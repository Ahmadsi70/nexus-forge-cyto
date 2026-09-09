"""
Gold-segment normalization and Rust batch enrichment.

Why: both image-derived and expert vector uploads must converge on one JSON contract
before the in-crate κ / spatial engine runs.

Uses `rust_bridge` for efficient Rust integration (FFI preferred, subprocess fallback).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any

from nexus_core.coord_adapter import TARGET_RING, segment_envelope_from_polygons
from nexus_core.runtime_paths import is_windows_host, repo_root, schema_path
from nexus_core.vector_ingest import FORMAT_JSON_STAGE, detect_vector_format, parse_vector_bytes
from nexus_core.rust_bridge import run_rust_enrichment  # Import the bridge

HEADLESS_RUNTIME_SCRIPT = repo_root() / "services" / "geometry_core" / "headless_pipeline_runtime.py"

_SERVICES = Path(__file__).resolve().parents[1]
if str(_SERVICES) not in sys.path:
    sys.path.insert(0, str(_SERVICES))


class PipelineError(Exception):
    """User-visible pipeline failure with HTTP status hint."""

    def __init__(self, message: str, *, status_code: int = 400) -> None:
        super().__init__(message)
        self.status_code = status_code


def _validate_gold_segment(payload: dict[str, Any]) -> None:
    """Ensure uploaded stage JSON satisfies the batch binary contract."""
    if not isinstance(payload, dict):
        raise PipelineError("gold segment root must be a JSON object")
    cells = payload.get("cells")
    count = payload.get("cell_count")
    if not isinstance(cells, list) or not cells:
        raise PipelineError("gold segment must include non-empty cells[]")
    if not isinstance(count, int) or count != len(cells):
        raise PipelineError("cell_count must match len(cells)")
    for idx, ring in enumerate(cells):
        if not isinstance(ring, list) or len(ring) != TARGET_RING:
            raise PipelineError(f"cells[{idx}] must be a {TARGET_RING}-point ring")


def gold_segment_from_vector_bytes(filename: str, raw: bytes) -> dict[str, Any]:
    """Parse vendor vector upload or accept an existing gold-segment JSON envelope."""
    fmt = detect_vector_format(filename, raw)
    if fmt == FORMAT_JSON_STAGE:
        payload = json.loads(raw.decode("utf-8"))
        _validate_gold_segment(payload)
        return payload

    records, fmt_id = parse_vector_bytes(filename, raw)
    polygons = [[(float(p[0]), float(p[1])) for p in rec["ring"]] for rec in records]
    return segment_envelope_from_polygons(
        polygons,
        format_hint=fmt_id,
        source_path=filename,
    )


def gold_segment_from_polygons(
    polygons: list[list[tuple[float, float]]],
    *,
    format_hint: str,
    source_path: str | None = None,
) -> dict[str, Any]:
    """Build gold segment from explicit polygon list (SAM 3 / manual lists)."""
    return segment_envelope_from_polygons(
        polygons,
        format_hint=format_hint,
        source_path=source_path,
    )


def run_rust_enrichment_legacy(gold_segment: dict[str, Any]) -> dict[str, Any]:
    """Invoke nexus-forge-cyto enrichment via rust_bridge (FFI or subprocess)."""
    try:
        return run_rust_enrichment(gold_segment)
    except RuntimeError as exc:
        raise PipelineError(str(exc), status_code=500) from exc


def build_headless_export_cmd(
    enriched_json: Path,
    *,
    output_dir: Path,
    schema_path_override: Path | None = None,
    knn_k: int = 8,
) -> tuple[list[str], dict[str, str]]:
    """Build subprocess argv/env for headless schema export (shared by UI and API)."""
    env = os.environ.copy()
    env["NEXUS_INPUT_JSON"] = str(enriched_json)
    env["NEXUS_OUTPUT_DIR"] = str(output_dir)
    env["NEXUS_SCHEMA_PATH"] = str(schema_path_override or schema_path())
    env["NEXUS_KNN_K"] = str(knn_k)
    cmd = [sys.executable, os.path.abspath(str(HEADLESS_RUNTIME_SCRIPT))]
    return cmd, env


def run_headless_schema_export(
    enriched_json: Path,
    *,
    output_dir: Path,
    schema_path_override: Path | None = None,
    knn_k: int = 8,
) -> dict[str, Any]:
    """Run headless bootstrap+stats and return validated stage_v2_final payload."""
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd, env = build_headless_export_cmd(
        enriched_json,
        output_dir=output_dir,
        schema_path_override=schema_path_override,
        knn_k=knn_k,
    )
    proc = subprocess.run(
        cmd,
        cwd=str(repo_root()),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    final_path = output_dir / "stage_v2_final.json"
    if proc.returncode != 0 or not final_path.is_file():
        detail = (proc.stderr or proc.stdout or "headless export failed").strip()
        raise PipelineError(
            f"Schema export failed: {detail[:2000]}",
            status_code=500,
        )
    return json.loads(final_path.read_text(encoding="utf-8"))
