# -*- coding: utf-8 -*-
"""
Standalone Data-to-Text reporter: reads validation JSON, calls local Ollama, runs SIRAT checks.

**Why path bootstrap**: The SIRAT package lives at `tools/sirat/` but imports use the top-level
name `sirat`, so this script adds `tools/` to `sys.path` before importing `sirat.*`.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

_TOOLS = Path(__file__).resolve().parent
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

_SERVICES = Path(__file__).resolve().parents[1] / "services"
if str(_SERVICES) not in sys.path:
    sys.path.insert(0, str(_SERVICES))
from nexus_core.runtime_paths import repo_root  # noqa: E402


def _ensure_sirat_import_path() -> None:
    tools_dir = Path(__file__).resolve().parent
    root = str(tools_dir)
    if root not in sys.path:
        sys.path.insert(0, root)


def _load_output_payload(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _total_cells_from_payload(payload: dict[str, Any]) -> int:
    """Declared **`cell_count`** vs **`len(cells)`** — same rule as the dashboard."""
    cells = payload.get("cells")
    total = int(payload.get("cell_count", 0)) if isinstance(payload.get("cell_count"), int) else 0
    if isinstance(cells, list) and cells:
        total = max(total, len(cells))
    return total


def _clinical_incomplete_message(payload: dict[str, Any]) -> str | None:
    """
    Return a human-readable problem when geometry exists but **`clinical`** is unusable (**why**: downstream
    metrics must not silently show zero malignant/normal).
    """
    total = _total_cells_from_payload(payload)
    if total <= 0:
        return None
    clinical = payload.get("clinical")
    if not isinstance(clinical, list) or len(clinical) == 0:
        return (
            f"Data incomplete: {total} cell(s) detected but `clinical` is missing or empty "
            "(expected one Normal/Malignant string per cell)."
        )
    if len(clinical) != total:
        return (
            f"Data incomplete: total_cells={total} but len(clinical)={len(clinical)} "
            "(label list length must match cell count)."
        )
    return None


def _clinical_counts(clinical: Any) -> tuple[int, int, int]:
    """Return (malignant, normal, unknown) for string labels in a list."""
    malignant = 0
    normal = 0
    unknown = 0
    if not isinstance(clinical, list):
        return 0, 0, 0
    for item in clinical:
        label = str(item).strip().lower()
        if label == "malignant":
            malignant += 1
        elif label == "normal":
            normal += 1
        else:
            unknown += 1
    return malignant, normal, unknown


def _polygon_area(points: list[list[float]]) -> float:
    """Shoelace area (signed); expects closed or open ring."""
    if len(points) < 3:
        return 0.0
    n = len(points)
    s = 0.0
    for i in range(n):
        x1, y1 = float(points[i][0]), float(points[i][1])
        x2, y2 = float(points[(i + 1) % n][0]), float(points[(i + 1) % n][1])
        s += x1 * y2 - x2 * y1
    return abs(s) * 0.5


def _polygon_perimeter(points: list[list[float]]) -> float:
    if len(points) < 2:
        return 0.0
    n = len(points)
    per = 0.0
    for i in range(n):
        x1, y1 = float(points[i][0]), float(points[i][1])
        x2, y2 = float(points[(i + 1) % n][0]), float(points[(i + 1) % n][1])
        per += math.hypot(x2 - x1, y2 - y1)
    return per


def _shape_summaries(cells: Any) -> dict[str, float]:
    """Aggregate non-medical geometry cues from 32-vertex rings."""
    if not isinstance(cells, list) or not cells:
        return {
            "mean_compactness": float("nan"),
            "mean_turning_energy": float("nan"),
        }
    comps: list[float] = []
    turns: list[float] = []
    for cell in cells:
        if not isinstance(cell, list) or len(cell) < 3:
            continue
        ring: list[list[float]] = []
        for pt in cell:
            if isinstance(pt, (list, tuple)) and len(pt) >= 2:
                ring.append([float(pt[0]), float(pt[1])])
        if len(ring) < 3:
            continue
        area = _polygon_area(ring)
        per = _polygon_perimeter(ring)
        if per > 1e-9:
            comps.append((4.0 * math.pi * area) / (per * per))
        energy = 0.0
        m = len(ring)
        for i in range(m):
            ax, ay = ring[i][0] - ring[i - 1][0], ring[i][1] - ring[i - 1][1]
            bx, by = ring[(i + 1) % m][0] - ring[i][0], ring[(i + 1) % m][1] - ring[i][1]
            la = math.hypot(ax, ay)
            lb = math.hypot(bx, by)
            if la < 1e-12 or lb < 1e-12:
                continue
            dot = (ax * bx + ay * by) / (la * lb)
            dot = max(-1.0, min(1.0, dot))
            energy += abs(math.acos(dot))
        turns.append(energy / max(1, m))

    def _mean(xs: list[float]) -> float:
        return float(sum(xs) / len(xs)) if xs else float("nan")

    return {
        "mean_compactness": _mean(comps),
        "mean_turning_energy": _mean(turns),
    }


def _build_prompt(stats: dict[str, Any]) -> str:
    data_blob = json.dumps(stats, ensure_ascii=False, indent=2)
    return (
        "You are a technical data interpreter. Based ONLY on the following cell data, provide a "
        "short 3-bullet-point summary of the cell distribution and curvature. Do NOT provide medical "
        f"advice. Data: {data_blob}"
    )


def _ollama_post_chat_deterministic(
    base_url: str,
    model: str,
    user_text: str,
    *,
    timeout_s: float = 180.0,
    temperature: float = 0.0,
) -> str:
    """POST **`/api/chat`** with fixed temperature for deterministic summaries."""
    import urllib.error
    import urllib.request

    url = base_url.rstrip("/") + "/api/chat"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": user_text}],
        "stream": False,
        "options": {"temperature": float(temperature)},
    }
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            raw = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Ollama HTTP {e.code}: {detail}") from e
    msg = (raw.get("message") or {}).get("content") or ""
    return str(msg).strip()


def _run_sirat(text: str):
    """Return guardrail result or None if import/processing fails."""
    try:
        _ensure_sirat_import_path()
        from sirat.core.sirat_axiomatic_core import SiratAxiomaticCore

        core = SiratAxiomaticCore()
        return core.process_biomedical(text, logprobs=None)
    except Exception as exc:
        print(f"WARN: SIRAT validation skipped ({type(exc).__name__}): {exc}", file=sys.stderr)
        return None


def main() -> int:
    root = repo_root()
    json_path = root / "tests" / "validation" / "output.json"
    out_md = root / "tests" / "validation" / "smart_report.md"

    if not json_path.is_file():
        print(f"ERROR: missing fixture JSON: {json_path}", file=sys.stderr)
        return 1

    payload = _load_output_payload(json_path)
    cells = payload.get("cells")
    total = _total_cells_from_payload(payload)

    if total <= 0:
        msg = (
            "No cells detected: the envelope lists zero cells — verify expert annotations and **`gold_segment.json`**."
        )
        print(f"ERROR: {msg}", file=sys.stderr)
        out_md.parent.mkdir(parents=True, exist_ok=True)
        out_md.write_text(
            "# Technical summary\n\n## No cells detected\n\n" + msg + "\n",
            encoding="utf-8",
        )
        print(f"Wrote diagnostic stub to {out_md}", file=sys.stderr)
        return 1

    incomplete = _clinical_incomplete_message(payload)
    if incomplete is not None:
        print(f"ERROR: {incomplete}", file=sys.stderr)
        out_md.parent.mkdir(parents=True, exist_ok=True)
        out_md.write_text(
            "# Technical summary\n\n"
            "## Data incomplete\n\n"
            f"{incomplete}\n\n"
            "Analysis Engine Unavailable: Please ensure Mojo Library is linked.\n",
            encoding="utf-8",
        )
        print(f"Wrote diagnostic stub to {out_md}", file=sys.stderr)
        return 1

    mal, norm, unk = _clinical_counts(payload.get("clinical"))
    shapes = _shape_summaries(cells if isinstance(cells, list) else [])

    calibration_lines: list[str] = []
    if payload.get("dynamic_calibration") or payload.get("pre_analysis"):
        calibration_lines.append("## Acquisition-aware calibration")
        calibration_lines.append("")
        calibration_lines.append(
            "Parameters were dynamically calibrated for this specific image's lighting and density."
        )
        calibration_lines.append("")

    stats: dict[str, Any] = {
        "total_cells": total,
        "malignant_count": mal,
        "normal_count": norm,
        "unknown_label_count": unk,
        "geometry_summary": {
            "mean_compactness_4piA_over_P2": shapes["mean_compactness"],
            "mean_boundary_turning_energy_proxy": shapes["mean_turning_energy"],
        },
    }

    prompt = _build_prompt(stats)

    base_url = "http://localhost:11434"
    model = "qwen2.5:1.5b"
    timeout_s = 180.0

    _ensure_sirat_import_path()
    try:
        from sirat.ollama.ollama_bridge import ollama_health
    except ImportError as e:
        print(f"ERROR: could not import ollama_bridge (is `tools/sirat` present?): {e}", file=sys.stderr)
        return 1

    if not ollama_health(base_url, timeout_s=5.0):
        print(
            f"ERROR: Ollama not reachable at {base_url} (GET /api/tags failed). "
            f"Start Ollama locally and ensure model {model!r} is available (e.g. `ollama pull {model}`).",
            file=sys.stderr,
        )
        return 1

    try:
        response_text = _ollama_post_chat_deterministic(
            base_url, model, prompt, timeout_s=timeout_s, temperature=0.0
        )
    except Exception as e:
        print(f"ERROR: Ollama /api/chat failed: {e}", file=sys.stderr)
        return 1

    if not response_text.strip():
        print("ERROR: empty model response from Ollama.", file=sys.stderr)
        return 1

    sirat_res = _run_sirat(response_text)

    lines: list[str] = []
    lines.append("# Technical summary (local LLM + optional SIRAT)")
    lines.append("")
    lines.extend(calibration_lines)
    lines.append("## Model output")
    lines.append("")
    lines.append(response_text.strip())
    lines.append("")
    lines.append("## SIRAT biomedical guardrail (text rules + optional 5-pillar when logprobs present)")
    lines.append("")
    if sirat_res is None:
        lines.append("_Validation could not be run; see stderr warnings._")
    else:
        lines.append(f"- **accepted**: `{sirat_res.accepted}`")
        lines.append(f"- **violated_axioms**: `{sirat_res.violated_axioms}`")
        if sirat_res.suggested_innovations:
            ids = [str(x.get("id", "")) for x in sirat_res.suggested_innovations]
            lines.append(f"- **suggested_innovation_ids**: {ids}")
        if sirat_res.core_report is not None:
            from sirat.core.sirat_axiomatic_core import SiratAxiomaticCore

            lines.append("")
            lines.append("```")
            lines.append(SiratAxiomaticCore().format_report(sirat_res.core_report))
            lines.append("```")

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
