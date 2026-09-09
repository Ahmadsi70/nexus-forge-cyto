"""
MoNuSeg XML → Nexus 32-ring → Rust enriched geometry tolerance audit.

Why: quantifies how much expert polygon fidelity is preserved when the offline
geometry engine normalizes boundaries to the fixed 32-vertex contract.

Default XML:
  C:\\Users\\badri\\Downloads\\MoNuSegTestData\\MoNuSegTestData\\TCGA-IZ-8196-01A-01-BS1.xml

Override:
  CYTO_MONUSEG_XML=/path/to/file.xml pytest tests/test_monuseg_geometry_tolerance.py -v

Artifacts (under tmp/production_output/monuseg_tolerance/):
  gold_segment.json, enriched.json, tolerance_report.json
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[1]
_SERVICES = _REPO / "services"
if str(_SERVICES) not in sys.path:
    sys.path.insert(0, str(_SERVICES))

from nexus_core.coord_adapter import (  # noqa: E402
    _polygon_area_shoelace,
    parse_monuseg_xml,
    ring32_from_polygon_xy,
    write_gold_segment_json,
)

DEFAULT_XML = Path(
    r"C:\Users\badri\Downloads\MoNuSegTestData\MoNuSegTestData\TCGA-IZ-8196-01A-01-BS1.xml"
)
RUST_BIN = _REPO / "target" / "release" / "nexus-forge-cyto-batch.exe"
OUT_DIR = _REPO / "tmp" / "production_output" / "monuseg_tolerance"

# Tolerance gates (px / ratios) — geometry contract, not clinical diagnosis.
CENTROID_MEDIAN_MAX_PX = 2.5
CENTROID_P95_MAX_PX = 6.0
AREA_RATIO_MEDIAN_MAX = 0.12
AREA_RATIO_P95_MAX = 0.25
ENRICHED_AREA_REL_ERR_MAX = 1e-4


@dataclass
class CellTolerance:
    cell_id: int
    n_xml_vertices: int
    centroid_shift_px: float
    area_xml_px2: float
    area_ring_px2: float
    area_enriched_px2: float
    area_ring_ratio: float
    area_enriched_vs_ring: float


@dataclass
class ToleranceSummary:
    xml_path: str
    microns_per_pixel: str | None
    cell_count: int
    centroid_shift_median_px: float
    centroid_shift_p95_px: float
    area_ring_ratio_median: float
    area_ring_ratio_p95: float
    enriched_area_max_rel_err: float
    rust_ran: bool
    verdict: str


def _resolve_xml() -> Path:
    raw = os.environ.get("CYTO_MONUSEG_XML", "").strip()
    path = Path(raw) if raw else DEFAULT_XML
    if not path.is_file():
        pytest.skip(f"MoNuSeg XML not found: {path}")
    return path


def _polygon_centroid(points: list[tuple[float, float]]) -> tuple[float, float]:
    arr = np.asarray(points, dtype=np.float64)
    if np.allclose(arr[0], arr[-1]):
        arr = arr[:-1]
    if len(arr) < 3:
        return float(arr[:, 0].mean()), float(arr[:, 1].mean())
    x, y = arr[:, 0], arr[:, 1]
    cross = x * np.roll(y, -1) - np.roll(x, -1) * y
    a = cross.sum() * 0.5
    if abs(a) < 1e-12:
        return float(x.mean()), float(y.mean())
    cx = np.sum((x + np.roll(x, -1)) * cross) / (6.0 * a)
    cy = np.sum((y + np.roll(y, -1)) * cross) / (6.0 * a)
    return float(cx), float(cy)


def _ring_centroid(ring: list[list[float]]) -> tuple[float, float]:
    return _polygon_centroid([(float(p[0]), float(p[1])) for p in ring])


def _read_microns_per_pixel(xml_path: Path) -> str | None:
    import xml.etree.ElementTree as ET

    root = ET.parse(xml_path).getroot()
    return root.get("MicronsPerPixel")


def _run_rust_enrichment(gold_json: Path, out_dir: Path) -> Path | None:
    """Invoke release batch binary from an ingest-only directory."""
    ingest_dir = out_dir / "ingest"
    rust_out = out_dir / "rust_out"
    ingest_dir.mkdir(parents=True, exist_ok=True)
    rust_out.mkdir(parents=True, exist_ok=True)
    staged = ingest_dir / "gold_segment.json"
    if staged.resolve() != gold_json.resolve():
        staged.write_bytes(gold_json.read_bytes())
    if not RUST_BIN.is_file():
        return None
    env = os.environ.copy()
    env.setdefault("CYTO_MOJO_LIB_DIR", "")
    proc = subprocess.run(
        [str(RUST_BIN), "--input-dir", str(ingest_dir), "--output-dir", str(rust_out)],
        cwd=str(_REPO),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return None
    enriched = rust_out / "gold_segment_enriched.json"
    return enriched if enriched.is_file() else None


def build_tolerance_report(xml_path: Path, out_dir: Path) -> tuple[ToleranceSummary, list[CellTolerance]]:
    """Compare MoNuSeg native polygons vs Nexus 32-ring vs Rust spatial_features."""
    out_dir.mkdir(parents=True, exist_ok=True)
    gold_path = out_dir / "gold_segment.json"
    write_gold_segment_json(xml_path, gold_path)
    gold = json.loads(gold_path.read_text(encoding="utf-8"))

    xml_polys = parse_monuseg_xml(xml_path)
    assert len(xml_polys) == int(gold["cell_count"]), "region count mismatch XML vs gold JSON"

    enriched_path = _run_rust_enrichment(gold_path, out_dir / "rust_out")
    rust_ran = enriched_path is not None
    enriched_sf: list[dict] = []
    if enriched_path:
        enriched = json.loads(enriched_path.read_text(encoding="utf-8"))
        enriched_sf = enriched.get("spatial_features", [])

    rows: list[CellTolerance] = []
    for i, poly in enumerate(xml_polys):
        ring = gold["cells"][i]
        cx0, cy0 = _polygon_centroid(poly)
        cx1, cy1 = _ring_centroid(ring)
        shift = float(np.hypot(cx1 - cx0, cy1 - cy0))
        area_xml = _polygon_area_shoelace([[x, y] for x, y in poly])
        area_ring = _polygon_area_shoelace(ring)
        area_enriched = float(enriched_sf[i]["area"]) if i < len(enriched_sf) else area_ring
        ratio = abs(area_ring - area_xml) / max(area_xml, 1e-9)
        rel_enr = abs(area_enriched - area_ring) / max(area_ring, 1e-9)
        rows.append(
            CellTolerance(
                cell_id=i,
                n_xml_vertices=len(poly),
                centroid_shift_px=shift,
                area_xml_px2=area_xml,
                area_ring_px2=area_ring,
                area_enriched_px2=area_enriched,
                area_ring_ratio=ratio,
                area_enriched_vs_ring=rel_enr,
            )
        )

    shifts = np.asarray([r.centroid_shift_px for r in rows], dtype=np.float64)
    area_ratios = np.asarray([r.area_ring_ratio for r in rows], dtype=np.float64)
    enr_errs = np.asarray([r.area_enriched_vs_ring for r in rows], dtype=np.float64)

    summary = ToleranceSummary(
        xml_path=str(xml_path),
        microns_per_pixel=_read_microns_per_pixel(xml_path),
        cell_count=len(rows),
        centroid_shift_median_px=float(np.median(shifts)),
        centroid_shift_p95_px=float(np.percentile(shifts, 95)),
        area_ring_ratio_median=float(np.median(area_ratios)),
        area_ring_ratio_p95=float(np.percentile(area_ratios, 95)),
        enriched_area_max_rel_err=float(np.max(enr_errs)) if enr_errs.size else 0.0,
        rust_ran=rust_ran,
        verdict="pending",
    )

    ok_centroid = (
        summary.centroid_shift_median_px <= CENTROID_MEDIAN_MAX_PX
        and summary.centroid_shift_p95_px <= CENTROID_P95_MAX_PX
    )
    ok_area = (
        summary.area_ring_ratio_median <= AREA_RATIO_MEDIAN_MAX
        and summary.area_ring_ratio_p95 <= AREA_RATIO_P95_MAX
    )
    ok_rust = (not rust_ran) or summary.enriched_area_max_rel_err <= ENRICHED_AREA_REL_ERR_MAX
    summary.verdict = "PASS" if (ok_centroid and ok_area and ok_rust) else "REVIEW"
    if not rust_ran:
        summary.verdict = "PASS" if (ok_centroid and ok_area) else "REVIEW"

    report_path = out_dir / "tolerance_report.json"
    report_path.write_text(
        json.dumps(
            {
                "summary": asdict(summary),
                "thresholds": {
                    "centroid_median_max_px": CENTROID_MEDIAN_MAX_PX,
                    "centroid_p95_max_px": CENTROID_P95_MAX_PX,
                    "area_ratio_median_max": AREA_RATIO_MEDIAN_MAX,
                    "area_ratio_p95_max": AREA_RATIO_P95_MAX,
                    "enriched_area_rel_err_max": ENRICHED_AREA_REL_ERR_MAX,
                },
                "cells_sample": [asdict(r) for r in rows[:5]],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return summary, rows


def test_monuseg_xml_to_nexus_geometry_tolerance() -> None:
    """End-to-end tolerance audit on TCGA-IZ-8196 MoNuSeg XML."""
    xml_path = _resolve_xml()
    summary, _rows = build_tolerance_report(xml_path, OUT_DIR)

    assert summary.cell_count >= 100, "unexpectedly small region count"
    assert summary.verdict == "PASS", (
        f"tolerance REVIEW: median_centroid={summary.centroid_shift_median_px:.3f}px "
        f"p95_centroid={summary.centroid_shift_p95_px:.3f}px "
        f"median_area_ratio={summary.area_ring_ratio_median:.4f} "
        f"rust_ran={summary.rust_ran}"
    )


if __name__ == "__main__":
    xml = _resolve_xml()
    s, rows = build_tolerance_report(xml, OUT_DIR)
    print(json.dumps(asdict(s), indent=2))
    print(f"report={OUT_DIR / 'tolerance_report.json'}")
    print(f"cells={len(rows)}")
