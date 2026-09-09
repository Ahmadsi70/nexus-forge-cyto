#!/usr/bin/env python3
"""
Full image pipeline: SAM 3 → gold segment → Rust κ enrich → schema export + mask PNG.

Why: one CLI for headless RunPod / batch runs without Streamlit or manual API calls.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SERVICES = Path(__file__).resolve().parents[1] / "services"
if str(_SERVICES) not in sys.path:
    sys.path.insert(0, str(_SERVICES))

from api_service.image_ingest import segment_image_bytes  # noqa: E402
from api_service.pipeline_core import (  # noqa: E402
    gold_segment_from_polygons,
    run_headless_schema_export,
    run_rust_enrichment,
)
from api_service.mask_utils import load_rgb_array  # noqa: E402


def _polygons_to_geojson(gold: dict, *, source: str) -> dict:
    features = []
    for idx, ring in enumerate(gold.get("cells") or []):
        coords = [[float(p[0]), float(p[1])] for p in ring]
        if coords and coords[0] != coords[-1]:
            coords.append(coords[0])
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": [coords]},
                "properties": {"cell_id": idx, "source": source},
            }
        )
    return {"type": "FeatureCollection", "features": features}


def _polygons_to_binary_mask(cells: list, height: int, width: int) -> "object":
    import cv2
    import numpy as np

    mask = np.zeros((height, width), dtype=np.uint8)
    for ring in cells:
        pts = np.array([[int(round(p[0])), int(round(p[1]))] for p in ring], dtype=np.int32)
        if len(pts) >= 3:
            cv2.fillPoly(mask, [pts], 255)
    return mask


def run(image_path: Path, output_dir: Path, *, text_prompt: str | None) -> dict:
    """Execute end-to-end pipeline; write artifacts under output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)
    raw = image_path.read_bytes()
    rgb = load_rgb_array(raw)
    h, w = rgb.shape[:2]

    polygons, fmt = segment_image_bytes(raw, text_prompt=text_prompt)
    gold = gold_segment_from_polygons(
        polygons,
        format_hint=fmt,
        source_path=str(image_path.resolve()),
    )
    gold_path = output_dir / "gold_segment.json"
    gold_path.write_text(json.dumps(gold, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    geo = _polygons_to_geojson(gold, source=fmt)
    geo_path = output_dir / "nuclei_coordinates.geojson"
    geo_path.write_text(json.dumps(geo, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    mask = _polygons_to_binary_mask(gold["cells"], h, w)
    mask_path = output_dir / "nuclei_mask_binary.png"
    from PIL import Image

    Image.fromarray(mask).save(mask_path)

    enriched = run_rust_enrichment(gold)
    enriched_path = output_dir / "gold_segment_enriched.json"
    enriched_path.write_text(json.dumps(enriched, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    final_path = output_dir / "stage_v2_final.json"
    export_error: str | None = None
    try:
        final = run_headless_schema_export(enriched_path, output_dir=output_dir)
        final_path.write_text(json.dumps(final, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except Exception as exc:
        export_error = str(exc)
        final_path = None

    summary = {
        "input": str(image_path.resolve()),
        "output_dir": str(output_dir.resolve()),
        "cell_count": gold["cell_count"],
        "source": fmt,
        "image_size": [w, h],
        "schema_export_error": export_error,
        "artifacts": {
            "coordinates_geojson": str(geo_path),
            "binary_mask_png": str(mask_path),
            "gold_segment_json": str(gold_path),
            "enriched_json": str(enriched_path),
            "stage_v2_final_json": str(final_path) if final_path else None,
        },
    }
    summary_path = output_dir / "pipeline_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main() -> int:
    p = argparse.ArgumentParser(description="Run full SAM3 → Rust geometry pipeline on one image.")
    p.add_argument("image", type=Path, help="Input PNG/JPG/TIF patch")
    p.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: tmp/production_output/<stem>)",
    )
    p.add_argument("--text-prompt", default="nucleus", help="SAM 3 text concept (default: nucleus)")
    args = p.parse_args()

    if not args.image.is_file():
        print(f"FATAL: image not found: {args.image}", file=sys.stderr)
        return 2

    out = args.output_dir
    if out is None:
        out = Path(__file__).resolve().parents[1] / "tmp" / "production_output" / args.image.stem

    try:
        run(args.image, out, text_prompt=args.text_prompt.strip() or None)
    except Exception as exc:
        print(f"pipeline failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
