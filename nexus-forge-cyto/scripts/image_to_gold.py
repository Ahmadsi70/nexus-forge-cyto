#!/usr/bin/env python3
"""CLI: patch image → gold segment JSON via SAM 3."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SERVICES = Path(__file__).resolve().parents[1] / "services"
if str(_SERVICES) not in sys.path:
    sys.path.insert(0, str(_SERVICES))

from api_service.image_ingest import segment_image_bytes  # noqa: E402
from api_service.pipeline_core import gold_segment_from_polygons  # noqa: E402


def _parse_bboxes(raw: str | None) -> list[list[float]] | None:
    if not raw:
        return None
    parsed = json.loads(raw)
    if not isinstance(parsed, list):
        raise ValueError("bboxes JSON must be a list of [x1,y1,x2,y2]")
    out: list[list[float]] = []
    for box in parsed:
        if not isinstance(box, list) or len(box) != 4:
            raise ValueError("each bbox must be [x1,y1,x2,y2]")
        out.append([float(v) for v in box])
    return out


def main() -> int:
    p = argparse.ArgumentParser(description="SAM 3 image patch → gold_segment.json")
    p.add_argument("image", type=Path, help="Input PNG/JPG/TIF path")
    p.add_argument("output", type=Path, help="Output gold_segment.json path")
    p.add_argument("--text-prompt", default="", help="Optional SAM 3 text concept prompt")
    p.add_argument("--bboxes", default="", help='Optional JSON list of exemplar boxes [[x1,y1,x2,y2],...]')
    args = p.parse_args()

    if not args.image.is_file():
        print(f"FATAL: image not found: {args.image}", file=sys.stderr)
        return 2

    raw = args.image.read_bytes()
    try:
        bboxes = _parse_bboxes(args.bboxes.strip() or None)
        polygons, fmt = segment_image_bytes(
            raw,
            bboxes=bboxes,
            text_prompt=args.text_prompt.strip() or None,
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 3
    except Exception as exc:
        print(f"segmentation failed: {exc}", file=sys.stderr)
        return 4

    gold = gold_segment_from_polygons(
        polygons,
        format_hint=fmt,
        source_path=str(args.image.resolve()),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(gold, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"cell_count": gold["cell_count"], "engine": fmt, "output": str(args.output.resolve())}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
