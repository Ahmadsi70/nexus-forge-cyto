#!/usr/bin/env python3
"""
Build gold segment JSON (`cell_count`, `cells` as 32-point rings) from MoNuSeg / ImageScope XML.

Usage:
  python gen_gold_from_xml.py <input.xml> <output_gold_segment.json>

**Why**: Rust `write_enriched_segment_output` consumes the same envelope as Streamlit / `debug_monuseg_run.py`.
"""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: gen_gold_from_xml.py <input.xml> <output_gold_segment.json>", file=sys.stderr)
        return 2
    repo = Path(__file__).resolve().parent.parent.parent
    sys.path.insert(0, str(repo / "services"))
    from nexus_core.coord_adapter import write_gold_segment_json  # noqa: E402

    xml_path = Path(sys.argv[1]).expanduser().resolve()
    out_path = Path(sys.argv[2]).expanduser().resolve()
    if not xml_path.is_file():
        print(f"error: XML not found: {xml_path}", file=sys.stderr)
        return 1
    write_gold_segment_json(xml_path, out_path)
    print(f"wrote {out_path} ({out_path.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
