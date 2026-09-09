#!/usr/bin/env python3
"""
Diagnose `real_cell.png` for cytology validation: shape, range, blank check, bit-depth hints.

**Why**: Zero detections often trace to empty tiles, inverted polarity, or unexpected dtype/range.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image


def main() -> int:
    root = Path(__file__).resolve().parent
    path = root / "real_cell.png"
    if not path.is_file():
        print(f"ERROR: missing {path}", file=sys.stderr)
        return 1

    im = np.array(Image.open(path))

    print("file:", path.resolve())
    print("shape:", im.shape)
    print("dtype:", im.dtype)
    print("min:", float(np.nanmin(im)), "max:", float(np.nanmax(im)), "mean:", float(np.nanmean(im)))

    if im.size > 0:
        if im.ndim == 2:
            constant = bool(np.all(im == im.flat[0]))
            std = float(np.std(im.astype(np.float64)))
        else:
            constant = bool(np.all(im == im.reshape(-1)[0]))
            std = float(np.std(im.astype(np.float64)))
        print("is_constant (single value):", constant)
        print("std (global):", std)
        if constant or std < 1e-6:
            print("WARNING: image appears blank or nearly constant — downstream pipelines may report zero cells.")

    dt = im.dtype
    if dt == np.uint16 or dt == np.int16:
        print(
            "NOTE: 16-bit image. For 8-bit RGB OpenCV paths, rescale e.g.\n"
            "  rgb8 = (np.clip(im.astype(np.float32) / float(np.iinfo(dt).max), 0, 1) * 255).astype(np.uint8)\n"
            "or use percentile stretch: p2,p99 = np.percentile(im, (1,99.8)); "
            "rgb8 = np.clip((im.astype(np.float32)-p2)/(p99-p2+1e-9)*255,0,255).astype(np.uint8)"
        )
    elif np.issubdtype(dt, np.floating):
        print(
            "NOTE: float image. Prefer explicit [0,255] or [0,1]uint8 pipeline before geometry / visualization."
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
