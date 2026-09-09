#!/usr/bin/env python3
"""
Deploy DSB2018 test TIFF pair into real_cell.png / real_mask.png under this directory.

Why a script: DSB ships `.tif`; validation tooling often expects `.png`; avoids fragile shell quoting.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image


def _dsb_test_root() -> Path:
    """Prefer WSL-mounted path when running under WSL; else native Windows tree."""
    candidates = [
        Path("/mnt/c/Users/badri/Downloads/dsb2018/dsb2018/test"),
        Path(r"C:\Users\badri\Downloads\dsb2018\dsb2018\test"),
    ]
    for p in candidates:
        if p.is_dir():
            return p
    return candidates[-1]


def main() -> int:
    repo_validation = Path(__file__).resolve().parent
    dsb_root = _dsb_test_root()
    img_dir = dsb_root / "images"
    msk_dir = dsb_root / "masks"

    if not img_dir.is_dir():
        print(f"ERROR: missing image dir {img_dir}", file=sys.stderr)
        return 1
    imgs = sorted(img_dir.glob("*.tif"))
    if not imgs:
        print(f"ERROR: no .tif images in {img_dir}", file=sys.stderr)
        return 1

    src_img = imgs[0]
    stem = src_img.stem
    src_msk = msk_dir / f"{stem}.tif"
    if not src_msk.is_file():
        print(f"ERROR: no matching mask for {stem}: {src_msk}", file=sys.stderr)
        return 1

    cell_out = repo_validation / "real_cell.png"
    mask_out = repo_validation / "real_mask.png"

    im = Image.open(src_img).convert("RGB")
    im.save(cell_out, format="PNG")
    print("cell:", src_img, "->", cell_out, "size=", im.size)

    marr = np.array(Image.open(src_msk))
    if marr.ndim == 2:
        fg = ((marr > 0).astype(np.uint8)) * 255
    else:
        fg = ((np.any(marr > 0, axis=2)).astype(np.uint8)) * 255
    Image.fromarray(fg).save(mask_out, format="PNG")
    print("mask:", src_msk, "->", mask_out, fg.shape)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
