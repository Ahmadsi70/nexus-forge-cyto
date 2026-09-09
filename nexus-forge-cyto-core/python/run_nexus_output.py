#!/usr/bin/env python3
"""Run Nexus-Fusion on an image and export (1) cell coordinates JSON and (2) a masked overlay image."""
import sys, os, json
import numpy as np
import cv2

from nexus_fusion import NexusFusion

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
_rust_bin = os.path.join(ROOT, "target", "release", "nexus-core-cli")
if os.name == "nt" and not os.path.exists(_rust_bin) and os.path.exists(_rust_bin + ".exe"):
    _rust_bin += ".exe"

MODEL = NexusFusion(
    onnx_path=os.path.join(ROOT, "models", "hovernet_finetuned.onnx"),
    fusion_joblib=os.path.join(ROOT, "models", "nexus_fusion_classifier.joblib"),
    rust_bin=_rust_bin,
)


def run(image_path, out_dir=None):
    img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    results = MODEL.predict_large(img)

    base = os.path.splitext(os.path.basename(image_path))[0]
    out_dir = out_dir or os.path.dirname(os.path.abspath(image_path))
    coords_path = os.path.join(out_dir, f"{base}_cells.json")
    mask_path = os.path.join(out_dir, f"{base}_masked.png")

    cells = [{"x": round(r["cx"], 2), "y": round(r["cy"], 2),
              "malignant": r["malignant"], "probability": round(r["probability"], 4)}
             for r in results]
    with open(coords_path, "w") as f:
        json.dump({"image": os.path.basename(image_path),
                   "n_cells": len(cells),
                   "n_malignant": sum(1 for c in cells if c["malignant"]),
                   "cells": cells}, f, indent=2)

    overlay = cv2.cvtColor(img, cv2.COLOR_RGB2BGR).copy()
    for r in results:
        color = (0, 0, 255) if r["malignant"] else (255, 0, 0)  # BGR: red=mali, blue=normal
        cx, cy = int(round(r["cx"])), int(round(r["cy"]))
        cv2.circle(overlay, (cx, cy), 6, color, 2)
        cv2.rectangle(overlay, (cx - 10, cy - 10), (cx + 10, cy + 10), color, 1)
    cv2.imwrite(mask_path, overlay)

    print(f"detected {len(cells)} cells ({sum(1 for c in cells if c['malignant'])} malignant)")
    print(f"coordinates -> {coords_path}")
    print(f"masked image -> {mask_path}")
    return coords_path, mask_path


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python run_nexus_output.py <image_path> [out_dir]")
        sys.exit(1)
    run(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
