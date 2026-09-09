#!/usr/bin/env python3
"""Demo: run the Nexus-Fusion malignancy classifier on a histology image (offline).

Setup:
  1. Build the Rust geometry engine:  cargo build -p nexus-forge-cyto-core --release --bin nexus-core-cli
  2. Install deps:  pip install -r requirements-fusion.txt
  3. Run:  python run_fusion_demo.py <image_path>

The pipeline: image -> ONNX HoVer-Net -> nuclei segmentation -> Rust geometry -> fusion classifier.
No tiatoolbox, no torch, no network.
"""
import sys
import os
import cv2

from nexus_fusion import NexusFusion

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))

_rust_bin = os.path.join(ROOT, "target", "release", "nexus-core-cli")
if os.name == "nt" and not os.path.exists(_rust_bin) and os.path.exists(_rust_bin + ".exe"):
    _rust_bin += ".exe"

DEFAULTS = {
    "onnx_path": os.path.join(ROOT, "models", "hovernet_finetuned.onnx"),
    "fusion_joblib": os.path.join(ROOT, "models", "nexus_fusion_classifier.joblib"),
    "rust_bin": _rust_bin,
}


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: python run_fusion_demo.py <image_path>")
        return 1

    img_path = sys.argv[1]
    if not os.path.exists(img_path):
        print(f"image not found: {img_path}")
        return 1

    for key, path in DEFAULTS.items():
        if not os.path.exists(path):
            print(f"missing {key}: {path}")
            print("hint: build the Rust engine with `cargo build -p nexus-forge-cyto-core --release --bin nexus-core-cli`")
            return 1

    model = NexusFusion(**DEFAULTS)

    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    results = model.predict_large(img)

    print(f"\n{len(results)} cells detected:")
    for r in sorted(results, key=lambda d: -d["probability"]):
        label = "MALIGNANT" if r["malignant"] else "normal"
        print(f"  ({r['cx']:.0f}, {r['cy']:.0f})  {label:<10} p={r['probability']:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
