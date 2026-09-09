#!/usr/bin/env python3
"""CLI: run the offline fusion pipeline on an image and print the cells JSON to stdout.

Used by the Rust backend (nexus-forge-cyto-api) to serve /analyze-image.
"""
import sys, os, json, time
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


def main() -> int:
    if len(sys.argv) < 2:
        print(json.dumps({"status": "error", "message": "no image path"}))
        return 1

    img_path = sys.argv[1]
    if not os.path.exists(img_path):
        print(json.dumps({"status": "error", "message": "image not found"}))
        return 1

    t0 = time.time()
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    t_load_ms = (time.time() - t0) * 1000.0

    results = MODEL.predict_large(img)
    t_total_ms = (time.time() - t0) * 1000.0

    cells = [{"x": round(r["cx"], 2), "y": round(r["cy"], 2),
              "is_cancer": r["malignant"], "probability": round(r["probability"], 4)}
             for r in results]

    out = {
        "status": "success",
        "cells": cells,
        "nodes_count": len(cells),
        "edges_count": 0,
        "malignant_count": sum(1 for c in cells if c["is_cancer"]),
        "timing_ms": {
            "total": round(t_total_ms, 1),
            "image_processing": round(t_load_ms, 1),
            "mojo_engine": 0.0,
        },
    }
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
