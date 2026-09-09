#!/usr/bin/env python3
"""
Nexus-Fusion — offline malignancy classifier (ONNX HoVer-Net + Rust geometry + GBM).

No tiatoolbox, no cloud, no network at runtime.
Pipeline:
  image -> ONNX HoVer-Net -> standalone post-processing -> per-cell contour
        -> Rust geometry engine (nexus-core-cli) -> 16 shape features
        -> fusion classifier -> per-cell malignancy

Dependencies: onnxruntime, numpy, scipy, scikit-image, opencv-python, joblib, scikit-learn.

Usage:
  from nexus_fusion import NexusFusion
  model = NexusFusion(onnx_path="models/hovernet_finetuned.onnx",
                      fusion_joblib="models/nexus_fusion_classifier.joblib",
                      rust_bin="target/release/nexus-core-cli")
  results = model.predict(image)   # image: (H,W,3) uint8
"""
import os, json, subprocess, tempfile
import numpy as np
import cv2
import joblib

from hovernet_postproc import postproc

OFF = 46  # HoVer-Net fast: 256 input -> 164 output (center-crop offset)


class NexusFusion:
    def __init__(self, onnx_path, fusion_joblib, rust_bin):
        import onnxruntime as ort
        self.sess = ort.InferenceSession(onnx_path, providers=[
            "CUDAExecutionProvider", "CPUExecutionProvider"
        ])
        self.clf = joblib.load(fusion_joblib)
        self.rust_bin = rust_bin

    @staticmethod
    def _softmax(x, axis):
        e = np.exp(x - x.max(axis=axis, keepdims=True))
        return e / e.sum(axis=axis, keepdims=True)

    @staticmethod
    def _resample_polygon(pts, n=32):
        pts = np.asarray(pts, dtype=np.float64)
        if len(pts) < 3:
            return None
        if not np.allclose(pts[0], pts[-1]):
            pts = np.vstack([pts, pts[0]])
        d = np.diff(pts, axis=0); seg = np.hypot(d[:, 0], d[:, 1]); total = seg.sum()
        if total < 1e-9:
            return None
        cum = np.concatenate([[0], np.cumsum(seg)]); t = np.linspace(0, total, n + 1)[:-1]
        return np.stack([np.interp(t, cum, pts[:, 0]), np.interp(t, cum, pts[:, 1])], 1)

    def _hovernet_instances(self, image):
        """image: (H,W,3) uint8 -> list of {cx, cy, ring(32,2), neop} in image coords."""
        x = image.astype(np.float32).transpose(2, 0, 1)[None]  # (1,3,256,256), model /255 internally
        o_np, o_hv, o_tp = self.sess.run(None, {"input": x})   # logits

        np_map = self._softmax(o_np, axis=1)[0, 1][..., None]   # (164,164,1) nucleus prob
        hv_map = o_hv[0].transpose(1, 2, 0)                      # (164,164,2)
        tp_soft = self._softmax(o_tp, axis=1)[0]                 # (6,164,164)
        tp_arg = tp_soft.argmax(axis=0)[..., None].astype(np.float32)  # (164,164,1)

        _, info = postproc(np_map, hv_map, tp_arg)

        instances = []
        for d in info.values():
            cx, cy = d["centroid"]                     # 164-space (x, y)
            neop = float(tp_soft[1, int(round(cy)), int(round(cx))])
            contour = np.asarray(d["contour"], dtype=np.float64)
            if contour.ndim != 2 or contour.shape[0] < 3:
                continue
            ring = self._resample_polygon(contour + OFF)
            if ring is None:
                continue
            instances.append({"cx": float(cx) + OFF, "cy": float(cy) + OFF,
                              "ring": ring, "neop": neop})
        return instances

    def _geometry_features(self, rings):
        """rings: list of (32,2) -> list of 16-dim normalized geometry features (via Rust)."""
        if not rings:
            return []
        cells = [[[float(x), float(y)] for x, y in r] for r in rings]
        with tempfile.TemporaryDirectory() as td:
            inp = os.path.join(td, "segment.json")
            outp = os.path.join(td, "segment_enriched.json")
            json.dump({"cell_count": len(cells), "cells": cells}, open(inp, "w"))
            subprocess.run([self.rust_bin, "single", "--input", inp, "--output", outp],
                           capture_output=True, check=True)
            sf = json.load(open(outp))["spatial_features"]
        areas = [f["area"] for f in sf]; knns = [f["knn_density"] for f in sf]
        med_area = float(np.median(areas)) if areas else 1.0
        med_knn = float(np.median(knns)) if knns else 1.0
        feats = []
        for f in sf:
            area_ratio = f["area"] / med_area if med_area > 1e-9 else 1.0
            knn_ratio = f["knn_density"] / med_knn if med_knn > 1e-9 else 1.0
            feats.append([
                area_ratio, f["circularity"], f["rbf_risk_score"], knn_ratio,
                f["eccentricity"], f["solidity"], f["convexity"], f["aspect_ratio"],
            ] + list(f["hu_moments"]))
        return feats

    def predict(self, image, threshold=0.5):
        """image: (256,256,3) uint8 -> list of {cx, cy, malignant, probability}."""
        instances = self._hovernet_instances(image)
        if not instances:
            return []
        rings = [i["ring"] for i in instances]
        geoms = self._geometry_features(rings)
        results = []
        for inst, g in zip(instances, geoms):
            feat = np.array(g + [inst["neop"]]).reshape(1, -1)
            prob = float(self.clf.predict_proba(feat)[0, 1])
            results.append({"cx": inst["cx"], "cy": inst["cy"],
                            "malignant": bool(prob > threshold), "probability": prob})
        return results

    def predict_large(self, image, threshold=0.5, tile=256, stride=164, margin=46):
        """image: (H,W,3) uint8 (any size) -> list of {cx, cy, malignant, probability}.

        Tiles with overlap so the model's 46px blind border is covered (no gaps, no duplicates).
        """
        H, W = image.shape[:2]
        if H <= tile and W <= tile:
            pad_h = max(0, tile - H)
            pad_w = max(0, tile - W)
            img = image if pad_h == 0 and pad_w == 0 else np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)))
            return self.predict(img, threshold)

        padded = cv2.copyMakeBorder(image, margin, margin, margin, margin, cv2.BORDER_REFLECT)
        PH, PW = padded.shape[:2]
        results = []
        for y in range(0, PH, stride):
            for x in range(0, PW, stride):
                patch = padded[y:y + tile, x:x + tile]
                ph = tile - patch.shape[0]
                pw = tile - patch.shape[1]
                if ph > 0 or pw > 0:
                    patch = np.pad(patch, ((0, ph), (0, pw), (0, 0)))
                for cell in self.predict(patch, threshold):
                    lcx, lcy = cell["cx"], cell["cy"]
                    if not (margin <= lcx < margin + stride and margin <= lcy < margin + stride):
                        continue  # keep only the tile's visible core -> no duplicate counting
                    gx = lcx + x - margin
                    gy = lcy + y - margin
                    if 0 <= gx < W and 0 <= gy < H:
                        results.append({"cx": gx, "cy": gy,
                                        "malignant": cell["malignant"], "probability": cell["probability"]})
        return results
