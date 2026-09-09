#!/usr/bin/env python3
"""
Nexus-Forge Cyto — HoVerNet + Fusion Classifier Benchmark

Pipeline: image -> ONNX HoVerNet -> nuclei seg + neoprob
        -> Python geometry (16 features) -> GBM fusion -> malignancy

Datasets: Pan-Cancer-Nuclei-Seg, Breast Histopathology, BACH
"""

import argparse, json, os, sys, time, io, itertools, warnings
from pathlib import Path
warnings.filterwarnings("ignore")

import numpy as np, cv2, joblib
from datasets import load_dataset
from PIL import Image
from scipy.spatial import ConvexHull, KDTree
from scipy.ndimage import label as sci_label


DATASETS = {
    "pan_cancer": {
        "path": "MedOtter/Pan-Cancer-Nuclei-Seg",
        "split": "train",
        "max_samples": 100,
        "image_key": "image",
        "desc": "Pan-Cancer H&E patches (1356, multiple cancers)",
    },
    "breast": {
        "path": "dbzadnen/breast-histopathology-images",
        "split": "train",
        "max_samples": 60,
        "image_key": "image",
        "desc": "Breast histopathology (IDC detection)",
    },
    "bach": {
        "path": "mezeidragos-lateral/bach-breast-histopathology-he-staining",
        "split": "train",
        "max_samples": 30,
        "image_key": "image",
        "desc": "BACH breast H&E staining",
    },
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="benchmark_hovernet.json")
    p.add_argument("--dataset", choices=list(DATASETS.keys()), default=None)
    p.add_argument("--samples", type=int, default=0, help="0 = dataset default")
    return p.parse_args()


# ─── HoVerNet + Fusion Pipeline ─────────────────────────────────────────────

OFF = 46  # HoVerNet: 256 input -> 164 output crop offset


def _softmax(x, axis):
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def _resample_polygon(pts, n=32):
    if len(pts) < 3:
        return None
    pts = np.asarray(pts, dtype=np.float64)
    if not np.allclose(pts[0], pts[-1]):
        pts = np.vstack([pts, pts[0]])
    seg = np.hypot(np.diff(pts[:, 0]), np.diff(pts[:, 1]))
    total = seg.sum()
    if total < 1e-9:
        return None
    cum = np.concatenate([[0], np.cumsum(seg)])
    t = np.linspace(0, total, n + 1)[:-1]
    return np.column_stack([np.interp(t, cum, pts[:, 0]),
                            np.interp(t, cum, pts[:, 1])])


def _hovernet_instances(image, sess):
    """image: (256,256,3) -> list of {cx, cy, ring, neop}"""
    x = image.astype(np.float32).transpose(2, 0, 1)[None]
    o_np, o_hv, o_tp = sess.run(None, {"input": x})

    from hovernet_postproc import postproc
    np_map = _softmax(o_np, axis=1)[0, 1][..., None]
    hv_map = o_hv[0].transpose(1, 2, 0)
    tp_soft = _softmax(o_tp, axis=1)[0]
    tp_arg = tp_soft.argmax(axis=0)[..., None].astype(np.float32)

    _, info = postproc(np_map, hv_map, tp_arg)

    instances = []
    for d in info.values():
        cx, cy = d["centroid"]
        neop = float(tp_soft[1, int(round(cy)), int(round(cx))])
        contour = np.asarray(d["contour"], dtype=np.float64)
        if contour.ndim != 2 or contour.shape[0] < 3:
            continue
        ring = _resample_polygon(contour + OFF)
        if ring is None:
            continue
        instances.append({
            "cx": float(cx) + OFF,
            "cy": float(cy) + OFF,
            "ring": ring,
            "neop": neop,
        })
    return instances


def _geometry_features(rings):
    """rings: [(32,2)] -> [[16 features]] using pure Python."""
    if not rings:
        return []

    # Extract polygon info
    areas = []
    perimeters = []
    hull_by_ring = []
    moments_by_ring = []
    centroids = []

    for r in rings:
        pts = np.array(r)
        # Area (shoelace)
        x, y = pts[:, 0], pts[:, 1]
        a = 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
        areas.append(a)
        # Perimeter
        p = np.sum(np.hypot(np.diff(x), np.diff(y)))
        perimeters.append(p)
        # Convex hull
        if len(pts) >= 4:
            try:
                hull = ConvexHull(pts)
                hull_pts = pts[hull.vertices]
                hull_a = hull.volume if hasattr(hull, 'volume') else 0
                hull_p = np.sum(np.hypot(np.diff(hull_pts[:, 0]),
                                          np.diff(hull_pts[:, 1])))
            except Exception:
                hull_a, hull_p = a, p
        else:
            hull_a, hull_p = a, p
        hull_by_ring.append((hull_a, hull_p))
        # Centroid
        cx = np.mean(x)
        cy = np.mean(y)
        centroids.append([cx, cy])
        # Hu moments
        m = cv2.moments(pts)
        hu = cv2.HuMoments(m).flatten()
        # Handle sign/log as OpenCV does
        hu = [np.sign(h) * np.log10(np.abs(h) + 1e-10) if np.abs(h) > 1e-10 else 0.0
              for h in hu]
        moments_by_ring.append(hu)

    # Normalized features
    med_area = float(np.median(areas)) if areas else 1.0

    # KNN density
    kdt = KDTree(centroids) if len(centroids) > 1 else None
    knn_densities = []
    if kdt is not None:
        for i, c in enumerate(centroids):
            dists, _ = kdt.query(c, k=min(9, len(centroids)))
            knn_densities.append(np.mean(dists[1:]) if len(dists) > 1 else 0.0)

    med_knn = float(np.median(knn_densities)) if knn_densities else 1.0

    # RBF risk (radial basis function density - inverse distance)
    rbf_risks = []
    if kdt is not None:
        for i, c in enumerate(centroids):
            dists, _ = kdt.query(c, k=min(9, len(centroids)))
            closest = [d for d in dists[1:] if d > 0]
            rbf = np.mean([np.exp(-d ** 2 / (2 * 50 ** 2)) for d in closest]) if closest else 0.0
            rbf_risks.append(rbf)

    features = []
    for i in range(len(rings)):
        a = areas[i]
        p = perimeters[i]
        ha, hp = hull_by_ring[i]

        area_ratio = a / med_area if med_area > 1e-9 else 1.0
        circularity = 4 * np.pi * a / (p * p + 1e-10)
        eccentricity = ha / (a + 1e-10) if a > 1e-9 else 1.0  # hull/area = solidity inverse
        solidity = a / (ha + 1e-10) if ha > 1e-9 else 1.0
        convexity = hp / (p + 1e-10) if p > 1e-9 else 1.0

        # Aspect ratio from bounding rectangle
        rect = cv2.minAreaRect(rings[i].astype(np.float32))
        (_, _), (w_, h_), _ = rect
        aspect_ratio = max(w_, h_) / (min(w_, h_) + 1e-10)

        rbf_risk = rbf_risks[i] if rbf_risks else 0.0
        knn_ratio = knn_densities[i] / med_knn if med_knn > 1e-9 else 1.0

        feats = [
            area_ratio, circularity, rbf_risk, knn_ratio,
            eccentricity, solidity, convexity, aspect_ratio,
        ] + moments_by_ring[i]  # 7 hu moments

        features.append(feats)

    return features


def _predict_image(image, sess, clf, threshold=0.5):
    """Run full pipeline on (256,256,3) image -> list of cell results."""
    instances = _hovernet_instances(image, sess)
    if not instances:
        return []
    rings = [i["ring"] for i in instances]
    geoms = _geometry_features(rings)
    results = []
    for inst, g in zip(instances, geoms):
        feat = np.array(g + [inst["neop"]], dtype=np.float64).reshape(1, -1)
        prob = float(clf.predict_proba(feat)[0, 1])
        results.append({
            "cx": inst["cx"],
            "cy": inst["cy"],
            "malignant": bool(prob > threshold),
            "probability": round(prob, 4),
            "neoplastic_prob": round(inst["neop"], 4),
        })
    return results


def predict_large(image, sess, clf, threshold=0.5, tile=256, stride=164, margin=46):
    """image: (H,W,3) any size -> tiled inference."""
    H, W = image.shape[:2]
    if H <= tile and W <= tile:
        pad_h = max(0, tile - H)
        pad_w = max(0, tile - W)
        img = image if pad_h == 0 and pad_w == 0 else np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)))
        return _predict_image(img, sess, clf, threshold)

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
            for cell in _predict_image(patch, sess, clf, threshold):
                lcx, lcy = cell["cx"], cell["cy"]
                if not (margin <= lcx < margin + stride and margin <= lcy < margin + stride):
                    continue
                gx = lcx + x - margin
                gy = lcy + y - margin
                if 0 <= gx < W and 0 <= gy < H:
                    results.append({
                        "cx": gx, "cy": gy,
                        "malignant": cell["malignant"],
                        "probability": cell["probability"],
                        "neoplastic_prob": cell["neoplastic_prob"],
                    })
    return results


# ─── Benchmark Runner ──────────────────────────────────────────────────────

class BenchmarkRunner:
    def __init__(self, output):
        self.output = output
        self.results = {
            "meta": {
                "pipeline": "HoVerNet ONNX -> Python Geometry -> Fusion GBM (AUC 0.92)",
                "model": "hovernet_finetuned.onnx (fold12 original retained)",
                "classifier": "nexus_fusion_classifier.joblib (Gradient Boosting, 16 feat)",
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            },
            "datasets": {},
            "aggregate": {},
        }

    def run_dataset(self, name, cfg, max_samples=None):
        ms = max_samples or cfg["max_samples"]
        print(f"\n{'='*60}")
        print(f"Dataset: {name} — {cfg['desc']}")
        print(f"{'='*60}")
        print(f"Loading {cfg['path']}...")

        ds = load_dataset(cfg["path"], split=cfg["split"], streaming=True)
        samples = list(itertools.islice(ds, ms))
        n = len(samples)

        print(f"Samples: {n}")

        # Load models once
        import onnxruntime as ort
        sess = ort.InferenceSession("/workspace/nexus/models/hovernet_finetuned.onnx",
                                    providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        clf = joblib.load("/workspace/nexus/models/nexus_fusion_classifier.joblib")

        per_sample = []
        errors = 0
        total_hovernet_time = 0.0
        total_geom_time = 0.0
        total_clf_time = 0.0

        for idx, sample in enumerate(samples):
            try:
                img = sample[cfg["image_key"]]
                rgb = np.array(img)

                # --- Pipeline timing ---
                t0 = time.perf_counter()

                # Step 1: HoVerNet segmentation
                t1 = time.perf_counter()
                instances = _hovernet_instances(
                    cv2.resize(rgb, (256, 256)), sess)
                t2 = time.perf_counter()

                # Step 2: Geometry features
                rings = [i["ring"] for i in instances]
                geoms = _geometry_features(rings)
                t3 = time.perf_counter()

                # Step 3: Fusion classification
                results = []
                for inst, g in zip(instances, geoms):
                    feat = np.array(g + [inst["neop"]], dtype=np.float64).reshape(1, -1)
                    prob = float(clf.predict_proba(feat)[0, 1])
                    results.append({
                        "cx": inst["cx"], "cy": inst["cy"],
                        "malignant": bool(prob > 0.5),
                        "probability": round(prob, 4),
                    })
                t4 = time.perf_counter()

                el = (t4 - t0) * 1000
                h_time = (t2 - t1) * 1000
                g_time = (t3 - t2) * 1000
                c_time = (t4 - t3) * 1000

                total_hovernet_time += h_time
                total_geom_time += g_time
                total_clf_time += c_time

                per_sample.append({
                    "idx": idx,
                    "total_ms": round(el, 1),
                    "hovernet_ms": round(h_time, 1),
                    "geom_ms": round(g_time, 1),
                    "clf_ms": round(c_time, 1),
                    "n_cells": len(results),
                    "n_malignant": sum(1 for r in results if r["malignant"]),
                })

            except Exception as e:
                errors += 1
                per_sample.append({"idx": idx, "error": str(e)[:300]})
                if errors > 10:
                    break

            if (idx + 1) % 20 == 0:
                print(f"  {idx+1}/{n} | errors={errors} | total cells={sum(s.get('n_cells',0) for s in per_sample if 'error' not in s)}")

        ok = [s for s in per_sample if "error" not in s]
        if not ok:
            return {"error": f"all {n} failed", "n_errors": errors}

        lats = [s["total_ms"] for s in ok]
        cells_per = [s["n_cells"] for s in ok]
        mal_per = [s["n_malignant"] for s in ok]

        stats = {
            "samples": n,
            "completed": len(ok),
            "errors": errors,
            "total_cells_detected": sum(cells_per),
            "total_malignant_predicted": sum(mal_per),
            "cells_per_image": {
                "mean": round(np.mean(cells_per), 1),
                "min": int(min(cells_per)),
                "max": int(max(cells_per)),
            },
            "latency_ms": {
                "mean": round(np.mean(lats), 1),
                "p50": round(float(np.percentile(lats, 50)), 1),
                "p95": round(float(np.percentile(lats, 95)), 1),
                "p99": round(float(np.percentile(lats, 99)), 1),
            },
            "timing_breakdown_ms": {
                "hovernet_mean": round(total_hovernet_time / len(ok), 1),
                "geometry_mean": round(total_geom_time / len(ok), 1),
                "classifier_mean": round(total_clf_time / len(ok), 1),
            },
            "throughput": {
                "images_per_sec": round(1.0 / (np.mean(lats) / 1000.0), 2),
                "cells_per_sec": round(sum(cells_per) / (sum(lats) / 1000.0), 2),
            },
        }

        print(f"\n  ✅ {name}:")
        print(f"     Cells: {stats['total_cells_detected']} ({stats['cells_per_image']['mean']}/img)")
        print(f"     Malignant: {stats['total_malignant_predicted']}")
        print(f"     Latency: {stats['latency_ms']['mean']}ms avg | p95: {stats['latency_ms']['p95']}ms")
        print(f"     Throughput: {stats['throughput']['images_per_sec']} img/s")
        print(f"     Breakdown: H={stats['timing_breakdown_ms']['hovernet_mean']}ms + "
              f"G={stats['timing_breakdown_ms']['geometry_mean']}ms + "
              f"C={stats['timing_breakdown_ms']['classifier_mean']}ms")

        self.results["datasets"][name] = stats
        return stats

    def save(self):
        Path(self.output).write_text(json.dumps(self.results, indent=2, default=str), encoding="utf-8")

    def run(self, dset_filter=None):
        names = [dset_filter] if dset_filter else list(DATASETS.keys())
        for n in names:
            if n not in DATASETS:
                print(f"Unknown dataset: {n}")
                continue
            try:
                self.run_dataset(n, DATASETS[n])
                self.save()
            except Exception as e:
                import traceback
                self.results["datasets"][n] = {"error": str(e)[:500],
                                                "traceback": traceback.format_exc()[:500]}
                self.save()

        # Aggregate
        lats = []
        cells = []
        for data in self.results["datasets"].values():
            if isinstance(data, dict) and "latency_ms" in data:
                lats.append(data["latency_ms"]["mean"])
                cells.append(data["total_cells_detected"])
        if lats:
            self.results["aggregate"] = {
                "datasets_benchmarked": len(lats),
                "avg_latency_ms": round(np.mean(lats), 1),
                "total_cells_detected": sum(cells),
                "avg_cells_per_dataset": round(np.mean(cells), 0),
            }
        self.save()
        print(f"\nDone → {self.output}")


def main():
    args = parse_args()
    runner = BenchmarkRunner(args.output)
    runner.run(dset_filter=args.dataset)
    # Print final summary
    data = runner.results
    print(f"\n{'='*60}")
    print("🏁 FINAL BENCHMARK SUMMARY")
    print(f"{'='*60}")
    for dset, stats in data.get("datasets", {}).items():
        if isinstance(stats, dict) and "latency_ms" in stats:
            print(f"  📁 {dset}:")
            print(f"     Latency: {stats['latency_ms']['mean']}ms avg | "
                  f"{stats['latency_ms']['p95']}ms p95")
            print(f"     Cells: {stats['total_cells_detected']} in {stats['completed']} images")
            print(f"     Throughput: {stats['throughput']['images_per_sec']} img/s")
    print(f"\n  📊 Aggregate: {data.get('aggregate', {})}")


if __name__ == "__main__":
    main()