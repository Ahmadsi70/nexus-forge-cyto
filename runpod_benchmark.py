#!/usr/bin/env python3
"""
Nexus-Forge Cyto — Comprehensive HTTP API Benchmark v2
Runs standard pathology datasets through the full pipeline.
"""

import argparse, json, os, sys, time, io, itertools, warnings
from pathlib import Path
warnings.filterwarnings("ignore")

import httpx, numpy as np
from datasets import load_dataset
from PIL import Image


DATASETS = {
    "pan_cancer": {
        "path": "MedOtter/Pan-Cancer-Nuclei-Seg",
        "split": "train",
        "max_samples": 50,
        "image_key": "image",
        "desc": "Pan-Cancer H&E patches (1356, multiple cancers)",
    },
    "breast": {
        "path": "dbzadnen/breast-histopathology-images",
        "split": "train",
        "max_samples": 40,
        "image_key": "image",
        "desc": "Breast histopathology (IDC detection)",
    },
    "bach": {
        "path": "mezeidragos-lateral/bach-breast-histopathology-he-staining",
        "split": "train",
        "max_samples": 20,
        "image_key": "image",
        "desc": "BACH breast H&E staining",
    },
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--host", default=os.environ.get("API_HOST", "localhost"))
    p.add_argument("--port", default=int(os.environ.get("API_PORT", "8810")))
    p.add_argument("--output", default="benchmark_results.json")
    p.add_argument("--dataset", choices=list(DATASETS.keys()), default=None)
    return p.parse_args()


class Runner:
    def __init__(self, host, port, output):
        self.base = f"http://{host}:{port}"
        self.output = output
        self.results = {
            "meta": {"host": host, "port": port,
                     "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
            "datasets": {}, "endpoint_latency": {}, "summary": {},
        }
        self.cl = httpx.Client(timeout=httpx.Timeout(300.0, connect=30.0))
        self._health()

    def _health(self):
        r = self.cl.get(f"{self.base}/health", timeout=10)
        r.raise_for_status()
        print(f"Server: {r.json()}")

    def _timed(self, method, path, **kw):
        t0 = time.perf_counter()
        r = self.cl.request(method, f"{self.base}{path}", **kw)
        el = (time.perf_counter() - t0) * 1000.0
        r.raise_for_status()
        return el, r.json()

    def bench_vector(self):
        print("\n[Ingest Vector]")
        res = []
        for nc in [1, 10, 50, 200]:
            lats = []
            for _ in range(10):
                geo = {"type": "FeatureCollection",
                       "features": [{"type": "Feature",
                                     "geometry": {"type": "Polygon",
                                                  "coordinates": [[[j,j],[j+10,j],[j+10,j+10],[j,j+10],[j,j]]]},
                                     "properties": {}} for j in range(nc)]}
                el, _ = self._timed(
                    "POST", "/v1/ingest/vector",
                    files={"file": ("t.geojson", json.dumps(geo).encode(), "application/geo+json")})
                lats.append(el)
            m = round(np.mean(lats), 2)
            res.append({"n_cells": nc, "mean_ms": m})
            print(f"  {nc:3d} cells => {m}ms")
        self.results["endpoint_latency"]["ingest_vector"] = res

    def bench_pipeline(self, name, cfg, max_samples=None):
        ms = max_samples or cfg["max_samples"]
        print(f"\n{'='*60}")
        print(f"Dataset: {name} — {cfg['desc']}")
        print(f"{'='*60}")
        print(f"  Loading {cfg['path']}...")
        ds = load_dataset(cfg["path"], split=cfg["split"], streaming=True)

        samples = list(itertools.islice(ds, ms))
        n = len(samples)
        print(f"  Samples: {n}")

        per = []
        errs = 0
        for idx, s in enumerate(samples):
            try:
                img = s[cfg["image_key"]]
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                buf.seek(0)
                el, result = self._timed(
                    "POST", "/analyze-image",
                    files={"file": (f"s{idx}.png", buf, "image/png")})
                malignant = sum(1 for c in result.get("cells", []) if c.get("is_cancer"))
                per.append({
                    "idx": idx,
                    "total_ms": el,
                    "cells": result.get("nodes_count", 0),
                    "malignant": malignant,
                    "timing": result.get("timing_ms", {}),
                })
                buf.close()
            except Exception as e:
                errs += 1
                per.append({"idx": idx, "error": str(e)[:300]})
                if errs > 10:
                    break
            if (idx + 1) % 15 == 0:
                print(f"  {idx+1}/{n} | err={errs}")

        ok = [s for s in per if "error" not in s]
        if not ok:
            return {"error": "all failed", "n_errors": errs}

        lats = [s["total_ms"] for s in ok]
        cells_per = [s["cells"] for s in ok]
        stats = {
            "samples": n, "completed": len(ok), "errors": errs,
            "total_cells": sum(s["cells"] for s in ok),
            "total_malignant": sum(s["malignant"] for s in ok),
            "cells_per_image_mean": round(np.mean(cells_per), 1),
            "latency_ms": {
                "mean": round(np.mean(lats), 1),
                "p50": round(np.percentile(lats, 50), 1),
                "p95": round(np.percentile(lats, 95), 1),
                "p99": round(np.percentile(lats, 99), 1),
            },
            "throughput_img_s": round(1.0 / (np.mean(lats) / 1000.0), 2),
        }
        print(f"  OK: {stats['latency_ms']['mean']}ms avg | "
              f"p95={stats['latency_ms']['p95']}ms | "
              f"{stats['throughput_img_s']} img/s | "
              f"{stats['cells_per_image_mean']} cells/img")
        return stats

    def save(self):
        Path(self.output).write_text(json.dumps(self.results, indent=2, default=str), encoding="utf-8")

    def run(self, dset_filter=None):
        print("\n=== ENDPOINT ===")
        self.bench_vector()
        print("\n=== PIPELINE ===")
        names = [dset_filter] if dset_filter else list(DATASETS.keys())
        for n in names:
            if n not in DATASETS:
                print(f"Skip {n}")
                continue
            try:
                self.results["datasets"][n] = self.bench_pipeline(n, DATASETS[n])
                self.save()
            except Exception as e:
                import traceback
                self.results["datasets"][n] = {"error": str(e)[:500]}
        lats = [d["latency_ms"]["mean"] for d in self.results["datasets"].values()
                if isinstance(d, dict) and "latency_ms" in d]
        if lats:
            self.results["summary"]["avg_latency_ms"] = round(np.mean(lats), 1)
        self.save()
        print(f"\nDone. Results: {self.output}")


def main():
    args = parse_args()
    Runner(args.host, args.port, args.output).run(dset_filter=args.dataset)


if __name__ == "__main__":
    main()