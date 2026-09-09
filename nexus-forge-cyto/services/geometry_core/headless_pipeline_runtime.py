import csv
import io
import json
import math
import os
import sys
from pathlib import Path

import jsonschema
import numpy as np
import pandas as pd
from scipy.spatial import KDTree

_SERVICES = Path(__file__).resolve().parents[1]
if str(_SERVICES) not in sys.path:
    sys.path.insert(0, str(_SERVICES))
from nexus_core.region_walker import (  # noqa: E402
    build_manifest,
    write_manifest,
)
from nexus_core.coord_adapter import (               # noqa: E402  — canonical ring impl
    ring_from_polygon_xy as _ring_from_polygon,       # polygon → ring
    _polygon_area_shoelace,                             # shoelace area
    radius_from_mapping as _radius_from_mapping,        # radius heuristic (canonical)
    TARGET_RING,                                        # 128
)
from nexus_core.vector_ingest import (  # noqa: E402
    FORMAT_JSON_STAGE,
    detect_vector_format,
    records_from_vector_path,
)
from nexus_core.runtime_paths import (  # noqa: E402
    default_input_json,
    hovernet_instances_path,
    production_output_dir,
    repo_root,
    schema_path,
)

# Geometry-only: skip HoverNet projection and PyG export (no torch/GNN).
GEOMETRY_ONLY_MODE = True

CLASS_TO_ID = {
    "Epithelial_Malignant": 0,
    "Inflammatory": 1,
    "Lymphocyte": 2,
    "Stromal": 3,
}


def _repo_root() -> Path:
    return repo_root()


def _paths() -> dict[str, Path]:
    root = _repo_root()
    input_json = default_input_json()
    out_dir = production_output_dir()
    schema = schema_path()
    hovernet_instances = hovernet_instances_path()
    return {
        "input": input_json,
        "schema": schema,
        "hovernet_instances": hovernet_instances,
        "out_dir": out_dir,
        "boot": out_dir / "stage_v2_bootstrap.json",
        "proj": out_dir / "stage_v2_projection.json",
        "spa": out_dir / "stage_v2_stats.json",
        "fin": out_dir / "stage_v2_final.json",
        "pt": out_dir / "graph_data.pt",
        "manifest": out_dir / "annotation_index_manifest.json",
    }


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _ring_from_centroid(cx: float, cy: float, radius: float = 4.0) -> list[list[float]]:
    """Generate a regular TARGET_RING-vertex ring centred at (cx, cy)."""
    out: list[list[float]] = []
    for i in range(TARGET_RING):
        t = (2.0 * math.pi * i) / TARGET_RING
        out.append([cx + radius * math.cos(t), cy + radius * math.sin(t)])
    return out


def _polygon_area_perimeter(ring: list[list[float]]) -> tuple[float, float]:
    """Area (shoelace) + perimeter for a 32-ring.  Delegates area to coord_adapter."""
    area = _polygon_area_shoelace(ring)
    n = len(ring)
    if n < 3:
        return (area, 0.0)
    perim = 0.0
    for i in range(n):
        j = (i + 1) % n
        perim += float(math.hypot(
            float(ring[j][0]) - float(ring[i][0]),
            float(ring[j][1]) - float(ring[i][1]),
        ))
    return (area, perim)


# NOTE: `_radius_from_mapping` is imported from coord_adapter above (single source of truth).
# The previous local copy was removed to eliminate duplication drift.


def _records_from_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        x_key = next((k for k in reader.fieldnames or [] if k.lower() in {"x", "centroid_x", "cx"}), None)
        y_key = next((k for k in reader.fieldnames or [] if k.lower() in {"y", "centroid_y", "cy"}), None)
        out: list[dict] = []
        for row in reader:
            if x_key and y_key:
                x = float(row[x_key])
                y = float(row[y_key])
            else:
                vals = [row[h] for h in (reader.fieldnames or []) if row[h] is not None and str(row[h]).strip()]
                if len(vals) < 2:
                    continue
                x = float(vals[0])
                y = float(vals[1])
            r = _radius_from_mapping(row)
            out.append({"centroid": (x, y), "ring": _ring_from_centroid(x, y, r)})
        return out


def _load_input_payload(input_path: Path, knn_k: int, manifest_path: Path | None = None) -> dict:
    suf = input_path.suffix.lower()
    if suf in {".json", ".geojson", ".xml"}:
        raw = input_path.read_bytes()
        fmt = detect_vector_format(input_path.name, raw)
        if fmt == FORMAT_JSON_STAGE:
            return _load(input_path)
        records, source_format = records_from_vector_path(input_path)
        print(f"[INFO] canonicalized {source_format} -> v2 payload (n={len(records)})")
        if manifest_path is not None:
            write_manifest(build_manifest(input_path, source_format, records), manifest_path)
        return _canonical_payload_from_records(records, input_path, source_format, knn_k)
    if suf == ".csv":
        records = _records_from_csv(input_path)
        print(f"[INFO] canonicalized CSV vectors -> v2 payload (n={len(records)})")
        return _canonical_payload_from_records(records, input_path, "csv_vectors", knn_k)
    if suf == ".html":
        records = _records_from_html(input_path)
        print(f"[INFO] canonicalized HTML tables -> v2 payload (n={len(records)})")
        return _canonical_payload_from_records(records, input_path, "html_table_vectors", knn_k)
    raise ValueError(
        f"Unsupported input format {suf!r}. Allowed: .json, .geojson, .csv, .xml, .html"
    )


def _records_from_html(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    out: list[dict] = []
    try:
        tables = pd.read_html(io.StringIO(text))
    except Exception:
        tables = []
    for df in tables:
        df.columns = [str(c).strip().lower() for c in df.columns]
        x_col = next((c for c in df.columns if c in {"x", "centroid_x", "cx"}), None)
        y_col = next((c for c in df.columns if c in {"y", "centroid_y", "cy"}), None)
        if not x_col or not y_col:
            continue
        for _, row in df.iterrows():
            x = float(row[x_col])
            y = float(row[y_col])
            r = _radius_from_mapping(row.to_dict())
            out.append({"centroid": (x, y), "ring": _ring_from_centroid(x, y, r)})
    return out


def _canonical_payload_from_records(
    records: list[dict], source_path: Path, source_format: str, knn_k: int
) -> dict:
    n = len(records)
    if n == 0:
        raise ValueError(f"No vector records parsed from {source_path}")

    cells = [r["ring"] for r in records]
    centroids = [r["centroid"] for r in records]

    return {
        "cell_count": n,
        "cells": cells,
        "clinical": ["Normal"] * n,
        "spatial_analysis": {
            "knn_k": int(knn_k),
            "rbf_sigma": 1.0,
            "knn_metric": "mean_euclidean_distance_to_k_neighbors",
        },
        "spatial_features": [
            {
                "cell_id": i,
                "centroid": [float(cx), float(cy)],
                "knn_density": 0.0,
                "rbf_risk_score": 0.0,
                "area": _polygon_area_perimeter(cells[i])[0],
                "perimeter": _polygon_area_perimeter(cells[i])[1],
                "circularity": 1.0,
                "eccentricity": 0.0,
                "distance_to_tumor_edge": 0.0,
            }
            for i, (cx, cy) in enumerate(centroids)
        ],
    }


def _save(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _validate(payload: dict, schema: dict, stage: str) -> None:
    jsonschema.validate(instance=payload, schema=schema)
    print(f"[PASS] schema gate: {stage}")


def _biology_labels_from_clinical(clinical: list[str]) -> list[str]:
    """Map Rust clinical strings to schema biology labels for mixing stats."""
    return [
        "Epithelial_Malignant" if c == "Malignant" else "Stromal"
        for c in clinical
    ]


def _bootstrap(payload: dict) -> dict:
    n = int(payload["cell_count"])
    payload["schema_version"] = "2.0.0"
    payload.setdefault(
        "spatial_analysis",
        {
            "knn_k": 8,
            "rbf_sigma": 1.0,
            "knn_metric": "mean_euclidean_distance_to_k_neighbors",
        },
    )
    clinical = payload.get("clinical", ["Normal"] * n)
    if len(clinical) != n:
        clinical = ["Normal"] * n
    if GEOMETRY_ONLY_MODE and "clinical" in payload:
        bio_labels = _biology_labels_from_clinical(clinical)
        bio_source = "clinical_sync_v2"
        bio_conf = [1.0 if c == "Malignant" else 0.0 for c in clinical]
    else:
        bio_labels = ["Stromal"] * n
        bio_source = "bootstrap_v2_default"
        bio_conf = [0.0] * n
    payload["cell_biology"] = {
        "class_schema_version": "1.0.0",
        "source": bio_source,
        "labels": bio_labels,
        "confidence": bio_conf,
        "projection": {"method": "kdtree_iou_fallback", "max_distance_px": 64.0, "iou_threshold": 0.05},
    }
    payload["advanced_spatial_stats"] = {
        "clustering_index": {
            "global_ci": 1.0,
            "classwise_ci": {
                "Epithelial_Malignant": 1.0,
                "Inflammatory": 1.0,
                "Lymphocyte": 1.0,
                "Stromal": 1.0,
            },
        },
        "mixing_score": {"value": 0.0, "immune_tumor_edges": 0, "eligible_edges": 1},
        "parameters": {
            "knn_k": int(payload["spatial_analysis"]["knn_k"]),
            "ci_reference": "csr_poisson_nn",
            "mixing_policy": "immune_tumor_boundary_edge_ratio",
        },
    }
    payload["graph_export_meta"] = {
        "node_feature_order": [
            "area",
            "perimeter",
            "circularity",
            "eccentricity",
            "knn_density",
            "rbf_risk_score",
            "distance_to_tumor_edge",
            "bio_confidence",
            "bio_is_epithelial_malignant",
            "bio_is_inflammatory",
            "bio_is_lymphocyte",
            "bio_is_stromal",
        ],
        "node_feature_dim": 12,
        "edge_policy": {"kind": "knn", "k": int(payload["spatial_analysis"]["knn_k"])},
        "tensor_dtypes": {"x": "float32", "edge_index": "int64"},
        "export_format": "pyg_data_dict_v1",
        "target_label_key": "clinical",
    }
    return payload


def _bbox_from_ring(ring: list[list[float]]) -> tuple[float, float, float, float]:
    xs = [float(p[0]) for p in ring]
    ys = [float(p[1]) for p in ring]
    return min(xs), min(ys), max(xs), max(ys)


def _bbox_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = (ix1 - ix0), (iy1 - iy0)
    if iw <= 0.0 or ih <= 0.0:
        return 0.0
    inter = iw * ih
    ua = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
    if ua <= 1e-12:
        return 0.0
    return inter / ua


def _load_instances(payload: dict, hovernet_instances_path: Path) -> list[dict]:
    if hovernet_instances_path.is_file():
        raw = _load(hovernet_instances_path)
        items = raw.get("instances", [])
        print(f"[INFO] projection source: {hovernet_instances_path} (instances={len(items)})")
        return items

    # Fallback synthetic instances from existing biological/clinical hints to keep pipeline deterministic.
    spatial = payload["spatial_features"]
    labels = payload["cell_biology"].get("labels", [])
    clinical = payload["clinical"]
    items = []
    for i, row in enumerate(spatial):
        cx, cy = float(row["centroid"][0]), float(row["centroid"][1])
        if i < len(labels):
            lab = labels[i]
        else:
            lab = "Epithelial_Malignant" if clinical[i] == "Malignant" else "Stromal"
        items.append(
            {
                "centroid": [cx, cy],
                "bbox": [cx - 4.0, cy - 4.0, cx + 4.0, cy + 4.0],
                "class_label": lab,
                "confidence": 0.75,
            }
        )
    print(f"[INFO] projection source: synthetic fallback (instances={len(items)})")
    return items


def _project(payload: dict, hovernet_instances_path: Path, max_distance_px: float = 64.0, iou_threshold: float = 0.05) -> dict:
    n = int(payload["cell_count"])
    centroids = np.asarray([[float(r["centroid"][0]), float(r["centroid"][1])] for r in payload["spatial_features"]], dtype=np.float64)
    tree = KDTree(centroids)
    cell_bboxes = [_bbox_from_ring(ring) for ring in payload["cells"]]
    instances = _load_instances(payload, hovernet_instances_path)

    best_conf = np.zeros((n,), dtype=np.float64)
    labels = ["Stromal"] * n
    mapped_by_kdtree = 0
    mapped_by_iou = 0

    for inst in instances:
        cx, cy = float(inst["centroid"][0]), float(inst["centroid"][1])
        label = str(inst["class_label"])
        conf = float(inst.get("confidence", 0.0))
        dist, idx = tree.query([cx, cy], k=1)
        target = int(idx)
        via_iou = False

        if float(dist) > max_distance_px:
            ibox = tuple(float(v) for v in inst.get("bbox", [cx - 4.0, cy - 4.0, cx + 4.0, cy + 4.0]))
            ious = np.asarray([_bbox_iou(ibox, b) for b in cell_bboxes], dtype=np.float64)
            best_idx = int(np.argmax(ious))
            best_iou = float(ious[best_idx])
            if best_iou >= iou_threshold:
                target = best_idx
                via_iou = True
            else:
                continue

        if conf >= best_conf[target]:
            labels[target] = label
            best_conf[target] = conf
            if via_iou:
                mapped_by_iou += 1
            else:
                mapped_by_kdtree += 1

    payload["cell_biology"]["labels"] = labels
    payload["cell_biology"]["confidence"] = best_conf.tolist()
    payload["cell_biology"]["source"] = "hovernet_projected_v1_kdtree_iou"
    payload["cell_biology"]["projection"] = {
        "method": "kdtree_iou_fallback",
        "max_distance_px": float(max_distance_px),
        "iou_threshold": float(iou_threshold),
    }
    print(
        "[PASS] project mapping: "
        f"kdtree={mapped_by_kdtree}, iou_fallback={mapped_by_iou}, "
        f"instances={len(instances)}, cells={n}, complexity=O(n log n)"
    )
    return payload


def _spatial(payload: dict) -> dict:
    centroids = [row["centroid"] for row in payload["spatial_features"]]
    labels = payload["cell_biology"]["labels"]
    n = len(centroids)
    if n <= 1:
        global_ci = 1.0
        nearest_idx = np.zeros((n,), dtype=np.int64)
    else:
        pts = np.asarray(centroids, dtype=np.float64)
        tree = KDTree(pts)
        dist, idx = tree.query(pts, k=2)
        nearest_dist = dist[:, 1]
        nearest_idx = idx[:, 1].astype(np.int64)
        obs = float(np.mean(nearest_dist))
        xs = pts[:, 0]
        ys = pts[:, 1]
        area = max(float((xs.max() - xs.min()) * (ys.max() - ys.min())), 1e-9)
        expected = 0.5 * math.sqrt(area / n)
        global_ci = obs / max(expected, 1e-9)

    immune = {"Inflammatory", "Lymphocyte"}
    tumor = {"Epithelial_Malignant"}
    immune_tumor_edges = 0
    eligible_edges = 0
    for i in range(n):
        li = labels[i]
        if n <= 1:
            continue
        lj = labels[int(nearest_idx[i])]
        if (li in immune) or (lj in immune) or (li in tumor) or (lj in tumor):
            eligible_edges += 1
            if ((li in immune) and (lj in tumor)) or ((lj in immune) and (li in tumor)):
                immune_tumor_edges += 1
    eligible_edges = max(1, eligible_edges)
    mixing_value = immune_tumor_edges / eligible_edges

    payload["advanced_spatial_stats"] = {
        "clustering_index": {
            "global_ci": float(global_ci),
            "classwise_ci": {
                "Epithelial_Malignant": float(global_ci),
                "Inflammatory": float(global_ci),
                "Lymphocyte": float(global_ci),
                "Stromal": float(global_ci),
            },
        },
        "mixing_score": {
            "value": float(mixing_value),
            "immune_tumor_edges": int(immune_tumor_edges),
            "eligible_edges": int(eligible_edges),
        },
        "parameters": {
            "knn_k": int(payload["spatial_analysis"]["knn_k"]),
            "ci_reference": "csr_poisson_nn",
            "mixing_policy": "immune_tumor_boundary_edge_ratio",
        },
    }
    return payload


def _knn_edge_index(centroids: np.ndarray, k: int) -> np.ndarray:
    n = centroids.shape[0]
    k_eff = max(1, min(k, n - 1))
    tree = KDTree(centroids)
    _, idx = tree.query(centroids, k=k_eff + 1)  # include self at position 0
    nbr = idx[:, 1:]  # drop self
    src = np.repeat(np.arange(n, dtype=np.int64), k_eff)
    dst = nbr.reshape(-1).astype(np.int64)
    return np.vstack([src, dst])


def _pyg(payload: dict, out_pt: Path, knn_k: int) -> dict:
    import torch  # optional AI path only; geometry-only skips this stage

    sf = payload["spatial_features"]
    labels = payload["cell_biology"]["labels"]
    conf = payload["cell_biology"]["confidence"]
    n = len(sf)
    x = np.empty((n, 12), dtype=np.float32)
    y = np.empty((n,), dtype=np.int64)
    for i, row in enumerate(sf):
        lab = labels[i]
        if lab not in CLASS_TO_ID:
            raise ValueError(
                f"Unknown canonical class label at node {i}: {lab!r}. "
                f"Allowed={sorted(CLASS_TO_ID.keys())}"
            )
        x[i] = np.asarray(
            [
                row["area"],
                row["perimeter"],
                row["circularity"],
                row["eccentricity"],
                row["knn_density"],
                row["rbf_risk_score"],
                row["distance_to_tumor_edge"],
                conf[i],
                1.0 if lab == "Epithelial_Malignant" else 0.0,
                1.0 if lab == "Inflammatory" else 0.0,
                1.0 if lab == "Lymphocyte" else 0.0,
                1.0 if lab == "Stromal" else 0.0,
            ],
            dtype=np.float32,
        )
        y[i] = CLASS_TO_ID[lab]

    centroids = np.asarray([[float(r["centroid"][0]), float(r["centroid"][1])] for r in sf], dtype=np.float64)
    edge_index = _knn_edge_index(centroids, knn_k)
    expected_edges = n * max(1, min(knn_k, n - 1))
    assert edge_index.shape == (2, expected_edges), f"edge_index shape {edge_index.shape} != (2, {expected_edges})"

    x_t = torch.from_numpy(x)
    edge_t = torch.from_numpy(edge_index)
    y_t = torch.from_numpy(y)
    torch.save({"x": x_t, "edge_index": edge_t, "y": y_t}, out_pt)
    print("[PASS] zero-copy path used: torch.from_numpy")

    payload["graph_export_meta"] = {
        "node_feature_order": [
            "area",
            "perimeter",
            "circularity",
            "eccentricity",
            "knn_density",
            "rbf_risk_score",
            "distance_to_tumor_edge",
            "bio_confidence",
            "bio_is_epithelial_malignant",
            "bio_is_inflammatory",
            "bio_is_lymphocyte",
            "bio_is_stromal",
        ],
        "node_feature_dim": 12,
        "edge_policy": {"kind": "knn", "k": int(knn_k)},
        "tensor_dtypes": {"x": "float32", "edge_index": "int64"},
        "export_format": "pyg_data_dict_v1",
        "target_label_key": "clinical",
    }
    print(f"[PASS] pyg edge_index shape: {tuple(edge_index.shape)} (k={knn_k})")
    return payload


def main() -> None:
    p = _paths()
    knn_k = int(os.environ.get("NEXUS_KNN_K", "8"))
    max_distance_px = float(os.environ.get("NEXUS_MAX_DISTANCE_PX", "64.0"))
    iou_threshold = float(os.environ.get("NEXUS_IOU_THRESHOLD", "0.05"))
    schema = _load(p["schema"])
    payload = _load_input_payload(p["input"], knn_k=knn_k, manifest_path=p["manifest"])
    payload = _bootstrap(payload)
    _validate(payload, schema, "bootstrap")
    _save(p["boot"], payload)

    if not GEOMETRY_ONLY_MODE:
        payload = _project(payload, p["hovernet_instances"], max_distance_px=max_distance_px, iou_threshold=iou_threshold)
        _validate(payload, schema, "project")
        _save(p["proj"], payload)

    payload = _spatial(payload)
    _validate(payload, schema, "spatial")
    _save(p["spa"], payload)

    if not GEOMETRY_ONLY_MODE:
        payload = _pyg(payload, p["pt"], knn_k=knn_k)
    if not GEOMETRY_ONLY_MODE:
        _validate(payload, schema, "pyg")
    _save(p["fin"], payload)
    print(f"[DONE] artifacts at {p['out_dir']}")


if __name__ == "__main__":
    main()
