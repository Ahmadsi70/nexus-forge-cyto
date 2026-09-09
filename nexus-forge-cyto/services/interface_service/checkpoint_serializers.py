"""
Checkpoint artifact serializers for decoupled pathology pipeline stages.

Why: emit vendor-native QuPath GeoJSON, compressed biostat tables, and signed
checkpoint bundles so humans can hand-carry artifacts between air-gapped stages.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

_SERVICES = Path(__file__).resolve().parents[1]
if str(_SERVICES) not in sys.path:
    sys.path.insert(0, str(_SERVICES))
from nexus_core.runtime_paths import portable_artifact_ref  # noqa: E402

logger = logging.getLogger(__name__)

GNN_CLASS_RGB: dict[str, list[int]] = {
    "Epithelial_Malignant": [229, 57, 53],   # red        — Malignant
    "Inflammatory": [255, 179, 0],            # amber      — other immune (macrophages, neutrophils)
    "Lymphocyte": [30, 136, 229],             # blue       — lymphocytes (was [255,193,7], visually
                                              #             near-identical to Inflammatory's amber).
                                              #             Use distinct blue per immune-gate
                                              #             convention so pathologists can tell the two
                                              #             immune subclasses apart at a glance.
    "Stromal": [67, 160, 71],                 # green      — Normal stroma
    "Malignant": [229, 57, 53],
    "Normal": [67, 160, 71],
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_checkpoint_manifest(artifacts: dict[str, Path]) -> dict[str, Any]:
    """Build SHA-256 manifest for cross-stage integrity validation."""
    manifest_artifacts: dict[str, Any] = {}
    for name, path in artifacts.items():
        if not path.is_file():
            continue
        manifest_artifacts[name] = {
            "sha256": _sha256_file(path),
            "bytes": int(path.stat().st_size),
        }
    return {
        "schema_version": "checkpoint_bundle_v1",
        "produced_at": datetime.now(timezone.utc).isoformat(),
        "artifacts": manifest_artifacts,
    }


def write_checkpoint_bundle_zip(
    output_zip: Path,
    *,
    stage_v2: Path,
    graph_pt: Path,
    extra_files: dict[str, Path] | None = None,
) -> Path:
    """Package Stage-A checkpoint files with cryptographic manifest.json."""
    files: dict[str, Path] = {
        "stage_v2_final.json": stage_v2,
        "graph_data.pt": graph_pt,
    }
    if extra_files:
        files.update(extra_files)

    manifest = build_checkpoint_manifest(files)
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    manifest_bytes = json.dumps(manifest, indent=2).encode("utf-8")

    with zipfile.ZipFile(output_zip, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("manifest.json", manifest_bytes)
        for arcname, path in files.items():
            if path.is_file():
                zf.write(path, arcname=arcname)
    return output_zip


def _finite_ring_coords(ring: Any) -> list[list[float]] | None:
    """Coerce ring vertices to finite floats; return None when malformed or non-finite."""
    try:
        coords = [[float(x), float(y)] for x, y in ring]
    except (ValueError, TypeError):
        return None
    for x, y in coords:
        if not math.isfinite(x) or not math.isfinite(y):
            return None
    return coords


def build_qupath_from_enriched(stage_path: Path) -> dict[str, Any]:
    """
    Geometry-only QuPath FeatureCollection from enriched/stage JSON.

    Why: export works without GNN prediction files — uses clinical[] labels from Rust κ.
    """
    stage = json.loads(stage_path.read_text(encoding="utf-8"))
    cells: list = stage.get("cells") or []
    clinical: list = stage.get("clinical") or []
    features: list[dict[str, Any]] = []
    for i, ring in enumerate(cells):
        label = str(clinical[i]) if i < len(clinical) else "Normal"
        if label not in {"Malignant", "Normal"}:
            label = "Normal"
        rgb = GNN_CLASS_RGB.get(label, [200, 200, 200])
        coords = _finite_ring_coords(ring)
        if coords is None or not coords:
            continue
        if coords[0] != coords[-1]:
            coords.append(coords[0])
        features.append(
            {
                "type": "Feature",
                "id": f"Cell_{i}",
                "geometry": {"type": "Polygon", "coordinates": [coords]},
                "properties": {
                    "objectType": "annotation",
                    "name": f"Cell_{i}",
                    "classification": {"name": label, "color": rgb},
                    "nexus_cell_id": i,
                },
            }
        )
    return {
        "type": "FeatureCollection",
        "features": features,
        "properties": {
            "schema_version": "qupath_annotations_geometry_v1",
            "source_stage": portable_artifact_ref(stage_path),
        },
    }


def write_qupath_from_enriched(stage_path: Path, output_path: Path) -> Path:
    """Persist geometry-only QuPath annotations beside other export artifacts."""
    payload = build_qupath_from_enriched(stage_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path


def build_qupath_annotations_geojson(
    stage_path: Path,
    predictions_path: Path,
) -> dict[str, Any]:
    """Build QuPath v0.6+ drag-and-drop FeatureCollection with GNN class names."""
    stage = json.loads(stage_path.read_text(encoding="utf-8"))
    preds_raw = json.loads(predictions_path.read_text(encoding="utf-8"))
    preds = preds_raw if isinstance(preds_raw, list) else preds_raw.get("records", [])

    cells: list = stage["cells"]
    if len(preds) != len(cells):
        logger.error(
            "Prediction/cell count mismatch in GeoJSON export: %s predictions vs %s cells",
            len(preds),
            len(cells),
        )

    features: list[dict[str, Any]] = []
    for i, ring in enumerate(cells):
        if i >= len(preds):
            logger.warning("Skipping cell %s — no matching prediction row", i)
            continue
        prow = preds[i]
        gnn_class = str(prow.get("predicted_class", "Stromal"))
        rgb = GNN_CLASS_RGB.get(gnn_class, [200, 200, 200])
        coords = _finite_ring_coords(ring)
        if coords is None:
            logger.warning(
                "Skipping cell %s with malformed, null, or non-finite coordinates in GeoJSON export",
                i,
            )
            continue
        if not coords:
            logger.warning("Skipping cell %s with empty coordinate ring in GeoJSON export", i)
            continue
        if coords[0] != coords[-1]:
            coords.append(coords[0])
        try:
            confidence = float(prow.get("confidence", 0.0))
            cell_id = int(prow.get("cell_id", i))
        except (ValueError, TypeError):
            logger.warning("Skipping cell %s with malformed prediction metadata", i)
            continue
        features.append(
            {
                "type": "Feature",
                "id": f"Cell_{i}",
                "geometry": {"type": "Polygon", "coordinates": [coords]},
                "properties": {
                    "objectType": "annotation",
                    "name": f"Cell_{i}",
                    "classification": {
                        "name": gnn_class,
                        "color": rgb,
                    },
                    "confidence": confidence,
                    "nexus_cell_id": cell_id,
                },
            }
        )

    return {
        "type": "FeatureCollection",
        "features": features,
        "properties": {
            "schema_version": "qupath_annotations_v1",
            "source_stage": portable_artifact_ref(stage_path),
            "source_predictions": portable_artifact_ref(predictions_path),
        },
    }


def write_qupath_annotations_geojson(
    stage_path: Path,
    predictions_path: Path,
    output_path: Path,
) -> Path:
    """Persist native QuPath-loadable annotation FeatureCollection."""
    payload = build_qupath_annotations_geojson(stage_path, predictions_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path


def _mean_incoming_attention(attention_csv: Path) -> dict[int, float]:
    if not attention_csv.is_file():
        return {}
    df = pd.read_csv(attention_csv)
    if "target_node" not in df.columns or "alpha_weight" not in df.columns:
        return {}
    if "layer" in df.columns:
        df = df[df["layer"] == "layer2"]
    if df.empty:
        return {}
    agg = df.groupby("target_node")["alpha_weight"].mean()
    return {int(k): float(v) for k, v in agg.items()}


def build_biostat_parquet_table(
    stage_path: Path,
    predictions_path: Path,
    attention_csv: Path,
) -> pd.DataFrame:
    """Merge morphometric traits, centroids, and GAT attention into one biostat table."""
    stage = json.loads(stage_path.read_text(encoding="utf-8"))
    preds_raw = json.loads(predictions_path.read_text(encoding="utf-8"))
    preds = preds_raw if isinstance(preds_raw, list) else preds_raw.get("records", [])
    spatial = stage.get("spatial_features", [])
    attn = _mean_incoming_attention(attention_csv)

    rows: list[dict[str, Any]] = []
    for i, sf in enumerate(spatial):
        prow = preds[i] if i < len(preds) else {}
        try:
            centroid = sf.get("centroid", [0.0, 0.0])
            if isinstance(centroid, (list, tuple)) and len(centroid) >= 2:
                cx, cy = float(centroid[0]), float(centroid[1])
            else:
                cx, cy = 0.0, 0.0
            if not math.isfinite(cx) or not math.isfinite(cy):
                raise ValueError(f"non-finite centroid at spatial index {i}")
            row = {
                "cell_id": int(sf.get("cell_id", i)),
                "predicted_class": str(prow.get("predicted_class", "")),
                "clinical_label": str(prow.get("clinical_label", "")),
                "confidence": float(prow.get("confidence", 0.0)),
                "centroid_x": cx,
                "centroid_y": cy,
                "area": float(sf.get("area", 0.0)),
                "perimeter": float(sf.get("perimeter", 0.0)),
                "circularity": float(sf.get("circularity", 0.0)),
                "eccentricity": float(sf.get("eccentricity", 0.0)),
                "knn_density": float(sf.get("knn_density", 0.0)),
                "rbf_risk_score": float(sf.get("rbf_risk_score", 0.0)),
                "distance_to_tumor_edge": float(sf.get("distance_to_tumor_edge", 0.0)),
                "gat_mean_incoming_alpha": float(attn.get(i, float("nan"))),
            }
            for key, val in row.items():
                if isinstance(val, float) and not math.isfinite(val) and key != "gat_mean_incoming_alpha":
                    raise ValueError(f"non-finite {key} at spatial index {i}")
            rows.append(row)
        except (ValueError, TypeError) as exc:
            logger.warning("Skipping biostat row %s due to malformed morphometry: %s", i, exc)
            continue
    return pd.DataFrame(rows)


def write_biostat_parquet(
    stage_path: Path,
    predictions_path: Path,
    attention_csv: Path,
    output_path: Path,
) -> Path:
    """Write Apache Parquet biostatistical table for fast downstream analytics."""
    df = build_biostat_parquet_table(stage_path, predictions_path, attention_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, output_path, compression="snappy")
    return output_path


def write_stage_b_interop_bundle(
    output_dir: Path,
    *,
    stage_path: Path,
    predictions_path: Path,
    attention_csv: Path,
) -> dict[str, Path]:
    """Emit all Stage-B interoperability artifacts into output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "qupath_annotations": output_dir / "qupath_annotations.geojson",
        "biostat_parquet": output_dir / "biostat_morphometry.parquet",
    }
    write_qupath_annotations_geojson(stage_path, predictions_path, paths["qupath_annotations"])
    write_biostat_parquet(stage_path, predictions_path, attention_csv, paths["biostat_parquet"])
    return paths


def read_file_bytes_if_exists(path: Path) -> bytes | None:
    """Safe file read helper for Streamlit download gates."""
    return path.read_bytes() if path.is_file() else None
