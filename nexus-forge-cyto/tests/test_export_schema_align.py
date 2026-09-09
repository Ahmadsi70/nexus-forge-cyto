"""Regression: UI export kwargs, schema ring=128 + innovations, QuPath geometry export."""

from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path

import jsonschema
import pytest

_REPO = Path(__file__).resolve().parents[1]
_SERVICES = _REPO / "services"
if str(_SERVICES) not in sys.path:
    sys.path.insert(0, str(_SERVICES))

from api_service.pipeline_core import build_headless_export_cmd  # noqa: E402
from interface_service.checkpoint_serializers import (  # noqa: E402
    build_qupath_from_enriched,
    write_qupath_from_enriched,
)
from interface_service.dashboard_shared import build_graph_export_cmd  # noqa: E402
from nexus_core.coord_adapter import TARGET_RING  # noqa: E402


def _minimal_stage_payload() -> dict:
    """Schema-valid stage fixture aligned with headless `_bootstrap` defaults."""
    ring = [[float(i), float((i * 3) % 17)] for i in range(TARGET_RING)]
    return {
        "schema_version": "2.0.0",
        "cell_count": 1,
        "cells": [ring],
        "clinical": ["Normal"],
        "spatial_analysis": {
            "knn_k": 8,
            "rbf_sigma": 1.0,
            "knn_metric": "mean_euclidean_distance_to_k_neighbors",
        },
        "spatial_features": [
            {
                "cell_id": 0,
                "centroid": [0.0, 0.0],
                "knn_density": 0.0,
                "rbf_risk_score": 0.0,
                "area": 1.0,
                "perimeter": 1.0,
                "circularity": 1.0,
                "eccentricity": 0.0,
                "distance_to_tumor_edge": 0.0,
            }
        ],
        "cell_biology": {
            "class_schema_version": "1.0.0",
            "source": "unit_test",
            "labels": ["Stromal"],
            "confidence": [0.0],
        },
        "advanced_spatial_stats": {
            "clustering_index": {
                "global_ci": 1.0,
                "classwise_ci": {"Stromal": 1.0},
            },
            "mixing_score": {
                "value": 0.0,
                "immune_tumor_edges": 0,
                "eligible_edges": 1,
            },
            "parameters": {
                "knn_k": 8,
                "ci_reference": "csr_poisson_nn",
                "mixing_policy": "immune_tumor_boundary_edge_ratio",
            },
        },
        "graph_export_meta": {
            "node_feature_order": ["area"],
            "node_feature_dim": 1,
            "edge_policy": {"kind": "knn", "k": 8},
            "tensor_dtypes": {"x": "float32", "edge_index": "int64"},
            "export_format": "pyg_data_dict_v1",
        },
        "innovations": {"kappa_mean": 0.0},
    }


def test_build_headless_export_cmd_kwarg_is_schema_path_override() -> None:
    sig = inspect.signature(build_headless_export_cmd)
    assert "schema_path_override" in sig.parameters
    assert "schema_path" not in sig.parameters


def test_dashboard_export_cmd_passes_schema_override(tmp_path: Path) -> None:
    enriched = tmp_path / "enriched.json"
    enriched.write_text("{}", encoding="utf-8")
    _cmd, env = build_graph_export_cmd(enriched)
    schema = Path(env["NEXUS_SCHEMA_PATH"])
    assert schema.name == "enriched_graph_v2.schema.json"
    assert schema.is_file()
    assert env["NEXUS_INPUT_JSON"] == str(enriched)


def test_schema_accepts_128_ring_and_innovations() -> None:
    schema = json.loads(
        (_REPO / "schemas" / "enriched_graph_v2.schema.json").read_text(encoding="utf-8")
    )
    cells_items = schema["properties"]["cells"]["items"]
    assert cells_items["minItems"] == TARGET_RING
    assert cells_items["maxItems"] == TARGET_RING
    assert "innovations" in schema["properties"]
    jsonschema.validate(instance=_minimal_stage_payload(), schema=schema)


def test_schema_rejects_wrong_ring_length() -> None:
    schema = json.loads(
        (_REPO / "schemas" / "enriched_graph_v2.schema.json").read_text(encoding="utf-8")
    )
    bad = _minimal_stage_payload()
    bad["cells"] = [[[0.0, 0.0]] * 32]
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(instance=bad, schema=schema)


def test_qupath_geometry_export(tmp_path: Path) -> None:
    stage = tmp_path / "stage.json"
    stage.write_text(json.dumps(_minimal_stage_payload()), encoding="utf-8")
    fc = build_qupath_from_enriched(stage)
    assert fc["type"] == "FeatureCollection"
    assert len(fc["features"]) == 1
    assert fc["features"][0]["properties"]["classification"]["name"] == "Normal"
    out = write_qupath_from_enriched(stage, tmp_path / "qupath_annotations.geojson")
    assert out.is_file()
    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert loaded["features"][0]["geometry"]["type"] == "Polygon"
