"""UI ingest must emit the same gold-segment envelope as api_service.pipeline_core."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
_SERVICES = _REPO / "services"
if str(_SERVICES) not in sys.path:
    sys.path.insert(0, str(_SERVICES))

from api_service.pipeline_core import gold_segment_from_vector_bytes  # noqa: E402
from interface_service.dashboard_shared import (  # noqa: E402
    normalize_vector_upload_to_json_bytes,
)
from nexus_core.coord_adapter import TARGET_RING  # noqa: E402
from nexus_core.vector_ingest import FORMAT_JSON_STAGE, FORMAT_SAM3_NUCLEI  # noqa: E402


_GEOJSON = {
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]]],
            },
            "properties": {},
        }
    ],
}


def test_vector_upload_matches_pipeline_core_envelope() -> None:
    raw = json.dumps(_GEOJSON).encode("utf-8")
    ui_bytes, ui_fmt = normalize_vector_upload_to_json_bytes("sample.geojson", raw)
    api_gold = gold_segment_from_vector_bytes("sample.geojson", raw)
    ui_gold = json.loads(ui_bytes.decode("utf-8"))
    assert ui_gold == api_gold
    assert ui_fmt == api_gold["annotation_adapter"]["format"]
    assert "dynamic_calibration" in ui_gold
    assert ui_gold["cell_count"] >= 1


def test_json_stage_passthrough_unchanged() -> None:
    gold = {"cell_count": 1, "cells": [[[float(i), float(i + 1)] for i in range(TARGET_RING)]]}
    raw = json.dumps(gold).encode("utf-8")
    ui_bytes, ui_fmt = normalize_vector_upload_to_json_bytes("stage.json", raw)
    assert ui_fmt == FORMAT_JSON_STAGE
    assert json.loads(ui_bytes.decode("utf-8")) == gold


def test_sam3_format_constant_importable_from_dashboard_module() -> None:
    """Regression: write_image_bootstrap references FORMAT_SAM3_NUCLEI."""
    assert FORMAT_SAM3_NUCLEI == "sam3_nuclei_derived"
