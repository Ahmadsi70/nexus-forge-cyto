"""HTTP API: dual ingress (image / vector) and Rust geometry enrichment."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

_REPO = Path(__file__).resolve().parents[1]
_SERVICES = _REPO / "services"
if str(_SERVICES) not in sys.path:
    sys.path.insert(0, str(_SERVICES))

from api_service.main import app  # noqa: E402
from nexus_core.coord_adapter import TARGET_RING  # noqa: E402

client = TestClient(app)

_MIN_GOLD = {
    "cell_count": 1,
    "cells": [[[float(i), float(i + 1)] for i in range(TARGET_RING)]],
}


def _fake_enriched(gold: dict) -> dict:
    return {
        **gold,
        "clinical": ["Normal"],
        "spatial_features": [{"cell_id": 0, "area": 100.0, "circularity": 0.9}],
    }


def test_health_ok() -> None:
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_ingest_vector_geojson() -> None:
    geo = {
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
    raw = json.dumps(geo).encode("utf-8")
    r = client.post(
        "/v1/ingest/vector",
        files={"file": ("sample.geojson", raw, "application/geo+json")},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["cell_count"] >= 1
    assert len(body["gold_segment"]["cells"]) == body["cell_count"]


def test_ingest_vector_gold_json_passthrough() -> None:
    raw = json.dumps(_MIN_GOLD).encode("utf-8")
    r = client.post(
        "/v1/ingest/vector",
        files={"file": ("gold_segment.json", raw, "application/json")},
    )
    assert r.status_code == 200
    assert r.json()["cell_count"] == 1


@patch("api_service.main.run_rust_enrichment")
def test_enrich_returns_spatial_payload(mock_enrich) -> None:
    mock_enrich.return_value = _fake_enriched(_MIN_GOLD)
    r = client.post("/v1/enrich", json={"gold_segment": _MIN_GOLD})
    assert r.status_code == 200
    out = r.json()
    assert "enriched" in out
    assert out["enriched"]["clinical"] == ["Normal"]
    mock_enrich.assert_called_once()


@patch("api_service.main.segment_image_bytes")
def test_ingest_image_uses_sam3(mock_segment) -> None:
    ring = [[float(i), float(i + 1)] for i in range(TARGET_RING)]
    mock_segment.return_value = ([[(p[0], p[1]) for p in ring]], "sam3_nuclei_derived")
    r = client.post(
        "/v1/ingest/image",
        files={"file": ("patch.png", b"\x89PNG\r\n\x1a\n", "image/png")},
    )
    assert r.status_code == 200
    assert r.json()["cell_count"] == 1
    assert r.json()["source"] == "sam3_nuclei_derived"
    mock_segment.assert_called_once()


@patch("api_service.main.segment_image_bytes")
def test_ingest_image_text_prompt_passed(mock_segment) -> None:
    ring = [[float(i), float(i + 1)] for i in range(TARGET_RING)]
    mock_segment.return_value = ([[(p[0], p[1]) for p in ring]], "sam3_nuclei_derived")
    r = client.post(
        "/v1/ingest/image?text_prompt=nucleus",
        files={"file": ("patch.png", b"\x89PNG\r\n\x1a\n", "image/png")},
    )
    assert r.status_code == 200
    assert r.json()["source"] == "sam3_nuclei_derived"
    mock_segment.assert_called_once()
    _, kwargs = mock_segment.call_args
    assert kwargs["text_prompt"] == "nucleus"


@patch("api_service.main.run_headless_schema_export")
def test_export_returns_stage_v2_final(mock_export) -> None:
    mock_export.return_value = {**_MIN_GOLD, "spatial_features": []}
    enriched = _fake_enriched(_MIN_GOLD)
    r = client.post("/v1/export", json={"enriched": enriched})
    assert r.status_code == 200
    out = r.json()
    assert "stage_v2_final" in out
    assert out["cell_count"] == 1
    mock_export.assert_called_once()
