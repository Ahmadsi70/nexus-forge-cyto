"""Unit tests for multi-vendor vector ingest (Stage A parsers)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_SERVICES = Path(__file__).resolve().parents[1] / "services"
if str(_SERVICES) not in sys.path:
    sys.path.insert(0, str(_SERVICES))

from nexus_core.vector_ingest import (  # noqa: E402
    FORMAT_APERIO_XML,
    FORMAT_ASAP_XML,
    FORMAT_QUPATH_GEOJSON,
    detect_vector_format,
    parse_vector_bytes,
    payload_from_records,
)


APERIO_XML = b"""<?xml version="1.0"?>
<Annotations>
  <Region Id="1">
    <Vertex X="0" Y="0"/>
    <Vertex X="10" Y="0"/>
    <Vertex X="10" Y="10"/>
    <Vertex X="0" Y="10"/>
  </Region>
</Annotations>
"""

ASAP_XML = b"""<?xml version="1.0"?>
<ASAPAnnotations>
  <Annotation Id="a1">
    <Coordinate X="1" Y="2"/>
    <Coordinate X="5" Y="2"/>
    <Coordinate X="5" Y="6"/>
    <Coordinate X="1" Y="6"/>
  </Annotation>
</ASAPAnnotations>
"""

QUPATH_GEOJSON = json.dumps(
    {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[0, 0], [8, 0], [8, 8], [0, 8], [0, 0]]],
                },
                "properties": {"classification": {"name": "Tumor"}},
            }
        ],
    }
).encode("utf-8")


def test_detect_aperio_xml_variant() -> None:
    assert detect_vector_format("sample.xml", APERIO_XML) == FORMAT_APERIO_XML


def test_detect_asap_xml_variant() -> None:
    assert detect_vector_format("sample.xml", ASAP_XML) == FORMAT_ASAP_XML


def test_parse_qupath_geojson_extracts_class_name() -> None:
    records, fmt = parse_vector_bytes("cells.geojson", QUPATH_GEOJSON)
    assert fmt == FORMAT_QUPATH_GEOJSON
    assert len(records) == 1
    assert records[0]["source_class"] == "Tumor"
    payload = payload_from_records(records)
    assert payload["cell_count"] == 1
    assert len(payload["cells"][0]) == 32


def test_parse_asap_xml_produces_rings() -> None:
    records, fmt = parse_vector_bytes("asap.xml", ASAP_XML)
    assert fmt == FORMAT_ASAP_XML
    assert len(records) == 1
    assert len(records[0]["ring"]) == 32


def test_parse_aperio_xml_produces_rings() -> None:
    records, fmt = parse_vector_bytes("aperio.xml", APERIO_XML)
    assert fmt == FORMAT_APERIO_XML
    assert len(records) == 1
    assert len(records[0]["ring"]) == 32
