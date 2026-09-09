"""Smoke tests for coord_adapter ring resampling and envelope parsing."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

_SERVICES = Path(__file__).resolve().parents[1] / "services"
if str(_SERVICES) not in sys.path:
    sys.path.insert(0, str(_SERVICES))

from nexus_core.coord_adapter import (  # noqa: E402
    TARGET_RING,
    ring32_from_polygon_xy,
    segment_envelope_from_file,
)


def test_resample_ring32_square() -> None:
    sq = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]
    ring = ring32_from_polygon_xy(sq)
    assert len(ring) == TARGET_RING
    assert len(ring[0]) == 2


def test_xml_parse_envelope() -> None:
    xml = """<?xml version='1.0'?>
    <Annotations><Annotation><Regions><Region><Vertices>
        <Vertex X="0" Y="0"/><Vertex X="40" Y="0"/><Vertex X="40" Y="40"/><Vertex X="0" Y="40"/>
    </Vertices></Region></Regions></Annotation></Annotations>"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".xml", delete=False, encoding="utf-8") as handle:
        handle.write(xml)
        path = Path(handle.name)
    try:
        env = segment_envelope_from_file(path)
        assert env["cell_count"] == 1
        assert len(env["cells"][0]) == TARGET_RING
    finally:
        path.unlink(missing_ok=True)


def test_csv_parse_envelope() -> None:
    csv_txt = (
        "polygon_id,x,y\n"
        "1,0,0\n1,10,0\n1,10,10\n1,0,10\n"
        "2,100,100\n2,110,100\n2,110,110\n2,100,110\n"
    )
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False, newline="", encoding="utf-8"
    ) as handle:
        handle.write(csv_txt)
        path = Path(handle.name)
    try:
        env = segment_envelope_from_file(path)
        assert env.get("gold_standard_source") == "CSV-Gold-Standard"
        assert env["cell_count"] == 2
        assert env["clinical_interpretation"]["gold_standard_source"] == "CSV-Gold-Standard"
    finally:
        path.unlink(missing_ok=True)
