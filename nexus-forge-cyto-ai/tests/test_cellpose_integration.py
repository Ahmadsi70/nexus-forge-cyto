"""Integration tests for the Cellpose nuclei input gate (Stage A)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

_SERVICES = Path(__file__).resolve().parents[1] / "services"
if str(_SERVICES) not in sys.path:
    sys.path.insert(0, str(_SERVICES))

from ingest_service import cellpose_ingest as ci  # noqa: E402
from nexus_core.vector_ingest import FORMAT_CELLPOSE_NUCLEI  # noqa: E402


class _FakeModel:
    def eval(self, img, **kw):
        mask = np.zeros((64, 64), dtype=np.int32)
        mask[16:48, 16:48] = 1  # one solid nucleus
        return mask, None, None


class _Upload:
    def __init__(self, name: str):
        self.name = name

    def getvalue(self) -> bytes:
        return b"fake-bytes"


def test_bootstrap_payload_yields_32_vertex_ring(monkeypatch):
    pytest.importorskip("skimage")
    monkeypatch.setattr(ci, "_resolve_cellpose_device", lambda: "cpu")
    monkeypatch.setattr(ci, "_load_image_from_bytes", lambda raw, fn, dev: np.zeros((64, 64), np.uint8))
    monkeypatch.setattr(ci, "_get_nuclei_model", lambda dev: _FakeModel())

    payload_str, fmt = ci.bootstrap_payload_from_image_bytes(b"\x89PNG", "patch.png")
    payload = json.loads(payload_str)

    assert fmt == FORMAT_CELLPOSE_NUCLEI
    assert payload["cell_count"] == 1
    assert len(payload["cells"][0]) == 32
    assert all(len(pt) == 2 for pt in payload["cells"][0])


def test_ingest_routing_vector_precedence(monkeypatch, tmp_path):
    import streamlit as st
    from interface_service import dashboard_shared as ds

    monkeypatch.setattr(ds, "INGEST_DIR", tmp_path)
    monkeypatch.setattr(ds, "INPUT_JSON", tmp_path / "uploaded_tissue.json")
    monkeypatch.setattr(st, "session_state", {}, raising=False)
    monkeypatch.setattr(
        ci, "bootstrap_payload_from_image_bytes",
        lambda raw, name: ('{"cell_count":1,"cells":[[[1.0,2.0]]]}', FORMAT_CELLPOSE_NUCLEI),
    )

    ds.write_cellpose_bootstrap(_Upload("patch.png"))
    assert ds.INPUT_JSON.is_file()
    assert st.session_state["last_vector_fmt"] == FORMAT_CELLPOSE_NUCLEI

    monkeypatch.setattr(
        ds, "normalize_vector_upload_to_json_bytes",
        lambda fn, raw: (b'{"cell_count":0,"cells":[]}', "aperio_imagescope_xml"),
    )

    def _boom(*a, **k):
        raise AssertionError("Cellpose must be bypassed when a vector is present")

    monkeypatch.setattr(ci, "bootstrap_payload_from_image_bytes", _boom)
    ds.write_vector_bootstrap(_Upload("a.xml"))
    assert st.session_state["last_vector_fmt"] == "aperio_imagescope_xml"
