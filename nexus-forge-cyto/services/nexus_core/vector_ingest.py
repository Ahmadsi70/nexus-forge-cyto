"""
Multi-vendor vector ingest: Aperio XML, QuPath GeoJSON, ASAP XML.

Why: Stage A must accept international pathology exchange formats while preserving
stable TARGET_RING (128-point) ring geometry for downstream κ and QuPath interoperability.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from lxml import etree

from nexus_core.region_walker import (
    _local_name,
    _ring32_from_centroid,
    _ring32_from_polygon,
    records_from_geojson,
    records_from_xml,
)

FORMAT_APERIO_XML = "aperio_imagescope_xml"
FORMAT_ASAP_XML = "asap_xml"
FORMAT_QUPATH_GEOJSON = "qupath_geojson"
FORMAT_JSON_STAGE = "nexus_stage_json"
FORMAT_SAM3_NUCLEI = "sam3_nuclei_derived"


def _xml_parser() -> etree.XMLParser:
    return etree.XMLParser(
        remove_blank_text=False,
        strip_cdata=False,
        resolve_entities=False,
        no_network=True,
        recover=False,
    )


def _vertices_from_xml_element(element: etree._Element) -> list[tuple[float, float]]:
    """Collect Aperio Vertex or ASAP Coordinate nodes under an annotation element."""
    verts: list[tuple[float, float]] = []
    for child in element.iter():
        tag = _local_name(child.tag)
        if tag not in {"Vertex", "Coordinate"}:
            continue
        xa = child.get("X") or child.get("x")
        ya = child.get("Y") or child.get("y")
        if xa is None or ya is None:
            continue
        verts.append((float(xa), float(ya)))
    return verts


def detect_xml_variant(raw: bytes) -> str:
    """Classify XML as ASAP (Coordinate) vs Aperio/ImageScope (Region/Vertex)."""
    root = etree.fromstring(raw, _xml_parser())
    has_coordinate = any(_local_name(el.tag) == "Coordinate" for el in root.iter())
    has_region = any(_local_name(el.tag) == "Region" for el in root.iter())
    has_vertex = any(_local_name(el.tag) == "Vertex" for el in root.iter())
    if has_coordinate and not (has_region or has_vertex):
        return FORMAT_ASAP_XML
    return FORMAT_APERIO_XML


def detect_vector_format(filename: str, raw: bytes) -> str:
    """Detect vendor format from extension and payload structure."""
    suf = Path(filename).suffix.lower()
    if suf == ".geojson":
        return FORMAT_QUPATH_GEOJSON
    if suf == ".json":
        try:
            payload = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON upload: {exc}") from exc
        if isinstance(payload, dict) and payload.get("type") == "FeatureCollection":
            return FORMAT_QUPATH_GEOJSON
        return FORMAT_JSON_STAGE
    if suf == ".xml":
        return detect_xml_variant(raw)
    raise ValueError(f"Unsupported vector extension {suf!r}; allowed: .xml, .geojson, .json")


def records_from_asap_xml(raw: bytes) -> list[dict[str, Any]]:
    """Parse ASAP XML annotations using Coordinate nodes."""
    root = etree.fromstring(raw, _xml_parser())
    out: list[dict[str, Any]] = []
    source_index = 0
    for element in root.iter():
        if _local_name(element.tag) != "Annotation":
            continue
        verts = _vertices_from_xml_element(element)
        if len(verts) < 3:
            continue
        cx = float(sum(p[0] for p in verts) / len(verts))
        cy = float(sum(p[1] for p in verts) / len(verts))
        out.append(
            {
                "source_index": source_index,
                "annotation_id": element.get("Id") or element.get("id"),
                "centroid": (cx, cy),
                "ring": _ring32_from_polygon(verts),
            }
        )
        source_index += 1
    if not out:
        raise ValueError("No ASAP Annotation/Coordinate polygons parsed from XML.")
    return out


def records_from_qupath_geojson_bytes(raw: bytes) -> list[dict[str, Any]]:
    """Parse QuPath-style GeoJSON FeatureCollection from in-memory bytes."""
    payload = json.loads(raw.decode("utf-8"))
    feats = payload.get("features", [])
    out: list[dict[str, Any]] = []
    for source_index, feat in enumerate(feats):
        if not isinstance(feat, dict):
            continue
        geom = feat.get("geometry", {})
        props = feat.get("properties", {}) if isinstance(feat.get("properties"), dict) else {}
        gtype = str(geom.get("type", ""))
        coords = geom.get("coordinates", [])
        source_class = _extract_qupath_class_name(props)
        rec: dict[str, Any] | None = None
        if gtype == "Point" and isinstance(coords, list) and len(coords) >= 2:
            x, y = float(coords[0]), float(coords[1])
            rec = {
                "source_index": source_index,
                "feature_id": feat.get("id") or props.get("id"),
                "centroid": (x, y),
                "ring": _ring32_from_centroid(x, y, 4.0),
                "source_class": source_class,
            }
        elif gtype == "Polygon":
            ring0 = coords[0] if coords and isinstance(coords[0], list) else []
            pts = [(float(p[0]), float(p[1])) for p in ring0 if isinstance(p, list) and len(p) >= 2]
            if pts:
                cx = float(sum(p[0] for p in pts) / len(pts))
                cy = float(sum(p[1] for p in pts) / len(pts))
                rec = {
                    "source_index": source_index,
                    "feature_id": feat.get("id") or props.get("id"),
                    "centroid": (cx, cy),
                    "ring": _ring32_from_polygon(pts),
                    "source_class": source_class,
                }
        elif gtype == "MultiPolygon" and isinstance(coords, list) and coords:
            ring0 = coords[0][0] if isinstance(coords[0], list) and coords[0] else []
            pts = [(float(p[0]), float(p[1])) for p in ring0 if isinstance(p, list) and len(p) >= 2]
            if pts:
                cx = float(sum(p[0] for p in pts) / len(pts))
                cy = float(sum(p[1] for p in pts) / len(pts))
                rec = {
                    "source_index": source_index,
                    "feature_id": feat.get("id") or props.get("id"),
                    "centroid": (cx, cy),
                    "ring": _ring32_from_polygon(pts),
                    "source_class": source_class,
                }
        if rec is not None:
            out.append(rec)
    if not out:
        raise ValueError("No QuPath GeoJSON polygon features parsed.")
    return out


def _extract_qupath_class_name(props: dict[str, Any]) -> str | None:
    """Read QuPath classification name from properties.classification.name."""
    classification = props.get("classification")
    if isinstance(classification, dict):
        name = classification.get("name")
        if name is not None:
            return str(name).strip()
    for key in ("class", "classification", "name"):
        val = props.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return None


def parse_vector_bytes(filename: str, raw: bytes) -> tuple[list[dict[str, Any]], str]:
    """Parse uploaded vector bytes into ring records plus detected format id."""
    fmt = detect_vector_format(filename, raw)
    if fmt == FORMAT_QUPATH_GEOJSON:
        records = records_from_qupath_geojson_bytes(raw)
    elif fmt == FORMAT_ASAP_XML:
        records = records_from_asap_xml(raw)
    elif fmt == FORMAT_APERIO_XML:
        root = etree.fromstring(raw, _xml_parser())
        records = _records_from_xml_root(root)
    elif fmt == FORMAT_JSON_STAGE:
        json.loads(raw.decode("utf-8"))
        return [], fmt
    else:
        raise ValueError(f"Unhandled vector format: {fmt}")
    return records, fmt


def _records_from_xml_root(root: etree._Element) -> list[dict[str, Any]]:
    """Parse Aperio/HALO Region/Vertex or Annotation-wrapped vertices from XML root."""
    out: list[dict[str, Any]] = []
    source_index = 0
    for element in root.iter():
        tag = _local_name(element.tag)
        if tag not in {"Region", "Annotation"}:
            continue
        verts = _vertices_from_xml_element(element)
        if len(verts) < 3:
            continue
        # Prefer explicit Region nodes; Annotation nodes may wrap Regions — skip empty shells
        if tag == "Annotation" and any(_local_name(c.tag) == "Region" for c in element.iter()):
            continue
        cx = float(sum(p[0] for p in verts) / len(verts))
        cy = float(sum(p[1] for p in verts) / len(verts))
        out.append(
            {
                "source_index": source_index,
                "region_id": element.get("Id") or element.get("id"),
                "centroid": (cx, cy),
                "ring": _ring32_from_polygon(verts),
            }
        )
        source_index += 1
    if not out:
        raise ValueError("No Aperio/ImageScope Region/Vertex polygons parsed from XML.")
    return out


def records_from_vector_path(path: Path) -> tuple[list[dict[str, Any]], str]:
    """Parse vector file from disk for orchestrator ingest."""
    raw = path.read_bytes()
    fmt = detect_vector_format(path.name, raw)
    if fmt == FORMAT_QUPATH_GEOJSON:
        return records_from_geojson(path), fmt
    if fmt == FORMAT_ASAP_XML:
        return records_from_asap_xml(raw), fmt
    if fmt == FORMAT_APERIO_XML:
        return records_from_xml(path), fmt
    if fmt == FORMAT_JSON_STAGE:
        return [], fmt
    raise ValueError(f"Unhandled vector format: {fmt}")


def payload_from_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Minimal Stage-A bootstrap payload from parsed vector records."""
    if not records:
        raise ValueError("No vector records were parsed from upload.")
    return {"cell_count": len(records), "cells": [r["ring"] for r in records]}
