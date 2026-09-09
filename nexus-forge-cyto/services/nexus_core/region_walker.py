"""
Shared annotation traversal for ingest/export parity.

Why: ingest and export must enumerate the same Region/Feature nodes in identical
document order so cell_id join keys remain stable across the pipeline.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterator

from lxml import etree

# --- Single-source-of-truth geometry helpers (delegates to coord_adapter) ---
from nexus_core.coord_adapter import (                    # noqa: E402  — canonical ring impl
    ring_from_polygon_xy as _ring32_from_polygon,       # polygon → ring
    radius_from_mapping as _radius_from_mapping,          # radius heuristic (canonical)
    _polygon_area_shoelace,                                # shoelace area
    TARGET_RING,                                           # 128
)

# Re-export for backward compatibility (used by vector_ingest, checkpoint_serializers, etc.)
__all__ = [
    "GNN_TO_CLINICAL",
    "clinical_label_from_gnn",
    "file_sha256",
    "records_from_xml",
    "records_from_geojson",
    "build_manifest",
    "write_manifest",
    "hash_xml_vertex_coords",
    "hash_geojson_coordinates",
    "assert_geometry_unchanged",
    "iter_xml_regions",
    "_xml_parser",
    "_ring32_from_centroid",
    "_ring32_from_polygon",
    "_polygon_area_perimeter",
    "_radius_from_mapping",
]

GNN_TO_CLINICAL: dict[str, str] = {
    "Epithelial_Malignant": "Malignant",
    "Inflammatory": "Immune",
    "Lymphocyte": "Immune",
    "Stromal": "Normal",
}


def clinical_label_from_gnn(gnn_class: str) -> str:
    """Map 4-class GNN taxonomy to simplified clinical labels for vendor export."""
    if gnn_class not in GNN_TO_CLINICAL:
        raise ValueError(
            f"Unknown GNN class {gnn_class!r}; allowed={sorted(GNN_TO_CLINICAL.keys())}"
        )
    return GNN_TO_CLINICAL[gnn_class]


def _local_name(tag: str) -> str:
    return tag.split("}")[-1] if "}" in tag else tag


def file_sha256(path: Path) -> str:
    """Return SHA-256 hex digest of a source annotation file for manifest validation."""
    return _file_sha256(path)


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _ring32_from_centroid(cx: float, cy: float, radius: float = 4.0) -> list[list[float]]:
    """Generate a regular TARGET_RING-vertex ring centred at (cx, cy)."""
    out: list[list[float]] = []
    for i in range(TARGET_RING):
        t = (2.0 * math.pi * i) / TARGET_RING
        out.append([cx + radius * math.cos(t), cy + radius * math.sin(t)])
    return out


def _polygon_area_perimeter(ring: list[list[float]]) -> tuple[float, float]:
    """Shoelace area + perimeter from a 32-ring.  Delegates to coord_adapter."""
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


def _xml_parser() -> etree.XMLParser:
    return etree.XMLParser(
        remove_blank_text=False,
        strip_cdata=False,
        resolve_entities=False,
        no_network=True,
        recover=False,
    )


def _vertices_from_region_element(region: etree._Element) -> list[tuple[float, float]]:
    verts: list[tuple[float, float]] = []
    for child in region.iter():
        tag = _local_name(child.tag)
        if tag not in {"Vertex", "Coordinate"}:
            continue
        xa = child.get("X") or child.get("x")
        ya = child.get("Y") or child.get("y")
        if xa is None or ya is None:
            continue
        verts.append((float(xa), float(ya)))
    return verts


def iter_xml_regions(root: etree._Element) -> Iterator[tuple[etree._Element, list[tuple[float, float]]]]:
    """Yield (Region element, vertices) in document order matching ingest contract."""
    for region in root.iter():
        if _local_name(region.tag) != "Region":
            continue
        verts = _vertices_from_region_element(region)
        if verts:
            yield region, verts


def records_from_xml(path: Path) -> list[dict[str, Any]]:
    """Parse HALO/ImageScope XML into ordered ingest records with join metadata."""
    tree = etree.parse(str(path), _xml_parser())
    root = tree.getroot()
    out: list[dict[str, Any]] = []
    for source_index, (region, verts) in enumerate(iter_xml_regions(root)):
        cx = float(sum(p[0] for p in verts) / len(verts))
        cy = float(sum(p[1] for p in verts) / len(verts))
        region_id = region.get("Id") or region.get("id")
        out.append(
            {
                "source_index": source_index,
                "region_id": str(region_id) if region_id is not None else None,
                "centroid": (cx, cy),
                "ring": _ring32_from_polygon(verts),
            }
        )
    return out


def _centroid_from_polygon_coords(coords: list) -> tuple[float, float]:
    ring0 = coords[0] if coords and isinstance(coords[0], list) else []
    if not ring0:
        return (0.0, 0.0)
    xs = [float(p[0]) for p in ring0 if isinstance(p, list) and len(p) >= 2]
    ys = [float(p[1]) for p in ring0 if isinstance(p, list) and len(p) >= 2]
    if not xs or not ys:
        return (0.0, 0.0)
    return (float(sum(xs) / len(xs)), float(sum(ys) / len(ys)))


# NOTE: `_radius_from_mapping` is imported from coord_adapter above (single source of truth).
# The previous local copy was removed to eliminate duplication drift.


def records_from_geojson(path: Path) -> list[dict[str, Any]]:
    """Parse GeoJSON FeatureCollection into ordered ingest records with join metadata."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    feats = payload.get("features", [])
    out: list[dict[str, Any]] = []
    for source_index, feat in enumerate(feats):
        if not isinstance(feat, dict):
            continue
        geom = feat.get("geometry", {})
        props = feat.get("properties", {}) if isinstance(feat.get("properties"), dict) else {}
        gtype = str(geom.get("type", ""))
        coords = geom.get("coordinates", [])
        feature_id = props.get("id") if props else None
        if feature_id is None:
            feature_id = feat.get("id")
        source_class = None
        classification = props.get("classification")
        if isinstance(classification, dict) and classification.get("name"):
            source_class = str(classification.get("name")).strip()
        rec: dict[str, Any] | None = None
        if gtype == "Point" and isinstance(coords, list) and len(coords) >= 2:
            x, y = float(coords[0]), float(coords[1])
            r = _radius_from_mapping(props)
            rec = {
                "source_index": source_index,
                "feature_id": feature_id,
                "centroid": (x, y),
                "ring": _ring32_from_centroid(x, y, r),
            }
            if source_class:
                rec["source_class"] = source_class
        elif gtype == "Polygon":
            ring0 = coords[0] if coords and isinstance(coords[0], list) else []
            pts = [(float(p[0]), float(p[1])) for p in ring0 if isinstance(p, list) and len(p) >= 2]
            if pts:
                cx, cy = _centroid_from_polygon_coords(coords)
                rec = {
                    "source_index": source_index,
                    "feature_id": feature_id,
                    "centroid": (cx, cy),
                    "ring": _ring32_from_polygon(pts),
                }
        elif gtype == "MultiPolygon" and isinstance(coords, list) and coords:
            ring0 = coords[0][0] if isinstance(coords[0], list) and coords[0] else []
            pts = [(float(p[0]), float(p[1])) for p in ring0 if isinstance(p, list) and len(p) >= 2]
            if pts:
                cx = float(sum(p[0] for p in pts) / len(pts))
                cy = float(sum(p[1] for p in pts) / len(pts))
                rec = {
                    "source_index": source_index,
                    "feature_id": feature_id,
                    "centroid": (cx, cy),
                    "ring": _ring32_from_polygon(pts),
                }
        if rec is not None:
            out.append(rec)
    return out


def build_manifest(source_path: Path, source_format: str, records: list[dict[str, Any]]) -> dict[str, Any]:
    """Build join-key manifest tying cell_id to source annotation indices and centroids."""
    manifest_records: list[dict[str, Any]] = []
    for cell_id, rec in enumerate(records):
        cx, cy = rec["centroid"]
        entry: dict[str, Any] = {
            "cell_id": cell_id,
            "source_index": int(rec.get("source_index", cell_id)),
            "centroid": [float(cx), float(cy)],
        }
        if rec.get("region_id") is not None:
            entry["region_id"] = rec["region_id"]
        if rec.get("feature_id") is not None:
            entry["feature_id"] = rec["feature_id"]
        manifest_records.append(entry)
    st = source_path.stat()
    return {
        "source_path": str(source_path.resolve()),
        "source_format": source_format,
        "source_sha256": _file_sha256(source_path),
        "source_mtime": float(st.st_mtime),
        "record_count": len(manifest_records),
        "records": manifest_records,
    }


def write_manifest(manifest: dict[str, Any], dest: Path) -> None:
    """Persist annotation index manifest for export-time join validation."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def hash_xml_vertex_coords(path: Path) -> str:
    """Digest all Region vertex coordinates in traversal order for integrity gates."""
    tree = etree.parse(str(path), _xml_parser())
    h = hashlib.sha256()
    for _, verts in iter_xml_regions(tree.getroot()):
        for x, y in verts:
            h.update(f"{x:.6f},{y:.6f};".encode("ascii"))
    return h.hexdigest()


def _collect_geojson_coordinate_strings(payload: dict[str, Any]) -> list[str]:
    out: list[str] = []
    for feat in payload.get("features", []):
        if not isinstance(feat, dict):
            continue
        geom = feat.get("geometry", {})
        coords = geom.get("coordinates")
        if coords is None:
            continue
        out.append(json.dumps(coords, separators=(",", ":"), ensure_ascii=False))
    return out


def hash_geojson_coordinates(path: Path) -> str:
    """Digest GeoJSON geometry coordinates per feature for integrity gates."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    h = hashlib.sha256()
    for s in _collect_geojson_coordinate_strings(payload):
        h.update(s.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def assert_geometry_unchanged(source_path: Path, output_path: Path, source_format: str) -> None:
    """Hard-fail when export mutates geometry instead of metadata-only injection."""
    if source_format in {"halo_xml_regions", "monuseg_xml"}:
        src = hash_xml_vertex_coords(source_path)
        out = hash_xml_vertex_coords(output_path)
    elif source_format == "geojson_features":
        src = hash_geojson_coordinates(source_path)
        out = hash_geojson_coordinates(output_path)
    else:
        raise ValueError(f"Unsupported manifest source_format for integrity gate: {source_format!r}")
    if src != out:
        raise ValueError(
            f"Geometry integrity breach: coordinate digest mismatch "
            f"(source={src[:16]}… output={out[:16]}…)"
        )
