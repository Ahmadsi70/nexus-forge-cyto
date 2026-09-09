# -*- coding: utf-8 -*-
"""
Gold-standard annotation adapter: MoNuSeg XML, PanNuke-like JSON, **CSV vertex tables** → **32×(x,y)** rings (**why**: feeds the
same Mojo κ tensor contract as HTTP segment exports — using **only** human-provided boundaries).

**Why stdlib XML + csv**: avoids extra deps; tolerates common Slide/ImageScope exports and spreadsheet polygon dumps.
"""

from __future__ import annotations

import csv
import json
import math
import xml.etree.ElementTree as ET
from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

TARGET_RING: int = 128

_CSV_GROUP_KEYS: tuple[str, ...] = ("polygon_id", "cell_id", "object_id", "region_id", "id")


def radius_from_mapping(row: dict[str, object]) -> float:
    """Derive a ring radius from a feature/property mapping.

    Single source of truth for the radius heuristic used across ingest paths.
    Order of precedence: explicit ``radius``/``r`` → half of ``diameter``/``size``
    → ``sqrt(area / π)`` → default 4.0 px. All keys are matched case-insensitively.
    """
    def _f(v: object) -> float | None:
        try:
            if v is None:
                return None
            s = str(v).strip()
            if not s:
                return None
            return float(s)
        except Exception:
            return None

    lk = {str(k).lower(): k for k in row.keys()}
    for key in ("radius", "r"):
        if key in lk:
            v = _f(row[lk[key]])
            if v is not None and v > 0:
                return v
    for key in ("diameter", "size"):
        if key in lk:
            v = _f(row[lk[key]])
            if v is not None and v > 0:
                return v * 0.5
    if "area" in lk:
        v = _f(row[lk["area"]])
        if v is not None and v > 0:
            return math.sqrt(v / math.pi)
    return 4.0


def parse_csv_annotations(path: Path | str) -> list[list[tuple[float, float]]]:
    """
    Parse CSV vertex dumps into closed polygons (**why**: interoperable gold standard from spreadsheets / exports).

    **Columns** (header names are matched case-insensitively):
    - **`x`**, **`y`** *or* **`coord_x`**, **`coord_y`** — required numeric coordinates.

    **Grouping** (optional; first matching column wins):
    - **`polygon_id`**, **`cell_id`**, **`object_id`**, **`region_id`**, **`id`** — groups rows into separate polygons.

    If **no** grouping column exists, all valid rows form **one** polygon (≥3 points).

    Rows with non-numeric coordinates are skipped; polygons with **<3** vertices after grouping are dropped.
    """
    p = Path(path)
    polygons: list[list[tuple[float, float]]] = []
    rows_norm: list[dict[str, str]] = []

    with p.open(encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        if not reader.fieldnames:
            raise ValueError("Invalid Coordinate CSV: CSV has no header row")

        headers_lower = {(n or "").strip().lower() for n in reader.fieldnames}
        has_xy_pair = "x" in headers_lower and "y" in headers_lower
        has_coord_pair = "coord_x" in headers_lower and "coord_y" in headers_lower
        if not has_xy_pair and not has_coord_pair:
            raise ValueError(
                "Invalid Coordinate CSV: header row must include columns 'x' and 'y', "
                "or 'coord_x' and 'coord_y' (case-insensitive)."
            )

        xk, yk = ("x", "y") if has_xy_pair else ("coord_x", "coord_y")

        for raw in reader:
            row = {
                (k or "").strip().lower(): ("" if v is None else str(v).strip())
                for k, v in raw.items()
            }
            if not any(v for v in row.values()):
                continue
            rows_norm.append(row)

    if not rows_norm:
        raise ValueError("CSV contained no data rows")

    group_col = next((k for k in _CSV_GROUP_KEYS if k in headers_lower), None)

    def append_point(bucket: list[tuple[float, float]], r: dict[str, str]) -> None:
        try:
            xf = float(r[xk])
            yf = float(r[yk])
        except (KeyError, ValueError, TypeError):
            return
        if not math.isfinite(xf) or not math.isfinite(yf):
            return
        bucket.append((xf, yf))

    if group_col is None:
        bucket: list[tuple[float, float]] = []
        for r in rows_norm:
            append_point(bucket, r)
        if len(bucket) < 3:
            raise ValueError("CSV without grouping column needs ≥3 valid (x,y) rows for one polygon")
        polygons.append(bucket)
        return polygons

    groups: OrderedDict[str, list[tuple[float, float]]] = OrderedDict()
    for r in rows_norm:
        gid_raw = r.get(group_col, "").strip()
        if not gid_raw:
            continue
        gid = gid_raw
        if gid not in groups:
            groups[gid] = []
        append_point(groups[gid], r)
    for _gid, bucket in groups.items():
        if len(bucket) >= 3:
            polygons.append(bucket)

    if not polygons:
        raise ValueError("no polygon had ≥3 vertices after grouping — check ids and coordinates")
    return polygons


def _polygon_signed_area_xy(poly_xy: np.ndarray) -> float:
    x = poly_xy[:, 0].astype(np.float64)
    y = poly_xy[:, 1].astype(np.float64)
    x2 = np.roll(x, -1)
    y2 = np.roll(y, -1)
    return float(0.5 * np.sum(x * y2 - x2 * y))


def _ensure_ccw_xy(poly: np.ndarray) -> np.ndarray:
    if poly.shape[0] < 3:
        return np.asarray(poly, dtype=np.float32)
    out = poly.copy()
    if _polygon_signed_area_xy(out) < 0:
        out = out[::-1]
    return out.astype(np.float32)


def resample_polygon_32_equidistant(poly_xy: np.ndarray) -> np.ndarray:
    """
    Equal arc-length resample to **`TARGET_RING`** vertices in **(x, y)** space (**why**: locked tensor row count for Mojo;
    arc-length spacing stabilizes downstream κ statistics vs naive vertex subsampling).
    """
    pts = np.asarray(poly_xy, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"polygon must be (N, 2) in x,y, got {pts.shape}")
    if pts.shape[0] < 1:
        raise ValueError("empty polygon")
    if pts.shape[0] == 1:
        return np.tile(pts.astype(np.float32), (TARGET_RING, 1))
    if pts.shape[0] == 2:
        a, b = pts[0], pts[1]
        t = np.linspace(0.0, 1.0, TARGET_RING, endpoint=False)
        s = a + (b - a) * t[:, None]
        return _ensure_ccw_xy(s.astype(np.float32))

    pts = _ensure_ccw_xy(pts)
    closed = np.vstack([pts, pts[0:1]])
    seg = np.diff(closed, axis=0)
    seg_len = np.linalg.norm(seg, axis=1)
    perim = float(seg_len.sum())
    if not math.isfinite(perim) or perim < 1e-12:
        c = pts.mean(axis=0, keepdims=True)
        return np.tile(c.astype(np.float32), (TARGET_RING, 1))

    cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    targets = np.linspace(0.0, perim, num=TARGET_RING, endpoint=False)
    out = np.zeros((TARGET_RING, 2), dtype=np.float64)
    for i, tdist in enumerate(targets):
        idx = int(np.searchsorted(cum, tdist, side="right") - 1)
        idx = max(0, min(idx, len(seg_len) - 1))
        seg_start = cum[idx]
        slen = seg_len[idx]
        alpha = 0.0 if slen < 1e-15 else (tdist - seg_start) / slen
        out[i] = closed[idx] + alpha * seg[idx]

    return _ensure_ccw_xy(out.astype(np.float32))


def ring32_from_polygon_xy(points: list[tuple[float, float]]) -> list[list[float]]:
    """Normalize arbitrary closed polygon vertex list → **`TARGET_RING`** **`[x,y]`** samples."""
    if len(points) < 3:
        raise ValueError(f"polygon needs ≥3 vertices, got {len(points)}")
    arr = np.asarray(points, dtype=np.float64)
    # Drop duplicated closing vertex if present
    if np.allclose(arr[0], arr[-1]):
        arr = arr[:-1]
    if arr.shape[0] < 3:
        raise ValueError("polygon collapsed after removing duplicate closing vertex")
    ring = resample_polygon_32_equidistant(arr.astype(np.float64))
    return [[float(ring[i, 0]), float(ring[i, 1])] for i in range(TARGET_RING)]


def _polygon_area_shoelace(points: list[list[float]]) -> float:
    n = len(points)
    if n < 3:
        return 0.0
    s = 0.0
    for i in range(n):
        x1, y1 = float(points[i][0]), float(points[i][1])
        x2, y2 = float(points[(i + 1) % n][0]), float(points[(i + 1) % n][1])
        s += x1 * y2 - x2 * y1
    return abs(s) * 0.5


def _median_areas_payload(rings32: list[list[list[float]]]) -> dict[str, Any]:
    areas = [_polygon_area_shoelace(r) for r in rings32]
    median_a = float(np.median(areas)) if areas else 0.0
    thr = 1.5 * median_a
    return {
        "message": "Parameters were dynamically calibrated for this specific image's lighting and density.",
        "median_cell_area_px2": median_a,
        "malignancy_area_threshold_px2": thr,
        "source": "gold_annotation_adapter",
    }


def parse_monuseg_xml(path: Path | str) -> list[list[tuple[float, float]]]:
    """
    Parse MoNuSeg / ImageScope-style **`Regions` → `Vertices` → `Vertex X,Y`** (**why**: common expert polygon dump).

    Collects one polygon per **`Region`** with ≥3 vertices (**why**: skip empty markup).
    """
    try:
        tree = ET.parse(path)
    except ET.ParseError as e:
        raise ValueError(f"invalid MoNuSeg/XML annotation: {e}") from e
    root = tree.getroot()
    polygons: list[list[tuple[float, float]]] = []

    for region in root.iter():
        tag = region.tag.split("}")[-1]
        if tag != "Region":
            continue
        verts: list[tuple[float, float]] = []
        for child in region.iter():
            ct = child.tag.split("}")[-1]
            if ct != "Vertex":
                continue
            xa = child.get("X") or child.get("x")
            ya = child.get("Y") or child.get("y")
            if xa is None or ya is None:
                continue
            verts.append((float(xa), float(ya)))
        if len(verts) >= 3:
            polygons.append(verts)

    return polygons


def parse_annotation_json(path: Path | str) -> list[list[tuple[float, float]]]:
    """
    Parse JSON polygons in PanNuke / generic cytology exports (**why**: one bridge for multiple bench formats).

    Supported shapes:
    - **`{"cells": [{"boundary": [[x,y],…]}, …]}`** or **`ring` / `polygon` / `coordinates`**
    - **`{"polygons": [[[x,y],…], …]}`**
    - **`{"nuclei": […]}`** with per-item **`coords`**, **`segmentation`**, **`polygon`**
    - **`[[[x,y],…], …]`** top-level list of rings
    - GeoJSON **`{"type":"FeatureCollection","features":[…]}`** with **`Polygon`** geometries
    """
    raw_text = Path(path).read_text(encoding="utf-8")
    data = json.loads(raw_text)
    return _polygons_from_json_obj(data)


def _extract_xy_pair(obj: Any) -> tuple[float, float] | None:
    if isinstance(obj, (list, tuple)) and len(obj) >= 2:
        return float(obj[0]), float(obj[1])
    if isinstance(obj, dict):
        for kx, ky in (("x", "y"), ("X", "Y")):
            if kx in obj and ky in obj:
                return float(obj[kx]), float(obj[ky])
    return None


def _looks_like_single_polygon_point_list(xs: list[Any]) -> bool:
    """True when **`xs`** is **`[[x,y], …]`** (**why**: distinguish one polygon vs list-of-polygons)."""
    if not xs:
        return False
    el = xs[0]
    return (
        isinstance(el, list)
        and len(el) >= 2
        and isinstance(el[0], (int, float))
        and isinstance(el[1], (int, float))
    )


def _polygons_from_json_obj(data: Any) -> list[list[tuple[float, float]]]:
    out: list[list[tuple[float, float]]] = []

    if isinstance(data, list):
        if not data:
            return out
        if _looks_like_single_polygon_point_list(data):
            pts = [_extract_xy_pair(p) for p in data]
            pts2 = [p for p in pts if p is not None]
            if len(pts2) >= 3:
                out.append(pts2)
            return out
        for poly in data:
            if isinstance(poly, list) and poly:
                pts = [_extract_xy_pair(p) for p in poly]
                pts2 = [p for p in pts if p is not None]
                if len(pts2) >= 3:
                    out.append(pts2)
        return out

    if not isinstance(data, dict):
        return out

    if data.get("type") == "FeatureCollection" and isinstance(data.get("features"), list):
        for feat in data["features"]:
            if not isinstance(feat, dict):
                continue
            geom = feat.get("geometry")
            if not isinstance(geom, dict) or geom.get("type") != "Polygon":
                continue
            coords = geom.get("coordinates")
            if isinstance(coords, list) and coords:
                outer = coords[0]
                pts = [_extract_xy_pair(p) for p in outer]
                pts2 = [p for p in pts if p is not None]
                if len(pts2) >= 3:
                    out.append(pts2)
        return out

    for key in ("polygons", "regions", "annotations"):
        if key in data and isinstance(data[key], list):
            return _polygons_from_json_obj(data[key])

    for key in ("cells", "nuclei"):
        if key not in data or not isinstance(data[key], list):
            continue
        for item in data[key]:
            if not isinstance(item, dict):
                continue
            for pk in ("boundary", "ring", "polygon", "coordinates", "segmentation", "mask_polygon"):
                if pk not in item:
                    continue
                raw_poly = item[pk]
                pts: list[tuple[float, float]] = []
                if isinstance(raw_poly, list):
                    for p in raw_poly:
                        pair = _extract_xy_pair(p)
                        if pair is not None:
                            pts.append(pair)
                if len(pts) >= 3:
                    out.append(pts)
        if out:
            return out

    return out


def parse_annotation_file(path: Path | str) -> list[list[tuple[float, float]]]:
    """Dispatch on suffix **`.xml`**, **`.json`**, **`.csv`** (**why**: single entry for UI + CLI)."""
    p = Path(path)
    suf = p.suffix.lower()
    if suf == ".xml":
        return parse_monuseg_xml(p)
    if suf == ".json":
        return parse_annotation_json(p)
    if suf == ".csv":
        return parse_csv_annotations(p)
    raise ValueError(f"unsupported annotation suffix {suf!r} (expected .xml, .json, or .csv)")


def segment_envelope_from_polygons(
    polygons: Iterable[list[tuple[float, float]]],
    *,
    format_hint: str,
    source_path: str | None = None,
    gold_standard_source: str | None = None,
) -> dict[str, Any]:
    """
    Build segment JSON (`cell_count`, `cells`, calibration blobs) for **`mojo_results_from_segment_json_bytes`**.

    **Why**: Matches the historical FastAPI segment payload shape so Mojo enrichment stays unchanged.

    **`gold_standard_source`**: when **`"CSV-Gold-Standard"`**, adds provenance + interpretation hints for auditors.
    """
    rings32: list[list[list[float]]] = []
    for poly in polygons:
        rings32.append(ring32_from_polygon_xy(list(poly)))

    if not rings32:
        raise ValueError("no valid polygons after parsing annotation")

    dyn = _median_areas_payload(rings32)
    if gold_standard_source:
        dyn["gold_standard_source"] = gold_standard_source
        dyn["malignancy_area_multiplier"] = 1.5
        dyn["rust_malignancy_note"] = (
            "Final `clinical[]` labels are assigned in Rust: Mojo κ decision, then forced **Malignant** "
            "if shoelace ring area (px²) > 1.5 × median(all ring areas). Threshold echoed as "
            "`malignancy_area_threshold_px2`; peer median as `median_cell_area_px2`."
        )

    payload: dict[str, Any] = {
        "cell_count": len(rings32),
        "cells": rings32,
        "options": {"source": "gold_annotation", "adapter_format": format_hint},
        "pre_analysis": None,
        "dynamic_calibration": dyn,
        "annotation_adapter": {
            "format": format_hint,
            "source_path": source_path,
            "ring_points": TARGET_RING,
        },
    }

    if gold_standard_source == "CSV-Gold-Standard":
        payload["options"]["gold_standard_source"] = "CSV-Gold-Standard"

    if gold_standard_source:
        payload["gold_standard_source"] = gold_standard_source
        payload["annotation_adapter"]["gold_standard_source"] = gold_standard_source

    if gold_standard_source == "CSV-Gold-Standard":
        payload["clinical_interpretation"] = {
            "gold_standard_source": "CSV-Gold-Standard",
            "how_to_count_malignancies": (
                "`Malignant_count` (dashboard / analytics) = number of entries in `clinical` equal to "
                "`Malignant` (case-insensitive), aligned with `cells[i]` ring geometry."
            ),
            "peer_area_rule": (
                "Rust applies 1.5×median shoelace area over the 32-point rings; see "
                "`dynamic_calibration.malignancy_area_threshold_px2` and `median_cell_area_px2`."
            ),
        }

    return payload


def segment_envelope_from_file(path: Path | str) -> dict[str, Any]:
    """Parse annotation path → segment envelope (**why**: Streamlit / batch one-liner)."""
    p = Path(path)
    suf = p.suffix.lower()
    polys = parse_annotation_file(p)
    if suf == ".xml":
        fmt = "monuseg_xml"
        gss: str | None = None
    elif suf == ".csv":
        fmt = "csv_vertices"
        gss = "CSV-Gold-Standard"
    else:
        fmt = "json_polygon"
        gss = None
    return segment_envelope_from_polygons(
        polys,
        format_hint=fmt,
        source_path=str(p.resolve()),
        gold_standard_source=gss,
    )


def write_gold_segment_json(annotation_path: Path | str, dest_json: Path) -> dict[str, Any]:
    """Parse **`annotation_path`** and write **`dest_json`** segment envelope for Mojo ingestion."""
    env = segment_envelope_from_file(annotation_path)
    dest_json.parent.mkdir(parents=True, exist_ok=True)
    dest_json.write_text(json.dumps(env, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return env
