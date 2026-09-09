"""
Tiled SAM 3 coverage helpers.

Why: dense MoNuSeg patches (~800 nuclei) need local exemplars per tile;
a single global SAM3 concept pass under-recalls.
"""

from __future__ import annotations

from typing import Iterator

import numpy as np


def iter_image_tiles(
    rgb: np.ndarray,
    *,
    tile_size: int = 512,
    overlap: int = 64,
) -> Iterator[tuple[int, int, np.ndarray]]:
    """Yield (x0, y0, crop) covering H×W×3 with stride = tile_size - overlap."""
    if rgb.ndim != 3:
        raise ValueError("rgb must be H×W×3")
    h, w = rgb.shape[:2]
    tile_size = max(64, int(tile_size))
    overlap = max(0, min(int(overlap), tile_size // 2))
    stride = max(1, tile_size - overlap)

    y_starts = list(range(0, max(1, h - tile_size + 1), stride))
    x_starts = list(range(0, max(1, w - tile_size + 1), stride))
    if not y_starts or y_starts[-1] + tile_size < h:
        y_starts.append(max(0, h - tile_size))
    if not x_starts or x_starts[-1] + tile_size < w:
        x_starts.append(max(0, w - tile_size))

    # de-dupe starts while preserving order
    def _uniq(xs: list[int]) -> list[int]:
        out: list[int] = []
        for v in xs:
            if v not in out:
                out.append(v)
        return out

    for y0 in _uniq(y_starts):
        for x0 in _uniq(x_starts):
            y1 = min(h, y0 + tile_size)
            x1 = min(w, x0 + tile_size)
            # Snap origin if final tile is shorter than tile_size
            y0_eff = max(0, y1 - (y1 - y0))
            x0_eff = max(0, x1 - (x1 - x0))
            crop = rgb[y0:y1, x0:x1]
            if crop.size == 0:
                continue
            yield x0, y0, crop


def shift_polygon(
    poly: list[tuple[float, float]],
    *,
    dx: float,
    dy: float,
) -> list[tuple[float, float]]:
    """Translate polygon vertices into global image coordinates."""
    return [(float(x) + dx, float(y) + dy) for x, y in poly]


def _centroid(poly: list[tuple[float, float]]) -> tuple[float, float]:
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    n = max(1, len(xs))
    return sum(xs) / n, sum(ys) / n


def dedupe_polygons_by_centroid(
    polygons: list[list[tuple[float, float]]],
    *,
    min_dist: float = 12.0,
) -> list[list[tuple[float, float]]]:
    """Keep largest-area polygon when centroids are closer than min_dist."""
    if not polygons:
        return []

    def _area(poly: list[tuple[float, float]]) -> float:
        # shoelace
        a = 0.0
        for i in range(len(poly) - 1):
            x1, y1 = poly[i]
            x2, y2 = poly[i + 1]
            a += x1 * y2 - x2 * y1
        return abs(a) * 0.5

    order = sorted(range(len(polygons)), key=lambda i: _area(polygons[i]), reverse=True)
    kept: list[list[tuple[float, float]]] = []
    centers: list[tuple[float, float]] = []
    for i in order:
        cx, cy = _centroid(polygons[i])
        if any((cx - kx) ** 2 + (cy - ky) ** 2 < min_dist**2 for kx, ky in centers):
            continue
        kept.append(polygons[i])
        centers.append((cx, cy))
    return kept
