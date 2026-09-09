"""Tests for dense-peak tiling and polygon dedupe (SAM3 recall path)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_SERVICES = Path(__file__).resolve().parents[1] / "services"
if str(_SERVICES) not in sys.path:
    sys.path.insert(0, str(_SERVICES))

from api_service.nuclei_proposer import propose_nuclei_exemplars  # noqa: E402
from api_service.tile_segment import (  # noqa: E402
    dedupe_polygons_by_centroid,
    iter_image_tiles,
    shift_polygon,
)


def _synthetic_he_patch(h: int = 256, w: int = 256) -> np.ndarray:
    rgb = np.full((h, w, 3), 235, dtype=np.uint8)
    rgb[..., 0] = 210
    rgb[..., 2] = 200
    centers = [(48, 52), (118, 76), (168, 142), (82, 168), (190, 48)]
    yy, xx = np.ogrid[:h, :w]
    for cx, cy in centers:
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= 11**2
        rgb[mask] = np.array([70, 50, 110], dtype=np.uint8)
    return rgb


def test_iter_tiles_covers_image_with_overlap() -> None:
    rgb = np.zeros((1000, 1000, 3), dtype=np.uint8)
    tiles = list(iter_image_tiles(rgb, tile_size=512, overlap=64))
    assert len(tiles) >= 4
    # Each tile has origin + crop
    for x0, y0, crop in tiles:
        assert crop.shape[0] <= 512 and crop.shape[1] <= 512
        assert x0 >= 0 and y0 >= 0
    # Bottom-right pixel must be covered by some tile
    covered = False
    for x0, y0, crop in tiles:
        if x0 + crop.shape[1] >= 1000 and y0 + crop.shape[0] >= 1000:
            covered = True
    assert covered


def test_shift_and_dedupe_polygons() -> None:
    poly_a = [(10.0, 10.0), (20.0, 10.0), (20.0, 20.0), (10.0, 20.0), (10.0, 10.0)]
    poly_b = shift_polygon(poly_a, dx=100.0, dy=50.0)
    assert poly_b[0] == (110.0, 60.0)
    # Near-duplicate of poly_a
    poly_dup = [(11.0, 11.0), (21.0, 11.0), (21.0, 21.0), (11.0, 21.0), (11.0, 11.0)]
    kept = dedupe_polygons_by_centroid([poly_a, poly_dup, poly_b], min_dist=8.0)
    assert len(kept) == 2


def test_dense_propose_returns_more_boxes_than_legacy_default() -> None:
    rgb = _synthetic_he_patch(320, 320)
    # Sprinkle many nuclei
    yy, xx = np.ogrid[:320, :320]
    for cx in range(30, 300, 28):
        for cy in range(30, 300, 28):
            mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= 8**2
            rgb[mask] = np.array([70, 50, 110], dtype=np.uint8)
    sparse = propose_nuclei_exemplars(rgb, max_positive=12, max_negative=0, peak_percentile=78.0)
    dense = propose_nuclei_exemplars(rgb, max_positive=80, max_negative=0, peak_percentile=70.0)
    assert sum(1 for L in dense.labels if L == 1) > sum(1 for L in sparse.labels if L == 1)
