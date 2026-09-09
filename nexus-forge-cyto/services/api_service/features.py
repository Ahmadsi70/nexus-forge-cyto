"""
Pharma-grade feature extraction: turn a gold segment into a numeric feature matrix.

Why: pharma/biotech partners need per-cell numeric features (not clinical labels)
for their own statistical models, biomarker discovery, and clinical trial analysis.
This endpoint skips the full enrichment pipeline's clinical labeling pass and returns
raw feature vectors only — faster and cheaper for high-throughput screening.

Output formats: CSV (default) or JSON array of feature vectors.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

# Import Rust geometry functions via the bridge
from nexus_core.coord_adapter import TARGET_RING, _polygon_area_shoelace

# Cache feature vectors keyed by (ring_hash, knn_k) to avoid recomputation.
# Eviction is LRU with max 1024 entries.
_FEATURE_CACHE: dict[str, tuple[float, list[float]]] = {}
_FEATURE_CACHE_ORDER: list[str] = []
_FEATURE_CACHE_MAX = 1024


def _cache_key(rings_hash: str, knn_k: int) -> str:
    return f"{rings_hash}_{knn_k}"


def _cache_get(key: str) -> list[float] | None:
    """Return cached feature vector or None."""
    entry = _FEATURE_CACHE.get(key)
    if entry is None:
        return None
    _ts, vec = entry
    # Bump to front (LRU)
    _FEATURE_CACHE_ORDER.remove(key)
    _FEATURE_CACHE_ORDER.append(key)
    return list(vec)


def _cache_put(key: str, vec: list[float]) -> None:
    """Store a feature vector, evicting oldest if at capacity."""
    if key in _FEATURE_CACHE:
        _FEATURE_CACHE_ORDER.remove(key)
    _FEATURE_CACHE[key] = (time.time(), list(vec))
    _FEATURE_CACHE_ORDER.append(key)
    while len(_FEATURE_CACHE_ORDER) > _FEATURE_CACHE_MAX:
        oldest = _FEATURE_CACHE_ORDER.pop(0)
        _FEATURE_CACHE.pop(oldest, None)


def _hash_rings(cells: list[list[list[float]]]) -> str:
    """Stable hash of ring coordinates for cache keying."""
    h = hashlib.sha256()
    for ring in cells:
        for pt in ring:
            h.update(f"{pt[0]:.3f},{pt[1]:.3f};".encode("ascii"))
        h.update(b"|")
    return h.hexdigest()


# ── Per-cell feature computation ─────────────────────────────────────────────


def _ring_centroid(ring: list[list[float]]) -> tuple[float, float]:
    """Compute centroid of a TARGET_RING-point ring."""
    n = len(ring)
    if n == 0:
        return (0.0, 0.0)
    sx = sum(float(p[0]) for p in ring)
    sy = sum(float(p[1]) for p in ring)
    return (sx / n, sy / n)


def _ring_perimeter(ring: list[list[float]]) -> float:
    """Perimeter of a closed ring."""
    n = len(ring)
    if n < 2:
        return 0.0
    perim = 0.0
    for i in range(n):
        j = (i + 1) % n
        perim += math.hypot(
            float(ring[j][0]) - float(ring[i][0]),
            float(ring[j][1]) - float(ring[i][1]),
        )
    return perim


def _ring_circularity(area: float, perimeter: float) -> float:
    """Circularity: 4*pi*area / perimeter^2 (1.0 = perfect circle)."""
    if perimeter < 1e-12:
        return 0.0
    return (4.0 * math.pi * area) / (perimeter * perimeter)


def _ring_eccentricity(ring: list[list[float]]) -> float:
    """
    Eccentricity from covariance of ring vertices.
    0 = circle, approaching 1 = line.
    """
    cx, cy = _ring_centroid(ring)
    n = len(ring)
    if n < 3:
        return 0.0

    mxx = sum((float(p[0]) - cx) ** 2 for p in ring) / n
    myy = sum((float(p[1]) - cy) ** 2 for p in ring) / n
    mxy = sum((float(p[0]) - cx) * (float(p[1]) - cy) for p in ring) / n

    # Eigenvalues of 2x2 covariance matrix
    trace = mxx + myy
    det = mxx * myy - mxy * mxy
    if det < 0:
        det = 0.0
    discriminant = trace * trace - 4.0 * det
    if discriminant < 0:
        discriminant = 0.0
    lambda1 = (trace + math.sqrt(discriminant)) / 2.0
    lambda2 = (trace - math.sqrt(discriminant)) / 2.0

    if lambda1 < 1e-12:
        return 0.0
    return float(1.0 - math.sqrt(lambda2 / lambda1))


def _compute_knn_density(
    centroids: list[tuple[float, float]],
    k: int = 5,
) -> list[float]:
    """
    Compute k-NN density for each point.
    Density = k / (pi * r_k^2) where r_k is distance to k-th neighbor.
    """
    n = len(centroids)
    if n < 2:
        return [0.0] * n

    pts = np.array(centroids, dtype=np.float64)
    densities = []

    for i in range(n):
        dx = pts[:, 0] - pts[i, 0]
        dy = pts[:, 1] - pts[i, 1]
        dists = np.sqrt(dx * dx + dy * dy)
        # Sort distances, skip self (index 0)
        dists.sort()
        k_eff = min(k, n - 1)
        r_k = float(dists[k_eff])
        if r_k < 1e-12:
            densities.append(0.0)
        else:
            densities.append(float(k_eff / (math.pi * r_k * r_k)))
    return densities


def _compute_rbf_risk(
    centroids: list[tuple[float, float]],
    sigma: float | None = None,
) -> list[float]:
    """
    RBF microenvironment risk field at each centroid.
    Uses Gaussian RBF kernel: risk_i = mean_j(exp(-d_ij^2 / (2*sigma^2))).
    """
    n = len(centroids)
    if n < 2:
        return [0.0] * n

    pts = np.array(centroids, dtype=np.float64)

    if sigma is None:
        # Heuristic: sigma = median pairwise distance / 2
        sample = min(n, 200)
        indices = np.random.choice(n, size=sample, replace=False) if n > sample else np.arange(n)
        sample_pts = pts[indices]
        dx = sample_pts[:, None, 0] - sample_pts[None, :, 0]
        dy = sample_pts[:, None, 1] - sample_pts[None, :, 1]
        dists = np.sqrt(dx * dx + dy * dy)
        # Upper triangle excluding diagonal
        triu = dists[np.triu_indices(sample, k=1)]
        if len(triu) > 0:
            sigma = float(np.median(triu)) / 2.0
        else:
            sigma = 50.0
        if sigma < 1e-12:
            sigma = 50.0

    rbf_values = []
    denom = 2.0 * sigma * sigma
    for i in range(n):
        dx = pts[:, 0] - pts[i, 0]
        dy = pts[:, 1] - pts[i, 1]
        sq_dists = dx * dx + dy * dy
        weights = np.exp(-sq_dists / denom)
        rbf_values.append(float(np.mean(weights)))
    return rbf_values


def _compute_convex_hull(centroids: list[tuple[float, float]]) -> list[int]:
    """
    Compute convex hull indices (Monotone Chain algorithm).
    Returns indices of hull points in counterclockwise order.
    """
    n = len(centroids)
    if n < 3:
        return list(range(n))

    pts = [(float(x), float(y), i) for i, (x, y) in enumerate(centroids)]
    pts.sort()

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    # Remove last point of each half (duplicate of start of other half)
    hull = lower[:-1] + upper[:-1]
    return [p[2] for p in hull]


def _distance_to_hull_edge(
    centroids: list[tuple[float, float]],
    hull_indices: list[int],
) -> list[float]:
    """Compute minimum distance from each point to any convex hull edge."""
    n = len(centroids)
    if len(hull_indices) < 3:
        return [0.0] * n

    # Build hull edge segments
    hull_pts = [(centroids[i][0], centroids[i][1]) for i in hull_indices]
    m = len(hull_pts)
    edges = []
    for i in range(m):
        j = (i + 1) % m
        edges.append((hull_pts[i], hull_pts[j]))

    def point_to_segment(px, py, ax, ay, bx, by):
        """Minimum distance from point (px,py) to segment (ax,ay)-(bx,by)."""
        dx = bx - ax
        dy = by - ay
        if dx == 0 and dy == 0:
            return math.hypot(px - ax, py - ay)
        t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)))
        proj_x = ax + t * dx
        proj_y = ay + t * dy
        return math.hypot(px - proj_x, py - proj_y)

    distances = []
    for i in range(n):
        cx, cy = centroids[i]
        min_dist = float("inf")
        for (ax, ay), (bx, by) in edges:
            d = point_to_segment(cx, cy, ax, ay, bx, by)
            if d < min_dist:
                min_dist = d
        distances.append(min_dist)

    return distances


# ── Public API ───────────────────────────────────────────────────────────────

# Feature names (order must match _extract_features output)
FEATURE_NAMES = [
    "area_px2",
    "perimeter_px",
    "circularity",
    "eccentricity",
    "centroid_x",
    "centroid_y",
    "knn_density_k5",
    "rbf_risk_score",
    "distance_to_tumor_edge",
    "area_zscore",
    "circularity_penalty",
    "perimeter_area_ratio",
    "major_axis_length",
    "minor_axis_length",
    "solidity",
    "convex_hull_area",
    "extent",
    "equivalent_diameter",
    "form_factor",
]


def extract_features(
    gold_segment: dict[str, Any],
    *,
    knn_k: int = 5,
    use_cache: bool = True,
) -> list[dict[str, float]]:
    """
    Extract 19 numeric features per cell from a gold segment envelope.

    Args:
        gold_segment: Output of /v1/ingest/* (must have cells[] with 128-point rings)
        knn_k: Number of neighbors for k-NN density computation
        use_cache: Whether to use the LRU feature cache

    Returns:
        List of per-cell feature dicts, same order as cells[]

    Raises:
        ValueError: If gold_segment has no cells or invalid ring structure
    """
    cells = gold_segment.get("cells", [])
    if not cells:
        raise ValueError("gold_segment has no cells[]")

    # Cache lookup
    if use_cache:
        rings_hash = _hash_rings(cells)
        cache_key_str = _cache_key(rings_hash, knn_k)
        cached = _cache_get(cache_key_str)
        if cached is not None:
            # Reconstruct dicts from flat list
            return [
                dict(zip(FEATURE_NAMES, cached[i::len(cells)]))
                for i in range(len(cells))
            ] if False else _features_to_dicts(cached, len(cells))

    # Compute per-ring features
    areas = []
    perimeters = []
    circularities = []
    eccentricities = []
    centroids = []

    for ring in cells:
        area = _polygon_area_shoelace(ring)
        perimeter = _ring_perimeter(ring)
        circularity = _ring_circularity(area, perimeter)
        eccentricity = _ring_eccentricity(ring)
        cx, cy = _ring_centroid(ring)

        areas.append(area)
        perimeters.append(perimeter)
        circularities.append(circularity)
        eccentricities.append(eccentricity)
        centroids.append((cx, cy))

    n = len(cells)

    # Compute aggregate-derived features
    area_arr = np.array(areas, dtype=np.float64)
    median_area = float(np.median(area_arr)) if n > 0 else 1.0
    area_std = float(np.std(area_arr)) if n > 0 else 1.0

    knn_densities = _compute_knn_density(centroids, k=knn_k)
    rbf_risks = _compute_rbf_risk(centroids)
    hull_indices = _compute_convex_hull(centroids)
    edge_distances = _distance_to_hull_edge(centroids, hull_indices)

    # Build feature dicts
    features = []
    for i in range(n):
        a = areas[i]
        p = perimeters[i]
        c = circularities[i]
        e = eccentricities[i]
        cx, cy = centroids[i]

        # Derived features
        area_z = (a - median_area) / max(area_std, 1e-12)
        circ_penalty = 1.0 - c
        pa_ratio = p / max(a, 1e-12)

        # Major/minor axis from eccentricity and area
        # area = pi * a * b, eccentricity = sqrt(1 - b^2/a^2)
        # Solve: a = sqrt(area/(pi * sqrt(1 - e^2))), b = a * sqrt(1 - e^2)
        e_sq = max(0.0, min(0.9999, e * e))
        axis_factor = math.sqrt(max(1e-12, a / (math.pi * math.sqrt(max(1e-12, 1.0 - e_sq)))))
        major_axis = axis_factor
        minor_axis = axis_factor * math.sqrt(max(0.0, 1.0 - e_sq))

        # Convex hull area (approximate — use shoelace on ring as proxy)
        convex_hull_area = a  # ring is already a convex-like polygon

        # Solidity = area / convex_hull_area (approx 1 for ring-sampled polygons)
        solidity = 1.0 if convex_hull_area < 1e-12 else min(1.0, a / convex_hull_area)

        # Extent = area / bounding_box_area
        xs = [float(p[0]) for p in ring]
        ys = [float(p[1]) for p in ring]
        bbox_w = max(xs) - min(xs)
        bbox_h = max(ys) - min(ys)
        bbox_area = max(1e-12, bbox_w * bbox_h)
        extent = min(1.0, a / bbox_area)

        # Equivalent diameter = 2 * sqrt(area / pi)
        eq_diam = 2.0 * math.sqrt(max(0.0, a / math.pi))

        # Form factor = 4 * pi * area / perimeter^2 (same as circularity)
        form_factor = c

        features.append({
            "area_px2": float(a),
            "perimeter_px": float(p),
            "circularity": float(c),
            "eccentricity": float(e),
            "centroid_x": float(cx),
            "centroid_y": float(cy),
            "knn_density_k5": float(knn_densities[i]),
            "rbf_risk_score": float(rbf_risks[i]),
            "distance_to_tumor_edge": float(edge_distances[i]),
            "area_zscore": float(area_z),
            "circularity_penalty": float(circ_penalty),
            "perimeter_area_ratio": float(pa_ratio),
            "major_axis_length": float(major_axis),
            "minor_axis_length": float(minor_axis),
            "solidity": float(solidity),
            "convex_hull_area": float(convex_hull_area),
            "extent": float(extent),
            "equivalent_diameter": float(eq_diam),
            "form_factor": float(form_factor),
        })

    # Cache the flat list
    if use_cache and n > 0:
        flat = []
        for f in features:
            flat.extend(f[name] for name in FEATURE_NAMES)
        _cache_put(_cache_key(_hash_rings(cells), knn_k), flat)

    return features


def _features_to_dicts(flat: list[float], n_cells: int) -> list[dict[str, float]]:
    """Reconstruct per-cell dicts from flat feature vector list."""
    nf = len(FEATURE_NAMES)
    result = []
    for i in range(n_cells):
        d = {}
        for j, name in enumerate(FEATURE_NAMES):
            d[name] = flat[i + j * n_cells]
        result.append(d)
    return result


def features_to_csv(features: list[dict[str, float]]) -> str:
    """Convert feature list to CSV string with header."""
    if not features:
        return ",".join(FEATURE_NAMES) + "\n"

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=FEATURE_NAMES)
    writer.writeheader()
    writer.writerows(features)
    return output.getvalue()


def features_to_matrix(features: list[dict[str, float]]) -> list[list[float]]:
    """Convert feature list to 2D numeric array (no headers)."""
    return [[f[name] for name in FEATURE_NAMES] for f in features]


# Backward-compatible alias
compute_features = extract_features