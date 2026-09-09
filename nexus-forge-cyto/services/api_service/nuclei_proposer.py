"""
H&E nuclei exemplar proposer for SAM 3 SemanticPredictor.

Why: pathology literature shows box exemplars beat text prompts; hematoxylin deconv
localizes nuclei before SAM3 draws precise instance polygons.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

# Ruifrok & Johnston stain vectors (H, E) in RGB optical-density space.
_STAIN_MATRIX = np.array(
    [
        [0.650, 0.704, 0.286],
        [0.268, 0.570, 0.776],
    ],
    dtype=np.float64,
)


@dataclass(frozen=True)
class NucleiExemplarPrompt:
    """SAM 3 exemplar boxes with parallel positive (1) / negative (0) labels."""

    bboxes: list[list[float]]
    labels: list[int]


def hematoxylin_channel(rgb: np.ndarray) -> np.ndarray:
    """Deconvolve H&E RGB → hematoxylin concentration map (high at nuclei)."""
    arr = np.asarray(rgb, dtype=np.float64)
    if arr.ndim != 3 or arr.shape[2] < 3:
        raise ValueError("rgb must be H×W×3")
    od = -np.log(np.clip(arr[..., :3] / 255.0, 1e-6, 1.0))
    flat = od.reshape(-1, 3).T
    conc, _, _, _ = np.linalg.lstsq(_STAIN_MATRIX.T, flat, rcond=None)
    h_map = conc[0].reshape(arr.shape[:2])
    h_map = h_map - float(h_map.min())
    peak = float(h_map.max())
    if peak > 0:
        h_map = h_map / peak
    return h_map.astype(np.float32)


def bbox_iou(a: list[float], b: list[float]) -> float:
    """Intersection-over-union for axis-aligned boxes [x1,y1,x2,y2]."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def nms_bboxes(
    boxes: list[list[float]],
    scores: list[float],
    *,
    iou_threshold: float = 0.35,
) -> list[int]:
    """Greedy NMS; returns indices kept sorted by score."""
    if not boxes:
        return []
    order = np.argsort(scores)[::-1]
    kept: list[int] = []
    for idx in order:
        i = int(idx)
        if all(bbox_iou(boxes[i], boxes[j]) <= iou_threshold for j in kept):
            kept.append(i)
    return kept


def _box_from_center(cx: float, cy: float, half: float, w: int, h: int) -> list[float]:
    x1 = max(0.0, cx - half)
    y1 = max(0.0, cy - half)
    x2 = min(float(w - 1), cx + half)
    y2 = min(float(h - 1), cy + half)
    return [x1, y1, x2, y2]


def _local_peaks(h_map: np.ndarray, *, min_distance: int, percentile: float) -> list[tuple[float, float, float]]:
    from scipy.ndimage import gaussian_filter, maximum_filter

    blur = gaussian_filter(h_map.astype(np.float32), sigma=1.0)
    thresh = float(np.percentile(blur, percentile))
    size = max(3, min_distance * 2 + 1)
    maxf = maximum_filter(blur, size=size)
    peaks_mask = (blur >= thresh) & (blur == maxf)
    ys, xs = np.where(peaks_mask)
    if len(xs) == 0:
        return []
    scores = blur[ys, xs]
    order = np.argsort(scores)[::-1]
    picked: list[tuple[float, float, float]] = []
    for idx in order:
        cx = float(xs[idx])
        cy = float(ys[idx])
        sc = float(scores[idx])
        if all((cx - px) ** 2 + (cy - py) ** 2 >= min_distance**2 for px, py, _ in picked):
            picked.append((cx, cy, sc))
    return picked


def _negative_stroma_boxes(
    h_map: np.ndarray,
    positive_centers: list[tuple[float, float]],
    *,
    max_negative: int,
    half_size: float,
    min_dist: float,
) -> list[list[float]]:
    """Sample low-H stroma regions as negative exemplars (PromptNucSeg-style)."""
    from scipy.ndimage import gaussian_filter

    h, w = h_map.shape
    blur = gaussian_filter(h_map.astype(np.float32), sigma=2.0)
    low_thresh = float(np.percentile(blur, 25))
    ys, xs = np.where(blur <= low_thresh)
    if len(xs) == 0:
        return []

    scores = low_thresh - blur[ys, xs]
    order = np.argsort(scores)[::-1]
    negatives: list[list[float]] = []
    for idx in order:
        cx = float(xs[idx])
        cy = float(ys[idx])
        if any((cx - px) ** 2 + (cy - py) ** 2 < min_dist**2 for px, py in positive_centers):
            continue
        box = _box_from_center(cx, cy, half_size, w, h)
        if box[2] - box[0] < 4 or box[3] - box[1] < 4:
            continue
        if any(bbox_iou(box, nb) > 0.2 for nb in negatives):
            continue
        negatives.append(box)
        if len(negatives) >= max_negative:
            break
    return negatives


def propose_nuclei_exemplars(
    rgb: np.ndarray,
    *,
    max_positive: int | None = None,
    max_negative: int | None = None,
    half_size: int | None = None,
    nms_iou: float | None = None,
    peak_percentile: float = 70.0,
) -> NucleiExemplarPrompt:
    """
    Hematoxylin peaks → NMS positive boxes + stroma negative boxes for SAM 3.

    Tunables via env: NEXUS_SAM3_MAX_POS_BOXES, NEXUS_SAM3_MAX_NEG_BOXES,
    NEXUS_SAM3_BOX_HALF, NEXUS_SAM3_NMS_IOU.
    """
    if rgb.size == 0:
        return NucleiExemplarPrompt(bboxes=[], labels=[])

    max_pos = max_positive if max_positive is not None else int(os.environ.get("NEXUS_SAM3_MAX_POS_BOXES", "80"))
    max_neg = max_negative if max_negative is not None else int(os.environ.get("NEXUS_SAM3_MAX_NEG_BOXES", "4"))
    half = float(half_size if half_size is not None else int(os.environ.get("NEXUS_SAM3_BOX_HALF", "14")))
    iou_thr = nms_iou if nms_iou is not None else float(os.environ.get("NEXUS_SAM3_NMS_IOU", "0.30"))
    peak_pct = float(os.environ.get("NEXUS_SAM3_PEAK_PERCENTILE", str(peak_percentile)))

    h_map = hematoxylin_channel(rgb)
    # Flat / empty slides: no stain contrast → no reliable nuclei peaks.
    if float(np.std(h_map)) < 0.02:
        return NucleiExemplarPrompt(bboxes=[], labels=[])

    h_img, w_img = h_map.shape
    # Dense MoNuSeg nuclei: allow closer peaks than the box half-size.
    min_distance = max(5, int(half * 0.45))

    peaks = _local_peaks(h_map, min_distance=min_distance, percentile=peak_pct)
    if not peaks:
        return NucleiExemplarPrompt(bboxes=[], labels=[])

    pos_boxes = [_box_from_center(cx, cy, half, w_img, h_img) for cx, cy, _ in peaks]
    pos_scores = [sc for _, _, sc in peaks]
    keep = nms_bboxes(pos_boxes, pos_scores, iou_threshold=iou_thr)[:max_pos]

    positive_boxes = [pos_boxes[i] for i in keep]
    positive_centers = [(peaks[i][0], peaks[i][1]) for i in keep]
    negative_boxes = _negative_stroma_boxes(
        h_map,
        positive_centers,
        max_negative=max_neg,
        half_size=half,
        min_dist=max(half * 2.5, 12.0),
    )

    bboxes = positive_boxes + negative_boxes
    labels = [1] * len(positive_boxes) + [0] * len(negative_boxes)
    return NucleiExemplarPrompt(bboxes=bboxes, labels=labels)


def auto_exemplar_bboxes(
    gray: np.ndarray,
    *,
    rgb: np.ndarray | None = None,
    max_boxes: int = 12,
    half_size: int = 18,
) -> list[list[float]]:
    """Backward-compatible positive-only box list (uses hematoxylin when rgb supplied)."""
    if rgb is not None:
        prompt = propose_nuclei_exemplars(
            rgb,
            max_positive=max_boxes,
            max_negative=0,
            half_size=half_size,
        )
        return prompt.bboxes

    # Legacy grayscale blob path (fallback)
    from scipy import ndimage

    if gray.size == 0:
        return []
    blur = ndimage.gaussian_filter(gray.astype(np.float32), sigma=1.2)
    thresh = float(np.percentile(blur, 82))
    binary = blur > thresh
    labeled, n = ndimage.label(binary)
    if n == 0:
        return []

    sizes = ndimage.sum(binary, labeled, range(1, n + 1))
    order = np.argsort(sizes)[::-1]
    h, w = gray.shape[:2]
    bboxes: list[list[float]] = []
    for idx in order[:max_boxes]:
        label = int(idx) + 1
        ys, xs = np.where(labeled == label)
        if len(xs) < 12:
            continue
        cx = float(xs.mean())
        cy = float(ys.mean())
        bboxes.append(_box_from_center(cx, cy, float(half_size), w, h))
    return bboxes
