"""H&E nuclei exemplar proposer: hematoxylin peaks, NMS, SAM3 positive/negative boxes."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_SERVICES = Path(__file__).resolve().parents[1] / "services"
if str(_SERVICES) not in sys.path:
    sys.path.insert(0, str(_SERVICES))

from api_service.nuclei_proposer import (  # noqa: E402
    bbox_iou,
    hematoxylin_channel,
    nms_bboxes,
    propose_nuclei_exemplars,
)


def _synthetic_he_patch(h: int = 256, w: int = 256) -> np.ndarray:
    """Pink stroma + dark hematoxylin-stained nuclei blobs."""
    rgb = np.full((h, w, 3), 235, dtype=np.uint8)
    rgb[..., 0] = 210
    rgb[..., 2] = 200
    centers = [(48, 52), (118, 76), (168, 142), (82, 168), (190, 48)]
    yy, xx = np.ogrid[:h, :w]
    for cx, cy in centers:
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= 11**2
        rgb[mask] = np.array([70, 50, 110], dtype=np.uint8)
    return rgb


def test_hematoxylin_peaks_at_nuclei_centers() -> None:
    rgb = _synthetic_he_patch()
    h = hematoxylin_channel(rgb)
    assert h.shape == rgb.shape[:2]
    # Nuclei regions should be brighter in H channel than mean stroma
    assert float(h[52, 48]) > float(np.percentile(h, 60))


def test_nms_removes_overlapping_boxes() -> None:
    boxes = [
        [10.0, 10.0, 30.0, 30.0],
        [12.0, 12.0, 32.0, 32.0],
        [100.0, 100.0, 130.0, 130.0],
    ]
    scores = [0.9, 0.8, 0.7]
    kept = nms_bboxes(boxes, scores, iou_threshold=0.3)
    assert len(kept) == 2
    assert bbox_iou(boxes[kept[0]], boxes[kept[1]]) < 0.3


def test_propose_returns_positive_and_negative_exemplars() -> None:
    rgb = _synthetic_he_patch()
    prompt = propose_nuclei_exemplars(rgb, max_positive=8, max_negative=3)
    assert len(prompt.bboxes) == len(prompt.labels)
    assert sum(prompt.labels) >= 3  # at least 3 positive
    assert 0 in prompt.labels  # at least one negative stroma box
    for box in prompt.bboxes:
        assert len(box) == 4
        assert box[2] > box[0] and box[3] > box[1]


def test_propose_empty_on_blank_image() -> None:
    rgb = np.full((64, 64, 3), 255, dtype=np.uint8)
    prompt = propose_nuclei_exemplars(rgb)
    assert prompt.bboxes == []
    assert prompt.labels == []
