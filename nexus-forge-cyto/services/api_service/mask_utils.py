"""Shared helpers: image decode and mask/outline → polygon conversion."""

from __future__ import annotations

import io
from typing import Iterable

import numpy as np


def load_rgb_array(raw: bytes) -> np.ndarray:
    """Decode PNG/JPEG/TIF bytes to H×W×3 uint8 array."""
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Pillow is required for image ingest (pip install Pillow).") from exc

    img = Image.open(io.BytesIO(raw))
    if img.mode != "RGB":
        img = img.convert("RGB")
    return np.asarray(img)


def load_grayscale_array(raw: bytes) -> np.ndarray:
    """Decode image bytes to single-channel uint8."""
    rgb = load_rgb_array(raw)
    if rgb.ndim == 3:
        return np.mean(rgb, axis=2).astype(np.uint8)
    return rgb.astype(np.uint8)


def masks_to_polygons(masks: Iterable[np.ndarray], *, min_area: float = 20.0) -> list[list[tuple[float, float]]]:
    """Convert binary masks to closed polygon vertex lists (pixel coordinates)."""
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("opencv-python is required for mask→polygon (pip install opencv-python-headless).") from exc

    polygons: list[list[tuple[float, float]]] = []
    for mask in masks:
        arr = np.asarray(mask)
        if arr.size == 0:
            continue
        binary = (arr > 0).astype(np.uint8) * 255
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(contour) < min_area:
            continue
        pts = [(float(p[0][0]), float(p[0][1])) for p in contour]
        if len(pts) < 3:
            continue
        if pts[0] != pts[-1]:
            pts.append(pts[0])
        polygons.append(pts)
    return polygons
