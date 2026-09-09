"""
SAM 3 concept / box segmentation → polygon list.

Why: Ultralytics SAM3 segments nuclei into precise contours from
hematoxylin-stained exemplars or explicit box prompts.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np

from api_service.mask_utils import load_rgb_array, masks_to_polygons
from api_service.nuclei_proposer import propose_nuclei_exemplars
from api_service.tile_segment import (
    dedupe_polygons_by_centroid,
    iter_image_tiles,
    shift_polygon,
)


class Sam3UnavailableError(RuntimeError):
    """Raised when SAM 3 / Ultralytics / weights are missing."""


def _sam3_model_path() -> Path:
    raw = os.environ.get("NEXUS_SAM3_MODEL_PATH", "").strip()
    if raw:
        p = Path(raw).expanduser()
        if p.is_file():
            return p
        raise Sam3UnavailableError(f"NEXUS_SAM3_MODEL_PATH not found: {p}")
    default = Path.home() / ".cache" / "nexus-forge" / "sam3.pt"
    if default.is_file():
        return default
    raise Sam3UnavailableError(
        "SAM 3 weights missing. Set NEXUS_SAM3_MODEL_PATH to sam3.pt (Ultralytics)."
    )


def _extract_masks_from_sam3_results(results) -> list[np.ndarray]:
    masks_out: list[np.ndarray] = []
    for result in results or []:
        if result is None:
            continue
        masks_obj = getattr(result, "masks", None)
        if masks_obj is None:
            continue
        data = getattr(masks_obj, "data", None)
        if data is None:
            continue
        arr = data.cpu().numpy() if hasattr(data, "cpu") else np.asarray(data)
        if arr.ndim == 3:
            for i in range(arr.shape[0]):
                masks_out.append(arr[i])
        elif arr.ndim == 2:
            masks_out.append(arr)
    return masks_out


def _predictor_overrides(model_path: Path) -> dict:
    overrides = {
        "conf": float(os.environ.get("NEXUS_SAM3_CONF", "0.15")),
        "task": "segment",
        "mode": "predict",
        "model": str(model_path),
        "save": False,
    }
    vocab = os.environ.get("NEXUS_SAM3_BPE_VOCAB", "").strip()
    if vocab:
        overrides["vocab"] = vocab
    return overrides


def _run_sam3_on_rgb(
    rgb: np.ndarray,
    *,
    predictor,
    bboxes: list[list[float]],
    labels: list[int] | None,
) -> list[list[tuple[float, float]]]:
    """Run one SAM3SemanticPredictor pass on an RGB crop; polygons in crop coords."""
    from PIL import Image

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        Image.fromarray(rgb).save(tmp.name)
        img_path = tmp.name
    try:
        predictor.set_image(img_path)
        if labels is not None:
            results = predictor(bboxes=bboxes, labels=labels)
        else:
            results = predictor(bboxes=bboxes)
        masks = _extract_masks_from_sam3_results(results)
        return masks_to_polygons(masks)
    finally:
        Path(img_path).unlink(missing_ok=True)


def _tiled_auto_segment(rgb: np.ndarray, *, predictor) -> list[list[tuple[float, float]]]:
    """Dense hematoxylin proposers per overlapping tile → SAM3 → global dedupe."""
    tile_size = int(os.environ.get("NEXUS_SAM3_TILE_SIZE", "512"))
    overlap = int(os.environ.get("NEXUS_SAM3_TILE_OVERLAP", "64"))
    dedupe_dist = float(os.environ.get("NEXUS_SAM3_DEDUPE_DIST", "12"))
    max_pos = int(os.environ.get("NEXUS_SAM3_MAX_POS_BOXES", "80"))
    max_neg = int(os.environ.get("NEXUS_SAM3_MAX_NEG_BOXES", "4"))

    all_polys: list[list[tuple[float, float]]] = []
    for x0, y0, crop in iter_image_tiles(rgb, tile_size=tile_size, overlap=overlap):
        prompt = propose_nuclei_exemplars(
            crop,
            max_positive=max_pos,
            max_negative=max_neg,
        )
        if not prompt.bboxes:
            continue
        local = _run_sam3_on_rgb(
            crop,
            predictor=predictor,
            bboxes=prompt.bboxes,
            labels=prompt.labels if prompt.labels else None,
        )
        for poly in local:
            all_polys.append(shift_polygon(poly, dx=float(x0), dy=float(y0)))

    return dedupe_polygons_by_centroid(all_polys, min_dist=dedupe_dist)


def sam3_image_bytes_to_polygons(
    raw: bytes,
    *,
    bboxes: list[list[float]] | None = None,
    text_prompt: str | None = None,
) -> list[list[tuple[float, float]]]:
    """
    Run SAM 3 segmentation; returns polygon lists in pixel space.

    Priority: explicit bboxes → text → tiled hematoxylin.
    """
    try:
        from ultralytics.models.sam import SAM3SemanticPredictor
    except ImportError as exc:
        raise Sam3UnavailableError(
            "Ultralytics SAM 3 not installed. pip install 'ultralytics>=8.3.237'"
        ) from exc

    model_path = _sam3_model_path()
    rgb = load_rgb_array(raw)

    if (text_prompt or "").strip():
        from PIL import Image

        predictor = SAM3SemanticPredictor(overrides=_predictor_overrides(model_path))
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            Image.fromarray(rgb).save(tmp.name)
            img_path = tmp.name
        try:
            predictor.set_image(img_path)
            results = predictor(text=[text_prompt.strip()])
            return masks_to_polygons(_extract_masks_from_sam3_results(results))
        finally:
            Path(img_path).unlink(missing_ok=True)

    if bboxes:
        predictor = SAM3SemanticPredictor(overrides=_predictor_overrides(model_path))
        return _run_sam3_on_rgb(rgb, predictor=predictor, bboxes=list(bboxes), labels=None)

    predictor = SAM3SemanticPredictor(overrides=_predictor_overrides(model_path))
    use_tiles = os.environ.get("NEXUS_SAM3_TILE", "1").strip() not in {"0", "false", "False"}
    if use_tiles:
        return _tiled_auto_segment(rgb, predictor=predictor)

    prompt = propose_nuclei_exemplars(rgb)
    if not prompt.bboxes:
        return []
    return _run_sam3_on_rgb(
        rgb,
        predictor=predictor,
        bboxes=prompt.bboxes,
        labels=prompt.labels if prompt.labels else None,
    )
