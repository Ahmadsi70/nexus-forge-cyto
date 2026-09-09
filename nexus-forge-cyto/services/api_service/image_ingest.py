"""
Image segmentation ingress — SAM 3 only (subprocess-isolated from Streamlit).

Why: Cellpose removed; SAM 3 exemplar/text prompts cover image → polygon ingress.
"""

from __future__ import annotations

from api_service.sam3_segment import Sam3UnavailableError, sam3_image_bytes_to_polygons
from nexus_core.vector_ingest import FORMAT_SAM3_NUCLEI


def segment_image_bytes(
    raw: bytes,
    *,
    bboxes: list[list[float]] | None = None,
    text_prompt: str | None = None,
) -> tuple[list[list[tuple[float, float]]], str]:
    """
    Segment nuclei/objects in a patch image via SAM 3.

    Returns (polygons, format_hint) for gold_segment construction.
    """
    try:
        polys = sam3_image_bytes_to_polygons(raw, bboxes=bboxes, text_prompt=text_prompt)
    except Sam3UnavailableError as exc:
        raise ValueError(str(exc)) from exc
    except Exception as exc:
        raise ValueError(f"SAM 3 segmentation failed: {exc}") from exc

    if not polys:
        raise ValueError("SAM 3 found no nuclei in the uploaded image.")

    return polys, FORMAT_SAM3_NUCLEI
