"""SAM 3 image ingress tests."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

_SERVICES = Path(__file__).resolve().parents[1] / "services"
if str(_SERVICES) not in sys.path:
    sys.path.insert(0, str(_SERVICES))

from api_service.image_ingest import segment_image_bytes  # noqa: E402
from api_service.sam3_segment import Sam3UnavailableError  # noqa: E402
from nexus_core.vector_ingest import FORMAT_SAM3_NUCLEI  # noqa: E402

_SQUARE = [[(0.0, 0.0), (20.0, 0.0), (20.0, 20.0), (0.0, 20.0), (0.0, 0.0)]]


@patch("api_service.image_ingest.sam3_image_bytes_to_polygons", return_value=_SQUARE)
def test_segment_image_bytes_uses_sam3(mock_sam3) -> None:
    polys, fmt = segment_image_bytes(b"fake")
    assert len(polys) == 1
    assert fmt == FORMAT_SAM3_NUCLEI
    mock_sam3.assert_called_once()


@patch(
    "api_service.image_ingest.sam3_image_bytes_to_polygons",
    side_effect=Sam3UnavailableError("no weights"),
)
def test_segment_raises_when_sam3_unavailable(_mock_sam3) -> None:
    with pytest.raises(ValueError, match="no weights"):
        segment_image_bytes(b"fake")


@patch("api_service.image_ingest.sam3_image_bytes_to_polygons", return_value=[])
def test_segment_raises_when_no_nuclei_found(_mock_sam3) -> None:
    with pytest.raises(ValueError, match="no nuclei"):
        segment_image_bytes(b"fake")
