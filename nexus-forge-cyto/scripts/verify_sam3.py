#!/usr/bin/env python3
"""Smoke-check SAM 3 runtime: CUDA, weights, Ultralytics predictor import."""
from __future__ import annotations

import sys
from pathlib import Path

_SERVICES = Path(__file__).resolve().parents[1] / "services"
if str(_SERVICES) not in sys.path:
    sys.path.insert(0, str(_SERVICES))

import torch  # noqa: E402
from api_service.sam3_segment import _sam3_model_path  # noqa: E402
from ultralytics.models.sam import SAM3SemanticPredictor  # noqa: E402

print("cuda:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
print("weights:", _sam3_model_path())
print("predictor:", SAM3SemanticPredictor)
print("sam3_ok")
