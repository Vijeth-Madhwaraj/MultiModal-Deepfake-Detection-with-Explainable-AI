"""Explainable rule-based fusion for deepfake detector results."""

from .engine import FusionEngine
from .models import InputReport, ValidationError
from .pipeline import InferenceCommand, PipelineError, VideoPipeline

__all__ = [
    "FusionEngine",
    "InferenceCommand",
    "InputReport",
    "PipelineError",
    "ValidationError",
    "VideoPipeline",
]
__version__ = "1.0.0"
