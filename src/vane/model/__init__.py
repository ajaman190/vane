"""Vane model package. Requires torch — keeping ``import vane`` CPU-light."""

from __future__ import annotations

try:
    import torch as _torch  # noqa: F401
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "vane.model requires PyTorch. Install torch to use the network "
        "(e.g. pip install torch). Core `import vane` stays available without torch."
    ) from e

from .config import DEV_CONFIG, FULL_TEXT_CONFIG, VaneConfig
from .confidence import cardinality_temperature, confidence
from .vane import ContextTooLongError, VaneModel, VaneOutput

__all__ = [
    "VaneConfig",
    "VaneModel",
    "VaneOutput",
    "DEV_CONFIG",
    "FULL_TEXT_CONFIG",
    "ContextTooLongError",
    "confidence",
    "cardinality_temperature",
]
