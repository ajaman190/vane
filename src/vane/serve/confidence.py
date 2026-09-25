"""Confidence readout for choice/score answers.

Single stdlib source of truth: ``vane.eval.metrics.confidence``.
Torch tensor C(q) lives only in ``vane.model.confidence``.
"""

from __future__ import annotations

from vane.eval.metrics import confidence

__all__ = ["confidence"]
