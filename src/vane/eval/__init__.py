"""Evaluation helpers for Vane: confidence, early exit, calibration, smoke metrics.

Pure numpy/stdlib — no torch required. Import as::

    from vane.eval import (
        confidence,
        early_exit_allowed,
        expected_calibration_error,
        evaluate_batch,
    )
"""

from __future__ import annotations

from vane.eval.metrics import (
    confidence,
    early_exit_allowed,
    eval_smoke,
    evaluate_batch,
    expected_calibration_error,
)

__all__ = [
    "confidence",
    "early_exit_allowed",
    "expected_calibration_error",
    "evaluate_batch",
    "eval_smoke",
]
