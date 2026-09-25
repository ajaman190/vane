"""Tests for vane.eval — confidence, early exit, ECE smoke. No HF / GPU."""

from __future__ import annotations

import math

import pytest

from vane.eval import (
    confidence,
    early_exit_allowed,
    eval_smoke,
    evaluate_batch,
    expected_calibration_error,
)


# --- Confidence -------------------------------------------------------------


def test_confidence_uniform_is_zero():
    q = [0.25, 0.25, 0.25, 0.25]
    assert confidence(q) == pytest.approx(0.0)


def test_confidence_point_mass_is_one():
    q = [1.0, 0.0, 0.0, 0.0]
    assert confidence(q) == pytest.approx(1.0)


def test_confidence_k4_max096():
    # architecture.md worked example: max q = 0.96 → C = 0.947
    q = [0.96, 0.02, 0.01, 0.01]
    expected = (4 * 0.96 - 1) / 3
    assert expected == pytest.approx(0.947, rel=1e-3)
    assert confidence(q) == pytest.approx(expected)
    assert confidence(q) == pytest.approx(0.947, rel=1e-3)


def test_confidence_rejects_scalar_noul():
    with pytest.raises(ValueError, match="k>=2"):
        confidence([0.97])


# --- Early exit -------------------------------------------------------------


def test_early_exit_one_interior_blocks():
    # Three sharp Cs and one interior — whole request stays on the stack.
    Cs = [0.95, 0.92, 0.30, 0.88]  # index 2 is interior
    tau = 0.80
    assert early_exit_allowed(Cs, tau=tau) is False


def test_early_exit_all_clear():
    Cs = [0.95, 0.92, 0.88, 0.91]
    assert early_exit_allowed(Cs, tau=0.80) is True


def test_early_exit_per_question_tau():
    Cs = [0.95, 0.70]
    # Second clears its own lower tau but would fail a shared 0.80.
    assert early_exit_allowed(Cs, tau=[0.90, 0.65]) is True
    assert early_exit_allowed(Cs, tau=0.80) is False


# --- ECE --------------------------------------------------------------------


def test_ece_finite_on_tiny_fixture():
    probs = [0.1, 0.3, 0.6, 0.9, 0.85, 0.2]
    outcomes = [0.0, 0.0, 1.0, 1.0, 1.0, 0.0]
    ece = expected_calibration_error(probs, outcomes, n_bins=5)
    assert math.isfinite(ece)
    assert ece >= 0.0


def test_ece_perfect_calibration_near_zero():
    # Perfectly matched probs/outcomes in one bin each → ECE ≈ 0.
    probs = [0.0, 1.0, 0.0, 1.0]
    outcomes = [0.0, 1.0, 0.0, 1.0]
    ece = expected_calibration_error(probs, outcomes, n_bins=2)
    assert ece == pytest.approx(0.0)


# --- Smoke / evaluate_batch -------------------------------------------------


def test_evaluate_batch_smoke_metrics():
    predictions = [0.1, 0.8, 0.7, 0.2]
    targets = [0.0, 1.0, 1.0, 0.0]
    metrics = evaluate_batch(predictions, targets)
    assert set(metrics) >= {"brier", "logloss", "ece"}
    assert math.isfinite(metrics["brier"])
    assert math.isfinite(metrics["logloss"])
    assert math.isfinite(metrics["ece"])
    assert metrics["brier"] >= 0.0
    assert metrics["logloss"] >= 0.0
    assert metrics["ece"] >= 0.0


def test_eval_smoke_alias():
    predictions = [[0.9, 0.05, 0.05], [0.2, 0.5, 0.3]]
    targets = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    a = evaluate_batch(predictions, targets)
    b = eval_smoke(predictions, targets)
    assert a == b
