"""Confidence, early-exit gate, ECE, and in-memory smoke metrics.

Formulas follow docs/architecture.md (Confidence, Early exit) and
docs/training.md (ECE release checks). No dataset download; no GPU.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

# ---------------------------------------------------------------------------
# Confidence
# ---------------------------------------------------------------------------


def confidence(q: Sequence[float]) -> float:
    """Confidence on a simplex with ``k >= 2`` support points.

    .. math::

        C(q) = \\frac{k \\cdot \\max_i q_i - 1}{k - 1}

    On a simplex, ``max q ∈ [1/k, 1]``. Uniform → 0; point mass → 1.
    Noul has no confidence field; do not call this on a scalar yes-prob.
    """
    k = len(q)
    if k < 2:
        raise ValueError(f"confidence requires simplex with k>=2, got k={k}")
    max_q = max(float(x) for x in q)
    return (k * max_q - 1.0) / (k - 1.0)


# ---------------------------------------------------------------------------
# Early exit
# ---------------------------------------------------------------------------


def early_exit_allowed(
    confidences_or_Cs: Sequence[float],
    tau: float | Sequence[float] | Mapping[Any, float] | None = None,
    *,
    thresholds: float | Sequence[float] | Mapping[Any, float] | None = None,
    types: Sequence[Any] | None = None,
) -> bool:
    """Return True iff every question clears its ``τ(type, k)``.

    Early exit skips later encoder blocks only when **every** choice/score
    confidence is at or above its threshold. One interior question blocks
    exit for the whole request (architecture.md).

    Parameters
    ----------
    confidences_or_Cs
        Per-question ``C(q)`` values already computed (noul is omitted by
        the caller — noul has no ``C``).
    tau, thresholds
        Scalar, parallel sequence, or map keyed by ``types``. Exactly one of
        ``tau`` / ``thresholds`` must be provided (``thresholds`` is an alias).
    types
        Optional keys into a mapping ``tau`` / ``thresholds``.
    """
    if tau is not None and thresholds is not None:
        raise ValueError("pass only one of tau or thresholds")
    if tau is None and thresholds is None:
        raise ValueError("tau (or thresholds) is required")
    thr = thresholds if thresholds is not None else tau
    assert thr is not None

    if not confidences_or_Cs:
        return True

    resolved = _resolve_thresholds(len(confidences_or_Cs), thr, types)
    return all(float(c) >= float(t) for c, t in zip(confidences_or_Cs, resolved))


def _resolve_thresholds(
    n: int,
    thr: float | Sequence[float] | Mapping[Any, float],
    types: Sequence[Any] | None,
) -> list[float]:
    if isinstance(thr, Mapping):
        if types is None:
            raise ValueError("types required when tau/thresholds is a mapping")
        if len(types) != n:
            raise ValueError(
                f"types length {len(types)} != confidences length {n}"
            )
        return [float(thr[t]) for t in types]
    if isinstance(thr, (int, float)):
        return [float(thr)] * n
    seq = list(thr)
    if len(seq) != n:
        raise ValueError(
            f"thresholds length {len(seq)} != confidences length {n}"
        )
    return [float(t) for t in seq]


# ---------------------------------------------------------------------------
# ECE (Naeini et al. binned)
# ---------------------------------------------------------------------------


def expected_calibration_error(
    probs: Sequence[float],
    outcomes: Sequence[float],
    *,
    n_bins: int = 10,
) -> float:
    """Binned expected calibration error (Naeini et al.).

    Partitions predicted probabilities into ``n_bins`` equal-width bins on
    ``[0, 1]`` and returns

    .. math::

        \\mathrm{ECE} = \\sum_m \\frac{|B_m|}{n}\\,|\\mathrm{acc}(B_m) - \\mathrm{conf}(B_m)|

    where ``acc`` is the mean outcome and ``conf`` is the mean predicted
    probability in bin ``m``. Empty bins contribute 0.

    ``probs`` and ``outcomes`` are parallel sequences in ``[0, 1]`` (binary
    predicted probability vs binary / soft outcome). No dataset download.
    """
    if n_bins < 1:
        raise ValueError(f"n_bins must be >= 1, got {n_bins}")
    if len(probs) != len(outcomes):
        raise ValueError(
            f"probs length {len(probs)} != outcomes length {len(outcomes)}"
        )
    n = len(probs)
    if n == 0:
        return 0.0

    # Equal-width bins on [0, 1]; last bin includes 1.0.
    bin_sums_p = [0.0] * n_bins
    bin_sums_y = [0.0] * n_bins
    bin_counts = [0] * n_bins

    for p, y in zip(probs, outcomes):
        p_f = float(p)
        y_f = float(y)
        if p_f < 0.0 or p_f > 1.0:
            raise ValueError(f"probability out of [0, 1]: {p_f}")
        idx = min(n_bins - 1, int(p_f * n_bins))
        bin_sums_p[idx] += p_f
        bin_sums_y[idx] += y_f
        bin_counts[idx] += 1

    ece = 0.0
    for m in range(n_bins):
        count = bin_counts[m]
        if count == 0:
            continue
        conf_m = bin_sums_p[m] / count
        acc_m = bin_sums_y[m] / count
        ece += (count / n) * abs(acc_m - conf_m)
    return ece


# ---------------------------------------------------------------------------
# Smoke / batch metrics
# ---------------------------------------------------------------------------


def evaluate_batch(
    predictions: Sequence[float] | Sequence[Sequence[float]],
    targets: Sequence[float] | Sequence[Sequence[float]],
    *,
    n_bins: int = 10,
) -> dict[str, float]:
    """In-memory metrics dict: ``brier``, ``logloss``, ``ece``.

    Accepts either binary 1-d vectors or multiclass rows (simplex vs target
    distribution / one-hot). No GPU. Tiny fixtures are enough for smoke.
    """
    pred_rows, tgt_rows = _as_rows(predictions, targets)
    brier = _brier(pred_rows, tgt_rows)
    logloss = _logloss(pred_rows, tgt_rows)

    # ECE on scalar confidence: for binary use p(yes); for multiclass use
    # max-prob vs correctness (hard argmax match to argmax of target).
    if len(pred_rows[0]) == 1:
        probs = [row[0] for row in pred_rows]
        outcomes = [row[0] for row in tgt_rows]
    else:
        probs = [max(row) for row in pred_rows]
        outcomes = [
            1.0 if _argmax(p) == _argmax(t) else 0.0
            for p, t in zip(pred_rows, tgt_rows)
        ]

    ece = expected_calibration_error(probs, outcomes, n_bins=n_bins)
    return {"brier": brier, "logloss": logloss, "ece": ece}


def eval_smoke(
    predictions: Sequence[float] | Sequence[Sequence[float]],
    targets: Sequence[float] | Sequence[Sequence[float]],
    *,
    n_bins: int = 10,
) -> dict[str, float]:
    """Alias for :func:`evaluate_batch` — tiny in-memory smoke metrics."""
    return evaluate_batch(predictions, targets, n_bins=n_bins)


def _as_rows(
    predictions: Sequence[float] | Sequence[Sequence[float]],
    targets: Sequence[float] | Sequence[Sequence[float]],
) -> tuple[list[list[float]], list[list[float]]]:
    if len(predictions) != len(targets):
        raise ValueError(
            f"predictions length {len(predictions)} != targets length {len(targets)}"
        )
    if len(predictions) == 0:
        raise ValueError("predictions/targets must be non-empty")

    pred_rows = [_row(x) for x in predictions]
    tgt_rows = [_row(x) for x in targets]
    widths = {len(r) for r in pred_rows} | {len(r) for r in tgt_rows}
    if len(widths) != 1:
        raise ValueError(f"inconsistent row widths: {sorted(widths)}")
    return pred_rows, tgt_rows


def _row(x: float | Sequence[float]) -> list[float]:
    if isinstance(x, (int, float)):
        return [float(x)]
    return [float(v) for v in x]


def _argmax(row: Sequence[float]) -> int:
    best_i = 0
    best_v = float(row[0])
    for i, v in enumerate(row):
        fv = float(v)
        if fv > best_v:
            best_v = fv
            best_i = i
    return best_i


def _brier(
    pred_rows: list[list[float]], tgt_rows: list[list[float]]
) -> float:
    n = len(pred_rows)
    total = 0.0
    for p, t in zip(pred_rows, tgt_rows):
        total += sum((pi - ti) ** 2 for pi, ti in zip(p, t))
    return total / n


def _logloss(
    pred_rows: list[list[float]],
    tgt_rows: list[list[float]],
    *,
    eps: float = 1e-12,
) -> float:
    n = len(pred_rows)
    total = 0.0
    for p, t in zip(pred_rows, tgt_rows):
        if len(p) == 1:
            q = min(1.0 - eps, max(eps, p[0]))
            y = t[0]
            total += -(y * math.log(q) + (1.0 - y) * math.log(1.0 - q))
        else:
            for pi, ti in zip(p, t):
                if ti == 0.0:
                    continue
                q = min(1.0 - eps, max(eps, pi))
                total += -ti * math.log(q)
    return total / n
