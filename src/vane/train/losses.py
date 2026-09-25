"""Proper scoring losses for Vane primitives."""

from __future__ import annotations

import torch
from torch import Tensor


def noul_loss(q: Tensor, pi: Tensor) -> Tensor:
    """Log loss + Brier for binary noul.

    L = -(π log q + (1-π) log(1-q)) + (q - π)^2
    """
    q = q.clamp(1e-7, 1.0 - 1e-7)
    pi = pi.to(dtype=q.dtype)
    log_loss = -(pi * torch.log(q) + (1.0 - pi) * torch.log(1.0 - q))
    brier = (q - pi) ** 2
    return (log_loss + brier).mean()


def choice_loss(q: Tensor, pi: Tensor) -> Tensor:
    """Log loss + Brier on a simplex.

    L = -Σ π_j log q_j + ||q - π||_2^2
    """
    q = q.clamp(min=1e-7)
    pi = pi.to(dtype=q.dtype)
    log_loss = -(pi * torch.log(q)).sum(dim=-1)
    brier = ((q - pi) ** 2).sum(dim=-1)
    return (log_loss + brier).mean()


def _is_unimodal(pi: Tensor, tol: float = 1e-6) -> Tensor:
    """Return bool mask [...] whether each simplex row is unimodal."""
    # Flatten to [N, L]
    orig_shape = pi.shape[:-1]
    L = pi.size(-1)
    flat = pi.reshape(-1, L)
    if L <= 2:
        return torch.ones(flat.size(0), dtype=torch.bool, device=pi.device).view(orig_shape)

    diffs = flat[:, 1:] - flat[:, :-1]  # [N, L-1]
    # Sign: +1 increasing, -1 decreasing, 0 flat
    signs = torch.sign(diffs)
    # Treat near-zero as flat
    signs = torch.where(diffs.abs() < tol, torch.zeros_like(signs), signs)

    # Walk: allow zeros; at most one transition from non-negative to non-positive
    # (up then down). Also allow pure monotone.
    ok = []
    for row in signs:
        phase = 0  # 0=start/up, 1=down
        good = True
        for s in row.tolist():
            if s == 0:
                continue
            if phase == 0:
                if s < 0:
                    phase = 1
            else:  # phase == 1 (descending)
                if s > 0:
                    good = False
                    break
        ok.append(good)
    return torch.tensor(ok, device=pi.device, dtype=torch.bool).view(orig_shape)


def ranked_probability_score(q: Tensor, pi: Tensor) -> Tensor:
    """RPS(q, π) = 1/(L-1) Σ_{i=0}^{L-2} (F_q(i) - F_π(i))^2."""
    L = q.size(-1)
    if L < 2:
        return torch.zeros(q.shape[:-1], device=q.device, dtype=q.dtype)
    Fq = q.cumsum(dim=-1)[..., :-1]
    Fp = pi.cumsum(dim=-1)[..., :-1]
    return ((Fq - Fp) ** 2).sum(dim=-1) / (L - 1)


def binary_entropy(alpha: Tensor, eps: float = 1e-7) -> Tensor:
    """H(α) = -α log α - (1-α) log(1-α)."""
    a = alpha.clamp(eps, 1.0 - eps)
    return -(a * torch.log(a) + (1.0 - a) * torch.log(1.0 - a))


def score_loss(
    q: Tensor,
    pi: Tensor,
    alpha: Tensor | None = None,
    beta: float = 1.0,
    gamma: float = 0.01,
) -> Tensor:
    """Score loss: log + Brier + β·1_unimodal·RPS + γ H(α).

    RPS applies only when π is unimodal.
    """
    q = q.clamp(min=1e-7)
    pi = pi.to(dtype=q.dtype)
    log_loss = -(pi * torch.log(q)).sum(dim=-1)
    brier = ((q - pi) ** 2).sum(dim=-1)
    loss = log_loss + brier

    uni = _is_unimodal(pi)
    rps = ranked_probability_score(q, pi)
    loss = loss + beta * uni.to(dtype=q.dtype) * rps

    if alpha is not None:
        loss = loss + gamma * binary_entropy(alpha)

    return loss.mean()


def _pava_isotonic(y: Tensor, w: Tensor | None = None) -> Tensor:
    """Pool-adjacent-violators isotonic regression (non-decreasing) on 1D ``y``.

    Assumes ``y`` is already ordered by the predictor.
    """
    n = y.numel()
    if n == 0:
        return y
    values = y.detach().float().tolist()
    weights = w.detach().float().tolist() if w is not None else [1.0] * n

    # Blocks: (sum_wy, sum_w, start, end)
    blocks: list[list[float]] = []
    for i in range(n):
        blocks.append([values[i] * weights[i], weights[i], float(i), float(i)])
        while len(blocks) >= 2:
            a, b = blocks[-2], blocks[-1]
            mean_a = a[0] / a[1]
            mean_b = b[0] / b[1]
            if mean_a <= mean_b + 1e-12:
                break
            # merge
            merged = [a[0] + b[0], a[1] + b[1], a[2], b[3]]
            blocks[-2:] = [merged]

    out = torch.empty(n, dtype=torch.float32, device=y.device)
    for s_wy, s_w, start, end in blocks:
        mean = s_wy / s_w
        out[int(start) : int(end) + 1] = mean
    return out.to(dtype=y.dtype)


def isotonic_residual(
    q: Tensor,
    y: Tensor,
    *,
    min_batch: int = 64,
    allow_small_batch: bool = False,
) -> Tensor:
    """Positive part of batch-mean Brier gap after isotonic recalibration.

    1. Fit isotonic map from predicted probability → outcome (PAVA on sorted q).
    2. Detach the mapped predictions.
    3. residual = relu( mean((q-y)^2) - mean((q_iso-y)^2) ).

    The positive part wraps the **batch mean**, not each row.
    Prefer batch size ≥ 64; set ``allow_small_batch=True`` for smoke tests.
    """
    q_flat = q.detach().reshape(-1).float()
    y_flat = y.detach().reshape(-1).float()
    n = q_flat.numel()
    if n < min_batch and not allow_small_batch:
        raise ValueError(
            f"isotonic_residual expects batch >= {min_batch}, got {n}; "
            "pass allow_small_batch=True for smoke tests"
        )
    if n == 0:
        return torch.zeros((), device=q.device, dtype=q.dtype)

    order = torch.argsort(q_flat)
    q_sorted = q_flat[order]
    y_sorted = y_flat[order]
    y_iso_sorted = _pava_isotonic(y_sorted)
    # Map back
    q_iso = torch.empty_like(q_flat)
    q_iso[order] = y_iso_sorted
    q_iso = q_iso.clamp(0.0, 1.0)

    brier_raw = ((q_flat - y_flat) ** 2).mean()
    brier_iso = ((q_iso - y_flat) ** 2).mean()
    gap = brier_raw - brier_iso
    # Positive part of batch-mean gap; detached map ⇒ no grad through PAVA
    residual = torch.relu(gap)
    # Re-attach to q graph so the residual can train the head (gap uses detached q_iso)
    # Training term: encourage lower Brier; use live q for raw Brier vs detached iso target
    live = q.reshape(-1).float()
    live_brier = ((live - y_flat) ** 2).mean()
    # residual value is relu(detached gap); gradient flows through live_brier only when gap>0
    return torch.relu(live_brier - brier_iso.detach())
