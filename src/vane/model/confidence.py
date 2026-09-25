"""Confidence C(q) and cardinality temperature helpers."""

from __future__ import annotations

import torch
from torch import Tensor


def confidence(
    q: Tensor,
    eps: float = 1e-8,
    *,
    k: int | Tensor | None = None,
    mask: Tensor | None = None,
) -> Tensor:
    """C(q) = (k * max(q) - 1) / (k - 1) for simplex q with k >= 2.

    Args:
        q: probabilities ``[..., K]`` (last dim is the simplex, possibly padded).
        k: True option/level count. Prefer this (or ``mask``) over the padded
           width so C(q) does not treat pad slots as support.
        mask: ``[..., K]`` bool; True = real option. ``k`` is derived as
           ``mask.sum(-1)`` when not passed explicitly.

    Returns:
        Confidence ``[...]`` in ``[0, 1]``.
    """
    if mask is not None:
        true_k = mask.sum(dim=-1).to(dtype=q.dtype).clamp(min=2.0)
        q_eff = q.masked_fill(~mask, -1.0e4)
        max_q = q_eff.max(dim=-1).values
        return ((true_k * max_q - 1.0) / (true_k - 1.0)).clamp(0.0, 1.0)

    if k is None:
        k_val: int | Tensor = q.size(-1)
    else:
        k_val = k

    if isinstance(k_val, int):
        if k_val < 2:
            raise ValueError(f"confidence requires k >= 2, got k={k_val}")
        max_q = q[..., :k_val].max(dim=-1).values
        return ((k_val * max_q - 1.0) / (k_val - 1.0)).clamp(0.0, 1.0)

    # Tensor k (per-row true cardinality)
    k_t = k_val.to(dtype=q.dtype).clamp(min=2.0)
    # Build a mask from varying k when possible (same padded width).
    width = q.size(-1)
    idx = torch.arange(width, device=q.device).expand(*q.shape)
    row_mask = idx < k_t.unsqueeze(-1)
    q_eff = q.masked_fill(~row_mask, -1.0e4)
    max_q = q_eff.max(dim=-1).values
    return ((k_t * max_q - 1.0) / (k_t - 1.0)).clamp(0.0, 1.0)


def cardinality_temperature(
    a: Tensor | float,
    b: Tensor | float,
    k: int | Tensor,
) -> Tensor:
    """T(type, k) = softplus(a + b log k)."""
    if isinstance(k, int):
        log_k = torch.log(torch.tensor(float(k), dtype=torch.float32))
        if isinstance(a, Tensor):
            log_k = log_k.to(device=a.device, dtype=a.dtype)
    else:
        log_k = torch.log(k.float().clamp(min=1.0))
        if isinstance(a, Tensor):
            log_k = log_k.to(device=a.device, dtype=a.dtype if a.dtype.is_floating_point else torch.float32)
    a_t = a if isinstance(a, Tensor) else torch.tensor(float(a))
    b_t = b if isinstance(b, Tensor) else torch.tensor(float(b), device=a_t.device, dtype=a_t.dtype)
    if isinstance(a, Tensor):
        a_t = a
        b_t = b if isinstance(b, Tensor) else torch.tensor(float(b), device=a.device, dtype=a.dtype)
    return torch.nn.functional.softplus(a_t + b_t * log_k)
