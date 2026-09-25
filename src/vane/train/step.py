"""Single training step for smoke and unit tests."""

from __future__ import annotations

import random
from typing import Any

import torch
from torch import Tensor
from torch.optim import Optimizer

from vane.model import VaneModel
from vane.train.device import batch_to_device, resolve_device
from vane.train.losses import choice_loss, isotonic_residual, noul_loss, score_loss


def one_train_step(
    model: VaneModel,
    batch: dict[str, Any],
    optim: Optimizer,
    *,
    isotonic: bool = False,
    isotonic_allow_small_batch: bool = True,
    force_full: bool = True,
    force_early: bool | None = None,
    early_exit_train_prob: float = 0.0,
    device: str | torch.device | None = None,
    rng: random.Random | None = None,
    gradient_clip: float = 1.0,
) -> dict[str, float]:
    """Run one forward / backward / optim step.

    Expected ``batch`` keys (all optional except ``input_ids``):

    - ``input_ids``: ``[B, N]``
    - ``attention_mask``: ``[B, N]``
    - ``noul_queries`` / ``noul_count`` / ``noul_target`` (π in [0,1], ``[B, Qn]``)
    - ``choice_options`` ``[B, Qc, K, D]``, ``choice_mask``, ``choice_target`` simplex
    - ``score_queries`` / ``score_count`` / ``score_n_levels`` / ``score_target``

    Early-head training: with probability ``early_exit_train_prob`` (or when
    ``force_early=True``) the step stops after block 1 and applies the proper
    score loss to the early head. Serve-time τ is **not** the training coin.
    """
    resolved = resolve_device(device) if device is not None else (
        batch["input_ids"].device
        if isinstance(batch.get("input_ids"), Tensor)
        else resolve_device("auto")
    )
    model.to(resolved)
    batch = batch_to_device(batch, resolved)

    model.train()
    optim.zero_grad(set_to_none=True)

    coin_early = False
    if force_early is True:
        coin_early = True
    elif force_early is False:
        coin_early = False
    elif early_exit_train_prob > 0.0:
        r = rng or random
        coin_early = r.random() < float(early_exit_train_prob)

    input_ids: Tensor = batch["input_ids"]
    out = model(
        input_ids,
        attention_mask=batch.get("attention_mask"),
        noul_queries=batch.get("noul_queries"),
        noul_count=batch.get("noul_count"),
        choice_options=batch.get("choice_options"),
        choice_mask=batch.get("choice_mask"),
        score_queries=batch.get("score_queries"),
        score_count=batch.get("score_count"),
        score_n_levels=batch.get("score_n_levels"),
        vision_patches=batch.get("vision_patches"),
        inputs_embeds=batch.get("inputs_embeds"),
        force_early=coin_early,
        force_full=(not coin_early) and force_full,
        use_early_exit=False,  # serve τ is not the training coin
    )

    loss = torch.zeros((), device=input_ids.device)
    parts: dict[str, float] = {}

    if out.noul is not None and "noul_target" in batch:
        ln = noul_loss(out.noul, batch["noul_target"])
        loss = loss + ln
        parts["noul"] = float(ln.detach())

    if out.choice is not None and "choice_target" in batch:
        q = out.choice
        pi = batch["choice_target"]
        mask = batch.get("choice_mask")
        if q.dim() == 4:
            q = q.reshape(-1, q.size(-1))
            pi = pi.reshape(-1, pi.size(-1))
            if mask is not None:
                mask = mask.reshape(-1, mask.size(-1))
        elif q.dim() == 3:
            q = q.reshape(-1, q.size(-1))
            pi = pi.reshape(-1, pi.size(-1))
            if mask is not None:
                mask = mask.reshape(-1, mask.size(-1))
        # Loss on true options only when a pad mask is present.
        if mask is not None:
            q = q.masked_fill(~mask, 0.0)
            pi = pi.masked_fill(~mask, 0.0)
            # Renormalize remaining mass on real options for a proper simplex.
            q_sum = q.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            pi_sum = pi.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            q = q / q_sum
            pi = pi / pi_sum
        lc = choice_loss(q, pi)
        loss = loss + lc
        parts["choice"] = float(lc.detach())

    if out.score is not None and "score_target" in batch:
        q = out.score
        pi = batch["score_target"]
        alpha = out.alpha
        if q.dim() == 3:
            q = q.reshape(-1, q.size(-1))
            pi = pi.reshape(-1, pi.size(-1))
            if alpha is not None:
                alpha = alpha.reshape(-1)
        ls = score_loss(q, pi, alpha=alpha)
        loss = loss + ls
        parts["score"] = float(ls.detach())

    if isotonic and out.noul is not None and "noul_target" in batch:
        li = isotonic_residual(
            out.noul,
            batch["noul_target"],
            allow_small_batch=isotonic_allow_small_batch,
        )
        loss = loss + li
        parts["isotonic"] = float(li.detach())

    if not loss.requires_grad:
        if out.noul is not None:
            loss = out.noul.sum() * 0.0
        elif out.choice is not None:
            loss = out.choice.sum() * 0.0
        elif out.score is not None:
            loss = out.score.sum() * 0.0
        else:
            loss = next(model.parameters()).sum() * 0.0

    loss.backward()
    # MPS can produce rare non-finite grads; skip the step rather than poison weights.
    grads_ok = True
    for p in model.parameters():
        if p.grad is not None and not torch.isfinite(p.grad).all():
            grads_ok = False
            break
    if not grads_ok or not torch.isfinite(loss):
        optim.zero_grad(set_to_none=True)
        parts["loss"] = float("nan") if not torch.isfinite(loss) else float(loss.detach())
        parts["skipped_nonfinite"] = 1.0
        parts["early"] = 1.0 if coin_early else 0.0
        parts["blocks_run"] = float(out.blocks_run)
        return parts

    if gradient_clip and gradient_clip > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(gradient_clip))
    optim.step()

    parts["loss"] = float(loss.detach())
    parts["early"] = 1.0 if coin_early else 0.0
    parts["blocks_run"] = float(out.blocks_run)
    return parts
