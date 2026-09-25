"""Fit a scalar temperature on the calibration split, after training.

The map is softmax(log q / T). It is fit on probabilities the training
batches never saw. a and b of T(type, k) stay at their initialized values
until a later fit replaces them; this scalar is what the checkpoint records.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
from torch import Tensor


_GRID = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0)


def _nll(probs: Tensor, target: Tensor, temperature: float) -> float:
    log_q = torch.log(probs.clamp(min=1e-8))
    scaled = torch.softmax(log_q / temperature, dim=-1)
    return float(-(target * torch.log(scaled.clamp(min=1e-8))).sum(dim=-1).mean())


def fit_probability_temperature(
    probs: Sequence[Sequence[float]] | Tensor,
    target: Sequence[Sequence[float]] | Tensor,
) -> float:
    """Return the grid temperature with the lowest negative log likelihood."""
    q = torch.as_tensor(probs, dtype=torch.float32)
    pi = torch.as_tensor(target, dtype=torch.float32)
    if q.ndim == 1:
        q = torch.stack([q, 1.0 - q], dim=-1)
        pi = torch.stack([pi, 1.0 - pi], dim=-1)
    best_t = 1.0
    best = math.inf
    for temperature in _GRID:
        loss = _nll(q, pi, temperature)
        if loss < best:
            best = loss
            best_t = float(temperature)
    return best_t


def fit_cardinality_temperature(
    groups: dict[str, tuple[Tensor, Tensor]],
) -> dict[str, float]:
    """Fit one scalar T per primitive on calibration probabilities.

    ``groups`` maps ``noul`` / ``choice`` / ``score`` to ``(probs, target)``.
    Empty groups are omitted.
    """
    fitted: dict[str, float] = {}
    for name, (probs, target) in groups.items():
        if probs.numel() == 0:
            continue
        if name == "noul":
            fitted[name] = fit_probability_temperature(
                probs.reshape(-1), target.reshape(-1)
            )
        else:
            fitted[name] = fit_probability_temperature(
                probs.reshape(-1, probs.size(-1)),
                target.reshape(-1, target.size(-1)),
            )
    return fitted


def fit_temperature_parameters(
    model: torch.nn.Module,
    batch: dict,
    *,
    steps: int = 30,
    lr: float = 0.05,
) -> dict[str, float]:
    """Fit a and b of T(type, k) = softplus(a + b log k) on one calibration batch.

    Every other parameter is frozen for these steps, then restored.
    The loss is log loss on the simplex the head returns, so the temperature
    parameters are the only ones that move.
    """
    from vane.train.losses import choice_loss, score_loss

    names = ("temp_a_choice", "temp_b_choice", "temp_a_score", "temp_b_score")
    trainable: list[Tensor] = []
    for module in (getattr(model, "heads", None), getattr(model, "early_heads", None)):
        if module is None:
            continue
        for name in names:
            param = getattr(module, name, None)
            if isinstance(param, torch.nn.Parameter):
                trainable.append(param)
    if not trainable:
        return {}
    frozen: list[Tensor] = []
    trainable_ids = {id(p) for p in trainable}
    for param in model.parameters():
        if id(param) not in trainable_ids and param.requires_grad:
            param.requires_grad_(False)
            frozen.append(param)
    opt = torch.optim.Adam(trainable, lr=lr)
    was_training = model.training
    model.train()
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        out = model(
            batch["input_ids"],
            attention_mask=batch.get("attention_mask"),
            noul_queries=batch.get("noul_queries"),
            noul_count=batch.get("noul_count"),
            choice_options=batch.get("choice_options"),
            choice_mask=batch.get("choice_mask"),
            score_queries=batch.get("score_queries"),
            score_count=batch.get("score_count"),
            score_n_levels=batch.get("score_n_levels"),
            inputs_embeds=batch.get("inputs_embeds"),
            force_full=True,
            use_early_exit=False,
        )
        loss = torch.zeros((), device=batch["input_ids"].device)
        if out.noul is not None and "noul_target" in batch:
            q = out.noul.clamp(1e-6, 1 - 1e-6)
            pi = batch["noul_target"]
            loss = loss + (-(pi * torch.log(q) + (1 - pi) * torch.log(1 - q))).mean()
        if out.choice is not None and "choice_target" in batch:
            loss = loss + choice_loss(out.choice.reshape(-1, out.choice.size(-1)), batch["choice_target"].reshape(-1, batch["choice_target"].size(-1)))
        if out.score is not None and "score_target" in batch:
            loss = loss + score_loss(
                out.score.reshape(-1, out.score.size(-1)),
                batch["score_target"].reshape(-1, batch["score_target"].size(-1)),
                alpha=None,
            )
        if loss.requires_grad:
            loss.backward()
            opt.step()
    if was_training:
        model.train()
    else:
        model.eval()
    for param in frozen:
        param.requires_grad_(True)
    heads = model.heads
    return {
        "a_choice": float(heads.temp_a_choice.detach()),
        "b_choice": float(heads.temp_b_choice.detach()),
        "a_score": float(heads.temp_a_score.detach()),
        "b_score": float(heads.temp_b_score.detach()),
    }


def fit_early_exit_tau(*_args: object, **_kwargs: object) -> None:
    """τ is still chosen on a calibration ECE match. Not fit in this version."""
    raise NotImplementedError(
        "fit_early_exit_tau needs paired early-exit and full-stack answers "
        "on the calibration split."
    )
