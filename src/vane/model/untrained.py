"""Untrained local forward used when no checkpoint is loaded."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

import torch

from vane.eval.metrics import confidence as confidence_py
from vane.model.config import DEV_CONFIG
from vane.model.vane import VaneModel

# One dev-sized network, created on first request. Random weights, fixed seed.
_DEV: VaneModel | None = None


def _get_dev() -> VaneModel:
    global _DEV
    if _DEV is None:
        torch.manual_seed(0)
        _DEV = VaneModel(DEV_CONFIG)
        _DEV.eval()
    return _DEV


def _stringify(state: Any) -> str:
    if isinstance(state, str):
        return state
    if isinstance(state, Mapping):
        payload = {k: v for k, v in state.items() if k != "images"}
        return json.dumps(payload, sort_keys=True, default=str)
    try:
        return json.dumps(state, sort_keys=True, default=str)
    except TypeError:
        return repr(state)


def _hash_tokens(text: str, vocab_size: int, max_len: int = 32) -> torch.Tensor:
    """Deterministic pseudo-tokens from text (no external tokenizer)."""
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    ids: list[int] = []
    # Expand digest stream
    blob = digest
    while len(ids) < max_len:
        for i in range(0, len(blob) - 1, 2):
            ids.append(int.from_bytes(blob[i : i + 2], "big") % vocab_size)
            if len(ids) >= max_len:
                break
        blob = hashlib.sha256(blob).digest()
    return torch.tensor([ids], dtype=torch.long)


def _option_vectors(names: list[str], d_model: int) -> torch.Tensor:
    """[1, K, D] vectors hashed from option names."""
    rows: list[torch.Tensor] = []
    for name in names:
        h = hashlib.sha256(name.encode("utf-8")).digest()
        vals = [
            ((h[i % len(h)] / 255.0) * 2.0 - 1.0) * 0.1 for i in range(d_model)
        ]
        rows.append(torch.tensor(vals, dtype=torch.float32))
    return torch.stack(rows, dim=0).unsqueeze(0)  # [1, K, D]


@torch.no_grad()
def forward_systemone(
    *,
    state: Any,
    questions: Mapping[str, Mapping[str, Any]],
    checkpoint: str = "vane-untrained-local-text",
) -> dict[str, Any]:
    """Run the dev-sized network once and map outputs to noul, choice, and score."""
    del checkpoint  # untrained smoke ignores checkpoint id for weights
    model = _get_dev()
    cfg = model.config
    d = cfg.d_model
    text = _stringify(state)
    input_ids = _hash_tokens(text, cfg.vocab_size, max_len=min(32, cfg.max_context))

    noul_ids: list[str] = []
    choice_ids: list[str] = []
    choice_names: list[list[str]] = []
    score_ids: list[str] = []
    score_levels: list[list[Any]] = []

    for qid, q in questions.items():
        qtype = q["type"]
        if qtype == "noul":
            noul_ids.append(qid)
        elif qtype == "choice":
            choice_ids.append(qid)
            choice_names.append(list(q["criteria"].keys()))
        elif qtype == "score":
            score_ids.append(qid)
            score_levels.append(list(q["criteria"]))
        else:
            raise ValueError(f"unknown question type {qtype!r}")

    kwargs: dict[str, Any] = {"force_full": True}
    if noul_ids:
        kwargs["noul_count"] = len(noul_ids)

    if choice_ids:
        # Pad all choice questions to the same K for a single batched tensor.
        max_k = max(len(n) for n in choice_names)
        opts = torch.zeros(1, len(choice_ids), max_k, d)
        mask = torch.zeros(1, len(choice_ids), max_k, dtype=torch.bool)
        for i, names in enumerate(choice_names):
            vec = _option_vectors(names, d)  # [1, K, D]
            k = len(names)
            opts[0, i, :k] = vec[0]
            mask[0, i, :k] = True
        kwargs["choice_options"] = opts
        kwargs["choice_mask"] = mask

    if score_ids:
        # One L for the batch: use max levels; pad criteria conceptually.
        max_l = max(len(lv) for lv in score_levels)
        kwargs["score_count"] = len(score_ids)
        kwargs["score_n_levels"] = max_l

    out = model(input_ids, **kwargs)
    answers: dict[str, Any] = {}

    if noul_ids and out.noul is not None:
        for i, qid in enumerate(noul_ids):
            answers[qid] = {"type": "noul", "noul": float(out.noul[0, i].item())}

    if choice_ids and out.choice is not None:
        for i, qid in enumerate(choice_ids):
            names = choice_names[i]
            k = len(names)
            probs = out.choice[0, i, :k].tolist()
            # Renormalize in case of padding mask zeros
            s = sum(probs) or 1.0
            probs = [p / s for p in probs]
            probabilities = {n: float(p) for n, p in zip(names, probs, strict=True)}
            choice_name = max(probabilities, key=probabilities.get)  # type: ignore[arg-type]
            answers[qid] = {
                "type": "choice",
                "choice": choice_name,
                "probabilities": probabilities,
                "confidence": float(confidence_py(probs)),
            }

    if score_ids and out.score is not None:
        for i, qid in enumerate(score_ids):
            levels = score_levels[i]
            L = len(levels)
            probs = out.score[0, i, :L].tolist()
            s = sum(probs) or 1.0
            probs = [p / s for p in probs]
            expectation = sum(j * probs[j] for j in range(L))
            answers[qid] = {
                "type": "score",
                "score": float(expectation),
                "legend": list(levels),
                "probabilities": {str(j): float(probs[j]) for j in range(L)},
                "confidence": float(confidence_py(probs)),
            }

    return answers
