"""Collate Example → tensors accepted by VaneModel / one_train_step.

Tokenizer for the text smoke
----------------------------
No 1B tokenizer is downloaded. This module uses a **whitespace + hash**
tokenizer: split on whitespace, map each token to ``sha256(token) % vocab_size``.
Documented here so smoke runs stay hermetic. Pad within the batch. Choice
temperature and C(q) must use the TRUE option count ``k`` (via ``choice_mask``),
not the padded width. Score uses the true ``L`` (``score_n_levels``), not
phantom padded levels.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from typing import Any, Mapping

import torch
from torch import Tensor

from vane.data.schema import Example, QuestionType, StateValue


def state_to_text(state: StateValue) -> str:
    if isinstance(state, str):
        return state
    if isinstance(state, Mapping):
        return json.dumps(dict(state), ensure_ascii=False, sort_keys=True, default=str)
    if isinstance(state, (list, tuple)):
        return "\n".join(str(x) for x in state)
    return str(state)


def hash_tokenize(
    text: str,
    *,
    vocab_size: int,
    max_length: int = 128,
) -> list[int]:
    """Whitespace + hash tokenizer (smoke only; no 1B tokenizer download)."""
    tokens = text.split() or [text[:64] or " "]
    ids: list[int] = []
    for tok in tokens[:max_length]:
        digest = hashlib.sha256(tok.encode("utf-8")).digest()
        ids.append(int.from_bytes(digest[:4], "big") % vocab_size)
    if not ids:
        ids = [0]
    return ids


def _option_vector(name: str, d_model: int) -> Tensor:
    digest = hashlib.sha256(name.encode("utf-8")).digest()
    vals = [((digest[i % len(digest)] / 255.0) * 2.0 - 1.0) * 0.1 for i in range(d_model)]
    return torch.tensor(vals, dtype=torch.float32)


def collate_examples(
    examples: Sequence[Example],
    *,
    d_model: int,
    vocab_size: int,
    max_length: int = 128,
    pad_value: int = 0,
) -> dict[str, Any]:
    """Pad a list of Examples into a training batch dict.

    True ``k`` / ``L`` are preserved via ``choice_mask`` and ``score_n_levels``.
    """
    if not examples:
        raise ValueError("collate_examples requires at least one example")

    token_rows = [
        hash_tokenize(state_to_text(ex.state), vocab_size=vocab_size, max_length=max_length)
        for ex in examples
    ]
    max_n = max(len(r) for r in token_rows)
    B = len(examples)
    input_ids = torch.full((B, max_n), pad_value, dtype=torch.long)
    attention_mask = torch.zeros(B, max_n, dtype=torch.bool)
    for i, row in enumerate(token_rows):
        n = len(row)
        input_ids[i, :n] = torch.tensor(row, dtype=torch.long)
        attention_mask[i, :n] = True

    batch: dict[str, Any] = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "corpora": [ex.corpus for ex in examples],
    }

    # --- Noul ---
    noul_targets: list[list[float]] = []
    for ex in examples:
        vals = [
            float(q.target)
            for q in ex.questions.values()
            if q.type is QuestionType.NOUL and isinstance(q.target, (int, float))
        ]
        noul_targets.append(vals)
    max_qn = max((len(v) for v in noul_targets), default=0)
    if max_qn > 0:
        noul_t = torch.zeros(B, max_qn, dtype=torch.float32)
        for i, vals in enumerate(noul_targets):
            for j, v in enumerate(vals):
                noul_t[i, j] = v
        batch["noul_count"] = max_qn
        batch["noul_target"] = noul_t

    # --- Choice ---
    choice_opts: list[list[tuple[str, ...]]] = []
    choice_tgts: list[list[dict[str, float]]] = []
    for ex in examples:
        opts_list: list[tuple[str, ...]] = []
        tgt_list: list[dict[str, float]] = []
        for q in ex.questions.values():
            if q.type is QuestionType.CHOICE and q.options is not None:
                opts_list.append(q.options)
                assert isinstance(q.target, Mapping)
                tgt_list.append(dict(q.target))
        choice_opts.append(opts_list)
        choice_tgts.append(tgt_list)
    max_qc = max((len(v) for v in choice_opts), default=0)
    max_k = 0
    for opts_list in choice_opts:
        for opts in opts_list:
            max_k = max(max_k, len(opts))
    if max_qc > 0 and max_k >= 2:
        options_t = torch.zeros(B, max_qc, max_k, d_model)
        mask_t = torch.zeros(B, max_qc, max_k, dtype=torch.bool)
        target_t = torch.zeros(B, max_qc, max_k)
        true_k_t = torch.zeros(B, max_qc, dtype=torch.long)
        for i, (opts_list, tgt_list) in enumerate(zip(choice_opts, choice_tgts)):
            for j, (opts, tgt) in enumerate(zip(opts_list, tgt_list)):
                k = len(opts)
                true_k_t[i, j] = k
                for t, name in enumerate(opts):
                    options_t[i, j, t] = _option_vector(name, d_model)
                    mask_t[i, j, t] = True
                    target_t[i, j, t] = float(tgt.get(name, 0.0))
        batch["choice_options"] = options_t
        batch["choice_mask"] = mask_t
        batch["choice_target"] = target_t
        batch["choice_true_k"] = true_k_t

    # --- Score ---
    score_levels: list[list[tuple[str, ...]]] = []
    score_tgts: list[list[tuple[float, ...]]] = []
    for ex in examples:
        lv_list: list[tuple[str, ...]] = []
        tg_list: list[tuple[float, ...]] = []
        for q in ex.questions.values():
            if q.type is QuestionType.SCORE and q.levels is not None:
                lv_list.append(q.levels)
                assert isinstance(q.target, Sequence)
                tg_list.append(tuple(float(x) for x in q.target))
        score_levels.append(lv_list)
        score_tgts.append(tg_list)
    max_qs = max((len(v) for v in score_levels), default=0)
    max_l = 0
    for lv_list in score_levels:
        for lv in lv_list:
            max_l = max(max_l, len(lv))
    if max_qs > 0 and max_l >= 2:
        # One L for the batch tensor: use max true L (not a phantom width
        # beyond any example's levels). Temperature / C(q) use this true max
        # when all rows share L; per-row true L stored in score_true_l.
        target_t = torch.zeros(B, max_qs, max_l)
        true_l_t = torch.zeros(B, max_qs, dtype=torch.long)
        for i, (lv_list, tg_list) in enumerate(zip(score_levels, score_tgts)):
            for j, (lv, tg) in enumerate(zip(lv_list, tg_list)):
                L = len(lv)
                true_l_t[i, j] = L
                for t in range(L):
                    target_t[i, j, t] = tg[t]
        batch["score_count"] = max_qs
        batch["score_n_levels"] = max_l  # true max L in batch — not padded phantom
        batch["score_target"] = target_t
        batch["score_true_l"] = true_l_t

    return batch
