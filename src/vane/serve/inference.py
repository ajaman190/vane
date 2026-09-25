"""Answer generation: prefer tiny VaneModel smoke path; always have a stub."""

from __future__ import annotations

import hashlib
import math
from typing import Any, Mapping, Sequence

from vane.serve.confidence import confidence

# Softmax temperature for the deterministic stub (keeps mass interior).
_STUB_TEMP = 1.0


def _softmax(logits: Sequence[float], temperature: float = _STUB_TEMP) -> list[float]:
    t = max(float(temperature), 1e-8)
    shifted = [float(x) / t for x in logits]
    m = max(shifted)
    exps = [math.exp(x - m) for x in shifted]
    s = sum(exps) or 1.0
    return [e / s for e in exps]


def _stable_logits(seed: str, k: int) -> list[float]:
    """Deterministic pseudo-logits from a seed string (no RNG / no torch)."""
    out: list[float] = []
    for i in range(k):
        digest = hashlib.sha256(f"{seed}:{i}".encode("utf-8")).digest()
        # Map 8 bytes to roughly N(0,1)-ish float
        n = int.from_bytes(digest[:8], "big") / float(2**64 - 1)
        out.append((n - 0.5) * 4.0)
    return out


def _answer_noul(seed: str) -> dict[str, Any]:
    logit = _stable_logits(seed, 1)[0]
    q = 1.0 / (1.0 + math.exp(-logit))
    return {"type": "noul", "noul": float(q)}


def _answer_choice(seed: str, criteria: Mapping[str, Any]) -> dict[str, Any]:
    names = list(criteria.keys())
    k = len(names)
    probs = _softmax(_stable_logits(seed, k))
    probabilities = {name: float(p) for name, p in zip(names, probs, strict=True)}
    choice_name = max(probabilities, key=probabilities.get)  # type: ignore[arg-type]
    values = list(probabilities.values())
    return {
        "type": "choice",
        "choice": choice_name,
        "probabilities": probabilities,
        "confidence": float(confidence(values)),
    }


def _answer_score(seed: str, criteria: Sequence[Any]) -> dict[str, Any]:
    levels = list(criteria)
    L = len(levels)
    probs = _softmax(_stable_logits(seed, L))
    # architecture.md: legend is an ordered array (lowest first), not a map.
    legend = [levels[i] for i in range(L)]
    probabilities = {str(i): float(probs[i]) for i in range(L)}
    expectation = sum(i * probs[i] for i in range(L))
    return {
        "type": "score",
        "score": float(expectation),
        "legend": legend,
        "probabilities": probabilities,
        "confidence": float(confidence(probs)),
    }


def _state_seed(state: Any) -> str:
    """Full-state hash so opposite meanings never share a noul seed.

    Uses the entire ``repr(state)`` (not a truncated prefix, not yes/no
    keywords alone) so affirmative vs negative payloads diverge.
    """
    digest = hashlib.sha256(
        repr(state).encode("utf-8", errors="surrogateescape")
    ).hexdigest()
    return digest


def stub_answers(state: Any, questions: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Deterministic schema-valid answers without torch or weights.

    Documented behaviour when no checkpoint is loaded: still returns noul /
    choice / score shapes with ``C(q)`` on choice and score. Noul depends on
    full state content via :func:`_state_seed`.
    """
    state_key = _state_seed(state)
    answers: dict[str, Any] = {}
    for qid, q in questions.items():
        seed = f"{state_key}|{qid}|{q.get('type')}"
        qtype = q["type"]
        if qtype == "noul":
            answers[qid] = _answer_noul(seed)
        elif qtype == "choice":
            answers[qid] = _answer_choice(seed, q["criteria"])
        elif qtype == "score":
            answers[qid] = _answer_score(seed, q["criteria"])
        else:
            raise ValueError(f"unknown question type {qtype!r}")
    return answers


def _try_model_answers(
    state: Any,
    questions: Mapping[str, Mapping[str, Any]],
    checkpoint: str,
) -> dict[str, Any] | None:
    """If torch is installed, run the untrained dev network; else None.

    Failures fall through to the deterministic stub so serve never crashes
    without weights.
    """
    try:
        import torch  # noqa: F401
    except ImportError:
        return None

    try:
        from vane.model.untrained import forward_systemone
    except ImportError:
        return None

    try:
        return forward_systemone(
            state=state, questions=questions, checkpoint=checkpoint
        )
    except Exception:
        return None


def generate_answers(
    state: Any,
    questions: Mapping[str, Mapping[str, Any]],
    checkpoint: str,
) -> dict[str, Any]:
    """Produce answers. Untrained dev network if torch is installed; otherwise a deterministic stub."""
    modeled = _try_model_answers(state, questions, checkpoint)
    if modeled is not None:
        # Ensure confidence fields exist on choice/score even if the model omitted them.
        return _ensure_confidence(modeled)
    return stub_answers(state, questions)


def _ensure_confidence(answers: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for qid, ans in answers.items():
        a = dict(ans)
        if a.get("type") == "choice" and "confidence" not in a:
            probs = a.get("probabilities") or {}
            a["confidence"] = float(confidence(list(probs.values())))
        elif a.get("type") == "score" and "confidence" not in a:
            probs = a.get("probabilities") or {}
            # probabilities may be keyed by str indices
            values = [float(probs[k]) for k in sorted(probs, key=lambda x: int(x))]
            a["confidence"] = float(confidence(values))
        out[qid] = a
    return out
