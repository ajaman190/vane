"""Request/response helpers for the serve surface (stdlib typing only)."""

from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

CONTEXT_LIMIT = 32768


def estimate_tokens(state: Any) -> int:
    """Rough token estimate used only to refuse oversized inputs.

    No real tokenizer ships with the package yet. We count whitespace-
    separated pieces of the textual payload and add a fixed budget per
    image entry. Requests past :data:`CONTEXT_LIMIT` are refused, not cropped.
    """
    images = 0
    text_parts: list[str] = []

    if isinstance(state, Mapping):
        payload = dict(state)
        imgs = payload.pop("images", None)
        if isinstance(imgs, (list, tuple)):
            images = len(imgs)
        elif imgs is not None:
            images = 1
        text_parts.append(_stringify(payload))
    elif isinstance(state, (list, tuple)):
        # list[str] or mixed; images only if nested mappings carry them
        for item in state:
            if isinstance(item, Mapping) and "images" in item:
                imgs = item.get("images")
                if isinstance(imgs, (list, tuple)):
                    images += len(imgs)
                elif imgs is not None:
                    images += 1
                rest = {k: v for k, v in item.items() if k != "images"}
                text_parts.append(_stringify(rest))
            else:
                text_parts.append(_stringify(item))
    else:
        text_parts.append(_stringify(state))

    text = " ".join(p for p in text_parts if p)
    # Conservative: whitespace tokens, floor 1 if non-empty
    word_tokens = len(text.split()) if text.strip() else 0
    # Also charge roughly 1 token per 4 chars to catch long unspaced pads
    char_tokens = (len(text) + 3) // 4 if text else 0
    text_tokens = max(word_tokens, char_tokens)
    # Fixed budget per image (patch tokens); refuse path only
    image_tokens = images * 256
    return text_tokens + image_tokens


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    except TypeError:
        return str(value)


def validate_questions(questions: Any) -> dict[str, Any]:
    if not isinstance(questions, Mapping) or len(questions) == 0:
        raise ValueError("questions must be a non-empty map")
    out: dict[str, Any] = {}
    for key, q in questions.items():
        if not isinstance(q, Mapping):
            raise ValueError(f"question {key!r} must be an object")
        qtype = q.get("type")
        if qtype not in ("noul", "choice", "score"):
            raise ValueError(
                f"question {key!r} type must be noul|choice|score, got {qtype!r}"
            )
        if qtype == "choice":
            criteria = q.get("criteria")
            if not isinstance(criteria, Mapping) or len(criteria) < 2:
                raise ValueError(
                    f"choice {key!r} requires criteria map with 2..255 options"
                )
            if len(criteria) > 255:
                raise ValueError(f"choice {key!r} has more than 255 options")
        if qtype == "score":
            criteria = q.get("criteria")
            if not isinstance(criteria, Sequence) or isinstance(criteria, (str, bytes)):
                raise ValueError(f"score {key!r} requires ordered criteria list")
            if not (2 <= len(criteria) <= 10):
                raise ValueError(f"score {key!r} needs 2..10 levels, got {len(criteria)}")
        out[str(key)] = dict(q)
    return out
