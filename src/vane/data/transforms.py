"""Allowed transforms of a human item. No new target for the original questions."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, replace
from typing import Any, Mapping

from vane.data.schema import Example, QuestionSpec, QuestionType, StateValue


@dataclass(frozen=True, slots=True)
class TransformOptions:
    """Which allowed transforms to apply. All are no-ops when False."""

    paraphrase: bool = False
    shuffle_options: bool = False
    flip_state: bool = False
    inject_distractor: bool = False
    seed: int | None = None


def apply_transforms(
    example: Example,
    options: TransformOptions | None = None,
    *,
    paraphrase: bool = False,
    shuffle_options: bool = False,
    flip_state: bool = False,
    inject_distractor: bool = False,
    seed: int | None = None,
    rng: random.Random | None = None,
) -> Example:
    """Apply allowed human-item transforms. Original question targets are preserved.

    Allowed (docs/data.md):
      - paraphrase (stub: whitespace normalize / identity)
      - shuffle option order (choice/score support order; π follows)
      - flip state between raw text and nested JSON
      - inject a random distractor question (construction-certain target)

    No model-written labels. Transforms never invent a new π for existing questions.
    """
    if options is not None:
        paraphrase = options.paraphrase
        shuffle_options = options.shuffle_options
        flip_state = options.flip_state
        inject_distractor = options.inject_distractor
        if seed is None:
            seed = options.seed

    rng = rng or random.Random(seed)

    state = paraphrase_state(example.state) if paraphrase else example.state
    if flip_state:
        state = flip_state_representation(state)

    questions: dict[str, QuestionSpec] = dict(example.questions)
    if shuffle_options:
        questions = {
            qid: shuffle_question_options(q, rng) for qid, q in questions.items()
        }

    if inject_distractor:
        questions = inject_distractor_question(questions, rng)

    return replace(example, state=state, questions=questions)


def paraphrase_state(state: StateValue) -> StateValue:
    """Paraphrase stub: normalize text whitespace; identity for structured state."""
    if isinstance(state, str):
        return " ".join(state.split())
    if isinstance(state, Mapping):
        return {k: paraphrase_state(v) if isinstance(v, (str, Mapping, list)) else v
                for k, v in state.items()}  # type: ignore[misc]
    if isinstance(state, list):
        return [paraphrase_state(x) if isinstance(x, (str, Mapping, list)) else x
                for x in state]
    return state


def flip_state_representation(state: StateValue) -> StateValue:
    """Flip state between raw text and nested JSON (dict).

    - str of JSON object → parsed dict
    - dict → compact JSON string
    - list[str] → JSON string; JSON array string → list
    Other values are returned unchanged.
    """
    if isinstance(state, str):
        text = state.strip()
        if not text:
            return state
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            # Wrap plain text as a single-field JSON object.
            return {"text": state}
        if isinstance(parsed, dict):
            return parsed
        if isinstance(parsed, list):
            return parsed
        return {"value": parsed}

    if isinstance(state, Mapping):
        return json.dumps(dict(state), ensure_ascii=False, separators=(",", ":"))

    if isinstance(state, list):
        return json.dumps(list(state), ensure_ascii=False, separators=(",", ":"))

    return state


def shuffle_question_options(
    question: QuestionSpec, rng: random.Random
) -> QuestionSpec:
    """Permute choice option order (or score level order). Target π follows.

    Noul is unchanged. Descriptions move with option names.
    """
    if question.type is QuestionType.NOUL:
        return question

    if question.type is QuestionType.CHOICE:
        assert question.options is not None
        assert isinstance(question.target, Mapping)
        order = list(question.options)
        rng.shuffle(order)
        new_target = {name: float(question.target[name]) for name in order}
        new_desc = None
        if question.option_descriptions is not None:
            new_desc = {
                name: question.option_descriptions.get(name) for name in order
            }
        return replace(
            question,
            options=tuple(order),
            target=new_target,
            option_descriptions=new_desc,
        )

    if question.type is QuestionType.SCORE:
        assert question.levels is not None
        assert isinstance(question.target, (list, tuple))
        # Score levels are ordered (ordinal). Shuffling breaks ordinality, so
        # we only shuffle a *display* copy for choice-like menus is wrong.
        # Allowed transform "shuffle option order" applies to choice options.
        # For score, preserve level order and target (identity).
        return question

    return question


def inject_distractor_question(
    questions: Mapping[str, QuestionSpec],
    rng: random.Random,
) -> dict[str, QuestionSpec]:
    """Inject a construction-certain distractor noul. Existing targets untouched.

    The distractor asks whether a nonce token appears in the question map key;
    by construction the answer is yes (π=1). This adds no model-written label
    and does not alter original targets.
    """
    out = dict(questions)
    nonce = f"distractor_{rng.randrange(1_000_000_000)}"
    # Ensure unique key.
    key = nonce
    while key in out:
        key = f"distractor_{rng.randrange(1_000_000_000)}"
    out[key] = QuestionSpec(
        type=QuestionType.NOUL,
        target=1.0,
        text=f"Is question id {key} present in this request?",
    )
    return out


def targets_equal(
    a: Mapping[str, QuestionSpec],
    b: Mapping[str, QuestionSpec],
    *,
    ignore_keys: set[str] | None = None,
) -> bool:
    """Compare soft targets for shared question ids (ignoring distractors)."""
    ignore = ignore_keys or set()
    shared = (set(a) & set(b)) - ignore
    for qid in shared:
        if not _target_close(a[qid].target, b[qid].target):
            return False
    return True


def _target_close(x: Any, y: Any, tol: float = 1e-9) -> bool:
    if isinstance(x, (int, float)) and isinstance(y, (int, float)):
        return abs(float(x) - float(y)) <= tol
    if isinstance(x, Mapping) and isinstance(y, Mapping):
        if set(x) != set(y):
            return False
        return all(abs(float(x[k]) - float(y[k])) <= tol for k in x)
    if isinstance(x, (list, tuple)) and isinstance(y, (list, tuple)):
        if len(x) != len(y):
            return False
        return all(abs(float(a) - float(b)) <= tol for a, b in zip(x, y))
    return x == y
