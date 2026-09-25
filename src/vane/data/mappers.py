"""Row → Example mappers for every text train-mix corpus in the registry.

Vision corpora stay registered but are not mapped for the text run.
Mappers never require network at import time.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from vane.data.banking77_intents import BANKING77_INTENTS
from vane.data.labels import soft_label_from_votes
from vane.data.registry import CorpusSpec
from vane.data.schema import Example, QuestionSpec, QuestionType, Split

# ---------------------------------------------------------------------------
# Shared option / label tables
# ---------------------------------------------------------------------------

NLI_LABELS: tuple[str, ...] = ("entailment", "neutral", "contradiction")
FEVER_LABELS: tuple[str, ...] = ("SUPPORTS", "REFUTES", "NOT ENOUGH INFO")
SST5_LEVELS: tuple[str, ...] = ("0", "1", "2", "3", "4")
HELPSTEER_AXES: tuple[str, ...] = (
    "helpfulness",
    "correctness",
    "coherence",
    "complexity",
    "verbosity",
)
GOEMOTIONS_LABELS: tuple[str, ...] = (
    "admiration",
    "amusement",
    "anger",
    "annoyance",
    "approval",
    "caring",
    "confusion",
    "curiosity",
    "desire",
    "disappointment",
    "disapproval",
    "disgust",
    "embarrassment",
    "excitement",
    "fear",
    "gratitude",
    "grief",
    "joy",
    "love",
    "nervousness",
    "optimism",
    "pride",
    "realization",
    "relief",
    "remorse",
    "sadness",
    "surprise",
    "neutral",
)

CLINC_OOS_INTENTS: tuple[str, ...] | None = None  # filled lazily from row meta if present

MASSIVE_INTENT_FALLBACK: tuple[str, ...] = tuple(f"intent_{i}" for i in range(60))


def _one_hot_choice(options: tuple[str, ...], name: str) -> dict[str, float]:
    if name not in options:
        raise ValueError(f"label {name!r} not in options")
    return {o: 1.0 if o == name else 0.0 for o in options}


def _one_hot_levels(levels: tuple[str, ...], idx: int) -> tuple[float, ...]:
    return tuple(1.0 if i == idx else 0.0 for i in range(len(levels)))


def _as_text(state: Any) -> str:
    if isinstance(state, str):
        return state
    if isinstance(state, Mapping):
        parts = [f"{k}: {v}" for k, v in state.items()]
        return "\n".join(parts)
    return str(state)


def _noul_from_label(label: Any) -> float | None:
    if label is None:
        return None
    if isinstance(label, str):
        low = label.strip().lower()
        if low in {"true", "yes", "1", "toxic", "spam", "phishing", "injection"}:
            return 1.0
        if low in {"false", "no", "0", "ham", "benign", "safe"}:
            return 0.0
        try:
            return 1.0 if float(low) >= 0.5 else 0.0
        except ValueError:
            return None
    if isinstance(label, (int, float)):
        v = float(label)
        if 0.0 <= v <= 1.0:
            return v
        return 1.0 if v > 0 else 0.0
    if isinstance(label, bool):
        return 1.0 if label else 0.0
    return None


# ---------------------------------------------------------------------------
# Per-corpus mappers
# ---------------------------------------------------------------------------


def map_massive(row: dict[str, Any], split: Split, spec: CorpusSpec) -> Example | None:
    text = row.get("utt", row.get("text", row.get("sentence")))
    intent = row.get("intent", row.get("label"))
    if text is None or intent is None:
        return None
    names = row.get("intent_names") or row.get("all_intents")
    label_text = row.get("label_text")
    if names and len(names) >= 2:
        options = tuple(str(x) for x in names)
    elif label_text is not None:
        # SetFit mirror: single label_text; build a stable 2-way menu for smoke
        # when the full intent catalog is not on the row. Prefer expanding with
        # a generic "other" so k>=2 without requiring all_intents.
        name = str(label_text)
        options = (name, "other_intent")
        return Example(
            state=str(text),
            questions={
                "intent": QuestionSpec(
                    type=QuestionType.CHOICE,
                    target=_one_hot_choice(options, name),
                    options=options,
                    text="Which intent?",
                )
            },
            split=split,
            corpus=spec.id,
            example_id=str(row["id"]) if "id" in row else None,
            meta={"locale": row.get("locale") or row.get("partition"), "soft": "one_hot_single_label"},
        )
    else:
        options = MASSIVE_INTENT_FALLBACK
    if isinstance(intent, int):
        if not 0 <= intent < len(options):
            return None
        name = options[intent]
    else:
        name = str(intent)
        if name not in options:
            options = options + (name,)
    return Example(
        state=str(text),
        questions={
            "intent": QuestionSpec(
                type=QuestionType.CHOICE,
                target=_one_hot_choice(options, name),
                options=options,
                text="Which intent?",
            )
        },
        split=split,
        corpus=spec.id,
        example_id=str(row["id"]) if "id" in row else None,
        meta={"locale": row.get("locale") or row.get("partition")},
    )


def map_clinc_oos(row: dict[str, Any], split: Split, spec: CorpusSpec) -> Example | None:
    text = row.get("text")
    intent = row.get("intent", row.get("label"))
    if text is None or intent is None:
        return None
    names = row.get("intent_names") or row.get("all_intents")
    label_text = row.get("label_text")
    if names and len(names) >= 2:
        options = tuple(str(x) for x in names)
        if isinstance(intent, int):
            if not 0 <= intent < len(options):
                return None
            name = options[intent]
        else:
            name = str(intent)
            if name not in options:
                options = options + (name,)
    elif label_text is not None:
        name = str(label_text)
        options = (name, "oos_or_other")
    elif isinstance(intent, str):
        name = intent
        options = (name, "oos_or_other")
    elif isinstance(intent, int):
        # No catalog on the row: keep k small for smoke (full 151-way needs feature names).
        name = f"intent_{intent}"
        options = (name, "oos_or_other")
    else:
        return None
    return Example(
        state=str(text),
        questions={
            "intent": QuestionSpec(
                type=QuestionType.CHOICE,
                target=_one_hot_choice(options, name),
                options=options,
                text="Which CLINC intent?",
            )
        },
        split=split,
        corpus=spec.id,
    )


def map_customer_support(
    row: dict[str, Any], split: Split, spec: CorpusSpec
) -> Example | None:
    text = (
        row.get("body")
        or row.get("text")
        or row.get("ticket")
        or row.get("content")
        or row.get("description")
    )
    label = row.get("queue") or row.get("label") or row.get("type") or row.get("category")
    if text is None or label is None:
        return None
    names = row.get("all_queues") or row.get("label_names") or row.get("all_intents")
    if names and len(names) >= 2:
        options = tuple(str(x) for x in names)
    else:
        # Minimal binary-or-small support so a lone label still trains choice.
        label_s = str(label)
        options = (label_s, f"other_than_{label_s}")
    if isinstance(label, int):
        if not 0 <= label < len(options):
            return None
        name = options[label]
    else:
        name = str(label)
        if name not in options:
            options = options + (name,)
    return Example(
        state=str(text),
        questions={
            "queue": QuestionSpec(
                type=QuestionType.CHOICE,
                target=_one_hot_choice(options, name),
                options=options,
                text="Which support queue?",
            )
        },
        split=split,
        corpus=spec.id,
    )


def map_banking77(row: dict[str, Any], split: Split, spec: CorpusSpec) -> Example | None:
    """BANKING77: HF rows are ``text`` + integer ``label`` only.

    Official 77 intent names are attached in code. Do **not** require
    ``all_intents``; do not drop every row for lacking that column.
    """
    text = row.get("text", row.get("query"))
    label = row.get("label", row.get("intent"))
    if text is None or label is None:
        return None
    options = BANKING77_INTENTS
    if "label_text" in row and row["label_text"] is not None:
        name = str(row["label_text"])
        if name not in options:
            # Tolerate alternate spellings by falling back to index when possible.
            if isinstance(label, int) and 0 <= int(label) < 77:
                name = options[int(label)]
            else:
                return None
    elif isinstance(label, int):
        if not 0 <= int(label) < 77:
            return None
        name = options[int(label)]
    else:
        name = str(label)
        if name not in options:
            return None
    target = _one_hot_choice(options, name)
    return Example(
        state=str(text),
        questions={
            "intent": QuestionSpec(
                type=QuestionType.CHOICE,
                target=target,
                options=options,
                text="Which banking intent?",
            )
        },
        split=split,
        corpus=spec.id,
    )


def map_boolq(row: dict[str, Any], split: Split, spec: CorpusSpec) -> Example | None:
    passage = row.get("passage", "")
    question = row.get("question", "")
    label = row.get("answer", row.get("label"))
    pi = _noul_from_label(label)
    if pi is None:
        return None
    return Example(
        state={"passage": passage, "question": question},
        questions={
            "answer": QuestionSpec(
                type=QuestionType.NOUL,
                target=pi,
                text=question or "Is the answer yes?",
            )
        },
        split=split,
        corpus=spec.id,
        example_id=str(row["id"]) if "id" in row else None,
    )


def map_mnli(row: dict[str, Any], split: Split, spec: CorpusSpec) -> Example | None:
    premise = row.get("premise", row.get("sentence1"))
    hypothesis = row.get("hypothesis", row.get("sentence2"))
    label = row.get("label")
    if premise is None or hypothesis is None or label is None:
        return None
    if isinstance(label, int) and label < 0:
        return None
    if isinstance(label, int):
        if not 0 <= label < 3:
            return None
        name = NLI_LABELS[label]
    else:
        name = str(label).lower()
        aliases = {
            "entailment": "entailment",
            "neutral": "neutral",
            "contradiction": "contradiction",
        }
        name = aliases.get(name, name)
        if name not in NLI_LABELS:
            return None
    return Example(
        state={"premise": premise, "hypothesis": hypothesis},
        questions={
            "nli": QuestionSpec(
                type=QuestionType.CHOICE,
                target=_one_hot_choice(NLI_LABELS, name),
                options=NLI_LABELS,
                text="Premise–hypothesis relation?",
            )
        },
        split=split,
        corpus=spec.id,
    )


def map_xnli(row: dict[str, Any], split: Split, spec: CorpusSpec) -> Example | None:
    return map_mnli(row, split, spec)


def map_anli(row: dict[str, Any], split: Split, spec: CorpusSpec) -> Example | None:
    return map_mnli(row, split, spec)


def map_fever(row: dict[str, Any], split: Split, spec: CorpusSpec) -> Example | None:
    claim = row.get("claim", row.get("text"))
    label = row.get("label")
    if claim is None or label is None:
        return None
    if isinstance(label, int):
        if not 0 <= label < len(FEVER_LABELS):
            return None
        name = FEVER_LABELS[label]
    else:
        raw = str(label).upper().replace(" ", "_")
        aliases = {
            "SUPPORTS": "SUPPORTS",
            "REFUTES": "REFUTES",
            "NOT_ENOUGH_INFO": "NOT ENOUGH INFO",
            "NOT ENOUGH INFO": "NOT ENOUGH INFO",
            "NEI": "NOT ENOUGH INFO",
        }
        name = aliases.get(raw, aliases.get(str(label).upper(), None))
        if name is None:
            return None
    return Example(
        state=str(claim),
        questions={
            "verdict": QuestionSpec(
                type=QuestionType.CHOICE,
                target=_one_hot_choice(FEVER_LABELS, name),
                options=FEVER_LABELS,
                text="FEVER verdict?",
            )
        },
        split=split,
        corpus=spec.id,
    )


def map_go_emotions(
    row: dict[str, Any], split: Split, spec: CorpusSpec
) -> Example | None:
    """GoEmotions soft targets.

    When the row has annotator vote lists, π = vote shares. Otherwise, if the
    file has a single label, a one-hot is acceptable — stated explicitly here.
    Multi-label presence lists are normalized to a simplex over present labels.
    """
    text = row.get("text")
    if text is None:
        return None
    options = GOEMOTIONS_LABELS

    # Raw multi-annotator votes: list of label-name lists or ints per rater.
    votes = row.get("votes") or row.get("annotator_labels")
    if votes and isinstance(votes, Sequence) and not isinstance(votes, (str, bytes)):
        flat: list[str] = []
        for v in votes:
            if isinstance(v, Sequence) and not isinstance(v, (str, bytes)):
                for item in v:
                    if isinstance(item, int) and 0 <= item < len(options):
                        flat.append(options[item])
                    else:
                        flat.append(str(item))
            elif isinstance(v, int) and 0 <= v < len(options):
                flat.append(options[v])
            else:
                flat.append(str(v))
        if flat:
            target = soft_label_from_votes(flat, labels=list(options))
            return Example(
                state=str(text),
                questions={
                    "emotion": QuestionSpec(
                        type=QuestionType.CHOICE,
                        target=target,
                        options=options,
                        text="Emotion?",
                    )
                },
                split=split,
                corpus=spec.id,
                meta={"soft": "annotator_vote_shares"},
            )

    labels = row.get("labels", row.get("label"))
    if labels is None:
        return None

    # Single integer label → one-hot (file has a single label).
    if isinstance(labels, int):
        if not 0 <= labels < len(options):
            return None
        # One-hot only because this file row has a single label (not vote shares).
        target = _one_hot_choice(options, options[labels])
        return Example(
            state=str(text),
            questions={
                "emotion": QuestionSpec(
                    type=QuestionType.CHOICE,
                    target=target,
                    options=options,
                    text="Emotion?",
                )
            },
            split=split,
            corpus=spec.id,
            meta={"soft": "one_hot_single_label"},
        )

    if isinstance(labels, str):
        if labels not in options:
            return None
        # One-hot only because this file row has a single label (not vote shares).
        target = _one_hot_choice(options, labels)
        return Example(
            state=str(text),
            questions={
                "emotion": QuestionSpec(
                    type=QuestionType.CHOICE,
                    target=target,
                    options=options,
                    text="Emotion?",
                )
            },
            split=split,
            corpus=spec.id,
            meta={"soft": "one_hot_single_label"},
        )

    if isinstance(labels, Sequence):
        present: list[str] = []
        for item in labels:
            if isinstance(item, int) and 0 <= item < len(options):
                present.append(options[item])
            else:
                s = str(item)
                if s in options:
                    present.append(s)
        if not present:
            return None
        if len(present) == 1:
            # One-hot only because this file row has a single label.
            target = _one_hot_choice(options, present[0])
            soft_kind = "one_hot_single_label"
        else:
            # Multi-label presence → equal mass over present emotions (no rater
            # vote column on this config); still a simplex over full support.
            target = soft_label_from_votes(present, labels=list(options))
            soft_kind = "multi_label_equal_mass"
        return Example(
            state=str(text),
            questions={
                "emotion": QuestionSpec(
                    type=QuestionType.CHOICE,
                    target=target,
                    options=options,
                    text="Emotion?",
                )
            },
            split=split,
            corpus=spec.id,
            meta={"soft": soft_kind},
        )
    return None


def map_civil_comments(
    row: dict[str, Any], split: Split, spec: CorpusSpec
) -> Example | None:
    text = row.get("text", row.get("comment_text"))
    tox = row.get("toxicity", row.get("target", row.get("label")))
    if text is None or tox is None:
        return None
    try:
        score = float(tox)
    except (TypeError, ValueError):
        return None
    score = min(1.0, max(0.0, score))
    # Noul: P(toxic); score: five bins over [0,1].
    levels = SST5_LEVELS
    idx = min(4, int(score * 5)) if score < 1.0 else 4
    return Example(
        state=str(text),
        questions={
            "toxic": QuestionSpec(
                type=QuestionType.NOUL,
                target=score,
                text="Is this comment toxic?",
            ),
            "toxicity_level": QuestionSpec(
                type=QuestionType.SCORE,
                target=_one_hot_levels(levels, idx),
                levels=levels,
                text="Toxicity level (low → high)",
            ),
        },
        split=split,
        corpus=spec.id,
    )


def map_prompt_injections(
    row: dict[str, Any], split: Split, spec: CorpusSpec
) -> Example | None:
    text = row.get("text", row.get("prompt"))
    label = row.get("label", row.get("injection"))
    pi = _noul_from_label(label)
    if text is None or pi is None:
        return None
    return Example(
        state=str(text),
        questions={
            "injection": QuestionSpec(
                type=QuestionType.NOUL,
                target=pi,
                text="Is this a prompt injection?",
            )
        },
        split=split,
        corpus=spec.id,
    )


def map_toxic_chat(row: dict[str, Any], split: Split, spec: CorpusSpec) -> Example | None:
    text = row.get("user_input") or row.get("text") or row.get("conversation")
    label = row.get("toxicity")
    if label is None:
        label = row.get("label")
    pi = _noul_from_label(label)
    if text is None or pi is None:
        return None
    return Example(
        state=str(text),
        questions={
            "toxic": QuestionSpec(
                type=QuestionType.NOUL,
                target=pi,
                text="Is this chat toxic?",
            )
        },
        split=split,
        corpus=spec.id,
    )


def map_phishing(row: dict[str, Any], split: Split, spec: CorpusSpec) -> Example | None:
    text = (
        row.get("Email Text")
        or row.get("text")
        or row.get("email")
        or row.get("body")
        or row.get("content")
    )
    label = row.get("Email Type") or row.get("label") or row.get("type")
    if text is None or label is None:
        return None
    if isinstance(label, str):
        low = label.lower()
        pi = 1.0 if "phish" in low else 0.0
    else:
        pi = _noul_from_label(label)
        if pi is None:
            return None
    return Example(
        state=str(text),
        questions={
            "phishing": QuestionSpec(
                type=QuestionType.NOUL,
                target=pi,
                text="Is this a phishing email?",
            )
        },
        split=split,
        corpus=spec.id,
    )


def map_enron_spam(row: dict[str, Any], split: Split, spec: CorpusSpec) -> Example | None:
    text = row.get("text") or row.get("email") or row.get("message")
    label = row.get("label") or row.get("label_text")
    pi = _noul_from_label(label)
    if text is None or pi is None:
        # SetFit/enron_spam often uses label 0/1 with text
        if text is not None and isinstance(label, str) and "spam" in label.lower():
            pi = 1.0
        elif text is not None and isinstance(label, str):
            pi = 0.0
        else:
            return None
    return Example(
        state=str(text),
        questions={
            "spam": QuestionSpec(
                type=QuestionType.NOUL,
                target=float(pi),
                text="Is this email spam?",
            )
        },
        split=split,
        corpus=spec.id,
    )


def map_helpsteer2(row: dict[str, Any], split: Split, spec: CorpusSpec) -> Example | None:
    prompt = row.get("prompt", "")
    response = row.get("response", "")
    if not prompt and not response:
        return None
    questions: dict[str, QuestionSpec] = {}
    for axis in HELPSTEER_AXES:
        if axis not in row or row[axis] is None:
            continue
        try:
            idx = int(row[axis])
        except (TypeError, ValueError):
            continue
        if not 0 <= idx <= 4:
            continue
        questions[axis] = QuestionSpec(
            type=QuestionType.SCORE,
            target=_one_hot_levels(SST5_LEVELS, idx),
            levels=SST5_LEVELS,
            text=f"HelpSteer2 {axis} (0–4)",
        )
    if not questions:
        return None
    return Example(
        state={"prompt": prompt, "response": response},
        questions=questions,
        split=split,
        corpus=spec.id,
    )


def map_sst5(row: dict[str, Any], split: Split, spec: CorpusSpec) -> Example | None:
    text = row.get("text", row.get("sentence"))
    label = row.get("label")
    if text is None or label is None:
        return None
    idx = int(label)
    if not 0 <= idx <= 4:
        return None
    return Example(
        state=str(text),
        questions={
            "sentiment": QuestionSpec(
                type=QuestionType.SCORE,
                target=_one_hot_levels(SST5_LEVELS, idx),
                levels=SST5_LEVELS,
                text="Sentiment (very negative → very positive)",
            )
        },
        split=split,
        corpus=spec.id,
    )


def map_ms_marco(row: dict[str, Any], split: Split, spec: CorpusSpec) -> Example | None:
    query = row.get("query") or row.get("question")
    if query is None:
        return None
    # passages may be dict with is_selected / passage_text
    passages = row.get("passages") or row.get("passage")
    if isinstance(passages, Mapping):
        texts = passages.get("passage_text") or passages.get("text") or []
        selected = passages.get("is_selected") or passages.get("selected") or []
        if texts and selected:
            # One example per (query, passage) pair would explode; take first selected
            # or first passage for smoke.
            for t, sel in zip(texts, selected):
                pi = 1.0 if int(sel) > 0 else 0.0
                return Example(
                    state={"query": query, "passage": t},
                    questions={
                        "relevant": QuestionSpec(
                            type=QuestionType.NOUL,
                            target=pi,
                            text="Is the passage relevant?",
                        )
                    },
                    split=split,
                    corpus=spec.id,
                )
        if texts:
            return Example(
                state={"query": query, "passage": texts[0]},
                questions={
                    "relevant": QuestionSpec(
                        type=QuestionType.NOUL,
                        target=0.0,
                        text="Is the passage relevant?",
                    )
                },
                split=split,
                corpus=spec.id,
            )
    answers = row.get("answers") or row.get("answer")
    if answers is not None:
        pi = 1.0 if answers else 0.0
        return Example(
            state=str(query),
            questions={
                "has_answer": QuestionSpec(
                    type=QuestionType.NOUL,
                    target=pi,
                    text="Does this query have an answer?",
                )
            },
            split=split,
            corpus=spec.id,
        )
    return None


def map_multi_woz(row: dict[str, Any], split: Split, spec: CorpusSpec) -> Example | None:
    # Dialogue-level success / turns. Shape varies by config.
    dialogue_id = row.get("dialogue_id") or row.get("id")
    turns = row.get("turns") or row.get("utterances") or row.get("dialogue")
    success = row.get("success") or row.get("goal_success") or row.get("label")
    if turns is None and "text" in row:
        state = str(row["text"])
    elif isinstance(turns, Sequence) and turns:
        parts: list[str] = []
        for t in turns:
            if isinstance(t, Mapping):
                speaker = t.get("speaker") or t.get("role") or ""
                utt = t.get("utterance") or t.get("text") or ""
                parts.append(f"{speaker}: {utt}".strip(": "))
            else:
                parts.append(str(t))
        state = "\n".join(parts)
    else:
        return None
    pi = _noul_from_label(success) if success is not None else 1.0
    if pi is None:
        pi = 1.0
    questions: dict[str, QuestionSpec] = {
        "goal_success": QuestionSpec(
            type=QuestionType.NOUL,
            target=float(pi),
            text="Was the dialogue goal successful?",
        )
    }
    # Optional ordinal progress score if present.
    progress = row.get("progress") or row.get("score")
    if progress is not None:
        try:
            idx = min(4, max(0, int(float(progress))))
            questions["progress"] = QuestionSpec(
                type=QuestionType.SCORE,
                target=_one_hot_levels(SST5_LEVELS, idx),
                levels=SST5_LEVELS,
                text="Dialogue progress (low → high)",
            )
        except (TypeError, ValueError):
            pass
    return Example(
        state=state,
        questions=questions,
        split=split,
        corpus=spec.id,
        example_id=str(dialogue_id) if dialogue_id is not None else None,
    )


def _image_bytes(image: Any) -> bytes | None:
    if image is None:
        return None
    if isinstance(image, bytes):
        return image
    if isinstance(image, dict) and isinstance(image.get("bytes"), bytes):
        return image["bytes"]
    save = getattr(image, "save", None)
    if callable(save):
        import io

        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return buf.getvalue()
    return None


def map_funsd(row: dict[str, Any], split: Split, spec: CorpusSpec) -> Example | None:
    """Form page. Noul: the page has a non-O NER tag. Human labels, closed question."""
    raw = _image_bytes(row.get("image"))
    if raw is None:
        return None
    tags = row.get("ner_tags") or row.get("labels") or []
    try:
        present = any(int(tag) != 0 for tag in tags)
    except (TypeError, ValueError):
        present = False
    return Example(
        state="form page",
        questions={
            "has_label": QuestionSpec(
                type=QuestionType.NOUL,
                target=1.0 if present else 0.0,
                text="Does this page contain a labeled form field?",
            )
        },
        split=split,
        corpus=spec.id,
        images=(raw,),
    )


def map_docvqa(row: dict[str, Any], split: Split, spec: CorpusSpec) -> Example | None:
    """Page plus question. Noul: the human answer is numeric. Not a generated span."""
    raw = _image_bytes(row.get("image"))
    if raw is None:
        return None
    answers = row.get("answers")
    text = ""
    if isinstance(answers, dict):
        values = answers.get("text") or []
        text = str(values[0]) if values else ""
    elif isinstance(answers, list) and answers:
        text = str(answers[0])
    digits = text.replace(",", "").replace(".", "").replace(" ", "")
    numeric = bool(digits) and digits.isdigit()
    question = str(row.get("question") or "document question")
    return Example(
        state=question,
        questions={
            "numeric": QuestionSpec(
                type=QuestionType.NOUL,
                target=1.0 if numeric else 0.0,
                text="Is the answer a number?",
            )
        },
        split=split,
        corpus=spec.id,
        images=(raw,),
    )


def map_rico(row: dict[str, Any], split: Split, spec: CorpusSpec) -> Example | None:
    """UI screenshot. Noul from a human clickable/activity flag when the row has one."""
    raw = _image_bytes(row.get("image") or row.get("screenshot"))
    if raw is None:
        return None
    flag = row.get("clickable")
    if flag is None:
        flag = row.get("is_clickable")
    if flag is None:
        return None
    return Example(
        state=str(row.get("activity_name") or row.get("label") or "ui screen"),
        questions={
            "clickable": QuestionSpec(
                type=QuestionType.NOUL,
                target=1.0 if bool(flag) else 0.0,
                text="Is the primary element clickable?",
            )
        },
        split=split,
        corpus=spec.id,
        images=(raw,),
    )


# ---------------------------------------------------------------------------
# Registry dispatch
# ---------------------------------------------------------------------------

MAPPERS: dict[str, Any] = {
    "AmazonScience/massive": map_massive,
    "clinc_oos": map_clinc_oos,
    "Tobi-Bueck/customer-support-tickets": map_customer_support,
    "PolyAI/banking77": map_banking77,
    "google/boolq": map_boolq,
    "nyu-mll/multi_nli": map_mnli,
    "xnli": map_xnli,
    "facebook/anli": map_anli,
    "fever": map_fever,
    "google-research-datasets/go_emotions": map_go_emotions,
    "google/civil_comments": map_civil_comments,
    "deepset/prompt-injections": map_prompt_injections,
    "lmsys/toxic-chat": map_toxic_chat,
    "zefang-liu/phishing-email-dataset": map_phishing,
    "SetFit/enron_spam": map_enron_spam,
    "nvidia/HelpSteer2": map_helpsteer2,
    "SetFit/sst5": map_sst5,
    "microsoft/ms_marco": map_ms_marco,
    "pfb30/multi_woz_2_2": map_multi_woz,
    "FUNSD": map_funsd,
    "DocVQA": map_docvqa,
    "Rico": map_rico,
}


def get_mapper(spec: CorpusSpec):
    """Return the built-in row mapper for a corpus, or raise."""
    fn = MAPPERS.get(spec.id)
    if fn is None:
        raise NotImplementedError(
            f"no built-in row mapper for {spec.id!r}; vision/holdout/probe "
            "corpora are not part of the text train mix"
        )
    return fn


# HF load_dataset kwargs that differ per corpus (config / name).
HF_LOAD_KWARGS: dict[str, dict[str, Any]] = {
    "AmazonScience/massive": {},  # SetFit mirror; no config name
    "clinc_oos": {"name": "plus"},
    "xnli": {"name": "en"},
    "facebook/anli": {},
    "fever": {},
    "google-research-datasets/go_emotions": {"name": "simplified"},
    "microsoft/ms_marco": {"name": "v1.1"},
    "lmsys/toxic-chat": {"name": "toxicchat0124"},
}

# Prefer an HF split other than plain "train" when streaming.
HF_SPLIT_OVERRIDES: dict[str, str] = {
    "facebook/anli": "train_r1",
}


__all__ = [
    "BANKING77_INTENTS",
    "MAPPERS",
    "HF_LOAD_KWARGS",
    "HF_SPLIT_OVERRIDES",
    "get_mapper",
    "map_banking77",
    "map_go_emotions",
    "map_boolq",
    "map_sst5",
]
