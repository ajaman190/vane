"""Training-example schema (not the HTTP request surface).

Splits are assigned before any checkpoint is selected. Soft targets π are
annotator vote shares; one-hot only when the label is certain.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal, Mapping, Sequence

SplitName = Literal["train", "development", "calibration", "test"]
"""Canonical split names. Assigned before checkpoint selection."""

SPLIT_NAMES: frozenset[str] = frozenset(
    ("train", "development", "calibration", "test")
)


class Split(str, Enum):
    """Corpus partitions. Fit never peeks at test; T/τ fit on calibration only."""

    TRAIN = "train"
    DEVELOPMENT = "development"
    CALIBRATION = "calibration"
    TEST = "test"


class QuestionType(str, Enum):
    """The three decision primitives. No fourth type."""

    NOUL = "noul"
    CHOICE = "choice"
    SCORE = "score"


StateValue = str | Mapping[str, Any] | Sequence[Any]
"""Encoded state: raw text, nested JSON-like mapping, or a list of strings."""

# Soft target π:
#   noul  -> float in [0, 1] = P(yes)
#   choice -> Mapping[option_name, float] simplex over options
#   score  -> Sequence[float] simplex over ordered levels (lowest first)
Target = float | Mapping[str, float] | Sequence[float]


@dataclass(frozen=True, slots=True)
class QuestionSpec:
    """One typed question with soft target π for training."""

    type: QuestionType
    target: Target
    #: Choice option names, length k with 2 <= k <= 255.
    options: tuple[str, ...] | None = None
    #: Optional per-option descriptions (null allowed).
    option_descriptions: Mapping[str, str | None] | None = None
    #: Score rubric levels, lowest first, length L with 2 <= L <= 10.
    levels: tuple[str, ...] | None = None
    #: Free-form prompt / stem for the question (not an HTTP field).
    text: str | None = None

    def __post_init__(self) -> None:
        qtype = (
            self.type
            if isinstance(self.type, QuestionType)
            else QuestionType(self.type)
        )
        if qtype is not self.type:
            object.__setattr__(self, "type", qtype)

        if qtype is QuestionType.NOUL:
            if not isinstance(self.target, (int, float)):
                raise TypeError("noul target must be a float in [0, 1]")
            if not 0.0 <= float(self.target) <= 1.0:
                raise ValueError("noul target must be in [0, 1]")
            if self.options is not None or self.levels is not None:
                raise ValueError("noul must not set options or levels")
        elif qtype is QuestionType.CHOICE:
            if self.options is None or len(self.options) < 2:
                raise ValueError("choice requires options with k >= 2")
            if len(self.options) > 255:
                raise ValueError("choice requires k <= 255")
            if len(set(self.options)) != len(self.options):
                raise ValueError("choice options must be unique")
            if not isinstance(self.target, Mapping):
                raise TypeError("choice target must be a mapping over option names")
            _check_simplex(dict(self.target), set(self.options), kind="choice")
        elif qtype is QuestionType.SCORE:
            if self.levels is None or len(self.levels) < 2:
                raise ValueError("score requires levels with L >= 2")
            if len(self.levels) > 10:
                raise ValueError("score requires L <= 10")
            if not isinstance(self.target, Sequence) or isinstance(
                self.target, (str, bytes)
            ):
                raise TypeError("score target must be a sequence over levels")
            if len(self.target) != len(self.levels):
                raise ValueError("score target length must match levels")
            _check_simplex_seq(tuple(float(x) for x in self.target), kind="score")
        else:  # pragma: no cover - enum exhaustiveness
            raise ValueError(f"unknown question type: {qtype!r}")


@dataclass(frozen=True, slots=True)
class Example:
    """One training/eval item: shared state plus typed questions with soft π."""

    state: StateValue
    questions: Mapping[str, QuestionSpec]
    split: Split
    corpus: str | None = None
    example_id: str | None = None
    images: tuple[bytes, ...] = field(default_factory=tuple)
    meta: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        split = self.split if isinstance(self.split, Split) else parse_split(self.split)
        if split is not self.split:
            object.__setattr__(self, "split", split)
        if not self.questions:
            raise ValueError("example must contain at least one question")
        # Freeze mapping copies so callers cannot mutate after construction.
        object.__setattr__(self, "questions", dict(self.questions))
        if self.meta and not isinstance(self.meta, dict):
            object.__setattr__(self, "meta", dict(self.meta))


def parse_split(name: str | Split) -> Split:
    """Parse and validate a split name.

    Splits must be assigned before any checkpoint is selected. Calibration
    alone fits T(type, k) and τ(type, k); test is opened once.
    """
    if isinstance(name, Split):
        return name
    key = str(name).strip().lower()
    # Common HF aliases → Vane split names.
    aliases = {
        "train": Split.TRAIN,
        "development": Split.DEVELOPMENT,
        "dev": Split.DEVELOPMENT,
        "validation": Split.DEVELOPMENT,
        "val": Split.DEVELOPMENT,
        "calibration": Split.CALIBRATION,
        "calib": Split.CALIBRATION,
        "test": Split.TEST,
    }
    if key not in aliases:
        raise ValueError(
            f"unknown split {name!r}; expected one of "
            f"{sorted(SPLIT_NAMES)} (aliases: dev/validation→development, "
            "calib→calibration)"
        )
    return aliases[key]


def validate_split_name(name: str | Split) -> SplitName:
    """Return the canonical SplitName string after validation."""
    return parse_split(name).value  # type: ignore[return-value]


def _check_simplex(
    target: Mapping[str, float],
    allowed: set[str],
    *,
    kind: str,
    tol: float = 1e-6,
) -> None:
    keys = set(target)
    if keys != allowed:
        raise ValueError(
            f"{kind} target keys {sorted(keys)} must equal options {sorted(allowed)}"
        )
    values = [float(target[k]) for k in allowed]
    if any(v < -tol for v in values):
        raise ValueError(f"{kind} target has negative mass")
    total = sum(values)
    if abs(total - 1.0) > tol:
        raise ValueError(f"{kind} target must sum to 1 (got {total})")


def _check_simplex_seq(
    values: Sequence[float], *, kind: str, tol: float = 1e-6
) -> None:
    if any(v < -tol for v in values):
        raise ValueError(f"{kind} target has negative mass")
    total = sum(values)
    if abs(total - 1.0) > tol:
        raise ValueError(f"{kind} target must sum to 1 (got {total})")
