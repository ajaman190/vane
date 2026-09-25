"""Soft-label helpers: π = annotator vote share."""

from __future__ import annotations

from collections import Counter
from collections.abc import Hashable, Sequence
from typing import TypeVar

T = TypeVar("T", bound=Hashable)


def soft_label_from_votes(
    votes: Sequence[T],
    *,
    labels: Sequence[T] | None = None,
) -> dict[T, float]:
    """Build soft target π from annotator votes.

    - Multiple annotators → vote-share simplex.
    - All votes agree (or a single vote) → one-hot on that label.
    - If ``labels`` is given, every label appears (zero mass when unvoted)
      so the simplex covers a fixed support (e.g. a choice option set).

    Raises:
        ValueError: empty ``votes``, or ``labels`` empty / missing voted keys.
    """
    if not votes:
        raise ValueError("votes must be non-empty")

    counts: Counter[T] = Counter(votes)
    total = sum(counts.values())

    if labels is not None:
        if not labels:
            raise ValueError("labels must be non-empty when provided")
        label_list = list(labels)
        if len(set(label_list)) != len(label_list):
            raise ValueError("labels must be unique")
        unknown = set(counts) - set(label_list)
        if unknown:
            raise ValueError(f"votes contain labels outside labels=: {unknown!r}")
        return {lab: counts.get(lab, 0) / total for lab in label_list}

    # Preserve first-seen order from votes for stable dicts.
    ordered: list[T] = []
    seen: set[T] = set()
    for v in votes:
        if v not in seen:
            seen.add(v)
            ordered.append(v)
    return {lab: counts[lab] / total for lab in ordered}


def soft_noul_from_votes(
    votes: Sequence[bool | int | str],
    *,
    yes_values: frozenset[object] = frozenset({True, 1, "yes", "Yes", "YES", "true"}),
) -> float:
    """Vote-share P(yes) for a noul question. One-hot (0 or 1) when unanimous."""
    if not votes:
        raise ValueError("votes must be non-empty")
    yes = sum(1 for v in votes if v in yes_values)
    return yes / len(votes)
