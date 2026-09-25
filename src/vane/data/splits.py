"""Split carving: development and calibration must not alias the same full val set.

Calibration is a held-out slice of train (or of validation if that is all that
exists), carved with a fixed seed **before** any checkpoint is chosen.
BANKING77 has no validation split: carve dev/calibration from train and never
touch the official test for selection.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, replace
from typing import Sequence

from vane.data.schema import Example, Split


@dataclass(frozen=True, slots=True)
class SplitBundle:
    """Train / development / calibration partitions (test stays untouched)."""

    train: list[Example]
    development: list[Example]
    calibration: list[Example]


def carve_splits(
    examples: Sequence[Example],
    *,
    seed: int = 0,
    development_fraction: float = 0.1,
    calibration_fraction: float = 0.1,
    source_split: Split = Split.TRAIN,
) -> SplitBundle:
    """Carve development and calibration from ``examples`` with a fixed seed.

    Does **not** alias development and calibration to the same full validation
    split. Fractions are taken sequentially from a shuffled copy; remaining
    rows stay in train. Rows originally marked test are refused.
    """
    if development_fraction < 0 or calibration_fraction < 0:
        raise ValueError("fractions must be non-negative")
    if development_fraction + calibration_fraction >= 1.0:
        raise ValueError("development_fraction + calibration_fraction must be < 1")

    pool = list(examples)
    for ex in pool:
        if ex.split is Split.TEST:
            raise ValueError(
                "carve_splits refuses test-split examples; never use official "
                "test for selection or calibration carving"
            )

    rng = random.Random(seed)
    rng.shuffle(pool)
    n = len(pool)
    n_cal = int(n * calibration_fraction)
    n_dev = int(n * development_fraction)
    # Ensure at least one row in each non-empty requested slice when possible.
    if calibration_fraction > 0 and n >= 3 and n_cal == 0:
        n_cal = 1
    if development_fraction > 0 and n >= 3 and n_dev == 0:
        n_dev = 1
    if n_cal + n_dev >= n and n >= 3:
        n_cal = max(1, min(n_cal, n // 3))
        n_dev = max(1, min(n_dev, (n - n_cal) // 2))

    cal_rows = pool[:n_cal]
    dev_rows = pool[n_cal : n_cal + n_dev]
    train_rows = pool[n_cal + n_dev :]

    def _retag(rows: list[Example], split: Split) -> list[Example]:
        return [replace(ex, split=split) for ex in rows]

    return SplitBundle(
        train=_retag(train_rows, Split.TRAIN),
        development=_retag(dev_rows, Split.DEVELOPMENT),
        calibration=_retag(cal_rows, Split.CALIBRATION),
    )


def carve_banking77(
    train_examples: Sequence[Example],
    *,
    seed: int = 0,
    development_fraction: float = 0.1,
    calibration_fraction: float = 0.1,
) -> SplitBundle:
    """BANKING77 has no validation split — carve from train only.

    Official test must never enter this function.
    """
    return carve_splits(
        train_examples,
        seed=seed,
        development_fraction=development_fraction,
        calibration_fraction=calibration_fraction,
        source_split=Split.TRAIN,
    )
