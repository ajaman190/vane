"""Streaming text-mix loader with per-corpus caps (laptop smoke friendly).

Full Civil Comments / MS MARCO / MNLI must not be downloaded onto a small
laptop for the smoke — streaming + ``max_examples_per_corpus`` is the smoke.
If a corpus 404s, is gated, or TLS-fails, log and continue; do not crash the
whole mix. Record which corpora actually yielded rows.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Mapping

from vane.data.mappers import HF_LOAD_KWARGS, HF_SPLIT_OVERRIDES, get_mapper
from vane.data.registry import CorpusRole, corpora_in_train_mix, get_corpus
from vane.data.schema import Example, Split, parse_split
from vane.data.splits import SplitBundle, carve_splits

logger = logging.getLogger(__name__)


@dataclass
class MixStats:
    """Which corpora yielded rows vs which were skipped (with error)."""

    yielded: dict[str, int] = field(default_factory=dict)
    skipped: dict[str, str] = field(default_factory=dict)

    def record_yield(self, corpus_id: str, n: int) -> None:
        self.yielded[corpus_id] = self.yielded.get(corpus_id, 0) + n

    def record_skip(self, corpus_id: str, error: str) -> None:
        self.skipped[corpus_id] = error


@dataclass
class TextMix:
    """Loaded (capped) examples plus yield/skip stats."""

    examples: list[Example]
    stats: MixStats
    splits: SplitBundle | None = None


def _config_get(config: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    cur: Any = config
    for key in keys:
        if not isinstance(cur, Mapping) or key not in cur:
            return default
        cur = cur[key]
    return cur


def load_text_mix(config: Mapping[str, Any] | None = None) -> TextMix:
    """Stream up to ``max_examples_per_corpus`` from each mapped text corpus.

    Parameters (via config mapping, all optional)
    ----------
    data.corpora : list[str]
        Registry ids to include; default = all ``TRAIN_MIX`` text corpora.
    data.max_examples_per_corpus : int
        Cap per corpus (smoke default 8–32).
    data.seed : int
        Used when carving development / calibration.
    data.development_fraction / data.calibration_fraction : float
        Held-out slices carved from the loaded pool (not the same full val).
    """
    config = dict(config or {})
    raw_cap = _config_get(config, "data", "max_examples_per_corpus", default=None)
    if raw_cap in (None, "", "null"):
        max_per = None
    else:
        max_per = int(raw_cap)
    seed = int(
        _config_get(config, "data", "seed", default=None)
        or _config_get(config, "seed", default=0)
        or 0
    )
    dev_frac = float(
        _config_get(config, "data", "development_fraction", default=0.1) or 0.1
    )
    cal_frac = float(
        _config_get(config, "data", "calibration_fraction", default=0.1) or 0.1
    )
    corpus_ids = _config_get(config, "data", "corpora", default=None)
    if corpus_ids is None:
        corpus_ids = [
            s.id
            for s in corpora_in_train_mix()
            if s.modality == "text" and s.role is CorpusRole.TRAIN_MIX
        ]

    # Lazy import so vane.data core stays torch/datasets free at import.
    from vane.data.loaders import iter_corpus

    stats = MixStats()
    examples: list[Example] = []

    for name in corpus_ids:
        try:
            spec = get_corpus(name)
        except KeyError as exc:
            stats.record_skip(str(name), f"unknown corpus: {exc}")
            logger.warning("skip corpus %s: unknown", name)
            continue
        if spec.role not in (CorpusRole.TRAIN_MIX, CorpusRole.VISION_TRAIN):
            stats.record_skip(spec.id, "not a train corpus")
            continue
        if spec.hf_id is None:
            stats.record_skip(spec.id, "no hf_id (probe)")
            continue

        try:
            mapper = get_mapper(spec)
        except NotImplementedError as exc:
            stats.record_skip(spec.id, str(exc))
            logger.warning("skip corpus %s: %s", spec.id, exc)
            continue

        n = 0
        try:
            for ex in iter_corpus(
                spec.id,
                Split.TRAIN,
                streaming=True,
                limit=max_per,
                mapper=mapper,
                download=False,
                hf_split=HF_SPLIT_OVERRIDES.get(spec.id),
                hf_kwargs=HF_LOAD_KWARGS.get(spec.id),
            ):
                examples.append(ex)
                n += 1
        except Exception as exc:  # noqa: BLE001 — smoke must continue on 404/gated/TLS
            stats.record_skip(spec.id, f"{type(exc).__name__}: {exc}")
            logger.warning("skip corpus %s: %s", spec.id, exc)
            continue

        if n == 0:
            stats.record_skip(spec.id, "mapped 0 rows")
            logger.warning("corpus %s yielded 0 rows", spec.id)
        else:
            stats.record_yield(spec.id, n)
            logger.info("corpus %s yielded %d rows", spec.id, n)

    splits = None
    if examples:
        # Carve before any checkpoint is chosen. Do not alias development and
        # calibration to the same full validation split.
        splits = carve_splits(
            examples,
            seed=seed,
            development_fraction=dev_frac,
            calibration_fraction=cal_frac,
        )
        # Training pool = carved train only. Two-pole pairs are added after,
        # so the calibration slice stays unimodal human labels.
        examples = list(splits.train)

    n_poles = int(_config_get(config, "data", "two_pole_pairs", default=0) or 0)
    if n_poles and splits is not None and splits.train:
        poles = _two_pole_pairs(list(splits.train), n_poles, seed)
        examples.extend(poles)
        if poles:
            stats.record_yield("two-pole", len(poles))

    return TextMix(examples=examples, stats=stats, splits=splits)


def _two_pole_pairs(examples: list[Example], n: int, seed: int) -> list[Example]:
    """Pair a low score with a high score. Both labels are human. The state is both texts.

    The target is an equal mixture, so the second CORN pole has something to fit.
    Levels must match. Pairs that share an argmax are skipped.
    """
    import random

    from vane.data.schema import QuestionSpec, QuestionType

    rows: list[tuple[Example, Any]] = []
    for ex in examples:
        for question in ex.questions.values():
            if question.type is not QuestionType.SCORE or not isinstance(question.target, tuple):
                continue
            if len(question.target) < 2:
                continue
            rows.append((ex, question))
    if len(rows) < 2:
        return []

    def pole(question: Any) -> int:
        target = question.target
        return max(range(len(target)), key=lambda i: target[i])

    rng = random.Random(seed)
    out: list[Example] = []
    for _ in range(n):
        left_ex, left_q = rows[rng.randrange(len(rows))]
        right_ex, right_q = rows[rng.randrange(len(rows))]
        if len(left_q.target) != len(right_q.target):
            continue
        if pole(left_q) == pole(right_q):
            continue
        mixed = tuple(
            0.5 * a + 0.5 * b for a, b in zip(left_q.target, right_q.target, strict=True)
        )
        total = sum(mixed) or 1.0
        mixed = tuple(x / total for x in mixed)
        state = f"{left_ex.state}\n{right_ex.state}"
        out.append(
            Example(
                state=state,
                questions={
                    "two_pole": QuestionSpec(
                        type=QuestionType.SCORE,
                        target=mixed,
                        levels=left_q.levels,
                        text=left_q.text,
                    )
                },
                split=Split.TRAIN,
                corpus="two-pole",
            )
        )
    return out


def iter_train_examples(config: Mapping[str, Any] | None = None):
    """Stream training rows. ``max_examples_per_corpus: null`` means the whole split."""
    config = dict(config or {})
    raw_cap = _config_get(config, "data", "max_examples_per_corpus", default=None)
    max_per = None if raw_cap in (None, "", "null") else int(raw_cap)
    corpus_ids = _config_get(config, "data", "corpora", default=None)
    if corpus_ids is None:
        corpus_ids = [
            s.id
            for s in corpora_in_train_mix()
            if s.modality == "text" and s.role is CorpusRole.TRAIN_MIX
        ]
    from vane.data.loaders import iter_corpus

    for name in corpus_ids:
        try:
            spec = get_corpus(name)
        except KeyError:
            continue
        if spec.role not in (CorpusRole.TRAIN_MIX, CorpusRole.VISION_TRAIN):
            continue
        if spec.hf_id is None:
            continue
        try:
            mapper = get_mapper(spec)
        except NotImplementedError:
            continue
        try:
            yield from iter_corpus(
                spec.id,
                Split.TRAIN,
                streaming=True,
                limit=max_per,
                mapper=mapper,
                download=False,
                hf_split=HF_SPLIT_OVERRIDES.get(spec.id),
                hf_kwargs=HF_LOAD_KWARGS.get(spec.id),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("skip corpus %s: %s", spec.id, exc)


def load_text_mix_from_yaml(path: str) -> TextMix:
    """Load a YAML config then :func:`load_text_mix`."""
    try:
        import yaml
    except ImportError as exc:
        raise ImportError(
            "PyYAML is required to load training configs. "
            "pip install pyyaml  (or pip install vane[train])"
        ) from exc
    with open(path, encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    # Resolve extends: like system_one base vs run yaml.
    if isinstance(config, dict) and "extends" in config:
        from pathlib import Path

        base_name = config["extends"]
        base_path = Path(path).parent / (
            base_name if str(base_name).endswith((".yaml", ".yml")) else f"{base_name}.yaml"
        )
        with open(base_path, encoding="utf-8") as bf:
            base = yaml.safe_load(bf) or {}
        config = _deep_merge(base, config)
    return load_text_mix(config)


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for key, value in override.items():
        if key == "extends":
            continue
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out
