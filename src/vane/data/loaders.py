"""Streaming loaders for public HF datasets (optional ``vane[data]`` extra).

This module never downloads corpora at import time. Callers that invoke
``iter_corpus`` must have ``datasets`` installed; tests must use in-memory
fixtures only and must not hit the network.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from typing import Any

from vane.data.registry import CORPUS_REGISTRY, CorpusSpec, get_corpus
from vane.data.schema import (
    Example,
    QuestionSpec,
    QuestionType,
    Split,
    parse_split,
)

RowMapper = Callable[[dict[str, Any], Split, CorpusSpec], Example | None]


def require_datasets() -> Any:
    """Import Hugging Face ``datasets``, or raise a clear InstallError."""
    try:
        import datasets  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            "Hugging Face datasets is required for corpus streaming. "
            "Install the optional extra: pip install vane[data]"
        ) from exc
    return datasets


def iter_corpus(
    name: str,
    split: Split | str,
    *,
    streaming: bool = True,
    limit: int | None = None,
    hf_split: str | None = None,
    mapper: RowMapper | None = None,
    download: bool = False,
    hf_kwargs: Mapping[str, Any] | None = None,
) -> Iterator[Example]:
    """Stream examples from a registered public HF corpus.

    **Streaming-only public API.** This package never materializes full
    training corpora. Keep ``download=False`` and ``streaming=True``.

    Parameters
    ----------
    name:
        Registry id or alias (e.g. ``\"BANKING77\"``, ``\"google/boolq\"``).
    split:
        Vane split name (train / development / calibration / test).
        Development and calibration are **not** silently aliased to the same
        full HF validation split by callers that carve locally — prefer
        :func:`vane.data.splits.carve_splits` after streaming train.
    streaming:
        Prefer HF streaming mode (no full local materialization).
    limit:
        Optional cap on yielded examples (useful for dry runs / smoke).
    hf_split:
        Override the HF split string; default maps from the Vane split.
    mapper:
        Optional row→Example converter. If omitted, a built-in mapper is used.
    download:
        Must stay False. Setting True raises ``ValueError``.
    hf_kwargs:
        Extra kwargs for ``datasets.load_dataset`` (e.g. ``name=`` config).
    """
    if download:
        raise ValueError(
            "iter_corpus is streaming-only: download=True is refused. "
            "Keep download=False and streaming=True (pip install vane[data]). "
            "Full corpus downloads are out of scope for this package API."
        )

    datasets = require_datasets()
    spec = get_corpus(name)
    vane_split = parse_split(split)

    if spec.entirely_held_out and vane_split is Split.TRAIN:
        raise ValueError(
            f"corpus {spec.id!r} is entirely held out and must not enter "
            "the training mix"
        )

    if spec.hf_id is None:
        raise ValueError(
            f"corpus {spec.id!r} has no hf_id (compiled/local probe); "
            "build examples in-memory instead of streaming"
        )

    hf_split_name = hf_split or _default_hf_split(spec, vane_split)
    row_mapper = mapper or _builtin_mapper(spec)
    load_kwargs: dict[str, Any] = dict(hf_kwargs or {})

    # streaming=True avoids a silent full download when the cache is cold.
    ds = datasets.load_dataset(
        spec.hf_id,
        split=hf_split_name,
        streaming=streaming,
        **load_kwargs,
    )

    n = 0
    for row in ds:
        if not isinstance(row, Mapping):
            row = dict(row)
        example = row_mapper(dict(row), vane_split, spec)
        if example is None:
            continue
        yield example
        n += 1
        if limit is not None and n >= limit:
            return


def examples_from_records(
    records: list[Example] | list[dict[str, Any]],
) -> list[Example]:
    """In-memory fixture helper: pass through Examples or build from dicts."""
    out: list[Example] = []
    for rec in records:
        if isinstance(rec, Example):
            out.append(rec)
        else:
            out.append(example_from_dict(rec))
    return out


def example_from_dict(data: Mapping[str, Any]) -> Example:
    """Build an ``Example`` from a plain dict (tests / fixtures)."""
    questions: dict[str, QuestionSpec] = {}
    raw_qs = data["questions"]
    for qid, q in raw_qs.items():
        qtype = q["type"] if isinstance(q["type"], QuestionType) else QuestionType(q["type"])
        opts = q.get("options")
        levels = q.get("levels")
        questions[qid] = QuestionSpec(
            type=qtype,
            target=q["target"],
            options=tuple(opts) if opts is not None else None,
            option_descriptions=q.get("option_descriptions"),
            levels=tuple(levels) if levels is not None else None,
            text=q.get("text"),
        )
    images = data.get("images") or ()
    return Example(
        state=data["state"],
        questions=questions,
        split=parse_split(data["split"]),
        corpus=data.get("corpus"),
        example_id=data.get("example_id"),
        images=tuple(images),
        meta=dict(data.get("meta") or {}),
    )


def _default_hf_split(spec: CorpusSpec, vane_split: Split) -> str:
    """Map Vane split → a conventional HF split string.

    Calibration / development are expected to be **carved** from train (or
    from a single validation pool) by :mod:`vane.data.splits` — this helper
    only picks an HF string when someone streams those names directly. It does
    **not** mean development and calibration are the same carved set.
    """
    if vane_split is Split.TRAIN:
        if "train" in spec.train_mix_hf_splits:
            return "train"
        if spec.train_mix_hf_splits:
            return sorted(spec.train_mix_hf_splits)[0]
        return "train"
    if vane_split is Split.DEVELOPMENT:
        return "validation"
    if vane_split is Split.CALIBRATION:
        # Prefer an explicit calibration HF split if ever present; otherwise
        # callers should carve from train rather than aliasing full validation.
        return "validation"
    return "test"


def _builtin_mapper(spec: CorpusSpec) -> RowMapper:
    """Dispatch to :mod:`vane.data.mappers` for every text train-mix corpus."""
    try:
        from vane.data.mappers import get_mapper

        return get_mapper(spec)
    except NotImplementedError:

        def _unsupported(
            row: dict[str, Any], split: Split, corpus: CorpusSpec
        ) -> Example | None:
            raise NotImplementedError(
                f"no built-in row mapper for {corpus.id!r}; pass mapper= to "
                "iter_corpus, or use examples_from_records for fixtures"
            )

        return _unsupported


# Re-export registry for callers that import loaders only.
__all__ = [
    "require_datasets",
    "iter_corpus",
    "examples_from_records",
    "example_from_dict",
    "CORPUS_REGISTRY",
]
