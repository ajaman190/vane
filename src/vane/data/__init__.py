"""Vane training data: schema, corpus registry, soft labels, transforms, loaders.

Core imports (schema / registry / labels / transforms / splits) stay free of
torch and of Hugging Face ``datasets``. Streaming loaders require
``pip install vane[data]``. Collate and the text mix pull torch/datasets lazily.

Splits (train / development / calibration / test) are assigned **before** any
checkpoint is selected. Soft targets π are annotator vote shares.
"""

from __future__ import annotations

from vane.data.labels import soft_label_from_votes, soft_noul_from_votes
from vane.data.loaders import (
    example_from_dict,
    examples_from_records,
    iter_corpus,
    require_datasets,
)
from vane.data.registry import (
    CORPUS_REGISTRY,
    CorpusRole,
    CorpusSpec,
    allows_hf_split_in_train_mix,
    corpora_in_train_mix,
    get_corpus,
    is_held_out,
    used_for_checkpoint_selection,
)
from vane.data.schema import (
    SPLIT_NAMES,
    Example,
    QuestionSpec,
    QuestionType,
    Split,
    SplitName,
    Target,
    parse_split,
    validate_split_name,
)
from vane.data.splits import SplitBundle, carve_banking77, carve_splits
from vane.data.transforms import (
    TransformOptions,
    apply_transforms,
    flip_state_representation,
    paraphrase_state,
    targets_equal,
)


def __getattr__(name: str):
    if name == "load_text_mix":
        from vane.data.mix import load_text_mix

        return load_text_mix
    if name == "TextMix":
        from vane.data.mix import TextMix

        return TextMix
    if name == "MixStats":
        from vane.data.mix import MixStats

        return MixStats
    if name == "collate_examples":
        from vane.data.collate import collate_examples

        return collate_examples
    if name == "BANKING77_INTENTS":
        from vane.data.banking77_intents import BANKING77_INTENTS

        return BANKING77_INTENTS
    if name == "map_banking77":
        from vane.data.mappers import map_banking77

        return map_banking77
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Schema
    "Split",
    "SplitName",
    "SPLIT_NAMES",
    "QuestionType",
    "QuestionSpec",
    "Example",
    "Target",
    "parse_split",
    "validate_split_name",
    # Registry
    "CORPUS_REGISTRY",
    "CorpusSpec",
    "CorpusRole",
    "get_corpus",
    "corpora_in_train_mix",
    "is_held_out",
    "allows_hf_split_in_train_mix",
    "used_for_checkpoint_selection",
    # Labels
    "soft_label_from_votes",
    "soft_noul_from_votes",
    # Transforms
    "TransformOptions",
    "apply_transforms",
    "paraphrase_state",
    "flip_state_representation",
    "targets_equal",
    # Splits
    "SplitBundle",
    "carve_splits",
    "carve_banking77",
    # Loaders (datasets optional)
    "require_datasets",
    "iter_corpus",
    "examples_from_records",
    "example_from_dict",
    # Lazy: mix / collate / banking77
    "load_text_mix",
    "TextMix",
    "MixStats",
    "collate_examples",
    "BANKING77_INTENTS",
    "map_banking77",
]
