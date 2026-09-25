"""Corpus registry: human-labeled public sets only. No model-written labels."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal

from vane.data.schema import QuestionType


class CorpusRole(str, Enum):
    """How a corpus participates in the data mix."""

    TRAIN_MIX = "train_mix"
    """Eligible for the fitting mix (subject to split resolutions)."""

    HOLDOUT = "holdout"
    """Entirely held out; never in training. Opened once for evaluation."""

    VISION_TRAIN = "vision_train"
    """Used only when building the vision checkpoint."""

    PROBE = "probe"
    """Synthetic / compiled probe (not a HF download target for training)."""


Modality = Literal["text", "vision"]


@dataclass(frozen=True, slots=True)
class CorpusSpec:
    """Catalog entry for an allowed human-labeled corpus."""

    id: str
    """Stable registry key (usually the HF dataset id)."""

    hf_id: str | None
    """Hugging Face dataset id when streamable; None for local/compiled probes."""

    role: CorpusRole
    primitives: frozenset[QuestionType]
    description: str
    modality: Modality = "text"
    #: Official / HF split names that may enter the training mix.
    train_mix_hf_splits: frozenset[str] = frozenset({"train"})
    #: Official splits that must not be used to select checkpoints.
    not_for_selection: frozenset[str] = field(default_factory=frozenset)
    #: Entire corpus is held out (not in training). Mirrors role=HOLDOUT.
    entirely_held_out: bool = False
    soft_labels: bool = False
    #: High-cardinality choice (e.g. BANKING77 k=77).
    high_k: bool = False
    notes: str = ""


def _choice(*extra: QuestionType) -> frozenset[QuestionType]:
    return frozenset((QuestionType.CHOICE, *extra))


def _noul(*extra: QuestionType) -> frozenset[QuestionType]:
    return frozenset((QuestionType.NOUL, *extra))


def _score(*extra: QuestionType) -> frozenset[QuestionType]:
    return frozenset((QuestionType.SCORE, *extra))


def _build_registry() -> dict[str, CorpusSpec]:
    """Human-labeled catalog from docs/data.md. No model-written labels."""
    entries: list[CorpusSpec] = [
        CorpusSpec(
            id="AmazonScience/massive",
            hf_id="SetFit/amazon_massive_intent_en-US",
            role=CorpusRole.TRAIN_MIX,
            primitives=_choice(),
            description="51 languages, 60 intents",
            notes=(
                "Languages not in the train split are held out and opened once. "
                "hf_id points at the SetFit parquet mirror (AmazonScience/massive "
                "dataset scripts are no longer supported on recent datasets)."
            ),
        ),
        CorpusSpec(
            id="clinc_oos",
            hf_id="clinc/clinc_oos",
            role=CorpusRole.TRAIN_MIX,
            primitives=_choice(),
            description="CLINC-150 in-scope plus out-of-scope",
        ),
        CorpusSpec(
            id="Tobi-Bueck/customer-support-tickets",
            hf_id="Tobi-Bueck/customer-support-tickets",
            role=CorpusRole.TRAIN_MIX,
            primitives=_choice(),
            description="Ticket labels",
        ),
        CorpusSpec(
            id="PolyAI/banking77",
            hf_id="mteb/banking77",
            role=CorpusRole.TRAIN_MIX,
            primitives=_choice(),
            description="13_083 queries, 77 intents; TRAIN split only for fitting",
            train_mix_hf_splits=frozenset({"train"}),
            not_for_selection=frozenset({"test"}),
            high_k=True,
            notes=(
                "BANKING77 train teaches k=77. BANKING77 official test (3080) "
                "is not used to select checkpoints; opened once as holdout. "
                "hf_id is the mteb parquet mirror (PolyAI/banking77 scripts "
                "unsupported on recent datasets); labels align with the "
                "official 77 intent names attached in code."
            ),
        ),
        CorpusSpec(
            id="google/boolq",
            hf_id="google/boolq",
            role=CorpusRole.TRAIN_MIX,
            primitives=_noul(),
            description="yes/no",
        ),
        CorpusSpec(
            id="nyu-mll/multi_nli",
            hf_id="nyu-mll/multi_nli",
            role=CorpusRole.TRAIN_MIX,
            primitives=_choice(),
            description="NLI",
            train_mix_hf_splits=frozenset({"train"}),
        ),
        CorpusSpec(
            id="xnli",
            hf_id="facebook/xnli",
            role=CorpusRole.TRAIN_MIX,
            primitives=_choice(),
            description="cross-lingual NLI",
            notes="Languages not in the train split are held out and opened once.",
        ),
        CorpusSpec(
            id="facebook/anli",
            hf_id="facebook/anli",
            role=CorpusRole.TRAIN_MIX,
            primitives=_choice(),
            description="adversarial NLI rounds 1–3",
        ),
        CorpusSpec(
            id="fever",
            hf_id="copenlu/fever_gold_evidence",
            role=CorpusRole.TRAIN_MIX,
            primitives=_choice(),
            description="supported / refuted / not-enough-info",
            notes=(
                "hf_id is a parquet FEVER mirror; the legacy fever dataset "
                "script is unsupported on recent datasets."
            ),
        ),
        CorpusSpec(
            id="google-research-datasets/go_emotions",
            hf_id="google-research-datasets/go_emotions",
            role=CorpusRole.TRAIN_MIX,
            primitives=_choice(),
            description="28 emotions, multi-annotator",
            soft_labels=True,
        ),
        CorpusSpec(
            id="google/civil_comments",
            hf_id="google/civil_comments",
            role=CorpusRole.TRAIN_MIX,
            primitives=_noul(QuestionType.SCORE),
            description="toxicity",
        ),
        CorpusSpec(
            id="deepset/prompt-injections",
            hf_id="deepset/prompt-injections",
            role=CorpusRole.TRAIN_MIX,
            primitives=_noul(),
            description="injection detect",
            not_for_selection=frozenset({"test"}),
            notes="prompt-injections test (n=116) opened once; not for selection.",
        ),
        CorpusSpec(
            id="lmsys/toxic-chat",
            hf_id="lmsys/toxic-chat",
            role=CorpusRole.TRAIN_MIX,
            primitives=_noul(),
            description="toxicity",
            not_for_selection=frozenset({"test"}),
            notes="toxic-chat test opened once; not for selection.",
        ),
        CorpusSpec(
            id="zefang-liu/phishing-email-dataset",
            hf_id="zefang-liu/phishing-email-dataset",
            role=CorpusRole.TRAIN_MIX,
            primitives=_noul(),
            description="phishing",
        ),
        CorpusSpec(
            id="SetFit/enron_spam",
            hf_id="SetFit/enron_spam",
            role=CorpusRole.TRAIN_MIX,
            primitives=_noul(),
            description="spam",
        ),
        CorpusSpec(
            id="nvidia/HelpSteer2",
            hf_id="nvidia/HelpSteer2",
            role=CorpusRole.TRAIN_MIX,
            primitives=_score(),
            description="five axes 0–4",
        ),
        CorpusSpec(
            id="SetFit/sst5",
            hf_id="SetFit/sst5",
            role=CorpusRole.TRAIN_MIX,
            primitives=_score(),
            description="SST-5 ordinal; TRAIN split only in the mix",
            train_mix_hf_splits=frozenset({"train"}),
            not_for_selection=frozenset({"test"}),
            notes=(
                "SST-5 train is in the mix so the ordinal head sees five levels. "
                "SST-5 test is not used to select checkpoints."
            ),
        ),
        CorpusSpec(
            id="microsoft/ms_marco",
            hf_id="microsoft/ms_marco",
            role=CorpusRole.TRAIN_MIX,
            primitives=_noul(),
            description="binary passage relevance",
        ),
        CorpusSpec(
            id="pfb30/multi_woz_2_2",
            hf_id="pfb30/multi_woz_2_2",
            role=CorpusRole.TRAIN_MIX,
            primitives=_noul(QuestionType.SCORE),
            description=(
                "label = verified goal success at dialogue end; prefixes are states"
            ),
        ),
        CorpusSpec(
            id="compiled-knowledge-quiz",
            hf_id=None,
            role=CorpusRole.PROBE,
            primitives=_noul(QuestionType.CHOICE),
            description=(
                "short public facts; dated holdout whose answer exists only "
                "in a passage in the state"
            ),
            train_mix_hf_splits=frozenset(),
            notes="Compiled locally. Dated fact holdout opened once.",
        ),
        CorpusSpec(
            id="long-document-probe",
            hf_id=None,
            role=CorpusRole.PROBE,
            primitives=_noul(),
            description=(
                "rare string at start / middle / end of a pad, and the pad "
                "without it; label = presence"
            ),
            train_mix_hf_splits=frozenset(),
        ),
        # --- Explicit entire holdout ---
        CorpusSpec(
            id="dair-ai/emotion",
            hf_id="dair-ai/emotion",
            role=CorpusRole.HOLDOUT,
            primitives=_choice(),
            description="entirely held out (not in training)",
            train_mix_hf_splits=frozenset(),
            entirely_held_out=True,
            notes="dair-ai/emotion is entirely held out; not in the training mix.",
        ),
        # --- Vision (vision checkpoint only) ---
        CorpusSpec(
            id="FUNSD",
            hf_id="nielsr/funsd",
            role=CorpusRole.VISION_TRAIN,
            primitives=_choice(),
            description="Form page image. Choice: field present or not; language.",
            modality="vision",
            notes="Vision checkpoint only.",
        ),
        CorpusSpec(
            id="DocVQA",
            hf_id="HuggingFaceM4/DocumentVQA",
            role=CorpusRole.VISION_TRAIN,
            primitives=_choice(),
            description=(
                "Page image plus a closed-label question. Answer is a short "
                "option list, not a generated span."
            ),
            modality="vision",
            notes="Vision checkpoint only.",
        ),
        CorpusSpec(
            id="Rico",
            hf_id="rico",
            role=CorpusRole.VISION_TRAIN,
            primitives=_choice(),
            description="UI state. Choice of the next action from a short menu.",
            modality="vision",
            notes="Vision checkpoint only. Listed corpus; no model-written labels.",
        ),
    ]
    registry = {e.id: e for e in entries}
    # Friendly aliases used in docs / tests.
    registry["MASSIVE"] = registry["AmazonScience/massive"]
    registry["BANKING77"] = registry["PolyAI/banking77"]
    registry["BoolQ"] = registry["google/boolq"]
    registry["SST-5"] = registry["SetFit/sst5"]
    registry["sst5"] = registry["SetFit/sst5"]
    return registry


CORPUS_REGISTRY: dict[str, CorpusSpec] = _build_registry()


def get_corpus(name: str) -> CorpusSpec:
    """Look up a corpus by registry id or alias."""
    try:
        return CORPUS_REGISTRY[name]
    except KeyError as exc:
        raise KeyError(
            f"unknown corpus {name!r}; known ids include "
            f"{sorted({s.id for s in CORPUS_REGISTRY.values()})}"
        ) from exc


def corpora_in_train_mix() -> list[CorpusSpec]:
    """Unique CorpusSpec entries eligible for the text training mix."""
    seen: set[str] = set()
    out: list[CorpusSpec] = []
    for spec in CORPUS_REGISTRY.values():
        if spec.id in seen:
            continue
        seen.add(spec.id)
        if spec.role is CorpusRole.TRAIN_MIX and not spec.entirely_held_out:
            out.append(spec)
    return out


def is_held_out(name: str) -> bool:
    """True if the corpus (or its documented test split policy) is a holdout."""
    spec = get_corpus(name)
    return spec.entirely_held_out or spec.role is CorpusRole.HOLDOUT


def allows_hf_split_in_train_mix(name: str, hf_split: str) -> bool:
    """Whether an official HF split may enter the fitting mix for this corpus."""
    spec = get_corpus(name)
    if spec.entirely_held_out or spec.role is CorpusRole.HOLDOUT:
        return False
    if spec.role is not CorpusRole.TRAIN_MIX:
        return False
    return hf_split in spec.train_mix_hf_splits


def used_for_checkpoint_selection(name: str, hf_split: str) -> bool:
    """False for documented not-for-selection splits (SST-5/BANKING77 test, …)."""
    spec = get_corpus(name)
    if hf_split in spec.not_for_selection:
        return False
    if spec.entirely_held_out:
        return False
    return True
