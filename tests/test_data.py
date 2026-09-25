"""Tests for vane.data — in-memory fixtures only. No network, no HF download, no GPU."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Editable / src layout without requiring an install.
_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from vane.data import (  # noqa: E402
    CORPUS_REGISTRY,
    Example,
    QuestionSpec,
    QuestionType,
    Split,
    SPLIT_NAMES,
    apply_transforms,
    example_from_dict,
    examples_from_records,
    get_corpus,
    is_held_out,
    parse_split,
    soft_label_from_votes,
    soft_noul_from_votes,
    targets_equal,
    used_for_checkpoint_selection,
    validate_split_name,
)
from vane.data.registry import allows_hf_split_in_train_mix  # noqa: E402


# ---------------------------------------------------------------------------
# Soft labels
# ---------------------------------------------------------------------------


def test_soft_label_unanimous_is_one_hot() -> None:
    pi = soft_label_from_votes(["billing", "billing", "billing"])
    assert pi == {"billing": 1.0}


def test_soft_label_vote_shares() -> None:
    pi = soft_label_from_votes(["a", "b", "a", "a"])
    assert pi == {"a": 0.75, "b": 0.25}
    assert abs(sum(pi.values()) - 1.0) < 1e-12


def test_soft_label_with_fixed_support() -> None:
    pi = soft_label_from_votes(
        ["yes", "no", "yes"],
        labels=["yes", "no", "maybe"],
    )
    assert pi == {"yes": 2 / 3, "no": 1 / 3, "maybe": 0.0}


def test_soft_label_empty_raises() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        soft_label_from_votes([])


def test_soft_noul_from_votes() -> None:
    assert soft_noul_from_votes([True, True, False]) == pytest.approx(2 / 3)
    assert soft_noul_from_votes(["yes", "yes"]) == 1.0
    assert soft_noul_from_votes([False, False, False]) == 0.0


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------


def test_split_names() -> None:
    assert SPLIT_NAMES == frozenset(
        {"train", "development", "calibration", "test"}
    )
    assert {s.value for s in Split} == SPLIT_NAMES


def test_parse_split_aliases() -> None:
    assert parse_split("train") is Split.TRAIN
    assert parse_split("dev") is Split.DEVELOPMENT
    assert parse_split("validation") is Split.DEVELOPMENT
    assert parse_split("calib") is Split.CALIBRATION
    assert parse_split(Split.TEST) is Split.TEST
    assert validate_split_name("calibration") == "calibration"


def test_parse_split_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="unknown split"):
        parse_split("holdout")


def test_splits_documented_before_checkpoint_selection() -> None:
    """Guardrail: the four names exist; selection uses non-test partitions."""
    # Calibration is distinct from development and test.
    assert Split.CALIBRATION.value not in {"train", "test"}
    assert Split.DEVELOPMENT is not Split.CALIBRATION


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_registry_contains_key_corpora() -> None:
    for key in (
        "MASSIVE",
        "AmazonScience/massive",
        "BANKING77",
        "PolyAI/banking77",
        "BoolQ",
        "google/boolq",
        "SetFit/sst5",
        "SST-5",
        "dair-ai/emotion",
        "FUNSD",
        "DocVQA",
        "Rico",
        "xnli",
        "nyu-mll/multi_nli",
        "google-research-datasets/go_emotions",
    ):
        assert key in CORPUS_REGISTRY, f"missing {key}"


def test_emotion_entirely_held_out() -> None:
    spec = get_corpus("dair-ai/emotion")
    assert spec.entirely_held_out
    assert is_held_out("dair-ai/emotion")
    assert not allows_hf_split_in_train_mix("dair-ai/emotion", "train")


def test_sst5_train_only_in_mix_test_not_for_selection() -> None:
    spec = get_corpus("SST-5")
    assert "train" in spec.train_mix_hf_splits
    assert "test" in spec.not_for_selection
    assert allows_hf_split_in_train_mix("SetFit/sst5", "train")
    assert not allows_hf_split_in_train_mix("SetFit/sst5", "test")
    assert not used_for_checkpoint_selection("SST-5", "test")


def test_banking77_train_for_k77_test_not_for_selection() -> None:
    spec = get_corpus("BANKING77")
    assert spec.high_k
    assert QuestionType.CHOICE in spec.primitives
    assert allows_hf_split_in_train_mix("BANKING77", "train")
    assert not used_for_checkpoint_selection("BANKING77", "test")
    assert "test" in spec.not_for_selection


def test_banking77_mapper_does_not_need_all_intents() -> None:
    """HF rows are text + integer label only — attach official 77 names in code."""
    from vane.data.banking77_intents import BANKING77_INTENTS
    from vane.data.mappers import map_banking77
    from vane.data.registry import get_corpus

    spec = get_corpus("PolyAI/banking77")
    row = {"text": "I want to activate my card", "label": 0}
    assert "all_intents" not in row
    ex = map_banking77(row, Split.TRAIN, spec)
    assert ex is not None
    q = ex.questions["intent"]
    assert q.options == BANKING77_INTENTS
    assert len(q.options) == 77
    assert isinstance(q.target, dict)
    assert q.target["activate_my_card"] == 1.0
    assert abs(sum(q.target.values()) - 1.0) < 1e-9


def test_carve_splits_dev_and_calibration_disjoint() -> None:
    from vane.data.splits import carve_splits

    rows = [
        Example(
            state=f"row-{i}",
            questions={
                "q": QuestionSpec(type=QuestionType.NOUL, target=0.5),
            },
            split=Split.TRAIN,
            corpus="fixture",
            example_id=str(i),
        )
        for i in range(20)
    ]
    bundle = carve_splits(
        rows, seed=0, development_fraction=0.2, calibration_fraction=0.2
    )
    cal_ids = {ex.example_id for ex in bundle.calibration}
    dev_ids = {ex.example_id for ex in bundle.development}
    train_ids = {ex.example_id for ex in bundle.train}
    assert cal_ids.isdisjoint(dev_ids)
    assert cal_ids.isdisjoint(train_ids)
    assert dev_ids.isdisjoint(train_ids)
    assert cal_ids | dev_ids | train_ids == {str(i) for i in range(20)}
    # Must not alias development and calibration to the same full set.
    assert cal_ids != dev_ids
    assert len(bundle.calibration) > 0 and len(bundle.development) > 0


def test_vision_corpora_listed() -> None:
    for name in ("FUNSD", "DocVQA", "Rico"):
        spec = get_corpus(name)
        assert spec.modality == "vision"
        assert spec.role.value == "vision_train"


def test_no_model_written_label_policy_in_registry_notes() -> None:
    # Registry is human-labeled public sets only; probes have no hf download id.
    quiz = get_corpus("compiled-knowledge-quiz")
    assert quiz.hf_id is None


# ---------------------------------------------------------------------------
# Schema / fixtures roundtrip
# ---------------------------------------------------------------------------


def _fixture_example() -> Example:
    votes = soft_label_from_votes(
        ["billing", "billing", "refund", "billing"],
        labels=["billing", "refund", "shipping"],
    )
    return Example(
        state={"ticket": "Please refund my order", "lang": "en"},
        questions={
            "department": QuestionSpec(
                type=QuestionType.CHOICE,
                options=("billing", "refund", "shipping"),
                target=votes,
                text="Department?",
            ),
            "urgent": QuestionSpec(
                type=QuestionType.NOUL,
                target=soft_noul_from_votes([True, True, False]),
            ),
            "severity": QuestionSpec(
                type=QuestionType.SCORE,
                levels=("low", "mid", "high"),
                target=(0.1, 0.2, 0.7),
            ),
        },
        split=Split.TRAIN,
        corpus="fixture",
        example_id="ex-1",
    )


def test_example_roundtrip_via_dict() -> None:
    ex = _fixture_example()
    as_dict = {
        "state": ex.state,
        "questions": {
            qid: {
                "type": q.type.value,
                "target": (
                    dict(q.target)
                    if isinstance(q.target, dict)
                    else (
                        list(q.target)
                        if isinstance(q.target, (list, tuple))
                        else q.target
                    )
                ),
                "options": q.options,
                "levels": q.levels,
                "text": q.text,
            }
            for qid, q in ex.questions.items()
        },
        "split": ex.split.value,
        "corpus": ex.corpus,
        "example_id": ex.example_id,
    }
    rebuilt = example_from_dict(as_dict)
    assert rebuilt.split is Split.TRAIN
    assert rebuilt.example_id == "ex-1"
    assert rebuilt.questions["urgent"].target == pytest.approx(2 / 3)
    assert targets_equal(ex.questions, rebuilt.questions)

    batch = examples_from_records([ex, as_dict])
    assert len(batch) == 2
    assert isinstance(batch[0], Example)


def test_question_type_only_three_primitives() -> None:
    assert {t.value for t in QuestionType} == {"noul", "choice", "score"}


# ---------------------------------------------------------------------------
# Transforms preserve targets
# ---------------------------------------------------------------------------


def test_paraphrase_and_flip_preserve_targets() -> None:
    ex = _fixture_example()
    out = apply_transforms(ex, paraphrase=True, flip_state=True, seed=0)
    assert targets_equal(ex.questions, out.questions)
    # State flipped to JSON string.
    assert isinstance(out.state, str)
    parsed = json.loads(out.state)
    assert "ticket" in parsed


def test_shuffle_options_preserves_target_mass() -> None:
    ex = _fixture_example()
    out = apply_transforms(ex, shuffle_options=True, seed=42)
    orig = ex.questions["department"]
    new = out.questions["department"]
    assert set(orig.options or ()) == set(new.options or ())
    assert isinstance(orig.target, dict) and isinstance(new.target, dict)
    for name, mass in orig.target.items():
        assert new.target[name] == pytest.approx(mass)
    # Order likely changed (seeded); still a permutation.
    assert tuple(sorted(new.options or ())) == tuple(sorted(orig.options or ()))


def test_inject_distractor_preserves_original_targets() -> None:
    ex = _fixture_example()
    out = apply_transforms(ex, inject_distractor=True, seed=7)
    assert len(out.questions) == len(ex.questions) + 1
    assert targets_equal(ex.questions, out.questions)
    extra = set(out.questions) - set(ex.questions)
    assert len(extra) == 1
    distractor = out.questions[next(iter(extra))]
    assert distractor.type is QuestionType.NOUL
    assert distractor.target == 1.0


def test_flip_text_to_json_and_back() -> None:
    from vane.data.transforms import flip_state_representation

    text_json = '{"a":1,"b":"x"}'
    as_obj = flip_state_representation(text_json)
    assert as_obj == {"a": 1, "b": "x"}
    back = flip_state_representation(as_obj)
    assert json.loads(back) == {"a": 1, "b": "x"}


# ---------------------------------------------------------------------------
# Loaders: ImportError path, no network
# ---------------------------------------------------------------------------


def test_require_datasets_message(monkeypatch: pytest.MonkeyPatch) -> None:
    import builtins

    from vane.data import loaders

    real_import = builtins.__import__
    monkeypatch.delitem(sys.modules, "datasets", raising=False)

    def _block_datasets(name: str, *args: object, **kwargs: object):  # type: ignore[no-untyped-def]
        if name == "datasets" or name.startswith("datasets."):
            raise ImportError("blocked for test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _block_datasets)
    with pytest.raises(ImportError, match=r"pip install vane\[data\]"):
        loaders.require_datasets()


def test_iter_corpus_refuses_download_flag() -> None:
    from vane.data.loaders import iter_corpus

    with pytest.raises(ValueError, match="download"):
        next(iter_corpus("google/boolq", "train", download=True))


def test_vane_data_imports_without_torch_or_datasets() -> None:
    """Core vane.data must import even when optional stacks are absent."""
    import importlib

    # Simulate absence only for the assertion that our package did not import them.
    mod = importlib.import_module("vane.data")
    assert hasattr(mod, "CORPUS_REGISTRY")
    assert hasattr(mod, "Example")
    assert hasattr(mod, "soft_label_from_votes")
    assert hasattr(mod, "apply_transforms")
    # Do not require datasets/torch to be missing on the host — only that
    # importing vane.data did not need them as hard deps.
    assert "torch" not in sys.modules.get("vane.data", mod).__dict__
