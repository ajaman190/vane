"""Model smoke tests. Skip cleanly when torch is not installed (core CI).

Requires the optional train/model extra: ``pip install torch``.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from vane.model import (  # noqa: E402
    FULL_TEXT_CONFIG,
    DEV_CONFIG,
    ContextTooLongError,
    VaneConfig,
    VaneModel,
    confidence,
)


@pytest.fixture
def smoke_model() -> VaneModel:
    torch.manual_seed(0)
    return VaneModel(DEV_CONFIG)


def test_smoke_forward_cpu(smoke_model: VaneModel) -> None:
    B, N, K = 2, 16, 3
    d = DEV_CONFIG.d_model
    input_ids = torch.randint(0, DEV_CONFIG.vocab_size, (B, N))
    choice_options = torch.randn(B, 1, K, d)
    choice_mask = torch.ones(B, 1, K, dtype=torch.bool)
    out = smoke_model(
        input_ids,
        noul_count=1,
        choice_options=choice_options,
        choice_mask=choice_mask,
        score_count=1,
        score_n_levels=3,
        force_full=True,
    )
    assert out.noul is not None and out.noul.shape == (B, 1)
    assert out.choice is not None and out.choice.shape == (B, 1, K)
    assert torch.allclose(out.choice.sum(-1), torch.ones(B, 1), atol=1e-5)
    assert out.score is not None and out.score.shape == (B, 1, 3)
    assert out.alpha is not None
    assert out.blocks_run == DEV_CONFIG.n_blocks


def test_confidence_uses_true_k_not_padded_width() -> None:
    """C(q) must use true option count k, not the padded tensor width."""
    # True k=2 point mass → C=1. Padded to width 4 with zeros must not change C.
    q = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    mask = torch.tensor([[True, True, False, False]])
    c_true = confidence(q, mask=mask)
    c_padded = confidence(q)  # would use k=4 incorrectly for a k=2 simplex
    assert float(c_true) == pytest.approx(1.0, abs=1e-5)
    # Padded-width formula: (4*1 - 1)/(4-1) = 1 still for point mass on first,
    # so also check a near-uniform-on-true-k case:
    q2 = torch.tensor([[0.5, 0.5, 0.0, 0.0]])
    assert float(confidence(q2, mask=mask)) == pytest.approx(0.0, abs=1e-5)
    assert float(confidence(q2, k=2)) == pytest.approx(0.0, abs=1e-5)
    # Using padded width k=4 on a k=2 uniform is wrong (not 0).
    assert float(confidence(q2)) != pytest.approx(0.0, abs=1e-3)


def test_early_exit_one_interior_blocks(smoke_model: VaneModel) -> None:
    """One interior question blocks exit for the whole request."""
    B, N, K = 1, 8, 4
    d = DEV_CONFIG.d_model
    input_ids = torch.randint(0, DEV_CONFIG.vocab_size, (B, N))

    # Peak confidence formula: need C >= tau. With tau=0, always exit after block 1.
    smoke_model.tau_choice = 0.0
    smoke_model.tau_score = 0.0
    out_exit = smoke_model(
        input_ids,
        choice_options=torch.randn(B, 1, K, d),
        choice_mask=torch.ones(B, 1, K, dtype=torch.bool),
        use_early_exit=True,
        force_full=False,
    )
    assert out_exit.blocks_run == 1
    assert out_exit.early_exit is not None and bool(out_exit.early_exit.all())

    # tau=1.01 impossible → interior always blocks → full stack
    smoke_model.tau_choice = 1.01
    smoke_model.tau_score = 1.01
    out_full = smoke_model(
        input_ids,
        choice_options=torch.randn(B, 2, K, d),  # two questions; both "interior" vs tau
        choice_mask=torch.ones(B, 2, K, dtype=torch.bool),
        use_early_exit=True,
        force_full=False,
    )
    assert out_full.blocks_run == DEV_CONFIG.n_blocks
    assert out_full.early_exit is not None and not bool(out_full.early_exit.any())

    # Mixed C(q): one sharp + one interior under mid τ → must NOT exit.
    # Direct gate check (matches vane.eval.early_exit_allowed semantics).
    smoke_model.tau_choice = 0.80
    smoke_model.tau_score = 0.80
    sharp = torch.tensor([[0.96, 0.02, 0.01, 0.01]])  # C ≈ 0.947
    interior = torch.tensor([[0.40, 0.30, 0.20, 0.10]])  # C ≈ 0.20
    from vane.model.confidence import confidence as conf_t

    cs = torch.stack([conf_t(sharp), conf_t(interior)], dim=1)  # [1, 2]
    assert float(cs[0, 0]) >= 0.80
    assert float(cs[0, 1]) < 0.80
    gate = smoke_model._early_exit_ok(
        {"choice_confidence": cs, "score_confidence": None},
        has_noul=False,
    )
    assert not bool(gate.item()), "one interior C must block early exit"

def test_refuse_context_longer_than_max(smoke_model: VaneModel) -> None:
    # Temporarily shrink max_context for a fast test
    smoke_model.config.max_context = 32
    input_ids = torch.randint(0, DEV_CONFIG.vocab_size, (1, 33))
    with pytest.raises(ContextTooLongError):
        smoke_model(input_ids, noul_count=1, force_full=True)


def test_option_permutation_equivariance(smoke_model: VaneModel) -> None:
    torch.manual_seed(1)
    B, N, K = 1, 8, 4
    d = DEV_CONFIG.d_model
    input_ids = torch.randint(0, DEV_CONFIG.vocab_size, (B, N))
    options = torch.randn(B, 1, K, d)
    mask = torch.ones(B, 1, K, dtype=torch.bool)

    out = smoke_model(
        input_ids, choice_options=options, choice_mask=mask, force_full=True
    )
    assert out.choice is not None
    q = out.choice[0, 0]

    perm = torch.tensor([2, 0, 3, 1])
    options_p = options[:, :, perm, :]
    out_p = smoke_model(
        input_ids, choice_options=options_p, choice_mask=mask, force_full=True
    )
    assert out_p.choice is not None
    q_p = out_p.choice[0, 0]
    # Permuting options must permute probabilities
    assert torch.allclose(q_p, q[perm], atol=1e-5, rtol=1e-4)


def test_full_text_config_is_object_only() -> None:
    """FULL_TEXT_CONFIG documents ~1B scale; do not instantiate in tests."""
    assert FULL_TEXT_CONFIG.max_context == 32768
    assert FULL_TEXT_CONFIG.window_size == 128
    est = FULL_TEXT_CONFIG.estimated_param_count()
    # Order-of-magnitude near 1B (allow 0.5B–2.5B for rough formula)
    assert 500_000_000 <= est <= 2_500_000_000
    assert isinstance(DEV_CONFIG, VaneConfig)
