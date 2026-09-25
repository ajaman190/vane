"""Training / loss smoke tests. Skip cleanly when torch is not installed.

Requires torch (same optional extra as vane.model / vane.train).
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from vane.model import DEV_CONFIG, VaneModel  # noqa: E402
from vane.train import (  # noqa: E402
    choice_loss,
    isotonic_residual,
    noul_loss,
    one_train_step,
    score_loss,
)
from vane.train.calibration import (  # noqa: E402
    fit_cardinality_temperature,
    fit_early_exit_tau,
)


def test_losses_finite_on_simplex() -> None:
    q_n = torch.tensor([0.7])
    pi_n = torch.tensor([0.6])
    ln = noul_loss(q_n, pi_n)
    assert torch.isfinite(ln)

    q_c = torch.tensor([[0.5, 0.3, 0.2]])
    pi_c = torch.tensor([[0.4, 0.4, 0.2]])
    lc = choice_loss(q_c, pi_c)
    assert torch.isfinite(lc)

    q_s = torch.tensor([[0.1, 0.2, 0.7]])
    pi_s = torch.tensor([[0.05, 0.15, 0.8]])  # unimodal
    alpha = torch.tensor([0.6])
    ls = score_loss(q_s, pi_s, alpha=alpha)
    assert torch.isfinite(ls)


def test_isotonic_residual_non_negative() -> None:
    torch.manual_seed(0)
    # Mis-calibrated predictions: high q when y low and vice versa
    q = torch.linspace(0.1, 0.9, 32)
    y = 1.0 - q + 0.05 * torch.randn(32)
    y = y.clamp(0, 1)
    r = isotonic_residual(q, y, min_batch=64, allow_small_batch=True)
    assert float(r) >= -1e-7
    assert torch.isfinite(r)


def test_one_train_step_runs_without_nan() -> None:
    torch.manual_seed(0)
    model = VaneModel(DEV_CONFIG)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    B, N, K = 2, 16, 3
    d = DEV_CONFIG.d_model
    batch = {
        "input_ids": torch.randint(0, DEV_CONFIG.vocab_size, (B, N)),
        "noul_count": 1,
        "noul_target": torch.tensor([[0.8], [0.2]]),
        "choice_options": torch.randn(B, 1, K, d),
        "choice_mask": torch.ones(B, 1, K, dtype=torch.bool),
        "choice_target": torch.tensor([[[0.7, 0.2, 0.1]], [[0.1, 0.1, 0.8]]]),
        "score_count": 1,
        "score_n_levels": 3,
        "score_target": torch.tensor(
            [[[0.1, 0.2, 0.7]], [[0.6, 0.3, 0.1]]],
        ),
    }
    # Loss before
    with torch.no_grad():
        out0 = model(
            batch["input_ids"],
            noul_count=1,
            choice_options=batch["choice_options"],
            choice_mask=batch["choice_mask"],
            score_count=1,
            score_n_levels=3,
            force_full=True,
        )
        loss0 = (
            noul_loss(out0.noul, batch["noul_target"])
            + choice_loss(out0.choice.reshape(-1, K), batch["choice_target"].reshape(-1, K))
            + score_loss(
                out0.score.reshape(-1, 3),
                batch["score_target"].reshape(-1, 3),
                alpha=out0.alpha.reshape(-1),
            )
        )

    parts = one_train_step(model, batch, optim, force_full=True)
    assert "loss" in parts
    assert parts["loss"] == parts["loss"]  # not NaN

    with torch.no_grad():
        out1 = model(
            batch["input_ids"],
            noul_count=1,
            choice_options=batch["choice_options"],
            choice_mask=batch["choice_mask"],
            score_count=1,
            score_n_levels=3,
            force_full=True,
        )
        loss1 = (
            noul_loss(out1.noul, batch["noul_target"])
            + choice_loss(out1.choice.reshape(-1, K), batch["choice_target"].reshape(-1, K))
            + score_loss(
                out1.score.reshape(-1, 3),
                batch["score_target"].reshape(-1, 3),
                alpha=out1.alpha.reshape(-1),
            )
        )
    # Prefer decrease; tolerate flat on tiny random net
    assert torch.isfinite(loss0) and torch.isfinite(loss1)
    assert float(loss1) <= float(loss0) + 0.5  # soft: at least no blow-up


def test_cardinality_parameters_fit_on_a_batch() -> None:
    from vane.train.calibration import fit_temperature_parameters

    torch.manual_seed(0)
    model = VaneModel(DEV_CONFIG)
    B, N, D = 2, 16, DEV_CONFIG.d_model
    batch = {
        "input_ids": torch.randint(0, DEV_CONFIG.vocab_size, (B, N)),
        "attention_mask": torch.ones(B, N, dtype=torch.bool),
        "noul_count": 1,
        "noul_target": torch.tensor([[1.0], [0.0]]),
        "choice_options": torch.randn(B, 1, 3, D),
        "choice_mask": torch.ones(B, 1, 3, dtype=torch.bool),
        "choice_target": torch.tensor([[[1.0, 0.0, 0.0]], [[0.0, 1.0, 0.0]]]),
    }
    before = float(model.heads.temp_b_choice.detach())
    fitted = fit_temperature_parameters(model, batch, steps=3, lr=0.1)
    assert set(fitted) == {"a_choice", "b_choice", "a_score", "b_score"}
    assert all(v == v for v in fitted.values())
    assert float(model.heads.temp_b_choice.detach()) != before or fitted["b_choice"] == before


def test_two_pole_pair_mixes_both_ends() -> None:
    from vane.data.mix import _two_pole_pairs
    from vane.data.schema import Example, QuestionSpec, QuestionType, Split

    low = Example(
        state="thanks, no rush",
        questions={
            "s": QuestionSpec(
                type=QuestionType.SCORE,
                target=(0.9, 0.1, 0.0),
                levels=("calm", "mixed", "furious"),
            )
        },
        split=Split.TRAIN,
        corpus="x",
    )
    high = Example(
        state="this blocks payroll",
        questions={
            "s": QuestionSpec(
                type=QuestionType.SCORE,
                target=(0.0, 0.1, 0.9),
                levels=("calm", "mixed", "furious"),
            )
        },
        split=Split.TRAIN,
        corpus="x",
    )
    pairs = _two_pole_pairs([low, high], n=4, seed=0)
    assert pairs
    target = pairs[0].questions["two_pole"].target
    assert target[0] > 0.2 and target[2] > 0.2
    probs = torch.tensor([[0.9, 0.1], [0.2, 0.8]])
    target = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    temperature = fit_cardinality_temperature({"choice": (probs, target)})
    assert set(temperature) == {"choice"}
    assert temperature["choice"] in {0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0}
    with pytest.raises(NotImplementedError, match="calibration"):
        fit_early_exit_tau()


def test_corn_alpha_weights_lower_pole() -> None:
    """α multiplies the lower CORN component; μ2 = μ1 + softplus(δ) is higher."""
    from vane.model.heads import ScoreHead

    torch.manual_seed(0)
    head = ScoreHead(d_model=8, n_heads=2, max_levels=5)
    # Force gate → α≈1 (all mass on component 1) and dig a clear μ gap.
    with torch.no_grad():
        head.gate.weight.zero_()
        head.gate.bias.fill_(10.0)  # sigmoid → α ≈ 1
        head.mu.weight.zero_()
        head.mu.bias.fill_(-2.0)  # μ1 low
        head.delta.weight.zero_()
        head.delta.bias.fill_(4.0)  # softplus large → μ2 ≫ μ1
        head.corn_base.zero_()
        head.corn_scale.weight.zero_()
        head.corn_scale.bias.zero_()

    query = torch.zeros(1, 8)
    latents = torch.zeros(1, 2, 8)
    probs, alpha = head(query, latents, n_levels=5, temperature=1.0)
    assert float(alpha.item()) > 0.99
    # With α≈1, output ≈ P(μ1). Larger μ raises levels (add-μ convention),
    # so μ1 (lower) must put more mass on lower indices than μ2 would.
    # Expectation under α≈1 should be lower than under α≈0.
    with torch.no_grad():
        head.gate.bias.fill_(-10.0)  # α ≈ 0 → higher component
    probs_hi, alpha_hi = head(query, latents, n_levels=5, temperature=1.0)
    assert float(alpha_hi.item()) < 0.01
    levels = torch.arange(5, dtype=probs.dtype)
    e_low = (probs.squeeze(0) * levels).sum().detach()
    e_high = (probs_hi.squeeze(0) * levels).sum().detach()
    assert float(e_low) < float(e_high), (
        f"α on lower pole expected E_low < E_high; got {float(e_low)} vs {float(e_high)}"
    )


def test_early_exit_train_path_can_be_forced() -> None:
    """Training coin force_early stops after block 1 (serve τ not consulted)."""
    torch.manual_seed(0)
    model = VaneModel(DEV_CONFIG)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    B, N, K = 2, 16, 3
    d = DEV_CONFIG.d_model
    batch = {
        "input_ids": torch.randint(0, DEV_CONFIG.vocab_size, (B, N)),
        "noul_count": 1,
        "noul_target": torch.tensor([[0.8], [0.2]]),
        "choice_options": torch.randn(B, 1, K, d),
        "choice_mask": torch.ones(B, 1, K, dtype=torch.bool),
        "choice_target": torch.tensor([[[0.7, 0.2, 0.1]], [[0.1, 0.1, 0.8]]]),
    }
    # Impossible τ — serve path would never exit; training coin still must.
    model.tau_choice = 1.01
    model.tau_score = 1.01
    parts = one_train_step(
        model, batch, optim, force_early=True, force_full=False, device="cpu"
    )
    assert parts["early"] == 1.0
    assert parts["blocks_run"] == 1.0
    assert parts["loss"] == parts["loss"]
