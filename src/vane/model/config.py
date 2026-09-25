"""Vane network configuration objects (no weights)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class VaneConfig:
    """Architecture hyperparameters for a Vane text tower.

    ``FULL_TEXT_CONFIG`` is a design-target object only — do not instantiate
    ``VaneModel(FULL_TEXT_CONFIG)`` in tests (too large for CPU smoke).
    """

    d_model: int = 32
    n_blocks: int = 2
    n_heads: int = 4
    d_ff: int | None = None
    window_size: int = 128
    max_context: int = 32768
    vocab_size: int = 256
    max_levels: int = 10
    dropout: float = 0.0
    # Cardinality temperature params T(type,k)=softplus(a + b log k)
    temp_a_choice: float = 0.0
    temp_b_choice: float = 0.0
    temp_a_score: float = 0.0
    temp_b_score: float = 0.0
    # Early-exit thresholds τ(type, k); scalars used as τ for all k in smoke
    tau_choice: float = 0.9
    tau_score: float = 0.9
    # Score loss coefficients (also referenced by train; stored for convenience)
    score_rps_beta: float = 1.0
    score_entropy_gamma: float = 0.01
    # Vision projection stub width (unused unless vision tokens are passed)
    vision_proj_dim: int | None = None
    # Latent-bank refinement steps on the uncertain path. 0 disables them.
    n_refine: int = 2
    # Frozen encoder hidden size. None means token ids use the learned embed.
    backbone_dim: int | None = None
    name: str = "vane"

    def __post_init__(self) -> None:
        if self.d_ff is None:
            self.d_ff = 4 * self.d_model
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
            )
        if self.window_size < 1:
            raise ValueError("window_size must be >= 1")
        if self.n_blocks < 1:
            raise ValueError("n_blocks must be >= 1")

    def estimated_param_count(self) -> int:
        """Rough parameter count for the text tower (embeddings + blocks + heads).

        Ignores tiny bias terms. Useful for documenting FULL_TEXT_CONFIG scale.
        """
        d = self.d_model
        ff = self.d_ff if self.d_ff is not None else 4 * d
        # token embed (+ tied unused) + learned pos within window
        embed = self.vocab_size * d + self.window_size * d
        # per block: window attn (4 d^2) + latent cross (2 d^2) + bank attn (4 d^2) + FFN (2 d ff)
        # + latent query (d) + norms (~2d) — count main mats
        per_block = (4 + 2 + 4) * d * d + 2 * d * ff
        blocks = self.n_blocks * per_block
        # option-set: 2 encoder layers ~ 2*(4d^2 + 2 d ff), plus noul/choice/score heads
        option_set = 2 * (4 * d * d + 2 * d * ff)
        heads = d * 1 + d * d + (2 * (self.max_levels - 1) + 3) * d
        vision = 0
        if self.vision_proj_dim is not None:
            vision = self.vision_proj_dim * d
        return embed + blocks + option_set + heads + vision


# Small randomly initialized config for tests and the dev training profile.
DEV_CONFIG = VaneConfig(
    d_model=32,
    n_blocks=2,
    n_heads=4,
    d_ff=64,
    window_size=128,
    max_context=32768,
    vocab_size=128,
    max_levels=5,
    dropout=0.0,
    tau_choice=0.95,
    tau_score=0.95,
    name="vane-dev",
)


# Design target ~1B text tower (config object only; do not materialize in tests).
#
# Param estimate (estimated_param_count):
#   d=1536, L=28, ff=6144, V=128000, w=128
#   embed ≈ V*d + w*d ≈ 197M
#   blocks ≈ 28 * (10*d^2 + 2*d*ff) ≈ 28 * (23.6M + 18.9M) ≈ 1.19B
#   heads/option-set ≈ tens of M
#   total ≈ 1.4B (order-of-magnitude design target near 1B–1.5B;
#   production start is a published multilingual encoder near 1B, not from-scratch).
FULL_TEXT_CONFIG = VaneConfig(
    d_model=1536,
    n_blocks=28,
    n_heads=24,
    d_ff=6144,
    window_size=128,
    max_context=32768,
    vocab_size=128_000,
    max_levels=10,
    dropout=0.0,
    vision_proj_dim=None,
    name="vane-1.0.0-text",
)


__all__ = ["VaneConfig", "DEV_CONFIG", "FULL_TEXT_CONFIG"]
