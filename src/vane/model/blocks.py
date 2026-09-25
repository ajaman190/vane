"""Encoder block: windowed attention → latent pool → latent bank → FFN."""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor

from .attention import FeedForward, LatentBankAttention, LatentPool, WindowedSelfAttention


class VaneBlock(nn.Module):
    """One stack block over token windows and the latent bank."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        window_size: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.window_size = window_size
        self.window_attn = WindowedSelfAttention(d_model, n_heads, window_size, dropout)
        self.latent_pool = LatentPool(d_model, n_heads, dropout)
        self.bank_attn = LatentBankAttention(d_model, n_heads, dropout)
        self.token_ff = FeedForward(d_model, d_ff, dropout)
        self.latent_ff = FeedForward(d_model, d_ff, dropout)
        # Broadcast latents back into their windows (additive residual)
        self.latent_to_token = nn.Linear(d_model, d_model)

    def forward(
        self,
        tokens: Tensor,
        pad_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """
        Args:
            tokens: [B, N, D]
            pad_mask: [B, N] True = valid
        Returns:
            tokens [B, N, D], latents [B, C, D]
        """
        B, N, D = tokens.shape
        w = self.window_size
        c = N // w

        tokens = self.window_attn(tokens, pad_mask)
        latents = self.latent_pool(tokens, w, pad_mask)

        latent_mask = None
        if pad_mask is not None:
            latent_mask = pad_mask.view(B, c, w).any(dim=-1)

        latents = self.bank_attn(latents, latent_mask)
        latents = self.latent_ff(latents)

        # Inject bank summary back into each window's tokens
        inj = self.latent_to_token(latents)  # [B, C, D]
        inj = inj.unsqueeze(2).expand(B, c, w, D).reshape(B, N, D)
        tokens = tokens + inj
        tokens = self.token_ff(tokens)
        return tokens, latents


class LatentRefiner(nn.Module):
    """A few steps on the latent bank only. No tokens are decoded.

    Used when the early head is still interior. Cost is c² per step, not n².
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, n_steps: int) -> None:
        super().__init__()
        self.steps = nn.ModuleList(
            nn.ModuleList(
                [
                    LatentBankAttention(d_model, n_heads),
                    FeedForward(d_model, d_ff),
                ]
            )
            for _ in range(n_steps)
        )

    def forward(self, latents: Tensor, latent_mask: Tensor | None = None) -> Tensor:
        for attn, ffn in self.steps:
            latents = attn(latents, latent_mask)
            latents = ffn(latents)
        return latents
