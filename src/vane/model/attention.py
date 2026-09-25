"""Windowed attention and latent-bank attention."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class MultiHeadAttention(nn.Module):
    """Standard MHA used for window, bank, and cross-attention."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        key_padding_mask: Tensor | None = None,
    ) -> Tensor:
        """
        Args:
            query: [B, Tq, D]
            key/value: [B, Tk, D]
            key_padding_mask: [B, Tk] True = valid (attend), False = pad
        """
        B, Tq, _ = query.shape
        Tk = key.size(1)
        q = self.q_proj(query).view(B, Tq, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(B, Tk, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(B, Tk, self.n_heads, self.head_dim).transpose(1, 2)

        scale = 1.0 / math.sqrt(self.head_dim)
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale  # [B, H, Tq, Tk]
        if key_padding_mask is not None:
            # mask shape [B, Tk] -> [B, 1, 1, Tk]
            attn = attn.masked_fill(~key_padding_mask[:, None, None, :], float("-inf"))
        weights = F.softmax(attn, dim=-1)
        weights = torch.nan_to_num(weights, nan=0.0)
        weights = self.dropout(weights)
        out = torch.matmul(weights, v)  # [B, H, Tq, Hd]
        out = out.transpose(1, 2).contiguous().view(B, Tq, -1)
        return self.out_proj(out)


class WindowedSelfAttention(nn.Module):
    """Full attention inside non-overlapping windows of width ``window_size``."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        window_size: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.window_size = window_size
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: Tensor, pad_mask: Tensor | None = None) -> Tensor:
        """
        Args:
            x: [B, N, D] with N multiple of window_size
            pad_mask: [B, N] True = valid token
        """
        B, N, D = x.shape
        w = self.window_size
        if N % w != 0:
            raise ValueError(f"sequence length {N} must be multiple of window_size {w}")
        c = N // w
        residual = x
        x = self.norm(x)
        # [B, C, W, D] -> [B*C, W, D]
        xw = x.view(B, c, w, D).reshape(B * c, w, D)
        mask_w = None
        if pad_mask is not None:
            mask_w = pad_mask.view(B, c, w).reshape(B * c, w)
        out = self.attn(xw, xw, xw, key_padding_mask=mask_w)
        out = out.view(B, c, w, D).reshape(B, N, D)
        return residual + out


class LatentPool(nn.Module):
    """One latent per window via attention pooling with a learned query."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: Tensor, window_size: int, pad_mask: Tensor | None = None) -> Tensor:
        """
        Args:
            x: [B, N, D]
        Returns:
            latents [B, C, D]
        """
        B, N, D = x.shape
        w = window_size
        c = N // w
        x = self.norm(x)
        xw = x.view(B, c, w, D).reshape(B * c, w, D)
        q = self.query.expand(B * c, 1, D)
        mask_w = None
        if pad_mask is not None:
            mask_w = pad_mask.view(B, c, w).reshape(B * c, w)
        lat = self.attn(q, xw, xw, key_padding_mask=mask_w)  # [B*C, 1, D]
        return lat.view(B, c, D)


class LatentBankAttention(nn.Module):
    """Latents attend to each other (full bank attention)."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, latents: Tensor, latent_mask: Tensor | None = None) -> Tensor:
        """
        Args:
            latents: [B, C, D]
            latent_mask: [B, C] True = window has ≥1 real token
        """
        residual = latents
        h = self.norm(latents)
        out = self.attn(h, h, h, key_padding_mask=latent_mask)
        return residual + out


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + self.net(x)
