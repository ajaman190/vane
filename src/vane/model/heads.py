"""Option-set head and primitive readouts (noul / choice / score)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .attention import MultiHeadAttention
from .confidence import cardinality_temperature


class OptionSetEncoder(nn.Module):
    """Two-layer set transformer with **no** position embeddings.

    Permuting options permutes outputs and nothing else.
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.0) -> None:
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=2, enable_nested_tensor=False)
        self.to_logit = nn.Linear(d_model, 1)

    def forward(self, option_states: Tensor, option_mask: Tensor | None = None) -> Tensor:
        """
        Args:
            option_states: [B, K, D]
            option_mask: [B, K] True = valid option
        Returns:
            logits [B, K]
        """
        src_key_padding_mask = None
        if option_mask is not None:
            # nn.TransformerEncoder: True = ignore
            src_key_padding_mask = ~option_mask
        h = self.encoder(option_states, src_key_padding_mask=src_key_padding_mask)
        logits = self.to_logit(h).squeeze(-1)
        if option_mask is not None:
            # Use a large negative (not -inf) so softmax stays finite on MPS.
            logits = logits.masked_fill(~option_mask, -1.0e4)
        return logits


class OptionCrossAttend(nn.Module):
    """Cross-attend option description vectors to state latents."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)

    def forward(
        self,
        options: Tensor,
        latents: Tensor,
        latent_mask: Tensor | None = None,
    ) -> Tensor:
        """
        Args:
            options: [B, K, D]
            latents: [B, C, D]
        Returns:
            [B, K, D]
        """
        q = self.norm_q(options)
        kv = self.norm_kv(latents)
        return options + self.attn(q, kv, kv, key_padding_mask=latent_mask)


def corn_from_conditionals(f: Tensor) -> Tensor:
    """Convert CORN conditional probs ``f`` [..., L-1] into level masses ``[..., L]``.

    f[..., r] = P(Y > r | Y > r-1) with Y > -1 always.
    Survival S_r = P(Y > r) = prod_{j=0}^{r} f[..., j]
    """
    # survival after each conditional: [..., L-1]
    log_f = torch.log(f.clamp(min=1e-8))
    log_surv = torch.cumsum(log_f, dim=-1)
    surv = torch.exp(log_surv)  # P(Y>0), P(Y>1), ..., P(Y>L-2)

    ones = torch.ones(*surv.shape[:-1], 1, device=f.device, dtype=f.dtype)
    zeros = torch.zeros(*surv.shape[:-1], 1, device=f.device, dtype=f.dtype)
    # P(Y > -1)=1, then surv, then P(Y > L-1)=0
    surv_ext = torch.cat([ones, surv, zeros], dim=-1)  # [..., L+1]
    probs = surv_ext[..., :-1] - surv_ext[..., 1:]  # [..., L]
    return probs.clamp(min=0.0)


class ScoreHead(nn.Module):
    """Two-component CORN ordinal head.

    μ2 = μ1 + softplus(δ); q = α P1 + (1-α) P2.
    α weights the **lower** component.
    """

    def __init__(self, d_model: int, n_heads: int, max_levels: int = 10) -> None:
        super().__init__()
        self.max_levels = max_levels
        self.query_proj = nn.Linear(d_model, d_model)
        self.cross = MultiHeadAttention(d_model, n_heads)
        self.token_cross = MultiHeadAttention(d_model, n_heads)
        self.gate = nn.Linear(d_model, 1)
        self.mu = nn.Linear(d_model, 1)
        self.delta = nn.Linear(d_model, 1)
        # Shared CORN base logits shape [max_levels-1]; shifted by μ
        self.corn_base = nn.Parameter(torch.zeros(max_levels - 1))
        self.corn_scale = nn.Linear(d_model, max_levels - 1)

    def forward(
        self,
        query: Tensor,
        latents: Tensor,
        n_levels: int,
        latent_mask: Tensor | None = None,
        temperature: Tensor | float = 1.0,
        tokens: Tensor | None = None,
        token_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """
        Args:
            query: [B, D] or [B, Q, D]
            latents: [B, C, D]
            n_levels: L in [2, max_levels]
            temperature: cardinality temperature T(type, L) applied to CORN
                logits before the sigmoid (same role as softmax(z/T) on choice).
        Returns:
            probs [B, L] or [B, Q, L], alpha [B] or [B, Q]
        """
        squeeze_q = False
        if query.dim() == 2:
            query = query.unsqueeze(1)
            squeeze_q = True
        B, Q, D = query.shape
        q = self.query_proj(query)  # [B, Q, D]
        # Cross-attend each question query to latents
        flat_q = q.reshape(B * Q, 1, D)
        lat = latents.unsqueeze(1).expand(B, Q, latents.size(1), D).reshape(B * Q, latents.size(1), D)
        mask = None
        if latent_mask is not None:
            mask = latent_mask.unsqueeze(1).expand(B, Q, -1).reshape(B * Q, -1)
        h = self.cross(flat_q, lat, lat, key_padding_mask=mask).squeeze(1)  # [B*Q, D]
        h = h.view(B, Q, D)
        if tokens is not None:
            flat_h = h.reshape(B * Q, 1, D)
            tok = tokens.unsqueeze(1).expand(B, Q, tokens.size(1), D).reshape(
                B * Q, tokens.size(1), D
            )
            tmask = None
            if token_mask is not None:
                tmask = token_mask.unsqueeze(1).expand(B, Q, -1).reshape(B * Q, -1)
            h = h + self.token_cross(flat_h, tok, tok, key_padding_mask=tmask).view(B, Q, D)

        alpha = torch.sigmoid(self.gate(h)).squeeze(-1)  # [B, Q]
        mu1 = self.mu(h).squeeze(-1)
        mu2 = mu1 + F.softplus(self.delta(h).squeeze(-1))
        scale = self.corn_scale(h)[..., : n_levels - 1]  # [B, Q, L-1]
        base = self.corn_base[: n_levels - 1]

        if isinstance(temperature, Tensor):
            t = temperature.clamp(min=1e-4)
            while t.dim() < 1:
                t = t.view(1)
        else:
            t = max(float(temperature), 1e-4)

        def component_probs(mu: Tensor) -> Tensor:
            # Larger μ → higher levels: ADD μ to CORN logits, then / T.
            # Subtracting μ would move mass DOWN, so μ2=μ1+softplus(δ) would
            # become the *lower* pole and α (on μ1) would weight the higher one.
            # With +, μ2 is the higher component and α weights the lower (μ1),
            # matching docs/architecture.md.
            logits = (base + scale + mu.unsqueeze(-1)) / t
            f = torch.sigmoid(logits)
            return corn_from_conditionals(f)

        p1 = component_probs(mu1)
        p2 = component_probs(mu2)
        # α weights the lower component (μ1); (1-α) weights the higher (μ2).
        probs = alpha.unsqueeze(-1) * p1 + (1.0 - alpha.unsqueeze(-1)) * p2
        probs = probs / probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        if squeeze_q:
            return probs.squeeze(1), alpha.squeeze(1)
        return probs, alpha


class NoulHead(nn.Module):
    """Binary yes/no via sigmoid after cross-attention to latents."""

    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        self.query_proj = nn.Linear(d_model, d_model)
        self.cross = MultiHeadAttention(d_model, n_heads)
        self.token_cross = MultiHeadAttention(d_model, n_heads)
        self.out = nn.Linear(d_model, 1)

    def forward(
        self,
        query: Tensor,
        latents: Tensor,
        latent_mask: Tensor | None = None,
        tokens: Tensor | None = None,
        token_mask: Tensor | None = None,
    ) -> Tensor:
        """
        Args:
            query: [B, Q, D]
            latents: [B, C, D]
        Returns:
            p(yes) [B, Q]
        """
        B, Q, D = query.shape
        q = self.query_proj(query)
        flat_q = q.reshape(B * Q, 1, D)
        lat = latents.unsqueeze(1).expand(B, Q, latents.size(1), D).reshape(B * Q, latents.size(1), D)
        mask = None
        if latent_mask is not None:
            mask = latent_mask.unsqueeze(1).expand(B, Q, -1).reshape(B * Q, -1)
        h = self.cross(flat_q, lat, lat, key_padding_mask=mask).squeeze(1)
        h = h.view(B, Q, D)
        if tokens is not None:
            flat_h = h.reshape(B * Q, 1, D)
            tok = tokens.unsqueeze(1).expand(B, Q, tokens.size(1), D).reshape(
                B * Q, tokens.size(1), D
            )
            tmask = None
            if token_mask is not None:
                tmask = token_mask.unsqueeze(1).expand(B, Q, -1).reshape(B * Q, -1)
            h = h + self.token_cross(flat_h, tok, tok, key_padding_mask=tmask).view(B, Q, D)
        return torch.sigmoid(self.out(h).squeeze(-1))


class ChoiceHead(nn.Module):
    """Option-set choice: cross-attend options to latents → set encoder → softmax / T."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.cross = OptionCrossAttend(d_model, n_heads, dropout)
        self.token_cross = OptionCrossAttend(d_model, n_heads, dropout)
        self.set_encoder = OptionSetEncoder(d_model, n_heads, d_ff, dropout)

    def forward(
        self,
        options: Tensor,
        latents: Tensor,
        option_mask: Tensor | None = None,
        latent_mask: Tensor | None = None,
        temperature: Tensor | float = 1.0,
        tokens: Tensor | None = None,
        token_mask: Tensor | None = None,
    ) -> Tensor:
        """
        Args:
            options: [B, K, D]
            latents: [B, C, D]
            temperature: scalar or [B]
        Returns:
            simplex [B, K]
        """
        states = self.cross(options, latents, latent_mask)
        if tokens is not None:
            states = states + self.token_cross(states, tokens, token_mask)
        logits = self.set_encoder(states, option_mask)
        if isinstance(temperature, Tensor):
            while temperature.dim() < logits.dim():
                temperature = temperature.unsqueeze(-1)
            logits = logits / temperature.clamp(min=1e-4)
        else:
            logits = logits / max(float(temperature), 1e-4)
        return F.softmax(logits, dim=-1)


class PrimitiveHeads(nn.Module):
    """Bundle of noul / choice / score heads plus temperature parameters."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        max_levels: int = 10,
        dropout: float = 0.0,
        temp_a_choice: float = 0.0,
        temp_b_choice: float = 0.0,
        temp_a_score: float = 0.0,
        temp_b_score: float = 0.0,
    ) -> None:
        super().__init__()
        self.noul = NoulHead(d_model, n_heads)
        self.choice = ChoiceHead(d_model, n_heads, d_ff, dropout)
        self.score = ScoreHead(d_model, n_heads, max_levels)
        self.temp_a_choice = nn.Parameter(torch.tensor(float(temp_a_choice)))
        self.temp_b_choice = nn.Parameter(torch.tensor(float(temp_b_choice)))
        self.temp_a_score = nn.Parameter(torch.tensor(float(temp_a_score)))
        self.temp_b_score = nn.Parameter(torch.tensor(float(temp_b_score)))
        # Learned default queries when caller does not supply question embeds
        self.default_noul_query = nn.Parameter(torch.randn(1, d_model) * 0.02)
        self.default_score_query = nn.Parameter(torch.randn(1, d_model) * 0.02)

    def choice_temperature(self, k: int | Tensor) -> Tensor:
        return cardinality_temperature(self.temp_a_choice, self.temp_b_choice, k)

    def score_temperature(self, k: int | Tensor) -> Tensor:
        return cardinality_temperature(self.temp_a_score, self.temp_b_score, k)
