"""Vane decision model: windowed text tower + option-set heads."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from .blocks import LatentRefiner, VaneBlock
from .confidence import confidence
from .config import VaneConfig
from .heads import PrimitiveHeads


class ContextTooLongError(ValueError):
    """Raised when tokenized state exceeds ``max_context`` (refused, not cropped)."""


@dataclass
class VaneOutput:
    """Forward outputs for a batch of typed questions."""

    noul: Tensor | None = None  # [B, Qn] P(yes)
    choice: Tensor | None = None  # [B, Qc, K] simplex
    score: Tensor | None = None  # [B, Qs, L] simplex
    alpha: Tensor | None = None  # [B, Qs] lower-component gate
    early_exit: Tensor | None = None  # [B] bool
    choice_confidence: Tensor | None = None  # [B, Qc]
    score_confidence: Tensor | None = None  # [B, Qs]
    latents: Tensor | None = None  # [B, C, D] final (or early) bank
    blocks_run: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "noul": self.noul,
            "choice": self.choice,
            "score": self.score,
            "alpha": self.alpha,
            "early_exit": self.early_exit,
            "choice_confidence": self.choice_confidence,
            "score_confidence": self.score_confidence,
            "latents": self.latents,
            "blocks_run": self.blocks_run,
        }


class VisionProjection(nn.Module):
    """Optional stub: project frozen vision patch vectors into the token stream."""

    def __init__(self, in_dim: int, d_model: int) -> None:
        super().__init__()
        self.proj = nn.Linear(in_dim, d_model)

    def forward(self, patches: Tensor) -> Tensor:
        return self.proj(patches)


class VaneModel(nn.Module):
    """Text-tower decision model (vision projection optional stub).

    Router is **not** part of this network — serve owns routing.
    """

    def __init__(self, config: VaneConfig) -> None:
        super().__init__()
        self.config = config
        d = config.d_model
        self.token_embed = nn.Embedding(config.vocab_size, d)
        # Position ids are **within-window only** (0..w-1); no global positions.
        self.window_pos_embed = nn.Embedding(config.window_size, d)
        self.blocks = nn.ModuleList(
            [
                VaneBlock(
                    d_model=d,
                    n_heads=config.n_heads,
                    d_ff=config.d_ff or 4 * d,
                    window_size=config.window_size,
                    dropout=config.dropout,
                )
                for _ in range(config.n_blocks)
            ]
        )
        self.final_norm = nn.LayerNorm(d)
        self.heads = PrimitiveHeads(
            d_model=d,
            n_heads=config.n_heads,
            d_ff=config.d_ff or 4 * d,
            max_levels=config.max_levels,
            dropout=config.dropout,
            temp_a_choice=config.temp_a_choice,
            temp_b_choice=config.temp_b_choice,
            temp_a_score=config.temp_a_score,
            temp_b_score=config.temp_b_score,
        )
        # Cheap early-exit head: share architecture, separate weights
        self.early_heads = PrimitiveHeads(
            d_model=d,
            n_heads=config.n_heads,
            d_ff=config.d_ff or 4 * d,
            max_levels=config.max_levels,
            dropout=config.dropout,
            temp_a_choice=config.temp_a_choice,
            temp_b_choice=config.temp_b_choice,
            temp_a_score=config.temp_a_score,
            temp_b_score=config.temp_b_score,
        )
        self.refiner = (
            LatentRefiner(d, config.n_heads, config.d_ff or 4 * d, config.n_refine)
            if config.n_refine > 0
            else None
        )
        self.input_proj = (
            nn.Linear(config.backbone_dim, d) if config.backbone_dim not in (None, d) else None
        )
        self.vision_proj: VisionProjection | None = None
        if config.vision_proj_dim is not None:
            self.vision_proj = VisionProjection(config.vision_proj_dim, d)

        self.tau_choice = config.tau_choice
        self.tau_score = config.tau_score

    def _refuse_if_too_long(self, n: int) -> None:
        if n > self.config.max_context:
            raise ContextTooLongError(
                f"context length {n} exceeds max_context={self.config.max_context}; "
                "refusing (not cropping)"
            )

    def _pad_to_windows(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        """Pad sequence length to a multiple of ``window_size``."""
        B, N = input_ids.shape
        self._refuse_if_too_long(N)
        w = self.config.window_size
        pad = (w - (N % w)) % w
        if pad:
            input_ids = nn.functional.pad(input_ids, (0, pad), value=0)
            if attention_mask is None:
                attention_mask = torch.ones(B, N, device=input_ids.device, dtype=torch.bool)
                attention_mask = nn.functional.pad(attention_mask, (0, pad), value=False)
            else:
                attention_mask = nn.functional.pad(
                    attention_mask.bool(), (0, pad), value=False
                )
        else:
            if attention_mask is None:
                attention_mask = torch.ones(B, N, device=input_ids.device, dtype=torch.bool)
            else:
                attention_mask = attention_mask.bool()
        return input_ids, attention_mask

    def embed_tokens(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        vision_patches: Tensor | None = None,
        inputs_embeds: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Embed and pad. Returns ``(tokens [B,N,D], pad_mask [B,N])``.

        ``inputs_embeds`` is a frozen encoder's hidden states. Token-token
        attention stays windowed. The learned ``token_embed`` is unused then.
        """
        if inputs_embeds is not None:
            if inputs_embeds.size(1) > self.config.max_context:
                self._refuse_if_too_long(inputs_embeds.size(1))
            tok = inputs_embeds
            if self.input_proj is not None:
                tok = self.input_proj(tok)
            B, N, _ = tok.shape
            w = self.config.window_size
            if attention_mask is None:
                pad_mask = torch.ones(B, N, device=tok.device, dtype=torch.bool)
            else:
                pad_mask = attention_mask.to(dtype=torch.bool)
            pad = (w - (N % w)) % w
            if pad:
                tok = nn.functional.pad(tok, (0, 0, 0, pad))
                pad_mask = nn.functional.pad(pad_mask, (0, pad), value=False)
            pos_ids = torch.arange(w, device=tok.device).repeat(tok.size(1) // w)
            tok = tok + self.window_pos_embed(pos_ids)[None, :, :]
            return tok, pad_mask

        input_ids, pad_mask = self._pad_to_windows(input_ids, attention_mask)
        B, N = input_ids.shape
        w = self.config.window_size
        tok = self.token_embed(input_ids)
        # Within-window positions only (no global / absolute sequence positions)
        pos_ids = torch.arange(w, device=input_ids.device).repeat(N // w)
        tok = tok + self.window_pos_embed(pos_ids)[None, :, :]

        if vision_patches is not None:
            if self.vision_proj is None:
                raise ValueError("vision_patches provided but vision_proj_dim is not configured")
            # Prepend projected patches (still subject to max_context on total length)
            v = self.vision_proj(vision_patches)  # [B, P, D]
            P = v.size(1)
            self._refuse_if_too_long(N + P)
            tok = torch.cat([v, tok], dim=1)
            v_mask = torch.ones(B, P, device=tok.device, dtype=torch.bool)
            pad_mask = torch.cat([v_mask, pad_mask], dim=1)
            # Re-pad to window multiple after prepend
            Np = tok.size(1)
            pad = (w - (Np % w)) % w
            if pad:
                tok = nn.functional.pad(tok, (0, 0, 0, pad))
                pad_mask = nn.functional.pad(pad_mask, (0, pad), value=False)
        return tok, pad_mask

    def encode(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        vision_patches: Tensor | None = None,
        n_blocks: int | None = None,
        inputs_embeds: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Run encoder blocks.

        Returns:
            tokens, latents, pad_mask
        """
        tokens, pad_mask = self.embed_tokens(
            input_ids, attention_mask, vision_patches, inputs_embeds=inputs_embeds
        )
        n_run = self.config.n_blocks if n_blocks is None else n_blocks
        latents = None
        for i in range(n_run):
            tokens, latents = self.blocks[i](tokens, pad_mask)
        assert latents is not None
        tokens = self.final_norm(tokens)
        return tokens, latents, pad_mask

    def _latent_mask(self, pad_mask: Tensor) -> Tensor:
        B, N = pad_mask.shape
        w = self.config.window_size
        return pad_mask.view(B, N // w, w).any(dim=-1)

    def _run_heads(
        self,
        heads: PrimitiveHeads,
        latents: Tensor,
        latent_mask: Tensor,
        *,
        noul_queries: Tensor | None,
        choice_options: Tensor | None,
        choice_mask: Tensor | None,
        score_queries: Tensor | None,
        score_n_levels: int | None,
        tokens: Tensor | None = None,
        token_mask: Tensor | None = None,
    ) -> dict[str, Tensor | None]:
        out: dict[str, Tensor | None] = {
            "noul": None,
            "choice": None,
            "score": None,
            "alpha": None,
            "choice_confidence": None,
            "score_confidence": None,
        }
        B = latents.size(0)

        if noul_queries is not None:
            if noul_queries.dim() == 2:
                noul_queries = noul_queries.unsqueeze(0).expand(B, -1, -1)
            out["noul"] = heads.noul(
                noul_queries, latents, latent_mask, tokens=tokens, token_mask=token_mask
            )

        if choice_options is not None:
            # choice_options: [B, Qc, K, D] or [B, K, D]
            if choice_options.dim() == 3:
                choice_options = choice_options.unsqueeze(1)
            B_, Qc, K, D = choice_options.shape
            flat_opt = choice_options.reshape(B_ * Qc, K, D)
            flat_mask = None
            if choice_mask is not None:
                if choice_mask.dim() == 2:
                    choice_mask = choice_mask.unsqueeze(1)
                flat_mask = choice_mask.reshape(B_ * Qc, K)
            lat_exp = latents.unsqueeze(1).expand(B_, Qc, latents.size(1), D).reshape(
                B_ * Qc, latents.size(1), D
            )
            lm = latent_mask.unsqueeze(1).expand(B_, Qc, -1).reshape(B_ * Qc, -1)
            # T(type, k) and C(q) use the TRUE option count, not padded width.
            if flat_mask is not None:
                true_k = flat_mask.sum(dim=-1).clamp(min=2)  # [B*Qc]
                T = heads.choice_temperature(true_k)  # type: ignore[arg-type]
            else:
                T = heads.choice_temperature(K)
            probs = heads.choice(
                flat_opt,
                lat_exp,
                flat_mask,
                lm,
                temperature=T,
                tokens=None if tokens is None else tokens.unsqueeze(1).expand(
                    B_, Qc, tokens.size(1), D
                ).reshape(B_ * Qc, tokens.size(1), D),
                token_mask=None if token_mask is None else token_mask.unsqueeze(1).expand(
                    B_, Qc, -1
                ).reshape(B_ * Qc, token_mask.size(-1)),
            )  # [B*Qc, K]
            probs = probs.view(B_, Qc, K)
            out["choice"] = probs
            if flat_mask is not None:
                out["choice_confidence"] = confidence(
                    probs, mask=flat_mask.view(B_, Qc, K)
                )
            else:
                out["choice_confidence"] = confidence(probs)

        if score_queries is not None:
            L = score_n_levels if score_n_levels is not None else 3
            if score_queries.dim() == 2:
                score_queries = score_queries.unsqueeze(0).expand(B, -1, -1)
            # Cardinality temperature / C(q) on the TRUE L (caller must pass
            # real level count, not a padded phantom width).
            T = heads.score_temperature(L)
            probs, alpha = heads.score(
                score_queries,
                latents,
                L,
                latent_mask,
                temperature=T,
                tokens=tokens,
                token_mask=token_mask,
            )
            out["score"] = probs
            out["alpha"] = alpha
            out["score_confidence"] = confidence(probs, k=L)

        return out

    def _early_exit_ok(
        self,
        head_out: dict[str, Tensor | None],
        *,
        has_noul: bool = False,  # noqa: ARG002 — noul never uses C; kept for call-site clarity
    ) -> Tensor:
        """Per-batch exit: every choice/score question must clear τ.

        Noul does not use C; noul questions never block exit on C.
        If there are **no** choice/score questions, exit is allowed (vacuous).
        One interior choice/score blocks exit for the whole request.
        """
        B = None
        clears: list[Tensor] = []

        cc = head_out.get("choice_confidence")
        if cc is not None:
            B = cc.size(0)
            # all questions in batch item must clear
            clears.append((cc >= self.tau_choice).all(dim=-1))

        sc = head_out.get("score_confidence")
        if sc is not None:
            B = sc.size(0)
            clears.append((sc >= self.tau_score).all(dim=-1))

        if not clears:
            # Only noul (or empty): allow early exit
            if B is None:
                # Infer batch from noul
                noul = head_out.get("noul")
                if noul is not None:
                    return torch.ones(noul.size(0), dtype=torch.bool, device=noul.device)
                return torch.ones(1, dtype=torch.bool)
            return torch.ones(B, dtype=torch.bool)

        ok = clears[0]
        for c in clears[1:]:
            ok = ok & c
        return ok

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        *,
        noul_queries: Tensor | None = None,
        noul_count: int | None = None,
        choice_options: Tensor | None = None,
        choice_mask: Tensor | None = None,
        score_queries: Tensor | None = None,
        score_count: int | None = None,
        score_n_levels: int | None = None,
        vision_patches: Tensor | None = None,
        inputs_embeds: Tensor | None = None,
        use_early_exit: bool = True,
        force_full: bool = False,
        force_early: bool = False,
    ) -> VaneOutput:
        """Encode state once; answer all typed questions.

        Args:
            input_ids: ``[B, N]`` token ids. ``N > max_context`` raises
                :class:`ContextTooLongError` (no crop).
            noul_queries: ``[B, Qn, D]`` or ``[Qn, D]``. If omitted and
                ``noul_count`` set, uses learned default query.
            choice_options: ``[B, Qc, K, D]`` option description vectors
                (no position ids in the set head).
            score_queries: ``[B, Qs, D]`` or use ``score_count``.
            use_early_exit: after block 1, exit iff every choice/score clears τ.
            force_full: always run all blocks (serve τ ignored).
            force_early: training coin — stop after block 1 and apply the
                early head; serve-time τ is not consulted.
        """
        B = input_ids.size(0)
        device = input_ids.device
        d = self.config.d_model

        if force_early and force_full:
            raise ValueError("force_early and force_full are mutually exclusive")

        if noul_queries is None and noul_count:
            noul_queries = self.heads.default_noul_query.expand(noul_count, d)
        if score_queries is None and score_count:
            score_queries = self.heads.default_score_query.expand(score_count, d)

        # --- Block 1 + early head ---
        tokens, pad_mask = self.embed_tokens(
            input_ids, attention_mask, vision_patches, inputs_embeds=inputs_embeds
        )
        tokens, latents = self.blocks[0](tokens, pad_mask)
        latent_mask = self._latent_mask(pad_mask)

        early_out = self._run_heads(
            self.early_heads,
            latents,
            latent_mask,
            noul_queries=noul_queries,
            choice_options=choice_options,
            choice_mask=choice_mask,
            score_queries=score_queries,
            score_n_levels=score_n_levels,
            tokens=tokens,
            token_mask=pad_mask,
        )
        exit_ok = self._early_exit_ok(early_out, has_noul=noul_queries is not None)

        if force_early:
            # Training path: coin says stop after block 1; τ is serve-only.
            tokens = self.final_norm(tokens)
            return VaneOutput(
                noul=early_out["noul"],
                choice=early_out["choice"],
                score=early_out["score"],
                alpha=early_out["alpha"],
                early_exit=torch.ones(B, dtype=torch.bool, device=device),
                choice_confidence=early_out["choice_confidence"],
                score_confidence=early_out["score_confidence"],
                latents=latents,
                blocks_run=1,
            )

        if force_full or not use_early_exit:
            exit_ok = torch.zeros(B, dtype=torch.bool, device=device)

        # Per-batch: if ANY item needs full stack, we run full for the batch
        # (per-question encoder skip would encode state twice).
        all_exit = bool(exit_ok.all().item()) if exit_ok.numel() else False
        blocks_run = 1

        if all_exit and self.config.n_blocks > 1:
            tokens = self.final_norm(tokens)
            return VaneOutput(
                noul=early_out["noul"],
                choice=early_out["choice"],
                score=early_out["score"],
                alpha=early_out["alpha"],
                early_exit=exit_ok,
                choice_confidence=early_out["choice_confidence"],
                score_confidence=early_out["score_confidence"],
                latents=latents,
                blocks_run=1,
            )

        # Run remaining blocks
        for i in range(1, self.config.n_blocks):
            tokens, latents = self.blocks[i](tokens, pad_mask)
            blocks_run += 1
        tokens = self.final_norm(tokens)
        latent_mask = self._latent_mask(pad_mask)
        if self.refiner is not None:
            latents = self.refiner(latents, latent_mask)

        full_out = self._run_heads(
            self.heads,
            latents,
            latent_mask,
            noul_queries=noul_queries,
            choice_options=choice_options,
            choice_mask=choice_mask,
            score_queries=score_queries,
            score_n_levels=score_n_levels,
            tokens=tokens,
            token_mask=pad_mask,
        )
        return VaneOutput(
            noul=full_out["noul"],
            choice=full_out["choice"],
            score=full_out["score"],
            alpha=full_out["alpha"],
            early_exit=torch.zeros(B, dtype=torch.bool, device=device),  # full path taken
            choice_confidence=full_out["choice_confidence"],
            score_confidence=full_out["score_confidence"],
            latents=latents,
            blocks_run=blocks_run,
        )
