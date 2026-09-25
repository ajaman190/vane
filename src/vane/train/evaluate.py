"""Evaluate a text checkpoint on the carved development slice."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from vane.train.yaml_config import active_model, config_dir, load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Eval Vane text checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument(
        "--config",
        type=str,
        default=str(config_dir() / "vane-text-1.0.0.yaml"),
    )
    parser.add_argument("--max-examples", type=int, default=32)
    args = parser.parse_args()

    import torch

    from vane.data.collate import collate_examples
    from vane.data.mix import load_text_mix
    from vane.eval.metrics import evaluate_batch
    from vane.model import VaneConfig, VaneModel
    from vane.train import resolve_device
    from vane.train.losses import choice_loss, noul_loss, score_loss

    config = load_config(args.config)
    device = resolve_device(config.get("training", {}).get("device", "auto"))

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    mcfg = ckpt.get("config", {}).get("model") or active_model(config)
    model_cfg = VaneConfig(
        d_model=int(mcfg.get("d_model", 64)),
        n_blocks=int(mcfg.get("n_blocks", 2)),
        n_heads=int(mcfg.get("n_heads", 4)),
        d_ff=int(mcfg.get("d_ff", 128)),
        window_size=int(mcfg.get("window_size", 128)),
        max_context=int(mcfg.get("max_context", 32768)),
        vocab_size=int(mcfg.get("vocab_size", 256)),
        max_levels=int(mcfg.get("max_levels", 5)),
        dropout=float(mcfg.get("dropout", 0.0)),
        name=str(mcfg.get("name", "vane-1.0.0-text")),
    )
    model = VaneModel(model_cfg)
    model.load_state_dict(ckpt["model_state"], strict=False)
    model.to(device)
    model.eval()

    # Held-out slice: prefer carved development; else reload mix and carve.
    mix = load_text_mix(config)
    held = []
    if mix.splits is not None and mix.splits.development:
        held = list(mix.splits.development)
    elif mix.examples:
        held = list(mix.examples)
    held = held[: args.max_examples]
    if not held:
        raise SystemExit("no held-out examples available for eval")

    print(f"Evaluating {len(held)} examples on {device}")
    print(f"checkpoint: {args.checkpoint}")

    batch = collate_examples(
        held,
        d_model=model_cfg.d_model,
        vocab_size=model_cfg.vocab_size,
        max_length=int(config.get("training", {}).get("max_seq_len", 128)),
    )
    batch = {
        k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()
    }

    with torch.no_grad():
        out = model(
            batch["input_ids"],
            attention_mask=batch.get("attention_mask"),
            noul_count=batch.get("noul_count"),
            choice_options=batch.get("choice_options"),
            choice_mask=batch.get("choice_mask"),
            score_count=batch.get("score_count"),
            score_n_levels=batch.get("score_n_levels"),
            force_full=True,
        )

    total_loss = 0.0
    n_terms = 0
    metrics: dict[str, float] = {}

    if out.noul is not None and "noul_target" in batch:
        ln = noul_loss(out.noul, batch["noul_target"])
        total_loss += float(ln)
        n_terms += 1
        pred = out.noul.detach().cpu().reshape(-1).tolist()
        tgt = batch["noul_target"].detach().cpu().reshape(-1).tolist()
        m = evaluate_batch(pred, tgt)
        metrics["noul_brier"] = m["brier"]
        metrics["noul_logloss"] = m["logloss"]
        # Binary accuracy at 0.5
        acc = sum(
            1 for p, t in zip(pred, tgt) if (p >= 0.5) == (t >= 0.5)
        ) / max(1, len(pred))
        metrics["noul_accuracy"] = acc

    if out.choice is not None and "choice_target" in batch:
        q = out.choice
        pi = batch["choice_target"]
        mask = batch.get("choice_mask")
        B, Qc, K = q.shape
        correct = 0
        total = 0
        brier_sum = 0.0
        for b in range(B):
            for c in range(Qc):
                if mask is not None:
                    m = mask[b, c]
                    k = int(m.sum().item())
                    if k < 2:
                        continue
                    probs = q[b, c, :k]
                    target = pi[b, c, :k]
                else:
                    probs = q[b, c]
                    target = pi[b, c]
                pred_i = int(probs.argmax().item())
                true_i = int(target.argmax().item())
                correct += int(pred_i == true_i)
                total += 1
                brier_sum += float(((probs - target) ** 2).sum().item())
        lc = choice_loss(q.reshape(-1, K), pi.reshape(-1, K))
        total_loss += float(lc)
        n_terms += 1
        metrics["choice_accuracy"] = correct / max(1, total)
        metrics["choice_brier"] = brier_sum / max(1, total)

    if out.score is not None and "score_target" in batch:
        q = out.score
        pi = batch["score_target"]
        ls = score_loss(q.reshape(-1, q.size(-1)), pi.reshape(-1, pi.size(-1)), alpha=out.alpha.reshape(-1) if out.alpha is not None else None)
        total_loss += float(ls)
        n_terms += 1
        metrics["score_brier"] = float(
            ((q - pi) ** 2).sum(dim=-1).mean().item()
        )

    metrics["loss"] = total_loss / max(1, n_terms)
    print("\nMetrics:")
    for k, v in metrics.items():
        assert v == v, f"non-finite metric {k}"
        print(f"  {k}: {v:.6f}")

    out_path = Path(args.checkpoint).with_suffix(".eval.json")
    out_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
