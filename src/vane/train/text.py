"""Train from a YAML in vane.train.configs.

``uv run vane-train --config <file>`` runs one full epoch on CUDA when an
NVIDIA GPU is present. The backbone tokenizer and frozen encoder are required.
The hash tokenizer is only for unit tests of the collate helper.
"""
from __future__ import annotations

import argparse
import json
import random

from vane.train.yaml_config import (
    active_model,
    checkpoint_dir,
    config_dir,
    load_config,
)


def _save_ckpt(ckpt_dir, step, model, mcfg, tcfg, config, losses, device, corpus_in_batches) -> None:
    import torch

    ckpt_path = ckpt_dir / f"step_{step:06d}.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": {"model": mcfg, "training": tcfg, "data": config.get("data", {})},
            "step": step,
            "losses": losses,
            "device": str(device),
            "batch_corpora": corpus_in_batches,
        },
        ckpt_path,
    )
    print(f"  wrote checkpoint {ckpt_path}")


def _train_batch(
    batch_ex, model, optim, encoder, vision, model_cfg, max_seq, device,
    early_prob, grad_clip, rng, corpus_in_batches,
):
    import torch

    from vane.data.collate import collate_examples, state_to_text
    from vane.train import one_train_step

    for ex in batch_ex:
        if ex.corpus:
            corpus_in_batches[ex.corpus] = corpus_in_batches.get(ex.corpus, 0) + 1
    batch = collate_examples(
        batch_ex,
        d_model=model_cfg.d_model,
        vocab_size=model_cfg.vocab_size,
        max_length=max_seq,
    )
    if encoder is not None:
        hidden, mask = encoder.embed_texts([state_to_text(ex.state) for ex in batch_ex], max_seq)
        batch["inputs_embeds"] = hidden
        batch["attention_mask"] = mask
        batch["input_ids"] = torch.zeros(hidden.size(0), hidden.size(1), dtype=torch.long)
    if vision is not None:
        images = []
        any_image = False
        for ex in batch_ex:
            if ex.images:
                images.append(ex.images[0])
                any_image = True
            else:
                images.append(None)
        if any_image:
            batch["vision_patches"] = vision.embed_images(images)
    return one_train_step(
        model,
        batch,
        optim,
        force_full=True,
        early_exit_train_prob=early_prob,
        device=device,
        rng=rng,
        gradient_clip=grad_clip,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Vane text checkpoint")
    parser.add_argument(
        "--config",
        type=str,
        default=str(config_dir() / "vane-text-1.0.0.yaml"),
    )
    parser.add_argument("--steps", type=int, default=None, help="Override max_steps")
    args = parser.parse_args()

    config = load_config(args.config)
    seed = int(config.get("training", {}).get("seed", config.get("data", {}).get("seed", 0)))
    random.seed(seed)

    mcfg = active_model(config)
    print("=" * 60)
    print("Vane training")
    print("=" * 60)
    print(f"config:  {args.config}")
    print(f"Select a YAML with:  uv run vane-train --config <path>")

    import torch

    from vane.data.collate import collate_examples
    from vane.model import VaneConfig, VaneModel
    from vane.train import one_train_step, resolve_device

    device = resolve_device(config.get("training", {}).get("device", "auto"))
    print(f"device:  {device}")
    print(f"model:   {config.get('model', {}).get('name')}  profile={config.get('model', {}).get('profile')}")
    print(f"backbone: {(config.get('model', {}).get('backbone') or {}).get('repo')} (load={ (config.get('model', {}).get('backbone') or {}).get('load') })")

    backbone_cfg = config.get("model", {}).get("backbone") or {}
    if not backbone_cfg.get("load"):
        raise SystemExit(
            "This config does not load a backbone. Training uses the backbone tokenizer "
            "and a frozen encoder. Set backbone.load: true. "
            "There is no hash-tokenizer training path."
        )

    backbone = mcfg.get("backbone") or {}
    encoder = None
    if backbone.get("load"):
        from vane.model.backbone import FrozenEncoder

        repo = str(backbone.get("repo") or "")
        if not repo:
            raise SystemExit("backbone.load is true but backbone.repo is empty")
        print(f"loading frozen encoder {repo}")
        encoder = FrozenEncoder.load(repo, device)
        mcfg["backbone_dim"] = encoder.dim
        print(f"frozen encoder dim {encoder.dim}; its weights are not trained")
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
        tau_choice=float(mcfg.get("tau_choice", 0.95)),
        tau_score=float(mcfg.get("tau_score", 0.95)),
        vision_proj_dim=mcfg.get("vision_proj_dim"),
        n_refine=int(mcfg.get("n_refine", 2)),
        backbone_dim=mcfg.get("backbone_dim"),
        name=str(mcfg.get("name", "vane-1.0.0-text")),
    )
    torch.manual_seed(seed)
    model = VaneModel(model_cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"parameters: {n_params:,}")

    tcfg = config.get("training", {})
    epochs = int(tcfg.get("epochs") or 1)
    step_cap = args.steps if args.steps is not None else tcfg.get("max_steps")
    max_steps = int(step_cap) if step_cap not in (None, "", "null") else None
    batch_size = int(tcfg.get("batch_size", 2))
    lr = float(tcfg.get("lr", 1e-3))
    early_prob = float(tcfg.get("early_exit_train_prob", 0.25))
    ckpt_every = int(tcfg.get("checkpoint_every", 4))
    max_seq = int(tcfg.get("max_seq_len", 128))

    vision = None
    vision_cfg = config.get("model", {}).get("vision") or {}
    if mcfg.get("modality") == "vision":
        if not vision_cfg.get("load", True):
            raise SystemExit("vision training requires model.vision.load: true")
        from vane.model.backbone import FrozenVision
        from vane.model.vane import VisionProjection

        vision = FrozenVision.load(str(vision_cfg["repo"]), device)
        print(f"frozen vision {vision.repo} dim {vision.dim}")
        model.vision_proj = VisionProjection(vision.dim, model_cfg.d_model).to(device)
        model.config.vision_proj_dim = vision.dim

    weight_decay = float(tcfg.get("weight_decay", 0.0))
    grad_clip = float(tcfg.get("gradient_clip", 1.0))
    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    ckpt_dir = checkpoint_dir(config)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Snapshot a parameter to prove learning.
    probe_param = next(model.parameters())
    with torch.no_grad():
        probe0 = probe_param.detach().float().abs().mean().item()

    rng = random.Random(seed)
    losses: list[float] = []
    corpus_in_batches: dict[str, int] = {}

    print(f"\nTraining {epochs} epoch(s) on {device}" + (f", cap {max_steps} steps" if max_steps else " (full pass)"))
    from vane.data.mix import iter_train_examples

    step = 0
    calib_rows: list = []
    calib_cap = int((config.get("data") or {}).get("calibration_size", 2048))
    for epoch in range(1, epochs + 1):
        buf: list = []
        for ex in iter_train_examples(config):
            if len(calib_rows) < calib_cap and rng.random() < float((config.get("data") or {}).get("calibration_fraction", 0.02)):
                calib_rows.append(ex)
                continue
            buf.append(ex)
            if len(buf) < batch_size:
                continue
            step += 1
            parts = _train_batch(
                buf, model, optim, encoder, vision, model_cfg, max_seq, device,
                early_prob, grad_clip, rng, corpus_in_batches,
            )
            buf = []
            losses.append(parts["loss"])
            print(
                f"  epoch {epoch} step {step:06d}  loss={parts['loss']:.6f}  "
                f"early={int(parts.get('early', 0))}  blocks={int(parts.get('blocks_run', 0))}"
            )
            if parts["loss"] != parts["loss"]:
                raise SystemExit("non-finite loss — aborting")
            if step % ckpt_every == 0:
                _save_ckpt(ckpt_dir, step, model, mcfg, tcfg, config, losses, device, corpus_in_batches)
            if max_steps is not None and step >= max_steps:
                break
        if max_steps is not None and step >= max_steps:
            break
    if step == 0:
        raise SystemExit("No training rows. Check corpus ids and network access.")
    _save_ckpt(ckpt_dir, step, model, mcfg, tcfg, config, losses, device, corpus_in_batches)

    with torch.no_grad():
        probe1 = probe_param.detach().float().abs().mean().item()
    print(f"\nparam mean-abs before={probe0:.8f} after={probe1:.8f}  Δ={probe1 - probe0:.8e}")
    print("Corpora seen in train batches:")
    for cid, n in sorted(corpus_in_batches.items(), key=lambda x: -x[1]):
        print(f"  {cid}: {n}")

    temperatures: dict[str, float] = {}
    calib = calib_rows
    if calib:
        from vane.train.calibration import fit_cardinality_temperature

        calib_batch = collate_examples(
            calib[:32],
            d_model=model_cfg.d_model,
            vocab_size=model_cfg.vocab_size,
            max_length=max_seq,
        )
        calib_batch = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in calib_batch.items()
        }
        model.eval()
        with torch.no_grad():
            calib_out = model(
                calib_batch["input_ids"],
                attention_mask=calib_batch.get("attention_mask"),
                noul_count=calib_batch.get("noul_count"),
                choice_options=calib_batch.get("choice_options"),
                choice_mask=calib_batch.get("choice_mask"),
                score_count=calib_batch.get("score_count"),
                score_n_levels=calib_batch.get("score_n_levels"),
                force_full=True,
            )
        groups = {}
        if calib_out.noul is not None and "noul_target" in calib_batch:
            groups["noul"] = (calib_out.noul.detach().cpu(), calib_batch["noul_target"].detach().cpu())
        if calib_out.choice is not None and "choice_target" in calib_batch:
            groups["choice"] = (
                calib_out.choice.detach().cpu(),
                calib_batch["choice_target"].detach().cpu(),
            )
        if groups:
            temperatures = fit_cardinality_temperature(groups)
            from vane.train.calibration import fit_temperature_parameters

            fitted_ab = fit_temperature_parameters(model, calib_batch, steps=20)
            (ckpt_dir / "temperature.json").write_text(
                json.dumps(
                    {
                        "scalar_temperature": temperatures,
                        "cardinality": fitted_ab,
                        "split": "calibration",
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            print(f"calibration temperatures: {temperatures} cardinality={fitted_ab}")

    meta_path = ckpt_dir / "last_run.json"
    meta_path.write_text(
        json.dumps(
            {
                "losses": losses,
                "checkpoint_dir": str(ckpt_dir),
                "device": str(device),
                "yielded": corpus_in_batches,
                "batch_corpora": corpus_in_batches,
                "param_delta": probe1 - probe0,
                "scalar_temperature": temperatures,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"run metadata → {meta_path}")
    print("done.")


if __name__ == "__main__":
    main()
