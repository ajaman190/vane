"""Tests for vane.load — local checkpoints only. No network."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("safetensors")


def _write_tiny_checkpoint(directory: Path) -> None:
    """Write config.json + a tiny model.safetensors under *directory*."""
    config = {
        "name": "vane-smoke-text",
        "modality": "text",
        "d_model": 8,
        "n_blocks": 1,
        "n_heads": 2,
        "d_ff": 16,
        "window_size": 8,
        "max_context": 64,
        "vocab_size": 32,
        "max_levels": 3,
        "dropout": 0.0,
    }
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )

    try:
        import torch
        from safetensors.torch import save_file

        tensors = {
            "token_embed.weight": torch.zeros(32, 8),
            "window_pos_embed.weight": torch.zeros(8, 8),
        }
        save_file(tensors, str(directory / "model.safetensors"))
        return
    except ImportError:
        pass

    import numpy as np
    from safetensors.numpy import save_file as save_numpy

    tensors_np = {
        "token_embed.weight": np.zeros((32, 8), dtype=np.float32),
        "window_pos_embed.weight": np.zeros((8, 8), dtype=np.float32),
    }
    save_numpy(tensors_np, str(directory / "model.safetensors"))


def test_import_vane_does_not_pull_torch() -> None:
    """``import vane`` must stay torch-free when torch was not already loaded."""
    for key in list(sys.modules):
        if key == "vane" or key.startswith("vane."):
            del sys.modules[key]

    # Only assert torch-free if torch was not already imported by the session.
    torch_already = "torch" in sys.modules
    import vane

    assert vane.__version__ == "0.1.0"
    if not torch_already:
        assert "torch" not in sys.modules
        assert "vane.load" not in sys.modules
        assert "vane.client" not in sys.modules
    # Lazy attribute resolves without requiring torch for the load *module* import
    # (load itself may import torch later only when reading weights with torch).
    assert callable(getattr(vane, "load"))


def test_load_local_temp_checkpoint(tmp_path: Path) -> None:
    from vane.load import LoadedCheckpoint, load

    ckpt_dir = tmp_path / "smoke-text"
    _write_tiny_checkpoint(ckpt_dir)

    loaded = load(ckpt_dir, modality="text")
    assert isinstance(loaded, LoadedCheckpoint)
    assert loaded.config["name"] == "vane-smoke-text"
    assert loaded.config["d_model"] == 8
    assert set(loaded.state_dict.keys()) >= {
        "token_embed.weight",
        "window_pos_embed.weight",
    }
    assert loaded.modality == "text"
    assert loaded.path is not None


def test_load_resolves_modality_subdir(tmp_path: Path) -> None:
    from vane.load import load

    root = tmp_path / "repo"
    _write_tiny_checkpoint(root / "text")
    vision = root / "vision"
    vision.mkdir(parents=True)
    (vision / "config.json").write_text(
        json.dumps({"name": "should-not-load", "modality": "vision"}),
        encoding="utf-8",
    )

    loaded = load(root, modality="text")
    assert loaded.config["name"] == "vane-smoke-text"
    assert loaded.config["modality"] == "text"


def test_text_and_vision_configs_ship_with_the_package() -> None:
    from vane.train.yaml_config import config_dir, load_config

    root = config_dir()
    assert (root / "base.yaml").is_file()
    assert (root / "vane-text-1.0.0.yaml").is_file()
    assert (root / "vane-vision-1.0.0.yaml").is_file()
    text = load_config(root / "vane-text-1.0.0.yaml")
    vision = load_config(root / "vane-vision-1.0.0.yaml")
    assert text["model"]["backbone"]["repo"]
    assert vision["model"]["vision"]["repo"]
    assert text["model"]["modality"] == "text"
    assert vision["model"]["modality"] == "vision"


def test_hub_import_error_message(monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing huggingface_hub → clear ImportError pointing at vane[hub]."""
    import builtins

    import vane.load as load_mod

    real_import = builtins.__import__

    def fake_import(name: str, globals=None, locals=None, fromlist=(), level=0):
        if name == "huggingface_hub" or name.startswith("huggingface_hub."):
            raise ImportError("blocked for test")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match=r"vane\[hub\]|huggingface_hub"):
        load_mod._load_from_hub("org/does-not-matter", modality="text")
