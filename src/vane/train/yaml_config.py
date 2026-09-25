"""Load a run YAML that extends base.yaml in this package."""

from __future__ import annotations

from importlib import resources
from pathlib import Path
from typing import Any


def package_dir() -> Path:
    return Path(__file__).resolve().parent


def config_dir() -> Path:
    return package_dir() / "configs"


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if key == "extends":
            continue
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("PyYAML is required. Install the train extra.") from exc

    if path is None:
        path = config_dir() / "vane-text-1.0.0.yaml"
    path = Path(path)
    if not path.is_file():
        candidate = config_dir() / path.name
        if candidate.is_file():
            path = candidate
    with open(path, encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ValueError(f"config root must be a mapping: {path}")
    if "extends" in config:
        base_name = str(config["extends"])
        if not base_name.endswith((".yaml", ".yml")):
            base_name = f"{base_name}.yaml"
        base_path = path.parent / base_name
        if not base_path.is_file():
            base_path = config_dir() / base_name
        base = load_config(base_path)
        config = deep_merge(base, config)
    return config


def active_model(config: dict[str, Any]) -> dict[str, Any]:
    """Dims for the selected profile, plus name, backbone, and vision fields."""
    model = dict(config.get("model") or {})
    profile = str(model.get("profile", "dev"))
    block = dict(model.get(profile) or {})
    for key in (
        "window_size",
        "max_context",
        "max_levels",
        "dropout",
        "tau_choice",
        "tau_score",
        "name",
        "vision_proj_dim",
        "n_refine",
        "backbone_dim",
    ):
        if key in model and key not in block:
            block[key] = model[key]
    vision = model.get("vision") or {}
    if block.get("vision_proj_dim") is None and vision.get("proj_dim") is not None:
        block["vision_proj_dim"] = vision["proj_dim"]
    block["profile"] = profile
    block["backbone"] = model.get("backbone")
    block["modality"] = model.get("modality", "text")
    return block


def checkpoint_dir(config: dict[str, Any]) -> Path:
    raw = Path(str((config.get("checkpoint") or {}).get("dir", "checkpoints/default")))
    if raw.is_absolute():
        return raw
    return package_dir() / raw


def bundled_config(name: str) -> Path:
    """Path to a YAML shipped inside the package."""
    path = config_dir() / name
    if path.is_file():
        return path
    root = resources.files("vane.train") / "configs" / name
    return Path(str(root))
