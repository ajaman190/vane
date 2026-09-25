"""Load a Vane checkpoint from a local directory or Hugging Face repo id.

Local layout (per modality directory)::

    config.json
    model.safetensors

Hugging Face repos that ship both modalities use sibling subdirectories
``text/`` and ``vision/``. ``load(..., modality=...)`` downloads only that
subtree via ``allow_patterns`` — never the sibling modality.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Mapping

Modality = Literal["text", "vision"]

_WEIGHTS_NAME = "model.safetensors"
_CONFIG_NAME = "config.json"


@dataclass
class LoadedCheckpoint:
    """Result of :func:`load`.

    Always exposes ``.config`` (parsed JSON) and ``.state_dict`` (tensor map).
    When ``torch`` is installed and the config looks like a :class:`VaneConfig`,
    ``.model`` may hold a constructed ``VaneModel`` with weights applied.
    """

    config: dict[str, Any]
    state_dict: Mapping[str, Any]
    model: Any | None = field(default=None, repr=False)
    modality: Modality = "text"
    path: str | None = None


def load(
    path_or_repo_id: str | Path,
    *,
    modality: Modality = "text",
) -> LoadedCheckpoint:
    """Load checkpoint config + weights for one modality.

    Parameters
    ----------
    path_or_repo_id:
        Local directory (absolute/relative path that exists as a directory),
        or a Hugging Face hub repo id (e.g. ``org/vane-1.0.0``).
    modality:
        ``\"text\"`` or ``\"vision\"``. Hub downloads use ``allow_patterns`` so
        only this modality's files are fetched.
    """
    if modality not in ("text", "vision"):
        raise ValueError(f"modality must be 'text' or 'vision', got {modality!r}")

    path = Path(path_or_repo_id)
    if path.is_dir():
        checkpoint_dir = _resolve_local_dir(path, modality)
        return _load_from_dir(checkpoint_dir, modality=modality)

    # Non-existent path → treat as Hugging Face repo id
    return _load_from_hub(str(path_or_repo_id), modality=modality)


def _resolve_local_dir(root: Path, modality: Modality) -> Path:
    """Prefer ``root/<modality>/`` when present; else ``root`` itself."""
    sub = root / modality
    if sub.is_dir() and (sub / _CONFIG_NAME).is_file():
        return sub
    if (root / _CONFIG_NAME).is_file():
        return root
    raise FileNotFoundError(
        f"No {_CONFIG_NAME} under {root} or {sub}. "
        "Expected a modality directory with config.json + model.safetensors."
    )


def _load_from_dir(checkpoint_dir: Path, *, modality: Modality) -> LoadedCheckpoint:
    config_path = checkpoint_dir / _CONFIG_NAME
    weights_path = checkpoint_dir / _WEIGHTS_NAME
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing {config_path}")
    if not weights_path.is_file():
        raise FileNotFoundError(f"Missing {weights_path}")

    with config_path.open(encoding="utf-8") as fh:
        config = json.load(fh)
    if not isinstance(config, dict):
        raise TypeError(f"{config_path} must contain a JSON object")

    state_dict = _read_safetensors(weights_path)
    model = _try_build_model(config, state_dict)
    return LoadedCheckpoint(
        config=config,
        state_dict=state_dict,
        model=model,
        modality=modality,
        path=str(checkpoint_dir.resolve()),
    )


def _load_from_hub(repo_id: str, *, modality: Modality) -> LoadedCheckpoint:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover - exercised when hub missing
        raise ImportError(
            "Loading from a Hugging Face repo id requires huggingface_hub. "
            "Install with: pip install 'vane[hub]' or pip install huggingface_hub"
        ) from exc

    # Only this modality's subtree — never the sibling (text vs vision).
    allow_patterns = [f"{modality}/*", f"{modality}/**"]
    local_root = Path(
        snapshot_download(
            repo_id=repo_id,
            allow_patterns=allow_patterns,
        )
    )
    checkpoint_dir = _resolve_local_dir(local_root, modality)
    return _load_from_dir(checkpoint_dir, modality=modality)


def _read_safetensors(weights_path: Path) -> dict[str, Any]:
    try:
        import safetensors  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "Reading model.safetensors requires the safetensors package. "
            "Install with: pip install safetensors (also in vane[hub])"
        ) from exc

    # Prefer torch tensors when torch is available; else numpy arrays.
    try:
        import torch  # noqa: F401
        from safetensors.torch import load_file as load_torch

        return dict(load_torch(str(weights_path)))
    except ImportError:
        from safetensors.numpy import load_file as load_numpy

        return dict(load_numpy(str(weights_path)))


def _try_build_model(
    config: Mapping[str, Any],
    state_dict: Mapping[str, Any],
) -> Any | None:
    """Optionally construct a VaneModel when torch + matching config are present."""
    try:
        import torch  # noqa: F401
        from vane.model.config import VaneConfig
        from vane.model.vane import VaneModel
    except ImportError:
        return None

    keys = (
        "d_model",
        "n_blocks",
        "n_heads",
        "d_ff",
        "window_size",
        "max_context",
        "vocab_size",
        "max_levels",
        "dropout",
        "name",
    )
    kwargs = {k: config[k] for k in keys if k in config}
    if "d_model" not in kwargs or "n_heads" not in kwargs:
        return None
    try:
        vane_cfg = VaneConfig(**kwargs)
        model = VaneModel(vane_cfg)
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        return model
    except Exception:
        # Smoke / partial weight files: still return tensors + config.
        return None


__all__ = ["LoadedCheckpoint", "load"]
