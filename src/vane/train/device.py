"""Device resolution for training. Prefer MPS on Apple Silicon; never require CUDA."""

from __future__ import annotations

from typing import Any

import torch


def resolve_device(requested: str | torch.device | None = "auto") -> torch.device:
    """Pick a torch device.

    ``auto`` uses CUDA when an NVIDIA GPU is present, then Apple Metal, then CPU.
    """
    if isinstance(requested, torch.device):
        return requested
    name = (requested or "auto").strip().lower()
    if name in {"auto", ""}:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if name == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        raise RuntimeError("config requested cuda but torch.cuda.is_available() is false")
    if name == "mps":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if name == "cpu":
        return torch.device("cpu")
    raise ValueError(f"unknown device {requested!r}; use auto|cuda|mps|cpu")


def batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    """Move tensor values in a batch dict onto ``device``; leave others as-is."""
    out: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            out[key] = value.to(device)
        else:
            out[key] = value
    return out
