"""Training helpers. Torch is imported only when a training symbol is used."""

from __future__ import annotations

_EXPORTS = {
    "noul_loss",
    "choice_loss",
    "score_loss",
    "isotonic_residual",
    "ranked_probability_score",
    "one_train_step",
    "fit_cardinality_temperature",
    "fit_early_exit_tau",
    "resolve_device",
    "batch_to_device",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    try:
        import torch  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "vane.train requires PyTorch. Install the train extra."
        ) from exc
    from vane.train.calibration import fit_cardinality_temperature, fit_early_exit_tau
    from vane.train.device import batch_to_device, resolve_device
    from vane.train.losses import (
        choice_loss,
        isotonic_residual,
        noul_loss,
        ranked_probability_score,
        score_loss,
    )
    from vane.train.step import one_train_step

    values = {
        "noul_loss": noul_loss,
        "choice_loss": choice_loss,
        "score_loss": score_loss,
        "isotonic_residual": isotonic_residual,
        "ranked_probability_score": ranked_probability_score,
        "one_train_step": one_train_step,
        "fit_cardinality_temperature": fit_cardinality_temperature,
        "fit_early_exit_tau": fit_early_exit_tau,
        "resolve_device": resolve_device,
        "batch_to_device": batch_to_device,
    }
    return values[name]
