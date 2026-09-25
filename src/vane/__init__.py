"""Vane: a decision model. One state in, typed questions (noul, choice, score),
one forward pass, probabilities out. No text generation.
"""

from __future__ import annotations

__version__ = "0.1.0"

# Client and load are lazy-imported so ``import vane`` never pulls
# httpx / torch / safetensors / huggingface_hub / serve.
def __getattr__(name: str):
    if name == "Client":
        from vane.client import Client

        return Client
    if name == "load":
        from vane.load import load

        return load
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["Client", "load", "__version__"]
