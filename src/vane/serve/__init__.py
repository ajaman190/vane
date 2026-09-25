"""Local HTTP serve surface for Vane (TypeSafe-compatible /v1/systemone)."""

from __future__ import annotations

from vane.serve.app import app, create_app, main

__all__ = ["app", "create_app", "main"]
