"""Field-check router: explicit model → images → text. No pixel inspection."""

from __future__ import annotations

from typing import Any, Literal, Mapping

Reason = Literal["explicit", "images", "text"]

DESIGN_TEXT = "vane-1.0.0-text"
DESIGN_VISION = "vane-1.0.0-vision"
ALIAS_TEXT = "vane-text"
ALIAS_VISION = "vane-vision"

# Served until real weights are loaded. Clearly marked untrained/local.
UNTRAINED_TEXT = "vane-untrained-local-text"
UNTRAINED_VISION = "vane-untrained-local-vision"

ALIASES = {
    ALIAS_TEXT: DESIGN_TEXT,
    ALIAS_VISION: DESIGN_VISION,
}

DESIGN_IDS = frozenset({DESIGN_TEXT, DESIGN_VISION})
UNTRAINED_IDS = frozenset({UNTRAINED_TEXT, UNTRAINED_VISION})

# Explicit ``model`` must be one of these; anything else → HTTP 400.
ALLOWED_MODELS = frozenset(
    {
        ALIAS_TEXT,
        ALIAS_VISION,
        DESIGN_TEXT,
        DESIGN_VISION,
        UNTRAINED_TEXT,
        UNTRAINED_VISION,
    }
)


class UnknownModelError(ValueError):
    """Raised when the caller passes an explicit model id outside ALLOWED_MODELS."""


def state_has_images(state: Any) -> bool:
    """True iff ``state`` has a non-empty ``images`` field (field check only)."""
    if isinstance(state, Mapping):
        images = state.get("images")
        if images is None:
            return False
        if isinstance(images, (list, tuple)):
            return len(images) > 0
        return True
    return False


def resolve_route(
    *,
    model: str | None,
    state: Any,
    weights_loaded: Mapping[str, bool] | None = None,
) -> dict[str, Any]:
    """Pick checkpoint id and routing reason.

    Until weights exist, aliases and design ids resolve to the matching
    untrained local id. ``reason`` stays ``explicit`` when the caller set
    ``model``; unresolved aliases are noted on the routing object.

    Raises
    ------
    UnknownModelError
        If ``model`` is non-empty and not in :data:`ALLOWED_MODELS`.
    """
    weights = weights_loaded or {}

    def _has_weights(design_id: str) -> bool:
        return bool(weights.get(design_id))

    def _serve_id(design_id: str) -> str:
        if _has_weights(design_id):
            return design_id
        return UNTRAINED_VISION if "vision" in design_id else UNTRAINED_TEXT

    routing: dict[str, Any] = {}

    if model is not None and str(model).strip() != "":
        requested = str(model).strip()
        reason: Reason = "explicit"
        routing["reason"] = reason
        routing["requested"] = requested

        if requested not in ALLOWED_MODELS:
            raise UnknownModelError(
                f"unknown model id {requested!r}; "
                f"allowed: {', '.join(sorted(ALLOWED_MODELS))}"
            )

        if requested in ALIASES:
            design = ALIASES[requested]
            routing["alias"] = requested
            if not _has_weights(design):
                routing["alias_unresolved"] = True
                routing["note"] = (
                    f"alias {requested!r} unresolved until weights exist; "
                    f"serving untrained local"
                )
            checkpoint = _serve_id(design)
            routing["checkpoint"] = checkpoint
            routing["design_id"] = design
            return routing

        if requested in DESIGN_IDS:
            if not _has_weights(requested):
                routing["note"] = (
                    f"design id {requested!r} has no weights; serving untrained local"
                )
            routing["checkpoint"] = _serve_id(requested)
            routing["design_id"] = requested
            return routing

        # UNTRAINED_IDS
        routing["checkpoint"] = requested
        return routing

    if state_has_images(state):
        routing["reason"] = "images"
        design = DESIGN_VISION
        routing["design_id"] = design
        routing["checkpoint"] = _serve_id(design)
        if not _has_weights(design):
            routing["note"] = "no vision weights; serving untrained local"
        return routing

    routing["reason"] = "text"
    design = DESIGN_TEXT
    routing["design_id"] = design
    routing["checkpoint"] = _serve_id(design)
    if not _has_weights(design):
        routing["note"] = "no text weights; serving untrained local"
    return routing
