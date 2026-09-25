"""FastAPI application: POST /v1/systemone, GET /v1/models.

No API key. An ``Authorization`` header, if present, is ignored.

Without weights the server still returns a valid schema under checkpoint ids
``vane-untrained-local-text`` / ``vane-untrained-local-vision``.

Run::

    vane-serve --host 127.0.0.1 --port 8000
    # or: python -m vane.serve
"""

from __future__ import annotations

import argparse
import os
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from vane.serve.inference import generate_answers
from vane.serve.router import (
    ALIAS_TEXT,
    ALIAS_VISION,
    DESIGN_TEXT,
    DESIGN_VISION,
    UNTRAINED_TEXT,
    UNTRAINED_VISION,
    UnknownModelError,
    resolve_route,
)
from vane.serve.schemas import CONTEXT_LIMIT, estimate_tokens, validate_questions


def create_app() -> FastAPI:
    """Build a fresh FastAPI app (useful for tests)."""
    application = FastAPI(title="Vane", version="0.1.0")

    @application.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @application.get("/v1/models")
    async def list_models() -> dict[str, list[dict[str, Any]]]:
        return {
            "models": [
                {
                    "name": ALIAS_TEXT,
                    "description": (
                        "Alias for the multilingual text checkpoint. "
                        "Resolves to a versioned id at request time "
                        "(untrained local until weights exist)."
                    ),
                    "release_date": "",
                },
                {
                    "name": ALIAS_VISION,
                    "description": (
                        "Alias for the text+image checkpoint. "
                        "Resolves to a versioned id at request time "
                        "(untrained local until weights exist)."
                    ),
                    "release_date": "",
                },
                {
                    "name": DESIGN_TEXT,
                    "description": (
                        "Design-target multilingual text checkpoint (~1B). "
                        "Unreleased; served as untrained local until weights exist."
                    ),
                    "release_date": "",
                },
                {
                    "name": DESIGN_VISION,
                    "description": (
                        "Design-target text+image checkpoint (shared text tower). "
                        "Unreleased; served as untrained local until weights exist."
                    ),
                    "release_date": "",
                },
                {
                    "name": UNTRAINED_TEXT,
                    "description": (
                        "Local untrained stub/smoke path for text requests. "
                        "Returns valid noul/choice/score schema without weights."
                    ),
                    "release_date": "",
                },
                {
                    "name": UNTRAINED_VISION,
                    "description": (
                        "Local untrained stub/smoke path for image requests. "
                        "Returns valid noul/choice/score schema without weights."
                    ),
                    "release_date": "",
                },
            ]
        }

    @application.post("/v1/systemone")
    async def systemone(request: Request) -> JSONResponse:
        # Authorization is intentionally ignored (no API key).
        try:
            body = await request.json()
        except Exception as exc:
            raise HTTPException(status_code=400, detail="invalid JSON body") from exc

        if not isinstance(body, dict):
            raise HTTPException(status_code=400, detail="body must be a JSON object")

        if "state" not in body:
            raise HTTPException(status_code=400, detail="state is required")
        if "questions" not in body:
            raise HTTPException(status_code=400, detail="questions is required")

        state = body["state"]
        model = body.get("model")

        try:
            questions = validate_questions(body["questions"])
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        n_tokens = estimate_tokens(state)
        if n_tokens > CONTEXT_LIMIT:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"context length {n_tokens} exceeds limit {CONTEXT_LIMIT}; "
                    "refused (not cropped)"
                ),
            )

        try:
            routing = resolve_route(model=model, state=state, weights_loaded={})
        except UnknownModelError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        checkpoint = routing["checkpoint"]
        answers = generate_answers(state, questions, checkpoint)

        payload = {
            "model": checkpoint,
            "answers": answers,
            "routing": routing,
            "usage": {
                "input_tokens": n_tokens,
                "output_tokens": len(answers),
            },
        }
        return JSONResponse(payload)

    return application


# Module-level app for ``uvicorn vane.serve.app:app``
app = create_app()


def main(argv: list[str] | None = None) -> None:
    """Console entry for ``vane-serve`` / ``python -m vane.serve``."""
    parser = argparse.ArgumentParser(prog="vane-serve", description="Run the Vane HTTP server")
    parser.add_argument(
        "--host",
        default=os.environ.get("VANE_HOST", "127.0.0.1"),
        help="Bind host (default 127.0.0.1; env VANE_HOST)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("VANE_PORT", "8000")),
        help="Bind port (default 8000; env VANE_PORT)",
    )
    args = parser.parse_args(argv)

    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "uvicorn is required to run vane-serve; pip install 'vane[serve]'"
        ) from exc

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
