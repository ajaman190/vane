"""Python Client against the in-process FastAPI app (no network)."""

from __future__ import annotations

import httpx
import pytest
from fastapi.testclient import TestClient

from vane.client import Client
from vane.presets import guard_questions, moderation_questions, triage_questions
from vane.serve.app import create_app
from vane.serve.router import UNTRAINED_TEXT


def _asgi_client(api_key: str | None = None) -> Client:
    """Wire Client → MockTransport → FastAPI TestClient (sync, no network)."""
    asgi_app = create_app()
    starlette = TestClient(asgi_app)

    def handler(request: httpx.Request) -> httpx.Response:
        response = starlette.request(
            request.method,
            request.url.path,
            content=request.content,
            headers={
                k: v
                for k, v in request.headers.items()
                if k.lower() != "host"
            },
        )
        return httpx.Response(
            status_code=response.status_code,
            headers=response.headers,
            content=response.content,
            request=request,
        )

    return Client("http://test", api_key, transport=httpx.MockTransport(handler))


@pytest.fixture()
def vane_client() -> Client:
    return _asgi_client(None)


def test_client_systemone(vane_client: Client) -> None:
    result = vane_client.systemone(
        state="refund ticket",
        questions={
            "billing": {
                "type": "choice",
                "instructions": "dept",
                "criteria": {"billing": None, "tech": None},
            }
        },
    )
    assert result["routing"]["checkpoint"] == UNTRAINED_TEXT
    assert result["model"] == UNTRAINED_TEXT
    assert result["answers"]["billing"]["type"] == "choice"
    assert "confidence" in result["answers"]["billing"]
    assert "usage" in result
    vane_client.close()


def test_client_models_returns_list(vane_client: Client) -> None:
    models = vane_client.models()
    assert isinstance(models, list)
    names = {m["name"] for m in models}
    assert UNTRAINED_TEXT in names
    assert "vane-text" in names
    assert "vane-vision" in names
    for m in models:
        assert "release_date" in m
    vane_client.close()


def test_client_base_url_only_no_auth_header() -> None:
    """Client(base_url) works; no Authorization when api_key is None."""
    seen: dict[str, str | None] = {"authorization": None}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["authorization"] = request.headers.get("authorization")
        return httpx.Response(
            200,
            json={"models": [{"name": "vane-text", "description": "", "release_date": ""}]},
            request=request,
        )

    with Client("http://example.test", transport=httpx.MockTransport(handler)) as c:
        c.models()
    assert seen["authorization"] is None


def test_client_optional_api_key_sends_bearer() -> None:
    seen: dict[str, str | None] = {"authorization": None}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["authorization"] = request.headers.get("authorization")
        return httpx.Response(
            200,
            json={"models": []},
            request=request,
        )

    with Client(
        "http://example.test",
        api_key="secret",
        transport=httpx.MockTransport(handler),
    ) as c:
        c.models()
    assert seen["authorization"] == "Bearer secret"


def test_client_with_asgi_and_ignored_key() -> None:
    client = _asgi_client("ignored-key")
    result = client.systemone(
        state="ok",
        questions={"n": {"type": "noul", "instructions": "?"}},
    )
    assert result["answers"]["n"]["type"] == "noul"
    client.close()


def test_presets_editable_maps() -> None:
    for factory in (triage_questions, guard_questions, moderation_questions):
        qs = factory()
        assert isinstance(qs, dict)
        assert len(qs) >= 1
        types = {q["type"] for q in qs.values()}
        assert types <= {"noul", "choice", "score"}
        # Editable: mutation must not leak into the next call.
        first_key = next(iter(qs))
        qs[first_key]["instructions"] = "mutated"
        again = factory()
        assert again[first_key]["instructions"] != "mutated"


def test_client_export() -> None:
    import vane

    assert vane.__version__ == "0.1.0"
    assert vane.Client is Client


def test_serve_main_exported() -> None:
    from vane.serve import main

    assert callable(main)
