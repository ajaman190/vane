"""HTTP client for a Vane (or TypeSafe-compatible) system-one server."""

from __future__ import annotations

from typing import Any, Literal, Mapping, NotRequired, TypedDict

try:
    import httpx
except ImportError:  # pragma: no cover - httpx is in core deps
    httpx = None  # type: ignore[assignment]


class NoulAnswer(TypedDict):
    type: Literal["noul"]
    noul: float


class ChoiceAnswer(TypedDict):
    type: Literal["choice"]
    choice: str
    probabilities: dict[str, float]
    confidence: float


class ScoreAnswer(TypedDict):
    type: Literal["score"]
    score: float
    legend: list[Any]
    probabilities: dict[str, float]
    confidence: float


Answer = NoulAnswer | ChoiceAnswer | ScoreAnswer


class Routing(TypedDict):
    checkpoint: str
    reason: Literal["explicit", "images", "text"]
    design_id: NotRequired[str]
    alias: NotRequired[str]
    alias_unresolved: NotRequired[bool]
    requested: NotRequired[str]
    note: NotRequired[str]


class Usage(TypedDict):
    input_tokens: int
    output_tokens: int


class SystemOneResult(TypedDict):
    model: str
    answers: dict[str, Answer]
    routing: Routing
    usage: Usage


class ModelCard(TypedDict):
    name: str
    description: str
    release_date: str
    aliases: NotRequired[list[str]]
    status: NotRequired[str]


class Client:
    """Minimal Python client: ``systemone`` and ``models``.

    Parameters
    ----------
    base_url:
        Server root, e.g. ``http://127.0.0.1:8000``.
        ``Client(base_url)`` is enough; no API key is required.
    api_key:
        Optional Bearer token. When ``None`` (default), no ``Authorization``
        header is sent. The current Vane server ignores the header either way.
    timeout:
        Request timeout in seconds.
    transport:
        Optional ``httpx`` transport (used in tests with ``ASGITransport`` /
        ``MockTransport``).
    """

    def __init__(
        self,
        base_url: str,
        api_key: str | None = None,
        *,
        timeout: float = 30.0,
        transport: Any | None = None,
    ) -> None:
        if httpx is None:
            raise ImportError(
                "httpx is required for vane.client.Client; "
                "install the vane package (core dependency)"
            )
        self.base_url = base_url.rstrip("/") + "/"
        self.api_key = api_key
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if api_key:  # omit Authorization when None or empty
            headers["Authorization"] = f"Bearer {api_key}"
        self._client = httpx.Client(
            base_url=self.base_url,
            headers=headers,
            timeout=timeout,
            transport=transport,
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> Client:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def systemone(
        self,
        *,
        state: Any,
        questions: Mapping[str, Any],
        model: str | None = None,
    ) -> SystemOneResult:
        """POST /v1/systemone."""
        body: dict[str, Any] = {"state": state, "questions": dict(questions)}
        if model is not None:
            body["model"] = model
        response = self._client.post("/v1/systemone", json=body)
        response.raise_for_status()
        return response.json()  # type: ignore[return-value]

    def models(self) -> list[ModelCard]:
        """GET /v1/models.

        The server returns ``{"models": [...]}``. This method returns the
        inner list for ergonomics (not the wrapping object).
        """
        response = self._client.get("/v1/models")
        response.raise_for_status()
        data = response.json()
        if isinstance(data, dict) and "models" in data:
            models = data["models"]
            if not isinstance(models, list):
                raise TypeError("GET /v1/models: 'models' must be a JSON list")
            return models  # type: ignore[return-value]
        if isinstance(data, list):
            # Legacy bare-list servers
            return data  # type: ignore[return-value]
        raise TypeError("GET /v1/models expected {\"models\": [...]} or a JSON list")
