"""HTTP schema, routing, and validation tests (no GPU; no API key)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from vane.serve.app import create_app
from vane.serve.confidence import confidence
from vane.serve.router import UNTRAINED_TEXT, UNTRAINED_VISION


@pytest.fixture()
def client() -> TestClient:
    return TestClient(create_app())


def test_empty_questions_400(client: TestClient) -> None:
    r = client.post(
        "/v1/systemone",
        json={"state": "hello", "questions": {}},
    )
    assert r.status_code == 400


def test_choice_one_option_400(client: TestClient) -> None:
    r = client.post(
        "/v1/systemone",
        json={
            "state": "hello",
            "questions": {
                "dept": {
                    "type": "choice",
                    "instructions": "Which?",
                    "criteria": {"only": "one"},
                }
            },
        },
    )
    assert r.status_code == 400


def test_noul_affirmative_vs_opposite_state(client: TestClient) -> None:
    q = {"ok": {"type": "noul", "instructions": "Is the outcome positive?"}}
    r_pos = client.post(
        "/v1/systemone",
        json={
            "state": {
                "text": "Payment succeeded. Customer confirmed delivery and thanked support."
            },
            "questions": q,
        },
    )
    r_neg = client.post(
        "/v1/systemone",
        json={
            "state": {
                "text": "Payment failed. Customer reports missing package and is furious."
            },
            "questions": q,
        },
    )
    assert r_pos.status_code == 200, r_pos.text
    assert r_neg.status_code == 200, r_neg.text
    noul_pos = r_pos.json()["answers"]["ok"]["noul"]
    noul_neg = r_neg.json()["answers"]["ok"]["noul"]
    assert r_pos.json()["answers"]["ok"]["type"] == "noul"
    assert 0.0 <= noul_pos <= 1.0
    assert 0.0 <= noul_neg <= 1.0
    assert noul_pos != noul_neg


def test_images_field_routes_vision_else_text(client: TestClient) -> None:
    q = {"signed": {"type": "noul", "instructions": "signed?"}}
    r_img = client.post(
        "/v1/systemone",
        json={
            "state": {"text": "invoice", "images": ["fake-bytes"]},
            "questions": q,
        },
    )
    r_txt = client.post(
        "/v1/systemone",
        json={
            "state": {"text": "invoice"},
            "questions": q,
        },
    )
    assert r_img.status_code == 200
    assert r_txt.status_code == 200
    assert r_img.json()["routing"]["reason"] == "images"
    assert r_img.json()["routing"]["checkpoint"] == UNTRAINED_VISION
    assert r_img.json()["model"] == UNTRAINED_VISION
    assert r_txt.json()["routing"]["reason"] == "text"
    assert r_txt.json()["routing"]["checkpoint"] == UNTRAINED_TEXT
    assert r_txt.json()["model"] == UNTRAINED_TEXT


def test_unknown_model_jev_latest_400(client: TestClient) -> None:
    r = client.post(
        "/v1/systemone",
        json={
            "state": "plain",
            "model": "jev-latest",
            "questions": {"y": {"type": "noul", "instructions": "?"}},
        },
    )
    assert r.status_code == 400


def test_unknown_model_jev_version_400(client: TestClient) -> None:
    r = client.post(
        "/v1/systemone",
        json={
            "state": "plain",
            "model": "jev-1.13.0",
            "questions": {"y": {"type": "noul", "instructions": "?"}},
        },
    )
    assert r.status_code == 400


def test_models_object_shape(client: TestClient) -> None:
    r = client.get("/v1/models")
    assert r.status_code == 200
    body = r.json()
    assert isinstance(body, dict)
    assert "models" in body
    models = body["models"]
    assert isinstance(models, list)
    names = {m["name"] for m in models}
    assert "vane-text" in names
    assert "vane-vision" in names
    assert UNTRAINED_TEXT in names
    for m in models:
        assert "name" in m
        assert "description" in m
        assert "release_date" in m


def test_authorization_bearer_still_200(client: TestClient) -> None:
    r = client.post(
        "/v1/systemone",
        headers={"Authorization": "Bearer anything"},
        json={
            "state": "hello",
            "questions": {"u": {"type": "noul", "instructions": "urgent?"}},
        },
    )
    assert r.status_code == 200


def test_no_authorization_still_200(client: TestClient) -> None:
    r = client.post(
        "/v1/systemone",
        json={
            "state": "hello",
            "questions": {"u": {"type": "noul", "instructions": "urgent?"}},
        },
    )
    assert r.status_code == 200
    data = r.json()
    assert "answers" in data
    assert "usage" in data
    assert "input_tokens" in data["usage"]
    assert "output_tokens" in data["usage"]
    assert data["model"] == UNTRAINED_TEXT


def test_confidence_formula_known_values() -> None:
    # Uniform → 0; point mass → 1; k=4 max=0.96 → 0.947
    assert confidence([0.25, 0.25, 0.25, 0.25]) == pytest.approx(0.0)
    assert confidence([1.0, 0.0, 0.0, 0.0]) == pytest.approx(1.0)
    assert confidence([0.96, 0.02, 0.01, 0.01]) == pytest.approx((4 * 0.96 - 1) / 3)


def test_systemone_noul_choice_score_schema(client: TestClient) -> None:
    body = {
        "state": "My refund failed for three days.",
        "questions": {
            "urgent": {"type": "noul", "instructions": "Is this urgent?"},
            "dept": {
                "type": "choice",
                "instructions": "Which team?",
                "criteria": {
                    "billing": "payments",
                    "technical": "bugs",
                    "sales": "pricing",
                },
            },
            "frustration": {
                "type": "score",
                "instructions": "Frustration",
                "criteria": ["Calm", "Frustrated", "Very angry"],
            },
        },
    }
    r = client.post("/v1/systemone", json=body)
    assert r.status_code == 200, r.text
    data = r.json()
    assert "answers" in data
    assert "routing" in data
    assert data["routing"]["reason"] == "text"
    assert data["routing"]["checkpoint"] == UNTRAINED_TEXT
    assert data["model"] == UNTRAINED_TEXT

    noul = data["answers"]["urgent"]
    assert noul["type"] == "noul"
    assert 0.0 <= noul["noul"] <= 1.0
    assert "confidence" not in noul

    choice = data["answers"]["dept"]
    assert choice["type"] == "choice"
    assert choice["choice"] in choice["probabilities"]
    probs = list(choice["probabilities"].values())
    assert abs(sum(probs) - 1.0) < 1e-6
    assert "confidence" in choice
    assert abs(choice["confidence"] - confidence(probs)) < 1e-9

    score = data["answers"]["frustration"]
    assert score["type"] == "score"
    assert "legend" in score and "probabilities" in score
    assert isinstance(score["legend"], list)
    assert score["legend"] == ["Calm", "Frustrated", "Very angry"]
    assert "confidence" in score
    sprobs = [score["probabilities"][k] for k in sorted(score["probabilities"], key=int)]
    assert abs(score["confidence"] - confidence(sprobs)) < 1e-9


def test_router_explicit_alias_returns_versioned_model(client: TestClient) -> None:
    r = client.post(
        "/v1/systemone",
        json={
            "state": "plain text",
            "model": "vane-vision",
            "questions": {"y": {"type": "noul", "instructions": "yes?"}},
        },
    )
    assert r.status_code == 200
    data = r.json()
    routing = data["routing"]
    assert routing["reason"] == "explicit"
    assert routing["checkpoint"] == UNTRAINED_VISION
    assert routing.get("alias_unresolved") is True
    assert data["model"] == UNTRAINED_VISION


def test_context_refuse_not_crop(client: TestClient) -> None:
    # ~40k char pad → char/4 estimate > 32768
    huge = "x" * 140_000
    r = client.post(
        "/v1/systemone",
        json={
            "state": huge,
            "questions": {"a": {"type": "noul", "instructions": "?"}},
        },
    )
    assert r.status_code == 400
    assert "refused" in r.json()["detail"].lower()


def test_score_not_list_400(client: TestClient) -> None:
    r = client.post(
        "/v1/systemone",
        json={
            "state": "x",
            "questions": {
                "s": {
                    "type": "score",
                    "instructions": "lvl",
                    "criteria": {"a": "b"},
                }
            },
        },
    )
    assert r.status_code == 400


def test_models_get_with_auth_header_200(client: TestClient) -> None:
    r = client.get("/v1/models", headers={"Authorization": "Bearer ignored"})
    assert r.status_code == 200
    assert "models" in r.json()
