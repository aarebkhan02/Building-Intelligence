"""Seed-level smoke tests: no credentials needed, placeholder behavior only."""

from fastapi.testclient import TestClient

from building_with_rag.app import app

client = TestClient(app)


def test_healthz_ok():
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_query_not_implemented():
    response = client.post(
        "/v1/query",
        json={"question": "What is BNS section 103?", "pattern": "hybrid"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["pattern"] == "hybrid"
    assert body["status"] == "not_implemented"
    assert body["results"] == []


def test_list_models():
    response = client.get("/v1/models")
    assert response.status_code == 200
    ids = {m["id"] for m in response.json()["data"]}
    assert ids == {
        "rag-semantic",
        "rag-hybrid",
        "rag-hybrid-reranked",
        "rag-structured",
        "rag-decomposition",
        "rag-hyde",
    }


def test_chat_completions_json():
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "rag-semantic",
            "messages": [{"role": "user", "content": "What is BNS section 103?"}],
            "stream": False,
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["choices"][0]["message"]["role"] == "assistant"
    assert "not implemented" in body["choices"][0]["message"]["content"]


def test_chat_completions_sse():
    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": "rag-semantic",
            "messages": [{"role": "user", "content": "What is BNS section 103?"}],
            "stream": True,
        },
    ) as response:
        assert response.status_code == 200
        raw = "".join(response.iter_text())
    assert raw.strip().endswith("data: [DONE]")


def test_chat_completions_unknown_model_returns_openai_error():
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "not-a-real-model",
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert response.status_code == 404
    assert "error" in response.json()
