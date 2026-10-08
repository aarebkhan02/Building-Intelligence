"""Story 3.2: streamed answers, bounded validation, confidence (fake provider, no network)."""

import json

import pytest
from fastapi.testclient import TestClient

from building_with_rag import pipeline
from building_with_rag.app import app
from building_with_rag.generation import answer
from building_with_rag.models import QueryResult, RetrievedChunk


class _Settings:
    generation_api_base_url = "http://fake"
    generation_api_key = "k"
    generation_model_name = "m"
    capstone_api_key = ""
    webui_demo_caller_id = "demo-public"


def _retrieval():
    chunks = [
        RetrievedChunk(chunk_id="c1", section_id="bns:303", act="BNS_2023", heading="Theft",
                       section_number=303, text="Theft is punished with imprisonment.")
    ]
    return QueryResult(pattern="semantic", status="ok", results=chunks)


@pytest.fixture
def fake(monkeypatch):
    state = {"drafts": [], "calls": 0, "verdicts": [], "fail_after": None}
    monkeypatch.setattr(answer, "get_settings", lambda: _Settings())
    monkeypatch.setattr("building_with_rag.app.get_settings", lambda: _Settings())
    monkeypatch.setattr("building_with_rag.app.pipeline.retrieve", lambda r: _retrieval())

    def stream(settings, messages):
        text = state["drafts"][state["calls"]]
        state["calls"] += 1
        for i in range(0, len(text), 7):
            yield text[i : i + 7]
        if state["fail_after"] is not None:
            raise answer._ProviderError("boom")

    def complete(settings, messages):
        return json.dumps({"results": [{"claim": 1, "supported": state["verdicts"].pop(0)}]})

    monkeypatch.setattr(answer, "_stream_provider", stream)
    monkeypatch.setattr(answer, "_complete", complete)
    return state


def _final(retrieval=None):
    events = list(answer.answer_stream("q", retrieval or _retrieval()))
    return events, events[-1].result


def test_pass_single_call_stream_equals_text(fake):
    fake["drafts"] = ["Theft is punished with imprisonment [E1]."]
    fake["verdicts"] = [True]
    events, g = _final()
    assert fake["calls"] == 1 and g.outcome == "answered" and g.confidence == "high"
    assert "".join(e.text for e in events if e.kind == "text") == g.text
    assert [a["status"] for a in g.attempts] == ["passed"]


def test_retry_then_pass_records_both_attempts(fake):
    fake["drafts"] = ["Theft is punished [E9].", "Theft is punished with imprisonment [E1]."]
    fake["verdicts"] = [True]
    events, g = _final()
    notices = [e.text for e in events if e.kind == "notice"]
    assert sum("DRAFT — checking evidence" in n for n in notices) == 2
    assert any("Retrying (attempt 2 of 2)" in n for n in notices)
    assert g.confidence == "high" and [i["check"] for i in g.issues] == ["citation_labels"]
    assert [a["status"] for a in g.attempts] == ["failed", "passed"]


def test_both_fail_low_confidence(fake):
    fake["drafts"] = ["Theft is punished [E9].", "Theft is punished [E1].\nAlso fines."]
    fake["verdicts"] = []
    _, g = _final()
    assert g.outcome == "malformed" and g.text == "" and g.confidence == "low"
    assert g.draft_answer and g.low_confidence_reason and len(g.issues) == 2
    out = pipeline.footer(g, True)
    assert "DRAFT — low confidence, not the final answer." in out


def test_provider_failure_mid_stream(fake):
    fake["drafts"] = ["Theft is punished [E1]."]
    fake["fail_after"] = True
    c = TestClient(app)
    with c.stream("POST", "/v1/chat/completions", json={
        "model": "rag-semantic", "stream": True,
        "messages": [{"role": "user", "content": "q"}]}) as r:
        raw = "".join(r.iter_text())
    assert r.status_code == 200
    assert "unchecked draft" in raw and '"finish_reason": "stop"' in raw
    assert raw.strip().endswith("data: [DONE]")
    fake["calls"] = 0
    _, g = _final()
    assert g.outcome == "unavailable" and g.confidence is None


def test_auth_required_when_key_set(fake, monkeypatch):
    s = _Settings()
    s.capstone_api_key = "secret"
    monkeypatch.setattr("building_with_rag.app.get_settings", lambda: s)
    c = TestClient(app)
    body = {"model": "rag-semantic", "messages": [{"role": "user", "content": "q"}]}
    r = c.post("/v1/chat/completions", json=body, headers={"Authorization": "Bearer nope"})
    assert r.status_code == 401 and r.json()["error"]["code"] == "invalid_api_key"
