"""Story 4.2: re-ranking selection, validation and failure handling (no network)."""

import httpx
import pytest

from building_with_rag import pipeline
from building_with_rag.generation.context import build_context
from building_with_rag.models import QueryRequest, QueryResult, RetrievedChunk
from building_with_rag.retrieval import rerank
from building_with_rag.retrieval.semantic import RetrievalError


class _Settings:
    rerank_api_key = "k"
    rerank_api_base_url = "http://fake/v1"
    rerank_model_name = "m"
    rerank_request_timeout_seconds = 5
    rerank_candidate_limit = 4
    rerank_send_limit = 3
    rerank_return_limit = 2


def _hybrid_result(n=4):
    chunks = [
        RetrievedChunk(
            chunk_id=f"c{i}", section_id=f"bns:{i}", act="BNS_2023", heading=f"H{i}",
            text=f"text {i}", score=0.1 - i / 100, fused_score=0.1 - i / 100, fused_rank=i,
        )
        for i in range(1, n + 1)
    ]
    return QueryResult(pattern="hybrid", status="ok", trace={"filters": {}}, results=chunks)


def _request(limit=5):
    return QueryRequest(question="q", pattern="hybrid-reranked", limit=limit)


@pytest.fixture
def settings(monkeypatch):
    s = _Settings()
    monkeypatch.setattr(rerank, "get_settings", lambda: s)
    return s


@pytest.fixture
def env(monkeypatch, settings):
    state = {"hybrid_calls": 0, "provider_calls": 0, "scores": {0: 0.1, 1: 0.9, 2: 0.5}}

    def run_hybrid(request):
        state["hybrid_calls"] += 1
        state["limit"] = request.limit
        return _hybrid_result()

    def provider(s, query, documents):
        state["provider_calls"] += 1
        return state["scores"], 42

    monkeypatch.setattr(rerank.hybrid, "run_hybrid", run_hybrid)
    monkeypatch.setattr(rerank, "call_provider", provider)
    state["settings"] = settings
    return state


def test_reorders_and_keeps_hybrid_fields(env):
    env["settings"].rerank_return_limit = 3
    result = rerank.run_hybrid_reranked(_request())
    assert env["limit"] == 4
    assert [r.chunk_id for r in result.results] == ["c2", "c3", "c1"]
    assert [r.rerank_rank for r in result.results] == [1, 2, 3]
    assert all(r.score == r.rerank_score for r in result.results)
    assert [r.fused_rank for r in result.results] == [2, 3, 1]
    assert result.trace["rerank"]["usage_tokens"] == 42


def test_omitted_before_and_after(env):
    result = rerank.run_hybrid_reranked(_request())
    assert [r.chunk_id for r in result.results] == ["c2", "c3"]
    by_id = {o.chunk_id: o for o in result.omitted_candidates}
    assert [o.chunk_id for o in result.omitted_candidates] == ["c1", "c4"]
    assert by_id["c4"].omitted_reason == "not_sent_to_reranker"
    assert by_id["c4"].rerank_score is None and by_id["c4"].rerank_rank is None
    assert by_id["c1"].omitted_reason == "below_return_limit"
    assert by_id["c1"].rerank_rank == 3 and by_id["c1"].rerank_score == 0.1
    t = result.trace["rerank"]
    assert (t["candidates"], t["sent"], t["returned"]) == (4, 3, 2)
    assert (t["omitted_before"], t["omitted_after"]) == (1, 1)


def test_empty_key_is_503_without_calls(env):
    env["settings"].rerank_api_key = ""
    with pytest.raises(RetrievalError) as exc:
        rerank.run_hybrid_reranked(_request())
    assert exc.value.status_code == 503 and exc.value.code == "retrieval_not_ready"
    assert "RERANK_API_KEY" in exc.value.message
    assert env["hybrid_calls"] == 0 and env["provider_calls"] == 0


def test_invalid_limits_are_503(env):
    env["settings"].rerank_send_limit = 9
    with pytest.raises(RetrievalError) as exc:
        rerank.run_hybrid_reranked(_request())
    assert exc.value.status_code == 503 and "RERANK_SEND_LIMIT" in exc.value.message
    assert env["hybrid_calls"] == 0


@pytest.fixture
def wire(monkeypatch, settings):
    """Real call_provider over a mock transport."""
    monkeypatch.setattr(rerank.hybrid, "run_hybrid", lambda r: _hybrid_result())

    def install(handler):
        client = httpx.Client(transport=httpx.MockTransport(handler))
        monkeypatch.setattr(rerank.httpx, "post", lambda url, **kw: client.post(url, **kw))

    return install


@pytest.mark.parametrize(
    "reply",
    [
        {"data": [{"index": 0, "relevance_score": 0.5}, {"index": 5, "relevance_score": 0.4}]},
        {"data": [{"index": 0, "relevance_score": 0.5}, {"index": 0, "relevance_score": 0.4}]},
        {"data": [{"index": 0}, {"index": 1, "relevance_score": 0.4}, {"index": 2}]},
        {"data": []},
    ],
)
def test_invalid_reply_is_502(wire, reply):
    wire(lambda request: httpx.Response(200, json=reply))
    with pytest.raises(RetrievalError) as exc:
        rerank.run_hybrid_reranked(_request())
    assert exc.value.status_code == 502 and exc.value.code == "retrieval_upstream_error"


@pytest.mark.parametrize("failure", ["timeout", "http500"])
def test_provider_failure_is_502(wire, failure):
    def handler(request):
        if failure == "timeout":
            raise httpx.ReadTimeout("slow")
        return httpx.Response(500, json={"error": "secret-body"})

    wire(handler)
    with pytest.raises(RetrievalError) as exc:
        rerank.run_hybrid_reranked(_request())
    assert exc.value.status_code == 502
    assert "secret-body" not in exc.value.message


def test_context_uses_only_final_results_and_pipeline_routes_here(env, monkeypatch):
    result = rerank.run_hybrid_reranked(_request())
    ctx = build_context(result.results)
    assert [e["chunk_id"] for e in ctx.entries] == ["c2", "c3"]
    monkeypatch.setattr(pipeline.semantic, "validate_scope", lambda r: [])
    monkeypatch.setitem(pipeline._RETRIEVERS, "hybrid-reranked", lambda r: "routed")
    assert pipeline.retrieve(_request()) == "routed"
