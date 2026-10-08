"""Regression guards for the bare-term / heading-stub / rate-limit failures (no network)."""

import httpx
import pytest

from building_with_rag.generation import answer
from building_with_rag.retrieval import hybrid


class _Chunks:
    def __init__(self):
        self.pipeline = None

    def aggregate(self, pipeline):
        self.pipeline = pipeline
        return []


def test_keyword_route_excludes_short_heading_chunks():
    chunks = _Chunks()
    hybrid.keyword_hits(chunks, "criminal breach of trust", {"access_level": ["public"]}, 20)
    match = next(stage["$match"] for stage in chunks.pipeline if "$match" in stage)
    assert match == {"$expr": {"$gte": [{"$strLenCP": "$text"}, hybrid.MIN_KEYWORD_CHARS]}}
    assert chunks.pipeline.index({"$match": match}) < chunks.pipeline.index({"$limit": 20})


def test_prompt_handles_bare_terms_and_requires_cited_lines():
    prompt = answer.SYSTEM_PROMPT
    assert "bare legal term" in prompt and "do not refuse because it is brief" in prompt
    assert "no introduction, no headings, no lines without a label" in prompt


def _settings():
    class S:
        generation_api_base_url = "http://fake"
        generation_api_key = "k"
        generation_model_name = "m"

    return S()


def test_complete_retries_rate_limit_then_succeeds(monkeypatch):
    replies = [
        httpx.Response(429, headers={"retry-after": "1"}),
        httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]}),
    ]
    monkeypatch.setattr(answer.httpx, "post", lambda *a, **k: replies.pop(0))
    monkeypatch.setattr(answer.time, "sleep", lambda s: None)
    assert answer._complete(_settings(), []) == "ok"


def test_complete_gives_up_after_bounded_retries(monkeypatch):
    calls = []

    def post(*a, **k):
        calls.append(1)
        return httpx.Response(429)

    monkeypatch.setattr(answer.httpx, "post", post)
    monkeypatch.setattr(answer.time, "sleep", lambda s: None)
    with pytest.raises(answer._ProviderError):
        answer._complete(_settings(), [])
    assert len(calls) == answer.RATE_LIMIT_RETRIES + 1


def test_stream_retries_rate_limit_then_streams(monkeypatch):
    sse = 'data: {"choices":[{"delta":{"content":"hi"}}]}\n\ndata: [DONE]\n\n'
    responses = [httpx.Response(429), httpx.Response(200, text=sse)]

    def handler(request):
        return responses.pop(0)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    monkeypatch.setattr(
        answer.httpx, "stream", lambda method, url, **kw: client.stream(method, url, **kw)
    )
    monkeypatch.setattr(answer.time, "sleep", lambda s: None)
    assert "".join(answer._stream_provider(_settings(), [])) == "hi"
