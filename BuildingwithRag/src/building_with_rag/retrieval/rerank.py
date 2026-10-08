"""Hybrid re-ranking: bounded hybrid candidates, one provider rerank call, final evidence."""

from __future__ import annotations

import math
import time
from typing import Any

import httpx

from building_with_rag.config import get_settings
from building_with_rag.models import QueryRequest, QueryResult, RetrievedChunk
from building_with_rag.retrieval import hybrid
from building_with_rag.retrieval.semantic import RetrievalError

MAX_CANDIDATES = 20
SCORE_MESSAGE = (
    "Re-ranking scores rank only; they do not prove a passage is correct or answers the question."
)
UPSTREAM_MESSAGE = "Re-ranking request failed; no re-ranked result returned."
NOT_SENT = "not_sent_to_reranker"
BELOW_LIMIT = "below_return_limit"


class _ProviderError(Exception):
    pass


def _not_ready(message: str) -> RetrievalError:
    return RetrievalError(503, "retrieval_not_ready", message)


def validate_settings(settings) -> None:
    """Raise 503 naming the offending setting (never its value)."""
    for env_name, attr in (
        ("RERANK_API_KEY", "rerank_api_key"),
        ("RERANK_API_BASE_URL", "rerank_api_base_url"),
        ("RERANK_MODEL_NAME", "rerank_model_name"),
    ):
        if not str(getattr(settings, attr) or "").strip():
            raise _not_ready(f"{env_name} is not set; hybrid-reranked requires it.")
    if settings.rerank_request_timeout_seconds < 1:
        raise _not_ready("RERANK_REQUEST_TIMEOUT_SECONDS must be at least 1.")
    candidate, send, ret = (
        settings.rerank_candidate_limit,
        settings.rerank_send_limit,
        settings.rerank_return_limit,
    )
    if not 1 <= candidate <= MAX_CANDIDATES:
        raise _not_ready(f"RERANK_CANDIDATE_LIMIT must be between 1 and {MAX_CANDIDATES}.")
    if not 1 <= send <= candidate:
        raise _not_ready("RERANK_SEND_LIMIT must be between 1 and RERANK_CANDIDATE_LIMIT.")
    if not 1 <= ret <= send:
        raise _not_ready("RERANK_RETURN_LIMIT must be between 1 and RERANK_SEND_LIMIT.")


def _document(chunk: RetrievedChunk) -> str:
    return f"{chunk.heading or ''}\n{chunk.text}"


def parse_reply(body: Any, sent: int) -> tuple[dict[int, float], int | None]:
    """Validate a provider reply into {index: score}. Anything unexpected is a failure."""
    data = body.get("data") if isinstance(body, dict) else None
    if not isinstance(data, list) or not data:
        raise _ProviderError("data missing")
    scores: dict[int, float] = {}
    for item in data:
        if not isinstance(item, dict):
            raise _ProviderError("bad item")
        index, score = item.get("index"), item.get("relevance_score")
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < sent:
            raise _ProviderError("bad index")
        if index in scores:
            raise _ProviderError("duplicate index")
        bad_type = isinstance(score, bool) or not isinstance(score, int | float)
        if bad_type or not math.isfinite(score):
            raise _ProviderError("bad score")
        scores[index] = float(score)
    usage = body.get("usage")
    tokens = usage.get("total_tokens") if isinstance(usage, dict) else None
    return scores, tokens if isinstance(tokens, int) and not isinstance(tokens, bool) else None


def call_provider(
    settings, query: str, documents: list[str]
) -> tuple[dict[int, float], int | None]:
    """One POST {base}/rerank; no retries. Raises _ProviderError on any failure."""
    url = settings.rerank_api_base_url.rstrip("/") + "/rerank"
    try:
        response = httpx.post(
            url,
            headers={"Authorization": f"Bearer {settings.rerank_api_key}"},
            json={"model": settings.rerank_model_name, "query": query, "documents": documents},
            timeout=settings.rerank_request_timeout_seconds,
        )
        response.raise_for_status()
        return parse_reply(response.json(), len(documents))
    except _ProviderError:
        raise
    except (httpx.HTTPError, ValueError) as exc:
        raise _ProviderError(type(exc).__name__) from None


def select(
    candidates: list[RetrievedChunk],
    scores: dict[int, float],
    send_limit: int,
    final_count: int,
) -> tuple[list[RetrievedChunk], list[RetrievedChunk]]:
    """Pure selection. `candidates` are in fused order; `scores` are keyed by sent index.

    Returns (final results by rerank_rank, omitted candidates in fused order).
    """
    sent = candidates[:send_limit]
    cut_before = candidates[send_limit:]
    order = sorted(range(len(sent)), key=lambda i: (-scores[i], sent[i].fused_rank or 0))
    final: list[RetrievedChunk] = []
    cut_after: list[RetrievedChunk] = []
    for rank, i in enumerate(order, start=1):
        chunk = sent[i].model_copy(update={"rerank_score": scores[i], "rerank_rank": rank})
        if rank <= final_count:
            chunk.score = chunk.rerank_score
            final.append(chunk)
        else:
            chunk.omitted_reason = BELOW_LIMIT
            cut_after.append(chunk)
    cut_before = [c.model_copy(update={"omitted_reason": NOT_SENT}) for c in cut_before]
    omitted = sorted(cut_before + cut_after, key=lambda c: c.fused_rank or 0)
    return final, omitted


def run_hybrid_reranked(request: QueryRequest) -> QueryResult:
    settings = get_settings()
    validate_settings(settings)

    inner = hybrid.run_hybrid(
        request.model_copy(update={"pattern": "hybrid", "limit": settings.rerank_candidate_limit})
    )
    trace: dict[str, Any] = {
        "mode": "hybrid-reranked",
        "query": request.question,
        "filters": inner.trace.get("filters"),
        "caller_id": inner.trace.get("caller_id"),
        "result_count": 0,
        "hybrid": {
            k: inner.trace.get(k)
            for k in (
                "embedding", "semantic", "keyword", "fusion", "contribution", "unresolved_hits"
            )
        },
    }
    if not inner.results:
        return QueryResult(
            pattern="hybrid-reranked",
            status="no_results",
            message="No passages matched. " + SCORE_MESSAGE,
            trace=trace,
        )

    candidates = inner.results
    sent = candidates[: settings.rerank_send_limit]
    started = time.monotonic()
    try:
        scores, usage_tokens = call_provider(
            settings, request.question, [_document(c) for c in sent]
        )
    except _ProviderError:
        raise RetrievalError(502, "retrieval_upstream_error", UPSTREAM_MESSAGE) from None
    latency_ms = round((time.monotonic() - started) * 1000)
    if len(scores) != len(sent):
        raise RetrievalError(502, "retrieval_upstream_error", UPSTREAM_MESSAGE)

    final_count = min(request.limit, settings.rerank_return_limit)
    results, omitted = select(candidates, scores, settings.rerank_send_limit, final_count)
    rerank_trace: dict[str, Any] = {
        "model": settings.rerank_model_name,
        "candidate_limit": settings.rerank_candidate_limit,
        "send_limit": settings.rerank_send_limit,
        "return_limit": settings.rerank_return_limit,
        "candidates": len(candidates),
        "sent": len(sent),
        "returned": len(results),
        "omitted_before": len(candidates) - len(sent),
        "omitted_after": len(sent) - len(results),
        "latency_ms": latency_ms,
    }
    if usage_tokens is not None:
        rerank_trace["usage_tokens"] = usage_tokens
    trace["rerank"] = rerank_trace
    trace["result_count"] = len(results)
    return QueryResult(
        pattern="hybrid-reranked",
        status="ok",
        message=SCORE_MESSAGE,
        trace=trace,
        results=results,
        omitted_candidates=omitted,
    )
