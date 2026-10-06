"""Shared mode registry and the single dispatch path for all RAG patterns."""

from __future__ import annotations

from building_with_rag.models import QueryRequest, QueryResult

MODE_BY_MODEL_ID: dict[str, str] = {
    "rag-semantic": "semantic",
    "rag-hybrid": "hybrid",
    "rag-hybrid-reranked": "hybrid-reranked",
    "rag-structured": "structured",
    "rag-decomposition": "decomposition",
    "rag-hyde": "hyde",
}

MODEL_ID_BY_MODE: dict[str, str] = {mode: model_id for model_id, mode in MODE_BY_MODEL_ID.items()}


def run_pattern(request: QueryRequest) -> QueryResult:
    """Single dispatch path for every RAG mode.

    Each mode is an honest not_implemented placeholder until its own story
    adds behavior.
    """
    if request.pattern not in MODEL_ID_BY_MODE:
        return QueryResult(
            pattern=request.pattern,
            status="error",
            message=f"Unknown pattern '{request.pattern}'.",
            trace={},
            results=[],
        )

    return QueryResult(
        pattern=request.pattern,
        status="not_implemented",
        message=f"Pattern '{request.pattern}' is not implemented yet.",
        trace={"mode": request.pattern, "model_id": MODEL_ID_BY_MODE[request.pattern]},
        results=[],
    )
