"""One shared path for /v1/query and /v1/chat/completions: retrieve → answer events."""

from __future__ import annotations

from collections.abc import Iterator

from fastapi import HTTPException

from building_with_rag.generation import answer
from building_with_rag.generation.answer import Event
from building_with_rag.models import GenerationResult, QueryRequest, QueryResult
from building_with_rag.registry import run_pattern
from building_with_rag.retrieval import hybrid, rerank, semantic

REAL_PATTERNS = frozenset({"semantic", "hybrid", "hybrid-reranked"})
_RETRIEVERS = {
    "semantic": semantic.run_semantic,
    "hybrid": hybrid.run_hybrid,
    "hybrid-reranked": rerank.run_hybrid_reranked,
}
UNAVAILABLE_DRAFT = "Answer generation unavailable — the text above is an unchecked draft."
UNAVAILABLE = "Answer generation unavailable."
UNJUDGED_DRAFT = "DRAFT — could not be checked; not the final answer."


def retrieve(request: QueryRequest) -> QueryResult:
    """Semantic/hybrid/hybrid-reranked → real retrieval; other modes → run_pattern.

    Errors raise before any streaming.
    """
    if request.pattern not in REAL_PATTERNS:
        return run_pattern(request)
    problems = semantic.validate_scope(request)
    if problems:
        raise HTTPException(status_code=422, detail=" ".join(problems))
    return _RETRIEVERS[request.pattern](request)  # may raise semantic.RetrievalError


def answer_events(question: str, retrieval: QueryResult) -> Iterator[Event]:
    return answer.answer_stream(question, retrieval)


def final_of(events: Iterator[Event]) -> GenerationResult:
    result = None
    for event in events:
        if event.kind == "final":
            result = event.result
    assert result is not None
    return result


def _source_line(c) -> str:
    act = c.act.split("_")[0]
    parts = [c.label, f"{act} §{c.section_number}" if c.section_number is not None else act]
    if c.heading:
        parts.append(c.heading)
    parts.append(c.section_id)
    return " · ".join(parts)


def footer(g: GenerationResult, streamed_text: bool) -> str:
    """Readable closing text for chat, derived from the same final result."""
    if g.outcome == "answered":
        sources = "\n".join(_source_line(c) for c in g.citations if not isinstance(c, str))
        return f"\n\nEvidence check passed — confidence: high\n\nSources:\n{sources}"
    if g.outcome == "insufficient_evidence":
        reason = str(g.trace.get("model_reason") or g.trace.get("reason") or "").strip()
        reason = reason if len(reason) <= 200 else reason[:199] + "…"
        sentence = "The retrieved passages do not support an answer"
        sentence += f": {reason.rstrip('.')}." if reason else "."
        return ("\n\n" if streamed_text else "") + sentence
    if g.confidence == "low":
        details = "\n".join(
            f"- attempt {i['attempt']} · {i['check']}: {i['detail']}" for i in g.issues[-4:]
        )
        return (
            "\n\nDRAFT — low confidence, not the final answer.\n"
            f"{g.low_confidence_reason}\n{details}"
        )
    if g.outcome == "unavailable":
        if streamed_text:
            return f"\n\n{UNAVAILABLE_DRAFT}"
        return UNAVAILABLE
    if g.draft_answer:
        return f"\n\n{UNJUDGED_DRAFT}"
    return "\n\nThe model reply could not be used; no answer is available."


def chat_pieces(question: str, retrieval: QueryResult) -> Iterator[str]:
    """Text pieces for the chat stream; same operation as /v1/query's generation."""
    if retrieval.pattern not in REAL_PATTERNS:
        yield retrieval.message or "not_implemented"
        return
    streamed = False
    final: GenerationResult | None = None
    try:
        for event in answer_events(question, retrieval):
            if event.kind == "final":
                final = event.result
            elif event.text:
                streamed = True
                yield event.text
    except Exception:  # safety net: never break an open stream
        yield ("\n\n" if streamed else "") + (UNAVAILABLE_DRAFT if streamed else UNAVAILABLE)
        return
    if final is not None:
        yield footer(final, streamed)
