"""Shared typed contracts for building-with-rag.

These names are fixed by docs/architecture.md and docs/stories/story-1-1-*.
Later stories extend them additively and never replace them with
simplified alternatives or parallel top-level fields.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class SemanticFilters(BaseModel):
    act: list[str] = Field(default_factory=list)
    status: list[str] = Field(default_factory=list)
    access_level: list[str] = Field(default_factory=list)


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    pattern: str
    caller_id: str | None = None
    filters: SemanticFilters | None = None
    limit: int = Field(default=5, ge=1, le=20)
    generate_answer: bool = False
    required_acts: list[str] | None = None
    chapter: str | None = None


class RetrievedChunk(BaseModel):
    chunk_id: str
    section_id: str
    act: str
    text: str
    heading: str | None = None
    score: float | None = None
    source_file: str | None = None
    source_page: int | None = None
    # Additive fields named by later stories only.
    semantic_score: float | None = None
    semantic_rank: int | None = None
    keyword_score: float | None = None
    keyword_rank: int | None = None
    fused_score: float | None = None
    fused_rank: int | None = None
    rerank_score: float | None = None
    rerank_rank: int | None = None


class OmittedCandidate(BaseModel):
    chunk_id: str
    omitted_reason: str


class SubquestionEvidence(BaseModel):
    subquestion: str
    status: Literal["evidenced", "no_evidence"]
    results: list[RetrievedChunk] = Field(default_factory=list)
    reason: str | None = None


class StructuredSignals(BaseModel):
    intent: Literal["exact_lookup", "filter", "aggregation"]
    act: str | None = None
    section_number: str | None = None
    chapter: str | None = None


class GenerationResult(BaseModel):
    outcome: Literal["answered", "insufficient_evidence", "unavailable", "malformed"]
    answer: str | None = None
    claims: list[str] = Field(default_factory=list)
    citations: list[str] = Field(default_factory=list)
    supporting_passages: list[RetrievedChunk] = Field(default_factory=list)
    provider: str | None = None
    model: str | None = None
    trace: dict[str, Any] = Field(default_factory=dict)
    context_outcome: str | None = None
    confidence: float | None = None
    draft_answer: str | None = None
    issues: list[str] = Field(default_factory=list)
    attempts: int = 0
    low_confidence_reason: str | None = None


class QueryResult(BaseModel):
    pattern: str
    status: str
    message: str | None = None
    trace: dict[str, Any] = Field(default_factory=dict)
    results: list[RetrievedChunk] = Field(default_factory=list)
    generation: GenerationResult | None = None
    # Additive fields; default to empty.
    omitted_candidates: list[OmittedCandidate] = Field(default_factory=list)
    subquestions: list[SubquestionEvidence] = Field(default_factory=list)
    hyde_direct_candidates: list[RetrievedChunk] = Field(default_factory=list)
    hyde_query_candidates: list[RetrievedChunk] = Field(default_factory=list)
    hyde_hypothetical_text_debug: str | None = None


class ChatMessage(BaseModel):
    role: Literal["system", "developer", "user", "assistant"]
    content: str


class RagOptions(BaseModel):
    """Strict: Open WebUI never sends identity, access level, or generation settings here."""

    model_config = ConfigDict(extra="forbid")

    pattern: str | None = None
    filters: SemanticFilters | None = None
    limit: int | None = Field(default=None, ge=1, le=20)
    required_acts: list[str] | None = None
    chapter: str | None = None


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    stream: bool = False
    n: int = 1
    rag_options: RagOptions | None = None
