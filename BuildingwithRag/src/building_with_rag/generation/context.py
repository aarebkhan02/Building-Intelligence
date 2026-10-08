"""Bounded, labelled evidence context built from a retrieval result (any real mode)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from building_with_rag.models import RetrievedChunk

MAX_PASSAGES = 5
MAX_CHARS = 12_000


@dataclass
class Context:
    entries: list[dict[str, Any]] = field(default_factory=list)
    passages: dict[str, RetrievedChunk] = field(default_factory=dict)
    omitted: int = 0
    chars: int = 0

    @property
    def outcome(self) -> str:
        return "assembled" if self.entries else "empty"

    @property
    def labels(self) -> dict[str, str]:
        return {e["label"]: e["chunk_id"] for e in self.entries}


def build_context(results: list[RetrievedChunk]) -> Context:
    """Take passages in score order until a budget is hit; never cut a passage."""
    ctx = Context()
    for chunk in results:
        if len(ctx.entries) >= MAX_PASSAGES or ctx.chars + len(chunk.text) > MAX_CHARS:
            break
        label = f"E{len(ctx.entries) + 1}"
        ctx.entries.append(
            {
                "label": label,
                "chunk_id": chunk.chunk_id,
                "section_id": chunk.section_id,
                "act": chunk.act,
                "act_label": chunk.act_label,
                "heading": chunk.heading,
                "chapter": chunk.chapter,
                "section_number": chunk.section_number,
                "status": chunk.status,
                "source_pdf": chunk.source_pdf,
                "needs_review": chunk.needs_review,
                "text": chunk.text,
            }
        )
        ctx.passages[label] = chunk
        ctx.chars += len(chunk.text)
    ctx.omitted = len(results) - len(ctx.entries)
    return ctx
