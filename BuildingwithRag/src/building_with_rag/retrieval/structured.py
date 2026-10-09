"""Structured exact retrieval: rule-based classifier + one read-only `sections` lookup."""

from __future__ import annotations

import logging
import re
from typing import Any

from pymongo.errors import PyMongoError

from building_with_rag.config import get_settings
from building_with_rag.ingestion import mongodb_schema as schema
from building_with_rag.models import QueryRequest, QueryResult, RetrievedChunk, StructuredSignals
from building_with_rag.retrieval import semantic

log = logging.getLogger(__name__)

SOURCE_MESSAGE = (
    "Exact record from the supplied corpus. status and source_status_version describe the "
    "source document and do not claim current legal applicability."
)
_CHAPTER_RE = re.compile(r"[A-Za-z0-9 .\-]{1,40}")
_AGGREGATION_RE = re.compile(r"\b(how many|count|total number)\b", re.I)
_FILTER_RE = re.compile(r"\b(list|all sections|which sections|sections? (?:in|under))\b", re.I)
_SECTION_RE = re.compile(r"(?:\bsections?|\bsec\.?|\bs\.|§)\s*(\d+)([A-Za-z]?)", re.I)
_BNS_RE = re.compile(r"\bBNS\b|Bharatiya\s+Nyaya\s+Sanhita", re.I)
_IPC_RE = re.compile(r"\bIPC\b|Indian\s+Penal\s+Code", re.I)
_PROJECTION = {"provenance": 0}


def validate_chapter(chapter: str | None) -> None:
    if chapter is not None and not _CHAPTER_RE.fullmatch(chapter):
        raise semantic.RetrievalError(
            422,
            "unsupported_option",
            "chapter must be 1-40 characters of letters, digits, spaces, '.' and '-'.",
        )


def validate_scope(request: QueryRequest) -> list[str]:
    """Like semantic.validate_scope, but chapter is accepted."""
    problems = []
    if request.caller_id is not None and request.caller_id != get_settings().webui_demo_caller_id:
        problems.append("caller_id does not match the configured caller.")
    if request.required_acts is not None:
        problems.append("required_acts is not supported in structured mode.")
    return problems


def _signals(status: str, reason: str, **kw: Any) -> StructuredSignals:
    return StructuredSignals(status=status, reason=reason, **kw)


def classify(question: str, chapter: str | None = None) -> StructuredSignals:
    """Pure rule-based classification; acts and sections are never guessed."""
    if _AGGREGATION_RE.search(question):
        return _signals(
            "recommendation",
            "Counting questions are recognised but not executed in structured mode.",
            intent="aggregation",
            chapter=chapter,
        )
    if _FILTER_RE.search(question):
        return _signals(
            "recommendation",
            "Listing/filtering questions are recognised but not executed in structured mode.",
            intent="filter",
            chapter=chapter,
        )

    refs = _SECTION_RE.findall(question)
    if not refs:
        return _signals(
            "recommendation",
            "Structured mode looks up one named section (e.g. 'BNS section 103'). "
            "For open questions use semantic or hybrid mode.",
            chapter=chapter,
        )

    def clarify(reason: str) -> StructuredSignals:
        return _signals("clarification_needed", reason, intent="exact_lookup", chapter=chapter)

    suffixed = [n + s for n, s in refs if s]
    if suffixed:
        return clarify(
            f"Section '{suffixed[0]}' is not a plain integer section number; "
            "only integer sections are supported."
        )
    numbers = list(dict.fromkeys(int(n) for n, _ in refs))
    if len(numbers) > 1:
        return clarify("Several section numbers were given; ask for one section at a time.")
    if not 1 <= numbers[0] <= 999:
        return clarify("The section number must be between 1 and 999.")

    has_bns, has_ipc = bool(_BNS_RE.search(question)), bool(_IPC_RE.search(question))
    if has_bns and has_ipc:
        return clarify("Both BNS and IPC were named; ask for one act at a time.")
    if not (has_bns or has_ipc):
        return clarify("The act is missing: say BNS or IPC (the act is never guessed).")
    return _signals(
        "ok",
        "Exact section lookup.",
        intent="exact_lookup",
        act="BNS_2023" if has_bns else "IPC_1860",
        section_number=numbers[0],
        chapter=chapter,
    )


def _result(
    status: str, message: str, trace: dict[str, Any], results: list[RetrievedChunk] | None = None
) -> QueryResult:
    trace["result_count"] = len(results or [])
    return QueryResult(
        pattern="structured", status=status, message=message, trace=trace, results=results or []
    )


def _to_chunk(doc: dict) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=doc["section_id"],
        section_id=doc["section_id"],
        act=doc.get("act") or "",
        text=doc.get("text") or "",
        heading=doc.get("heading") or "",
        score=1.0,  # exact-match marker, not a similarity
        act_label=doc.get("act_label"),
        status=doc.get("status"),
        chapter=doc.get("chapter"),
        chapter_title=doc.get("chapter_title"),
        section_number=doc.get("section_number"),
        source_pdf=doc.get("source_pdf"),
        source_sha256=doc.get("source_sha256"),
        needs_review=doc.get("needs_review"),
    )


def run_structured(request: QueryRequest) -> QueryResult:
    settings = get_settings()
    problems = validate_scope(request)
    if problems:
        raise semantic.RetrievalError(422, "unsupported_option", " ".join(problems))
    validate_chapter(request.chapter)

    signals = classify(request.question, request.chapter)
    filters = semantic.effective_filters(request)
    trace: dict[str, Any] = {
        "mode": "structured",
        "signals": signals.model_dump(),
        "mongodb_called": False,
        "collection": schema.SECTIONS_COLLECTION,
        "filters": filters,
        "caller_id": settings.webui_demo_caller_id,
        "result_count": 0,
    }

    if signals.status == "ok" and "act" in filters and signals.act not in filters["act"]:
        signals = _signals(
            "clarification_needed",
            f"The question names {signals.act} but the act filter excludes it.",
            intent="exact_lookup",
            chapter=request.chapter,
        )
        trace["signals"] = signals.model_dump()
    if signals.status != "ok":
        return _result(signals.status, signals.reason, trace)

    if not settings.mongodb_uri:
        raise semantic._not_ready("MONGODB_URI must be set in .env.")

    predicate: dict[str, Any] = {
        "act": signals.act,
        "section_number": signals.section_number,
        "access_level": {"$in": filters["access_level"]},
    }
    if "status" in filters:
        predicate["status"] = {"$in": filters["status"]}
    if signals.chapter is not None:
        predicate["chapter"] = signals.chapter

    trace["mongodb_called"] = True
    try:
        doc = semantic._db(settings)[schema.SECTIONS_COLLECTION].find_one(predicate, _PROJECTION)
    except PyMongoError as exc:
        log.warning("structured lookup failed: %s", type(exc).__name__)
        raise semantic.RetrievalError(
            502, "retrieval_upstream_error", f"Upstream error ({type(exc).__name__}); retry later."
        ) from None

    if doc is None:
        return _result(
            "not_found",
            "This corpus has no matching record. That says nothing about whether the law "
            "has such a section.",
            trace,
        )
    trace["record"] = {
        "section_id": doc["section_id"],
        "status": doc.get("status"),
        "source_status_version": doc.get("source_status_version"),
    }
    return _result("ok", SOURCE_MESSAGE, trace, [_to_chunk(doc)])
