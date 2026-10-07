"""Semantic retrieval: embed the question, run Atlas $vectorSearch, resolve source passages."""

from __future__ import annotations

from typing import Any

from pymongo import MongoClient
from pymongo.errors import PyMongoError
from voyageai import Client as VoyageClient
from voyageai.error import VoyageError

from building_with_rag.config import get_settings
from building_with_rag.ingestion import mongodb_schema as schema
from building_with_rag.models import QueryRequest, QueryResult, RetrievedChunk

INGEST_HINT = "Run: uv run python -m building_with_rag.ingestion.ingest"
MIN_CANDIDATES = 50
MAX_CANDIDATES = 200
CANDIDATE_MULTIPLIER = 10
SERVER_ACCESS_LEVELS = [schema.DEFAULT_ACCESS_LEVEL]
SCORE_MESSAGE = (
    "Scores rank similarity only; they do not prove a passage is correct or answers the question."
)

_mongo: MongoClient | None = None
_voyage: VoyageClient | None = None
_ready = False


class RetrievalError(Exception):
    def __init__(self, status_code: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message


def reset() -> None:
    """Drop cached clients and readiness (tests, credential changes)."""
    global _mongo, _voyage, _ready
    _mongo = _voyage = None
    _ready = False


def num_candidates(limit: int) -> int:
    """At least `limit` and 50, at most 200; 10x limit in between."""
    return min(MAX_CANDIDATES, max(limit, MIN_CANDIDATES, limit * CANDIDATE_MULTIPLIER))


def validate_scope(request: QueryRequest) -> list[str]:
    """Return rejection messages for fields semantic mode cannot honour."""
    settings = get_settings()
    problems = []
    if request.caller_id is not None and request.caller_id != settings.webui_demo_caller_id:
        problems.append("caller_id does not match the configured caller.")
    if request.required_acts is not None:
        problems.append("required_acts is not supported in semantic mode.")
    if request.chapter is not None:
        problems.append("chapter is not supported in semantic mode.")
    return problems


def effective_filters(request: QueryRequest) -> dict[str, list[str]]:
    """Server-fixed access_level; caller lists only narrow (empty = no narrowing)."""
    f = request.filters
    out: dict[str, list[str]] = {"access_level": list(SERVER_ACCESS_LEVELS)}
    if f and f.act:
        out["act"] = list(dict.fromkeys(f.act))
    if f and f.status:
        out["status"] = list(dict.fromkeys(f.status))
    if f and f.access_level:
        out["access_level"] = [v for v in SERVER_ACCESS_LEVELS if v in f.access_level]
    return out


def _mongo_filter(filters: dict[str, list[str]]) -> dict[str, Any]:
    return {"$and": [{k: {"$in": v}} for k, v in filters.items()]}


def _not_ready(message: str) -> RetrievalError:
    return RetrievalError(503, "retrieval_not_ready", message)


def _db(settings):
    global _mongo
    if _mongo is None:
        _mongo = MongoClient(
            settings.mongodb_uri,
            serverSelectionTimeoutMS=5000,
            connectTimeoutMS=5000,
            socketTimeoutMS=15000,
        )
    testing = settings.app_env == "testing"
    name = settings.mongodb_test_db_name if testing else settings.mongodb_db_name
    return _mongo[name]


def _voyage_client(settings) -> VoyageClient:
    global _voyage
    if _voyage is None:
        _voyage = VoyageClient(api_key=settings.voyage_api_key, max_retries=1, timeout=15)
    return _voyage


def _ensure_ready(db) -> None:
    global _ready
    if _ready:
        return
    emb = db[schema.EMBEDDINGS_COLLECTION]
    indexes = emb.list_search_indexes()
    idx = next((i for i in indexes if i.get("name") == schema.VECTOR_INDEX_NAME), None)
    if idx is None:
        raise _not_ready(f"Vector index is missing. {INGEST_HINT}")
    if not (idx.get("queryable") or idx.get("status") == "READY"):
        raise _not_ready("Vector index is not ready yet. Wait for READY and retry.")
    sample = emb.find_one({}, {"model": 1, "model_version": 1, "dimensions": 1})
    if sample is None:
        raise _not_ready(f"Embeddings collection is empty. {INGEST_HINT}")
    if (
        sample.get("model") != schema.EMBEDDING_MODEL
        or sample.get("model_version") != schema.EMBEDDING_MODEL_VERSION
        or sample.get("dimensions") != schema.EMBEDDING_DIMENSIONS
    ):
        raise _not_ready("Stored embeddings do not match the configured model/dimensions.")
    _ready = True


def _embed_query(settings, question: str) -> list[float]:
    result = _voyage_client(settings).embed(
        texts=[question], model=schema.EMBEDDING_MODEL, input_type="query"
    )
    vector = result.embeddings[0]
    if len(vector) != schema.EMBEDDING_DIMENSIONS:
        raise _not_ready(
            f"Query embedding has {len(vector)} dimensions; expected {schema.EMBEDDING_DIMENSIONS}."
        )
    return vector


def _to_chunk(hit: dict, chunk: dict, section: dict) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk["chunk_id"],
        section_id=chunk["section_id"],
        act=chunk.get("act") or section.get("act") or "",
        text=chunk.get("text") or "",
        heading=section.get("heading") or "",
        score=hit.get("score"),
        chunk_index=chunk.get("chunk_index"),
        act_label=section.get("act_label"),
        status=section.get("status") or chunk.get("status"),
        chapter=section.get("chapter"),
        chapter_title=section.get("chapter_title"),
        section_number=section.get("section_number"),
        source_pdf=section.get("source_pdf"),
        source_sha256=section.get("source_sha256"),
        needs_review=section.get("needs_review"),
    )


def run_semantic(request: QueryRequest) -> QueryResult:
    settings = get_settings()
    if not settings.mongodb_uri or not settings.voyage_api_key:
        raise _not_ready("MONGODB_URI and VOYAGE_API_KEY must be set in .env.")

    filters = effective_filters(request)
    limit = request.limit
    candidates = num_candidates(limit)
    ignored = []
    if request.generate_answer:
        ignored.append("generate_answer: no answer is generated in semantic retrieval.")
    trace: dict[str, Any] = {
        "mode": "semantic",
        "query": request.question,
        "embedding": {
            "model": schema.EMBEDDING_MODEL,
            "input_type": "query",
            "dimensions": schema.EMBEDDING_DIMENSIONS,
        },
        "index": schema.VECTOR_INDEX_NAME,
        "limit": limit,
        "num_candidates": candidates,
        "filters": filters,
        "caller_id": settings.webui_demo_caller_id,
        "result_count": 0,
        "ignored": ignored,
        "unresolved_hits": 0,
    }

    try:
        db = _db(settings)
        _ensure_ready(db)
        mongo_filter = _mongo_filter(filters)
        emb = db[schema.EMBEDDINGS_COLLECTION]

        if emb.find_one(mongo_filter, {"_id": 1}) is None:
            return QueryResult(
                pattern="semantic",
                status="no_results",
                message="No indexed passages match the filters. " + SCORE_MESSAGE,
                trace=trace,
            )

        vector = _embed_query(settings, request.question)
        hits = list(
            emb.aggregate(
                [
                    {
                        "$vectorSearch": {
                            "index": schema.VECTOR_INDEX_NAME,
                            "path": "vector",
                            "queryVector": vector,
                            "numCandidates": candidates,
                            "limit": limit,
                            "filter": mongo_filter,
                        }
                    },
                    {
                        "$project": {
                            "_id": 0,
                            "chunk_id": 1,
                            "score": {"$meta": "vectorSearchScore"},
                        }
                    },
                ]
            )
        )
        chunks = {
            c["_id"]: c
            for c in db[schema.CHUNKS_COLLECTION].find(
                {"_id": {"$in": [h["chunk_id"] for h in hits]}}
            )
        }
        section_ids = list({c["section_id"] for c in chunks.values()})
        found = db[schema.SECTIONS_COLLECTION].find({"_id": {"$in": section_ids}})
        sections = {s["_id"]: s for s in found}
    except RetrievalError:
        raise
    except (PyMongoError, VoyageError) as exc:
        global _ready
        _ready = False
        raise RetrievalError(
            502, "retrieval_upstream_error", f"Upstream error ({type(exc).__name__}); retry later."
        ) from None

    results = []
    for hit in hits:
        chunk = chunks.get(hit["chunk_id"])
        section = sections.get(chunk["section_id"]) if chunk else None
        if chunk is None or section is None:
            trace["unresolved_hits"] += 1
            continue
        results.append(_to_chunk(hit, chunk, section))

    if hits and not results:
        raise _not_ready("Vector hits could not be resolved to chunks/sections.")

    trace["result_count"] = len(results)
    return QueryResult(
        pattern="semantic",
        status="ok" if results else "no_results",
        message=SCORE_MESSAGE if results else "No passages matched. " + SCORE_MESSAGE,
        trace=trace,
        results=results,
    )
