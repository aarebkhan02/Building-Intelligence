"""Hybrid retrieval: Atlas Search keyword route + semantic route, fused with RRF."""

from __future__ import annotations

from typing import Any

from pymongo.errors import PyMongoError
from voyageai.error import VoyageError

from building_with_rag.config import get_settings
from building_with_rag.ingestion import mongodb_schema as schema
from building_with_rag.models import QueryRequest, QueryResult
from building_with_rag.retrieval import semantic
from building_with_rag.retrieval.semantic import RetrievalError

RRF_K = 60
# BM25 length-normalises toward heading-only stubs; keep them out of the keyword route.
MIN_KEYWORD_CHARS = 100
KEYWORD_HINT = "Run: uv run python -m building_with_rag.ingestion.keyword_index"
SCORE_MESSAGE = (
    "Fused score ranks only (reciprocal rank fusion of semantic and keyword ranks); "
    "it does not prove a passage is correct or answers the question."
)
_ROUTE_FIELDS = (
    "semantic_score",
    "semantic_rank",
    "keyword_score",
    "keyword_rank",
    "fused_score",
    "fused_rank",
)

_ready = False


def reset() -> None:
    global _ready
    _ready = False


def route_depth(limit: int) -> int:
    return max(limit, min(50, max(20, 4 * limit)))


def fuse(semantic_hits: list[dict], keyword_hits: list[dict], limit: int) -> list[dict]:
    """RRF over ranks. Hits are {chunk_id, score}, best first. No I/O."""
    fused: dict[str, dict[str, Any]] = {}
    for route, hits in (("semantic", semantic_hits), ("keyword", keyword_hits)):
        for rank, hit in enumerate(hits, start=1):
            entry = fused.setdefault(
                hit["chunk_id"],
                {
                    "chunk_id": hit["chunk_id"],
                    "semantic_score": None,
                    "semantic_rank": None,
                    "keyword_score": None,
                    "keyword_rank": None,
                    "fused_score": 0.0,
                },
            )
            if entry[f"{route}_rank"] is not None:
                continue  # duplicate hit within one route: keep the best rank
            entry[f"{route}_score"] = hit.get("score")
            entry[f"{route}_rank"] = rank
            entry["fused_score"] += 1.0 / (RRF_K + rank)
    ordered = sorted(
        fused.values(),
        key=lambda e: (
            -e["fused_score"],
            e["semantic_rank"] if e["semantic_rank"] is not None else float("inf"),
            e["chunk_id"],
        ),
    )
    for position, entry in enumerate(ordered, start=1):
        entry["fused_rank"] = position
    return ordered[:limit]


def _ensure_ready(db) -> None:
    global _ready
    if _ready:
        return
    semantic.check_vector_ready(db)
    chunks = db[schema.CHUNKS_COLLECTION]
    idx = next(
        (i for i in chunks.list_search_indexes() if i.get("name") == schema.KEYWORD_INDEX_NAME),
        None,
    )
    if idx is None:
        raise RetrievalError(
            503, "retrieval_not_ready", f"Keyword index is missing. {KEYWORD_HINT}"
        )
    if not (idx.get("queryable") or idx.get("status") == "READY"):
        raise RetrievalError(
            503, "retrieval_not_ready", "Keyword index is not ready yet. Wait for READY and retry."
        )
    _ready = True


def keyword_hits(chunks, question: str, filters: dict[str, list[str]], depth: int) -> list[dict]:
    """Atlas Search `text` operator on chunks.text; hits are {chunk_id, section_id, score}."""
    return list(
        chunks.aggregate(
            [
                {
                    "$search": {
                        "index": schema.KEYWORD_INDEX_NAME,
                        "compound": {
                            "must": [{"text": {"query": question, "path": "text"}}],
                            "filter": [
                                {"in": {"path": path, "value": values}}
                                for path, values in filters.items()
                            ],
                        },
                    }
                },
                {"$match": {"$expr": {"$gte": [{"$strLenCP": "$text"}, MIN_KEYWORD_CHARS]}}},
                {"$limit": depth},
                {
                    "$project": {
                        "_id": 0,
                        "chunk_id": 1,
                        "section_id": 1,
                        "score": {"$meta": "searchScore"},
                    }
                },
            ]
        )
    )


def run_hybrid(request: QueryRequest) -> QueryResult:
    settings = get_settings()
    if not settings.mongodb_uri or not settings.voyage_api_key:
        raise RetrievalError(
            503, "retrieval_not_ready", "MONGODB_URI and VOYAGE_API_KEY must be set in .env."
        )

    filters = semantic.effective_filters(request)
    limit = request.limit
    depth = route_depth(limit)
    candidates = semantic.num_candidates(depth)
    trace: dict[str, Any] = {
        "mode": "hybrid",
        "query": request.question,
        "embedding": {
            "model": schema.EMBEDDING_MODEL,
            "input_type": "query",
            "dimensions": schema.EMBEDDING_DIMENSIONS,
        },
        "filters": filters,
        "caller_id": settings.webui_demo_caller_id,
        "result_count": 0,
        "unresolved_hits": 0,
        "semantic": {
            "index": schema.VECTOR_INDEX_NAME,
            "limit": depth,
            "num_candidates": candidates,
            "hit_count": 0,
        },
        "keyword": {
            "index": schema.KEYWORD_INDEX_NAME,
            "path": "text",
            "operator": "text",
            "min_chars": MIN_KEYWORD_CHARS,
            "limit": depth,
            "hit_count": 0,
        },
        "fusion": {
            "method": "rrf",
            "k": RRF_K,
            "weights": {"semantic": 1.0, "keyword": 1.0},
            "route_depth": depth,
        },
        "contribution": {"both": 0, "semantic_only": 0, "keyword_only": 0},
    }

    try:
        db = semantic._db(settings)
        _ensure_ready(db)
        mongo_filter = semantic._mongo_filter(filters)
        emb = db[schema.EMBEDDINGS_COLLECTION]

        if emb.find_one(mongo_filter, {"_id": 1}) is None:
            return QueryResult(
                pattern="hybrid",
                status="no_results",
                message="No indexed passages match the filters. " + SCORE_MESSAGE,
                trace=trace,
            )

        vector = semantic._embed_query(settings, request.question)
        sem_hits = semantic.vector_hits(emb, vector, candidates, depth, mongo_filter)
        kw_hits = keyword_hits(db[schema.CHUNKS_COLLECTION], request.question, filters, depth)
        trace["semantic"]["hit_count"] = len(sem_hits)
        trace["keyword"]["hit_count"] = len(kw_hits)

        fused = fuse(sem_hits, kw_hits, limit)
        chunks = {
            c["_id"]: c
            for c in db[schema.CHUNKS_COLLECTION].find(
                {"_id": {"$in": [e["chunk_id"] for e in fused]}}
            )
        }
        section_ids = list({c["section_id"] for c in chunks.values()})
        found = db[schema.SECTIONS_COLLECTION].find({"_id": {"$in": section_ids}})
        sections = {s["_id"]: s for s in found}
    except RetrievalError:
        raise
    except (PyMongoError, VoyageError) as exc:
        reset()
        raise RetrievalError(
            502, "retrieval_upstream_error", f"Upstream error ({type(exc).__name__}); retry later."
        ) from None

    results = []
    for entry in fused:
        chunk = chunks.get(entry["chunk_id"])
        section = sections.get(chunk["section_id"]) if chunk else None
        if chunk is None or section is None:
            trace["unresolved_hits"] += 1
            continue
        item = semantic._to_chunk({"score": entry["fused_score"]}, chunk, section)
        for key in _ROUTE_FIELDS:
            setattr(item, key, entry[key])
        results.append(item)
        if entry["semantic_rank"] is not None and entry["keyword_rank"] is not None:
            trace["contribution"]["both"] += 1
        elif entry["semantic_rank"] is not None:
            trace["contribution"]["semantic_only"] += 1
        else:
            trace["contribution"]["keyword_only"] += 1

    if fused and not results:
        raise RetrievalError(
            503, "retrieval_not_ready", "Hybrid hits could not be resolved to chunks/sections."
        )

    trace["result_count"] = len(results)
    return QueryResult(
        pattern="hybrid",
        status="ok" if results else "no_results",
        message=SCORE_MESSAGE if results else "No passages matched. " + SCORE_MESSAGE,
        trace=trace,
        results=results,
    )
