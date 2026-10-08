"""Story 4.1: hybrid fusion and run_hybrid with faked searches (no network)."""

import pytest

from building_with_rag.models import QueryRequest
from building_with_rag.retrieval import hybrid, semantic


def _hits(*ids):
    return [{"chunk_id": c, "score": 1.0 / (i + 1)} for i, c in enumerate(ids)]


def test_fuse_both_routes_rank_first_with_rrf_score():
    out = hybrid.fuse(_hits("a", "b", "c"), _hits("x", "c"), limit=5)
    assert out[0]["chunk_id"] == "c"
    assert (out[0]["semantic_rank"], out[0]["keyword_rank"]) == (3, 2)
    assert out[0]["fused_score"] == pytest.approx(1 / 63 + 1 / 62)
    assert [e["fused_rank"] for e in out] == list(range(1, len(out) + 1))


def test_fuse_single_route_leaves_other_route_none():
    out = {e["chunk_id"]: e for e in hybrid.fuse(_hits("a"), _hits("x"), limit=5)}
    assert out["a"]["keyword_rank"] is None and out["a"]["keyword_score"] is None
    assert out["x"]["semantic_rank"] is None and out["x"]["semantic_score"] is None


def test_fuse_ties_semantic_rank_then_chunk_id_and_limit():
    # a (sem 1) and x (kw 1) tie on score; semantic rank wins, missing last.
    out = hybrid.fuse(_hits("a"), _hits("x"), limit=1)
    assert [e["chunk_id"] for e in out] == ["a"]
    # both keyword-only, equal rank impossible; equal-score keyword-only vs keyword-only by id
    out = hybrid.fuse([], _hits("m", "n"), limit=5)
    assert [e["chunk_id"] for e in out] == ["m", "n"]
    again = hybrid.fuse(_hits("b", "a"), [], limit=5)
    assert [e["chunk_id"] for e in again] == ["b", "a"]


class _Settings:
    mongodb_uri = "mongodb://fake"
    voyage_api_key = "k"
    webui_demo_caller_id = "demo-public"


class _Coll:
    def __init__(self, docs):
        self.docs = docs

    def find(self, query, *a, **k):
        ids = query["_id"]["$in"]
        return [d for d in self.docs if d["_id"] in ids]

    def find_one(self, *a, **k):
        return {"_id": 1}


class _DB(dict):
    pass


def _db():
    chunks = [
        {"_id": "c1", "chunk_id": "c1", "section_id": "bns:303", "act": "BNS_2023", "text": "t1"},
        {"_id": "c2", "chunk_id": "c2", "section_id": "bns:316", "act": "BNS_2023", "text": "t2"},
    ]
    sections = [
        {"_id": "bns:303", "heading": "Theft", "act": "BNS_2023"},
        {"_id": "bns:316", "heading": "Breach", "act": "BNS_2023"},
    ]
    return _DB(chunks=_Coll(chunks), sections=_Coll(sections), embeddings=_Coll([]))


@pytest.fixture
def fakes(monkeypatch):
    hybrid.reset()
    monkeypatch.setattr(hybrid, "get_settings", lambda: _Settings())
    monkeypatch.setattr(semantic, "_db", lambda settings: _db())
    monkeypatch.setattr(semantic, "_embed_query", lambda settings, q: [0.0])
    monkeypatch.setattr(hybrid, "_ensure_ready", lambda db: None)
    monkeypatch.setattr(semantic, "vector_hits", lambda *a: _hits("c1", "c2"))
    monkeypatch.setattr(hybrid, "keyword_hits", lambda *a: _hits("c2"))
    yield
    hybrid.reset()


def test_run_hybrid_sets_fused_score_and_route_fields(fakes):
    result = hybrid.run_hybrid(QueryRequest(question="theft", pattern="hybrid", limit=2))
    assert result.status == "ok" and result.pattern == "hybrid"
    first, second = result.results
    assert first.chunk_id == "c2" and first.score == first.fused_score
    assert (first.semantic_rank, first.keyword_rank, first.fused_rank) == (2, 1, 1)
    assert second.keyword_rank is None and second.keyword_score is None
    assert result.trace["contribution"] == {"both": 1, "semantic_only": 1, "keyword_only": 0}
    assert result.trace["fusion"]["route_depth"] == hybrid.route_depth(2)


def test_run_hybrid_missing_keyword_index_is_not_ready(monkeypatch):
    hybrid.reset()
    monkeypatch.setattr(hybrid, "get_settings", lambda: _Settings())
    monkeypatch.setattr(semantic, "_db", lambda settings: _db())
    monkeypatch.setattr(semantic, "check_vector_ready", lambda db: None)
    _db_coll = _Coll([])
    _db_coll.list_search_indexes = lambda: []
    monkeypatch.setattr(semantic, "_db", lambda settings: _DB(chunks=_db_coll))
    with pytest.raises(semantic.RetrievalError) as exc:
        hybrid.run_hybrid(QueryRequest(question="theft", pattern="hybrid"))
    assert exc.value.status_code == 503 and exc.value.code == "retrieval_not_ready"
    assert "keyword_index" in exc.value.message
