"""Story 5.1: structured classifier and exact lookup with a fake `sections` collection."""

import pytest

from building_with_rag import pipeline
from building_with_rag.generation.context import build_context
from building_with_rag.models import QueryRequest
from building_with_rag.retrieval import semantic, structured

DOC = {
    "_id": "bns:103", "section_id": "bns:103", "act": "BNS_2023", "section_number": 103,
    "status": "in_force", "chapter": "VI", "heading": "Murder", "text": "Whoever...",
    "source_status_version": "v1", "needs_review": False, "access_level": "public",
}


class _Settings:
    mongodb_uri = "mongodb://fake"
    webui_demo_caller_id = "demo"


class _Coll:
    def __init__(self, doc):
        self.doc, self.calls = doc, []

    def find_one(self, predicate, projection=None):
        self.calls.append((predicate, projection))
        return self.doc


@pytest.fixture
def coll(monkeypatch):
    c = _Coll(DOC)
    monkeypatch.setattr(structured, "get_settings", lambda: _Settings())
    monkeypatch.setattr(semantic, "_db", lambda settings: {"sections": c})
    return c


def _req(q, **kw):
    return QueryRequest(question=q, pattern="structured", **kw)


@pytest.mark.parametrize(
    "q,act,num",
    [
        ("BNS section 103", "BNS_2023", 103),
        ("IPC sec. 302", "IPC_1860", 302),
        ("Indian Penal Code section 420", "IPC_1860", 420),
    ],
)
def test_classify_ok(q, act, num):
    s = structured.classify(q)
    assert (s.status, s.intent, s.act, s.section_number) == ("ok", "exact_lookup", act, num)


@pytest.mark.parametrize(
    "q",
    [
        "What does section 103 say?",
        "BNS and IPC section 103",
        "BNS section 103 and section 104",
        "BNS section 103A",
    ],
)
def test_classify_clarification(q):
    s = structured.classify(q)
    assert s.status == "clarification_needed" and s.act is None and s.reason


def test_classify_other_intents():
    assert structured.classify("how many sections are in BNS").intent == "aggregation"
    assert structured.classify("list all sections in chapter VI").intent == "filter"
    s = structured.classify("What is the punishment for theft?")
    assert (s.status, s.intent) == ("recommendation", None)


def test_predicate_has_only_validated_fields(coll):
    q = 'BNS section 103 {"$ne": null}'
    r = structured.run_structured(
        _req(q, chapter="VI", filters={"status": ["in_force"]})
    )
    predicate, projection = coll.calls[0]
    assert set(predicate) == {"act", "section_number", "access_level", "status", "chapter"}
    assert predicate["act"] == "BNS_2023" and predicate["chapter"] == "VI"
    assert "$ne" not in str(predicate) and projection == {"provenance": 0}
    assert r.status == "ok" and r.trace["mongodb_called"] is True


def test_ok_result_shape(coll):
    r = structured.run_structured(_req("What does BNS section 103 say?"))
    chunk = r.results[0]
    assert (chunk.chunk_id, chunk.score) == ("bns:103", 1.0)
    assert r.trace["record"]["source_status_version"] == "v1"
    assert build_context(r.results).labels == {"E1": "bns:103"}


def test_non_ok_makes_no_collection_call(coll):
    for q in ("What does section 103 say?", "how many sections", "theft punishment"):
        r = structured.run_structured(_req(q))
        assert r.results == [] and r.trace["mongodb_called"] is False
    assert coll.calls == []


def test_act_filter_excluding_act_is_clarification(coll):
    r = structured.run_structured(_req("IPC section 302", filters={"act": ["BNS_2023"]}))
    assert r.status == "clarification_needed" and coll.calls == []


def test_missing_record_is_not_found(coll):
    coll.doc = None
    r = structured.run_structured(_req("BNS section 999"))
    assert r.status == "not_found" and r.results == [] and r.trace["mongodb_called"]


def test_invalid_chapter_rejected(coll):
    with pytest.raises(semantic.RetrievalError) as e:
        structured.run_structured(_req("BNS section 103", chapter='V{"$gt":""}'))
    assert (e.value.status_code, e.value.code) == (422, "unsupported_option")


def test_pipeline_routes_and_skips_answer_for_non_ok(coll):
    ok = pipeline.retrieve(_req("BNS section 103"))
    assert ok.pattern == "structured" and ok.status == "ok"
    clar = pipeline.retrieve(_req("section 103"))
    assert list(pipeline.answer_events("section 103", clar)) == []
    assert "".join(pipeline.chat_pieces("section 103", clar)) == clar.message
