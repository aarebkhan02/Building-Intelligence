"""Parser and context tests for Story 3.1 (no network)."""

import json

from building_with_rag.generation.answer import _Malformed, parse_output
from building_with_rag.generation.context import build_context
from building_with_rag.models import RetrievedChunk


def _ctx(n=2, text="x"):
    return build_context(
        [RetrievedChunk(chunk_id=f"c{i}", section_id=f"bns:{i}", act="BNS_2023", text=text)
         for i in range(n)]
    )


def _bad(content, ctx):
    try:
        parse_output(content, ctx)
    except _Malformed:
        return True
    return False


def test_unknown_label_non_json_and_missing_claims_are_malformed():
    ctx = _ctx()
    unknown = json.dumps({"outcome": "answered", "answer": "a",
                          "claims": [{"text": "t", "evidence": ["E9"]}]})
    no_claims = json.dumps({"outcome": "answered", "answer": "a", "claims": []})
    assert _bad(unknown, ctx)
    assert _bad("not json", ctx)
    assert _bad(no_claims, ctx)


def test_valid_answer_and_fence():
    ctx = _ctx()
    payload = json.dumps({"outcome": "answered", "answer": "a", "reason": "r",
                          "claims": [{"text": "t", "evidence": ["E2", "E1"]}]})
    parsed = parse_output(f"```json\n{payload}\n```", ctx)
    assert parsed["claims"][0].evidence_labels == ["E2", "E1"]


def test_insufficient_must_be_empty():
    ctx = _ctx()
    ok = json.dumps({"outcome": "insufficient_evidence", "answer": "", "claims": []})
    assert parse_output(ok, ctx)["outcome"] == "insufficient_evidence"
    bad = json.dumps({"outcome": "insufficient_evidence", "answer": "a", "claims": []})
    assert _bad(bad, ctx)


def test_context_budget_never_cuts_passage():
    ctx = _ctx(n=8, text="y" * 3000)
    assert len(ctx.entries) == 4 and ctx.omitted == 4 and ctx.outcome == "assembled"
    assert build_context([]).outcome == "empty"
