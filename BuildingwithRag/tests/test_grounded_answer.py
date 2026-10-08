"""Claim-derivation and context tests for Story 3.1/3.2 (no network)."""

from building_with_rag.generation.answer import derive_claims, extract_labels
from building_with_rag.generation.context import build_context
from building_with_rag.models import RetrievedChunk


def _ctx(n=2, text="x"):
    return build_context(
        [RetrievedChunk(chunk_id=f"c{i}", section_id=f"bns:{i}", act="BNS_2023", text=text)
         for i in range(n)]
    )


def test_labels_and_claims_from_text():
    assert extract_labels("a [E2, E1] b [E2][E3]") == ["E2", "E1", "E3"]
    claims = derive_claims("Intro:\n- Theft is punished [E1].\n- Fines apply too. [E2]\nNo label here.")
    assert [c.evidence_labels for c in claims] == [["E1"], ["E2"], []]


def test_context_budget_never_cuts_passage():
    ctx = _ctx(n=8, text="y" * 3000)
    assert len(ctx.entries) == 4 and ctx.omitted == 4 and ctx.outcome == "assembled"
    assert build_context([]).outcome == "empty"
