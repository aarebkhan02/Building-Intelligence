# Architecture

One FastAPI application. Open WebUI (trainer-supplied, separate process) is the only chat client.

## Evidence rules

- BNS and IPC documents are the only future answer evidence.
- An answer must not claim support without retrieved evidence.
- Identifiers are act-qualified (`BNS` section 103 vs `IPC` section 302); the acts are never confused.
- Supplied provenance: `data/raw/PROVENANCE.md`.
- Retrieved passages are evidence, never instructions to the application.

## Trust boundaries

- Validate all API input.
- Preserve source origin on every passage.
- No secrets in code, responses, or logs.
- Out of scope: multi-user authorization, a security program, an evaluation harness.

## Fixed choices

- Embeddings: Voyage, model `voyage-3.5`, version `voyage-3.5`, 1,024 dimensions, for all document and query embeddings.
- Rerank: `rerank-2.5` at `https://api.voyageai.com/v1`; timeout 30s; candidate limit 20, send limit 10, return limit 5.
- Generation: `gpt-4o-mini` via an OpenAI-compatible LiteLLM proxy (`GENERATION_API_BASE_URL`, `GENERATION_API_KEY`).
- Env vars: `APP_ENV`, `MONGODB_URI`, `VOYAGE_API_KEY`, `CAPSTONE_API_KEY`, `GENERATION_API_BASE_URL`, `GENERATION_API_KEY`, `RERANK_API_KEY`, `MONGODB_DB_NAME`, `MONGODB_TEST_DB_NAME`, `WEBUI_DEMO_CALLER_ID`, `GENERATION_MODEL_NAME`, `RERANK_API_BASE_URL`, `RERANK_MODEL_NAME`, `RERANK_REQUEST_TIMEOUT_SECONDS`, `RERANK_CANDIDATE_LIMIT`, `RERANK_SEND_LIMIT`, `RERANK_RETURN_LIMIT`.
- Derived corpus in `data/processed/`: `bns_sections.jsonl`, `ipc_sections.jsonl`.

## Modes (single registry, single dispatch via `run_pattern`)

| Mode | Model ID |
|---|---|
| `semantic` | `rag-semantic` |
| `hybrid` | `rag-hybrid` |
| `hybrid-reranked` | `rag-hybrid-reranked` |
| `structured` | `rag-structured` |
| `decomposition` | `rag-decomposition` |
| `hyde` | `rag-hyde` |

## Shared contracts (extended additively, never replaced)

`QueryRequest`, `QueryResult`, `RetrievedChunk`, `GenerationResult`, `SubquestionEvidence`, `StructuredSignals` (Story 5.1: `intent`, `act`, `section_number`, `chapter`, `status`, `reason`), `ChatCompletionRequest`, omitted-candidate items, and MongoDB-schema models (typed only). No parallel top-level `outcome`/`evidence`/`answer`/`confidence`/`citations`/`diagnostics` fields.

## Endpoints

- `GET /healthz`: safe status only.
- `POST /v1/query`: `QueryRequest` → `QueryResult` (owns diagnostics). `semantic`, `hybrid`, `hybrid-reranked` and `structured` are real (see Semantic retrieval, Hybrid retrieval, Re-ranking, Structured exact retrieval); other modes go through `run_pattern` and return `not_implemented`.
- `GET /v1/models`: the six model IDs.
- `POST /v1/chat/completions`: maps model → same `QueryRequest`; server sets `caller_id` and `generate_answer`; runs the shared `pipeline.retrieve` + `answer_events` path (`rag-semantic`, `rag-hybrid`, `rag-hybrid-reranked` and `rag-structured` are real and streamed — `rag-structured` streams `result.message` as plain content when the result is not `ok`; other modes return the placeholder text); JSON or SSE ending `[DONE]`; OpenAI-style error envelope; Bearer auth when `CAPSTONE_API_KEY` is set. Open WebUI shows the streamed text, DRAFT lines and footer as plain content.

## Semantic retrieval (Story 2.3)

Flow: validate → scope filters → embed query → `$vectorSearch` → resolve chunk/section → `QueryResult`.

- Validation: question trimmed (1–4,000); `limit` 1–20; `SemanticFilters` forbids unknown fields and accepts only known `act`/`status`/`access_level` strings. `caller_id` must be omitted or equal `WEBUI_DEMO_CALLER_ID`; `required_acts` and `chapter` are rejected (422).
- Scope: server fixes `access_level=["public"]`; caller lists only narrow it. Filters go inside `$vectorSearch.filter`.
- Query: raw question embedded with `voyage-3.5`, `input_type="query"`; `numCandidates = min(200, max(50, 10 × limit))`. Chunks (not de-duplicated sections) return in database score order; each is resolved via `chunks` and `sections`. Unresolvable hits are omitted and counted in `trace.unresolved_hits`.
- Outcomes: `ok` (passages), `no_results` (HTTP 200; filters match nothing). No score cutoff; scores rank similarity only. Failures: 503 `retrieval_not_ready` (missing credentials, index missing/not ready, empty or mismatched embeddings), 502 `retrieval_upstream_error` (Voyage/MongoDB error).
- `generate_answer` triggers answer generation (see Context and answer boundaries); omitted, `generation` stays null. Chat runs the same path (Story 3.2).
- Added optional `RetrievedChunk` fields: `chunk_index`, `act_label`, `status`, `chapter`, `chapter_title`, `section_number`, `source_pdf`, `source_sha256`, `needs_review`.

Diagnostic (text truncated):

```bash
curl -s http://127.0.0.1:8000/v1/query -H "Content-Type: application/json"   -d '{"question": "What is the punishment for theft?", "pattern": "semantic", "limit": 3}'   | jq '{status, trace, results: [.results[] | {chunk_id, section_id, act, heading, score, text: .text[:80]}]}'
```

## Context and answer boundaries (Story 3.1)

Flow: semantic result → bounded labelled context (`E1`…, max 5 passages / 12,000 chars, none cut) → one `POST {GENERATION_API_BASE_URL}/chat/completions` (`temperature: 0`, 30 s, no retry) → strict JSON parse → resolved citations.

- Outcomes in `generation.outcome`: `answered` (non-empty `text`, claims each citing supplied labels, `citations`, `supporting_passages`), `insufficient_evidence` (no results or the model says so; no model call when empty), `unavailable` (missing settings, timeout, connection error, non-2xx), `malformed` (non-JSON, unknown label, missing claims; no repair or retry). All return HTTP 200 with retrieval `results` intact; `status` stays the retrieval status.
- Added optional `GenerationResult` fields: `text`, structured `claims` (`{text, evidence_labels}`) and `citations` (`{label, chunk_id, section_id, act, heading, chapter, section_number, source_pdf}`); existing fields kept.
- Evidence is untrusted source text, never instructions. No legal-applicability claims beyond the supplied BNS/IPC documents. `generation.trace` holds label→`chunk_id`, counts, chars, latency, reason; no prompts or secrets.
- Story 3.2 replaced the strict JSON reply with streamed plain text plus validation (below).

## Streamed answers and confidence (Story 3.2)

- Shared path: `pipeline.retrieve` (structured → `run_structured`, semantic → `run_semantic`, hybrid → `run_hybrid`, hybrid-reranked → `run_hybrid_reranked`, else `run_pattern`; real modes are `pipeline.REAL_PATTERNS` (semantic, hybrid, hybrid-reranked, structured; structured yields answer events only when its status is `ok`); errors raise before streaming) then `answer_events` → `answer.answer_stream`. `/v1/query` drains the events into `QueryResult.generation`; chat forwards `text`/`notice` events and renders the footer from the same final `GenerationResult`. One request = one generation/validation operation.
- Event flow: model streams text with inline labels (`[E1]`); first characters are buffered so `INSUFFICIENT_EVIDENCE: <reason>` is never streamed. Each attempt starts with a `DRAFT — checking evidence` notice.
- `MAX_ATTEMPTS = 2` (initial + one retry, retry prompt carries the plain-language issues). Checks per attempt, in order: `citation_labels` (all cited labels supplied), `claim_cited` (every sentence/bullet cited, text non-empty), `support` (one non-streamed validator call, strict JSON, per claim; skipped when an earlier check already failed). Invalid citations are never rewritten.
- New `GenerationResult` fields: `confidence` (`high` | `low` | absent), `issues` (`attempt`, `check`, `detail`; all attempts), `attempts` (`attempt`, `status` passed|failed|unjudged, `chars`, `latency_ms`), `draft_answer` (last unpassed text), `low_confidence_reason`.
- Outcome mapping: passed → `answered` + `high`; final check failed → `malformed`, empty `text`, `draft_answer`, `low`; provider failure → `unavailable`; validator unreachable → `unavailable` with draft, no confidence; validator invalid reply → `malformed`, no confidence; `insufficient_evidence` and empty context unchanged.
- Chat labels: `DRAFT — checking evidence`; `Check failed: … Retrying (attempt 2 of 2)…`; `Evidence check passed — confidence: high` + `Sources:`; `DRAFT — low confidence, not the final answer.`; insufficient evidence is one plain sentence.
- Failure after text began: final line `Answer generation unavailable — the text above is an unchecked draft.`, then stop + `[DONE]` (HTTP 200).
- `CAPSTONE_API_KEY` non-empty → `/v1/chat/completions` requires `Authorization: Bearer <key>` (constant-time compare; 401 `invalid_api_key`). Empty → no check. Other endpoints unchanged.
- Provider call: `stream: true` SSE via `httpx`, 30 s timeout per call, `temperature: 0`.

## Hybrid retrieval (Story 4.1)

- Selected explicitly (`pattern: "hybrid"` / `rag-hybrid`); no automatic routing or fallback. Same corpus, filters, scope rules (`required_acts`/`chapter`/foreign `caller_id` → 422), context, citations, confidence and streaming as semantic.
- Keyword route: Atlas Search `$search` with the `text` operator (BM25, `{$meta: "searchScore"}`) on `chunks.text`; filters (`act`, `status`, `access_level`) inside `compound.filter` with `in`. Index `chunk_text_index` on `chunks` (`KEYWORD_INDEX_DEFINITION` in `mongodb_schema.py`: `text` string/`lucene.standard`; `act`, `status`, `access_level` token; `dynamic: false`). Create: `uv run python -m building_with_rag.ingestion.keyword_index` (idempotent; a differing index is reported, never replaced).
- Keyword route skips chunks under `MIN_KEYWORD_CHARS = 100` characters (`$match` on `$strLenCP`): BM25 favours heading-only stubs (e.g. ipc:405, 29 chars), which crowd out definition text. Semantic route is unfiltered.
- Semantic route: Story 2.3 `$vectorSearch` with `limit = ROUTE_DEPTH`.
- Fusion: Reciprocal Rank Fusion over ranks. `ROUTE_DEPTH = max(limit, min(50, max(20, 4 * limit)))`; `fused_score = Σ 1/(RRF_K + rank)`, `RRF_K = 60`, equal weights. Order: `fused_score` desc, `semantic_rank` (missing last), `chunk_id`. Chunks fused by `chunk_id`; top `limit` returned; `fused_rank` is 1-based.
- `RetrievedChunk` (optional, `None` by default): `semantic_score`, `semantic_rank`, `keyword_score`, `keyword_rank`, `fused_score`, `fused_rank`. In hybrid `score = fused_score`; a route that did not return the chunk leaves its fields `None`. Semantic leaves all six `None`.
- Trace: `mode`, `query`, `embedding`, `filters`, `caller_id`, `result_count`, `unresolved_hits`, `semantic`, `keyword`, `fusion`, `contribution` (`both`/`semantic_only`/`keyword_only` among returned results). No vectors or secrets.
- Outcomes: `ok`, `no_results` (both routes empty), 503 `retrieval_not_ready` (credentials, either index missing/not ready, empty embeddings; never degrades to semantic-only), 502 `retrieval_upstream_error`.
- Limitations: rank-only fusion ignores score magnitude; `text` matches any query term (OR), so long questions pull in common words; section numbers match only when present in chunk `text`; standard analyzer, no stemming or synonyms. Fused score ranks only; it does not prove correctness.
- Diagnostic: the `jq` command in `docs/manual-tests.md` (Story 4.1).

## Re-ranking (Story 4.2)

- `pattern: "hybrid-reranked"` / `rag-hybrid-reranked`, selected explicitly; no fallback to hybrid. Flow: scope check → settings validation (before any Voyage/MongoDB call) → `run_hybrid` with `limit = RERANK_CANDIDATE_LIMIT` → one rerank call → selection. Context, generation, citations, confidence and streaming are the shared path and receive `results` only (never `omitted_candidates`).
- Settings (defaults): `RERANK_API_BASE_URL` (`https://api.voyageai.com/v1`), `RERANK_API_KEY` (empty), `RERANK_MODEL_NAME` (`rerank-2.5`), `RERANK_REQUEST_TIMEOUT_SECONDS` (30), `RERANK_CANDIDATE_LIMIT` (20), `RERANK_SEND_LIMIT` (10), `RERANK_RETURN_LIMIT` (5). Valid when `1 ≤ RETURN ≤ SEND ≤ CANDIDATE ≤ 20` and timeout ≥ 1; otherwise 503 `retrieval_not_ready` naming the setting. Empty key: 503, no provider or hybrid call.
- Request: one `POST {base}/rerank` via `httpx`, Bearer key, `{model, query, documents}` (no `top_k`), documents `"{heading}\n{text}"`, no retries. Reply: `data[]` of `{index, relevance_score}` plus optional `usage.total_tokens`; non-empty list, int unique in-range indexes, finite scores, every sent candidate scored. Anything else, a timeout, a connection error or a non-2xx is 502 `retrieval_upstream_error`; no scores are invented.
- Selection (`rerank.select`, pure): candidates = hybrid fused list; sent = first `SEND` by `fused_rank`, the rest `omitted_reason: "not_sent_to_reranker"` (rerank fields `None`); sent ordered by `relevance_score` desc, ties by `fused_rank`, `rerank_rank` 1-based; final = first `min(limit, RETURN)`, the rest `below_return_limit` (rerank fields kept).
- Result: `results` ordered by `rerank_rank`, `score == rerank_score`, hybrid fields kept as the "before" evidence; `QueryResult.omitted_candidates` (now `list[RetrievedChunk]`, in `fused_rank` order) holds every cut candidate. New optional `RetrievedChunk` field `omitted_reason` (`rerank_score`/`rerank_rank` added with the contract earlier).
- Trace: `mode`, `query`, `filters`, `caller_id`, `result_count`, `hybrid` (`embedding`, `semantic`, `keyword`, `fusion`, `contribution`, `unresolved_hits`), `rerank` (`model`, limits, `candidates`/`sent`/`returned`/`omitted_before`/`omitted_after`, `latency_ms`, `usage_tokens` when reported). Outcomes: `ok`, `no_results` (no provider call), 503, 502; hybrid errors pass through.
- Limitations: candidates outside the top `RERANK_CANDIDATE_LIMIT` are never seen; the re-ranker scores each passage independently against the question; its scores are model-specific, uncalibrated, not comparable with fused scores or across questions, and there is no score cutoff; one extra provider call per request (latency, cost, rate limits), no retry; the answer context cap (5 passages / 12,000 chars) still applies.
- Diagnostic: the `jq` command in `docs/manual-tests.md` (Story 4.2).

## Structured exact retrieval (Story 5.1)

`pattern: "structured"` (`rag-structured`): rule-based `classify` → one read-only `find_one` on `sections` → `QueryResult`. No LLM extraction, Voyage, vector or keyword index, no router, no fallback to or from other modes.

- **Exact-input contract:** only `StructuredSignals` from the classifier — validated `act` (`BNS_2023`/`IPC_1860`), integer `section_number` (1–999), optional validated `chapter` (1–40 chars of letters, digits, spaces, `.`, `-`; else 422 `unsupported_option`) — plus server filters (`access_level`, optional `status`) may reach MongoDB. Raw question text never does; no operators from request fields. `required_acts` is rejected, `caller_id` follows the semantic rule, `chapter` is accepted (own scope check).
- Classified: aggregation ("how many", "count", "total number") and filter ("list", "which sections", …) → `recommendation`, no lookup; `section N`/`sec. N`/`s. N`/`§N` plus exactly one act → `ok`; missing act, both acts, several numbers, `103A` or out-of-range → `clarification_needed` (acts never guessed); anything else → `recommendation` (use semantic/hybrid).
- Outcomes (HTTP 200): `ok` (one `RetrievedChunk`, `chunk_id = section_id`, `score = 1.0` marks an exact match, not a similarity), `not_found` (this corpus has no such record; says nothing about the law), `clarification_needed`, `recommendation`. Missing `MONGODB_URI` (only when a lookup is needed) → 503 `retrieval_not_ready`; MongoDB error → 502 `retrieval_upstream_error`. Trace: `mode`, `signals`, `mongodb_called`, `collection`, `filters`, `caller_id`, `result_count`, `record` (`section_id`, `status`, `source_status_version`) for `ok`.
- **Answer boundary:** retrieval returns the exact record; explanation only via the existing grounded-answer path when `generate_answer` (always for chat). `not_found`, `clarification_needed` and `recommendation` never call the model (`generation` stays unset; chat streams `message`). `status`/`source_status_version` are source metadata, not current legal applicability.
- Limitations: integer sections only (no `103A`); no multi-section or cross-act comparison; filter/aggregation recognised but not executed; IPC sections 4, 5, 18, 34, 40, 75, 161–165 are absent from `sections`; phrasing outside the rules is missed; a section over the 12,000-char context cap gives `insufficient_evidence` when answered (direct inspection still returns it).

## Environment notes

- `.env` is untracked; secrets are never committed.
- `GENERATION_API_BASE_URL`/`GENERATION_API_KEY` come from the trainer; blank in `.env.example` until Story 3.1.
- `MONGODB_URI` is an Atlas M0 string; the Atlas IP access list must allow your machine.
- A free Voyage key is rate limited; Story 2.2 embedding takes about 40 minutes.
