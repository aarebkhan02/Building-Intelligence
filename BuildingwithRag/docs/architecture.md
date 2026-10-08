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

`QueryRequest`, `QueryResult`, `RetrievedChunk`, `GenerationResult`, `SubquestionEvidence`, `StructuredSignals`, `ChatCompletionRequest`, omitted-candidate items, and MongoDB-schema models (typed only). No parallel top-level `outcome`/`evidence`/`answer`/`confidence`/`citations`/`diagnostics` fields.

## Endpoints

- `GET /healthz`: safe status only.
- `POST /v1/query`: `QueryRequest` → `QueryResult` (owns diagnostics). `semantic` is real (see Semantic retrieval); other modes go through `run_pattern` and return `not_implemented`.
- `GET /v1/models`: the six model IDs.
- `POST /v1/chat/completions`: maps model → same `QueryRequest`; server sets `caller_id` and `generate_answer`; runs the shared `pipeline.retrieve` + `answer_events` path (`rag-semantic` is real and streamed; other modes return the placeholder text); JSON or SSE ending `[DONE]`; OpenAI-style error envelope; Bearer auth when `CAPSTONE_API_KEY` is set. Open WebUI shows the streamed text, DRAFT lines and footer as plain content.

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

- Shared path: `pipeline.retrieve` (semantic → `run_semantic`, else `run_pattern`; errors raise before streaming) then `answer_events` → `answer.answer_stream`. `/v1/query` drains the events into `QueryResult.generation`; chat forwards `text`/`notice` events and renders the footer from the same final `GenerationResult`. One request = one generation/validation operation.
- Event flow: model streams text with inline labels (`[E1]`); first characters are buffered so `INSUFFICIENT_EVIDENCE: <reason>` is never streamed. Each attempt starts with a `DRAFT — checking evidence` notice.
- `MAX_ATTEMPTS = 2` (initial + one retry, retry prompt carries the plain-language issues). Checks per attempt, in order: `citation_labels` (all cited labels supplied), `claim_cited` (every sentence/bullet cited, text non-empty), `support` (one non-streamed validator call, strict JSON, per claim; skipped when an earlier check already failed). Invalid citations are never rewritten.
- New `GenerationResult` fields: `confidence` (`high` | `low` | absent), `issues` (`attempt`, `check`, `detail`; all attempts), `attempts` (`attempt`, `status` passed|failed|unjudged, `chars`, `latency_ms`), `draft_answer` (last unpassed text), `low_confidence_reason`.
- Outcome mapping: passed → `answered` + `high`; final check failed → `malformed`, empty `text`, `draft_answer`, `low`; provider failure → `unavailable`; validator unreachable → `unavailable` with draft, no confidence; validator invalid reply → `malformed`, no confidence; `insufficient_evidence` and empty context unchanged.
- Chat labels: `DRAFT — checking evidence`; `Check failed: … Retrying (attempt 2 of 2)…`; `Evidence check passed — confidence: high` + `Sources:`; `DRAFT — low confidence, not the final answer.`; insufficient evidence is one plain sentence.
- Failure after text began: final line `Answer generation unavailable — the text above is an unchecked draft.`, then stop + `[DONE]` (HTTP 200).
- `CAPSTONE_API_KEY` non-empty → `/v1/chat/completions` requires `Authorization: Bearer <key>` (constant-time compare; 401 `invalid_api_key`). Empty → no check. Other endpoints unchanged.
- Provider call: `stream: true` SSE via `httpx`, 30 s timeout per call, `temperature: 0`.

## Environment notes

- `.env` is untracked; secrets are never committed.
- `GENERATION_API_BASE_URL`/`GENERATION_API_KEY` come from the trainer; blank in `.env.example` until Story 3.1.
- `MONGODB_URI` is an Atlas M0 string; the Atlas IP access list must allow your machine.
- A free Voyage key is rate limited; Story 2.2 embedding takes about 40 minutes.
