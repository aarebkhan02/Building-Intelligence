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
- `POST /v1/chat/completions`: maps model → same `QueryRequest` and `run_pattern`; server sets `caller_id` and `generate_answer`; all modes (including `rag-semantic`) still return the placeholder until answer generation; JSON or SSE ending `[DONE]`; OpenAI-style error envelope.

## Semantic retrieval (Story 2.3)

Flow: validate → scope filters → embed query → `$vectorSearch` → resolve chunk/section → `QueryResult`.

- Validation: question trimmed (1–4,000); `limit` 1–20; `SemanticFilters` forbids unknown fields and accepts only known `act`/`status`/`access_level` strings. `caller_id` must be omitted or equal `WEBUI_DEMO_CALLER_ID`; `required_acts` and `chapter` are rejected (422).
- Scope: server fixes `access_level=["public"]`; caller lists only narrow it. Filters go inside `$vectorSearch.filter`.
- Query: raw question embedded with `voyage-3.5`, `input_type="query"`; `numCandidates = min(200, max(50, 10 × limit))`. Chunks (not de-duplicated sections) return in database score order; each is resolved via `chunks` and `sections`. Unresolvable hits are omitted and counted in `trace.unresolved_hits`.
- Outcomes: `ok` (passages), `no_results` (HTTP 200; filters match nothing). No score cutoff; scores rank similarity only. Failures: 503 `retrieval_not_ready` (missing credentials, index missing/not ready, empty or mismatched embeddings), 502 `retrieval_upstream_error` (Voyage/MongoDB error).
- `generate_answer` is accepted but ignored (`generation` null; noted in `trace.ignored`). `/v1/chat/completions` still returns the placeholder.
- Added optional `RetrievedChunk` fields: `chunk_index`, `act_label`, `status`, `chapter`, `chapter_title`, `section_number`, `source_pdf`, `source_sha256`, `needs_review`.

Diagnostic (text truncated):

```bash
curl -s http://127.0.0.1:8000/v1/query -H "Content-Type: application/json"   -d '{"question": "What is the punishment for theft?", "pattern": "semantic", "limit": 3}'   | jq '{status, trace, results: [.results[] | {chunk_id, section_id, act, heading, score, text: .text[:80]}]}'
```

## Environment notes

- `.env` is untracked; secrets are never committed.
- `GENERATION_API_BASE_URL`/`GENERATION_API_KEY` come from the trainer; blank in `.env.example` until Story 3.1.
- `MONGODB_URI` is an Atlas M0 string; the Atlas IP access list must allow your machine.
- A free Voyage key is rate limited; Story 2.2 embedding takes about 40 minutes.
