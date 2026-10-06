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
- `POST /v1/query`: `QueryRequest` → `run_pattern` → `QueryResult` (owns diagnostics).
- `GET /v1/models`: the six model IDs.
- `POST /v1/chat/completions`: maps model → same `QueryRequest` and `run_pattern`; server sets `caller_id` and `generate_answer`; JSON or SSE ending `[DONE]`; OpenAI-style error envelope.

## Environment notes

- `.env` is untracked; secrets are never committed.
- `GENERATION_API_BASE_URL`/`GENERATION_API_KEY` come from the trainer; blank in `.env.example` until Story 3.1.
- `MONGODB_URI` is an Atlas M0 string; the Atlas IP access list must allow your machine.
- A free Voyage key is rate limited; Story 2.2 embedding takes about 40 minutes.
