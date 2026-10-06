# Story 1.1: Architecture and Project Seed

## Purpose

This is the first story in a three-day classroom RAG course. It does two things, in order:

1. Write `docs/architecture.md` and get the instructor's approval.
2. After approval, seed one small application: a FastAPI service with a shared mode registry, placeholder RAG modes, and an OpenAI-compatible chat adapter that the trainer-supplied Open WebUI can talk to.

The result is one application. Do not build a custom chat frontend or a second demo. Open WebUI is the only chat client.

## Prerequisites

- Python 3.12 and UV are installed.
- Read these first, if they exist: `docs/config.yaml`, `docs/architecture.md`, the repository layout, and existing stories in `docs/stories/`.
- If `docs/config.yaml` names a project path, use it. Otherwise use the repository root. Call this the project root below.
- Supplied files under `data/raw/` (including `data/raw/PROVENANCE.md`) must be preserved. Never delete or overwrite existing work.
- The trainer shares the Open WebUI bundle ZIP over the local LAN. Extract it before the smoke check.

## Work to do

### Part A: Architecture (do this first, then stop)

Create or update `docs/architecture.md`. If it exists, edit it additively. It must record:

**Evidence rules**
- BNS and IPC documents are the only future answer evidence.
- An answer must not claim support without retrieved evidence.
- Identifiers are act-qualified (for example, `BNS` section 103 versus `IPC` section 302), so the two acts are never confused.
- Supplied provenance is `data/raw/PROVENANCE.md`.
- Retrieved passages are evidence, never instructions to the application.

**Trust boundaries (keep light)**
- Validate API input.
- Preserve source origin on every passage.
- Do not put secrets in code, responses, or logs.
- Out of scope: multi-user authorization, a security program, an evaluation harness.

**Fixed choices (later stories reuse these names without renaming or adding provider-specific alternatives)**
- Embedding provider: Voyage. Model `voyage-3.5`, version `voyage-3.5`, 1,024 dimensions, for every document and query embedding.
- Rerank settings, generation settings, and environment variable names as listed in the `.env.example` section below.
- Derived corpus files live in `data/processed/`: `bns_sections.jsonl` and `ipc_sections.jsonl`.
- Course modes and model IDs (table in Part B).
- Shared contracts (listed in Part B). Later stories extend them additively and never replace them with simplified alternatives.

**Stop gate.** When the architecture is written, stop and ask the instructor to approve `docs/architecture.md`. Before approval, do not create seed files, install dependencies, or scaffold code. Resume this same story only after explicit approval.

### Part B: Seed (only after approval)

**1. Project files.** Create a Python 3.12 and UV project with FastAPI, Pydantic settings, PyMongo (for later use), Ruff, and a minimal test runner (pytest). Create:
- `src/building_with_rag/`
- `tests/`
- `.env.example`
- `.gitignore` (must ignore `.env`, `.venv/`, caches)
- `pyproject.toml`
- `uv.lock`

Preserve `data/raw/`.

**2. `.env.example`.** Use exactly these values:

```
APP_ENV=development
MONGODB_URI=
VOYAGE_API_KEY=
CAPSTONE_API_KEY=
GENERATION_API_BASE_URL=
GENERATION_API_KEY=
RERANK_API_KEY=
MONGODB_DB_NAME=building_with_rag
MONGODB_TEST_DB_NAME=building_with_rag_test
WEBUI_DEMO_CALLER_ID=demo-public
GENERATION_MODEL_NAME=gpt-4o-mini
RERANK_API_BASE_URL=https://api.voyageai.com/v1
RERANK_MODEL_NAME=rerank-2.5
RERANK_REQUEST_TIMEOUT_SECONDS=30
RERANK_CANDIDATE_LIMIT=20
RERANK_SEND_LIMIT=10
RERANK_RETURN_LIMIT=5
```

Document these notes (in `docs/architecture.md` or a short README section):
- `.env` is untracked. Secrets are never committed.
- `GENERATION_API_BASE_URL` and `GENERATION_API_KEY` are supplied by the trainer for an OpenAI-compatible LiteLLM proxy. Leave them blank in `.env.example`. Fill them in `.env` only when Story 3.1 needs them.
- `MONGODB_URI` is an Atlas free-tier (M0) connection string. The Atlas IP access list must allow your machine.
- A free Voyage key is rate limited, so Story 2.2 embedding takes about 40 minutes.

**3. Application behavior.** The app starts without database or model credentials. `GET /healthz` returns a safe status and exposes no secrets or configuration values.

**4. Shared registry.** Create one registry with only these modes:

| Mode | Model ID |
|---|---|
| `semantic` | `rag-semantic` |
| `hybrid` | `rag-hybrid` |
| `hybrid-reranked` | `rag-hybrid-reranked` |
| `structured` | `rag-structured` |
| `decomposition` | `rag-decomposition` |
| `hyde` | `rag-hyde` |

Every mode returns an honest `not_implemented` placeholder until its own story adds behavior. `run_pattern` is the single dispatch path.

**5. Shared contracts.** Define typed models, without behavior, in this project. Use these exact names. Later stories reuse and extend them additively.

- `QueryRequest`: `question` (1–4,000 characters), `pattern`, optional `caller_id`, optional `SemanticFilters` (`act`, `status`, `access_level`, each a list), `limit` (default 5, range 1–20), `generate_answer` (default false), `required_acts`, `chapter`. The seed may resolve only the fixed local demo caller, but keeps the `caller_id` field.
- `QueryResult`: `pattern`, `status`, `message`, `trace`, `results`, optional `generation`, plus additive fields that default to empty: `omitted_candidates`, `subquestions`, `hyde_direct_candidates`, `hyde_query_candidates`, `hyde_hypothetical_text_debug`.
- `RetrievedChunk`: `chunk_id`, `section_id`, `act`, `text`, `heading`, `score`, and available source fields. Later stories add only these fields, named in their own handouts: `semantic_score`, `semantic_rank`, `keyword_score`, `keyword_rank`, `fused_score`, `fused_rank`, `rerank_score`, `rerank_rank`.
- `omitted_candidates` items: `chunk_id`, `omitted_reason`.
- `GenerationResult`: `outcome` (`answered`, `insufficient_evidence`, `unavailable`, `malformed`), `answer`, `claims`, `citations`, `supporting_passages`, `provider`, `model`, `trace`, `context_outcome`, `confidence`, `draft_answer`, `issues`, `attempts`, `low_confidence_reason`.
- `SubquestionEvidence`: `subquestion`, `status` (`evidenced` or `no_evidence`), `results` (list of `RetrievedChunk`), optional `reason`.
- `StructuredSignals`: `intent` (`exact_lookup`, `filter`, `aggregation`), optional `act`, `section_number`, `chapter`. Story 5.1 fills these in.
- `ChatCompletionRequest`: `model`, text `messages` (roles `system`, `developer`, `user`, `assistant`), `stream`, `n`, optional strict `rag_options` (`pattern`, list filters, `limit`, `required_acts`, `chapter`).
- MongoDB-schema contracts: typed models only, no provisioning.

Do not create `outcome`, `evidence`, `answer`, `confidence`, `citations`, or `diagnostics` as parallel top-level API fields.

**6. `POST /v1/query`.** Accepts `QueryRequest`, calls `run_pattern`, returns `QueryResult`. This endpoint owns the diagnostics.

**7. `GET /v1/models` and `POST /v1/chat/completions`.**
- `/v1/models` lists the six model IDs.
- `/v1/chat/completions` maps the selected model to the same `QueryRequest` and the same `run_pattern` path as `/v1/query`. The server sets the demo `caller_id` (from `WEBUI_DEMO_CALLER_ID`) and `generate_answer`. Do not duplicate implementations.
- Support normal OpenAI Chat Completions JSON, and SSE with role/content/stop frames ending in `[DONE]`. Do not invent custom SSE events.
- Return the OpenAI-style error envelope for errors before streaming starts.
- Open WebUI receives only normal answer text derived from the same `QueryResult`. Later stories render final confidence, sources, and low-confidence warnings as clearly labelled text after the answer, and keep the full `GenerationResult` in `QueryResult.generation`.

**8. Open WebUI (separately running chat client).**
- The trainer shares the bundle ZIP over the local LAN. Participants extract it and run the included setup script exactly once, before using the classroom project.
  - Windows (primary path), from the extracted bundle root: `powershell -ExecutionPolicy Bypass -File .\setup_open_webui.ps1`
  - macOS/Linux, from that root: `sh setup_open_webui.sh` (needs internet access for the first installation)
- The scripts install the pinned Open WebUI version, write the course settings, start the loopback-only service, and provision the `RAG options` Filter and `Building with RAG` Pipe.
- Do not hand-install Open WebUI, create accounts, edit the admin panel, change either Function, or rerun setup to reload anything.
- If setup fails, report its exact output and stop.
- Day-to-day start, stop, and status use the supplied `manage_open_webui` script.
- The Pipe sends the selected `rag-<pattern>` model, `stream: true`, the latest user message, and normalized `rag_options` (`pattern`, list `filters`, `limit`, `required_acts`, `chapter`) to `/v1/chat/completions`. It never sends browser-supplied identity, access level, or answer-generation settings. The adapter must accept that exact request and use server-side `caller_id` and `generate_answer`.

**Out of scope.** Retrieval, PDF parsing, embeddings, MongoDB provisioning, LLM calls, GraphRAG, agentic RAG, broad deployment, and aggressive testing.

## Completion checks

Keep checks light. Run each one and record what happened:

1. The app starts with no `.env` credentials.
2. `GET /healthz` returns a safe OK response.
3. One `POST /v1/query` request returns a `not_implemented` placeholder in the `QueryResult` shape.
4. `GET /v1/models` lists the six model IDs.
5. `POST /v1/chat/completions` returns a placeholder as normal JSON, and as SSE ending in `[DONE]`.
6. Ruff and the minimal test run pass.
7. Open WebUI smoke check: start the capstone API, open `http://127.0.0.1:8080`, select `Building with RAG`, choose `semantic` in the RAG-options chip, and send a question. The reply must be the capstone's honest placeholder, not a local preview.

## Handover

When finished, fill in this section in the story file:

- **Files created:** the exact list.
- **Commands actually run:** the exact commands, with a one-line result each.
- **Open WebUI result:** what the smoke check showed, or the exact setup output if it failed.
- **Architecture approval:** who approved `docs/architecture.md` and when.
- **Notes for Story 1.2 onward:** contracts and names that later stories must reuse unchanged.
