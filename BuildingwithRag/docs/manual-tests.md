# Manual tests

## Story 1.1 — Architecture and Project Seed

What it adds: FastAPI project seed with health, query, model-listing, and OpenAI-compatible chat endpoints — all returning honest `not_implemented` placeholders.

Prerequisite: start the API — `uv run uvicorn building_with_rag.app:app --host 127.0.0.1 --port 8000`

### Health

```bash
curl -s http://127.0.0.1:8000/healthz
```

Expected: `{"status":"ok"}`.

### List models

```bash
curl -s http://127.0.0.1:8000/v1/models | python3 -c "import json,sys; [print(m['id']) for m in json.load(sys.stdin)['data']]"
```

Expected: six lines: `rag-semantic`, `rag-hybrid`, `rag-hybrid-reranked`, `rag-structured`, `rag-decomposition`, `rag-hyde`.

### Query — each RAG mode (semantic)

```bash
curl -s http://127.0.0.1:8000/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is theft?", "pattern": "semantic", "limit": 3}'
```

Expected (Story 2.3, needs ingested data and keys): `"status":"ok"`, up to 3 `results` in non-increasing `score` order, populated `trace`. Not `not_implemented`.

### Query — hybrid

```bash
curl -s http://127.0.0.1:8000/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is theft?", "pattern": "hybrid"}'
```

Expected: `"status":"not_implemented"`, message references `hybrid`.

### Query — hybrid-reranked

```bash
curl -s http://127.0.0.1:8000/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is theft?", "pattern": "hybrid-reranked"}'
```

Expected: `"status":"not_implemented"`, message references `hybrid-reranked`.

### Query — structured

```bash
curl -s http://127.0.0.1:8000/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is theft?", "pattern": "structured"}'
```

Expected: `"status":"not_implemented"`, message references `structured`.

### Query — decomposition

```bash
curl -s http://127.0.0.1:8000/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is theft?", "pattern": "decomposition"}'
```

Expected: `"status":"not_implemented"`, message references `decomposition`.

### Query — hyde

```bash
curl -s http://127.0.0.1:8000/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is theft?", "pattern": "hyde"}'
```

Expected: `"status":"not_implemented"`, message references `hyde`.

### Query — empty question (failure)

```bash
curl -s http://127.0.0.1:8000/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "", "pattern": "semantic"}'
```

Expected: 422 validation error (question below min_length 1).

### Chat completions — JSON (non-streaming)

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "rag-hybrid-reranked", "messages": [{"role": "user", "content": "What is theft?"}]}'
```

Expected: `"object":"chat.completion"`, `"finish_reason":"stop"`, content contains `not implemented yet` (`rag-semantic` is real since Story 3.2).

### Chat completions — streaming

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "rag-hybrid-reranked", "messages": [{"role": "user", "content": "What is theft?"}], "stream": true}'
```

Expected: SSE `data:` frames with `delta` role then content, ending with `data: [DONE]`.

### Chat completions — invalid model (failure)

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "gpt-4", "messages": [{"role": "user", "content": "Hello"}]}'
```

Expected: 404 with an OpenAI-style `error` envelope (`Unknown model 'gpt-4'.`).

### Chat completions — no user message (failure)

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "rag-hybrid-reranked", "messages": [{"role": "system", "content": "You are helpful."}]}'
```

Expected: 400 with an OpenAI-style `error` envelope (`At least one user message is required.`).

## Story 2.1 — Document Ingestion

What it adds: PDF extraction script that produces inspectable JSONL section-level corpora from the BNS and IPC bare act PDFs.

Prerequisite: Story 1.1 complete, `data/raw/` PDFs and `PROVENANCE.md` present, `uv sync` done.

### Run extraction

```bash
uv run python scripts/extract_sections.py
```

Expected: prints parser name/version, record counts (~358 BNS, ~500 IPC), count of `needs_review` flags, count of empty-text records. Both `data/processed/bns_sections.jsonl` and `data/processed/ipc_sections.jsonl` exist.

### Re-run (skip)

```bash
uv run python scripts/extract_sections.py
```

Expected: prints "BNS corpus up to date — skipping" and "IPC corpus up to date — skipping". No records appended or overwritten.
## Story 2.2 — MongoDB, Chunks, Embeddings, and Vector Index

What it adds: an ingestion runner that loads the JSONL corpora into MongoDB as `sources`, `sections`, `chunks`, and `embeddings`, embeds every chunk with Voyage, and creates the Atlas `vector_index` on `embeddings.vector`.

Prerequisite: Story 2.1 complete (both JSONL files present), and `.env` holds `MONGODB_URI` (Atlas cluster), `MONGODB_DB_NAME`, `VOYAGE_API_KEY`. First run takes roughly 35–40 minutes on a free Voyage key.

### Run ingestion

```bash
uv run python -m building_with_rag.ingestion.ingest
```

Expected: steps 1–10 print in order. `sources=2`, `sections=858` with the supplied PDFs, `chunks` > 0, and `embeddings == chunks: True`. `bns:1 chunks` shows one or more with `all linked: True`, sample vector length 1024, index status `READY`, and the sample query for "punishment for theft" prints `chunk_id`, `section_id`, heading, and score (or `vector query pending — index not ready`).

### Re-run (skip)

```bash
uv run python -m building_with_rag.ingestion.ingest
```

Expected: sources and sections report skipped, chunks report all skipped with 0 inserted/replaced, `Embeddings: 0 inserted, <n> skipped, 0 deleted` (no Voyage calls), and the existing `vector_index` is reused rather than recreated.

### Missing MongoDB URI (failure)

```bash
MONGODB_URI= uv run python -m building_with_rag.ingestion.ingest
```

Expected: stops at step 1 with `MongoDB unavailable: MONGODB_URI is empty. Set it in .env and re-run.` Nothing is written and no Voyage call is made.

## Story 2.3 — Semantic Retrieval

What it adds: `POST /v1/query` with `pattern: "semantic"` embeds the question, runs `$vectorSearch`, and returns ranked passages. Chat for `rag-semantic` was a placeholder until Story 3.2.

Prerequisite: Story 2.2 complete (`vector_index` READY), `.env` has `MONGODB_URI` and `VOYAGE_API_KEY`, API running as in Story 1.1. Output below is truncated; never print full passages.

### Semantic query

```bash
curl -s http://127.0.0.1:8000/v1/query -H "Content-Type: application/json"   -d '{"question": "What is the punishment for theft?", "pattern": "semantic", "limit": 3}'   | jq '{status, trace, results: [.results[] | {chunk_id, section_id, act, heading, score, text: .text[:80]}]}'
```

Expected: `status` `ok`, at most 3 results in non-increasing `score`, populated `trace`.

### No results (filters)

```bash
curl -s http://127.0.0.1:8000/v1/query -H "Content-Type: application/json"   -d '{"question": "theft", "pattern": "semantic", "filters": {"act": ["IPC_1860"], "status": ["in_force"]}}'   | jq '{status, n: (.results | length)}'
```

Expected: HTTP 200, `status` `no_results`, `n` 0 (IPC is repealed).

### Invalid input (failure)

```bash
curl -s http://127.0.0.1:8000/v1/query -H "Content-Type: application/json"   -d '{"question": "theft", "pattern": "semantic", "filters": {"act": {"$ne": "x"}}}'
```

Expected: 422. Also 422: unknown filter field, `limit` 21, foreign `caller_id`, `required_acts`, `chapter`.

### Missing Voyage key (failure)

Start the API with `VOYAGE_API_KEY=` empty. Expected: `/healthz` still `ok`; semantic query returns 503 `retrieval_not_ready`, not `no_results`.

## Story 3.1 — Grounded Answer Generation

What it adds: `generate_answer: true` on a semantic `/v1/query` returns `generation` (outcome, answer, claims, citations) built only from the retrieved passages. Chat streaming arrives in Story 3.2.

Prerequisite: Story 2.3 complete, `.env` has `GENERATION_API_BASE_URL` and `GENERATION_API_KEY` (trainer-supplied). Output truncated; never print full passages.

### Answerable question

```bash
curl -s http://127.0.0.1:8000/v1/query -H "Content-Type: application/json"   -d '{"question": "What is the punishment for theft under the BNS?", "pattern": "semantic", "limit": 5, "generate_answer": true}'   | jq '{status, g: (.generation | {outcome, model, provider, context_outcome, text: (.text[:300]), claims: [.claims[] | {t: .text[:80], e: .evidence_labels}], citations: [.citations[] | {label, chunk_id, section_id, act, heading}], trace}), ctx: [.results[] | {chunk_id, section_id, act, score}]}'
```

Expected: `outcome` `answered`, `confidence` `high`, non-empty `text`, each claim has labels, every citation `chunk_id` is in `ctx` and `trace.labels`, `section_id` prefix matches `act`.

### Unsupported question

Same command with `"question": "What is the GST rate on restaurant services?"`. Expected: `insufficient_evidence`, empty `text`, no claims or citations, no `confidence`, `results` still present.

### Generation unavailable (failure)

Start the API with `GENERATION_API_KEY=` empty and repeat the first request. Expected: HTTP 200, `outcome` `unavailable`, empty `text`, `results` present.

### Without `generate_answer`

Omit the flag. Expected: `generation` is `null`, no model call.

### Claim and context tests

```bash
uv run pytest tests/test_grounded_answer.py
```

## Story 3.2 — Streamed Answers with Confidence in Open WebUI

What it adds: `rag-semantic` chat runs the same retrieval + generation path as `/v1/query`, streams the answer, and ends with an evidence-check footer.

Prerequisite: Story 3.1 complete; restart the API so it runs the new code (`uv run uvicorn building_with_rag.app:app --host 127.0.0.1 --port 8000`; an old server on the port keeps the old behavior). Never print passages.

### Postman

1. `POST http://127.0.0.1:8000/v1/query`, Body → raw JSON:
   `{"question": "What is the punishment for theft under the BNS?", "pattern": "semantic", "limit": 5, "generate_answer": true}`
   Expected: `generation.outcome` `answered`, `confidence` `high`, `attempts` (status `passed`), `citations`. Low confidence shows `outcome` `malformed`, empty `text`, `confidence` `low`, `draft_answer`, `issues`.
2. Same request with `"question": "What is the GST rate on restaurant services?"`. Expected: `insufficient_evidence`, no `confidence`, `results` intact.
3. `POST http://127.0.0.1:8000/v1/chat/completions`, raw JSON:
   `{"model": "rag-semantic", "stream": false, "messages": [{"role": "user", "content": "What is the punishment for theft under the BNS?"}]}`
   Expected: `choices[0].message.content` starts `DRAFT — checking evidence`, then the answer, then `Evidence check passed — confidence: high` and `Sources:`. Set `"stream": true` to see `data:` chunks ending `data: [DONE]` (Postman shows them after completion).
4. If `CAPSTONE_API_KEY` is set: add header `Authorization: Bearer <key>` (Authorization tab → Bearer Token). A wrong token returns 401 `invalid_api_key`; unset, no header is needed.
5. `rag-hybrid-reranked` in the chat body, or `"pattern": "hybrid-reranked"` on `/v1/query`, still returns `not_implemented`.

### Open WebUI

1. Admin Settings → Connections → add an OpenAI connection: URL `http://host.docker.internal:8000/v1` (Open WebUI in Docker; use `http://127.0.0.1:8000/v1` if it runs natively), key = `CAPSTONE_API_KEY` (any text if unset). Or use the Pipe: see `open_webui_functions/open-webui-operations.md` (re-paste after changes).
2. Start a chat with the `rag-semantic` model (or `Building with RAG`) and ask: "What is the punishment for theft under the BNS?" Expected: `DRAFT — checking evidence`, answer with `[E1]` labels, then `Evidence check passed — confidence: high` and `Sources:`.
3. Ask: "What is the GST rate on restaurant services?" Expected: one insufficient-evidence sentence, no confidence line.
4. A `Check failed: … Retrying (attempt 2 of 2)…` line, or a closing `DRAFT — low confidence, not the final answer.`, is expected behavior when checks fail; earlier drafts stay visible.
5. Provider rate limits (HTTP 429) show `Answer generation unavailable`; wait a minute and ask again.

Offline: `uv run pytest tests/test_streamed_confidence.py`

## Story 4.1 — Hybrid search

Prerequisite: Stories 2.2, 3.2 complete; `uv run python -m building_with_rag.ingestion.keyword_index` ends with `chunk_text_index` queryable; restart the API. Voyage free keys allow about 3 requests/minute: space the calls. Never print passages.

```bash
curl -s http://127.0.0.1:8000/v1/query -H "Content-Type: application/json"   -d '{"question":"criminal breach of trust","pattern":"hybrid","limit":5}'   | jq '{status, t: (.trace | {semantic, keyword, fusion, contribution}), r: [.results[] | {section_id, score, sr: .semantic_rank, kr: .keyword_rank, fr: .fused_rank, text: .text[:60]}]}'
```

1. Expected: `status` `ok`, non-increasing `score` (= `fused_score`), `fr` = position, each result has `sr` and/or `kr`, trace shows both routes, `fusion`, `contribution`.
2. Add `"generate_answer": true` and compare with `"pattern": "semantic"`: sections, order, and answer may differ; neither mode is better in general.
3. `rag-hybrid` in chat (Open WebUI or `/v1/chat/completions`) streams the same DRAFT/confidence/Sources behavior as `rag-semantic`.
4. Keyword index missing or not ready: hybrid returns 503 `retrieval_not_ready` naming the keyword-index command; semantic still returns `ok`.
5. `hybrid-reranked`, `structured`, `decomposition`, `hyde` still return `not_implemented`.
