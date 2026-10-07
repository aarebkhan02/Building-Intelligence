"""FastAPI application: health check, query endpoint, and OpenAI-compatible chat adapter."""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import AsyncIterator

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse

from building_with_rag.config import get_settings
from building_with_rag.generation.answer import generate_answer
from building_with_rag.models import ChatCompletionRequest, QueryRequest, QueryResult
from building_with_rag.registry import MODE_BY_MODEL_ID, MODEL_ID_BY_MODE, run_pattern
from building_with_rag.retrieval import semantic

_OUTCOME_NOTES = {
    "answered": "Answer generated from the cited passages.",
    "insufficient_evidence": "The retrieved passages do not support an answer.",
    "unavailable": "Answer generation is unavailable; retrieved passages are still returned.",
    "malformed": "The model reply was invalid and was discarded; passages are still returned.",
}

app = FastAPI(title="building-with-rag", version="0.1.0")


@app.get("/healthz")
def healthz() -> dict[str, str]:
    """Safe status only; never exposes secrets or configuration values."""
    return {"status": "ok"}


@app.post("/v1/query", response_model=QueryResult)
def query(request: QueryRequest) -> QueryResult | JSONResponse:
    if request.pattern != "semantic":
        return run_pattern(request)
    problems = semantic.validate_scope(request)
    if problems:
        raise HTTPException(status_code=422, detail=" ".join(problems))
    try:
        result = semantic.run_semantic(request)
    except semantic.RetrievalError as exc:
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": {"code": exc.code, "message": exc.message}},
        )
    if request.generate_answer:
        result.generation = generate_answer(request.question, result)
        result.message = f"{result.message} {_OUTCOME_NOTES[result.generation.outcome]}"
    return result


@app.get("/v1/models")
def list_models() -> dict[str, object]:
    now = int(time.time())
    return {
        "object": "list",
        "data": [
            {"id": model_id, "object": "model", "created": now, "owned_by": "building-with-rag"}
            for model_id in MODEL_ID_BY_MODE.values()
        ],
    }


def _openai_error(message: str, status_code: int, error_type: str = "invalid_request_error") -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"error": {"message": message, "type": error_type, "param": None, "code": None}},
    )


def _build_query_request(chat_request: ChatCompletionRequest) -> QueryRequest:
    settings = get_settings()

    pattern = MODE_BY_MODEL_ID.get(chat_request.model)
    if pattern is None:
        raise HTTPException(status_code=404, detail=f"Unknown model '{chat_request.model}'.")

    user_messages = [m for m in chat_request.messages if m.role == "user"]
    if not user_messages:
        raise HTTPException(status_code=400, detail="At least one user message is required.")
    question = user_messages[-1].content

    options = chat_request.rag_options
    if options and options.pattern:
        pattern = options.pattern

    return QueryRequest(
        question=question,
        pattern=pattern,
        caller_id=settings.webui_demo_caller_id,
        filters=options.filters if options else None,
        limit=options.limit if options and options.limit else 5,
        generate_answer=True,
        required_acts=options.required_acts if options else None,
        chapter=options.chapter if options else None,
    )


async def _stream_chat_completion(
    completion_id: str, created: int, model: str, content: str
) -> AsyncIterator[str]:
    role_chunk = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
    }
    yield f"data: {json.dumps(role_chunk)}\n\n"

    content_chunk = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}],
    }
    yield f"data: {json.dumps(content_chunk)}\n\n"

    stop_chunk = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    yield f"data: {json.dumps(stop_chunk)}\n\n"
    yield "data: [DONE]\n\n"


@app.post("/v1/chat/completions")
def chat_completions(chat_request: ChatCompletionRequest) -> object:
    try:
        query_request = _build_query_request(chat_request)
    except HTTPException as exc:
        return _openai_error(str(exc.detail), exc.status_code)

    result = run_pattern(query_request)
    answer_text = result.message or "not_implemented"
    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())

    if chat_request.stream:
        return StreamingResponse(
            _stream_chat_completion(completion_id, created, chat_request.model, answer_text),
            media_type="text/event-stream",
        )

    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": created,
        "model": chat_request.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": answer_text},
                "finish_reason": "stop",
            }
        ],
    }
