"""FastAPI application: health check, query endpoint, and OpenAI-compatible chat adapter."""

from __future__ import annotations

import hmac
import json
import time
import uuid
from collections.abc import Iterator

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse

from building_with_rag import pipeline
from building_with_rag.config import get_settings
from building_with_rag.models import ChatCompletionRequest, QueryRequest, QueryResult
from building_with_rag.registry import MODE_BY_MODEL_ID, MODEL_ID_BY_MODE
from building_with_rag.retrieval import semantic

_LOW_CONFIDENCE_NOTE = (
    "The answer failed its evidence check and was withheld as a draft; "
    "passages and issues are still returned."
)
_OUTCOME_NOTES = {
    "answered": "Answer generated from the cited passages.",
    "insufficient_evidence": "The retrieved passages do not support an answer.",
    "unavailable": "Answer generation is unavailable; retrieved passages are still returned.",
    "malformed": "The model reply was invalid and was discarded; passages are still returned.",
}



def _note(generation) -> str:
    if generation.confidence == "low":
        return _LOW_CONFIDENCE_NOTE
    return _OUTCOME_NOTES[generation.outcome]


app = FastAPI(title="building-with-rag", version="0.1.0")


@app.get("/healthz")
def healthz() -> dict[str, str]:
    """Safe status only; never exposes secrets or configuration values."""
    return {"status": "ok"}


@app.post("/v1/query", response_model=QueryResult)
def query(request: QueryRequest) -> QueryResult | JSONResponse:
    try:
        result = pipeline.retrieve(request)
    except semantic.RetrievalError as exc:
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": {"code": exc.code, "message": exc.message}},
        )
    if request.generate_answer and request.pattern == "semantic":
        result.generation = pipeline.final_of(pipeline.answer_events(request.question, result))
        result.message = f"{result.message} {_note(result.generation)}"
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


def _openai_error(
    message: str,
    status_code: int,
    error_type: str = "invalid_request_error",
    code: str | None = None,
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"error": {"message": message, "type": error_type, "param": None, "code": code}},
    )


def _authorized(authorization: str | None) -> bool:
    key = get_settings().capstone_api_key
    if not key:
        return True
    expected = f"Bearer {key}".encode()
    return hmac.compare_digest((authorization or "").encode(), expected)

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


def _chunk(completion_id: str, created: int, model: str, choices: list[dict]) -> str:
    chunk = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": choices,
    }
    return f"data: {json.dumps(chunk)}\n\n"


def _stream_chat_completion(
    completion_id: str, created: int, model: str, n: int, pieces: Iterator[str]
) -> Iterator[str]:
    indexes = range(max(1, n))
    yield _chunk(
        completion_id, created, model,
        [{"index": i, "delta": {"role": "assistant"}, "finish_reason": None} for i in indexes],
    )
    for piece in pieces:
        if piece:
            yield _chunk(
                completion_id, created, model,
                [{"index": i, "delta": {"content": piece}, "finish_reason": None} for i in indexes],
            )
    yield _chunk(
        completion_id, created, model,
        [{"index": i, "delta": {}, "finish_reason": "stop"} for i in indexes],
    )
    yield "data: [DONE]\n\n"


@app.post("/v1/chat/completions")
def chat_completions(
    chat_request: ChatCompletionRequest, authorization: str | None = Header(default=None)
) -> object:
    if not _authorized(authorization):
        return _openai_error(
            "Invalid API key.", 401, "invalid_request_error", code="invalid_api_key"
        )
    try:
        query_request = _build_query_request(chat_request)
        result = pipeline.retrieve(query_request)
    except HTTPException as exc:
        return _openai_error(str(exc.detail), exc.status_code)
    except semantic.RetrievalError as exc:
        return _openai_error(exc.message, exc.status_code, "api_error", code=exc.code)

    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())
    pieces = pipeline.chat_pieces(query_request.question, result)

    if chat_request.stream:
        return StreamingResponse(
            _stream_chat_completion(
                completion_id, created, chat_request.model, chat_request.n, pieces
            ),
            media_type="text/event-stream",
        )

    content = "".join(pieces)
    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": created,
        "model": chat_request.model,
        "choices": [
            {
                "index": i,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
            for i in range(max(1, chat_request.n))
        ],
    }
