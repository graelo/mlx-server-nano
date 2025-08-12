"""
MLX Server Nano FastAPI app and routes
"""

import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Any, Generator, Callable

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

import json
from .model_manager import (
    generate_response_with_tools,
    generate_response_stream,
    generate_response_with_tools_cached,
    generate_response_stream_cached,
    get_conversation_cache_stats,
    start_model_unloader,
    stop_model_unloader,
)
from .schemas import ChatCompletionRequest, Message, Tool, ToolCall

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    logger.info("Starting MLX Server Nano")
    await start_model_unloader()
    yield
    logger.info("Shutting down MLX Server Nano")
    await stop_model_unloader()


app = FastAPI(
    title="MLX OpenAI-Compatible API",
    version="0.2.0",
    description="OpenAI-compatible API server for Apple Silicon using MLX",
    lifespan=lifespan,
)


def create_streaming_response(
    model_name: str,
    messages: list[Message],
    tools: list[Tool] | None,
    max_tokens: int | None,
    temperature: float | None,
    stop: str | list[str] | None,
    conversation_id: str | None = None,  # Add conversation_id parameter
) -> Callable[[], Generator[str, None, None]]:
    def generate() -> Generator[str, None, None]:
        logger.debug("Starting streaming generation")
        try:
            completion_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
            created_time = int(time.time())
            stream_generator = generate_response_stream_cached(
                model_name=model_name,
                messages=messages,
                tools=tools,
                conversation_id=conversation_id,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=stop,
            )
            chunk_count = 0
            final_finish_reason = "stop"  # Default finish reason
            accumulated_response = ""  # Track full response for tool call parsing

            for chunk_data in stream_generator:
                # Handle new tuple format: (text_chunk, finish_reason)
                if isinstance(chunk_data, tuple):
                    chunk_text, finish_reason = chunk_data
                    if finish_reason is not None:
                        # This is the final chunk indicator
                        final_finish_reason = finish_reason

                        # If we have tool calls, send them in OpenAI format instead of raw text
                        if finish_reason == "tool_calls" and accumulated_response:
                            from .model_manager import parse_tool_calls

                            tool_calls = parse_tool_calls(accumulated_response)
                            if tool_calls:
                                # Send tool calls as delta
                                tool_chunk = {
                                    "id": completion_id,
                                    "object": "chat.completion.chunk",
                                    "created": created_time,
                                    "model": model_name,
                                    "system_fingerprint": None,
                                    "choices": [
                                        {
                                            "index": 0,
                                            "delta": {
                                                "tool_calls": [
                                                    {
                                                        "id": tc.id,
                                                        "type": tc.type,
                                                        "function": tc.function,
                                                    }
                                                    for tc in tool_calls
                                                ]
                                            },
                                            "logprobs": None,
                                            "finish_reason": None,
                                        }
                                    ],
                                }
                                logger.debug(
                                    f"Full streamed OpenAI API response: {json.dumps(tool_chunk, indent=2)}"
                                )
                                yield f"data: {json.dumps(tool_chunk)}\n\n"
                        break
                else:
                    # Backward compatibility: treat as text chunk
                    chunk_text = chunk_data

                if chunk_text:  # Only yield non-empty chunks
                    accumulated_response += chunk_text

                    # Check if we're in a tool call - don't stream tool call format as content
                    # Tool calls start with [TOOL_CALLS], so detect them early
                    is_tool_call_content = "[TOOL_CALLS]" in accumulated_response

                    # For regular text (not tool calls), stream as content
                    if not is_tool_call_content:
                        chunk_count += 1
                        logger.debug(f"Yielding chunk {chunk_count}: {chunk_text}")
                        chunk_data = {
                            "id": completion_id,
                            "object": "chat.completion.chunk",
                            "created": created_time,
                            "model": model_name,
                            "system_fingerprint": None,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"content": chunk_text},
                                    "logprobs": None,
                                    "finish_reason": None,
                                }
                            ],
                        }
                        yield f"data: {json.dumps(chunk_data)}\n\n"
                    else:
                        logger.debug(f"Skipping tool call content chunk: {chunk_text}")

            # Send final chunk with the determined finish reason
            final_chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created_time,
                "model": model_name,
                "system_fingerprint": None,
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "logprobs": None,
                        "finish_reason": final_finish_reason,
                    }
                ],
            }
            logger.info(
                f"Streaming completed with finish_reason: {final_finish_reason}"
            )
            yield f"data: {json.dumps(final_chunk)}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            logger.error(f"Streaming generation failed: {e}", exc_info=True)
            error_chunk = {
                "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model_name,
                "system_fingerprint": None,
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "logprobs": None,
                        "finish_reason": "stop",
                    }
                ],
                "error": {"message": str(e), "type": "internal_error"},
            }
            yield f"data: {json.dumps(error_chunk)}\n\n"
            yield "data: [DONE]\n\n"

    return generate


def _create_streaming_headers() -> dict[str, str]:
    return {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Headers": "*",
    }


def _create_chat_response_message(
    content: str, tool_calls: list[ToolCall]
) -> dict[str, Any]:
    response_message: dict[str, Any] = {
        "role": "assistant",
        "content": content if content is not None else "",
    }
    if tool_calls:
        logger.debug(f"Adding {len(tool_calls)} tool calls to response")
        response_message["tool_calls"] = [
            {"id": tc.id, "type": tc.type, "function": tc.function} for tc in tool_calls
        ]
    return response_message


def _create_chat_completion_response(
    model: str,
    message: dict,
    tool_calls: list[ToolCall],
) -> dict[str, Any]:
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": "tool_calls" if tool_calls else "stop",
                "logprobs": None,  # OpenAI compatibility
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
        "system_fingerprint": None,  # OpenAI compatibility
    }


@app.post("/v1/chat/completions")
def chat_completion(body: ChatCompletionRequest) -> object:
    logger.info(
        f"Chat completion request - model: {body.model}, messages: {len(body.messages)}, stream: {body.stream}"
    )

    # Log the stop parameter if provided
    if body.stop is not None:
        logger.info(f"Stop words received: {body.stop}")

    logger.debug(f"Request body: {body}")
    if body.stream:
        logger.info("Processing streaming request")
        generator = create_streaming_response(
            model_name=body.model,
            messages=body.messages,
            tools=body.tools,
            max_tokens=body.max_tokens,
            temperature=body.temperature,
            stop=body.stop,
            conversation_id=body.user,
        )
        logger.info("Created streaming generator, returning StreamingResponse")
        return StreamingResponse(
            generator(),
            media_type="text/event-stream",
            headers=_create_streaming_headers(),
        )
    logger.info("Processing non-streaming request")
    try:
        content, tool_calls = generate_response_with_tools_cached(
            model_name=body.model,
            messages=body.messages,
            tools=body.tools,
            conversation_id=body.user,
            max_tokens=body.max_tokens,
            temperature=body.temperature,
            stop=body.stop,
        )
        logger.info(
            f"Response generated - content: {len(content) if content else 0} chars, tool_calls: {len(tool_calls)}"
        )
        # Ensure content is always a string
        response_message = _create_chat_response_message(
            content if content is not None else "", tool_calls
        )
        response = _create_chat_completion_response(
            body.model, response_message, tool_calls
        )
        logger.info(f"Chat completion successful - response ID: {response['id']}")
        logger.debug(f"Full response: {response}")
        return response
    except Exception as e:
        logger.error(
            f"Chat completion failed: {type(e).__name__}: {str(e)}", exc_info=True
        )
        error_detail = f"Internal error: {type(e).__name__}: {str(e)}"
        raise HTTPException(status_code=500, detail=error_detail)


@app.get("/v1/models")
def list_models() -> dict[str, object]:
    logger.info("Models list requested")
    try:
        available_models = []
        logger.info("Returning empty models list (on-demand loading supported)")
        return {
            "object": "list",
            "data": available_models,
        }
    except Exception as e:
        logger.error(
            f"Failed to list models: {type(e).__name__}: {str(e)}", exc_info=True
        )
        raise HTTPException(status_code=500, detail=f"Failed to list models: {str(e)}")


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "healthy"}


@app.get("/admin/cache/stats")
def get_cache_statistics() -> dict[str, int | float | bool]:
    """Get conversation cache statistics."""
    logger.info("Cache statistics requested")
    try:
        stats = get_conversation_cache_stats()
        logger.debug(f"Cache stats: {stats}")
        return stats
    except Exception as e:
        logger.error(f"Failed to get cache statistics: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Failed to get cache statistics: {str(e)}"
        )
