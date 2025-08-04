"""
MLX Server Nano - OpenAI-compatible API server for Apple Silicon

A lightweight FastAPI server that provides OpenAI-compatible chat completion endpoints
using Apple's MLX framework for running language models efficiently on Apple Silicon.

Features:
- OpenAI API compatibility
- Streaming and non-streaming responses
- Tool calling support
- Automatic model management with caching
- Multi-model support with appropriate chat templates
"""

import argparse
import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Generator, Callable

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

from .config import config
from .model_manager import (
    generate_response_with_tools,
    generate_response_stream,
    start_model_unloader,
    stop_model_unloader,
)
from .schemas import ChatCompletionRequest, Message, Tool

# Set up logging
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage application lifespan for background tasks."""
    # Startup
    logger.info("Starting MLX Server Nano")
    await start_model_unloader()
    yield
    # Shutdown
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
) -> Callable[[], Generator[str, None, None]]:
    """
    Create a streaming response generator for chat completions.

    Args:
        model_name: Name of the model to use for generation
        messages: List of conversation messages
        tools: Optional list of available tools
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature

    Returns:
        Generator function that yields SSE-formatted streaming chunks
    """

    def generate() -> Generator[str, None, None]:
        """Generator function that streams chat completion chunks in OpenAI format."""
        logger.debug("Starting streaming generation")

        try:
            # Create the completion ID and metadata
            completion_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
            created_time = int(time.time())

            # Get the streaming generator from model_manager
            logger.debug("Creating streaming generator from model_manager")
            stream_generator = generate_response_stream(
                model_name=model_name,
                messages=messages,
                tools=tools,
                max_tokens=max_tokens,
                temperature=temperature,
            )

            logger.debug("Starting to iterate over stream_generator")
            chunk_count = 0

            # Stream each chunk as it comes from the model
            for chunk in stream_generator:
                chunk_count += 1
                logger.debug(f"Yielding chunk {chunk_count}: {chunk}")

                chunk_data = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created_time,
                    "model": model_name,
                    "choices": [
                        {"index": 0, "delta": {"content": chunk}, "finish_reason": None}
                    ],
                }
                yield f"data: {json.dumps(chunk_data)}\n\n"

            logger.debug(f"Finished streaming {chunk_count} chunks")

            # Send final chunk with finish_reason
            final_chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created_time,
                "model": model_name,
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": "stop",  # Note: tool calls in streaming need future enhancement
                    }
                ],
            }
            logger.debug("Sending final chunk")
            yield f"data: {json.dumps(final_chunk)}\n\n"

            # Send the [DONE] marker
            logger.debug("Sending [DONE] marker")
            yield "data: [DONE]\n\n"

        except Exception as e:
            logger.error(f"Streaming generation failed: {e}", exc_info=True)

            # Send error chunk in OpenAI format
            error_chunk = {
                "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model_name,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "error": {"message": str(e), "type": "internal_error"},
            }
            yield f"data: {json.dumps(error_chunk)}\n\n"
            yield "data: [DONE]\n\n"

    return generate


def _create_streaming_headers() -> dict[str, str]:
    """Create common headers for streaming responses."""
    return {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Headers": "*",
    }


def _create_chat_response_message(content: str, tool_calls: list) -> dict:
    """Create a properly formatted response message."""
    response_message = {"role": "assistant", "content": content}

    if tool_calls:
        logger.debug(f"Adding {len(tool_calls)} tool calls to response")
        response_message["tool_calls"] = [
            {"id": tc.id, "type": tc.type, "function": tc.function} for tc in tool_calls
        ]

    return response_message


def _create_chat_completion_response(
    model: str,
    message: dict,
    tool_calls: list,
) -> dict:
    """Create a properly formatted chat completion response."""
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
            }
        ],
        "usage": {
            "prompt_tokens": 0,  # MLX doesn't provide token counts
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    }


@app.post("/v1/chat/completions")
def chat_completion(body: ChatCompletionRequest) -> object:
    """
    Handle chat completion requests with support for both streaming and non-streaming modes.

    Supports:
    - OpenAI-compatible API format
    - Tool calling functionality
    - Real-time streaming responses
    - Non-streaming batch responses
    """
    logger.info(
        f"Chat completion request - model: {body.model}, messages: {len(body.messages)}, stream: {body.stream}"
    )
    logger.debug(f"Request body: {body}")

    # Handle streaming requests
    if body.stream:
        logger.info("Processing streaming request")

        generator = create_streaming_response(
            model_name=body.model,
            messages=body.messages,
            tools=body.tools,
            max_tokens=body.max_tokens,
            temperature=body.temperature,
        )

        logger.info("Created streaming generator, returning StreamingResponse")
        return StreamingResponse(
            generator(),
            media_type="text/event-stream",
            headers=_create_streaming_headers(),
        )

    # Handle non-streaming requests
    logger.info("Processing non-streaming request")

    try:
        # Generate response with tool calling support
        logger.debug("Calling generate_response_with_tools")
        content, tool_calls = generate_response_with_tools(
            model_name=body.model,
            messages=body.messages,
            tools=body.tools,
            max_tokens=body.max_tokens,
            temperature=body.temperature,
        )

        logger.info(
            f"Response generated - content: {len(content) if content else 0} chars, tool_calls: {len(tool_calls)}"
        )

        # Build the response
        response_message = _create_chat_response_message(content, tool_calls)
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
    """
    List available models.

    Note: This server supports on-demand loading of any MLX-compatible model
    from Hugging Face Hub. Specify the model name directly in your requests.
    Popular models are available from the mlx-community organization.
    """
    logger.info("Models list requested")
    try:
        # Since we support on-demand loading from HF Hub, return minimal response
        available_models = []

        # Note: Could optionally add a few example models here, but keeping it empty
        # to avoid confusion about what models are "available" vs "can be loaded"

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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=config.host, port=config.port)


def main() -> None:
    """Entry point for the mlx-server-nano command"""
    import uvicorn

    parser = argparse.ArgumentParser(
        description="MLX Server Nano - OpenAI-compatible API server"
    )
    parser.add_argument(
        "--host", default=config.host, help=f"Host to bind to (default: {config.host})"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=config.port,
        help=f"Port to bind to (default: {config.port})",
    )
    parser.add_argument(
        "--log-level",
        default=config.log_level,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help=f"Log level (default: {config.log_level})",
    )
    parser.add_argument(
        "--reload", action="store_true", help="Enable auto-reload for development"
    )

    args = parser.parse_args()

    # Update config with command line arguments
    config.host = args.host
    config.port = args.port
    config.log_level = args.log_level

    # Set up logging
    log_level = getattr(logging, config.log_level, logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )

    # Set specific loggers
    logging.getLogger("mlx_server_nano").setLevel(log_level)
    logging.getLogger("uvicorn").setLevel(logging.INFO)

    logger.info("MLX Server Nano starting up")
    logger.info(f"Log level set to: {config.log_level}")

    print(f"Starting MLX Server Nano on {config.host}:{config.port}")
    print(f"Log level: {config.log_level}")
    print(f"Using Hugging Face cache (HF_HOME: {os.environ.get('HF_HOME', 'default')})")

    logger.info(f"Starting server on {config.host}:{config.port}")
    logger.debug(f"HF_HOME: {os.environ.get('HF_HOME', 'not set')}")
    logger.debug(
        f"HUGGINGFACE_HUB_CACHE: {os.environ.get('HUGGINGFACE_HUB_CACHE', 'not set')}"
    )

    uvicorn.run(
        "mlx_server_nano.main:app",
        host=config.host,
        port=config.port,
        log_level=config.log_level.lower(),
        reload=args.reload,
    )
