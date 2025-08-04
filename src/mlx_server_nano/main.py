from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
import time
import uuid
import argparse
import os
import logging
import traceback
import json

from .schemas import ChatCompletionRequest
from .model_manager import (
    generate_response_with_tools,
    generate_response_stream,
    get_available_models,
)
from .config import config

# Set up logging
logger = logging.getLogger(__name__)

app = FastAPI(title="MLX OpenAI-Compatible API", version="0.2.0")


def create_streaming_response(
    model_name: str, messages, tools, max_tokens, temperature
):
    """Create a streaming response generator for chat completions"""

    def generate():
        print("[DEBUG] Starting streaming generator")
        logger.debug("Starting streaming generator")
        try:
            # Create the completion ID and metadata
            completion_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
            created_time = int(time.time())

            # Get the streaming generator from model_manager
            print("[DEBUG] Creating streaming generator from model_manager")
            logger.debug("Creating streaming generator from model_manager")
            stream_generator = generate_response_stream(
                model_name=model_name,
                messages=messages,
                tools=tools,
                max_tokens=max_tokens,
                temperature=temperature,
            )

            print("[DEBUG] Starting to iterate over stream_generator")
            logger.debug("Starting to iterate over stream_generator")
            # Stream each chunk as it comes from the model
            chunk_count = 0
            for chunk in stream_generator:
                chunk_count += 1
                print(f"[DEBUG] Yielding chunk {chunk_count}: {chunk}")
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
                        "finish_reason": "stop",  # For now, always stop (tool calls in streaming need more work)
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


@app.post("/v1/chat/completions")
def chat_completion(body: ChatCompletionRequest):
    print(f"[DEBUG] Chat completion request - stream: {body.stream}")
    logger.info(
        f"Chat completion request - model: {body.model}, messages: {len(body.messages)}, stream: {body.stream}"
    )
    logger.debug(f"Request body: {body}")

    # Handle streaming requests
    if body.stream:
        print("[DEBUG] === STREAMING REQUEST DETECTED ===")
        logger.info("=== STREAMING REQUEST DETECTED ===")
        logger.info("Handling streaming request")
        generator = create_streaming_response(
            model_name=body.model,
            messages=body.messages,
            tools=body.tools,
            max_tokens=body.max_tokens,
            temperature=body.temperature,
        )
        print("[DEBUG] Created streaming generator, returning StreamingResponse")
        logger.info("Created streaming generator, returning StreamingResponse")
        return StreamingResponse(
            generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "*",
            },
        )

    # Handle non-streaming requests (existing logic)
    print("[DEBUG] === NON-STREAMING REQUEST ===")
    logger.info("=== NON-STREAMING REQUEST ===")
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

        # Prepare the response message
        response_message = {
            "role": "assistant",
            "content": content,
        }

        # Add tool calls if any were detected
        if tool_calls:
            logger.debug(f"Adding {len(tool_calls)} tool calls to response")
            response_message["tool_calls"] = [
                {"id": tc.id, "type": tc.type, "function": tc.function}
                for tc in tool_calls
            ]

        # Build the response
        response = {
            "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": body.model,
            "choices": [
                {
                    "index": 0,
                    "message": response_message,
                    "finish_reason": "tool_calls" if tool_calls else "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 0,  # MLX doesn't provide token counts
                "completion_tokens": 0,
                "total_tokens": 0,
            },
        }

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
def list_models():
    """List available models"""
    logger.info("Models list requested")
    try:
        available_model_names = get_available_models()
        available_models = []

        for model_name in available_model_names:
            available_models.append(
                {
                    "id": model_name,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "mlx-server",
                }
            )

        logger.info(f"Returning {len(available_models)} models")
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
def health_check():
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=config.host, port=config.port)


def main():
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
