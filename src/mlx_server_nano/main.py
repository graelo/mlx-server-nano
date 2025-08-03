from fastapi import FastAPI, HTTPException
import time
import uuid
import argparse
from pathlib import Path

from .schemas import ChatCompletionRequest
from .model_manager import generate_response_with_tools
from .config import config

app = FastAPI(title="MLX OpenAI-Compatible API", version="0.2.0")


@app.post("/v1/chat/completions")
def chat_completion(body: ChatCompletionRequest):
    try:
        # Generate response with tool calling support
        content, tool_calls = generate_response_with_tools(
            model_name=body.model,
            messages=body.messages,
            tools=body.tools,
            max_tokens=body.max_tokens,
            temperature=body.temperature,
        )

        # Prepare the response message
        response_message = {
            "role": "assistant",
            "content": content,
        }

        # Add tool calls if any were detected
        if tool_calls:
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

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/models")
def list_models():
    """List available models"""
    models_dir = Path(config.model_cache_dir)
    available_models = []

    if models_dir.exists():
        for model_dir in models_dir.iterdir():
            if model_dir.is_dir():
                available_models.append(
                    {
                        "id": model_dir.name,
                        "object": "model",
                        "created": int(time.time()),
                        "owned_by": "mlx-server",
                    }
                )

    # Fallback to hardcoded models if no models found
    if not available_models:
        available_models = [
            {
                "id": "qwen3-30b-a3b-instruct-2507",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "mlx-server",
            },
            {
                "id": "devstral-small-2507",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "mlx-server",
            },
        ]

    return {
        "object": "list",
        "data": available_models,
    }


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
        "--model-cache-dir",
        default=config.model_cache_dir,
        help=f"Directory containing models (default: {config.model_cache_dir})",
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
    config.model_cache_dir = args.model_cache_dir
    config.log_level = args.log_level

    print(f"Starting MLX Server Nano on {config.host}:{config.port}")
    print(f"Model cache directory: {config.model_cache_dir}")
    print(f"Log level: {config.log_level}")

    uvicorn.run(
        "mlx_server_nano.main:app",
        host=config.host,
        port=config.port,
        log_level=config.log_level.lower(),
        reload=args.reload,
    )
