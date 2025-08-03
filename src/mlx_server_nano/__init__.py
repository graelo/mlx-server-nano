"""
MLX Server Nano - OpenAI-compatible API server for Apple Silicon

A lightweight, production-ready FastAPI server that provides OpenAI-compatible
chat completion endpoints using Apple's MLX framework for efficient language
model inference on Apple Silicon.

Key Features:
- OpenAI API compatibility (chat completions, models)
- Streaming and non-streaming responses
- Tool calling support with model-specific parsers
- Automatic model management with intelligent caching
- Multi-model support (Devstral, Qwen3, and more)
- Production-ready error handling and logging

Usage:
    from mlx_server_nano.main import app
    # or
    $ mlx-server-nano --host 0.0.0.0 --port 8000

Version: 0.2.0
"""

__version__ = "0.2.0"
__author__ = "graelo"
__description__ = "OpenAI-compatible API server for Apple Silicon using MLX"

from .app import app
from .config import config
from .schemas import ChatCompletionRequest, ChatCompletionResponse

__all__ = ["app", "config", "ChatCompletionRequest", "ChatCompletionResponse"]
