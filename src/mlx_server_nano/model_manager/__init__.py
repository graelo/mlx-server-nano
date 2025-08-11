"""
Model Manager Package

Public API for MLX Server Nano model management functionality.
This module provides the main interface for model loading, text generation,
and tool calling capabilities.

Key Functions:
- generate_response_with_tools: Non-streaming generation with tool support
- generate_response_stream: Streaming generation with tool support
- start_model_unloader: Start background model management
- stop_model_unloader: Stop background model management
- parse_tool_calls: Parse tool calls from model responses
"""

# Import the main public functions from their respective modules
from .generation import (
    generate_response_with_tools,
    generate_response_stream,
    _setup_generation_kwargs,
    _try_generate_with_fallback,
)
from .background_tasks import (
    start_model_unloader,
    stop_model_unloader,
    _schedule_unload,
)
from .tool_calling import parse_tool_calls, has_tool_calls
from .cache import load_model, get_current_time, _unload_model, MODEL_IDLE_TIMEOUT
from .stop_sequences import _get_stop_sequences

# Import submodules for direct access in tests
from . import cache, background_tasks

# Import mlx_lm functions for test mocking compatibility
from mlx_lm.utils import load
from mlx_lm.generate import generate, stream_generate

# Public API exports
__all__ = [
    "generate_response_with_tools",
    "generate_response_stream",
    "start_model_unloader",
    "stop_model_unloader",
    "parse_tool_calls",
    "has_tool_calls",
    "load_model",
    "get_current_time",
    "_unload_model",
    "_schedule_unload",
    "_setup_generation_kwargs",
    "_get_stop_sequences",
    "_try_generate_with_fallback",
    "MODEL_IDLE_TIMEOUT",
    # Expose submodules for direct access in tests
    "cache",
    "background_tasks",
    # Expose mlx_lm functions for test mocking
    "load",
    "generate",
    "stream_generate",
]
