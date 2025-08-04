"""
Model Management for MLX Server Nano

Handles model loading, unloading, caching, and text generation using MLX framework.
Provides both streaming and non-streaming generation capabilities with automatic
model lifecycle management.

Features:
- Automatic model loading from Hugging Face Hub
- Model caching with idle timeout unloading
- Thread-safe model management
- Streaming and batch text generation
- Tool calling integration
"""

import logging
import threading
import time
from typing import List, Optional, Tuple

from mlx_lm import generate, load, stream_generate

from .chat_templates import format_messages_for_model
from .config import config
from .schemas import Tool, ToolCall
from .tool_calling import get_tool_parser

# Set up logging
logger = logging.getLogger(__name__)

# Model cache configuration
MODEL_IDLE_TIMEOUT = config.model_idle_timeout

# Global model cache state
_loaded_model = None
_model_name = None
_last_used_time = 0
_unload_timer = None
_lock = threading.Lock()


def get_current_time():
    """Get current timestamp. Separate function for easier testing."""
    return time.time()


def _unload_model():
    """Internal function to unload the current model and reset state."""
    global _loaded_model, _model_name
    logger.info(f"Unloading model '{_model_name}' due to inactivity")
    _loaded_model = None
    _model_name = None


def _schedule_unload():
    """Schedule model unloading after idle timeout."""
    global _unload_timer

    def unload_after_timeout():
        """Unload model if it has been idle for the timeout period."""
        time.sleep(MODEL_IDLE_TIMEOUT)
        with _lock:
            if get_current_time() - _last_used_time >= MODEL_IDLE_TIMEOUT:
                _unload_model()

    _unload_timer = threading.Thread(target=unload_after_timeout, daemon=True)
    _unload_timer.start()


def load_model(name: str):
    """
    Load model using Hugging Face Hub and MLX framework.

    Args:
        name: Model name/path from Hugging Face Hub

    Returns:
        Tuple of (model, tokenizer) from MLX

    Note:
        The mlx-lm library handles downloading from HF Hub automatically
        and uses the standard HF cache location (respects HF_HOME, HUGGINGFACE_HUB_CACHE, etc.)
    """
    global _loaded_model, _model_name, _last_used_time

    logger.info(f"Loading model: {name}")

    with _lock:
        _last_used_time = get_current_time()

        # Return cached model if already loaded
        if _model_name == name and _loaded_model:
            logger.info(f"Model '{name}' already loaded, reusing cached model")
            _schedule_unload()
            return _loaded_model

        logger.info(f"Loading model '{name}' from Hugging Face Hub...")

        try:
            logger.debug(f"Calling mlx_lm.load() with model name: {name}")
            # mlx-lm will automatically download from HF Hub and cache using HF's cache system
            model, tokenizer = load(name)
            logger.info(f"Successfully loaded model '{name}'")

        except Exception as e:
            logger.error(
                f"Failed to load model '{name}': {type(e).__name__}: {str(e)}",
                exc_info=True,
            )
            raise RuntimeError(f"Failed to load model '{name}': {e}")

        _loaded_model = (model, tokenizer)
        _model_name = name

        _schedule_unload()
        logger.info(f"Model '{name}' loaded and cached successfully")
        return _loaded_model


def _setup_generation_kwargs(model_name: str, **kwargs) -> dict:
    """
    Setup generation parameters for MLX models.

    Args:
        model_name: Name of the model for model-specific parameters
        **kwargs: Additional generation parameters

    Returns:
        Dictionary of generation parameters
    """
    generation_kwargs = {
        "max_tokens": kwargs.get("max_tokens", config.default_max_tokens),
    }

    # Add model-specific stop strings
    if "qwen" in model_name.lower():
        generation_kwargs["stop_strings"] = ["✿RESULT✿:", "✿RETURN✿:", "<|im_end|>"]
        logger.debug("Added Qwen stop strings")

    return generation_kwargs


def _try_generate_with_fallback(
    model, tokenizer, prompt: str, **generation_kwargs
) -> str:
    """
    Try to generate text with fallback to streaming if regular generate fails.

    Args:
        model: MLX model instance
        tokenizer: MLX tokenizer instance
        prompt: Input prompt
        **generation_kwargs: Generation parameters

    Returns:
        Generated text response
    """
    try:
        # First try regular generate
        response = generate(
            model=model, tokenizer=tokenizer, prompt=prompt, **generation_kwargs
        )
        logger.debug("Used generate() successfully")
        return response
    except Exception as gen_error:
        logger.warning(f"generate() failed, trying stream_generate: {gen_error}")

        # Fall back to stream_generate and collect all chunks
        response_parts = []
        for chunk in stream_generate(model, tokenizer, prompt, **generation_kwargs):
            response_parts.append(chunk)
        response = "".join(response_parts)
        logger.debug("Fell back to stream_generate successfully")
        return response


def generate_response_with_tools(
    model_name: str, messages: List, tools: Optional[List[Tool]] = None, **kwargs
) -> Tuple[Optional[str], List[ToolCall]]:
    """
    Generate response with tool calling support.

    Args:
        model_name: Name of the model to use
        messages: List of conversation messages
        tools: Optional list of available tools
        **kwargs: Additional generation parameters

    Returns:
        Tuple of (content, tool_calls) where content is the text response
        and tool_calls is a list of detected tool calls
    """
    logger.info(
        f"Generating response - model: {model_name}, messages: {len(messages)}, tools: {len(tools) if tools else 0}"
    )
    logger.debug(f"Messages: {messages}")
    logger.debug(f"Tools: {tools}")
    logger.debug(f"Kwargs: {kwargs}")

    # Load model
    try:
        model, tokenizer = load_model(model_name)
        logger.debug("Model loaded successfully for generation")
    except Exception as e:
        logger.error(f"Failed to load model for generation: {e}", exc_info=True)
        raise

    # Format the prompt with tools
    try:
        logger.debug("Formatting messages for model")
        prompt = format_messages_for_model(messages, model_name, tools)
        logger.debug(f"Formatted prompt length: {len(prompt)} characters")
        logger.debug(f"Formatted prompt preview: {prompt[:200]}...")
    except Exception as e:
        logger.error(f"Failed to format messages: {e}", exc_info=True)
        raise

    # Setup generation parameters
    generation_kwargs = _setup_generation_kwargs(model_name, **kwargs)
    logger.info(f"Generation parameters: {generation_kwargs}")

    # Generate text response
    try:
        logger.debug("Starting text generation")
        response = _try_generate_with_fallback(
            model, tokenizer, prompt, **generation_kwargs
        )
        logger.info(
            f"Generation completed - response length: {len(response)} characters"
        )
        logger.debug(f"Generated response: {response}")
    except Exception as e:
        logger.error(f"Text generation failed: {e}", exc_info=True)
        raise

    # Parse tool calls from response
    try:
        logger.debug("Parsing tool calls from response")
        parser = get_tool_parser(model_name)
        content, tool_calls = parser.parse_tool_calls(response)
        logger.info(
            f"Tool parsing completed - content: {len(content) if content else 0} chars, tool_calls: {len(tool_calls)}"
        )
        logger.debug(f"Parsed content: {content}")
        logger.debug(f"Parsed tool calls: {tool_calls}")
    except Exception as e:
        logger.error(f"Tool call parsing failed: {e}", exc_info=True)
        raise

    return content, tool_calls


def generate_response_stream(
    model_name: str, messages: list, tools: Optional[List[Tool]] = None, **kwargs
):
    """
    Generate streaming response with tool calling support.

    Args:
        model_name: Name of the model to use
        messages: List of conversation messages
        tools: Optional list of available tools
        **kwargs: Additional generation parameters

    Yields:
        Individual text chunks as they are generated

    Note:
        Tool calls are processed after full response completion in streaming mode.
        This is a limitation that could be improved with smarter parsing.
    """
    logger.info(f"Starting streaming generation for model: {model_name}")

    # Load model (load_model has its own locking)
    model, tokenizer = load_model(model_name)

    # Update last used time and schedule unload
    with _lock:
        global _last_used_time
        _last_used_time = get_current_time()
        _schedule_unload()

    # Format messages for the model
    try:
        prompt = format_messages_for_model(messages, model_name, tools)
        logger.debug(f"Formatted prompt for streaming: {prompt}")
    except Exception as e:
        logger.error(f"Failed to format messages: {e}", exc_info=True)
        raise

    # Setup generation parameters
    generation_kwargs = _setup_generation_kwargs(model_name, **kwargs)
    logger.info(f"Streaming generation parameters: {generation_kwargs}")

    try:
        logger.debug("Starting streaming text generation with stream_generate")

        # Use stream_generate to yield chunks as they come
        full_response = ""
        for chunk in stream_generate(model, tokenizer, prompt, **generation_kwargs):
            # Extract text from GenerationResponse object
            chunk_text = chunk.text if hasattr(chunk, "text") else str(chunk)
            full_response += chunk_text
            yield chunk_text

        logger.info(
            f"Streaming generation completed - total length: {len(full_response)} characters"
        )
        logger.debug(f"Full streamed response: {full_response}")

    except Exception as e:
        logger.error(f"Streaming text generation failed: {e}", exc_info=True)
        raise
