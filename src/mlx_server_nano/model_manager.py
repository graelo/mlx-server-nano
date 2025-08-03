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

import asyncio
import gc
import json
import logging
import random
import re
import string
import threading
import time
from typing import List, Optional, Tuple, Union

import mlx.core as mx
from mlx_lm.generate import generate, stream_generate
from mlx_lm.utils import load
from mistral_common.tokens.tokenizers.base import SpecialTokenPolicy

from .config import config
from .schemas import Tool, ToolCall, tools_to_openai_format

# Set up logging
logger = logging.getLogger(__name__)

# Model cache configuration
MODEL_IDLE_TIMEOUT = config.model_idle_timeout

# Global model cache state
_loaded_model = None
_model_name = None
_last_used_time = 0
_lock = threading.Lock()

# Background task management for model unloading
_model_unloader_task: Optional[asyncio.Task] = None
_unload_requested = asyncio.Event()
_shutdown_requested = asyncio.Event()


def _convert_messages_to_dicts(messages):
    """
    Convert Pydantic Message objects to dictionaries for MLX-LM compatibility.

    Args:
        messages: List of Message objects or dictionaries

    Returns:
        List of dictionaries
    """
    converted_messages = []
    for msg in messages:
        if hasattr(msg, "model_dump"):
            # Pydantic v2 model
            converted_messages.append(msg.model_dump())
        elif hasattr(msg, "dict"):
            # Pydantic v1 model
            converted_messages.append(msg.dict())
        else:
            # Already a dictionary
            converted_messages.append(msg)
    return converted_messages


def parse_tool_calls(response: str) -> List[ToolCall]:
    """
    Parse tool calls from model response using MLX-LM compatible format.

    Supports multiple formats:
    - Mistral format: [TOOL_CALLS]function_name[ARGS]{"arg1": "value1"}
    - Multiple calls: [TOOL_CALLS]func1[ARGS]{...}[TOOL_CALLS]func2[ARGS]{...}

    Args:
        response: The model response text

    Returns:
        List of parsed ToolCall objects
    """
    tool_calls = []

    # Mistral format: [TOOL_CALLS]function_name[ARGS]{"arguments": "json"}
    mistral_pattern = r"\[TOOL_CALLS\]([^\[]+)\[ARGS\](\{.*?\})"
    matches = re.findall(mistral_pattern, response, re.DOTALL)

    for function_name, args_json in matches:
        function_name = function_name.strip()
        try:
            # Parse the JSON arguments
            arguments = json.loads(args_json.strip())

            # Create tool call
            tool_call = ToolCall(
                id="".join(
                    random.choices(string.ascii_letters + string.digits, k=9)
                ),  # 9 alphanumeric characters
                type="function",
                function={
                    "name": function_name,
                    "arguments": json.dumps(arguments),  # Store as JSON string
                },
            )
            tool_calls.append(tool_call)

        except json.JSONDecodeError as e:
            logger.warning(
                f"Failed to parse tool call arguments for {function_name}: {e}"
            )
            continue

    return tool_calls


def has_tool_calls(response: str) -> bool:
    """
    Simple tool call detection using MLX-LM parsing.

    Args:
        response: The model response text

    Returns:
        True if response contains tool calls, False otherwise
    """
    tool_calls = parse_tool_calls(response)
    return len(tool_calls) > 0


def get_current_time() -> float:
    """Get current timestamp. Separate function for easier testing."""
    return time.time()


async def _model_unloader_background_task() -> None:
    """Background task that handles model unloading based on idle timeout."""
    logger.info("Model unloader background task started")

    while True:
        try:
            # Wait for either an unload request or shutdown
            done, pending = await asyncio.wait(
                [
                    asyncio.create_task(_unload_requested.wait()),
                    asyncio.create_task(_shutdown_requested.wait()),
                ],
                return_when=asyncio.FIRST_COMPLETED,
            )

            # Cancel any pending tasks
            for task in pending:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

            # Check what completed
            if _shutdown_requested.is_set():
                logger.info("Model unloader background task shutting down")
                break

            if _unload_requested.is_set():
                # Clear the event for next time
                _unload_requested.clear()

                # Wait for the idle timeout period
                try:
                    await asyncio.wait_for(
                        _shutdown_requested.wait(), timeout=MODEL_IDLE_TIMEOUT
                    )
                    # If we get here, shutdown was requested
                    logger.info(
                        "Model unloader background task shutting down during timeout"
                    )
                    break
                except asyncio.TimeoutError:
                    # Timeout expired, check if we should unload
                    with _lock:
                        if (
                            get_current_time() - _last_used_time >= MODEL_IDLE_TIMEOUT
                            and _loaded_model is not None
                        ):
                            logger.info(
                                f"Unloading model '{_model_name}' due to inactivity"
                            )
                            _unload_model()

        except Exception as e:
            logger.error(f"Error in model unloader background task: {e}")
            await asyncio.sleep(1)  # Brief pause before retrying


def _unload_model():
    """Internal function to unload the current model and reset state with proper memory cleanup."""
    global _loaded_model, _model_name

    if _loaded_model is None:
        return  # Nothing to unload

    model_name = _model_name
    logger.info(f"Unloading model '{model_name}' and freeing memory")

    # Clear the model references first
    _loaded_model = None
    _model_name = None

    # Force garbage collection to release Python objects
    gc.collect()

    # Clear MLX memory cache/buffers using the new API
    try:
        # Use the new MLX API (replaces deprecated mx.metal.clear_cache)
        mx.clear_cache()
        logger.debug("Cleared MLX memory cache")

    except Exception as e:
        logger.warning(f"Could not clear MLX cache: {e}")

    # Multiple rounds of garbage collection for stubborn references
    for i in range(3):
        collected = gc.collect()
        if collected > 0:
            logger.debug(f"GC round {i + 1}: collected {collected} objects")

    # Force memory release by explicitly deleting any lingering references
    try:
        # Get all local variables that might hold model references
        import sys

        frame = sys._getframe()
        while frame:
            if frame.f_locals:
                for var_name, var_value in list(frame.f_locals.items()):
                    if (
                        hasattr(var_value, "__class__")
                        and "mlx" in str(type(var_value)).lower()
                    ):
                        try:
                            del frame.f_locals[var_name]
                        except Exception:
                            pass
            frame = frame.f_back
    except Exception:
        pass  # Best effort cleanup

    # Final garbage collection
    gc.collect()

    logger.info(f"Model '{model_name}' unloaded and memory freed")


def _schedule_unload():
    """Schedule model unloading after idle timeout."""
    # Simply set the event to trigger the background task
    if not _shutdown_requested.is_set():
        _unload_requested.set()


async def start_model_unloader():
    """Start the model unloader background task."""
    global _model_unloader_task

    if _model_unloader_task is None or _model_unloader_task.done():
        _model_unloader_task = asyncio.create_task(_model_unloader_background_task())
        logger.info("Model unloader background task created")


async def stop_model_unloader():
    """Stop the model unloader background task."""
    global _model_unloader_task

    _shutdown_requested.set()

    if _model_unloader_task and not _model_unloader_task.done():
        try:
            await asyncio.wait_for(_model_unloader_task, timeout=5.0)
        except asyncio.TimeoutError:
            logger.warning("Model unloader task did not stop gracefully, cancelling")
            _model_unloader_task.cancel()
            try:
                await _model_unloader_task
            except asyncio.CancelledError:
                pass

    logger.info("Model unloader background task stopped")


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

    # Handle temperature parameter
    temperature = kwargs.get("temperature")
    if temperature is not None:
        generation_kwargs["temp"] = temperature
        logger.debug(f"Using temperature: {temperature}")

    # Note: stop_strings handling moved to manual detection in generation functions
    # since some MLX versions don't support stop_strings parameter

    return generation_kwargs


def _get_stop_sequences(
    model_name: str, stop_param: Optional[Union[str, List[str]]] = None
) -> List[str]:
    """
    Get stop sequences from request parameter or model-specific defaults.

    Args:
        model_name: Name of the model
        stop_param: Stop parameter from request (overrides model defaults)

    Returns:
        List of stop sequences
    """
    stop_sequences = []

    if stop_param is not None:
        # Request provided stop sequences - use them and override defaults
        if isinstance(stop_param, str):
            stop_sequences = [stop_param]
            logger.info(f"Using stop string from request: [{stop_param}]")
        elif isinstance(stop_param, list):
            stop_sequences = stop_param
            logger.info(f"Using stop strings from request: {stop_param}")
    else:
        # Use model-specific default stop sequences
        model_name_lower = model_name.lower()

        if "qwen" in model_name_lower:
            stop_sequences = ["✿RESULT✿:", "✿RETURN✿:", "<|im_end|>"]
            logger.debug(
                f"Using Qwen stop sequences for '{model_name}': {stop_sequences}"
            )
        elif "llama" in model_name_lower:
            stop_sequences = ["<|eot_id|>"]
            logger.debug(
                f"Using Llama stop sequences for '{model_name}': {stop_sequences}"
            )
        elif "mistral" in model_name_lower or "devstral" in model_name_lower:
            # Modern Mistral models with MLX-LM use clean tool calling format
            # No special stop sequences needed - they naturally complete tool calls
            stop_sequences = []
            logger.debug(
                f"Using no stop sequences for modern Mistral model '{model_name}' (clean tool calling format)"
            )
        else:
            logger.debug(
                f"No default stop sequences configured for model '{model_name}'"
            )

    return stop_sequences


def _contains_tool_calls(
    response: str, model_name: str, tools: Optional[List[Tool]] = None
) -> bool:
    """
    Check if the response contains tool calls by parsing it.

    Args:
        response: The generated response text
        model_name: Name of the model for parser selection
        tools: Available tools

    Returns:
        True if response contains tool calls, False otherwise
    """
    if not tools:
        logger.debug("No tools provided, cannot detect tool calls")
        return False

    try:
        tool_calls = parse_tool_calls(response)

        logger.debug(
            f"Tool call detection - Model: {model_name}, Using native MLX-LM parsing"
        )
        logger.debug(f"Tool call detection - Response length: {len(response)}")
        logger.debug(f"Tool call detection - Found {len(tool_calls)} tool calls")

        if tool_calls:
            logger.debug(f"Tool call detection - Tool calls found: {tool_calls}")

        return len(tool_calls) > 0
    except Exception as e:
        logger.debug(f"Error checking for tool calls: {e}")
        return False


def _check_stop_sequences(text: str, stop_sequences: List[str]) -> Optional[str]:
    """
    Check if any stop sequence appears in the text.

    Args:
        text: Text to check
        stop_sequences: List of stop sequences to look for

    Returns:
        The stop sequence that was found, or None if none found
    """
    for stop_seq in stop_sequences:
        if stop_seq in text:
            return stop_seq
    return None


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
    model_name: str,
    messages: List,
    tools: Optional[List[Tool]] = None,
    stop: Optional[Union[str, List[str]]] = None,
    **kwargs,
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

    # Format the prompt with tools using native MLX-LM
    try:
        logger.debug("Formatting messages using native MLX-LM apply_chat_template")
        openai_tools = tools_to_openai_format(tools)
        messages_dict = _convert_messages_to_dicts(messages)

        # Get tokenized version for generation
        prompt = tokenizer.apply_chat_template(
            messages_dict, tools=openai_tools, add_generation_prompt=True
        )
        logger.debug(f"Formatted prompt length: {len(prompt)} tokens")

        # Get text version for debugging by decoding the tokens
        try:
            formatted_prompt = tokenizer._tokenizer.decode(
                prompt, special_token_policy=SpecialTokenPolicy.KEEP
            )
            logger.debug(f"Formatted prompt text: {formatted_prompt}")
        except Exception as decode_error:
            logger.warning(f"Could not decode prompt for debugging: {decode_error}")

    except Exception as e:
        logger.error(f"Failed to format messages: {e}", exc_info=True)
        raise

    # Setup generation parameters (without stop_strings)
    generation_kwargs = _setup_generation_kwargs(model_name, **kwargs)
    logger.info(f"Generation parameters: {generation_kwargs}")

    # Note: Stop sequences will be handled after generation for non-streaming

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
        logger.debug("Parsing tool calls from response using native MLX-LM parsing")
        tool_calls = parse_tool_calls(response)

        # For compatibility, we return the original response as content
        # since our new parser doesn't separate content from tool calls
        content = response

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
    model_name: str,
    messages: list,
    tools: Optional[List[Tool]] = None,
    stop: Optional[Union[str, List[str]]] = None,
    **kwargs,
):
    """
    Generate streaming response with tool calling support.

    Args:
        model_name: Name of the model to use
        messages: List of conversation messages
        tools: Optional list of available tools
        stop: Stop words/sequences to terminate generation
        **kwargs: Additional generation parameters

    Yields:
        Tuple of (text_chunk, finish_reason) where finish_reason is None for intermediate chunks
        and a string ("stop", "tool_calls", "length") for the final chunk

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

    # Format messages using native MLX-LM
    try:
        logger.debug("Formatting messages using native MLX-LM apply_chat_template")
        openai_tools = tools_to_openai_format(tools)
        messages_dict = _convert_messages_to_dicts(messages)

        # Get tokenized version for generation
        prompt = tokenizer.apply_chat_template(
            messages_dict, tools=openai_tools, add_generation_prompt=True
        )
        logger.debug(f"Formatted prompt length: {len(prompt)} tokens")

        # Get text version for debugging by decoding the tokens
        try:
            formatted_prompt = tokenizer._tokenizer.decode(
                prompt, special_token_policy=SpecialTokenPolicy.KEEP
            )
            logger.debug(f"Formatted prompt text: {formatted_prompt}")
        except Exception as decode_error:
            logger.warning(f"Could not decode prompt for debugging: {decode_error}")

    except Exception as e:
        logger.error(f"Failed to format messages: {e}", exc_info=True)
        raise

    # Setup generation parameters (without stop_strings)
    generation_kwargs = _setup_generation_kwargs(model_name, **kwargs)
    logger.info(f"Streaming generation parameters: {generation_kwargs}")

    # Get stop sequences for manual detection
    stop_sequences = _get_stop_sequences(model_name, stop)

    try:
        logger.debug("Starting streaming text generation with stream_generate")

        # Use stream_generate to yield chunks as they come
        full_response = ""
        last_chunk = None
        for chunk in stream_generate(model, tokenizer, prompt, **generation_kwargs):
            # Extract text from GenerationResponse object
            chunk_text = chunk.text if hasattr(chunk, "text") else str(chunk)
            full_response += chunk_text
            last_chunk = chunk

            # Check for stop sequences after each chunk
            if stop_sequences:
                found_stop = _check_stop_sequences(full_response, stop_sequences)
                if found_stop:
                    # Stop sequence found - determine finish reason
                    logger.debug(f"Stop sequence '{found_stop}' found in response")
                    logger.debug(f"Full response so far: {repr(full_response)}")

                    # First check if the response contains actual tool calls
                    if _contains_tool_calls(full_response, model_name, tools):
                        finish_reason = "tool_calls"
                        logger.info(
                            f"Generation stopped due to tool calls detected in response (stop sequence: {found_stop})"
                        )
                    # Then check if stop sequence itself is tool-related
                    elif any(
                        tool_indicator in found_stop.lower()
                        for tool_indicator in [
                            "tool",
                            "function",
                            "[/tool",
                            "tool_call",
                            "b_inst",
                        ]
                    ):
                        finish_reason = "tool_calls"
                        logger.info(
                            f"Generation stopped due to tool-related stop sequence: {found_stop}"
                        )
                    else:
                        finish_reason = "stop"
                        logger.info(
                            f"Generation stopped due to stop sequence: {found_stop}"
                        )

                    # Yield the chunk before the stop sequence
                    if chunk_text:
                        yield chunk_text, None

                    # Yield final indication with finish reason
                    yield "", finish_reason
                    return

            yield chunk_text, None  # None indicates this is not the final chunk

        # Determine finish reason based on the final state
        finish_reason = "stop"  # Default finish reason

        # Check if we have information from the MLX response about why it stopped
        if last_chunk and hasattr(last_chunk, "finish_reason"):
            if last_chunk.finish_reason == "length":
                finish_reason = "length"

        # Check if the final response contains tool calls (even if no stop sequence was hit)
        if _contains_tool_calls(full_response, model_name, tools):
            finish_reason = "tool_calls"
            logger.info(
                "Final response contains tool calls, setting finish_reason to tool_calls"
            )

        logger.info(
            f"Streaming generation completed - total length: {len(full_response)} characters, finish_reason: {finish_reason}"
        )
        logger.debug(f"Full streamed native response: {full_response}")

        # Yield final indication with finish reason
        yield "", finish_reason

    except Exception as e:
        logger.error(f"Streaming text generation failed: {e}", exc_info=True)
        raise
