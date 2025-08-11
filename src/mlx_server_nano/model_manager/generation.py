"""
Text Generation

Handles both streaming and non-streaming text generation using MLX framework.
Provides tool calling integration and model-specific generation parameter handling.

Features:
- Streaming and non-streaming generation
- Tool call detection and parsing integration
- Generation parameter setup and validation
- Fallback generation methods for compatibility
- MLX-LM handles stop sequences automatically
"""

import logging
from typing import List, Optional, Tuple, Union

from mlx_lm.generate import generate, stream_generate

from ..config import config
from ..schemas import Tool, ToolCall
from .cache import load_model, update_last_used_time
from .message_formatting import format_messages_for_generation
from .tool_calling import parse_tool_calls, _contains_tool_calls

# Set up logging
logger = logging.getLogger(__name__)


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

    # Handle temperature parameter (temporarily disabled due to MLX-LM compatibility)
    # temperature = kwargs.get("temperature")
    # if temperature is not None:
    #     generation_kwargs["temperature"] = temperature
    #     logger.debug(f"Using temperature: {temperature}")

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
        prompt = format_messages_for_generation(messages, tools or [], tokenizer)
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
    update_last_used_time()
    from .background_tasks import _schedule_unload

    _schedule_unload()

    # Format messages using native MLX-LM
    try:
        prompt = format_messages_for_generation(messages, tools or [], tokenizer)
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
        last_chunk = None
        for chunk in stream_generate(model, tokenizer, prompt, **generation_kwargs):
            # Extract text from GenerationResponse object
            chunk_text = chunk.text if hasattr(chunk, "text") else str(chunk)
            full_response += chunk_text
            last_chunk = chunk

            yield chunk_text, None  # None indicates this is not the final chunk

        # Determine finish reason based on the final state
        finish_reason = "stop"  # Default finish reason

        # Check if we have information from the MLX response about why it stopped
        if last_chunk and hasattr(last_chunk, "finish_reason"):
            if last_chunk.finish_reason == "length":
                finish_reason = "length"

        # Check if the final response contains tool calls
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
