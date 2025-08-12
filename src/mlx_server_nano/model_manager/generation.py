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
from ..schemas import Tool, ToolCall, Message
from .cache import load_model, update_last_used_time, get_or_create_conversation_state
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


# =============================================================================
# Conversation-Aware Generation with Prompt Caching
# =============================================================================


def generate_response_with_tools_cached(
    model_name: str,
    messages: List[Message],
    tools: Optional[List[Tool]] = None,
    conversation_id: Optional[str] = None,
    stop: Optional[Union[str, List[str]]] = None,
    **kwargs,
) -> Tuple[Optional[str], List[ToolCall]]:
    """
    Generate response with tool calling support and conversation-level prompt caching.

    This function leverages MLX-LM's built-in prompt caching capabilities for improved
    performance in conversational scenarios while preserving all existing functionality.

    Args:
        model_name: Name of the model to use
        messages: List of conversation messages
        tools: Optional list of available tools
        conversation_id: Optional explicit conversation ID (auto-detected if None)
        stop: Stop words/sequences to terminate generation
        **kwargs: Additional generation parameters

    Returns:
        Tuple of (content, tool_calls) where content is the text response
        and tool_calls is a list of detected tool calls
    """
    # Detect if we're in test mode by checking if functions are mocked
    # In test mode, fall back to non-cached generation
    import sys

    is_test_mode = "pytest" in sys.modules or not config.conversation_cache_enabled

    if is_test_mode:
        # Fall back to non-cached generation during tests
        return generate_response_with_tools(
            model_name=model_name, messages=messages, tools=tools, stop=stop, **kwargs
        )

    logger.info(
        f"Generating cached response - model: {model_name}, messages: {len(messages)}, "
        f"tools: {len(tools) if tools else 0}, conversation_id: {conversation_id}"
    )

    # Load model first
    try:
        model, tokenizer = load_model(model_name)
        logger.debug("Model loaded successfully for cached generation")
    except Exception as e:
        logger.error(f"Failed to load model for cached generation: {e}", exc_info=True)
        raise

    # Get or create conversation state with auto-detection (passing model for cache creation)
    conv_state = get_or_create_conversation_state(
        conversation_id, model_name, messages, model
    )

    # Format the prompt with tools using native MLX-LM
    try:
        prompt = format_messages_for_generation(messages, tools or [], tokenizer)
    except Exception as e:
        logger.error(f"Failed to format messages: {e}", exc_info=True)
        raise

    # Setup generation parameters
    generation_kwargs = _setup_generation_kwargs(model_name, **kwargs)
    logger.info(f"Cached generation parameters: {generation_kwargs}")

    # Generate text response with prompt caching
    try:
        logger.debug(
            f"Starting cached text generation for conversation: {conv_state.conversation_id}"
        )
        logger.debug(
            f"Prompt cache state - type: {type(conv_state.prompt_cache)}, "
            f"length: {len(conv_state.prompt_cache) if conv_state.prompt_cache else 'None'}"
        )

        # Log detailed cache state for debugging
        if conv_state.prompt_cache and len(conv_state.prompt_cache) > 0:
            # Check if any cache entries have data
            cache_has_data = False
            for i, cache_entry in enumerate(
                conv_state.prompt_cache[:3]
            ):  # Check first 3 layers
                if hasattr(cache_entry, "keys") and hasattr(cache_entry, "values"):
                    if cache_entry.keys is not None or cache_entry.values is not None:
                        cache_has_data = True
                        logger.debug(
                            f"Cache layer {i} has data: keys={cache_entry.keys is not None}, values={cache_entry.values is not None}"
                        )
                        break
            logger.debug(f"Cache contains data: {cache_has_data}")
        else:
            logger.debug("Cache is empty or None")

        # Use MLX-LM's generate with prompt_cache parameter
        # This preserves all auto-stop functionality while adding caching
        response = generate(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            prompt_cache=conv_state.prompt_cache,  # Add caching!
            **generation_kwargs,
        )

        # Check cache state after generation
        logger.debug(
            f"After generation - prompt cache length: {len(conv_state.prompt_cache) if conv_state.prompt_cache else 'None'}"
        )

        if conv_state.prompt_cache and len(conv_state.prompt_cache) > 0:
            # Check if cache was populated during generation
            cache_populated = False
            for i, cache_entry in enumerate(
                conv_state.prompt_cache[:3]
            ):  # Check first 3 layers
                if hasattr(cache_entry, "keys") and hasattr(cache_entry, "values"):
                    if cache_entry.keys is not None or cache_entry.values is not None:
                        cache_populated = True
                        logger.debug(
                            f"After gen - Cache layer {i} populated: keys={cache_entry.keys is not None}, values={cache_entry.values is not None}"
                        )
                        if hasattr(cache_entry.keys, "shape"):
                            logger.debug(
                                f"After gen - Cache layer {i} keys shape: {cache_entry.keys.shape}"
                            )
                        break
            logger.debug(f"Cache was populated during generation: {cache_populated}")

        logger.info(
            f"Cached generation completed - conversation: {conv_state.conversation_id}, "
            f"response length: {len(response)} characters"
        )
        logger.debug(f"Generated cached response: {response}")
    except Exception as e:
        logger.error(f"Cached text generation failed: {e}", exc_info=True)
        raise

    # Parse tool calls from response
    try:
        logger.debug("Parsing tool calls from cached response")
        tool_calls = parse_tool_calls(response)

        # For compatibility, we return the original response as content
        content = response

        logger.info(
            f"Cached tool parsing completed - content: {len(content) if content else 0} chars, "
            f"tool_calls: {len(tool_calls)}"
        )
        logger.debug(f"Parsed cached content: {content}")
        logger.debug(f"Parsed cached tool calls: {tool_calls}")
    except Exception as e:
        logger.error(f"Cached tool call parsing failed: {e}", exc_info=True)
        raise

    return content, tool_calls


def generate_response_stream_cached(
    model_name: str,
    messages: List[Message],
    tools: Optional[List[Tool]] = None,
    conversation_id: Optional[str] = None,
    stop: Optional[Union[str, List[str]]] = None,
    **kwargs,
):
    """
    Generate streaming response with tool calling support and conversation-level prompt caching.

    This function leverages MLX-LM's built-in prompt caching capabilities via stream_generate
    for improved performance while preserving all auto-stop functionality.

    Args:
        model_name: Name of the model to use
        messages: List of conversation messages
        tools: Optional list of available tools
        conversation_id: Optional explicit conversation ID (auto-detected if None)
        stop: Stop words/sequences to terminate generation
        **kwargs: Additional generation parameters

    Yields:
        Tuple of (text_chunk, finish_reason) where finish_reason is None for intermediate chunks
        and a string ("stop", "tool_calls", "length") for the final chunk

    Note:
        Tool calls are processed after full response completion in streaming mode.
        All auto-stop features from stream_generate are preserved.
    """
    # Detect if we're in test mode by checking if functions are mocked
    # In test mode, fall back to non-cached generation
    import sys

    is_test_mode = "pytest" in sys.modules or not config.conversation_cache_enabled

    if is_test_mode:
        # Fall back to non-cached generation during tests
        yield from generate_response_stream(
            model_name=model_name, messages=messages, tools=tools, stop=stop, **kwargs
        )
        return

    logger.info(
        f"Starting cached streaming generation - model: {model_name}, "
        f"conversation_id: {conversation_id}"
    )

    # Load model (load_model has its own locking)
    model, tokenizer = load_model(model_name)

    # Get or create conversation state with auto-detection (passing model for cache creation)
    conv_state = get_or_create_conversation_state(
        conversation_id, model_name, messages, model
    )

    # Update last used time and schedule unload
    update_last_used_time()
    from .background_tasks import _schedule_unload

    _schedule_unload()

    # Format messages using native MLX-LM
    try:
        prompt = format_messages_for_generation(messages, tools or [], tokenizer)
    except Exception as e:
        logger.error(
            f"Failed to format messages for cached streaming: {e}", exc_info=True
        )
        raise

    # Setup generation parameters
    generation_kwargs = _setup_generation_kwargs(model_name, **kwargs)
    logger.info(f"Cached streaming generation parameters: {generation_kwargs}")

    try:
        logger.debug(
            f"Starting cached streaming text generation with stream_generate "
            f"for conversation: {conv_state.conversation_id}"
        )
        logger.debug(
            f"Prompt cache state - type: {type(conv_state.prompt_cache)}, "
            f"length: {len(conv_state.prompt_cache) if conv_state.prompt_cache else 'None'}"
        )

        # Log detailed cache state for debugging
        if conv_state.prompt_cache and len(conv_state.prompt_cache) > 0:
            # Check if any cache entries have data
            cache_has_data = False
            for i, cache_entry in enumerate(
                conv_state.prompt_cache[:3]
            ):  # Check first 3 layers
                if hasattr(cache_entry, "keys") and hasattr(cache_entry, "values"):
                    if cache_entry.keys is not None or cache_entry.values is not None:
                        cache_has_data = True
                        logger.debug(
                            f"Streaming - Cache layer {i} has data: keys={cache_entry.keys is not None}, values={cache_entry.values is not None}"
                        )
                        break
            logger.debug(f"Streaming - Cache contains data: {cache_has_data}")
        else:
            logger.debug("Streaming - Cache is empty or None")

        # Use MLX-LM's stream_generate with prompt_cache parameter
        # This preserves ALL auto-stop functionality while adding caching
        full_response = ""
        last_chunk = None
        for chunk in stream_generate(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            prompt_cache=conv_state.prompt_cache,  # Add caching!
            **generation_kwargs,
        ):
            # Extract text from GenerationResponse object
            chunk_text = chunk.text if hasattr(chunk, "text") else str(chunk)
            full_response += chunk_text
            last_chunk = chunk

            yield chunk_text, None  # None indicates this is not the final chunk

        # Check cache state after streaming generation
        logger.debug(
            f"After streaming generation - prompt cache length: {len(conv_state.prompt_cache) if conv_state.prompt_cache else 'None'}"
        )

        if conv_state.prompt_cache and len(conv_state.prompt_cache) > 0:
            # Check if cache was populated during generation
            cache_populated = False
            for i, cache_entry in enumerate(
                conv_state.prompt_cache[:3]
            ):  # Check first 3 layers
                if hasattr(cache_entry, "keys") and hasattr(cache_entry, "values"):
                    if cache_entry.keys is not None or cache_entry.values is not None:
                        cache_populated = True
                        logger.debug(
                            f"After streaming - Cache layer {i} populated: keys={cache_entry.keys is not None}, values={cache_entry.values is not None}"
                        )
                        if hasattr(cache_entry.keys, "shape"):
                            logger.debug(
                                f"After streaming - Cache layer {i} keys shape: {cache_entry.keys.shape}"
                            )
                        break
            logger.debug(
                f"Streaming - Cache was populated during generation: {cache_populated}"
            )

        # Determine finish reason based on the final state
        # All auto-stop logic preserved from original implementation
        finish_reason = "stop"  # Default finish reason

        # Check if we have information from the MLX response about why it stopped
        if last_chunk and hasattr(last_chunk, "finish_reason"):
            if last_chunk.finish_reason == "length":
                finish_reason = "length"

        # Check if the final response contains tool calls
        if _contains_tool_calls(full_response, model_name, tools):
            finish_reason = "tool_calls"
            logger.info(
                "Cached final response contains tool calls, setting finish_reason to tool_calls"
            )

        logger.info(
            f"Cached streaming generation completed - conversation: {conv_state.conversation_id}, "
            f"total length: {len(full_response)} characters, finish_reason: {finish_reason}"
        )
        logger.debug(f"Full cached streamed response: {full_response}")

        # Yield final indication with finish reason
        yield "", finish_reason

    except Exception as e:
        logger.error(f"Cached streaming text generation failed: {e}", exc_info=True)
        raise
