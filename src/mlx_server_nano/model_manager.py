import time
import threading
import logging
from typing import Optional, List, Tuple

from mlx_lm import load, generate, stream_generate
from .schemas import Tool, ToolCall
from .tool_calling import get_tool_parser
from .chat_templates import format_messages_for_model

# Set up logging
logger = logging.getLogger(__name__)

MODEL_IDLE_TIMEOUT = 300  # seconds

_loaded_model = None
_model_name = None
_last_used_time = 0
_unload_timer = None
_lock = threading.Lock()


def get_current_time():
    return time.time()


def unload_model():
    global _loaded_model, _model_name
    logger.info(f"Unloading model '{_model_name}' due to inactivity")
    print(f"[MODEL] Unloading model '{_model_name}' due to inactivity.")
    _loaded_model = None
    _model_name = None


def _schedule_unload():
    global _unload_timer
    # Cancel existing timer if it exists and is still active
    if _unload_timer and _unload_timer.is_alive():
        # For threads, we can't cancel them directly, but we can rely on the time check
        pass  # The unload_later function will check if enough time has passed

    def unload_later():
        time.sleep(MODEL_IDLE_TIMEOUT)
        with _lock:
            if get_current_time() - _last_used_time >= MODEL_IDLE_TIMEOUT:
                unload_model()

    _unload_timer = threading.Thread(target=unload_later, daemon=True)
    _unload_timer.start()


def load_model(name: str):
    """
    Load model using only Hugging Face Hub and its standard cache.

    The mlx-lm library handles downloading from HF Hub automatically
    and uses the standard HF cache location (respects HF_HOME, HUGGINGFACE_HUB_CACHE, etc.)
    """
    global _loaded_model, _model_name, _last_used_time

    logger.info(f"load_model called with name: {name}")

    with _lock:
        _last_used_time = get_current_time()

        if _model_name == name and _loaded_model:
            logger.info(f"Model '{name}' already loaded, reusing cached model")
            _schedule_unload()
            return _loaded_model

        logger.info(f"Loading model '{name}' from Hugging Face Hub...")
        print(f"[MODEL] Loading model '{name}' from Hugging Face Hub...")

        try:
            # mlx-lm will automatically download from HF Hub and cache using HF's cache system
            logger.debug(f"Calling mlx_lm.load() with model name: {name}")
            model, tokenizer = load(
                name
            )  # Note: mlx_lm.load returns (model, tokenizer)
            logger.info(
                f"Successfully loaded model '{name}' - model type: {type(model)}, tokenizer type: {type(tokenizer)}"
            )
            print(f"[MODEL] Successfully loaded model '{name}'")

        except Exception as e:
            logger.error(
                f"Failed to load model '{name}': {type(e).__name__}: {str(e)}",
                exc_info=True,
            )
            print(f"[MODEL] Failed to load model '{name}': {e}")
            raise RuntimeError(f"Failed to load model '{name}': {e}")

        _loaded_model = (model, tokenizer)  # Store as (model, tokenizer)
        _model_name = name

        _schedule_unload()
        logger.info(f"Model '{name}' loaded and cached successfully")
        return _loaded_model


def get_available_models() -> List[str]:
    """Get list of popular MLX-compatible models from Hugging Face Hub"""
    logger.debug("get_available_models called")

    # Return popular MLX-compatible models that are known to work well
    popular_models = [
        "mlx-community/Qwen2.5-7B-Instruct-4bit",
        "mlx-community/Qwen2.5-14B-Instruct-4bit",
        "mlx-community/Qwen2.5-32B-Instruct-4bit",
        "mlx-community/Qwen2.5-72B-Instruct-4bit",
        "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",
        "mlx-community/Meta-Llama-3.1-70B-Instruct-4bit",
        "mlx-community/Mistral-7B-Instruct-v0.3-4bit",
        "mlx-community/Mixtral-8x7B-Instruct-v0.1-4bit",
        "mlx-community/CodeLlama-7b-Instruct-hf-4bit",
        "mlx-community/Nous-Hermes-2-Mixtral-8x7B-DPO-4bit",
    ]

    logger.info(f"Returning {len(popular_models)} available MLX models")
    print(f"[MODEL] Available popular MLX models: {len(popular_models)} models")
    return popular_models


def generate_response_with_tools(
    model_name: str, messages: List, tools: Optional[List[Tool]] = None, **kwargs
) -> Tuple[Optional[str], List[ToolCall]]:
    """Generate response with tool calling support"""

    logger.info(
        f"generate_response_with_tools called - model: {model_name}, messages: {len(messages)}, tools: {len(tools) if tools else 0}"
    )
    logger.debug(f"Messages: {messages}")
    logger.debug(f"Tools: {tools}")
    logger.debug(f"Kwargs: {kwargs}")

    try:
        model, tokenizer = load_model(model_name)  # Now correctly ordered
        logger.debug("Model loaded successfully for generation")
    except Exception as e:
        logger.error(f"Failed to load model for generation: {e}", exc_info=True)
        raise

    try:
        # Format the prompt with tools
        logger.debug("Formatting messages for model")
        prompt = format_messages_for_model(messages, model_name, tools)
        logger.debug(f"Formatted prompt length: {len(prompt)} characters")
        logger.debug(f"Formatted prompt preview: {prompt[:200]}...")
    except Exception as e:
        logger.error(f"Failed to format messages: {e}", exc_info=True)
        raise

    # Set up generation parameters - only use parameters that are known to work
    generation_kwargs = {
        "max_tokens": kwargs.get("max_tokens", 512),
    }

    # Add stop strings for Qwen3 models
    if "qwen" in model_name.lower():
        generation_kwargs["stop_strings"] = ["✿RESULT✿:", "✿RETURN✿:", "<|im_end|>"]
        logger.debug("Added Qwen stop strings")

    logger.info(f"Generation parameters: {generation_kwargs}")

    try:
        logger.debug("Starting text generation with generate")

        # Use regular generate
        try:
            response = generate(
                model=model, tokenizer=tokenizer, prompt=prompt, **generation_kwargs
            )
            logger.debug("Used generate successfully")
        except Exception as gen_error:
            logger.warning(f"generate() failed, trying stream_generate: {gen_error}")
            # Fall back to stream_generate
            response_parts = []
            for chunk in stream_generate(model, tokenizer, prompt, **generation_kwargs):
                response_parts.append(chunk)
            response = "".join(response_parts)
            logger.debug("Fell back to stream_generate successfully")

        logger.info(
            f"Generation completed - response length: {len(response)} characters"
        )
        logger.debug(f"Generated response: {response}")
    except Exception as e:
        logger.error(f"Text generation failed: {e}", exc_info=True)
        raise

    try:
        # Parse tool calls from response
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
    Yields individual chunks as they are generated.
    """
    print(f"[DEBUG] generate_response_stream called with:")
    print(f"[DEBUG]   model_name: {model_name} (type: {type(model_name)})")
    print(f"[DEBUG]   messages: {messages} (type: {type(messages)})")
    print(f"[DEBUG]   tools: {tools} (type: {type(tools)})")
    print(f"[DEBUG]   kwargs: {kwargs}")

    print(f"[DEBUG] Starting streaming generation for model: {model_name}")
    logger.info(f"Starting streaming generation for model: {model_name}")

    # Load model if needed (load_model has its own locking)
    print("[DEBUG] Loading model...")
    model, tokenizer = load_model(model_name)

    # Update last used time and schedule unload
    with _lock:
        global _last_used_time
        _last_used_time = get_current_time()
        _schedule_unload()
        print("[DEBUG] Updated last used time...")

    # Format messages outside the lock
    print("[DEBUG] Formatting messages...")
    try:
        prompt = format_messages_for_model(messages, model_name, tools)
        print(f"[DEBUG] Formatted prompt for streaming: {prompt[:100]}...")
        logger.debug(f"Formatted prompt for streaming: {prompt}")
    except Exception as e:
        logger.error(f"Failed to format messages: {e}", exc_info=True)
        raise

    # Set up generation parameters
    print("[DEBUG] Setting up generation parameters...")
    generation_kwargs = {
        "max_tokens": kwargs.get("max_tokens", 512),
    }

    # Add stop strings for Qwen3 models
    if "qwen" in model_name.lower():
        generation_kwargs["stop_strings"] = ["✿RESULT✿:", "✿RETURN✿:", "<|im_end|>"]
        logger.debug("Added Qwen stop strings for streaming")

    print(f"[DEBUG] Streaming generation parameters: {generation_kwargs}")
    logger.info(f"Streaming generation parameters: {generation_kwargs}")

    try:
        print("[DEBUG] Starting streaming text generation with stream_generate")
        logger.debug("Starting streaming text generation with stream_generate")

        # Use stream_generate to yield chunks as they come
        full_response = ""
        print("[DEBUG] About to call stream_generate...")
        for chunk in stream_generate(model, tokenizer, prompt, **generation_kwargs):
            print(f"[DEBUG] Got chunk: {chunk}")
            # Extract text from GenerationResponse object
            chunk_text = chunk.text if hasattr(chunk, "text") else str(chunk)
            print(f"[DEBUG] Extracted text: {chunk_text}")
            full_response += chunk_text
            yield chunk_text

        print(
            f"[DEBUG] Streaming generation completed - total length: {len(full_response)} characters"
        )
        logger.info(
            f"Streaming generation completed - total length: {len(full_response)} characters"
        )
        logger.debug(f"Full streamed response: {full_response}")

        # Note: For streaming, tool calls are processed after full response is complete
        # This is a limitation of the current approach - could be improved with smarter parsing

    except Exception as e:
        print(f"[DEBUG] Streaming text generation failed: {e}")
        logger.error(f"Streaming text generation failed: {e}", exc_info=True)
        raise
