"""
Stop Sequence Management

Handles stop sequence configuration and detection for different model types.
Provides model-specific default stop sequences and manual stop detection.

Features:
- Model-specific stop sequence defaults
- Custom stop sequence override support
- Stop sequence detection in generated text
- Model type recognition and configuration
"""

import logging
from typing import List, Optional, Union

# Set up logging
logger = logging.getLogger(__name__)


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
