"""
Message Formatting and Conversion

Handles message conversion and prompt formatting for MLX-LM compatibility.
Converts between Pydantic models and dictionaries, and manages chat template application.

Features:
- Pydantic model to dictionary conversion
- Chat template application via MLX-LM tokenizers
- Message format compatibility handling
- Debug logging for formatted prompts
"""

import logging
from typing import List

from mistral_common.tokens.tokenizers.base import SpecialTokenPolicy

from ..schemas import tools_to_openai_format, Tool

# Set up logging
logger = logging.getLogger(__name__)


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


def format_messages_for_generation(messages: List, tools: List[Tool], tokenizer) -> str:
    """
    Format messages for generation using MLX-LM chat template.

    Args:
        messages: List of conversation messages
        tools: List of available tools
        tokenizer: MLX tokenizer instance

    Returns:
        Formatted prompt tokens for generation

    Raises:
        Exception: If message formatting fails
    """
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

        return prompt

    except Exception as e:
        logger.error(f"Failed to format messages: {e}", exc_info=True)
        raise
