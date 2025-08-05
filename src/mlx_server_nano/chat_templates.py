"""
Chat template formatting for different language models

Provides model-specific chat template formatting for conversation messages and tool calling.
Each model requires specific formatting to work optimally with their training templates.

Supported Models:
- Devstral: Uses [INST]/[/INST], [SYSTEM_PROMPT], [AVAILABLE_TOOLS], [TOOL_CALLS] format
- Qwen3: Uses <|im_start|>/<|im_end|> format with ✿FUNCTION✿/✿ARGS✿ tool calling
- Generic: Basic fallback format for unknown models
"""

import json
import logging
from typing import Optional

from .schemas import Message, Tool

# Set up logging
logger = logging.getLogger(__name__)


def format_messages_for_model(
    messages: list[Message],
    model_name: str,
    tools: Optional[list[Tool]] = None,
    chat_template: str = "none",
) -> str:
    """
    Format messages using model-specific chat template.

    Args:
        messages: List of conversation messages
        model_name: Name of the model to format for
        tools: Optional list of available tools
        chat_template: Chat template to use. Options: "none", "devstral", "qwen3"

    Returns:
        Formatted prompt string ready for the model

    Raises:
        Exception: If formatting fails for any reason
    """
    logger.debug(
        f"Formatting messages for model: {model_name}, message count: {len(messages)}, tools: {len(tools) if tools else 0}, chat_template: {chat_template}"
    )

    if chat_template == "none":
        logger.debug("Using raw message formatting (templates disabled)")
        return format_raw_messages(messages, tools)

    try:
        if chat_template == "devstral":
            logger.debug("Using Devstral formatting")
            return format_devstral_messages(messages, tools)
        elif chat_template == "qwen3":
            logger.debug("Using Qwen formatting")
            return format_qwen3_messages(messages, tools)
        else:
            logger.debug("Using generic formatting")
            return format_generic_messages(messages, tools)
    except Exception as e:
        logger.error(
            f"Failed to format messages for model {model_name}: {e}", exc_info=True
        )
        raise


def format_raw_messages(
    messages: list[Message], tools: Optional[list[Tool]] = None
) -> str:
    """
    Pass messages through without template formatting.
    Assumes client has already applied appropriate formatting.

    Args:
        messages: List of conversation messages
        tools: Optional list of available tools (currently ignored in raw mode)

    Returns:
        Raw concatenated message content
    """
    prompt = ""

    # Simple concatenation of message content
    for message in messages:
        if message.content:
            prompt += message.content

    return prompt


def format_devstral_messages(
    messages: list[Message], tools: Optional[list[Tool]] = None
) -> str:
    """
    Format messages for Devstral using its chat template.

    Args:
        messages: List of conversation messages
        tools: Optional list of available tools

    Returns:
        Formatted prompt string for Devstral model
    """
    prompt = ""

    # Find the last user message for tool insertion
    last_user_idx = -1
    for i, msg in enumerate(messages):
        if msg.role == "user":
            last_user_idx = i

    # Check if the last message is from user (expecting assistant response)
    expecting_response = len(messages) > 0 and messages[-1].role == "user"

    for i, message in enumerate(messages):
        if message.role == "system":
            prompt += f"[SYSTEM_PROMPT]{message.content}[/SYSTEM_PROMPT]\n"

        elif message.role == "user":
            # Add tools to the last user message if available
            if i == last_user_idx and tools:
                from .tool_calling import get_tool_parser

                parser = get_tool_parser("devstral")
                tools_str = parser.format_tools_for_prompt(tools)
                prompt += f"[AVAILABLE_TOOLS]{tools_str}[/AVAILABLE_TOOLS]\n"

            # For final user message expecting response, leave open for model to close
            if i == len(messages) - 1 and expecting_response:
                prompt += (
                    f"[INST]{message.content}"  # No [/INST] - let model generate it
                )
            else:
                prompt += f"[INST]{message.content}[/INST]\n"

        elif message.role == "assistant":
            if message.content:
                prompt += f"{message.content}\n"
            if message.tool_calls:
                # Format tool calls in Devstral format
                tool_calls_json = []
                for tc in message.tool_calls:
                    tool_calls_json.append(
                        {
                            "name": tc["function"]["name"],
                            "arguments": json.loads(tc["function"]["arguments"]),
                        }
                    )
                prompt += f"[TOOL_CALLS]{json.dumps(tool_calls_json)}]\n"

        elif message.role == "tool":
            prompt += f'[TOOL_RESULTS]{{"content": {message.content}}}[/TOOL_RESULTS]\n'

    return prompt


def format_qwen3_messages(
    messages: list[Message], tools: Optional[list[Tool]] = None
) -> str:
    """
    Format messages for Qwen3 using its chat template.

    Args:
        messages: List of conversation messages
        tools: Optional list of available tools

    Returns:
        Formatted prompt string for Qwen3 model
    """
    prompt = ""

    # Handle system message and tools
    system_msg = "You are a helpful assistant."
    for message in messages:
        if message.role == "system":
            system_msg = message.content
            break

    prompt += f"<|im_start|>system\n{system_msg}"

    if tools:
        from .tool_calling import get_tool_parser

        parser = get_tool_parser("qwen3")
        tools_str = parser.format_tools_for_prompt(tools)
        prompt += f"\n\n{tools_str}"

    prompt += "<|im_end|>\n"

    # Process conversation messages
    for message in messages:
        if message.role == "system":
            continue  # Already handled
        elif message.role == "user":
            prompt += f"<|im_start|>user\n{message.content}<|im_end|>\n"
        elif message.role == "assistant":
            prompt += "<|im_start|>assistant\n"
            if message.tool_calls:
                for tc in message.tool_calls:
                    func_name = tc["function"]["name"]
                    func_args = tc["function"]["arguments"]
                    prompt += f"✿FUNCTION✿: {func_name}\n✿ARGS✿: {func_args}\n"
            if message.content:
                prompt += f"✿RETURN✿: {message.content}"
            prompt += "<|im_end|>\n"
        elif message.role == "tool":
            prompt += f"✿RESULT✿: {message.content}\n"

    # Add generation prompt
    prompt += "<|im_start|>assistant\n"

    return prompt


def format_generic_messages(
    messages: list[Message], tools: Optional[list[Tool]] = None
) -> str:
    """
    Generic message formatting for unknown models.

    Args:
        messages: List of conversation messages
        tools: Optional list of available tools (ignored in generic format)

    Returns:
        Basic formatted prompt string
    """
    prompt = ""
    for message in messages:
        prompt += f"{message.role}: {message.content}\n"
    prompt += "assistant:"
    return prompt
