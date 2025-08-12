"""
Tool Calling Support

Handles tool call parsing and detection from model responses using MLX-LM compatible formats.
Supports multiple tool calling formats including Mistral and other common patterns.

Features:
- Tool call parsing from model responses
- Multiple format support (Mistral, etc.)
- Tool call detection and validation
- JSON argument parsing with error handling
"""

import json
import logging
import random
import re
import string
from typing import Optional

from ..schemas import Tool, ToolCall

# Set up logging
logger = logging.getLogger(__name__)


def parse_tool_calls(response: str) -> list[ToolCall]:
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
    pattern = r"\[TOOL_CALLS\]([^\[]+)\[ARGS\]"

    for match in re.finditer(pattern, response):
        function_name = match.group(1).strip()
        start_pos = match.end()

        # Find the opening brace
        brace_start = response.find("{", start_pos)
        if brace_start == -1:
            continue

        # Extract JSON by counting braces, respecting quotes
        brace_count = 0
        in_quotes = False
        escaped = False

        for i, char in enumerate(response[brace_start:], brace_start):
            if escaped:
                escaped = False
                continue

            if char == "\\" and in_quotes:
                escaped = True
            elif char == '"' and not escaped:
                in_quotes = not in_quotes
            elif not in_quotes:
                if char == "{":
                    brace_count += 1
                elif char == "}":
                    brace_count -= 1
                    if brace_count == 0:
                        json_str = response[brace_start : i + 1]
                        break
        else:
            continue  # No matching closing brace found

        try:
            arguments = json.loads(json_str)
            tool_call = ToolCall(
                id="".join(random.choices(string.ascii_letters + string.digits, k=9)),
                type="function",
                function={
                    "name": function_name,
                    "arguments": json.dumps(arguments),
                },
            )
            tool_calls.append(tool_call)
        except json.JSONDecodeError as e:
            logger.warning(
                f"Failed to parse tool call arguments for {function_name}: {e}"
            )

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


def _contains_tool_calls(
    response: str, model_name: str, tools: Optional[list[Tool]] = None
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
