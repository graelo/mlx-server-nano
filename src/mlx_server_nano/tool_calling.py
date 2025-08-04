"""
Tool calling support for MLX Server Nano

Provides model-specific tool call parsing and formatting for different language models.
Supports both Devstral and Qwen3 model formats with extensible parser architecture.

Features:
- Model-specific tool call formatting
- Tool call parsing from model responses
- Extensible parser architecture for new models
- JSON validation and error handling
"""

import json
import logging
import re
import uuid
from typing import Optional

from .schemas import Tool, ToolCall

# Set up logging
logger = logging.getLogger(__name__)


class ToolCallParser:
    """Base class for model-specific tool call parsing."""

    def format_tools_for_prompt(self, tools: list[Tool]) -> str:
        """
        Format tools for inclusion in the prompt.

        Args:
            tools: List of available tools

        Returns:
            Formatted string to include in model prompt
        """
        raise NotImplementedError

    def parse_tool_calls(self, response: str) -> tuple[Optional[str], list[ToolCall]]:
        """
        Parse tool calls from model response.

        Args:
            response: Raw model response text

        Returns:
            Tuple of (content, tool_calls) where content is the cleaned text
            and tool_calls is a list of parsed tool calls
        """
        raise NotImplementedError


class DevstralToolCallParser(ToolCallParser):
    """Tool call parser for Devstral models using [AVAILABLE_TOOLS] and [TOOL_CALLS] format."""

    def format_tools_for_prompt(self, tools: list[Tool]) -> str:
        """Format tools for Devstral's [AVAILABLE_TOOLS] format."""
        tool_specs = []
        for tool in tools:
            tool_spec = {
                "name": tool.function.name,
                "description": tool.function.description,
                "parameters": tool.function.parameters,
            }
            tool_specs.append(tool_spec)
        return json.dumps(tool_specs, indent=2)

    def parse_tool_calls(self, response: str) -> tuple[Optional[str], list[ToolCall]]:
        """Parse Devstral's [TOOL_CALLS][...] format."""
        tool_calls = []
        content = response

        # Look for [TOOL_CALLS][...] pattern
        tool_call_pattern = r"\[TOOL_CALLS\]\[(.*?)\]"
        matches = re.findall(tool_call_pattern, response, re.DOTALL)

        for match in matches:
            try:
                # Parse the JSON array of tool calls
                calls_data = json.loads(f"[{match}]")
                for call_data in calls_data:
                    tool_call = ToolCall(
                        id=f"call_{uuid.uuid4().hex[:8]}",
                        function={
                            "name": call_data["name"],
                            "arguments": json.dumps(call_data["arguments"]),
                        },
                    )
                    tool_calls.append(tool_call)

                # Remove tool calls from content
                content = re.sub(
                    r"\[TOOL_CALLS\].*?\]", "", content, flags=re.DOTALL
                ).strip()

            except (json.JSONDecodeError, KeyError) as e:
                logger.error(f"Failed to parse tool call: {e}")
                continue

        return content if content else None, tool_calls


class Qwen3ToolCallParser(ToolCallParser):
    """Tool call parser for Qwen3 models using ✿FUNCTION✿/✿ARGS✿ format."""

    def format_tools_for_prompt(self, tools: list[Tool]) -> str:
        """Format tools for Qwen3's system prompt format."""
        tool_descriptions = []
        for tool in tools:
            desc = f"### {tool.function.name}\n\n"
            desc += f"{tool.function.name}: {tool.function.description} "
            desc += f"Parameters: {json.dumps(tool.function.parameters)} "
            desc += "Format the arguments as a JSON object.\n\n"
            tool_descriptions.append(desc)

        function_names = [tool.function.name for tool in tools]
        tools_section = "## Tools\n\nYou have access to the following tools:\n\n"
        tools_section += "".join(tool_descriptions)
        tools_section += "## When you need to call a tool, please insert the following command in your reply:\n\n"
        tools_section += f"✿FUNCTION✿: The tool to use, should be one of [{', '.join(function_names)}]\n"
        tools_section += "✿ARGS✿: The input of the tool\n"
        tools_section += "✿RESULT✿: Tool results\n"
        tools_section += "✿RETURN✿: Reply based on tool results\n"

        return tools_section

    def parse_tool_calls(self, response: str) -> tuple[Optional[str], list[ToolCall]]:
        """Parse Qwen3's ✿FUNCTION✿/✿ARGS✿ format."""
        tool_calls = []
        content = response

        # Look for ✿FUNCTION✿ and ✿ARGS✿ patterns
        function_pattern = r"✿FUNCTION✿:\s*([^\n]+)"
        args_pattern = r"✿ARGS✿:\s*(\{[^✿]*\})"

        function_matches = re.findall(function_pattern, response)
        args_matches = re.findall(args_pattern, response, re.DOTALL)

        # Pair functions with their arguments
        for i, func_name in enumerate(function_matches):
            if i < len(args_matches):
                try:
                    # Validate JSON arguments
                    args_str = args_matches[i].strip()
                    json.loads(args_str)  # Validate JSON

                    tool_call = ToolCall(
                        id=f"call_{uuid.uuid4().hex[:8]}",
                        function={"name": func_name.strip(), "arguments": args_str},
                    )
                    tool_calls.append(tool_call)

                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse tool arguments: {e}")
                    continue

        # Remove tool call markers from content
        if tool_calls:
            content = re.sub(r"✿FUNCTION✿:[^\n]*\n", "", content)
            content = re.sub(r"✿ARGS✿:[^✿]*", "", content)
            content = content.replace("✿RESULT✿:", "").replace("✿RETURN✿:", "").strip()

        return content if content else None, tool_calls


def get_tool_parser(model_name: str) -> ToolCallParser:
    """
    Get the appropriate tool parser for a model.

    Args:
        model_name: Name of the model to get parser for

    Returns:
        ToolCallParser instance appropriate for the model
    """
    logger.debug(f"Getting tool parser for model: {model_name}")
    model_lower = model_name.lower()

    if "devstral" in model_lower:
        logger.debug("Using DevstralToolCallParser")
        return DevstralToolCallParser()
    elif "qwen3" in model_lower or "qwen" in model_lower:
        logger.debug("Using Qwen3ToolCallParser")
        return Qwen3ToolCallParser()
    else:
        logger.debug("Using default Qwen3ToolCallParser for unknown model")
        # Default to Qwen3 format for unknown models
        return Qwen3ToolCallParser()
