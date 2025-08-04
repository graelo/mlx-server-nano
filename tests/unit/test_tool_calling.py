"""
Unit tests for tool calling functionality.

Tests model-specific tool call parsing and formatting.
"""

import pytest

from mlx_server_nano.tool_calling import (
    ToolCallParser,
    DevstralToolCallParser,
    Qwen3ToolCallParser,
    get_tool_parser,
)
from mlx_server_nano.schemas import Tool, Function


@pytest.mark.unit
class TestToolCallParser:
    """Test cases for base ToolCallParser class."""

    def test_base_parser_not_implemented(self):
        """Test that base parser methods raise NotImplementedError."""
        parser = ToolCallParser()

        with pytest.raises(NotImplementedError):
            parser.format_tools_for_prompt([])

        with pytest.raises(NotImplementedError):
            parser.parse_tool_calls("")


@pytest.mark.unit
class TestDevstralToolCallParser:
    """Test cases for Devstral-specific tool call parsing."""

    def setup_method(self):
        """Set up test fixtures."""
        self.parser = DevstralToolCallParser()
        self.sample_tool = Tool(
            function=Function(
                name="get_weather",
                description="Get current weather for a location",
                parameters={
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "City name"}
                    },
                    "required": ["location"],
                },
            )
        )

    def test_format_tools_for_prompt_empty(self):
        """Test formatting empty tools list."""
        parser = DevstralToolCallParser()
        result = parser.format_tools_for_prompt([])
        assert result == ""

    def test_format_tools_for_prompt_single_tool(self):
        """Test formatting single tool."""
        parser = DevstralToolCallParser()
        tools = [self.sample_tool]

        result = parser.format_tools_for_prompt(tools)

        # Devstral format is JSON, not markdown headers
        assert "get_weather" in result
        assert "Get current weather for a location" in result
        assert "location" in result
        assert "City name" in result

    def test_format_tools_for_prompt_multiple_tools(self):
        """Test formatting multiple tools."""
        parser = DevstralToolCallParser()

        tool2 = Tool(
            function=Function(
                name="calculate",
                description="Perform calculation",
                parameters={
                    "type": "object",
                    "properties": {"expression": {"type": "string"}},
                },
            )
        )

        tools = [self.sample_tool, tool2]
        result = parser.format_tools_for_prompt(tools)

        assert "get_weather" in result
        assert "calculate" in result
        # DevStral format is JSON array, not markdown with headers
        assert '"name"' in result  # Check for JSON structure

    def test_parse_tool_calls_no_tools(self):
        """Test parsing response with no tool calls."""
        parser = DevstralToolCallParser()
        response = "This is just a regular response without any function calls."

        content, tool_calls = parser.parse_tool_calls(response)

        assert content == response
        assert tool_calls == []

    def test_parse_tool_calls_single_tool(self):
        """Test parsing response with single tool call."""
        parser = DevstralToolCallParser()
        response = """I'll help you get the weather.

[TOOL_CALLS][{"name": "get_weather", "arguments": {"location": "San Francisco"}}]

The function has been called."""

        content, tool_calls = parser.parse_tool_calls(response)

        assert "I'll help you get the weather." in content
        assert "The function has been called." in content
        assert "[TOOL_CALLS]" not in content

        assert len(tool_calls) == 1
        assert tool_calls[0].name == "get_weather"
        # Note: arguments are stored as JSON string in Devstral format
        assert "San Francisco" in tool_calls[0].function["arguments"]
        assert tool_calls[0].id.startswith("call_")

    def test_parse_tool_calls_multiple_tools(self):
        """Test parsing response with multiple tool calls."""
        parser = DevstralToolCallParser()
        response = """I'll help with both requests.

[TOOL_CALLS][{"name": "get_weather", "arguments": {"location": "San Francisco"}}, {"name": "calculate", "arguments": {"expression": "2 + 2"}}]

Both functions called."""

        content, tool_calls = parser.parse_tool_calls(response)

        assert len(tool_calls) == 2
        assert tool_calls[0].name == "get_weather"
        assert tool_calls[1].name == "calculate"
        assert "[TOOL_CALLS]" not in content

    def test_parse_tool_calls_invalid_json(self):
        """Test parsing response with invalid JSON in tool call."""
        parser = DevstralToolCallParser()
        response = """I'll call a function.

[TOOL_CALLS][{"name": "get_weather", "arguments": {invalid json}}]

Function called."""

        content, tool_calls = parser.parse_tool_calls(response)

        # Should still return remaining content but no tool calls due to invalid JSON
        assert "I'll call a function." in content
        assert "Function called." in content
        assert len(tool_calls) == 0

    def test_parse_tool_calls_missing_fields(self):
        """Test parsing tool call with missing required fields."""
        parser = DevstralToolCallParser()
        response = """[TOOL_CALLS][{"arguments": {"location": "SF"}}]"""

        content, tool_calls = parser.parse_tool_calls(response)

        # Should not create tool call if name is missing
        assert len(tool_calls) == 0


@pytest.mark.unit
class TestQwen3ToolCallParser:
    """Test cases for Qwen3-specific tool call parsing."""

    def test_format_tools_for_prompt_empty(self):
        """Test formatting empty tools list."""
        parser = Qwen3ToolCallParser()
        result = parser.format_tools_for_prompt([])
        assert result == ""

    def test_format_tools_for_prompt_single_tool(self):
        """Test formatting single tool."""
        parser = Qwen3ToolCallParser()

        tool = Tool(
            function=Function(
                name="get_weather",
                description="Get weather",
                parameters={
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                },
            )
        )

        result = parser.format_tools_for_prompt([tool])

        assert "✿FUNCTION✿" in result
        assert "get_weather" in result
        assert "Get weather" in result

    def test_parse_tool_calls_single_tool(self):
        """Test parsing Qwen3-style tool call."""
        parser = Qwen3ToolCallParser()
        response = """I'll get the weather for you.

✿FUNCTION✿: get_weather
✿ARGS✿: {"location": "New York"}

The weather function has been called."""

        content, tool_calls = parser.parse_tool_calls(response)

        assert "I'll get the weather for you." in content
        assert "The weather function has been called." in content
        assert "✿FUNCTION✿" not in content

        assert len(tool_calls) == 1
        assert tool_calls[0].name == "get_weather"

    def test_parse_tool_calls_multiple_tools(self):
        """Test parsing multiple Qwen3-style tool calls."""
        parser = Qwen3ToolCallParser()
        response = """✿FUNCTION✿: func1
✿ARGS✿: {"param": "value1"}

✿FUNCTION✿: func2
✿ARGS✿: {"param": "value2"}"""

        content, tool_calls = parser.parse_tool_calls(response)

        assert len(tool_calls) == 2
        assert tool_calls[0].name == "func1"
        assert tool_calls[1].name == "func2"


@pytest.mark.unit
class TestGetToolParser:
    """Test cases for get_tool_parser function."""

    def test_get_devstral_parser(self):
        """Test getting Devstral parser."""
        parser = get_tool_parser("devstral")
        assert isinstance(parser, DevstralToolCallParser)

        parser = get_tool_parser("Devstral-Model-Name")
        assert isinstance(parser, DevstralToolCallParser)

    def test_get_qwen_parser(self):
        """Test getting Qwen3 parser."""
        parser = get_tool_parser("qwen")
        assert isinstance(parser, Qwen3ToolCallParser)

        parser = get_tool_parser("Qwen3-Model-Name")
        assert isinstance(parser, Qwen3ToolCallParser)

    def test_get_default_parser(self):
        """Test getting default parser for unknown models."""
        parser = get_tool_parser("unknown-model")
        assert isinstance(parser, DevstralToolCallParser)

        parser = get_tool_parser("gpt-4")
        assert isinstance(parser, DevstralToolCallParser)

    def test_case_insensitive_matching(self):
        """Test that model name matching is case insensitive."""
        assert isinstance(get_tool_parser("DEVSTRAL"), DevstralToolCallParser)
        assert isinstance(get_tool_parser("QWEN"), Qwen3ToolCallParser)
        assert isinstance(get_tool_parser("DeVsTrAl"), DevstralToolCallParser)


@pytest.mark.unit
class TestToolCallIntegration:
    """Integration tests for tool calling functionality."""

    def test_devstral_end_to_end(self):
        """Test complete Devstral tool calling flow."""
        parser = DevstralToolCallParser()

        # Create tools
        tools = [
            Tool(
                function=Function(
                    name="get_weather",
                    description="Get weather info",
                    parameters={
                        "type": "object",
                        "properties": {"location": {"type": "string"}},
                        "required": ["location"],
                    },
                )
            )
        ]

        # Format for prompt
        prompt_addition = parser.format_tools_for_prompt(tools)
        assert "get_weather" in prompt_addition

        # Simulate model response
        model_response = """I'll check the weather for you.

[TOOL_CALLS][{"name": "get_weather", "arguments": {"location": "Boston"}}]

Weather function called successfully."""

        # Parse response
        content, tool_calls = parser.parse_tool_calls(model_response)

        assert len(tool_calls) == 1
        assert tool_calls[0].name == "get_weather"

        # Arguments are stored as JSON string, so parse them
        import json

        args = json.loads(tool_calls[0].arguments)
        assert args["location"] == "Boston"
        assert "Weather function called successfully." in content

    def test_qwen_end_to_end(self):
        """Test complete Qwen3 tool calling flow."""
        parser = Qwen3ToolCallParser()

        tools = [
            Tool(
                function=Function(
                    name="calculate",
                    description="Do math",
                    parameters={"type": "object"},
                )
            )
        ]

        # Format and parse
        prompt_addition = parser.format_tools_for_prompt(tools)
        assert "calculate" in prompt_addition

        model_response = """✿FUNCTION✿: calculate
✿ARGS✿: {"expr": "5+5"}"""

        content, tool_calls = parser.parse_tool_calls(model_response)

        assert len(tool_calls) == 1
        assert tool_calls[0].name == "calculate"
