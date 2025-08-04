"""
Unit tests for Pydantic schemas.

Tests request/response schemas for OpenAI API compatibility.
"""

import pytest
from pydantic import ValidationError

from mlx_server_nano.schemas import (
    Message,
    Function,
    Tool,
    ToolChoice,
    ChatCompletionRequest,
    ToolCall,
)


@pytest.mark.unit
class TestMessage:
    """Test cases for Message schema."""

    def test_basic_message(self):
        """Test creating a basic message."""
        message = Message(role="user", content="Hello")

        assert message.role == "user"
        assert message.content == "Hello"
        assert message.tool_calls is None
        assert message.tool_call_id is None

    def test_message_with_tool_calls(self):
        """Test message with tool calls."""
        tool_calls = [
            {
                "id": "call_123",
                "type": "function",
                "function": {"name": "test_func", "arguments": "{}"},
            }
        ]

        message = Message(
            role="assistant", content="I'll call a function", tool_calls=tool_calls
        )

        assert message.role == "assistant"
        assert message.content == "I'll call a function"
        assert message.tool_calls is not None
        assert len(message.tool_calls) == 1
        assert message.tool_calls[0]["id"] == "call_123"

    def test_message_with_tool_call_id(self):
        """Test message with tool call ID (for tool responses)."""
        message = Message(
            role="tool", content="Function result", tool_call_id="call_123"
        )

        assert message.role == "tool"
        assert message.content == "Function result"
        assert message.tool_call_id == "call_123"

    def test_message_content_optional(self):
        """Test that message content can be None."""
        message = Message(role="assistant", tool_calls=[])

        assert message.role == "assistant"
        assert message.content is None
        assert message.tool_calls == []


@pytest.mark.unit
class TestFunction:
    """Test cases for Function schema."""

    def test_basic_function(self):
        """Test creating a basic function definition."""
        function = Function(
            name="get_weather",
            description="Get weather info",
            parameters={
                "type": "object",
                "properties": {"location": {"type": "string"}},
            },
        )

        assert function.name == "get_weather"
        assert function.description == "Get weather info"
        assert function.parameters["type"] == "object"
        assert "location" in function.parameters["properties"]

    def test_function_complex_parameters(self):
        """Test function with complex parameter schema."""
        parameters = {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City and state"},
                "units": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "default": "celsius",
                },
            },
            "required": ["location"],
        }

        function = Function(
            name="get_weather", description="Get current weather", parameters=parameters
        )

        assert function.parameters["required"] == ["location"]
        assert "enum" in function.parameters["properties"]["units"]


@pytest.mark.unit
class TestTool:
    """Test cases for Tool schema."""

    def test_basic_tool(self):
        """Test creating a basic tool definition."""
        function = Function(
            name="test_func", description="Test function", parameters={"type": "object"}
        )

        tool = Tool(function=function)

        assert tool.type == "function"  # default value
        assert tool.function.name == "test_func"

    def test_tool_custom_type(self):
        """Test tool with custom type."""
        function = Function(
            name="test_func", description="Test function", parameters={"type": "object"}
        )

        tool = Tool(type="custom", function=function)

        assert tool.type == "custom"


@pytest.mark.unit
class TestToolChoice:
    """Test cases for ToolChoice schema."""

    def test_basic_tool_choice(self):
        """Test creating a basic tool choice."""
        choice = ToolChoice(type="auto")

        assert choice.type == "auto"
        assert choice.function is None

    def test_tool_choice_with_function(self):
        """Test tool choice with specific function."""
        choice = ToolChoice(type="function", function={"name": "get_weather"})

        assert choice.type == "function"
        assert choice.function is not None
        assert choice.function["name"] == "get_weather"


@pytest.mark.unit
class TestChatCompletionRequest:
    """Test cases for ChatCompletionRequest schema."""

    def test_minimal_request(self):
        """Test creating a minimal chat completion request."""
        request = ChatCompletionRequest(
            model="test-model", messages=[Message(role="user", content="Hello")]
        )

        assert request.model == "test-model"
        assert request.messages is not None
        assert len(request.messages) == 1
        assert request.messages[0].role == "user"

        # Test defaults
        assert request.max_tokens is None
        assert request.temperature is None
        assert request.stream is False
        assert request.tools is None

    def test_complete_request(self):
        """Test creating a complete chat completion request."""
        messages = [
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi there!"),
            Message(role="user", content="How are you?"),
        ]

        tools = [
            Tool(
                type="function",
                function=Function(
                    name="get_weather",
                    description="Get weather",
                    parameters={"type": "object"},
                ),
            )
        ]

        request = ChatCompletionRequest(
            model="gpt-4",
            messages=messages,
            max_tokens=100,
            temperature=0.7,
            top_p=0.9,
            stream=True,
            tools=tools,
            tool_choice="auto",
        )

        assert request.model == "gpt-4"
        assert request.messages is not None
        assert len(request.messages) == 3
        assert request.max_tokens == 100
        assert request.temperature == 0.7
        assert request.top_p == 0.9
        assert request.stream is True
        assert request.tools is not None
        assert len(request.tools) == 1
        assert request.tool_choice == "auto"

    def test_request_validation_empty_messages(self):
        """Test that empty messages list raises validation error."""
        with pytest.raises(ValidationError):
            ChatCompletionRequest(model="test-model", messages=[])

    def test_request_validation_invalid_temperature(self):
        """Test validation of temperature parameter."""
        # Valid temperature
        request = ChatCompletionRequest(
            model="test-model",
            messages=[Message(role="user", content="test")],
            temperature=0.5,
        )
        assert request.temperature == 0.5

        # Temperature too high should still be accepted (API will handle)
        request = ChatCompletionRequest(
            model="test-model",
            messages=[Message(role="user", content="test")],
            temperature=2.5,
        )
        assert request.temperature == 2.5

    def test_request_with_message_objects(self):
        """Test request with Message objects instead of dicts."""
        message = Message(role="user", content="Hello")

        # Should work with mixed types
        request = ChatCompletionRequest(
            model="test-model",
            messages=[message, Message(role="assistant", content="Hi")],
        )

        assert len(request.messages) == 2


@pytest.mark.unit
class TestToolCall:
    """Test cases for ToolCall schema."""

    def test_basic_tool_call(self):
        """Test creating a basic tool call."""
        tool_call = ToolCall(
            id="call_123",
            function={
                "name": "get_weather",
                "arguments": {"location": "San Francisco"},
            },
        )

        assert tool_call.id == "call_123"
        assert tool_call.name == "get_weather"
        assert tool_call.arguments is not None
        assert isinstance(tool_call.arguments, dict)
        assert tool_call.arguments["location"] == "San Francisco"

    def test_tool_call_with_string_arguments(self):
        """Test tool call with JSON string arguments."""
        tool_call = ToolCall(
            id="call_456",
            function={"name": "calculate", "arguments": '{"x": 5, "y": 10}'},
        )

        assert tool_call.id == "call_456"
        assert tool_call.name == "calculate"
        assert tool_call.arguments == '{"x": 5, "y": 10}'

    def test_tool_call_empty_arguments(self):
        """Test tool call with empty arguments."""
        tool_call = ToolCall(
            id="call_789", function={"name": "no_args_func", "arguments": {}}
        )

        assert tool_call.id == "call_789"
        assert tool_call.name == "no_args_func"
        assert tool_call.arguments == {}
