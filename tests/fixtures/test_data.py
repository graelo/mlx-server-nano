"""
Test fixtures and data for MLX Server Nano tests.

Provides common test data, mock responses, and utility functions
used across multiple test modules.
"""

import pytest


# Sample model responses for different model types
DEVSTRAL_SIMPLE_RESPONSE = "Hello! How can I help you today?"

DEVSTRAL_TOOL_RESPONSE = """I'll help you get the weather information for San Francisco.

✿FUNCTION✿
{"name": "get_weather", "arguments": {"location": "San Francisco", "units": "celsius"}}
✿FUNCTION✿

I've called the weather function to get the current conditions for San Francisco."""

QWEN_TOOL_RESPONSE = """I'll check the weather for you in San Francisco.

<tool_call>
{"name": "get_weather", "arguments": {"location": "San Francisco", "units": "celsius"}}
</tool_call>

The weather function has been invoked to retrieve the current conditions."""

MULTI_TOOL_RESPONSE = """I'll help you with both requests.

✿FUNCTION✿
{"name": "get_weather", "arguments": {"location": "San Francisco"}}
✿FUNCTION✿

✿FUNCTION✿
{"name": "calculate", "arguments": {"expression": "25 * 1.8 + 32"}}
✿FUNCTION✿

I've called both the weather and calculation functions."""


# Sample chat requests
@pytest.fixture
def basic_chat_messages():
    """Basic chat conversation messages."""
    return [
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "Hi there! How can I help you?"},
        {"role": "user", "content": "What's 2+2?"},
    ]


@pytest.fixture
def weather_tool():
    """Weather tool definition."""
    return {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather conditions for a specified location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "units": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "Temperature units",
                        "default": "celsius",
                    },
                },
                "required": ["location"],
            },
        },
    }


@pytest.fixture
def calculator_tool():
    """Calculator tool definition."""
    return {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Perform mathematical calculations",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Mathematical expression to evaluate",
                    }
                },
                "required": ["expression"],
            },
        },
    }


@pytest.fixture
def search_tool():
    """Search tool definition."""
    return {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for information",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "num_results": {
                        "type": "integer",
                        "description": "Number of results to return",
                        "default": 5,
                        "minimum": 1,
                        "maximum": 20,
                    },
                },
                "required": ["query"],
            },
        },
    }


@pytest.fixture
def multiple_tools(weather_tool, calculator_tool, search_tool):
    """Multiple tool definitions."""
    return [weather_tool, calculator_tool, search_tool]


@pytest.fixture
def chat_with_tools_request(basic_chat_messages, weather_tool):
    """Chat request with tools."""
    return {
        "model": "test-model",
        "messages": basic_chat_messages,
        "tools": [weather_tool],
        "tool_choice": "auto",
        "max_tokens": 100,
        "temperature": 0.7,
    }


@pytest.fixture
def streaming_chat_request():
    """Streaming chat request."""
    return {
        "model": "test-model",
        "messages": [{"role": "user", "content": "Tell me a short story"}],
        "stream": True,
        "max_tokens": 200,
        "temperature": 0.8,
    }


# Mock model responses
@pytest.fixture
def mock_model_responses():
    """Collection of mock model responses for different scenarios."""
    return {
        "simple": DEVSTRAL_SIMPLE_RESPONSE,
        "devstral_tool": DEVSTRAL_TOOL_RESPONSE,
        "qwen_tool": QWEN_TOOL_RESPONSE,
        "multi_tool": MULTI_TOOL_RESPONSE,
        "streaming_chunks": ["Once", " upon", " a", " time", "..."],
        "error_response": "I encountered an error processing your request.",
        "long_response": "This is a very long response. " * 100,
        "empty_response": "",
        "json_response": '{"status": "success", "data": {"result": 42}}',
    }


# Test model configurations
@pytest.fixture
def test_models():
    """Test model configurations."""
    return {
        "small_devstral": {
            "name": "graelo/Devstral-Small-2507-4bits",
            "type": "devstral",
            "supports_tools": True,
            "max_context": 8192,
        },
        "qwen_model": {
            "name": "Qwen/Qwen3-0.5B-Instruct",
            "type": "qwen",
            "supports_tools": True,
            "max_context": 32768,
        },
        "generic_model": {
            "name": "microsoft/DialoGPT-medium",
            "type": "generic",
            "supports_tools": False,
            "max_context": 1024,
        },
    }


# Error scenarios
@pytest.fixture
def error_scenarios():
    """Common error scenarios for testing."""
    return {
        "model_load_error": {
            "error": Exception("Failed to load model"),
            "expected_status": 500,
        },
        "generation_error": {
            "error": Exception("Generation failed"),
            "expected_status": 500,
        },
        "timeout_error": {
            "error": TimeoutError("Request timed out"),
            "expected_status": 504,
        },
        "validation_error": {
            "error": ValueError("Invalid input"),
            "expected_status": 422,
        },
        "memory_error": {"error": MemoryError("Out of memory"), "expected_status": 507},
    }


# API response templates
@pytest.fixture
def openai_response_template():
    """OpenAI-compatible response template."""
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "test-model",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,  # To be filled
                    "tool_calls": None,  # To be filled if needed
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


@pytest.fixture
def streaming_response_template():
    """Streaming response chunk template."""
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion.chunk",
        "created": 1234567890,
        "model": "test-model",
        "choices": [
            {
                "index": 0,
                "delta": {
                    "role": "assistant",
                    "content": None,  # To be filled
                },
                "finish_reason": None,
            }
        ],
    }


# Test utilities
@pytest.fixture
def model_name_variants():
    """Different model name variants for testing."""
    return {
        "devstral": [
            "devstral",
            "Devstral",
            "DEVSTRAL",
            "devstral-small",
            "graelo/Devstral-Small-2507-4bits",
            "Devstral-7B-v0.1",
        ],
        "qwen": [
            "qwen",
            "Qwen",
            "QWEN",
            "qwen3",
            "Qwen/Qwen3-0.5B-Instruct",
            "qwen2-7b",
        ],
        "other": ["gpt-4", "claude-3", "llama-2", "mistral-7b", "phi-3-mini"],
    }


@pytest.fixture
def performance_test_data():
    """Data for performance testing."""
    return {
        "small_request": {
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 10,
        },
        "medium_request": {
            "messages": [
                {"role": "user", "content": "What is machine learning?"},
                {"role": "assistant", "content": "Machine learning is..."},
                {"role": "user", "content": "Can you explain more?"},
            ],
            "max_tokens": 100,
        },
        "large_request": {
            "messages": [{"role": "user", "content": "x" * 1000}],
            "max_tokens": 500,
        },
        "concurrent_requests": 10,
        "timeout_seconds": 30,
    }


# Memory test helpers
@pytest.fixture
def memory_test_config():
    """Configuration for memory testing."""
    return {
        "initial_memory_mb": 50,  # Expected baseline memory
        "model_memory_mb": 200,  # Expected memory after loading model
        "memory_threshold_mb": 100,  # Threshold for memory increase
        "gc_rounds": 3,  # Number of garbage collection rounds
        "cleanup_delay_seconds": 2,  # Delay after cleanup before measuring
    }


# Validation test data
@pytest.fixture
def invalid_requests():
    """Invalid request examples for validation testing."""
    return {
        "missing_model": {"messages": [{"role": "user", "content": "test"}]},
        "empty_messages": {"model": "test-model", "messages": []},
        "invalid_role": {
            "model": "test-model",
            "messages": [{"role": "invalid", "content": "test"}],
        },
        "negative_max_tokens": {
            "model": "test-model",
            "messages": [{"role": "user", "content": "test"}],
            "max_tokens": -1,
        },
        "invalid_temperature": {
            "model": "test-model",
            "messages": [{"role": "user", "content": "test"}],
            "temperature": -1,
        },
        "malformed_tool": {
            "model": "test-model",
            "messages": [{"role": "user", "content": "test"}],
            "tools": [{"invalid": "tool"}],
        },
    }


# Regression test data
@pytest.fixture
def regression_test_cases():
    """Test cases for regression testing."""
    return [
        {
            "name": "basic_completion",
            "request": {
                "model": "test-model",
                "messages": [{"role": "user", "content": "Hello"}],
            },
            "expected_fields": ["id", "object", "created", "model", "choices"],
        },
        {
            "name": "completion_with_tools",
            "request": {
                "model": "test-model",
                "messages": [{"role": "user", "content": "What's the weather?"}],
                "tools": [
                    {
                        "type": "function",
                        "function": {"name": "get_weather", "parameters": {}},
                    }
                ],
            },
            "expected_fields": ["id", "object", "created", "model", "choices"],
        },
        {
            "name": "streaming_completion",
            "request": {
                "model": "test-model",
                "messages": [{"role": "user", "content": "Tell me a story"}],
                "stream": True,
            },
            "expected_format": "text/plain",
        },
    ]
