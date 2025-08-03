"""
Global test configuration and fixtures.

Provides common test setup, configuration, and fixtures for use across all test modules.
"""

import asyncio
import os
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from mlx_server_nano.app import app
from mlx_server_nano.config import ServerConfig
import mlx_server_nano.model_manager as model_manager


# Pytest configuration
pytest_plugins = ["pytest_asyncio"]


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    policy = asyncio.get_event_loop_policy()
    loop = policy.new_event_loop()
    asyncio.set_event_loop(loop)
    yield loop
    loop.close()


@pytest.fixture
def test_config():
    """Provide a test configuration with reduced timeouts."""
    return ServerConfig(
        host="127.0.0.1",
        port=8000,
        model_idle_timeout=2,  # Short timeout for faster tests
        default_max_tokens=100,
        default_temperature=0.1,
        log_level="DEBUG",
    )


@pytest.fixture
def test_env_vars(test_config):
    """Set up test environment variables."""
    original_values = {}
    test_vars = {
        "MLX_MODEL_IDLE_TIMEOUT": str(test_config.model_idle_timeout),
        "MLX_DEFAULT_MAX_TOKENS": str(test_config.default_max_tokens),
        "MLX_DEFAULT_TEMPERATURE": str(test_config.default_temperature),
        "MLX_LOG_LEVEL": test_config.log_level,
    }

    # Store original values and set test values
    for key, value in test_vars.items():
        original_values[key] = os.environ.get(key)
        os.environ[key] = value

    yield test_vars

    # Restore original values
    for key, original_value in original_values.items():
        if original_value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = original_value


@pytest.fixture
def client():
    """Provide a FastAPI test client."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
async def clean_model_manager():
    """Clean model manager state before and after tests with proper async cleanup."""
    # Save original state
    original_model = model_manager._loaded_model
    original_name = model_manager._model_name
    original_last_used = model_manager._last_used_time
    original_events = (
        model_manager._unload_requested,
        model_manager._shutdown_requested,
    )
    original_task = model_manager._model_unloader_task

    # Create fresh events for this test
    model_manager._unload_requested = asyncio.Event()
    model_manager._shutdown_requested = asyncio.Event()
    model_manager._model_unloader_task = None

    # Clean model state
    model_manager._loaded_model = None
    model_manager._model_name = None
    model_manager._last_used_time = 0

    yield

    # Cleanup and restore
    if (
        model_manager._model_unloader_task
        and not model_manager._model_unloader_task.done()
    ):
        model_manager._shutdown_requested.set()
        try:
            await asyncio.wait_for(model_manager._model_unloader_task, timeout=1.0)
        except asyncio.TimeoutError:
            if model_manager._model_unloader_task is not None:
                model_manager._model_unloader_task.cancel()
                try:
                    await model_manager._model_unloader_task  # pyright: ignore[reportGeneralTypeIssues]
                except asyncio.CancelledError:
                    pass

    # Restore original state (best effort)
    model_manager._loaded_model = original_model
    model_manager._model_name = original_name
    model_manager._last_used_time = original_last_used
    model_manager._unload_requested, model_manager._shutdown_requested = original_events
    model_manager._model_unloader_task = original_task


@pytest.fixture
def mock_mlx_load():
    """Mock the mlx_lm.load function to avoid real model loading."""
    with patch("mlx_server_nano.model_manager.load") as mock_load:
        # Create mock model and tokenizer
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.bos_token = "<s>"
        mock_tokenizer.eos_token = "</s>"
        mock_load.return_value = (mock_model, mock_tokenizer)
        yield mock_load


@pytest.fixture
def mock_mlx_generate():
    """Mock the mlx_lm.generate function."""
    with patch("mlx_server_nano.model_manager.generate") as mock_generate:
        mock_generate.return_value = "Test response"
        yield mock_generate


@pytest.fixture
def mock_mlx_stream_generate():
    """Mock the mlx_lm.stream_generate function."""
    with patch("mlx_server_nano.model_manager.stream_generate") as mock_stream:
        mock_stream.return_value = iter(["Test", " response", " chunk"])
        yield mock_stream


@pytest.fixture
def mock_mlx_clear_cache():
    """Mock the mlx.core.clear_cache function."""
    with patch("mlx.core.clear_cache") as mock_clear:
        yield mock_clear


@pytest.fixture
def mock_all_mlx(
    mock_mlx_load, mock_mlx_generate, mock_mlx_stream_generate, mock_mlx_clear_cache
):
    """Composite fixture that provides all MLX mocking."""
    return {
        "load": mock_mlx_load,
        "generate": mock_mlx_generate,
        "stream_generate": mock_mlx_stream_generate,
        "clear_cache": mock_mlx_clear_cache,
    }


@pytest.fixture
async def mock_mlx_model():
    """Mock MLX model and tokenizer for testing without actual model loading."""
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()

    # Configure mock methods
    mock_model.generate = MagicMock(return_value="Mocked response")
    mock_tokenizer.encode = MagicMock(return_value=[1, 2, 3, 4])
    mock_tokenizer.decode = MagicMock(return_value="decoded text")

    with patch("mlx_lm.load", return_value=(mock_model, mock_tokenizer)):
        with patch("mlx_lm.generate", return_value="Mocked response"):
            with patch(
                "mlx_lm.stream_generate", return_value=["Mock", " stream", " response"]
            ):
                yield mock_model, mock_tokenizer


@pytest.fixture
def sample_chat_request():
    """Provide a sample chat completion request."""
    return {
        "model": "test-model",
        "messages": [{"role": "user", "content": "Hello, how are you?"}],
        "max_tokens": 50,
        "temperature": 0.1,
        "stream": False,
    }


@pytest.fixture
def sample_tool_definition():
    """Provide a sample tool definition for testing."""
    return {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "The city and state"}
                },
                "required": ["location"],
            },
        },
    }


@pytest.fixture
def sample_chat_request_with_tools(sample_tool_definition):
    """Provide a chat request with tools."""
    return {
        "model": "test-model",
        "messages": [
            {"role": "user", "content": "What's the weather in San Francisco?"}
        ],
        "tools": [sample_tool_definition],
        "max_tokens": 50,
        "temperature": 0.1,
        "stream": False,
    }


@pytest.fixture
async def started_unloader():
    """Start and stop the model unloader for tests."""
    from mlx_server_nano.model_manager import start_model_unloader, stop_model_unloader

    await start_model_unloader()
    yield
    await stop_model_unloader()


@pytest.fixture
def mock_memory_monitoring():
    """Mock psutil for memory monitoring in tests."""
    with patch("psutil.Process") as mock_process:
        mock_memory_info = MagicMock()
        mock_memory_info.rss = 1024 * 1024 * 100  # 100 MB
        mock_process.return_value.memory_info.return_value = mock_memory_info
        yield mock_process


# Test data fixtures
@pytest.fixture
def devstral_tool_response():
    """Sample Devstral tool calling response."""
    return """Sure! I'll help you get the weather information.

[TOOL_CALLS][{"name": "get_weather", "arguments": {"location": "San Francisco"}}]

The weather function has been called for San Francisco."""


@pytest.fixture
def qwen_tool_response():
    """Sample Qwen tool calling response."""
    return """I'll help you get the weather information for San Francisco.

✿FUNCTION✿: get_weather
✿ARGS✿: {"location": "San Francisco"}

The weather function has been called."""


# Pytest markers for different test categories
def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line(
        "markers", "unit: Unit tests that don't require external dependencies"
    )
    config.addinivalue_line(
        "markers",
        "integration: Integration tests that may require models or external services",
    )
    config.addinivalue_line("markers", "slow: Tests that take longer to run")
    config.addinivalue_line("markers", "memory: Tests that monitor memory usage")
    config.addinivalue_line("markers", "api: Tests for API endpoints")
    config.addinivalue_line(
        "markers", "model: Tests that involve model loading/unloading"
    )
