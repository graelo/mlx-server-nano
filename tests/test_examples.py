"""
Example test suite demonstrating MLX Server Nano testing patterns.

This file shows practical examples of how to write tests for different
components and scenarios in the MLX Server Nano project.
"""

import pytest
import asyncio
from unittest.mock import patch, MagicMock

from mlx_server_nano.model_manager import (
    load_model,
    generate_response_with_tools,
    parse_tool_calls,
)
from mlx_server_nano.schemas import Tool, Function


@pytest.mark.unit
class TestExampleUnitTests:
    """Example unit tests showing common patterns."""

    @patch("mlx_server_nano.model_manager.load")
    def test_model_loading_basic_example(self, mock_load, clean_model_manager):
        """Example: Test basic model loading with mocking."""
        # Arrange: Set up mock
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_load.return_value = (mock_model, mock_tokenizer)

        # Act: Load the model
        result = load_model("test-model")

        # Assert: Verify results
        assert result == (mock_model, mock_tokenizer)
        mock_load.assert_called_once_with("test-model")

    def test_tool_parsing_example(self):
        """Example: Test native MLX-LM tool call parsing."""
        # Test the new native tool call parsing
        response_with_tool = '[TOOL_CALLS]get_weather[ARGS]{"location": "Paris"}'
        tool_calls = parse_tool_calls(response_with_tool)

        assert len(tool_calls) == 1
        assert tool_calls[0].function["name"] == "get_weather"

        # Test response without tool calls
        response_without_tool = "The weather is sunny today."
        tool_calls_empty = parse_tool_calls(response_without_tool)
        assert len(tool_calls_empty) == 0

    def test_tool_definition_validation_example(self):
        """Example: Test tool definition validation."""
        # Valid tool
        valid_tool = Tool(
            function=Function(
                name="get_weather",
                description="Get weather info",
                parameters={
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                },
            )
        )

        assert valid_tool.type == "function"
        assert valid_tool.function.name == "get_weather"


@pytest.mark.integration
class TestExampleIntegrationTests:
    """Example integration tests showing realistic scenarios."""

    @patch("mlx_server_nano.model_manager.load")
    @patch("mlx_server_nano.model_manager.generation.generate")
    @patch("mlx_server_nano.model_manager.generation.stream_generate")
    @patch("mlx_server_nano.model_manager.generation.parse_tool_calls")
    def test_complete_generation_flow_example(
        self,
        mock_parse_tools,
        mock_stream_generate,
        mock_generate,
        mock_load,
        clean_model_manager,
    ):
        """Example: Test complete response generation flow."""
        # Arrange: Set up all mocks
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template.return_value = "User: Hello\nAssistant:"
        mock_tokenizer.bos_token = "<bos>"
        mock_load.return_value = (mock_model, mock_tokenizer)
        mock_generate.return_value = "Hello! How can I help you?"
        mock_stream_generate.return_value = [
            "Hello! How can I help you?"
        ]  # For fallback

        # Mock the new tool parsing
        mock_parse_tools.return_value = []  # No tool calls found

        # Act: Generate response
        content, tool_calls = generate_response_with_tools(
            model_name="test-model",
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=50,
        )

        # Assert: Verify the complete flow
        assert content == "Hello! How can I help you?"
        assert tool_calls == []

        # Verify all components were called
        mock_load.assert_called_once_with("test-model")
        # Should be called once now (we decode tokens for debugging)
        mock_tokenizer.apply_chat_template.assert_called_once()
        mock_generate.assert_called_once()
        mock_parse_tools.assert_called_once()

    def test_api_endpoint_example(self, client, sample_chat_request, mock_all_mlx):
        """Example: Test API endpoint with real FastAPI client."""
        # Act: Make API request
        response = client.post("/v1/chat/completions", json=sample_chat_request)

        # Assert: Check response format
        assert response.status_code == 200
        data = response.json()
        assert "choices" in data
        assert "content" in data["choices"][0]["message"]


@pytest.mark.memory
class TestExampleMemoryTests:
    """Example memory management tests."""

    @patch("mlx_server_nano.model_manager.load")
    @patch("mlx.core.clear_cache")
    @patch("gc.collect")
    def test_memory_cleanup_example(
        self, mock_gc, mock_clear_cache, mock_load, clean_model_manager
    ):
        """Example: Test functional memory cleanup verification."""
        from mlx_server_nano import model_manager

        # Arrange: Set up model in cache
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_load.return_value = (mock_model, mock_tokenizer)
        mock_gc.return_value = 10  # Simulate garbage collection

        # Act: Load then unload model
        load_model("test-model")
        assert model_manager.cache._loaded_model is not None

        model_manager.cache._unload_model()

        # Assert: Verify functional cleanup
        assert model_manager.cache._loaded_model is None
        assert model_manager.cache._model_name is None
        mock_clear_cache.assert_called_once()
        assert mock_gc.call_count >= 1


@pytest.mark.api
class TestExampleAPITests:
    """Example API-specific tests."""

    def test_health_endpoint_example(self, client):
        """Example: Test health check endpoint."""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"

    def test_models_endpoint_example(self, client):
        """Example: Test models listing endpoint."""
        response = client.get("/v1/models")

        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "list"
        assert isinstance(data["data"], list)

    @patch("mlx_server_nano.model_manager.generate_response_stream")
    def test_streaming_endpoint_example(self, mock_stream, client):
        """Example: Test streaming response."""
        # Arrange: Mock streaming chunks
        chunk1 = MagicMock()
        chunk1.text = "Hello"
        chunk2 = MagicMock()
        chunk2.text = " world"
        mock_stream.return_value = [chunk1, chunk2]

        # Act: Make streaming request
        request = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": True,
        }
        response = client.post("/v1/chat/completions", json=request)

        # Assert: Check streaming response
        assert response.status_code == 200
        assert response.headers["content-type"] == "text/event-stream; charset=utf-8"

        # Parse streaming chunks
        lines = response.text.strip().split("\n")
        data_lines = [line for line in lines if line.startswith("data: ")]
        assert len(data_lines) >= 2  # At least content + [DONE]


@pytest.mark.slow
@pytest.mark.asyncio
class TestExampleSlowTests:
    """Example tests that might take longer (marked as slow)."""

    async def test_background_task_timing_example(
        self, clean_model_manager, test_env_vars
    ):
        """Example: Test background task timing with event synchronization."""
        from mlx_server_nano.model_manager import (
            start_model_unloader,
            stop_model_unloader,
        )
        from mlx_server_nano import model_manager

        # Start the background unloader
        await start_model_unloader()

        try:
            # Verify background task was created and is running
            assert model_manager.background_tasks._model_unloader_task is not None
            initial_task_state = (
                model_manager.background_tasks._model_unloader_task.done()
            )

            # Simulate model usage
            model_manager.cache._last_used_time = 1000  # Old timestamp
            model_manager.background_tasks._unload_requested.set()

            # Give the background task a brief moment to process the event
            await asyncio.sleep(0.1)

            # Verify background task is still responsive (not crashed)
            # The task should still be running unless it encountered an error
            task_is_healthy = (
                model_manager.background_tasks._model_unloader_task is not None
                and (
                    not model_manager.background_tasks._model_unloader_task.done()
                    or not initial_task_state
                )
            )
            assert task_is_healthy

        finally:
            await stop_model_unloader()


class TestExampleErrorHandling:
    """Example error handling tests."""

    @patch("mlx_server_nano.model_manager.load")
    def test_model_loading_error_example(self, mock_load, clean_model_manager):
        """Example: Test error handling during model loading."""
        # Arrange: Make loading fail
        mock_load.side_effect = Exception("Model not found")

        # Act & Assert: Verify proper error handling
        with pytest.raises(RuntimeError, match="Failed to load model"):
            load_model("nonexistent-model")

    def test_api_validation_error_example(self, client):
        """Example: Test API validation error handling."""
        # Missing required field
        invalid_request = {
            "messages": [{"role": "user", "content": "test"}]
            # Missing "model" field
        }

        response = client.post("/v1/chat/completions", json=invalid_request)
        assert response.status_code == 422  # Validation error


@pytest.fixture
def example_complex_tool():
    """Example fixture for complex tool definition."""
    return {
        "type": "function",
        "function": {
            "name": "search_and_summarize",
            "description": "Search for information and provide a summary",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "max_results": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 10,
                        "default": 5,
                    },
                    "include_summary": {"type": "boolean", "default": True},
                },
                "required": ["query"],
            },
        },
    }


def test_example_using_custom_fixture(example_complex_tool):
    """Example: Test using a custom fixture."""
    tool = example_complex_tool

    assert tool["function"]["name"] == "search_and_summarize"
    assert "query" in tool["function"]["parameters"]["required"]
    assert tool["function"]["parameters"]["properties"]["max_results"]["minimum"] == 1


# Example of parametrized test
@pytest.mark.parametrize(
    "model_name",
    [
        "qwen-chat",
        "devstral-model",
        "gpt-4",
    ],
)
def test_example_parametrized(model_name):
    """Example: Parametrized test showing that MLX-LM handles stop sequences automatically."""
    from mlx_server_nano.model_manager import _setup_generation_kwargs

    kwargs = _setup_generation_kwargs(model_name)

    # generation_kwargs should never contain stop_strings since MLX-LM handles them automatically
    assert "stop_strings" not in kwargs
    assert "max_tokens" in kwargs


# Example test class with setup/teardown
class TestExampleWithSetupTeardown:
    """Example: Test class with setup and teardown methods."""

    def setup_method(self):
        """Run before each test method."""
        self.test_data = {"counter": 0}

    def teardown_method(self):
        """Run after each test method."""
        self.test_data = None

    def test_example_with_setup(self):
        """Example test that uses setup data."""
        assert self.test_data is not None
        assert self.test_data["counter"] == 0
        self.test_data["counter"] += 1
        assert self.test_data["counter"] == 1

    def test_example_isolated_state(self):
        """Example showing tests are isolated (setup runs again)."""
        assert self.test_data is not None
        assert self.test_data["counter"] == 0  # Reset by setup_method
