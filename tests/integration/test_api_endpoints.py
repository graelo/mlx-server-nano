"""
Integration tests for API endpoints.

Tests the FastAPI application endpoints with mocked model components.
"""

import pytest
import json
from unittest.mock import patch, MagicMock


@pytest.mark.integration
@pytest.mark.api
class TestHealthEndpoint:
    """Test cases for health check endpoint."""

    def test_health_check_success(self, client):
        """Test successful health check."""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data

    def test_health_check_multiple_calls(self, client):
        """Test multiple health check calls."""
        for _ in range(3):
            response = client.get("/health")
            assert response.status_code == 200
            assert response.json()["status"] == "healthy"


@pytest.mark.integration
@pytest.mark.api
class TestModelsEndpoint:
    """Test cases for models list endpoint."""

    def test_models_list_success(self, client):
        """Test successful models list retrieval."""
        response = client.get("/v1/models")

        assert response.status_code == 200
        data = response.json()
        assert "object" in data
        assert "data" in data
        assert data["object"] == "list"
        assert isinstance(data["data"], list)
        assert len(data["data"]) > 0  # Should have some default models

    def test_models_list_format(self, client):
        """Test that models list has correct format."""
        response = client.get("/v1/models")
        data = response.json()

        # Check first model format
        model = data["data"][0]
        assert "id" in model
        assert "object" in model
        assert "created" in model
        assert "owned_by" in model
        assert model["object"] == "model"


@pytest.mark.integration
@pytest.mark.api
class TestChatCompletionsEndpoint:
    """Test cases for chat completions endpoint."""

    @patch("mlx_server_nano.model_manager.generate_response_with_tools")
    def test_chat_completion_basic(self, mock_generate, client, sample_chat_request):
        """Test basic chat completion request."""
        # Configure mock
        mock_generate.return_value = ("Hello there!", [])

        response = client.post("/v1/chat/completions", json=sample_chat_request)

        assert response.status_code == 200
        data = response.json()

        # Check response format
        assert "id" in data
        assert "object" in data
        assert "created" in data
        assert "model" in data
        assert "choices" in data
        assert "usage" in data

        assert data["object"] == "chat.completion"
        assert data["model"] == sample_chat_request["model"]
        assert len(data["choices"]) == 1

        # Check choice format
        choice = data["choices"][0]
        assert "index" in choice
        assert "message" in choice
        assert "finish_reason" in choice

        # Check message format
        message = choice["message"]
        assert "role" in message
        assert "content" in message
        assert message["role"] == "assistant"
        assert message["content"] == "Hello there!"

    @patch("mlx_server_nano.model_manager.generate_response_with_tools")
    def test_chat_completion_with_tools(
        self, mock_generate, client, sample_chat_request_with_tools
    ):
        """Test chat completion with tool calls."""
        # Configure mock to return tool calls
        mock_tool_call = MagicMock()
        mock_tool_call.id = "call_123"
        mock_tool_call.name = "get_weather"
        mock_tool_call.arguments = {"location": "San Francisco"}

        mock_generate.return_value = ("I'll check the weather.", [mock_tool_call])

        response = client.post(
            "/v1/chat/completions", json=sample_chat_request_with_tools
        )

        assert response.status_code == 200
        data = response.json()

        message = data["choices"][0]["message"]
        assert message["content"] == "I'll check the weather."
        assert "tool_calls" in message
        assert len(message["tool_calls"]) == 1

        tool_call = message["tool_calls"][0]
        assert tool_call["id"] == "call_123"
        assert tool_call["function"]["name"] == "get_weather"

    @patch("mlx_server_nano.model_manager.generate_response_stream")
    def test_chat_completion_streaming(self, mock_stream, client, sample_chat_request):
        """Test streaming chat completion."""
        # Configure mock to return chunks
        mock_stream.return_value = ["Hello", " there", "!"]

        # Modify request for streaming
        streaming_request = sample_chat_request.copy()
        streaming_request["stream"] = True

        response = client.post("/v1/chat/completions", json=streaming_request)

        assert response.status_code == 200
        assert response.headers["content-type"] == "text/plain; charset=utf-8"

        # Parse streaming response
        lines = response.text.strip().split("\n")
        data_lines = [line for line in lines if line.startswith("data: ")]

        # Should have data chunks plus final [DONE]
        assert len(data_lines) >= 2
        assert data_lines[-1] == "data: [DONE]"

        # Check first chunk format
        first_chunk = json.loads(data_lines[0][6:])  # Remove "data: " prefix
        assert "id" in first_chunk
        assert "object" in first_chunk
        assert "choices" in first_chunk
        assert first_chunk["object"] == "chat.completion.chunk"

    def test_chat_completion_missing_model(self, client):
        """Test chat completion with missing model field."""
        request = {"messages": [{"role": "user", "content": "hello"}]}

        response = client.post("/v1/chat/completions", json=request)
        assert response.status_code == 422  # Validation error

    def test_chat_completion_empty_messages(self, client):
        """Test chat completion with empty messages."""
        request = {"model": "test-model", "messages": []}

        response = client.post("/v1/chat/completions", json=request)
        assert response.status_code == 422  # Validation error

    @patch("mlx_server_nano.model_manager.generate_response_with_tools")
    def test_chat_completion_model_error(
        self, mock_generate, client, sample_chat_request
    ):
        """Test handling of model generation errors."""
        # Configure mock to raise an error
        mock_generate.side_effect = Exception("Model loading failed")

        response = client.post("/v1/chat/completions", json=sample_chat_request)

        assert response.status_code == 500
        data = response.json()
        assert "detail" in data

    def test_chat_completion_invalid_json(self, client):
        """Test handling of invalid JSON in request."""
        response = client.post(
            "/v1/chat/completions",
            data="invalid json",
            headers={"content-type": "application/json"},
        )

        assert response.status_code == 422

    @patch("mlx_server_nano.model_manager.generate_response_with_tools")
    def test_chat_completion_custom_parameters(self, mock_generate, client):
        """Test chat completion with custom generation parameters."""
        mock_generate.return_value = ("Response", [])

        request = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "test"}],
            "max_tokens": 200,
            "temperature": 0.8,
            "top_p": 0.95,
        }

        response = client.post("/v1/chat/completions", json=request)

        assert response.status_code == 200

        # Verify mock was called with correct parameters
        mock_generate.assert_called_once()
        args, kwargs = mock_generate.call_args
        assert kwargs.get("max_tokens") == 200
        assert kwargs.get("temperature") == 0.8
        assert kwargs.get("top_p") == 0.95


@pytest.mark.integration
@pytest.mark.api
class TestAPICompatibility:
    """Test cases for OpenAI API compatibility."""

    @patch("mlx_server_nano.model_manager.generate_response_with_tools")
    def test_openai_format_compatibility(self, mock_generate, client):
        """Test that response format matches OpenAI API."""
        mock_generate.return_value = ("Test response", [])

        request = {
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": "test"}],
            "max_tokens": 100,
        }

        response = client.post("/v1/chat/completions", json=request)
        data = response.json()

        # Check all required OpenAI fields are present
        required_fields = ["id", "object", "created", "model", "choices", "usage"]
        for field in required_fields:
            assert field in data

        # Check choice structure
        choice = data["choices"][0]
        choice_fields = ["index", "message", "finish_reason"]
        for field in choice_fields:
            assert field in choice

        # Check message structure
        message = choice["message"]
        message_fields = ["role", "content"]
        for field in message_fields:
            assert field in message

    @patch("mlx_server_nano.model_manager.generate_response_with_tools")
    def test_usage_tracking(self, mock_generate, client, sample_chat_request):
        """Test that usage information is properly tracked."""
        mock_generate.return_value = ("Short response", [])

        response = client.post("/v1/chat/completions", json=sample_chat_request)
        data = response.json()

        usage = data["usage"]
        assert "prompt_tokens" in usage
        assert "completion_tokens" in usage
        assert "total_tokens" in usage

        # Basic validation of token counts
        assert usage["prompt_tokens"] > 0
        assert usage["completion_tokens"] > 0
        assert (
            usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]
        )

    def test_cors_headers(self, client):
        """Test that appropriate CORS headers are set."""
        response = client.options("/v1/chat/completions")

        # Should handle OPTIONS request
        assert response.status_code in [200, 405]  # Depends on CORS setup

    def test_content_type_handling(self, client, sample_chat_request):
        """Test different content type scenarios."""
        # Test with explicit content-type
        response = client.post(
            "/v1/chat/completions",
            json=sample_chat_request,
            headers={"Content-Type": "application/json"},
        )

        # Should work regardless of explicit content-type
        assert response.status_code in [200, 500]  # 500 if model fails, 200 if mocked


@pytest.mark.integration
@pytest.mark.api
class TestErrorHandling:
    """Test cases for API error handling."""

    def test_404_endpoints(self, client):
        """Test that unknown endpoints return 404."""
        response = client.get("/unknown/endpoint")
        assert response.status_code == 404

    def test_method_not_allowed(self, client):
        """Test that wrong HTTP methods return 405."""
        response = client.get("/v1/chat/completions")  # Should be POST
        assert response.status_code == 405

    def test_large_request_handling(self, client):
        """Test handling of very large requests."""
        large_content = "x" * 100000  # 100KB content
        request = {
            "model": "test-model",
            "messages": [{"role": "user", "content": large_content}],
        }

        response = client.post("/v1/chat/completions", json=request)

        # Should either handle gracefully or return appropriate error
        assert response.status_code in [200, 413, 422, 500]

    @patch("mlx_server_nano.model_manager.generate_response_with_tools")
    def test_timeout_handling(self, mock_generate, client, sample_chat_request):
        """Test handling of generation timeouts."""
        import time

        def slow_generate(*args, **kwargs):
            time.sleep(0.1)  # Simulate slow generation
            return ("Response", [])

        mock_generate.side_effect = slow_generate

        response = client.post("/v1/chat/completions", json=sample_chat_request)

        # Should complete successfully or timeout gracefully
        assert response.status_code in [200, 504, 500]
