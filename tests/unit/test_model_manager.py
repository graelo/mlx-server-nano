"""
Unit tests for model manager functionality.

Tests model loading, caching, and lifecycle management with mocked MLX components.
"""

import pytest
from unittest.mock import MagicMock, patch
import time

from mlx_server_nano import model_manager
from mlx_server_nano.model_manager import (
    load_model,
    get_current_time,
    _unload_model,
    _schedule_unload,
    start_model_unloader,
    stop_model_unloader,
    generate_response_with_tools,
    generate_response_stream,
    _setup_generation_kwargs,
    _get_stop_sequences,
    _try_generate_with_fallback,
)


@pytest.mark.unit
class TestGetCurrentTime:
    """Test cases for get_current_time function."""

    def test_get_current_time_returns_float(self):
        """Test that get_current_time returns a float timestamp."""
        timestamp = get_current_time()
        assert isinstance(timestamp, float)
        assert timestamp > 0

    def test_get_current_time_changes(self):
        """Test that get_current_time returns different values over time."""
        time1 = get_current_time()
        time.sleep(0.001)  # Small delay
        time2 = get_current_time()
        assert time2 > time1


@pytest.mark.unit
class TestUnloadModel:
    """Test cases for model unloading functionality."""

    def test_unload_model_when_no_model_loaded(self, clean_model_manager):
        """Test unloading when no model is loaded."""
        # Ensure no model is loaded
        model_manager._loaded_model = None
        model_manager._model_name = None

        # Should not raise any errors
        _unload_model()

        assert model_manager._loaded_model is None
        assert model_manager._model_name is None

    @patch("mlx.core.clear_cache")
    @patch("gc.collect")
    def test_unload_model_with_loaded_model(
        self, mock_gc_collect, mock_clear_cache, clean_model_manager
    ):
        """Test unloading when a model is loaded."""
        # Set up loaded model
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        model_manager._loaded_model = (mock_model, mock_tokenizer)
        model_manager._model_name = "test-model"

        # Configure mocks
        mock_gc_collect.return_value = 5  # Simulate collected objects

        # Unload model
        _unload_model()

        # Verify state is cleared
        assert model_manager._loaded_model is None
        assert model_manager._model_name is None

        # Verify cleanup was called
        mock_clear_cache.assert_called_once()
        assert mock_gc_collect.call_count >= 1  # Called multiple times

    @patch("mlx.core.clear_cache")
    def test_unload_model_clear_cache_error(
        self, mock_clear_cache, clean_model_manager
    ):
        """Test handling of clear_cache errors."""
        # Set up loaded model
        model_manager._loaded_model = (MagicMock(), MagicMock())
        model_manager._model_name = "test-model"

        # Make clear_cache raise an error
        mock_clear_cache.side_effect = Exception("Cache clear failed")

        # Should not raise, just log warning
        _unload_model()

        # Model should still be unloaded
        assert model_manager._loaded_model is None
        assert model_manager._model_name is None


@pytest.mark.unit
class TestScheduleUnload:
    """Test cases for schedule unload functionality."""

    def test_schedule_unload_sets_event(self):
        """Test that schedule unload sets the unload event."""
        # Reset events
        model_manager._unload_requested.clear()
        model_manager._shutdown_requested.clear()

        _schedule_unload()

        assert model_manager._unload_requested.is_set()

    def test_schedule_unload_when_shutdown_requested(self):
        """Test that schedule unload doesn't set event when shutdown is requested."""
        model_manager._shutdown_requested.set()
        model_manager._unload_requested.clear()

        _schedule_unload()

        # Should not set unload event when shutdown is requested
        assert not model_manager._unload_requested.is_set()

        # Clean up
        model_manager._shutdown_requested.clear()


@pytest.mark.unit
class TestLoadModel:
    """Test cases for model loading functionality."""

    @patch("mlx_server_nano.model_manager.load")
    def test_load_model_success(self, mock_load, clean_model_manager):
        """Test successful model loading."""
        # Configure mock
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_load.return_value = (mock_model, mock_tokenizer)

        # Load model
        result = load_model("test-model")

        # Verify result
        assert result == (mock_model, mock_tokenizer)
        assert model_manager._loaded_model == (mock_model, mock_tokenizer)
        assert model_manager._model_name == "test-model"
        assert model_manager._last_used_time > 0

        # Verify mlx_lm.load was called
        mock_load.assert_called_once_with("test-model")

    @patch("mlx_server_nano.model_manager.load")
    def test_load_model_caching(self, mock_load, clean_model_manager):
        """Test that model loading uses cache when same model requested."""
        # Configure mock
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_load.return_value = (mock_model, mock_tokenizer)

        # Load model first time
        result1 = load_model("test-model")

        # Load same model again
        result2 = load_model("test-model")

        # Should return same objects
        assert result1 == result2
        assert result1 is result2

        # mlx_lm.load should only be called once
        mock_load.assert_called_once_with("test-model")

    @patch("mlx_server_nano.model_manager.load")
    def test_load_different_model_replaces_cache(self, mock_load, clean_model_manager):
        """Test that loading different model replaces cached model."""
        # Configure mocks for different models
        mock_model1 = MagicMock()
        mock_tokenizer1 = MagicMock()
        mock_model2 = MagicMock()
        mock_tokenizer2 = MagicMock()

        mock_load.side_effect = [
            (mock_model1, mock_tokenizer1),
            (mock_model2, mock_tokenizer2),
        ]

        # Load first model
        result1 = load_model("model-1")
        assert model_manager._model_name == "model-1"

        # Load different model
        result2 = load_model("model-2")
        assert model_manager._model_name == "model-2"
        assert result1 != result2

        # Both calls should go to mlx_lm.load
        assert mock_load.call_count == 2

    @patch("mlx_server_nano.model_manager.load")
    def test_load_model_error_handling(self, mock_load, clean_model_manager):
        """Test error handling in model loading."""
        # Make load raise an error
        mock_load.side_effect = Exception("Loading failed")

        # Should raise RuntimeError
        with pytest.raises(RuntimeError, match="Failed to load model"):
            load_model("failing-model")

        # Model state should remain clean
        assert model_manager._loaded_model is None
        assert model_manager._model_name is None


@pytest.mark.unit
class TestSetupGenerationKwargs:
    """Test cases for generation kwargs setup."""

    def test_setup_generation_kwargs_defaults(self):
        """Test setup with default values."""
        kwargs = _setup_generation_kwargs("test-model")

        assert "max_tokens" in kwargs
        assert isinstance(kwargs["max_tokens"], int)

    def test_setup_generation_kwargs_custom_max_tokens(self):
        """Test setup with custom max_tokens."""
        kwargs = _setup_generation_kwargs("test-model", max_tokens=200)

        assert kwargs["max_tokens"] == 200

    def test_setup_generation_kwargs_qwen_stop_strings(self):
        """Test that qwen3 template gets special stop strings."""
        kwargs = _setup_generation_kwargs("qwen-model")

        # Template manager should provide stop sequences for qwen models
        stop_sequences = _get_stop_sequences("qwen-model")

        # generation_kwargs no longer contains stop_strings
        assert "stop_strings" not in kwargs

        # But stop sequences are properly configured for qwen3 template
        assert len(stop_sequences) > 0
        assert "✿RESULT✿:" in stop_sequences
        assert "✿RETURN✿:" in stop_sequences
        assert "<|im_end|>" in stop_sequences

    def test_setup_generation_kwargs_non_qwen_no_stop_strings(self):
        """Test that models without template configuration don't get stop strings."""
        kwargs = _setup_generation_kwargs("gpt-4")

        # Models without template configuration should have no stop sequences
        stop_sequences = _get_stop_sequences("gpt-4")

        assert "stop_strings" not in kwargs
        assert len(stop_sequences) == 0

    def test_setup_generation_kwargs_devstral_stop_strings(self):
        """Test that devstral template gets special stop strings."""
        kwargs = _setup_generation_kwargs("devstral-model")

        # Template manager should provide stop sequences for devstral models
        stop_sequences = _get_stop_sequences("devstral-model")

        # generation_kwargs no longer contains stop_strings
        assert "stop_strings" not in kwargs

        # Modern Devstral models use clean tool calling format - no stop sequences needed
        assert len(stop_sequences) == 0

    def test_setup_generation_kwargs_stop_param_override(self):
        """Test that stop parameter overrides template defaults."""
        # Test with string stop parameter
        stop_sequences = _get_stop_sequences("qwen-model", "custom_stop")
        assert stop_sequences == ["custom_stop"]

        # Test with list stop parameter
        stop_sequences = _get_stop_sequences("qwen-model", ["stop1", "stop2"])
        assert stop_sequences == ["stop1", "stop2"]


@pytest.mark.unit
class TestTryGenerateWithFallback:
    """Test cases for generation with fallback functionality."""

    @patch("mlx_server_nano.model_manager.generate")
    def test_generate_success(self, mock_generate):
        """Test successful generation without fallback."""
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_generate.return_value = "Generated response"

        result = _try_generate_with_fallback(
            mock_model, mock_tokenizer, "test prompt", max_tokens=50
        )

        assert result == "Generated response"
        mock_generate.assert_called_once_with(
            model=mock_model,
            tokenizer=mock_tokenizer,
            prompt="test prompt",
            max_tokens=50,
        )

    @patch("mlx_server_nano.model_manager.stream_generate")
    @patch("mlx_server_nano.model_manager.generate")
    def test_generate_fallback_to_stream(self, mock_generate, mock_stream_generate):
        """Test fallback to stream_generate when generate fails."""
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        # Make generate fail
        mock_generate.side_effect = Exception("Generate failed")

        # Configure stream_generate to return chunks
        mock_stream_generate.return_value = ["Hello", " world", "!"]

        result = _try_generate_with_fallback(mock_model, mock_tokenizer, "test prompt")

        assert result == "Hello world!"
        mock_generate.assert_called_once()
        mock_stream_generate.assert_called_once()


@pytest.mark.unit
@pytest.mark.asyncio
class TestModelUnloaderLifecycle:
    """Test cases for model unloader background task lifecycle."""

    async def test_start_model_unloader(self):
        """Test starting the model unloader."""
        # Ensure clean state
        await stop_model_unloader()

        # Start unloader
        await start_model_unloader()

        # Should have created a task
        assert model_manager._model_unloader_task is not None
        assert not model_manager._model_unloader_task.done()

        # Clean up
        await stop_model_unloader()

    async def test_stop_model_unloader(self):
        """Test stopping the model unloader."""
        # Start unloader
        await start_model_unloader()

        # Stop unloader
        await stop_model_unloader()

        # Task should be done and shutdown requested
        assert model_manager._shutdown_requested.is_set()
        if model_manager._model_unloader_task:
            assert model_manager._model_unloader_task.done()

    async def test_start_unloader_multiple_times(self):
        """Test that starting unloader multiple times doesn't create multiple tasks."""
        # Clean state
        await stop_model_unloader()

        # Start multiple times
        await start_model_unloader()
        task1 = model_manager._model_unloader_task

        await start_model_unloader()
        task2 = model_manager._model_unloader_task

        # Should reuse task if still running
        if task1 and not task1.done():
            assert task1 is task2

        # Clean up
        await stop_model_unloader()


@pytest.mark.unit
class TestGenerateResponseIntegration:
    """Test cases for response generation functions (mocked)."""

    @patch("mlx_server_nano.model_manager.load_model")
    @patch("mlx_server_nano.model_manager._try_generate_with_fallback")
    @patch("mlx_server_nano.model_manager.parse_tool_calls")
    def test_generate_response_with_tools_success(
        self, mock_parse_tools, mock_generate, mock_load_model
    ):
        """Test successful response generation with tools."""
        # Configure mocks
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template.return_value = "formatted prompt"
        mock_load_model.return_value = (mock_model, mock_tokenizer)
        mock_generate.return_value = "model response"

        # Mock the new tool parsing
        mock_parse_tools.return_value = []  # No tool calls found

        # Call function
        content, tool_calls = generate_response_with_tools(
            "test-model", [{"role": "user", "content": "hello"}]
        )

        # Verify result - content should be the original response, tool_calls empty
        assert content == "model response"
        assert tool_calls == []

        # Verify calls
        mock_load_model.assert_called_once_with("test-model")
        # apply_chat_template should be called once now (we decode tokens for debugging)
        mock_tokenizer.apply_chat_template.assert_called_once()
        mock_generate.assert_called_once()
        mock_parse_tools.assert_called_once_with("model response")

    @patch("mlx_server_nano.model_manager.load_model")
    def test_generate_response_with_tools_load_error(self, mock_load_model):
        """Test error handling when model loading fails."""
        mock_load_model.side_effect = Exception("Load failed")

        with pytest.raises(Exception):
            generate_response_with_tools("test-model", [])

    @patch("mlx_server_nano.model_manager.load_model")
    @patch("mlx_server_nano.model_manager.stream_generate")
    def test_generate_response_stream_success(self, mock_stream, mock_load_model):
        """Test successful streaming response generation."""
        # Configure mocks
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template.return_value = "formatted prompt"
        mock_load_model.return_value = (mock_model, mock_tokenizer)

        # Mock stream chunks with text attribute
        chunk1 = MagicMock()
        chunk1.text = "Hello"
        chunk2 = MagicMock()
        chunk2.text = " world"
        mock_stream.return_value = [chunk1, chunk2]

        # Call function and collect results
        chunks = list(generate_response_stream("test-model", []))

        # Verify result - expecting tuples with finish_reason
        expected_chunks = [("Hello", None), (" world", None), ("", "stop")]
        assert chunks == expected_chunks

        # Verify calls
        mock_load_model.assert_called_once()
        # apply_chat_template should be called once now (we decode tokens for debugging)
        mock_tokenizer.apply_chat_template.assert_called_once()
        mock_stream.assert_called_once()
