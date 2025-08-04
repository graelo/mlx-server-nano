"""
Integration tests for model management functionality.

Tests model loading, unloading, and lifecycle management with real interactions
but mocked MLX components for speed.
"""

import asyncio
import pytest
from unittest.mock import patch, MagicMock

from mlx_server_nano import model_manager
from mlx_server_nano.model_manager import (
    start_model_unloader,
    stop_model_unloader,
    load_model,
    generate_response_with_tools,
    generate_response_stream,
)


@pytest.mark.integration
@pytest.mark.model
@pytest.mark.asyncio
class TestModelUnloaderIntegration:
    """Integration tests for the model unloader background task."""

    async def test_model_unloader_lifecycle(self, clean_model_manager, test_env_vars):
        """Test complete model unloader lifecycle."""
        # Ensure clean state
        await stop_model_unloader()
        model_manager._shutdown_requested.clear()
        model_manager._unload_requested.clear()

        # Start unloader
        await start_model_unloader()
        assert model_manager._model_unloader_task is not None
        assert not model_manager._model_unloader_task.done()

        # Trigger unload event
        model_manager._unload_requested.set()

        # Wait a short time
        await asyncio.sleep(0.1)

        # Stop unloader
        await stop_model_unloader()
        assert model_manager._shutdown_requested.is_set()

    @patch("mlx_lm.load")
    async def test_model_loading_with_unloader(
        self, mock_load, clean_model_manager, test_env_vars
    ):
        """Test model loading integration with unloader."""
        # Configure mock
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_load.return_value = (mock_model, mock_tokenizer)

        # Start unloader
        await start_model_unloader()

        try:
            # Load model
            result = load_model("test-model")
            assert result == (mock_model, mock_tokenizer)
            assert model_manager._loaded_model is not None
            assert model_manager._model_name == "test-model"

            # Verify unload is scheduled
            assert (
                model_manager._unload_requested.is_set()
                or model_manager._last_used_time > 0
            )

        finally:
            await stop_model_unloader()

    @patch("mlx_lm.load")
    async def test_model_auto_unload_timing(
        self, mock_load, clean_model_manager, test_env_vars
    ):
        """Test that model is unloaded after timeout."""
        # Configure mock
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_load.return_value = (mock_model, mock_tokenizer)

        # Start unloader with short timeout (from test_env_vars)
        await start_model_unloader()

        try:
            # Load model
            load_model("test-model")
            assert model_manager._loaded_model is not None

            # Wait for timeout + margin (test timeout is 2 seconds)
            await asyncio.sleep(3)

            # Model should be unloaded
            assert model_manager._loaded_model is None
            assert model_manager._model_name is None

        finally:
            await stop_model_unloader()

    @patch("mlx_lm.load")
    async def test_model_cache_reuse_resets_timer(
        self, mock_load, clean_model_manager, test_env_vars
    ):
        """Test that reusing cached model resets the unload timer."""
        # Configure mock
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_load.return_value = (mock_model, mock_tokenizer)

        await start_model_unloader()

        try:
            # Load model
            load_model("test-model")
            first_load_time = model_manager._last_used_time

            # Wait a bit
            await asyncio.sleep(1)

            # Load same model again (should hit cache)
            load_model("test-model")
            second_load_time = model_manager._last_used_time

            # Timer should be reset
            assert second_load_time > first_load_time
            assert model_manager._loaded_model is not None

        finally:
            await stop_model_unloader()


@pytest.mark.integration
@pytest.mark.model
class TestModelGenerationIntegration:
    """Integration tests for model generation with lifecycle management."""

    @patch("mlx_lm.load")
    @patch("mlx_lm.generate")
    @patch("mlx_server_nano.model_manager.format_messages_for_model")
    @patch("mlx_server_nano.model_manager.get_tool_parser")
    def test_generate_response_complete_flow(
        self,
        mock_parser_func,
        mock_format,
        mock_generate,
        mock_load,
        clean_model_manager,
    ):
        """Test complete response generation flow."""
        # Configure mocks
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_load.return_value = (mock_model, mock_tokenizer)
        mock_format.return_value = "formatted prompt"
        mock_generate.return_value = "model response"

        mock_parser = MagicMock()
        mock_parser.parse_tool_calls.return_value = ("content", [])
        mock_parser_func.return_value = mock_parser

        # Generate response
        content, tool_calls = generate_response_with_tools(
            "test-model", [{"role": "user", "content": "hello"}], max_tokens=100
        )

        # Verify result
        assert content == "content"
        assert tool_calls == []

        # Verify model was loaded and cached
        assert model_manager._loaded_model == (mock_model, mock_tokenizer)
        assert model_manager._model_name == "test-model"
        assert model_manager._last_used_time > 0

        # Verify all steps were called
        mock_load.assert_called_once_with("test-model")
        mock_format.assert_called_once()
        mock_generate.assert_called_once()
        mock_parser.parse_tool_calls.assert_called_once_with("model response")

    @patch("mlx_lm.load")
    @patch("mlx_lm.stream_generate")
    @patch("mlx_server_nano.model_manager.format_messages_for_model")
    def test_generate_stream_complete_flow(
        self, mock_format, mock_stream, mock_load, clean_model_manager
    ):
        """Test complete streaming generation flow."""
        # Configure mocks
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_load.return_value = (mock_model, mock_tokenizer)
        mock_format.return_value = "formatted prompt"

        # Mock streaming chunks
        chunk1 = MagicMock()
        chunk1.text = "Hello"
        chunk2 = MagicMock()
        chunk2.text = " world"
        mock_stream.return_value = [chunk1, chunk2]

        # Generate streaming response
        chunks = list(
            generate_response_stream(
                "test-model", [{"role": "user", "content": "hello"}]
            )
        )

        # Verify result
        assert chunks == ["Hello", " world"]

        # Verify model state
        assert model_manager._loaded_model == (mock_model, mock_tokenizer)
        assert model_manager._last_used_time > 0

    @patch("mlx_lm.load")
    def test_multiple_model_requests_caching(self, mock_load, clean_model_manager):
        """Test that multiple requests for same model use cache."""
        # Configure mock
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_load.return_value = (mock_model, mock_tokenizer)

        # Load same model multiple times
        result1 = load_model("test-model")
        result2 = load_model("test-model")
        result3 = load_model("test-model")

        # Should all return same objects
        assert result1 == result2 == result3

        # Should only call mlx_lm.load once
        mock_load.assert_called_once_with("test-model")

    @patch("mlx_lm.load")
    def test_different_models_replace_cache(self, mock_load, clean_model_manager):
        """Test that loading different models replaces cache."""
        # Configure mocks for different models
        model1, tokenizer1 = MagicMock(), MagicMock()
        model2, tokenizer2 = MagicMock(), MagicMock()

        mock_load.side_effect = [(model1, tokenizer1), (model2, tokenizer2)]

        # Load different models
        result1 = load_model("model-1")
        assert model_manager._model_name == "model-1"

        result2 = load_model("model-2")
        assert model_manager._model_name == "model-2"

        # Should load both models
        assert mock_load.call_count == 2
        assert result1 != result2


@pytest.mark.integration
@pytest.mark.model
@pytest.mark.memory
class TestMemoryManagement:
    """Integration tests for memory management functionality."""

    @patch("mlx_lm.load")
    @patch("mlx.core.clear_cache")
    @patch("gc.collect")
    def test_model_unload_memory_cleanup(
        self, mock_gc, mock_clear_cache, mock_load, clean_model_manager
    ):
        """Test that model unloading performs proper memory cleanup."""
        # Configure mocks
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_load.return_value = (mock_model, mock_tokenizer)
        mock_gc.return_value = 10  # Simulate collected objects

        # Load then unload model
        load_model("test-model")
        assert model_manager._loaded_model is not None

        # Manually trigger unload (normally done by background task)
        model_manager._unload_model()

        # Verify state is cleared
        assert model_manager._loaded_model is None
        assert model_manager._model_name is None

        # Verify cleanup was called
        mock_clear_cache.assert_called_once()
        assert mock_gc.call_count >= 1

    @patch("psutil.Process")
    def test_memory_monitoring_integration(self, mock_process, mock_memory_monitoring):
        """Test integration with memory monitoring (if psutil available)."""
        # This test verifies the monitoring infrastructure works
        # without actually measuring real memory

        mock_memory_info = MagicMock()
        mock_memory_info.rss = 1024 * 1024 * 100  # 100 MB
        mock_process.return_value.memory_info.return_value = mock_memory_info

        # Get memory usage (simulated)
        import os

        try:
            import psutil

            process = psutil.Process(os.getpid())
            memory_mb = process.memory_info().rss / 1024 / 1024
            assert memory_mb == 100.0
        except ImportError:
            pytest.skip("psutil not available")


@pytest.mark.integration
@pytest.mark.model
@pytest.mark.slow
class TestConcurrentModelAccess:
    """Integration tests for concurrent model access patterns."""

    @patch("mlx_lm.load")
    async def test_concurrent_model_loading(self, mock_load, clean_model_manager):
        """Test concurrent requests for same model."""
        # Configure mock with delay to simulate loading time
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        async def slow_load(model_name):
            await asyncio.sleep(0.1)  # Simulate loading time
            return (mock_model, mock_tokenizer)

        mock_load.side_effect = lambda name: slow_load(name)

        # Start multiple concurrent loads
        async def load_task():
            return load_model("test-model")

        tasks = [asyncio.create_task(load_task()) for _ in range(5)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # All should succeed or fail consistently
        success_count = sum(1 for r in results if not isinstance(r, Exception))
        assert success_count > 0  # At least some should succeed

        # If any succeeded, they should all return the same model
        if success_count > 0:
            successful_results = [r for r in results if not isinstance(r, Exception)]
            first_result = successful_results[0]
            assert all(r == first_result for r in successful_results)

    @patch("mlx_lm.load")
    def test_model_access_thread_safety(self, mock_load, clean_model_manager):
        """Test thread safety of model loading."""
        import threading

        # Configure mock
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_load.return_value = (mock_model, mock_tokenizer)

        results = []
        errors = []

        def load_in_thread():
            try:
                result = load_model("test-model")
                results.append(result)
            except Exception as e:
                errors.append(e)

        # Start multiple threads
        threads = [threading.Thread(target=load_in_thread) for _ in range(10)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        # All threads should succeed
        assert len(errors) == 0
        assert len(results) == 10

        # All should return the same model (cached)
        assert all(r == results[0] for r in results)


@pytest.mark.integration
@pytest.mark.model
class TestModelConfigurationIntegration:
    """Integration tests for model configuration handling."""

    @patch("mlx_lm.load")
    def test_model_specific_configurations(self, mock_load, clean_model_manager):
        """Test that model-specific configurations are applied."""
        # Configure mock
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_load.return_value = (mock_model, mock_tokenizer)

        # Test Qwen model configuration
        from mlx_server_nano.model_manager import _setup_generation_kwargs

        qwen_kwargs = _setup_generation_kwargs("qwen-test-model", max_tokens=100)
        assert "stop_strings" in qwen_kwargs
        assert qwen_kwargs["max_tokens"] == 100

        # Test non-Qwen model
        other_kwargs = _setup_generation_kwargs("gpt-4", max_tokens=200)
        assert "stop_strings" not in other_kwargs
        assert other_kwargs["max_tokens"] == 200

    def test_environment_variable_integration(self, test_env_vars):
        """Test that environment variables are properly integrated."""
        # test_env_vars fixture sets various MLX_* environment variables

        from mlx_server_nano.config import ServerConfig

        config = ServerConfig.from_env()

        # Should use test values
        assert config.model_idle_timeout == 2  # From test_config fixture
        assert config.default_max_tokens == 100
        assert config.log_level == "DEBUG"
