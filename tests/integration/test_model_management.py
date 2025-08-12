"""
Integration tests for model management functionality.

Tests model loading, unloading, and lifecycle management with real interactions
but mocked MLX components for speed.
"""

import asyncio
import pytest
from unittest.mock import patch, MagicMock

from mlx_server_nano import model_manager
from mlx_server_nano.model_manager.cache_manager import model_cache
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

    async def test_model_unloader_lifecycle(
        self,
        clean_model_manager,
        test_env_vars: dict[str, int | str | float | bool],
    ) -> None:
        """Test complete model unloader lifecycle."""
        # Ensure clean state
        await stop_model_unloader()
        model_manager.background_tasks._shutdown_requested.clear()
        model_manager.background_tasks._unload_requested.clear()

        # Start unloader
        await start_model_unloader()
        assert model_manager.background_tasks._model_unloader_task is not None
        assert not model_manager.background_tasks._model_unloader_task.done()

        # Trigger unload event
        model_manager.background_tasks._unload_requested.set()

        # Wait a short time
        await asyncio.sleep(0.1)

        # Stop unloader
        await stop_model_unloader()
        assert model_manager.background_tasks._shutdown_requested.is_set()

    @patch("mlx_server_nano.model_manager.load")
    async def test_model_loading_with_unloader(
        self,
        mock_load: MagicMock,
        clean_model_manager,
        test_env_vars: dict[str, int | str | float | bool],
    ) -> None:
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
            assert model_cache._loaded_model is not None
            assert model_cache._model_name == "test-model"

            # Verify unload is scheduled
            assert (
                model_manager.background_tasks._unload_requested.is_set()
                or model_cache._last_used_time > 0
            )

        finally:
            await stop_model_unloader()

    @patch("mlx_server_nano.model_manager.load")
    async def test_model_auto_unload_timing(
        self, mock_load, clean_model_manager, test_env_vars
    ):
        """Test that model is unloaded after timeout."""
        # Configure mock
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_load.return_value = (mock_model, mock_tokenizer)

        # Create a completion event for synchronization
        unload_completed = asyncio.Event()

        # Patch _unload_model to signal completion
        original_unload = model_manager.cache.unload_model

        def synchronized_unload():
            result = original_unload()
            unload_completed.set()
            return result

        # Use test timeout value
        test_timeout = 2  # From test_env_vars

        with (
            patch.object(model_manager.cache, "unload_model", synchronized_unload),
            patch.object(
                model_manager.background_tasks, "MODEL_IDLE_TIMEOUT", test_timeout
            ),
        ):
            # Start unloader with short timeout
            await start_model_unloader()

            try:
                # Load model
                load_model("test-model")
                assert model_cache._loaded_model is not None

                # Simulate that the model was loaded long ago by setting an old timestamp
                # This ensures the unload condition will be met after the timeout
                old_time = model_manager.get_current_time() - (test_timeout + 1)
                model_cache._last_used_time = old_time

                # Wait for actual unload completion (with timeout)
                await asyncio.wait_for(
                    unload_completed.wait(), timeout=test_timeout + 2.0
                )

                # Now assert the model is unloaded
                assert model_cache._loaded_model is None
                assert model_cache._model_name is None

            finally:
                await stop_model_unloader()

    @patch("mlx_server_nano.model_manager.load")
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
            first_load_time = model_cache._last_used_time

            # Wait a bit
            await asyncio.sleep(1)

            # Load same model again (should hit cache)
            load_model("test-model")
            second_load_time = model_cache._last_used_time

            # Timer should be reset
            assert second_load_time > first_load_time
            assert model_cache._loaded_model is not None

        finally:
            await stop_model_unloader()


@pytest.mark.integration
@pytest.mark.model
class TestModelGenerationIntegration:
    """Integration tests for model generation with lifecycle management."""

    @patch("mlx_server_nano.model_manager.load")
    @patch("mlx_server_nano.model_manager.generation.generate")
    @patch("mlx_server_nano.model_manager.generation.stream_generate")
    @patch("mlx_server_nano.model_manager.generation.parse_tool_calls")
    def test_generate_response_complete_flow(
        self,
        mock_parse_tools,
        mock_stream_generate,
        mock_generate,
        mock_load,
        clean_model_manager,
    ):
        """Test complete response generation flow."""
        # Configure mocks
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template.return_value = "formatted prompt"
        mock_tokenizer.bos_token = "<bos>"  # Add proper bos_token
        mock_load.return_value = (mock_model, mock_tokenizer)
        mock_generate.return_value = "model response"
        mock_stream_generate.return_value = ["model response"]  # For fallback

        # Mock the new tool parsing
        mock_parse_tools.return_value = []  # No tool calls found

        # Generate response
        content, tool_calls = generate_response_with_tools(
            "test-model", [{"role": "user", "content": "hello"}], max_tokens=100
        )

        # Verify result - content should be the original response
        assert content == "model response"
        assert tool_calls == []

        # Verify model was loaded and cached
        assert model_cache._loaded_model == (mock_model, mock_tokenizer)
        assert model_cache._model_name == "test-model"
        assert model_cache._last_used_time > 0

        # Verify all steps were called
        mock_load.assert_called_once_with("test-model")
        # Should be called once now (we decode tokens for debugging)
        mock_tokenizer.apply_chat_template.assert_called_once()
        mock_generate.assert_called_once()
        mock_parse_tools.assert_called_once_with("model response", "test-model")

    @patch("mlx_server_nano.model_manager.load")
    @patch("mlx_server_nano.model_manager.generation.stream_generate")
    def test_generate_stream_complete_flow(
        self, mock_stream, mock_load, clean_model_manager
    ):
        """Test complete streaming generation flow."""
        # Configure mocks
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template.return_value = "formatted prompt"
        mock_tokenizer.bos_token = "<bos>"  # Add proper bos_token
        mock_load.return_value = (mock_model, mock_tokenizer)

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

        # Verify result - expecting tuples with finish_reason
        expected_chunks = [("Hello", None), (" world", None), ("", "stop")]
        assert chunks == expected_chunks

        # Verify model state
        assert model_cache._loaded_model == (mock_model, mock_tokenizer)
        assert model_cache._last_used_time > 0

    @patch("mlx_server_nano.model_manager.load")
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

    @patch("mlx_server_nano.model_manager.load")
    def test_different_models_replace_cache(self, mock_load, clean_model_manager):
        """Test that loading different models replaces cache."""
        # Configure mocks for different models
        model1, tokenizer1 = MagicMock(), MagicMock()
        model2, tokenizer2 = MagicMock(), MagicMock()

        mock_load.side_effect = [(model1, tokenizer1), (model2, tokenizer2)]

        # Load different models
        result1 = load_model("model-1")
        assert model_cache._model_name == "model-1"

        result2 = load_model("model-2")
        assert model_cache._model_name == "model-2"

        # Should load both models
        assert mock_load.call_count == 2
        assert result1 != result2


@pytest.mark.integration
@pytest.mark.model
@pytest.mark.memory
class TestMemoryManagement:
    """Integration tests for memory management functionality."""

    @patch("mlx_server_nano.model_manager.load")
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
        assert model_cache._loaded_model is not None

        # Manually trigger unload (normally done by background task)
        model_manager.cache.unload_model()

        # Verify state is cleared
        assert model_cache._loaded_model is None
        assert model_cache._model_name is None

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

    @patch("mlx_server_nano.model_manager.load")
    async def test_concurrent_model_loading(self, mock_load, clean_model_manager):
        """Test concurrent requests for same model."""
        # Configure mock with delay to simulate loading time
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        def slow_load(model_name):
            import time

            time.sleep(0.1)  # Simulate loading time with blocking sleep
            return (mock_model, mock_tokenizer)

        mock_load.side_effect = slow_load

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

    @patch("mlx_server_nano.model_manager.load")
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

    @patch("mlx_server_nano.model_manager.load")
    def test_model_specific_configurations(self, mock_load, clean_model_manager):
        """Test that model-specific configurations are applied."""
        # Configure mock
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_load.return_value = (mock_model, mock_tokenizer)

        # Test model configuration
        from mlx_server_nano.model_manager import _setup_generation_kwargs

        qwen_kwargs = _setup_generation_kwargs("qwen-test-model", max_tokens=100)
        assert qwen_kwargs["max_tokens"] == 100
        # Ensure no stop_strings in generation kwargs (MLX-LM handles them automatically)
        assert "stop_strings" not in qwen_kwargs

        # Test non-Qwen model
        other_kwargs = _setup_generation_kwargs("gpt-4", max_tokens=200)
        assert other_kwargs["max_tokens"] == 200
        assert "stop_strings" not in other_kwargs

    def test_environment_variable_integration(self, test_env_vars):
        """Test that environment variables are properly integrated."""
        # test_env_vars fixture sets various MLX_* environment variables

        from mlx_server_nano.config import ServerConfig

        config = ServerConfig.from_env()

        # Should use test values
        assert config.model_idle_timeout == 2  # From test_config fixture
        assert config.default_max_tokens == 100
        assert config.log_level == "DEBUG"
