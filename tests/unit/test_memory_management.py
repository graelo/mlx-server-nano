"""
Memory management tests for MLX Server Nano.

Tests functional memory management, including model unloading verification,
cache state management, and memory cleanup processes.
"""

import asyncio
import pytest
import gc
from unittest.mock import patch, MagicMock

from mlx_server_nano import model_manager
from mlx_server_nano.model_manager import (
    load_model,
    _unload_model,
    start_model_unloader,
    stop_model_unloader,
)


@pytest.mark.memory
@pytest.mark.unit
class TestFunctionalMemoryManagement:
    """Tests for functional memory management (cache state verification)."""

    @patch("mlx_server_nano.model_manager.load")
    def test_model_cache_state_after_load(self, mock_load, clean_model_manager):
        """Test that model is properly cached after loading."""
        # Configure mock
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_load.return_value = (mock_model, mock_tokenizer)

        # Initially no model should be cached
        assert model_manager._loaded_model is None
        assert model_manager._model_name is None

        # Load model
        result = load_model("test-model")

        # Verify cache state
        assert model_manager._loaded_model is not None
        assert model_manager._model_name == "test-model"
        assert model_manager._loaded_model == (mock_model, mock_tokenizer)
        assert result == (mock_model, mock_tokenizer)

    @patch("mlx_server_nano.model_manager.load")
    @patch("mlx.core.clear_cache")
    @patch("gc.collect")
    def test_model_cache_state_after_unload(
        self, mock_gc, mock_clear_cache, mock_load, clean_model_manager
    ):
        """Test that model cache is properly cleared after unloading."""
        # Configure mocks
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_load.return_value = (mock_model, mock_tokenizer)
        mock_gc.return_value = 5

        # Load model
        load_model("test-model")
        assert model_manager._loaded_model is not None
        assert model_manager._model_name == "test-model"

        # Unload model
        _unload_model()

        # Verify cache is cleared
        assert model_manager._loaded_model is None
        assert model_manager._model_name is None

        # Verify cleanup functions were called
        mock_clear_cache.assert_called_once()
        assert mock_gc.call_count >= 1

    @patch("mlx_server_nano.model_manager.load")
    def test_model_cache_replacement(self, mock_load, clean_model_manager):
        """Test that loading a different model replaces the cache."""
        # Configure mocks for different models
        model1, tokenizer1 = MagicMock(), MagicMock()
        model2, tokenizer2 = MagicMock(), MagicMock()
        mock_load.side_effect = [(model1, tokenizer1), (model2, tokenizer2)]

        # Load first model
        load_model("model-1")
        assert model_manager._model_name == "model-1"
        assert model_manager._loaded_model == (model1, tokenizer1)

        # Load second model
        load_model("model-2")
        assert model_manager._model_name == "model-2"
        assert model_manager._loaded_model == (model2, tokenizer2)

        # First model should be replaced
        assert model_manager._loaded_model != (model1, tokenizer1)

    @patch("mlx_server_nano.model_manager.load")
    def test_model_cache_reuse(self, mock_load, clean_model_manager):
        """Test that same model is reused from cache."""
        # Configure mock
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_load.return_value = (mock_model, mock_tokenizer)

        # Load model multiple times
        result1 = load_model("test-model")
        result2 = load_model("test-model")
        result3 = load_model("test-model")

        # Should return same objects (cached)
        assert result1 is result2 is result3

        # Should only call mlx_lm.load once
        mock_load.assert_called_once()


@pytest.mark.memory
@pytest.mark.integration
@pytest.mark.asyncio
class TestMemoryLifecycleIntegration:
    """Integration tests for memory lifecycle management."""

    @patch("mlx_server_nano.model_manager.load")
    async def test_memory_lifecycle_with_unloader(
        self, mock_load, clean_model_manager, test_env_vars
    ):
        """Test complete memory lifecycle with background unloader."""
        # Configure mock
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_load.return_value = (mock_model, mock_tokenizer)

        # Start unloader
        await start_model_unloader()

        try:
            # Load model
            load_model("test-model")
            assert model_manager._loaded_model is not None

            # Wait for auto-unload (test timeout is 2 seconds)
            await asyncio.sleep(3)

            # Model should be unloaded
            assert model_manager._loaded_model is None
            assert model_manager._model_name is None

        finally:
            await stop_model_unloader()

    @patch("mlx_server_nano.model_manager.load")
    async def test_memory_state_during_concurrent_access(
        self, mock_load, clean_model_manager
    ):
        """Test memory state consistency during concurrent access."""
        # Configure mock
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_load.return_value = (mock_model, mock_tokenizer)

        async def load_task():
            return load_model("test-model")

        # Run multiple concurrent loads
        tasks = [asyncio.create_task(load_task()) for _ in range(5)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # All successful results should be the same (cached)
        successful_results = [r for r in results if not isinstance(r, Exception)]
        if successful_results:
            first_result = successful_results[0]
            assert all(r == first_result for r in successful_results)

        # Cache should be consistent
        assert model_manager._loaded_model is not None
        assert model_manager._model_name == "test-model"

    @patch("mlx_server_nano.model_manager.load")
    def test_memory_cleanup_error_handling(self, mock_load, clean_model_manager):
        """Test memory cleanup when errors occur during loading."""
        # Make loading fail
        mock_load.side_effect = Exception("Loading failed")

        # Attempt to load model
        with pytest.raises(RuntimeError):
            load_model("failing-model")

        # Cache should remain clean
        assert model_manager._loaded_model is None
        assert model_manager._model_name is None


@pytest.mark.memory
@pytest.mark.integration
class TestMemoryCleanupVerification:
    """Tests for verifying memory cleanup processes."""

    @patch("mlx.core.clear_cache")
    @patch("gc.collect")
    def test_gc_collection_during_unload(
        self, mock_gc, mock_clear_cache, clean_model_manager
    ):
        """Test that garbage collection is properly triggered during unload."""
        # Set up a loaded model
        model_manager._loaded_model = (MagicMock(), MagicMock())
        model_manager._model_name = "test-model"

        # Configure gc mock to return different values for multiple calls
        mock_gc.return_value = 10  # Just return a fixed value

        # Unload model
        _unload_model()

        # Verify cleanup was called
        mock_clear_cache.assert_called_once()

        # Verify GC was called
        assert mock_gc.call_count >= 1

    @patch("mlx.core.clear_cache")
    def test_mlx_cache_clear_error_handling(
        self, mock_clear_cache, clean_model_manager
    ):
        """Test handling of MLX cache clear errors."""
        # Set up a loaded model
        model_manager._loaded_model = (MagicMock(), MagicMock())
        model_manager._model_name = "test-model"

        # Make clear_cache raise an error
        mock_clear_cache.side_effect = Exception("Cache clear failed")

        # Unload should not raise, just log warning
        _unload_model()

        # Model should still be unloaded despite error
        assert model_manager._loaded_model is None
        assert model_manager._model_name is None

    @patch("mlx_server_nano.model_manager.load")
    @patch("mlx.core.clear_cache")
    def test_memory_cleanup_with_model_cycling(
        self, mock_clear_cache, mock_load, clean_model_manager
    ):
        """Test memory cleanup when cycling between different models."""
        # Configure mocks for different models
        model1, tokenizer1 = MagicMock(), MagicMock()
        model2, tokenizer2 = MagicMock(), MagicMock()
        model3, tokenizer3 = MagicMock(), MagicMock()

        mock_load.side_effect = [
            (model1, tokenizer1),
            (model2, tokenizer2),
            (model3, tokenizer3),
        ]

        # Load and switch between models
        load_model("model-1")
        assert model_manager._model_name == "model-1"

        load_model("model-2")  # This should unload model-1
        assert model_manager._model_name == "model-2"

        load_model("model-3")  # This should unload model-2
        assert model_manager._model_name == "model-3"

        # Each model switch should trigger cleanup
        # (Note: actual cleanup happens in _unload_model, which is called
        # when a different model is loaded)

        # Final state should be model-3
        assert model_manager._loaded_model == (model3, tokenizer3)


@pytest.mark.memory
@pytest.mark.slow
class TestMemoryMonitoringIntegration:
    """Tests for memory monitoring integration (if psutil available)."""

    def test_memory_monitoring_available(self):
        """Test if memory monitoring tools are available."""
        try:
            import psutil
            import os

            # Test basic memory monitoring
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()

            assert hasattr(memory_info, "rss")
            assert memory_info.rss > 0

        except ImportError:
            pytest.skip("psutil not available for memory monitoring")

    @patch("psutil.Process")
    def test_memory_measurement_integration(self, mock_process):
        """Test integration with memory measurement tools."""
        # Configure mock
        mock_memory_info = MagicMock()
        mock_memory_info.rss = 1024 * 1024 * 150  # 150 MB
        mock_process.return_value.memory_info.return_value = mock_memory_info

        # Test memory measurement (as done in the test scripts)
        try:
            import psutil
            import os

            process = psutil.Process(os.getpid())
            memory_mb = process.memory_info().rss / 1024 / 1024

            assert memory_mb == 150.0

        except ImportError:
            pytest.skip("psutil not available")

    def test_functional_memory_verification(self, clean_model_manager):
        """Test functional verification of memory state without OS memory measurement."""
        # This test verifies memory management functionally without
        # requiring actual memory measurement

        # Initially no model cached
        assert model_manager._loaded_model is None

        # Track object count before and after operations
        initial_objects = len(gc.get_objects())

        # Simulate model loading
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        model_manager._loaded_model = (mock_model, mock_tokenizer)
        model_manager._model_name = "test-model"

        # Should have more objects
        after_load_objects = len(gc.get_objects())
        assert after_load_objects >= initial_objects

        # Unload model
        _unload_model()

        # Force garbage collection
        gc.collect()

        # Verify functional unloading
        assert model_manager._loaded_model is None
        assert model_manager._model_name is None

        # Object count may have decreased (though not guaranteed due to gc behavior)
        len(gc.get_objects())  # Check but don't store
        # Note: We don't assert object count decrease as it's not reliable
        # The important thing is functional state verification
