"""
Model Cache Manager

Principled cache management for MLX models using the CacheManager interface.
Handles model loading, unloading, and memory management.
"""

import threading
import time
from typing import Any, Dict, Optional, Tuple
import logging

from .base import CacheManager
from .model_cache import (
    load_model as _load_model,
    unload_model as _unload_model,
    get_loaded_model as _get_loaded_model,
    get_cache_state as _get_cache_state,
    update_last_used_time as _update_last_used_time,
)

logger = logging.getLogger(__name__)


class ModelCacheManager(CacheManager):
    """
    Cache manager for MLX models.

    Provides principled model caching with configurable memory management,
    idle timeouts, and thread-safe operations.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize model cache manager.

        Args:
            config: Dictionary with model cache settings:
                - model_idle_timeout: Timeout in seconds for model unloading
                - memory_management_enabled: Enable automatic memory cleanup
                - preload_models: List of models to preload
        """
        super().__init__(config)
        self._lock = threading.Lock()
        self.model_idle_timeout = config.get("model_idle_timeout", 300)
        self.memory_management_enabled = config.get("memory_management_enabled", True)
        self.preload_models = config.get("preload_models", [])

        # Initialize any preload models if configured
        if self.preload_models:
            self._preload_models()

    def _preload_models(self):
        """Preload models specified in configuration."""
        for model_name in self.preload_models:
            try:
                logger.info(f"Preloading model: {model_name}")
                self.get_cache(model_name)
            except Exception as e:
                logger.warning(f"Failed to preload model {model_name}: {e}")

    def get_cache(self, key: Optional[str] = None) -> Tuple[Any, Any]:
        """
        Get or load a model.

        Args:
            key: Name of the model to load

        Returns:
            Tuple of (model, tokenizer)
        """
        if not key:
            # Return currently loaded model if any
            loaded = _get_loaded_model()
            if loaded:
                return loaded
            raise ValueError("No model name provided and no model currently loaded")

        with self._lock:
            logger.debug(f"Requesting model: {key}")
            return _load_model(key)

    def save_cache(self, cache: Any, identifier: str) -> None:
        """
        Save model cache to disk.

        Note: MLX models are typically saved via HuggingFace cache.
        This implementation logs the action but doesn't perform disk I/O.
        """
        logger.info(
            f"Model cache save requested for {identifier} (handled by HF cache)"
        )

    def load_cache(self, identifier: str) -> Optional[Tuple[Any, Any]]:
        """
        Load model from cache/disk.

        Args:
            identifier: Model name to load

        Returns:
            Model tuple if successful, None otherwise
        """
        try:
            return self.get_cache(identifier)
        except Exception as e:
            logger.warning(f"Failed to load model cache for {identifier}: {e}")
            return None

    def clear_cache(self, identifier: Optional[str] = None) -> None:
        """
        Clear model cache.

        Args:
            identifier: Specific model to unload. If None, unload current model.
        """
        with self._lock:
            if identifier is None:
                # Unload currently loaded model
                _unload_model()
                logger.info("Cleared current model cache")
            else:
                # For now, we only support unloading the current model
                current_model_name, _ = _get_cache_state()
                if current_model_name == identifier:
                    _unload_model()
                    logger.info(f"Cleared model cache for {identifier}")
                else:
                    logger.warning(f"Model {identifier} is not currently loaded")

    def update_usage(self) -> None:
        """Update the last used time for cache management."""
        _update_last_used_time()

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get model cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        current_model_name, last_used_time = _get_cache_state()
        current_time = time.time()

        return {
            "current_model": current_model_name,
            "last_used_time": last_used_time,
            "idle_time_seconds": current_time - last_used_time if last_used_time else 0,
            "model_idle_timeout": self.model_idle_timeout,
            "memory_management_enabled": self.memory_management_enabled,
            "is_loaded": current_model_name is not None,
        }
