"""
Model Cache Management

Handles model loading, unloading, caching, and memory management using MLX framework.
Provides thread-safe model lifecycle management with automatic cleanup.

Features:
- Model loading from Hugging Face Hub via MLX-LM
- Thread-safe model caching with global state
- Memory cleanup and garbage collection
- Model idle timeout configuration
"""

import gc
import logging
import threading
import time
from typing import Optional, Tuple

import mlx.core as mx

from ..config import config

# Set up logging
logger = logging.getLogger(__name__)

# Model cache configuration
MODEL_IDLE_TIMEOUT = config.model_idle_timeout

# Global model cache state
_loaded_model = None
_model_name = None
_last_used_time = 0
_lock = threading.Lock()


def get_current_time() -> float:
    """Get current timestamp. Separate function for easier testing."""
    return time.time()


def _unload_model():
    """Internal function to unload the current model and reset state with proper memory cleanup."""
    global _loaded_model, _model_name

    if _loaded_model is None:
        return  # Nothing to unload

    model_name = _model_name
    logger.info(f"Unloading model '{model_name}' and freeing memory")

    # Clear the model references first
    _loaded_model = None
    _model_name = None

    # Force garbage collection to release Python objects
    gc.collect()

    # Clear MLX memory cache/buffers using the new API
    try:
        # Use the new MLX API (replaces deprecated mx.metal.clear_cache)
        mx.clear_cache()
        logger.debug("Cleared MLX memory cache")

    except Exception as e:
        logger.warning(f"Could not clear MLX cache: {e}")

    # Multiple rounds of garbage collection for stubborn references
    for i in range(3):
        collected = gc.collect()
        if collected > 0:
            logger.debug(f"GC round {i + 1}: collected {collected} objects")

    # Force memory release by explicitly deleting any lingering references
    try:
        # Get all local variables that might hold model references
        import sys

        frame = sys._getframe()
        while frame:
            if frame.f_locals:
                for var_name, var_value in list(frame.f_locals.items()):
                    if (
                        hasattr(var_value, "__class__")
                        and "mlx" in str(type(var_value)).lower()
                    ):
                        try:
                            del frame.f_locals[var_name]
                        except Exception:
                            pass
            frame = frame.f_back
    except Exception:
        pass  # Best effort cleanup

    # Final garbage collection
    gc.collect()

    logger.info(f"Model '{model_name}' unloaded and memory freed")


def load_model(name: str):
    """
    Load model using Hugging Face Hub and MLX framework.

    Args:
        name: Model name/path from Hugging Face Hub

    Returns:
        Tuple of (model, tokenizer) from MLX

    Note:
        The mlx-lm library handles downloading from HF Hub automatically
        and uses the standard HF cache location (respects HF_HOME, HUGGINGFACE_HUB_CACHE, etc.)
    """
    global _loaded_model, _model_name, _last_used_time

    logger.info(f"Loading model: {name}")

    with _lock:
        _last_used_time = get_current_time()

        # Return cached model if already loaded
        if _model_name == name and _loaded_model:
            logger.info(f"Model '{name}' already loaded, reusing cached model")
            # Import here to avoid circular dependency
            from .background_tasks import _schedule_unload

            _schedule_unload()
            return _loaded_model

        logger.info(f"Loading model '{name}' from Hugging Face Hub...")

        try:
            logger.debug(f"Calling mlx_lm.load() with model name: {name}")
            # mlx-lm will automatically download from HF Hub and cache using HF's cache system
            # Use a qualified import to allow patching at the module level
            import mlx_server_nano.model_manager as model_manager_mod

            model, tokenizer = model_manager_mod.load(name)
            logger.info(f"Successfully loaded model '{name}'")

        except Exception as e:
            logger.error(
                f"Failed to load model '{name}': {type(e).__name__}: {str(e)}",
                exc_info=True,
            )
            raise RuntimeError(f"Failed to load model '{name}': {e}")

        _loaded_model = (model, tokenizer)
        _model_name = name

        # Import here to avoid circular dependency
        from .background_tasks import _schedule_unload

        _schedule_unload()
        logger.info(f"Model '{name}' loaded and cached successfully")
        return _loaded_model


def update_last_used_time():
    """Update the last used time for the cached model."""
    global _last_used_time
    with _lock:
        _last_used_time = get_current_time()


def get_cache_state() -> Tuple[Optional[str], float]:
    """Get current cache state for background task management."""
    with _lock:
        return _model_name, _last_used_time


def get_loaded_model():
    """Get the currently loaded model if any."""
    with _lock:
        return _loaded_model
