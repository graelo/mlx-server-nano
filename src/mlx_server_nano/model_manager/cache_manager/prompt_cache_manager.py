"""
Prompt Cache Manager

Principled cache management for MLX prompt/KV caches using the CacheManager interface.
Supports different cache types like KVCache, QuantizedKVCache, RotatingKVCache, and more.
"""

import os
import time
from typing import Any, Dict, Optional, List
import logging

from .base import CacheManager
from .prompt_cache import (
    ConversationState,
    get_or_create_conversation_state as _get_or_create_conversation_state,
    cleanup_expired_conversations as _cleanup_expired_conversations,
    get_conversation_cache_stats as _get_conversation_cache_stats,
    _conversation_states,
    _conversation_lock,
)
from ...schemas import Message

logger = logging.getLogger(__name__)


class PromptCacheManager(CacheManager):
    """
    Cache manager for MLX prompt/KV caches.

    Supports different cache types and provides conversation-level caching
    with automatic detection and management.

    Available cache types:
    - KVCache: Standard key-value cache
    - QuantizedKVCache: Memory-efficient quantized cache
    - RotatingKVCache: Fixed-size rotating cache
    - ChunkedKVCache: Chunked processing cache
    - ConcatenateKVCache: Concatenated cache for multiple inputs
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize prompt cache manager with configuration.

        Args:
            config: Dictionary with cache settings:
                - cache_type: "KVCache", "QuantizedKVCache", "RotatingKVCache",
                             "ChunkedKVCache", "ConcatenateKVCache"
                - quantization_bits: For quantized caches (default: 8)
                - quantization_group_size: For quantized caches (default: 64)
                - max_tokens: Max tokens for rotating caches (default: 4096)
                - chunk_size: For chunked caches (default: 512)
                - cache_persistence_dir: Directory for disk persistence
                - conversation_idle_timeout: Timeout for conversation expiry
                - max_conversations: Maximum number of cached conversations
                - auto_detect_conversations: Enable conversation detection
        """
        super().__init__(config)
        self.caches = {}  # conversation_id -> cache object
        self.cache_type = config.get("cache_type", "KVCache")
        self.quantization_bits = config.get("quantization_bits", 8)
        self.quantization_group_size = config.get("quantization_group_size", 64)
        self.max_tokens = config.get("max_tokens", 4096)
        self.chunk_size = config.get("chunk_size", 512)
        self.cache_persistence_dir = os.path.expanduser(
            config.get("cache_persistence_dir", "~/.cache/mlx-server-nano")
        )
        self.conversation_idle_timeout = config.get("conversation_idle_timeout", 300)
        self.max_conversations = config.get("max_conversations", 10)
        self.auto_detect_conversations = config.get("auto_detect_conversations", True)

        # Ensure persistence directory exists
        os.makedirs(self.cache_persistence_dir, exist_ok=True)

        logger.info(f"Initialized PromptCacheManager with cache_type={self.cache_type}")

    def get_cache(self, key: Optional[str] = None) -> Any:
        """
        Get or create a cache for a conversation.

        Args:
            key: Optional conversation ID to retrieve existing cache

        Returns:
            Cache object (KVCache, QuantizedKVCache, RotatingKVCache, etc.)
        """
        if key and key in self.caches:
            logger.debug(f"Retrieved existing cache for conversation: {key}")
            return self.caches[key]

        cache = self._create_cache(self.cache_type)
        if key:
            self.caches[key] = cache
            logger.debug(f"Created new {self.cache_type} cache for conversation: {key}")

        return cache

    def _create_cache(self, cache_type: str) -> Any:
        """
        Create a cache instance based on type.

        Args:
            cache_type: Type of cache to create

        Returns:
            Instantiated cache object
        """
        try:
            if cache_type == "KVCache":
                return self._create_kv_cache()
            elif cache_type == "QuantizedKVCache":
                return self._create_quantized_kv_cache()
            elif cache_type == "RotatingKVCache":
                return self._create_rotating_kv_cache()
            elif cache_type == "ChunkedKVCache":
                return self._create_chunked_kv_cache()
            elif cache_type == "ConcatenateKVCache":
                return self._create_concatenate_kv_cache()
            else:
                logger.warning(
                    f"Unknown cache type: {cache_type}, falling back to KVCache"
                )
                return self._create_kv_cache()

        except Exception as e:
            logger.warning(f"Failed to create {cache_type} cache: {e}, using fallback")
            return []  # Return empty list as fallback

    def _create_kv_cache(self) -> Any:
        """Create a standard KVCache."""
        try:
            from mlx_lm.models.cache import KVCache

            cache = KVCache()
            logger.debug("Created KVCache")
            return cache
        except ImportError:
            logger.warning("MLX-LM KVCache not available")
            return []

    def _create_quantized_kv_cache(self) -> Any:
        """Create a QuantizedKVCache with specified quantization settings."""
        try:
            from mlx_lm.models.cache import QuantizedKVCache

            cache = QuantizedKVCache(
                bits=self.quantization_bits, group_size=self.quantization_group_size
            )
            logger.debug(
                f"Created QuantizedKVCache with {self.quantization_bits} bits, "
                f"group_size {self.quantization_group_size}"
            )
            return cache
        except ImportError:
            logger.warning("MLX-LM QuantizedKVCache not available")
            return []

    def _create_rotating_kv_cache(self) -> Any:
        """Create a RotatingKVCache with specified max tokens."""
        try:
            from mlx_lm.models.cache import RotatingKVCache

            cache = RotatingKVCache(max_size=self.max_tokens)
            logger.debug(f"Created RotatingKVCache with max_size {self.max_tokens}")
            return cache
        except ImportError:
            logger.warning("MLX-LM RotatingKVCache not available")
            return []

    def _create_chunked_kv_cache(self) -> Any:
        """Create a ChunkedKVCache with specified chunk size."""
        try:
            from mlx_lm.models.cache import ChunkedKVCache

            cache = ChunkedKVCache(chunk_size=self.chunk_size)
            logger.debug(f"Created ChunkedKVCache with chunk_size {self.chunk_size}")
            return cache
        except ImportError:
            logger.warning("MLX-LM ChunkedKVCache not available")
            return []

    def _create_concatenate_kv_cache(self) -> Any:
        """Create a ConcatenateKVCache for multiple input scenarios."""
        try:
            from mlx_lm.models.cache import ConcatenateKVCache

            cache = ConcatenateKVCache()
            logger.debug("Created ConcatenateKVCache")
            return cache
        except ImportError:
            logger.warning("MLX-LM ConcatenateKVCache not available")
            return []

    def get_conversation_cache(
        self,
        conversation_id: Optional[str],
        model_name: str,
        messages: List[Message],
        model=None,
    ) -> ConversationState:
        """
        Get or create conversation state with cache.

        This bridges the new cache manager with the existing conversation logic.

        Args:
            conversation_id: Optional conversation ID
            model_name: Model name for this conversation
            messages: Current messages
            model: Model instance for cache creation

        Returns:
            ConversationState with appropriate cache
        """
        return _get_or_create_conversation_state(
            conversation_id, model_name, messages, model
        )

    def save_cache(self, cache: Any, identifier: str) -> None:
        """
        Save cache to disk using MLX-LM's save_prompt_cache function.

        Args:
            cache: Cache object to save
            identifier: Unique identifier for the cache file
        """
        try:
            cache_path = os.path.join(self.cache_persistence_dir, f"{identifier}.cache")

            # Use MLX-LM's official save function
            from mlx_lm.models.cache import save_prompt_cache

            save_prompt_cache(cache_path, cache)
            logger.info(f"Saved {type(cache).__name__} to {cache_path}")

        except ImportError:
            logger.warning("MLX-LM save_prompt_cache not available")
        except Exception as e:
            logger.error(f"Failed to save cache {identifier}: {e}")

    def load_cache(self, identifier: str) -> Optional[Any]:
        """
        Load cache from disk using MLX-LM's load_prompt_cache function.

        Args:
            identifier: Unique identifier for the cache file

        Returns:
            Loaded cache object or None if loading fails
        """
        try:
            cache_path = os.path.join(self.cache_persistence_dir, f"{identifier}.cache")

            if not os.path.exists(cache_path):
                logger.debug(f"Cache file does not exist: {cache_path}")
                return None

            # Use MLX-LM's official load function
            from mlx_lm.models.cache import load_prompt_cache

            cache = load_prompt_cache(cache_path)
            logger.info(f"Loaded cache from {cache_path}")
            return cache

        except ImportError:
            logger.warning("MLX-LM load_prompt_cache not available")
            return None
        except Exception as e:
            logger.error(f"Failed to load cache {identifier}: {e}")
            return None

    def trim_cache(self, cache: Any, max_tokens: int) -> None:
        """Trim cache to specified token limit."""
        try:
            # For RotatingKVCache
            if hasattr(cache, "trim"):
                cache.trim(max_tokens)
                logger.debug(f"Trimmed cache to {max_tokens} tokens")
            # For other cache types, use MLX-LM trimming if available
            elif hasattr(cache, "__len__") and len(cache) > 0:
                try:
                    from mlx_lm.models.cache import (
                        can_trim_prompt_cache,
                        trim_prompt_cache,
                    )

                    if can_trim_prompt_cache(cache):
                        # Calculate how much to trim
                        if hasattr(cache[0], "keys") and cache[0].keys is not None:
                            current_size = (
                                cache[0].keys.shape[2]
                                if len(cache[0].keys.shape) > 2
                                else 0
                            )
                            if current_size > max_tokens:
                                tokens_to_trim = current_size - max_tokens
                                trim_prompt_cache(cache, tokens_to_trim)
                                logger.debug(
                                    f"Trimmed cache from {current_size} to {max_tokens} tokens"
                                )
                except ImportError:
                    logger.debug("MLX-LM cache trimming not available")
            else:
                logger.debug("Cache type does not support trimming")

        except Exception as e:
            logger.warning(f"Failed to trim cache: {e}")

    def clear_cache(self, identifier: Optional[str] = None) -> None:
        """
        Clear cache(s).

        Args:
            identifier: Optional specific conversation to clear. If None, clear all.
        """
        if identifier is None:
            # Clear all caches
            self.caches.clear()
            _cleanup_expired_conversations()
            logger.info("Cleared all prompt caches")
        else:
            # Clear specific cache
            if identifier in self.caches:
                del self.caches[identifier]

            # Also clear from conversation states
            with _conversation_lock:
                if identifier in _conversation_states:
                    del _conversation_states[identifier]

            logger.info(f"Cleared cache for conversation: {identifier}")

    def cleanup_expired_caches(self) -> None:
        """Remove expired conversation caches."""
        _cleanup_expired_conversations()

        # Also clean up our internal cache mapping
        current_time = time.time()
        expired_ids = []

        for conv_id in self.caches.keys():
            # Check if conversation state exists and is expired
            with _conversation_lock:
                if conv_id in _conversation_states:
                    conv_state = _conversation_states[conv_id]
                    if (
                        current_time - conv_state.last_used
                    ) > self.conversation_idle_timeout:
                        expired_ids.append(conv_id)
                else:
                    # Conversation state doesn't exist, remove cache
                    expired_ids.append(conv_id)

        for conv_id in expired_ids:
            if conv_id in self.caches:
                del self.caches[conv_id]

        if expired_ids:
            logger.info(f"Cleaned up {len(expired_ids)} expired caches")

    def get_cache_info(self) -> Dict[str, Any]:
        """
        Get information about current cache state.

        Returns:
            Dictionary with cache statistics and information
        """
        info = {
            "cache_type": self.cache_type,
            "active_conversations": len(self.caches),
            "conversation_stats": _get_conversation_cache_stats(),
            "config": {
                "quantization_bits": self.quantization_bits,
                "quantization_group_size": self.quantization_group_size,
                "max_tokens": self.max_tokens,
                "chunk_size": self.chunk_size,
                "max_conversations": self.max_conversations,
                "conversation_idle_timeout": self.conversation_idle_timeout,
            },
        }

        # Add cache-specific information
        cache_sizes = {}
        for conv_id, cache in self.caches.items():
            try:
                if hasattr(cache, "__len__") and len(cache) > 0:
                    if hasattr(cache[0], "keys") and cache[0].keys is not None:
                        size = (
                            cache[0].keys.shape[2]
                            if len(cache[0].keys.shape) > 2
                            else 0
                        )
                        cache_sizes[conv_id] = size
                    else:
                        cache_sizes[conv_id] = len(cache)
                else:
                    cache_sizes[conv_id] = 0
            except Exception:
                cache_sizes[conv_id] = "unknown"

        info["cache_sizes"] = cache_sizes
        return info

    def optimize_caches(self) -> None:
        """
        Optimize all active caches by trimming and cleanup.
        """
        logger.info("Starting cache optimization...")

        # First cleanup expired caches
        self.cleanup_expired_caches()

        # Then optimize remaining caches
        optimized_count = 0
        for conv_id, cache in self.caches.items():
            try:
                self._optimize_single_cache(cache, self.max_tokens)
                optimized_count += 1
            except Exception as e:
                logger.warning(f"Failed to optimize cache for {conv_id}: {e}")

        logger.info(f"Optimized {optimized_count} caches")

    def _optimize_single_cache(self, cache: Any, max_tokens: int) -> None:
        """
        Optimize a single cache instance.

        Args:
            cache: Cache object to optimize
            max_tokens: Maximum tokens to keep in cache
        """
        try:
            if isinstance(cache, list) and len(cache) > 0:
                from mlx_lm.models.cache import (
                    can_trim_prompt_cache,
                    trim_prompt_cache,
                )

                if can_trim_prompt_cache(cache):
                    # Calculate how much to trim
                    if hasattr(cache[0], "keys") and cache[0].keys is not None:
                        current_size = (
                            cache[0].keys.shape[2]
                            if len(cache[0].keys.shape) > 2
                            else 0
                        )
                        if current_size > max_tokens:
                            tokens_to_trim = current_size - max_tokens
                            trim_prompt_cache(cache, tokens_to_trim)
                            logger.debug(
                                f"Trimmed cache from {current_size} to {max_tokens} tokens"
                            )
        except ImportError:
            logger.debug("MLX-LM cache trimming not available")
        except Exception as e:
            logger.warning(f"Failed to trim single cache: {e}")

    def switch_cache_type(self, new_cache_type: str) -> None:
        """
        Switch to a different cache type. This will clear existing caches.

        Args:
            new_cache_type: New cache type to use
        """
        logger.info(f"Switching cache type from {self.cache_type} to {new_cache_type}")

        # Clear existing caches since they may not be compatible
        self.clear_cache()

        # Update cache type
        self.cache_type = new_cache_type

        logger.info(f"Successfully switched to {new_cache_type}")

    def get_cache_recommendations(self) -> Dict[str, str]:
        """
        Get recommendations for optimal cache configuration based on usage patterns.

        Returns:
            Dictionary with recommendations
        """
        recommendations = {}
        stats = _get_conversation_cache_stats()

        # Analyze conversation patterns
        if stats["total_conversations"] > 0:
            avg_messages = stats["total_conversations"] / max(
                1, len(_conversation_states)
            )

            if avg_messages > 20:
                recommendations["cache_type"] = (
                    "Consider RotatingKVCache for long conversations to manage memory"
                )
            elif stats["total_conversations"] > 50:
                recommendations["cache_type"] = (
                    "Consider QuantizedKVCache to reduce memory usage with many conversations"
                )
            elif avg_messages < 5:
                recommendations["cache_type"] = (
                    "Standard KVCache is optimal for short conversations"
                )

        # Memory-based recommendations
        active_caches = len(self.caches)
        if active_caches > self.max_conversations * 0.8:
            recommendations["memory"] = (
                "Consider increasing conversation_idle_timeout or max_conversations"
            )

        # Performance recommendations
        if self.cache_type == "QuantizedKVCache" and self.quantization_bits > 8:
            recommendations["performance"] = (
                "Consider reducing quantization_bits to 8 for better performance"
            )

        return recommendations

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get statistics about cache usage.

        Returns:
            Dictionary with cache statistics
        """
        conversation_stats = _get_conversation_cache_stats()

        return {
            **conversation_stats,
            "cache_type": self.cache_type,
            "quantization_bits": self.quantization_bits,
            "max_tokens": self.max_tokens,
            "cache_persistence_dir": self.cache_persistence_dir,
            "internal_caches": len(self.caches),
            "max_conversations": self.max_conversations,
        }
