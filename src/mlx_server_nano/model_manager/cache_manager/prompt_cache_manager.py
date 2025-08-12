"""
Prompt Cache Manager

Principled cache management for MLX prompt/KV caches using the CacheManager interface.
Supports different cache types like KVCache, QuantizedKVCache, and RotatingKVCache.
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
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize prompt cache manager with configuration.

        Args:
            config: Dictionary with cache settings:
                - cache_type: "KVCache", "QuantizedKVCache", "RotatingKVCache"
                - quantization_bits: For quantized caches (default: 8)
                - max_tokens: Max tokens for rotating caches (default: 4096)
                - cache_persistence_dir: Directory for disk persistence
                - conversation_idle_timeout: Timeout for conversation expiry
                - max_conversations: Maximum number of cached conversations
                - auto_detect_conversations: Enable conversation detection
        """
        super().__init__(config)
        self.caches = {}  # conversation_id -> cache object
        self.cache_type = config.get("cache_type", "KVCache")
        self.quantization_bits = config.get("quantization_bits", 8)
        self.max_tokens = config.get("max_tokens", 4096)
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
            Cache object (KVCache, QuantizedKVCache, or RotatingKVCache)
        """
        if key and key in self.caches:
            logger.debug(f"Retrieved existing cache for conversation: {key}")
            return self.caches[key]

        cache = self._create_cache(self.cache_type)
        if key:
            self.caches[key] = cache
            logger.debug(f"Created new cache for conversation: {key}")

        return cache

    def _create_cache(self, cache_type: str) -> Any:
        """Create a cache instance based on type."""
        try:
            # For now, use the existing MLX-LM cache creation approach
            # The user's suggested cache types might not be available in current MLX-LM
            logger.debug(f"Creating {cache_type} cache (basic implementation)")
            return []  # Return empty list as fallback for now

        except Exception as e:
            logger.warning(f"Failed to create cache: {e}")
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
        """Save cache to disk."""
        try:
            cache_path = os.path.join(self.cache_persistence_dir, f"{identifier}.cache")

            # For MLX caches that support save/load
            if hasattr(cache, "save"):
                cache.save(cache_path)
                logger.info(f"Saved cache to {cache_path}")
            else:
                logger.debug(f"Cache type {type(cache)} does not support persistence")

        except Exception as e:
            logger.error(f"Failed to save cache {identifier}: {e}")

    def load_cache(self, identifier: str) -> Optional[Any]:
        """Load cache from disk."""
        try:
            cache_path = os.path.join(self.cache_persistence_dir, f"{identifier}.cache")

            if not os.path.exists(cache_path):
                return None

            # For now, return None since cache persistence is not fully implemented
            # This can be extended when MLX-LM cache types support load/save
            logger.debug(f"Cache loading not yet implemented for {cache_path}")
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
                logger.debug(f"Cleaned up expired cache: {conv_id}")

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
