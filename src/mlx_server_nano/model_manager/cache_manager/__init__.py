"""
Cache Manager Package

Provides unified cache management for MLX Server Nano with separate
handling for model caching and prompt/conversation caching.

Modules:
- base: Abstract base class for cache managers
- model_cache: Handles model loading, unloading, and memory management
- prompt_cache: Manages conversation-level prompt caching with MLX-LM
- model_cache_manager: Principled model cache management
- prompt_cache_manager: Principled prompt cache management
"""

# Import the original functions for backward compatibility
from .model_cache import (
    get_current_time,
    load_model,
    unload_model,
    update_last_used_time,
    get_cache_state,
    get_loaded_model,
)

from .prompt_cache import (
    ConversationState,
    detect_conversation_continuation,
    generate_conversation_hash,
    get_or_create_conversation_state,
    cleanup_expired_conversations,
    get_conversation_cache_stats,
)

# Import the new principled cache managers
from .base import CacheManager
from .model_cache_manager import ModelCacheManager
from .prompt_cache_manager import PromptCacheManager


__all__ = [
    # Model cache functions (backward compatibility)
    "get_current_time",
    "load_model",
    "unload_model",
    "update_last_used_time",
    "get_cache_state",
    "get_loaded_model",
    # Prompt cache functions and classes (backward compatibility)
    "ConversationState",
    "detect_conversation_continuation",
    "generate_conversation_hash",
    "get_or_create_conversation_state",
    "cleanup_expired_conversations",
    "get_conversation_cache_stats",
    # New principled cache managers
    "CacheManager",
    "ModelCacheManager",
    "PromptCacheManager",
]
