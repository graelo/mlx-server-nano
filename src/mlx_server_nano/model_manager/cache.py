"""
Model Cache Management

Handles model loading, unloading, caching, and memory management using MLX framework.
Provides thread-safe model lifecycle management with automatic cleanup.
Now includes conversation-level prompt caching for enhanced performance.

Features:
- Model loading from Hugging Face Hub via MLX-LM
- Thread-safe model caching with global state
- Memory cleanup and garbage collection
- Model idle timeout configuration
- Conversation-level prompt caching using MLX-LM's built-in capabilities
- Automatic conversation detection and cache management
"""

import gc
import hashlib
import logging
import os
import threading
import time
from typing import Dict, List, Optional, Tuple

import mlx.core as mx

from ..config import config
from ..schemas import Message

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


def unload_model():
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


# =============================================================================
# Conversation-Level Prompt Caching
# =============================================================================


class ConversationState:
    """
    Manages conversation-level state and prompt caching.

    Uses MLX-LM's built-in prompt caching capabilities for enhanced performance
    in conversational scenarios.
    """

    def __init__(
        self, conversation_id: str, model_name: str, conversation_hash: str, model=None
    ):
        """Initialize conversation state."""
        self.conversation_id = conversation_id
        self.model_name = model_name
        self.conversation_hash = conversation_hash
        # Initialize prompt cache using MLX-LM's official function
        self.prompt_cache = self._create_prompt_cache(model)
        self.created_at = time.time()
        self.last_accessed = time.time()
        self.message_count = 0

    def _create_prompt_cache(self, model):
        """Create prompt cache using MLX-LM's official function."""
        if model is None:
            # If no model provided, return empty list as fallback
            return []

        try:
            from mlx_lm.models.cache import make_prompt_cache

            cache = make_prompt_cache(model)
            logger.debug(f"Created MLX-LM prompt cache: {type(cache)}")
            return cache
        except ImportError:
            logger.warning("MLX-LM cache module not available, using empty list")
            return []
        except Exception as e:
            logger.warning(
                f"Failed to create MLX-LM prompt cache: {e}, using empty list"
            )
            return []

    def update_usage(self, messages: List[Message]):
        """Update conversation state with new messages."""
        self.last_used = time.time()
        self.message_history = messages.copy()
        self.message_count = len(messages)

    def should_expire(self) -> bool:
        """Check if this conversation should be expired based on idle time."""
        idle_time = time.time() - self.last_used
        return idle_time > config.conversation_idle_timeout

    def estimate_tokens(self) -> int:
        """Rough estimate of tokens used in this conversation."""
        if not self.message_history:
            return 0
        # Very rough estimate: 4 chars per token
        total_chars = sum(len(msg.content or "") for msg in self.message_history)
        return total_chars // 4


# Global conversation cache state
_conversation_states: Dict[str, ConversationState] = {}
_conversation_lock = threading.Lock()


def detect_conversation_continuation(
    new_messages: List[Message], cached_conversations: Dict[str, ConversationState]
) -> Optional[str]:
    """
    Detect if new messages are a continuation of an existing conversation.

    Algorithm:
    1. Check if new messages start with the same sequence as any cached conversation
    2. For efficiency, compare up to the last N messages from cached conversations
    3. Return conversation ID if overlap ratio exceeds threshold

    Args:
        new_messages: New messages to check
        cached_conversations: Existing cached conversations

    Returns:
        Conversation ID if match found, None otherwise
    """
    if not config.auto_detect_conversations or not new_messages:
        return None

    new_content = [(msg.role, msg.content or "") for msg in new_messages]

    best_match_id = None
    best_overlap = 0.0

    for conv_id, conv_state in cached_conversations.items():
        if not conv_state.message_history:
            continue

        cached_content = [
            (msg.role, msg.content or "") for msg in conv_state.message_history
        ]

        # Check if new messages start with the same sequence as cached conversation
        # This handles the case where new_messages = cached_messages + additional_messages
        overlap_count = 0
        min_length = min(len(new_content), len(cached_content))

        for i in range(min_length):
            if new_content[i] == cached_content[i]:
                overlap_count += 1
            else:
                break

        # Calculate overlap ratio - how much of the cached conversation is matched
        if len(cached_content) > 0:
            overlap_ratio = overlap_count / len(cached_content)
        else:
            overlap_ratio = 0.0

        # Only consider this a continuation if:
        # 1. We match a significant portion of the cached conversation
        # 2. The new messages are longer (adding to the conversation)
        if (
            overlap_ratio > best_overlap
            and overlap_ratio >= config.conversation_detection_threshold
            and len(new_content)
            >= len(cached_content)  # New conversation should be longer
        ):
            best_overlap = overlap_ratio
            best_match_id = conv_id

    if best_match_id:
        logger.debug(
            f"Detected conversation continuation: {best_match_id} (overlap: {best_overlap:.2f})"
        )

    return best_match_id


def generate_conversation_hash(messages: List[Message]) -> str:
    """Generate stable hash from conversation content for cache key."""
    if not messages:
        return "empty"

    # Create deterministic hash from message content and roles
    content_parts = []
    for msg in messages[-3:]:  # Only use last 3 messages for hash
        content_parts.append(f"{msg.role}:{msg.content or ''}")

    content_str = "|".join(content_parts)
    return hashlib.sha256(content_str.encode()).hexdigest()[:16]


def get_or_create_conversation_state(
    conversation_id: Optional[str], model_name: str, messages: List[Message], model=None
) -> ConversationState:
    """
    Get existing conversation state or create new one.

    Args:
        conversation_id: Explicit conversation ID or None for auto-detection
        model_name: Model name for this conversation
        messages: Current messages for conversation detection

    Returns:
        ConversationState object for this conversation
    """
    print(
        f"[DEBUG] get_or_create_conversation_state called with conv_id={conversation_id}, model={model_name}"
    )

    with _conversation_lock:
        print(f"[DEBUG] Acquired lock")

        # Try auto-detection if no explicit ID provided
        if not conversation_id and config.auto_detect_conversations:
            print(f"[DEBUG] Attempting auto-detection")
            conversation_id = detect_conversation_continuation(
                messages, _conversation_states
            )
            print(f"[DEBUG] Auto-detection result: {conversation_id}")

        # Fall back to content-based hash if still no ID
        if not conversation_id:
            print(f"[DEBUG] Generating hash-based ID")
            conversation_id = f"auto_{generate_conversation_hash(messages)}"
            print(f"[DEBUG] Generated ID: {conversation_id}")

        print(f"[DEBUG] Final conversation_id: {conversation_id}")

        # Get existing or create new conversation state
        if conversation_id in _conversation_states:
            print(f"[DEBUG] Found existing conversation")
            conv_state = _conversation_states[conversation_id]
            # Verify model matches
            if conv_state.model_name != model_name:
                logger.warning(
                    f"Model mismatch for conversation {conversation_id}: "
                    f"cached={conv_state.model_name}, requested={model_name}. Creating new conversation."
                )
                # Create new conversation with different ID
                conversation_id = f"{conversation_id}_{model_name}"
                print(f"[DEBUG] Model mismatch, new ID: {conversation_id}")
                # Note: this will fall through to the creation logic below

        # Create new conversation state if needed
        if conversation_id not in _conversation_states:
            print("[DEBUG] Creating new conversation state")
            # Clean up expired conversations if we're at capacity
            _cleanup_expired_conversations_unsafe()

            # If still at capacity, remove oldest conversation
            if len(_conversation_states) >= config.max_conversations:
                oldest_id = min(
                    _conversation_states.keys(),
                    key=lambda k: _conversation_states[k].last_used,
                )
                del _conversation_states[oldest_id]
                logger.info(f"Removed oldest conversation {oldest_id} to make space")

            # Generate conversation hash for new conversation
            conversation_hash = generate_conversation_hash(messages)
            _conversation_states[conversation_id] = ConversationState(
                conversation_id, model_name, conversation_hash, model
            )
            logger.info(f"Created new conversation state: {conversation_id}")
            print(f"[DEBUG] Created new conversation: {conversation_id}")

        conv_state = _conversation_states[conversation_id]
        print(f"[DEBUG] Updating usage")
        conv_state.update_usage(messages)

        print(f"[DEBUG] Returning conversation state")
        return conv_state

        return conv_state


def cleanup_expired_conversations():
    """Remove expired conversations from cache."""
    with _conversation_lock:
        _cleanup_expired_conversations_unsafe()


def _cleanup_expired_conversations_unsafe():
    """Remove expired conversations from cache. Assumes lock is already held."""
    expired_ids = [
        conv_id
        for conv_id, conv_state in _conversation_states.items()
        if conv_state.should_expire()
    ]

    for conv_id in expired_ids:
        del _conversation_states[conv_id]
        logger.info(f"Expired conversation cache: {conv_id}")


def get_conversation_cache_stats() -> Dict[str, int | float | bool]:
    """Get statistics about conversation caching."""
    with _conversation_lock:
        total_conversations = len(_conversation_states)
        total_messages = sum(
            conv.message_count for conv in _conversation_states.values()
        )
        avg_age = 0

        if _conversation_states:
            current_time = time.time()
            avg_age = sum(
                current_time - conv.created_at for conv in _conversation_states.values()
            ) / len(_conversation_states)

        return {
            "total_conversations": total_conversations,
            "total_messages": total_messages,
            "average_age_seconds": avg_age,
            "cache_enabled": config.conversation_cache_enabled,
            "auto_detection_enabled": config.auto_detect_conversations,
        }
