"""
Prompt/Conversation Cache Management

Handles conversation-level prompt caching using MLX-LM's built-in KV cache capabilities.
Provides conversation detection, cache management, and memory optimization.

Features:
- Conversation-level prompt caching using MLX-LM's built-in capabilities
- Automatic conversation detection and cache management
- Cache trimming for optimal performance
- Thread-safe conversation state management
"""

import hashlib
import logging
import threading
import time
from typing import Optional, Dict, Any, List

from ...config import config
from ...schemas import Message

# Set up logging
logger = logging.getLogger(__name__)


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

            # Based on performance testing, optimal cache size for conversation extensions
            # is around 256-512 tokens. Larger caches hurt performance due to overhead.
            # Use None (default) which seems to work best for small conversations
            cache = make_prompt_cache(model, max_kv_size=None)
            logger.debug(
                f"Created MLX-LM prompt cache with default size: {type(cache)}"
            )
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

        # Trim cache if it's getting too large for optimal performance
        self._trim_cache_if_needed()

    def _trim_cache_if_needed(self):
        """Trim cache if it exceeds optimal size for performance."""
        if not self.prompt_cache or len(self.prompt_cache) == 0:
            return

        try:
            from mlx_lm.models.cache import can_trim_prompt_cache, trim_prompt_cache

            # Check if cache can be trimmed and if it's getting large
            if can_trim_prompt_cache(self.prompt_cache):
                # Check current cache size
                if (
                    hasattr(self.prompt_cache[0], "keys")
                    and self.prompt_cache[0].keys is not None
                ):
                    current_size = (
                        self.prompt_cache[0].keys.shape[2]
                        if len(self.prompt_cache[0].keys.shape) > 2
                        else 0
                    )

                    # If cache is larger than 512 tokens, trim to 256 tokens for optimal performance
                    if current_size > 512:
                        tokens_to_trim = current_size - 256
                        logger.debug(
                            f"Trimming cache from {current_size} to 256 tokens (trimming {tokens_to_trim})"
                        )

                        # trim_prompt_cache modifies the cache in place and returns the trimmed cache
                        trim_prompt_cache(self.prompt_cache, tokens_to_trim)
                        # Check if the operation succeeded by checking cache size after
                        if (
                            hasattr(self.prompt_cache[0], "keys")
                            and self.prompt_cache[0].keys is not None
                        ):
                            new_size = (
                                self.prompt_cache[0].keys.shape[2]
                                if len(self.prompt_cache[0].keys.shape) > 2
                                else 0
                            )
                            actual_trimmed = current_size - new_size
                            if actual_trimmed > 0:
                                logger.info(
                                    f"Successfully trimmed {actual_trimmed} tokens from cache for conversation {self.conversation_id}"
                                )
                            else:
                                logger.debug(
                                    "Cache trim operation completed but may not have reduced size as expected"
                                )

        except ImportError:
            logger.debug("MLX-LM cache trimming not available")
        except Exception as e:
            logger.warning(f"Failed to trim cache: {e}")

    def should_expire(self) -> bool:
        """Check if this conversation should be expired based on idle time."""
        idle_time = time.time() - self.last_used
        return idle_time > config.conversation_idle_timeout

    def estimate_tokens(self) -> int:
        """Rough estimate of tokens used in this conversation."""
        if not hasattr(self, "message_history") or not self.message_history:
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
        if not hasattr(conv_state, "message_history") or not conv_state.message_history:
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
        model: Model instance for cache creation

    Returns:
        ConversationState object for this conversation
    """
    logger.debug(
        f"get_or_create_conversation_state called with conv_id={conversation_id}, model={model_name}"
    )

    with _conversation_lock:
        logger.debug("Acquired lock")

        # Try auto-detection if no explicit ID provided
        if not conversation_id and config.auto_detect_conversations:
            logger.debug("Attempting auto-detection")
            conversation_id = detect_conversation_continuation(
                messages, _conversation_states
            )
            logger.debug(f"Auto-detection result: {conversation_id}")

        # Fall back to content-based hash if still no ID
        if not conversation_id:
            logger.debug("Generating hash-based ID")
            conversation_id = f"auto_{generate_conversation_hash(messages)}"
            logger.debug(f"Generated ID: {conversation_id}")

        logger.debug(f"Final conversation_id: {conversation_id}")

        # Get existing or create new conversation state
        if conversation_id in _conversation_states:
            logger.debug("Found existing conversation")
            conv_state = _conversation_states[conversation_id]
            # Verify model matches
            if conv_state.model_name != model_name:
                logger.warning(
                    f"Model mismatch for conversation {conversation_id}: "
                    f"cached={conv_state.model_name}, requested={model_name}. Creating new conversation."
                )
                # Create new conversation with different ID
                conversation_id = f"{conversation_id}_{model_name}"
                logger.info(f"Model mismatch, new ID: {conversation_id}")
                # Note: this will fall through to the creation logic below

        # Create new conversation state if needed
        if conversation_id not in _conversation_states:
            logger.debug("Creating new conversation state")
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

        conv_state = _conversation_states[conversation_id]
        logger.debug("Updating usage")
        conv_state.update_usage(messages)

        logger.debug("Returning conversation state")
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


def get_conversation_cache_stats() -> Dict[str, Any]:
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
