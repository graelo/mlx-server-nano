"""
Configuration management for MLX Server Nano

Handles server configuration with environment variable support and sensible defaults.
Configuration can be set via environment variables with MLX_ prefix or programmatically.

Environment Variables:
- MLX_SERVER_HOST: Server host address (default: 0.0.0.0)
- MLX_SERVER_PORT: Server port (default: 8000)
- MLX_MODEL_IDLE_TIMEOUT: Model idle timeout in seconds (default: 300)
- MLX_DEFAULT_MAX_TOKENS: Default max tokens for generation (default: 512)
- MLX_DEFAULT_TEMPERATURE: Default temperature for generation (default: 0.7)
- MLX_LOG_LEVEL: Logging level (default: INFO)

Conversation Caching Environment Variables:
- MLX_CONVERSATION_CACHE_ENABLED: Enable conversation caching (default: true)
- MLX_CONVERSATION_IDLE_TIMEOUT: Conversation idle timeout in seconds (default: 300)
- MLX_MAX_CONVERSATIONS: Maximum number of cached conversations (default: 10)
- MLX_CACHE_QUANTIZATION_ENABLED: Enable cache quantization (default: true)
- MLX_CACHE_QUANTIZATION_BITS: Quantization bits for cache (default: 8)
- MLX_MAX_CACHED_TOKENS_PER_CONVERSATION: Max tokens per conversation cache (default: 4096)
- MLX_CONVERSATION_DETECTION_THRESHOLD: Threshold for conversation detection (default: 0.7)
- MLX_SEMANTIC_SIMILARITY_THRESHOLD: Threshold for semantic similarity (default: 0.85)
- MLX_PREDICTIVE_CACHE_WARMING: Enable predictive cache warming (default: false)
- MLX_CACHE_PERSISTENCE_DIR: Directory for cache persistence (default: ~/.cache/mlx-server-nano)
- MLX_AUTO_DETECT_CONVERSATIONS: Enable automatic conversation detection (default: true)
"""

import os
from dataclasses import dataclass


@dataclass
class ServerConfig:
    """Server configuration with environment variable support."""

    host: str = "0.0.0.0"
    port: int = 8000
    model_idle_timeout: int = 300
    default_max_tokens: int = 512
    default_temperature: float = 0.7
    log_level: str = "INFO"

    # Conversation caching configuration
    conversation_cache_enabled: bool = True
    conversation_idle_timeout: int = 300  # seconds
    max_conversations: int = 10
    cache_quantization_enabled: bool = True
    cache_quantization_bits: int = 8
    max_cached_tokens_per_conversation: int = 4096
    conversation_detection_threshold: float = 0.7  # 70% message overlap
    semantic_similarity_threshold: float = 0.85
    predictive_cache_warming: bool = False
    cache_persistence_dir: str = "~/.cache/mlx-server-nano"
    auto_detect_conversations: bool = True  # Enable automatic conversation detection

    @classmethod
    def from_env(cls) -> "ServerConfig":
        """Create configuration from environment variables."""
        return cls(
            host=os.environ.get("MLX_SERVER_HOST", "0.0.0.0"),
            port=int(os.environ.get("MLX_SERVER_PORT", "8000")),
            model_idle_timeout=int(os.environ.get("MLX_MODEL_IDLE_TIMEOUT", "300")),
            default_max_tokens=int(os.environ.get("MLX_DEFAULT_MAX_TOKENS", "512")),
            default_temperature=float(os.environ.get("MLX_DEFAULT_TEMPERATURE", "0.7")),
            log_level=os.environ.get("MLX_LOG_LEVEL", "INFO"),
            # Conversation caching from environment
            conversation_cache_enabled=os.environ.get(
                "MLX_CONVERSATION_CACHE_ENABLED", "true"
            ).lower()
            == "true",
            conversation_idle_timeout=int(
                os.environ.get("MLX_CONVERSATION_IDLE_TIMEOUT", "300")
            ),
            max_conversations=int(os.environ.get("MLX_MAX_CONVERSATIONS", "10")),
            cache_quantization_enabled=os.environ.get(
                "MLX_CACHE_QUANTIZATION_ENABLED", "true"
            ).lower()
            == "true",
            cache_quantization_bits=int(
                os.environ.get("MLX_CACHE_QUANTIZATION_BITS", "8")
            ),
            max_cached_tokens_per_conversation=int(
                os.environ.get("MLX_MAX_CACHED_TOKENS_PER_CONVERSATION", "4096")
            ),
            conversation_detection_threshold=float(
                os.environ.get("MLX_CONVERSATION_DETECTION_THRESHOLD", "0.7")
            ),
            semantic_similarity_threshold=float(
                os.environ.get("MLX_SEMANTIC_SIMILARITY_THRESHOLD", "0.85")
            ),
            predictive_cache_warming=os.environ.get(
                "MLX_PREDICTIVE_CACHE_WARMING", "false"
            ).lower()
            == "true",
            cache_persistence_dir=os.environ.get(
                "MLX_CACHE_PERSISTENCE_DIR", "~/.cache/mlx-server-nano"
            ),
            auto_detect_conversations=os.environ.get(
                "MLX_AUTO_DETECT_CONVERSATIONS", "true"
            ).lower()
            == "true",
        )


# Global config instance
config = ServerConfig.from_env()
