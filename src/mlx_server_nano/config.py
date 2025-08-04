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
        )


# Global config instance
config = ServerConfig.from_env()
