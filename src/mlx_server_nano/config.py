import os
from dataclasses import dataclass


@dataclass
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    model_cache_dir: str = "models"
    model_idle_timeout: int = 300
    default_max_tokens: int = 512
    default_temperature: float = 0.7
    log_level: str = "INFO"

    @classmethod
    def from_env(cls) -> "ServerConfig":
        return cls(
            host=os.environ.get("MLX_SERVER_HOST", "0.0.0.0"),
            port=int(os.environ.get("MLX_SERVER_PORT", "8000")),
            model_cache_dir=os.environ.get("MLX_MODEL_CACHE_DIR", "models"),
            model_idle_timeout=int(os.environ.get("MLX_MODEL_IDLE_TIMEOUT", "300")),
            default_max_tokens=int(os.environ.get("MLX_DEFAULT_MAX_TOKENS", "512")),
            default_temperature=float(os.environ.get("MLX_DEFAULT_TEMPERATURE", "0.7")),
            log_level=os.environ.get("MLX_LOG_LEVEL", "INFO"),
        )


# Global config instance
config = ServerConfig.from_env()
