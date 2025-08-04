"""
Unit tests for configuration management.

Tests the ServerConfig class and environment variable handling.
"""

import os
import pytest
from unittest.mock import patch

from mlx_server_nano.config import ServerConfig


@pytest.mark.unit
class TestServerConfig:
    """Test cases for ServerConfig class."""

    def test_default_values(self):
        """Test that default configuration values are correct."""
        config = ServerConfig()

        assert config.host == "0.0.0.0"
        assert config.port == 8000
        assert config.model_idle_timeout == 300
        assert config.default_max_tokens == 512
        assert config.default_temperature == 0.7
        assert config.log_level == "INFO"

    def test_custom_values(self):
        """Test creating config with custom values."""
        config = ServerConfig(
            host="127.0.0.1",
            port=9000,
            model_idle_timeout=600,
            default_max_tokens=1024,
            default_temperature=0.5,
            log_level="DEBUG",
        )

        assert config.host == "127.0.0.1"
        assert config.port == 9000
        assert config.model_idle_timeout == 600
        assert config.default_max_tokens == 1024
        assert config.default_temperature == 0.5
        assert config.log_level == "DEBUG"

    def test_from_env_defaults(self):
        """Test from_env with no environment variables set."""
        with patch.dict(os.environ, {}, clear=True):
            config = ServerConfig.from_env()

            assert config.host == "0.0.0.0"
            assert config.port == 8000
            assert config.model_idle_timeout == 300
            assert config.default_max_tokens == 512
            assert config.default_temperature == 0.7
            assert config.log_level == "INFO"

    def test_from_env_with_variables(self):
        """Test from_env with environment variables set."""
        env_vars = {
            "MLX_SERVER_HOST": "192.168.1.1",
            "MLX_SERVER_PORT": "3000",
            "MLX_MODEL_IDLE_TIMEOUT": "120",
            "MLX_DEFAULT_MAX_TOKENS": "256",
            "MLX_DEFAULT_TEMPERATURE": "0.9",
            "MLX_LOG_LEVEL": "WARNING",
        }

        with patch.dict(os.environ, env_vars, clear=True):
            config = ServerConfig.from_env()

            assert config.host == "192.168.1.1"
            assert config.port == 3000
            assert config.model_idle_timeout == 120
            assert config.default_max_tokens == 256
            assert config.default_temperature == 0.9
            assert config.log_level == "WARNING"

    def test_from_env_partial_variables(self):
        """Test from_env with only some environment variables set."""
        env_vars = {"MLX_SERVER_PORT": "4000", "MLX_LOG_LEVEL": "ERROR"}

        with patch.dict(os.environ, env_vars, clear=True):
            config = ServerConfig.from_env()

            # Set variables should use env values
            assert config.port == 4000
            assert config.log_level == "ERROR"

            # Unset variables should use defaults
            assert config.host == "0.0.0.0"
            assert config.model_idle_timeout == 300
            assert config.default_max_tokens == 512
            assert config.default_temperature == 0.7

    def test_from_env_type_conversion(self):
        """Test that environment variables are properly converted to correct types."""
        env_vars = {
            "MLX_SERVER_PORT": "8080",  # str -> int
            "MLX_MODEL_IDLE_TIMEOUT": "60",  # str -> int
            "MLX_DEFAULT_MAX_TOKENS": "128",  # str -> int
            "MLX_DEFAULT_TEMPERATURE": "1.2",  # str -> float
        }

        with patch.dict(os.environ, env_vars, clear=True):
            config = ServerConfig.from_env()

            assert isinstance(config.port, int)
            assert isinstance(config.model_idle_timeout, int)
            assert isinstance(config.default_max_tokens, int)
            assert isinstance(config.default_temperature, float)

            assert config.port == 8080
            assert config.model_idle_timeout == 60
            assert config.default_max_tokens == 128
            assert config.default_temperature == 1.2

    def test_from_env_invalid_types(self):
        """Test handling of invalid environment variable values."""
        # Test invalid port (not a number)
        with patch.dict(os.environ, {"MLX_SERVER_PORT": "not_a_number"}, clear=True):
            with pytest.raises(ValueError):
                ServerConfig.from_env()

        # Test invalid timeout (not a number)
        with patch.dict(os.environ, {"MLX_MODEL_IDLE_TIMEOUT": "invalid"}, clear=True):
            with pytest.raises(ValueError):
                ServerConfig.from_env()

        # Test invalid temperature (not a number)
        with patch.dict(
            os.environ, {"MLX_DEFAULT_TEMPERATURE": "not_float"}, clear=True
        ):
            with pytest.raises(ValueError):
                ServerConfig.from_env()

    def test_config_immutability(self):
        """Test that config objects maintain their values."""
        config1 = ServerConfig(port=8001)
        config2 = ServerConfig(port=8002)

        assert config1.port == 8001
        assert config2.port == 8002

        # Creating a new config shouldn't affect existing ones
        ServerConfig.from_env()
        assert config1.port == 8001
        assert config2.port == 8002
