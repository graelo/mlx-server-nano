"""
Basic validation tests to ensure test infrastructure is working.

These tests verify that the test setup is functional and provide quick validation
of core components without requiring complex mocking.
"""

import pytest
from unittest.mock import patch, MagicMock

from mlx_server_nano.config import ServerConfig
from mlx_server_nano.schemas import Message, Function, Tool


@pytest.mark.unit
class TestBasicFunctionality:
    """Basic tests to validate test infrastructure."""

    def test_config_creation(self):
        """Test basic config creation works."""
        config = ServerConfig()
        assert config.host == "0.0.0.0"
        assert config.port == 8000  # Corrected from 8080
        assert isinstance(config.model_idle_timeout, int)

    def test_schema_creation(self):
        """Test basic schema creation works."""
        message = Message(role="user", content="Hello")
        assert message.role == "user"
        assert message.content == "Hello"

        function = Function(
            name="test_func",
            description="A test function",
            parameters={"type": "object", "properties": {}},
        )
        assert function.name == "test_func"

        tool = Tool(function=function)
        assert tool.type == "function"
        assert tool.function == function

    @patch("mlx_server_nano.model_manager.load")
    def test_mocking_works(self, mock_load, clean_model_manager):
        """Test that mocking infrastructure works."""
        from mlx_server_nano.model_manager import load_model

        # Configure mock
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.bos_token = "<s>"
        mock_load.return_value = (mock_model, mock_tokenizer)

        # Call function
        result = load_model("test-model")

        # Verify mock was called and returned expected values
        mock_load.assert_called_once_with("test-model")
        assert result == (mock_model, mock_tokenizer)


@pytest.mark.integration
class TestBasicIntegration:
    """Basic integration tests."""

    def test_import_statements(self):
        """Test that all main modules can be imported."""
        import mlx_server_nano.main
        from mlx_server_nano.config import ServerConfig
        from mlx_server_nano.schemas import Message
        import mlx_server_nano.model_manager

        # Just test they can be imported without errors
        assert hasattr(mlx_server_nano, "app")
        assert ServerConfig is not None
        assert Message is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
