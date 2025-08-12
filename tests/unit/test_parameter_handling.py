"""
Test to verify that temperature and top_p parameters are correctly passed through
to the MLX-LM generation pipeline.
"""

from unittest.mock import patch, MagicMock

from mlx_server_nano.model_manager.generation import _setup_generation_kwargs


class TestParameterHandling:
    """Test proper handling of temperature and top_p parameters."""

    def test_temperature_and_top_p_create_sampler(self):
        """Test that temperature and top_p parameters create a sampler."""
        with patch("mlx_lm.sample_utils.make_sampler") as mock_make_sampler:
            mock_sampler = MagicMock()
            mock_make_sampler.return_value = mock_sampler

            kwargs = _setup_generation_kwargs(
                "test-model", temperature=0.8, top_p=0.9, max_tokens=100
            )

            # Verify make_sampler was called with correct parameters
            mock_make_sampler.assert_called_once_with(temp=0.8, top_p=0.9)

            # Verify the sampler is included in kwargs
            assert "sampler" in kwargs
            assert kwargs["sampler"] == mock_sampler
            assert kwargs["max_tokens"] == 100

    def test_temperature_only_creates_sampler(self):
        """Test that only temperature parameter creates sampler with default top_p."""
        with patch("mlx_lm.sample_utils.make_sampler") as mock_make_sampler:
            mock_sampler = MagicMock()
            mock_make_sampler.return_value = mock_sampler

            kwargs = _setup_generation_kwargs(
                "test-model", temperature=0.7, max_tokens=50
            )

            # Verify make_sampler was called with temperature and default top_p
            mock_make_sampler.assert_called_once_with(temp=0.7, top_p=0.0)

            assert "sampler" in kwargs
            assert kwargs["sampler"] == mock_sampler

    def test_top_p_only_creates_sampler(self):
        """Test that only top_p parameter creates sampler with default temperature."""
        with patch("mlx_lm.sample_utils.make_sampler") as mock_make_sampler:
            mock_sampler = MagicMock()
            mock_make_sampler.return_value = mock_sampler

            kwargs = _setup_generation_kwargs("test-model", top_p=0.95, max_tokens=200)

            # Verify make_sampler was called with top_p and default temperature
            mock_make_sampler.assert_called_once_with(temp=0.0, top_p=0.95)

            assert "sampler" in kwargs
            assert kwargs["sampler"] == mock_sampler

    def test_no_sampling_parameters_no_sampler(self):
        """Test that no sampling parameters means no sampler is created."""
        with patch("mlx_lm.sample_utils.make_sampler") as mock_make_sampler:
            kwargs = _setup_generation_kwargs("test-model", max_tokens=100)

            # Verify make_sampler was NOT called
            mock_make_sampler.assert_not_called()

            # Verify no sampler in kwargs
            assert "sampler" not in kwargs
            assert kwargs["max_tokens"] == 100

    def test_zero_values_create_sampler(self):
        """Test that zero values for temperature/top_p still create a sampler."""
        with patch("mlx_lm.sample_utils.make_sampler") as mock_make_sampler:
            mock_sampler = MagicMock()
            mock_make_sampler.return_value = mock_sampler

            kwargs = _setup_generation_kwargs("test-model", temperature=0.0, top_p=0.0)

            # Verify make_sampler was called even with zero values
            mock_make_sampler.assert_called_once_with(temp=0.0, top_p=0.0)

            assert "sampler" in kwargs
            assert kwargs["sampler"] == mock_sampler
