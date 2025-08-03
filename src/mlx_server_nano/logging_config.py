import logging
import sys
from .config import config


def setup_logging() -> None:
    """Setup logging configuration"""
    log_level = getattr(logging, config.log_level, logging.INFO)

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # Set MLX related loggers to appropriate levels
    logging.getLogger("mlx").setLevel(logging.INFO)
    logging.getLogger("uvicorn").setLevel(logging.INFO)


# Create a logger for this module
logger = logging.getLogger(__name__)
