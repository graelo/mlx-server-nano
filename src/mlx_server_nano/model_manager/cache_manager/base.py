"""
Base Cache Manager

Provides a common interface for different types of cache management.
This serves as the foundation for both model caching and prompt caching.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class CacheManager(ABC):
    """
    Abstract base class for cache managers.

    Provides a common interface for different caching strategies.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize cache manager with configuration.

        Args:
            config: Dictionary with cache settings specific to implementation
        """
        self.config = config
        logger.debug(f"Initialized {self.__class__.__name__} with config: {config}")

    @abstractmethod
    def get_cache(self, key: Optional[str] = None) -> Any:
        """
        Get or create a cache for a given key.

        Args:
            key: Optional identifier to retrieve existing cache

        Returns:
            Cache object appropriate for the implementation
        """
        pass

    @abstractmethod
    def save_cache(self, cache: Any, identifier: str) -> None:
        """
        Save cache to persistent storage.

        Args:
            cache: Cache object to save
            identifier: Unique identifier for the cache
        """
        pass

    @abstractmethod
    def load_cache(self, identifier: str) -> Optional[Any]:
        """
        Load cache from persistent storage.

        Args:
            identifier: Unique identifier for the cache

        Returns:
            Cache object if found, None otherwise
        """
        pass

    @abstractmethod
    def clear_cache(self, identifier: Optional[str] = None) -> None:
        """
        Clear cache(s).

        Args:
            identifier: Optional specific cache to clear. If None, clear all.
        """
        pass

    @abstractmethod
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get statistics about cache usage.

        Returns:
            Dictionary with cache statistics
        """
        pass
