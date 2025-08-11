"""
Background Task Management

Handles background model unloading automation with configurable idle timeouts.
Manages asyncio tasks for automatic model cleanup when models are not in use.

Features:
- Automatic model unloading after idle timeout
- Graceful task shutdown and cleanup
- Event-driven unload scheduling
- Thread-safe task management
"""

import asyncio
import logging
from typing import Optional

from .cache import MODEL_IDLE_TIMEOUT, get_cache_state, get_current_time
from . import cache

# Set up logging
logger = logging.getLogger(__name__)

# Background task management for model unloading
_model_unloader_task: Optional[asyncio.Task] = None
_unload_requested = asyncio.Event()
_shutdown_requested = asyncio.Event()


async def _model_unloader_background_task() -> None:
    """Background task that handles model unloading based on idle timeout."""
    logger.info("Model unloader background task started")

    while True:
        try:
            # Wait for either an unload request or shutdown
            done, pending = await asyncio.wait(
                [
                    asyncio.create_task(_unload_requested.wait()),
                    asyncio.create_task(_shutdown_requested.wait()),
                ],
                return_when=asyncio.FIRST_COMPLETED,
            )

            # Cancel any pending tasks
            for task in pending:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

            # Check what completed
            if _shutdown_requested.is_set():
                logger.info("Model unloader background task shutting down")
                break

            if _unload_requested.is_set():
                # Clear the event for next time
                _unload_requested.clear()

                # Wait for the idle timeout period, but also listen for new unload requests
                try:
                    done, pending = await asyncio.wait(
                        [
                            asyncio.create_task(_shutdown_requested.wait()),
                            asyncio.create_task(_unload_requested.wait()),
                        ],
                        timeout=MODEL_IDLE_TIMEOUT,
                        return_when=asyncio.FIRST_COMPLETED,
                    )

                    # Cancel any pending tasks
                    for task in pending:
                        task.cancel()
                        try:
                            await task
                        except asyncio.CancelledError:
                            pass

                    # Check what completed
                    if _shutdown_requested.is_set():
                        logger.info(
                            "Model unloader background task shutting down during timeout"
                        )
                        break
                    elif _unload_requested.is_set():
                        # New unload request received during timeout, immediately check condition
                        _unload_requested.clear()
                        model_name, last_used_time = get_cache_state()
                        if (
                            get_current_time() - last_used_time >= MODEL_IDLE_TIMEOUT
                            and model_name is not None
                        ):
                            logger.info(
                                f"Unloading model '{model_name}' due to inactivity (immediate check)"
                            )
                            cache._unload_model()
                        # Continue the loop to start a new timeout period
                        continue
                    else:
                        # Timeout expired, check if we should unload
                        model_name, last_used_time = get_cache_state()
                        if (
                            get_current_time() - last_used_time >= MODEL_IDLE_TIMEOUT
                            and model_name is not None
                        ):
                            logger.info(
                                f"Unloading model '{model_name}' due to inactivity"
                            )
                            cache._unload_model()

                except Exception as timeout_error:
                    logger.error(f"Error during timeout wait: {timeout_error}")
                    await asyncio.sleep(1)

        except Exception as e:
            logger.error(f"Error in model unloader background task: {e}")
            await asyncio.sleep(1)  # Brief pause before retrying


def _schedule_unload():
    """Schedule model unloading after idle timeout."""
    # Simply set the event to trigger the background task
    if not _shutdown_requested.is_set():
        _unload_requested.set()


async def start_model_unloader():
    """Start the model unloader background task."""
    global _model_unloader_task

    if _model_unloader_task is None or _model_unloader_task.done():
        _model_unloader_task = asyncio.create_task(_model_unloader_background_task())
        logger.info("Model unloader background task created")


async def stop_model_unloader():
    """Stop the model unloader background task."""
    global _model_unloader_task

    _shutdown_requested.set()

    if _model_unloader_task and not _model_unloader_task.done():
        try:
            await asyncio.wait_for(_model_unloader_task, timeout=5.0)
        except asyncio.TimeoutError:
            logger.warning("Model unloader task did not stop gracefully, cancelling")
            _model_unloader_task.cancel()
            try:
                await _model_unloader_task
            except asyncio.CancelledError:
                pass

    logger.info("Model unloader background task stopped")
