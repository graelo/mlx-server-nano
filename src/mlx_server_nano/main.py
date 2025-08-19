"""
MLX Server Nano CLI entry point (Typer-based)

Usage:
    python -m mlx_server_nano.main [OPTIONS]
    # or, if installed as a script:
    mlx-server-nano [OPTIONS]

Options:
    --host         Host to bind to (default: 127.0.0.1)
    --port         Port to bind to (default: 8000)
    --log-level    Log level (DEBUG, INFO, WARNING, ERROR)
    --reload       Enable auto-reload for development

Cache Options:
    --cache-type          Cache type (KVCache, QuantizedKVCache, RotatingKVCache, ChunkedKVCache, ConcatenateKVCache)
    --cache-max-size      Maximum cache size for RotatingKVCache
    --cache-chunk-size    Chunk size for ChunkedKVCache
    --max-conversations   Maximum number of cached conversations
    --cache-enabled       Enable/disable conversation caching
    --cache-timeout       Cache idle timeout in seconds

This CLI launches the FastAPI app defined in mlx_server_nano.app.
Note: Since there's only one command, Typer automatically runs it without needing 'serve'.
"""

import os
import logging
import typer
import uvicorn
from .config import config, CacheType

app_cli = typer.Typer()


@app_cli.command()
def serve(
    host: str = typer.Option(config.host, help="Host to bind to"),
    port: int = typer.Option(config.port, help="Port to bind to"),
    log_level: str = typer.Option(
        config.log_level, help="Log level", show_choices=True, case_sensitive=False
    ),
    reload: bool = typer.Option(False, help="Enable auto-reload for development"),
    # Cache options
    cache_type: CacheType = typer.Option(
        config.cache_type,
        case_sensitive=False,
        help="Cache type (KVCache, QuantizedKVCache, RotatingKVCache, ChunkedKVCache, ConcatenateKVCache)",
    ),
    cache_max_size: int = typer.Option(
        config.cache_max_size, help="Maximum cache size for RotatingKVCache"
    ),
    cache_chunk_size: int = typer.Option(
        config.cache_chunk_size, help="Chunk size for ChunkedKVCache"
    ),
    max_conversations: int = typer.Option(
        config.max_conversations, help="Maximum number of cached conversations"
    ),
    cache_enabled: bool = typer.Option(
        config.conversation_cache_enabled, help="Enable/disable conversation caching"
    ),
    cache_timeout: int = typer.Option(
        config.conversation_idle_timeout, help="Cache idle timeout in seconds"
    ),
):
    """Start the MLX Server Nano FastAPI server."""

    # Update config singleton with CLI arguments
    config.host = host
    config.port = port
    config.log_level = log_level
    config.cache_type = cache_type
    config.cache_max_size = cache_max_size
    config.cache_chunk_size = cache_chunk_size
    config.max_conversations = max_conversations
    config.conversation_cache_enabled = cache_enabled
    config.conversation_idle_timeout = cache_timeout

    # Set up logging first
    log_level_val = getattr(logging, config.log_level, logging.INFO)
    logging.basicConfig(
        level=log_level_val,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )
    logging.getLogger("mlx_server_nano").setLevel(log_level_val)
    logging.getLogger("uvicorn").setLevel(logging.INFO)

    print(f"Starting MLX Server Nano on {config.host}:{config.port}")
    print(f"Log level: {config.log_level}")
    print("Using native MLX-LM integration (no custom templates)")
    print(f"Using Hugging Face cache (HF_HOME: {os.environ.get('HF_HOME', 'default')})")
    print(f"Cache enabled: {config.conversation_cache_enabled}")
    if config.conversation_cache_enabled:
        print(f"Cache type: {config.cache_type}")
        print(f"Max conversations: {config.max_conversations}")
        print(f"Cache timeout: {config.conversation_idle_timeout}s")
        if config.cache_type == "RotatingKVCache":
            print(f"Cache max size: {config.cache_max_size}")
        if config.cache_type == "ChunkedKVCache":
            print(f"Cache chunk size: {config.cache_chunk_size}")

    uvicorn.run(
        "mlx_server_nano.app:app",
        host=config.host,
        port=config.port,
        log_level=config.log_level.lower(),
        reload=reload,
    )
