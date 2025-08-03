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

This CLI launches the FastAPI app defined in mlx_server_nano.app.
Note: Since there's only one command, Typer automatically runs it without needing 'serve'.
"""

import os
import logging
import typer
import uvicorn
from .config import config

app_cli = typer.Typer()


@app_cli.command()
def serve(
    host: str = typer.Option(config.host, help="Host to bind to"),
    port: int = typer.Option(config.port, help="Port to bind to"),
    log_level: str = typer.Option(
        config.log_level, help="Log level", show_choices=True, case_sensitive=False
    ),
    reload: bool = typer.Option(False, help="Enable auto-reload for development"),
):
    """Start the MLX Server Nano FastAPI server."""

    # Update config singleton
    config.host = host
    config.port = port
    config.log_level = log_level

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

    uvicorn.run(
        "mlx_server_nano.app:app",
        host=config.host,
        port=config.port,
        log_level=config.log_level.lower(),
        reload=reload,
    )
