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
    templates_dir: str = typer.Option(
        config.templates_dir,
        "--templates-dir",
        help="Directory containing Jinja2 templates and config.yaml for extensible chat template system (required)",
    ),
):
    """Start the MLX Server Nano FastAPI server."""
    # Validate required arguments
    if not templates_dir:
        print("Error: --templates-dir is required for the extensible template system.")
        print("Please specify a directory containing Jinja2 templates and config.yaml")
        print("Example: mlx-server-nano --templates-dir ./templates")
        raise typer.Exit(1)

    # Update config singleton
    config.host = host
    config.port = port
    config.log_level = log_level
    config.templates_dir = templates_dir

    # Set up logging first so template manager logs are visible
    log_level_val = getattr(logging, config.log_level, logging.INFO)
    logging.basicConfig(
        level=log_level_val,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )
    logging.getLogger("mlx_server_nano").setLevel(log_level_val)
    logging.getLogger("uvicorn").setLevel(logging.INFO)

    # Initialize template manager after logging is set up
    from .template_manager import initialize_template_manager, get_template_manager

    initialize_template_manager(config.templates_dir)

    # Get template manager info for startup messages
    template_manager = get_template_manager()
    template_status = (
        "enabled" if template_manager and template_manager.enabled else "disabled"
    )

    print(f"Starting MLX Server Nano on {config.host}:{config.port}")
    print(f"Log level: {config.log_level}")
    template_count = (
        len(template_manager.rules)
        if template_manager and template_manager.enabled
        else 0
    )
    print(
        f"Templates: {template_status} ({template_count} rules) - {config.templates_dir}"
    )
    print(f"Using Hugging Face cache (HF_HOME: {os.environ.get('HF_HOME', 'default')})")

    uvicorn.run(
        "mlx_server_nano.app:app",
        host=config.host,
        port=config.port,
        log_level=config.log_level.lower(),
        reload=reload,
    )
