"""
Chat template formatting for different language models

Provides extensible chat template management using the TemplateManager system.
All template logic is now handled by Jinja2 templates configured in the templates directory.

Legacy hardcoded templates have been removed in favor of the flexible template system.
"""

import logging
from typing import Optional

from .schemas import Message, Tool

# Set up logging
logger = logging.getLogger(__name__)


def format_messages_for_model(
    messages: list[Message],
    model_name: str,
    tools: Optional[list[Tool]] = None,
) -> str:
    """
    Format messages using the extensible template system.

    Args:
        messages: List of conversation messages
        model_name: Name of the model to format for
        tools: Optional list of available tools

    Returns:
        Formatted prompt string ready for the model

    Raises:
        RuntimeError: If template system is not initialized or no template found
        Exception: If formatting fails for any reason
    """
    logger.debug(
        f"Formatting messages for model: {model_name}, message count: {len(messages)}, tools: {len(tools) if tools else 0}"
    )

    # Use template manager (extensible system)
    from .template_manager import get_template_manager

    template_manager = get_template_manager()

    if not template_manager or not template_manager.enabled:
        raise RuntimeError(
            "Template system not initialized. Please start the server with --templates-dir to specify a templates directory."
        )

    formatted = template_manager.format_messages(messages, model_name, tools)
    if formatted is not None:
        logger.debug(
            f"Successfully formatted using template manager for model: {model_name}"
        )
        return formatted
    else:
        # Template manager is enabled but no template matched
        available_templates = template_manager.list_available_templates()
        raise RuntimeError(
            f"No template found for model '{model_name}'. "
            f"Available templates: {available_templates}. "
            f"Consider adding a matching rule in your config.yaml or using the default template."
        )
