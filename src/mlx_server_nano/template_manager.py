"""
Template Manager for MLX Server Nano

Provides extensible chat template management with Jinja2 support and Unsloth compatibility.
Replaces hardcoded template logic with configurable, regex-based template selection.

Features:
- Jinja2 template loading and caching
- YAML configuration for template-to-model mapping
- Regex-based automatic model matching
- Stop sequence configuration per template
- Fallback to default templates
- Unsloth GGUF template compatibility

Configuration Format (config.yaml):
```yaml
template_rules:
  - pattern: ".*qwen.*"
    template: "qwen3.jinja2"
    stop_sequences: ["✿RESULT✿:", "✿RETURN✿:", "<|im_end|>"]
  - pattern: ".*devstral.*"
    template: "devstral.jinja2"
    stop_sequences: ["[/B_INST]", "[/TOOL_CALLS]"]

default_template: "default.jinja2"
default_stop_sequences: []
```
"""

import logging
import re
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from jinja2 import Environment, FileSystemLoader, Template, TemplateNotFound

# Set up logging
logger = logging.getLogger(__name__)


class TemplateRule:
    """Represents a template matching rule from configuration."""

    def __init__(
        self, pattern: str, template: str, stop_sequences: Optional[List[str]] = None
    ):
        self.pattern = pattern
        self.compiled_pattern = re.compile(pattern, re.IGNORECASE)
        self.template = template
        self.stop_sequences = stop_sequences or []

    def matches(self, model_name: str) -> bool:
        """Check if this rule matches the given model name."""
        return bool(self.compiled_pattern.search(model_name))


class TemplateManager:
    """
    Manages Jinja2 chat templates with configurable model matching.

    This class provides the core infrastructure for the extensible template system,
    supporting Unsloth GGUF templates and automatic model-to-template matching.
    """

    def __init__(self, templates_dir: Optional[str] = None):
        """
        Initialize the template manager.

        Args:
            templates_dir: Directory containing template files and config.yaml.
                          If None, template system is disabled and falls back to hardcoded logic.
        """
        self.templates_dir = Path(templates_dir) if templates_dir else None
        self.enabled = self.templates_dir is not None

        # Template system state
        self.env: Optional[Environment] = None
        self.template_cache: Dict[str, Template] = {}
        self.rules: List[TemplateRule] = []
        self.default_template: Optional[str] = None
        self.default_stop_sequences: List[str] = []

        # Initialize if templates directory is provided
        if self.enabled:
            self._initialize()

    def _initialize(self):
        """Initialize the template system by loading configuration and setting up Jinja2."""
        if not self.templates_dir or not self.templates_dir.exists():
            logger.warning(f"Templates directory not found: {self.templates_dir}")
            self.enabled = False
            return

        try:
            # Set up Jinja2 environment
            self.env = Environment(
                loader=FileSystemLoader(str(self.templates_dir)),
                trim_blocks=True,
                lstrip_blocks=True,
                # Add security considerations for production use
                autoescape=False,  # Templates contain code, not HTML
            )

            # Add custom filters if needed for Unsloth compatibility
            self.env.filters["tojson"] = self._json_filter

            # Load configuration
            self._load_config()

            logger.info(
                f"Template manager initialized with {len(self.rules)} rules from {self.templates_dir}"
            )

        except Exception as e:
            logger.error(f"Failed to initialize template manager: {e}", exc_info=True)
            self.enabled = False

    def _load_config(self):
        """Load template configuration from config.yaml."""
        if not self.templates_dir:
            logger.warning("Templates directory not set")
            return

        config_path = self.templates_dir / "config.yaml"

        if not config_path.exists():
            logger.warning(
                f"No config.yaml found in {self.templates_dir}, using empty configuration"
            )
            return

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config_data = yaml.safe_load(f)

            # Parse template rules
            rules_data = config_data.get("template_rules", [])
            self.rules = []

            for rule_data in rules_data:
                try:
                    rule = TemplateRule(
                        pattern=rule_data["pattern"],
                        template=rule_data["template"],
                        stop_sequences=rule_data.get("stop_sequences", []),
                    )
                    self.rules.append(rule)
                    logger.debug(
                        f"Loaded template rule: {rule.pattern} -> {rule.template}"
                    )
                except KeyError as e:
                    logger.error(f"Invalid template rule configuration: missing {e}")
                except re.error as e:
                    logger.error(
                        f"Invalid regex pattern '{rule_data.get('pattern', '')}': {e}"
                    )

            # Load defaults
            self.default_template = config_data.get("default_template")
            self.default_stop_sequences = config_data.get("default_stop_sequences", [])

            logger.info(
                f"Loaded template configuration: {len(self.rules)} rules, default: {self.default_template}"
            )

        except Exception as e:
            logger.error(f"Failed to load template configuration: {e}", exc_info=True)

    def _json_filter(self, value: Any, indent: Optional[int] = None) -> str:
        """Custom JSON filter for Jinja2 templates (Unsloth compatibility)."""
        import json

        return json.dumps(value, ensure_ascii=False, indent=indent)

    def get_template_for_model(
        self, model_name: str
    ) -> Tuple[Optional[Template], List[str]]:
        """
        Get the appropriate template and stop sequences for a model.

        Args:
            model_name: Name of the model to get template for

        Returns:
            Tuple of (Template object or None, stop_sequences list)
        """
        if not self.enabled or not self.env:
            logger.debug("Template manager not enabled, returning None")
            return None, []

        # Find matching rule
        for rule in self.rules:
            if rule.matches(model_name):
                logger.debug(
                    f"Model '{model_name}' matched rule: {rule.pattern} -> {rule.template}"
                )
                template = self._get_template(rule.template)
                return template, rule.stop_sequences

        # No rule matched, try default template
        if self.default_template:
            logger.debug(
                f"Model '{model_name}' using default template: {self.default_template}"
            )
            template = self._get_template(self.default_template)
            return template, self.default_stop_sequences

        logger.debug(f"No template found for model '{model_name}'")
        return None, []

    def _get_template(self, template_name: str) -> Optional[Template]:
        """
        Get a template by name with caching.

        Args:
            template_name: Name of the template file

        Returns:
            Template object or None if not found
        """
        if not self.env:
            logger.error("Jinja2 environment not initialized")
            return None

        # Check cache first
        if template_name in self.template_cache:
            return self.template_cache[template_name]

        try:
            template = self.env.get_template(template_name)
            self.template_cache[template_name] = template
            logger.debug(f"Loaded and cached template: {template_name}")
            return template
        except TemplateNotFound:
            logger.error(f"Template not found: {template_name}")
            return None
        except Exception as e:
            logger.error(
                f"Failed to load template '{template_name}': {e}", exc_info=True
            )
            return None

    def format_messages(
        self,
        messages: List,
        model_name: str,
        tools: Optional[List] = None,
        **template_vars,
    ) -> Optional[str]:
        """
        Format messages using the appropriate template for the model.

        Args:
            messages: List of conversation messages
            model_name: Name of the model
            tools: Optional list of available tools
            **template_vars: Additional variables to pass to the template

        Returns:
            Formatted prompt string or None if no template available
        """
        template, _ = self.get_template_for_model(model_name)

        if not template:
            logger.debug(f"No template available for model '{model_name}'")
            return None

        try:
            # Convert Pydantic models to dictionaries for template compatibility
            messages_dict = []
            for msg in messages:
                if hasattr(msg, "model_dump"):
                    # Pydantic v2 model
                    messages_dict.append(msg.model_dump())
                elif hasattr(msg, "dict"):
                    # Pydantic v1 model
                    messages_dict.append(msg.dict())
                else:
                    # Already a dictionary
                    messages_dict.append(msg)

            tools_dict = None
            if tools:
                tools_dict = []
                for tool in tools:
                    if hasattr(tool, "model_dump"):
                        # Pydantic v2 model
                        tools_dict.append(tool.model_dump())
                    elif hasattr(tool, "dict"):
                        # Pydantic v1 model
                        tools_dict.append(tool.dict())
                    else:
                        # Already a dictionary
                        tools_dict.append(tool)

            # Prepare template variables with Unsloth compatibility
            template_context = {
                "messages": messages_dict,
                "tools": tools_dict,
                "bos_token": "<|begin_of_text|>",  # Common BOS token
                "eos_token": "<|end_of_text|>",  # Common EOS token
                **template_vars,
            }

            # Render the template
            result = template.render(**template_context)
            logger.debug(
                f"Successfully formatted {len(messages)} messages using template for '{model_name}'"
            )
            return result

        except Exception as e:
            logger.error(
                f"Failed to render template for model '{model_name}': {e}",
                exc_info=True,
            )
            return None

    def get_stop_sequences(self, model_name: str) -> List[str]:
        """
        Get stop sequences for a model based on template configuration.

        Args:
            model_name: Name of the model

        Returns:
            List of stop sequences for the model
        """
        if not self.enabled:
            return []

        _, stop_sequences = self.get_template_for_model(model_name)
        return stop_sequences

    def list_available_templates(self) -> List[str]:
        """
        List all available template files in the templates directory.

        Returns:
            List of template file names
        """
        if not self.enabled or not self.templates_dir:
            return []

        try:
            template_files = []
            for file_path in self.templates_dir.glob("*.jinja2"):
                template_files.append(file_path.name)
            return sorted(template_files)
        except Exception as e:
            logger.error(f"Failed to list templates: {e}")
            return []

    def validate_template(self, template_name: str) -> bool:
        """
        Validate that a template can be loaded and parsed.

        Args:
            template_name: Name of the template to validate

        Returns:
            True if template is valid, False otherwise
        """
        if not self.enabled:
            return False

        try:
            template = self._get_template(template_name)
            return template is not None
        except Exception:
            return False


# Global template manager instance (initialized later)
_template_manager: Optional[TemplateManager] = None


def initialize_template_manager(templates_dir: Optional[str] = None) -> None:
    """
    Initialize the global template manager instance.

    Args:
        templates_dir: Directory containing template files and config.yaml
    """
    global _template_manager
    _template_manager = TemplateManager(templates_dir)
    logger.info(f"Template manager initialized (enabled: {_template_manager.enabled})")


def get_template_manager() -> Optional[TemplateManager]:
    """
    Get the global template manager instance.

    Returns:
        TemplateManager instance or None if not initialized
    """
    return _template_manager


def is_template_system_enabled() -> bool:
    """
    Check if the template system is enabled and ready.

    Returns:
        True if template system is enabled, False otherwise
    """
    return _template_manager is not None and _template_manager.enabled
