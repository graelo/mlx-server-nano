"""
Unit tests for tool calling functionality.

Tests the tool call parsing, detection, and error handling capabilities
of the MLX Server Nano tool calling system.
"""

from mlx_server_nano.model_manager.tool_calling import parse_tool_calls, has_tool_calls


class TestToolCallParsing:
    """Test tool call parsing functionality."""

    def test_simple_tool_call_parsing(self):
        """Test parsing a simple tool call."""
        response = '[TOOL_CALLS]get_weather[ARGS]{"location": "Paris"}'
        tool_calls = parse_tool_calls(response)

        assert len(tool_calls) == 1
        assert tool_calls[0].function["name"] == "get_weather"
        assert tool_calls[0].type == "function"
        assert '"location": "Paris"' in tool_calls[0].function["arguments"]

    def test_complex_json_with_nested_braces(self):
        """Test parsing tool call with complex JSON containing nested braces and quotes.

        This test specifically addresses the bug where tool calls with nested braces
        in the arguments (like shell commands with {}) would fail to parse correctly.
        """
        # This is the exact case that was failing
        response = '[TOOL_CALLS]run_terminal_command[ARGS]{"command": "find . -name \\"*.py\\" -exec wc -l {} +", "waitForCompletion": true}'
        tool_calls = parse_tool_calls(response)

        assert len(tool_calls) == 1, "Should parse exactly one tool call"
        assert tool_calls[0].function["name"] == "run_terminal_command"
        assert tool_calls[0].type == "function"

        # Verify the JSON was parsed correctly
        import json

        args = json.loads(tool_calls[0].function["arguments"])
        assert args["command"] == 'find . -name "*.py" -exec wc -l {} +'
        assert args["waitForCompletion"] is True

    def test_multiple_tool_calls(self):
        """Test parsing multiple tool calls in one response."""
        response = '[TOOL_CALLS]get_weather[ARGS]{"location": "Paris"}[TOOL_CALLS]get_time[ARGS]{"timezone": "UTC"}'
        tool_calls = parse_tool_calls(response)

        assert len(tool_calls) == 2
        assert tool_calls[0].function["name"] == "get_weather"
        assert tool_calls[1].function["name"] == "get_time"

    def test_tool_call_with_complex_nested_json(self):
        """Test tool call with deeply nested JSON structures."""
        response = '[TOOL_CALLS]complex_function[ARGS]{"config": {"database": {"host": "localhost", "settings": {"timeout": 30}}, "features": ["auth", "cache"]}, "debug": true}'
        tool_calls = parse_tool_calls(response)

        assert len(tool_calls) == 1
        assert tool_calls[0].function["name"] == "complex_function"

        import json

        args = json.loads(tool_calls[0].function["arguments"])
        assert args["config"]["database"]["host"] == "localhost"
        assert args["config"]["features"] == ["auth", "cache"]
        assert args["debug"] is True

    def test_tool_call_with_escaped_quotes(self):
        """Test tool call parsing with escaped quotes in JSON."""
        response = '[TOOL_CALLS]execute_script[ARGS]{"script": "echo \\"Hello World\\"", "params": ["-v"]}'
        tool_calls = parse_tool_calls(response)

        assert len(tool_calls) == 1
        assert tool_calls[0].function["name"] == "execute_script"

        import json

        args = json.loads(tool_calls[0].function["arguments"])
        assert args["script"] == 'echo "Hello World"'
        assert args["params"] == ["-v"]

    def test_tool_call_with_regex_patterns(self):
        """Test tool call parsing with regex patterns containing special characters."""
        response = '[TOOL_CALLS]search_files[ARGS]{"pattern": "\\\\w+\\\\.py$", "flags": "i", "replacement": "test_{}.py"}'
        tool_calls = parse_tool_calls(response)

        assert len(tool_calls) == 1
        assert tool_calls[0].function["name"] == "search_files"

    def test_invalid_json_handling(self):
        """Test that invalid JSON in tool calls is handled gracefully."""
        response = (
            '[TOOL_CALLS]broken_function[ARGS]{"invalid": json, "missing": quote}'
        )
        tool_calls = parse_tool_calls(response)

        # Should not crash, but return empty list due to JSON parsing error
        assert len(tool_calls) == 0

    def test_malformed_tool_call_structure(self):
        """Test handling of malformed tool call structures."""
        # Missing [ARGS]
        response1 = '[TOOL_CALLS]function_name{"param": "value"}'
        tool_calls1 = parse_tool_calls(response1)
        assert len(tool_calls1) == 0

        # Missing [TOOL_CALLS]
        response2 = 'function_name[ARGS]{"param": "value"}'
        tool_calls2 = parse_tool_calls(response2)
        assert len(tool_calls2) == 0

    def test_empty_and_whitespace_responses(self):
        """Test parsing empty or whitespace-only responses."""
        assert len(parse_tool_calls("")) == 0
        assert len(parse_tool_calls("   ")) == 0
        assert len(parse_tool_calls("\n\t  \n")) == 0

    def test_response_with_tool_calls_and_text(self):
        """Test parsing response that contains both tool calls and regular text."""
        response = 'I need to check the weather. [TOOL_CALLS]get_weather[ARGS]{"location": "Paris"} The forecast looks good!'
        tool_calls = parse_tool_calls(response)

        assert len(tool_calls) == 1
        assert tool_calls[0].function["name"] == "get_weather"

    def test_tool_call_id_generation(self):
        """Test that tool call IDs are generated and unique."""
        response = '[TOOL_CALLS]test_function[ARGS]{"param": "value"}'
        tool_calls1 = parse_tool_calls(response)
        tool_calls2 = parse_tool_calls(response)

        assert len(tool_calls1) == 1
        assert len(tool_calls2) == 1
        assert tool_calls1[0].id != tool_calls2[0].id  # IDs should be unique
        assert len(tool_calls1[0].id) == 9  # Should be 9 characters


class TestToolCallDetection:
    """Test tool call detection functionality."""

    def test_has_tool_calls_positive(self):
        """Test that has_tool_calls correctly identifies tool calls."""
        response = '[TOOL_CALLS]get_weather[ARGS]{"location": "Paris"}'
        assert has_tool_calls(response) is True

    def test_has_tool_calls_negative(self):
        """Test that has_tool_calls correctly identifies non-tool responses."""
        response = "This is just a regular response with no tool calls."
        assert has_tool_calls(response) is False

    def test_has_tool_calls_with_complex_case(self):
        """Test tool call detection with the complex nested braces case."""
        response = '[TOOL_CALLS]run_terminal_command[ARGS]{"command": "find . -name \\"*.py\\" -exec wc -l {} +", "waitForCompletion": true}'
        assert has_tool_calls(response) is True

    def test_has_tool_calls_empty_response(self):
        """Test tool call detection with empty response."""
        assert has_tool_calls("") is False
        assert has_tool_calls("   ") is False


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_tool_call_with_very_long_json(self):
        """Test parsing tool calls with very long JSON arguments."""
        # Create a long JSON string
        long_data = {"data": ["item_" + str(i) for i in range(1000)]}
        import json

        long_json = json.dumps(long_data)
        response = f"[TOOL_CALLS]process_data[ARGS]{long_json}"

        tool_calls = parse_tool_calls(response)
        assert len(tool_calls) == 1
        assert tool_calls[0].function["name"] == "process_data"

    def test_tool_call_with_unicode_characters(self):
        """Test parsing tool calls with Unicode characters."""
        response = '[TOOL_CALLS]translate_text[ARGS]{"text": "Hello 世界! 🌍", "target_lang": "français"}'
        tool_calls = parse_tool_calls(response)

        assert len(tool_calls) == 1
        assert tool_calls[0].function["name"] == "translate_text"

        import json

        args = json.loads(tool_calls[0].function["arguments"])
        assert args["text"] == "Hello 世界! 🌍"
        assert args["target_lang"] == "français"

    def test_tool_call_with_unbalanced_braces_in_strings(self):
        """Test tool calls with unbalanced braces inside string values."""
        response = '[TOOL_CALLS]format_template[ARGS]{"template": "Hello {name}, welcome to {place}!", "vars": {"name": "John", "place": "Paris"}}'
        tool_calls = parse_tool_calls(response)

        assert len(tool_calls) == 1
        assert tool_calls[0].function["name"] == "format_template"

    def test_partial_tool_call_at_end_of_response(self):
        """Test handling of incomplete tool call at end of response."""
        response = '[TOOL_CALLS]incomplete_function[ARGS]{"param": "value"'
        tool_calls = parse_tool_calls(response)

        # Should handle gracefully and return empty list
        assert len(tool_calls) == 0

    def test_large_response(self):
        """Test handling of large responses with real-world complex content.

        This test uses a realistic codebase investigation report with complex formatting,
        code blocks, configuration examples, and special characters to ensure the parser
        can handle real-world tool call scenarios.
        """
        # Real-world tool call with complex content including code blocks, special chars, etc.
        response = '[TOOL_CALLS]create_new_file[ARGS]{"filepath": "CODEBASE_INVESTIGATION.md", "contents": "# Codebase Investigation Report\\n\\n## Overview\\nThis document summarizes the investigation of the MLX Server Nano codebase, including its structure, configuration, and potential areas for improvement.\\n\\n## Repository Structure\\nThe repository has the following structure:\\n\\n```\\n.github/\\n.python-version\\nREADME.md\\nTESTING.md\\ndebug_caching.py\\ndebug_mlx_functions.py\\ndebug_stream_generate.py\\npyproject.toml\\npytest.ini\\nsimple_cache_test.py\\nsrc/\\n  mlx_server_nano/\\n    __init__.py\\n    app.py\\n    config.py\\n    logging_config.py\\n    main.py\\n    model_manager/\\n    schemas.py\\ntest_caching_manual.py\\ntest_real_caching.py\\ntest_streaming_cache.py\\ntests/\\n```\\n\\n## Configuration Files\\n\\n### pytest.ini\\nThe `pytest.ini` file is well-configured with clear test organization and markers:\\n\\n```ini\\n[pytest]\\ntestpaths = tests\\npython_files = test_*.py *_test.py\\npython_classes = Test*\\npython_functions = test_*\\naddopts = -v --tb=short --strict-markers --disable-warnings -ra\\nmarkers =\\n    unit: Unit tests that don\'t require external dependencies\\n    integration: Integration tests that may require models or external services\\n    slow: Tests that take longer to run\\n    memory: Tests that monitor memory usage\\n    api: Tests for API endpoints\\n    model: Tests that involve model loading/unloading\\nasyncio_mode = auto\\nfilterwarnings =\\n    ignore::DeprecationWarning\\n    ignore::PendingDeprecationWarning\\n```\\n\\n### pyproject.toml\\nThe `pyproject.toml` file includes project metadata, dependencies, and build system configuration:\\n\\n```toml\\n[project]\\nname = \\"mlx-server-nano\\"\\nversion = \\"0.1.0\\"\\ndescription = \\"Add your description here\\"\\nreadme = \\"README.md\\"\\nauthors = [{ name = \\"graelo\\", email = \\"graelo@graelo.cc\\" }]\\nrequires-python = \\">=3.12\\"\\ndependencies = [\\n  \\"fastapi>=0.116.1\\",\\n  \\"mistral-common>=1.8.3\\",\\n  \\"mlx-lm\\",\\n  \\"uvicorn[standard]>=0.35.0\\",\\n  \\"typer>=0.12.3\\",\\n]\\n\\n[project.scripts]\\nmlx-server-nano = \\"mlx_server_nano.main:app_cli\\"\\n\\n[build-system]\\nrequires = [\\"uv_build>=0.8.4,<0.9.0\\"]\\nbuild-backend = \\"uv_build\\"\\n\\n[tool.uv.sources]\\nmlx-lm = { git = \\"https://github.com/graelo/mlx-lm\\", branch = \\"feat/mistral\\" }\\n\\n[dependency-groups]\\ndev = [\\"psutil>=7.0.0\\", \\"ruff>=0.7.0\\", \\"pyright>=1.1.0\\"]\\ntest = [\\n  \\"pytest>=8.0.0\\",\\n  \\"pytest-asyncio>=0.21.0\\",\\n  \\"pytest-cov>=4.0.0\\",\\n  \\"pytest-mock>=3.10.0\\",\\n  \\"httpx>=0.24.0\\",          # For TestClient\\n]\\n\\n[tool.coverage.run]\\nsource = [\\"src/mlx_server_nano\\"]\\nomit = [\\"*/tests/*\\", \\"*/__pycache__/*\\", \\"*/venv/*\\", \\"*/.venv/*\\"]\\n\\n[tool.coverage.report]\\nexclude_lines = [\\n  \\"pragma: no cover\\",\\n  \\"def __repr__\\",\\n  \\"raise AssertionError\\",\\n  \\"raise NotImplementedError\\",\\n  \\"if __name__ == .__main__.:\\",\\n  \\"if TYPE_CHECKING:\\",\\n]\\n```\\n\\n## Main Application Code\\n\\n### main.py\\nThe `main.py` file provides the CLI entry point for the application:\\n\\n```python\\nimport os\\nimport logging\\nimport typer\\nimport uvicorn\\nfrom .config import config\\n\\napp_cli = typer.Typer()\\n\\n@app_cli.command()\\ndef serve(\\n    host: str = typer.Option(config.host, help=\\"Host to bind to\\"),\\n    port: int = typer.Option(config.port, help=\\"Port to bind to\\"),\\n    log_level: str = typer.Option(\\n        config.log_level, help=\\"Log level\\", show_choices=True, case_sensitive=False\\n    ),\\n    reload: bool = typer.Option(False, help=\\"Enable auto-reload for development\\"),\\n):\\n    \\"\\"\\"Start the MLX Server Nano FastAPI server.\\"\\"\\"\\n\\n    # Update config singleton\\n    config.host = host\\n    config.port = port\\n    config.log_level = log_level\\n\\n    # Set up logging first\\n    log_level_val = getattr(logging, config.log_level, logging.INFO)\\n    logging.basicConfig(\\n        level=log_level_val,\\n        format=\\"%(asctime)s - %(name)s - %(levelname)s - %(message)s\\",\\n        handlers=[logging.StreamHandler()],\\n    )\\n    logging.getLogger(\\"mlx_server_nano\\").setLevel(log_level_val)\\n    logging.getLogger(\\"uvicorn\\").setLevel(logging.INFO)\\n\\n    print(f\\"Starting MLX Server Nano on {config.host}:{config.port}\\")\\n    print(f\\"Log level: {config.log_level}\\")\\n    print(\\"Using native MLX-LM integration (no custom templates)\\")\\n    print(f\\"Using Hugging Face cache (HF_HOME: {os.environ.get(\'HF_HOME\', \'default\')})\\")\\n\\n    uvicorn.run(\\n        \\"mlx_server_nano.app:app\\",\\n        host=config.host,\\n        port=config.port,\\n        log_level=config.log_level.lower(),\\n        reload=reload,\\n    )\\n```\\n\\n## Potential Improvements\\n\\n### Code Documentation\\n- Add more detailed docstrings to functions and classes to explain their purpose and usage.\\n- Consider adding type hints to functions for better code documentation and IDE support.\\n\\n### Error Handling\\n- Add validation for configuration values to catch invalid settings early.\\n- Add error handling for cases where environment variables might have invalid values.\\n\\n### Testing\\n- Ensure that tests cover different scenarios, including edge cases and error conditions.\\n- Consider adding integration tests to verify the interaction between different components.\\n\\n### Configuration Management\\n- Add a method to validate configuration values after loading to catch any invalid settings early.\\n- Consider adding a `classifiers` section to the `pyproject.toml` to provide more metadata about the project.\\n\\n### Logging\\n- Add more specific log levels for different components or modules.\\n- Add support for log rotation or file-based logging for production environments.\\n\\n## Conclusion\\nThe MLX Server Nano codebase is well-structured and follows good practices for configuration management, logging, and error handling. The potential improvements identified in this report can help enhance the maintainability, readability, and robustness of the codebase."}'
        tool_calls = parse_tool_calls(response)

        assert len(tool_calls) == 1, "Should parse exactly one tool call"
        assert tool_calls[0].function["name"] == "create_new_file"
        assert tool_calls[0].type == "function"

        # Verify the content was parsed correctly
        import json

        args = json.loads(tool_calls[0].function["arguments"])
        assert args["filepath"] == "CODEBASE_INVESTIGATION.md"
        assert "Codebase Investigation Report" in args["contents"]
        assert "pyproject.toml" in args["contents"]
        assert "```python" in args["contents"]  # Verify code blocks are preserved
        assert "mlx-server-nano" in args["contents"]  # Verify project name is preserved
