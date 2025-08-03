# Testing Guide for MLX Server Nano

This document provides comprehensive guidance for testing MLX Server Nano, including setup, running tests, and understanding the test architecture.

## Overview

The test suite for MLX Server Nano is designed to provide:
- **Comprehensive coverage** of all core functionality
- **Fast execution** through strategic mocking
- **Memory management verification** without OS dependency
- **API compatibility testing** with OpenAI standards
- **Integration testing** for real-world scenarios
- **Regression testing** for production confidence

## Test Structure

```
tests/
├── conftest.py              # Global test configuration and fixtures
├── pytest.ini              # Pytest configuration
├── unit/                    # Unit tests (fast, isolated)
│   ├── test_config.py       # Configuration management tests
│   ├── test_schemas.py      # Pydantic schema validation tests
│   ├── test_tool_calling.py # Tool calling parser tests
│   ├── test_model_manager.py # Model management unit tests
│   └── test_memory_management.py # Memory management tests
├── integration/             # Integration tests (slower, more realistic)
│   ├── test_api_endpoints.py # FastAPI endpoint tests
│   └── test_model_management.py # Model lifecycle integration tests
└── fixtures/
    ├── test_data.py         # Common test data and fixtures
    └── __init__.py
```

## Test Categories and Markers

Tests are organized using pytest markers for selective execution:

- `@pytest.mark.unit` - Fast unit tests, heavily mocked
- `@pytest.mark.integration` - Integration tests with real component interaction
- `@pytest.mark.api` - API endpoint and compatibility tests
- `@pytest.mark.model` - Model loading/unloading tests
- `@pytest.mark.memory` - Memory management and cleanup tests
- `@pytest.mark.slow` - Tests that take longer to execute

## Quick Start

### 1. Install Test Dependencies

```bash
# Install test dependencies
uv sync --group test
```

### 2. Run Basic Tests

```bash
# Run basic working tests (recommended for quick verification)
uv run pytest tests/test_basic_validation.py tests/unit/test_config.py -v

# Run all unit tests
uv run pytest -m unit -v

# Run with coverage
uv run pytest -m unit --cov=src/mlx_server_nano --cov-report=html -v
```

### 3. Run Specific Test Categories

```bash
# API endpoint tests
uv run pytest -m api -v

# Memory management tests
uv run pytest -m memory -v

# Model management tests
uv run pytest -m model -v

# Integration tests
uv run pytest -m integration -v
```

## Detailed Test Commands

### Using pytest directly

Standard pytest commands work with our configured markers:

```bash
# Basic commands
uv run pytest -m unit -v           # Unit tests
uv run pytest -m integration -v    # Integration tests
uv run pytest -v                   # All tests
uv run pytest -m "not slow" -v     # Fast tests only (excludes slow)

# With options
uv run pytest -m unit -v --cov=src/mlx_server_nano --cov-report=html
uv run pytest --cov=src/mlx_server_nano --cov-report=html

# Specific test categories
uv run pytest -m memory -v         # Memory management tests
uv run pytest -m api -v            # API endpoint tests
uv run pytest -m model -v          # Model lifecycle tests

# Utility commands
uv sync --group test              # Install test dependencies
uv run pytest --cov=src/mlx_server_nano --cov-report=html  # Generate coverage report
rm -rf .pytest_cache htmlcov .coverage  # Clean test artifacts
```

### Using Pytest Directly

For more control, use pytest directly:

```bash
# Run all tests
uv run pytest

# Run specific test files
uv run pytest tests/unit/test_config.py
uv run pytest tests/integration/

# Run tests with markers
uv run pytest -m "unit"
uv run pytest -m "integration and not slow"
uv run pytest -m "memory"

# Run with coverage
uv run pytest --cov=src/mlx_server_nano --cov-report=html

# Run with verbose output
uv run pytest -v

# Run specific test method
uv run pytest tests/unit/test_config.py::TestServerConfig::test_default_values
```

## Test Architecture

### Mocking Strategy

The test suite uses strategic mocking to achieve fast, reliable tests:

1. **MLX Components**: `mlx_lm.load`, `mlx_lm.generate`, `mlx.core.clear_cache` are mocked
2. **External Dependencies**: Network calls, file operations are mocked
3. **System Resources**: Memory monitoring uses controlled mocks
4. **Background Tasks**: AsyncIO tasks are properly managed

### Key Fixtures

From `conftest.py`:

- `test_config` - Test-specific configuration with reduced timeouts
- `test_env_vars` - Environment variables for testing
- `client` - FastAPI test client
- `clean_model_manager` - Clean model manager state
- `mock_mlx_model` - Mocked MLX model and tokenizer
- `sample_chat_request` - Basic chat completion request
- `started_unloader` - Background task management

### Memory Testing Approach

Memory tests focus on **functional verification** rather than OS memory measurement:

1. **Cache State Verification**: Ensure models are properly cached/cleared
2. **Cleanup Function Calls**: Verify garbage collection and cache clearing
3. **Reference Management**: Check that object references are properly released
4. **Lifecycle Management**: Test complete load/unload cycles

This approach is faster and more reliable than monitoring actual memory usage.

## Test Coverage Areas

### Core Functionality
- ✅ Configuration management with environment variables
- ✅ Pydantic schema validation
- ✅ Model loading and caching
- ✅ Model unloading and memory cleanup
- ✅ Background task lifecycle management
- ✅ Tool calling for different model types (Devstral, Qwen)

### API Compatibility
- ✅ OpenAI-compatible request/response formats
- ✅ Chat completion endpoints (streaming and non-streaming)
- ✅ Error handling and validation
- ✅ Tool calling integration
- ✅ Usage tracking and token counting

### Integration Scenarios
- ✅ Complete request/response cycles
- ✅ Model lifecycle with background unloader
- ✅ Concurrent request handling
- ✅ Configuration integration with environment variables
- ✅ Error propagation through the stack

### Memory Management
- ✅ Model cache state management
- ✅ Automatic unloading after timeout
- ✅ Memory cleanup verification
- ✅ Cache replacement when switching models
- ✅ Error handling during cleanup

## Performance Testing

### Memory Management Tests

```bash
# Run memory-specific tests
uv run pytest -m memory -v

# Test functional memory management
uv run pytest -m "memory" -v
```

These tests verify:
- Model cache state transitions
- Cleanup function execution
- Background task timing
- Concurrent access safety

### Load Testing

For load testing with real models:

```bash
# Run integration tests (may load real models)
uv run pytest -m integration -v

# Run slow tests that may involve actual model loading
uv run pytest -m slow -v
```

## Continuous Integration

### GitHub Actions / CI Setup

Example CI configuration:

```yaml
- name: Install dependencies
  run: uv sync --group test

- name: Run fast tests
  run: uv run pytest -m "not slow" -v --cov=src/mlx_server_nano --cov-report=html

- name: Run integration tests
  run: uv run pytest -m integration -v

- name: Upload coverage
  uses: codecov/codecov-action@v3
```

### Local Pre-commit

```bash
# Before committing, run:
uv run pytest -m "not slow" -v --cov=src/mlx_server_nano --cov-report=html
uv run ruff check
```

## Debugging Tests

### Running Individual Tests

```bash
# Run a specific test file
uv run pytest tests/unit/test_model_manager.py -v

# Run a specific test class
uv run pytest tests/unit/test_model_manager.py::TestLoadModel -v

# Run a specific test method
uv run pytest tests/unit/test_config.py::TestServerConfig::test_from_env_with_variables -v -s
```

### Debug Mode

```bash
# Run with Python debugger on failure
uv run pytest --pdb

# Keep temporary files for inspection
uv run pytest --basetemp=/tmp/pytest-debug

# Show print statements
uv run pytest -s

# Show locals in tracebacks
uv run pytest --tb=long -vv
```

### Log Output

```bash
# See debug logs during tests
MLX_LOG_LEVEL=DEBUG uv run pytest -v -s

# Capture log output
uv run pytest --log-cli-level=DEBUG --log-cli-format='%(asctime)s [%(levelname)8s] %(name)s: %(message)s'
```

## Adding New Tests

### Unit Test Example

```python
@pytest.mark.unit
def test_new_functionality(clean_model_manager):
    """Test description."""
    # Arrange
    # Act
    # Assert
```

### Integration Test Example

```python
@pytest.mark.integration
@pytest.mark.asyncio
async def test_async_functionality(started_unloader):
    """Test description."""
    # Async test implementation
```

### Memory Test Example

```python
@pytest.mark.memory
def test_memory_management(clean_model_manager):
    """Test functional memory management."""
    # Verify cache state changes
    # Don't rely on OS memory measurements
```

## Best Practices

### Test Design
1. **Use appropriate markers** for test categorization
2. **Mock external dependencies** for unit tests
3. **Use fixtures** for common setup/teardown
4. **Test edge cases** and error conditions
5. **Keep tests focused** on single functionality

### Performance
1. **Prefer unit tests** for speed
2. **Use mocks extensively** to avoid real model loading
3. **Clean up resources** with proper fixtures
4. **Group slow tests** with appropriate markers

### Reliability
1. **Avoid timing dependencies** in tests
2. **Use deterministic test data**
3. **Handle async operations properly**
4. **Clean state between tests**

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure `uv sync --group test` was run
2. **Model Loading Failures**: Check if tests are properly mocked
3. **Async Test Issues**: Use `@pytest.mark.asyncio` decorator
4. **State Leakage**: Use `clean_model_manager` fixture
5. **Environment Variables**: Use `test_env_vars` fixture

### Environment Issues

```bash
# Reset test environment
rm -rf .pytest_cache htmlcov .coverage
uv sync --group test

# Check test environment
uv run python -c "import pytest; print(pytest.__version__)"
uv run python -c "import mlx_server_nano; print('Import successful')"
```

### Memory Test Issues

If memory tests are failing:
1. Check that mocks are properly configured
2. Verify fixture usage for clean state
3. Ensure tests focus on functional verification, not OS memory
4. Use `clean_model_manager` fixture consistently

## Coverage Reports

Generate coverage reports:

```bash
# Generate HTML coverage report
uv run pytest --cov=src/mlx_server_nano --cov-report=html

# View report
open htmlcov/index.html
```

Coverage targets:
- **Overall**: > 90%
- **Core modules**: > 95%
- **Critical paths**: 100%

## Contributing to Tests

When adding new features:

1. **Add unit tests** for new functions/classes
2. **Add integration tests** for new endpoints/workflows  
3. **Update fixtures** if new test data is needed
4. **Add markers** for appropriate test categorization
5. **Update this documentation** for significant changes

The test suite is designed to grow with the project while maintaining fast execution and reliable results.
