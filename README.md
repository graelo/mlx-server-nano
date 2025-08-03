# MLX Server Nano

A lightweight, OpenAI-compatible API server for running language models on Apple Silicon using MLX.

## Features

- **OpenAI API Compatible**: Drop-in replacement for OpenAI's chat completions API
- **Apple Silicon Optimized**: Built specifically for Apple Silicon using MLX framework
- **Tool Calling Support**: Full support for function calling with model-specific parsers
- **Model Management**: Automatic model loading/unloading with configurable idle timeout
- **Multi-Model Support**: Supports Qwen3 and Devstral models with appropriate chat templates

## Installation

```bash
# Install from source
pip install -e .

# Or using uv
uv pip install -e .
```

## Usage

### Start the server

```bash
# Start with default settings
mlx-server-nano

# Start with custom host and port
mlx-server-nano --host 127.0.0.1 --port 9000

# Start with custom model directory
mlx-server-nano --model-cache-dir /path/to/models

# Start with debug logging
mlx-server-nano --log-level DEBUG

# Start with auto-reload for development
mlx-server-nano --reload

# Show all options
mlx-server-nano --help
```

The server will start on `http://localhost:8000` by default.

### Command Line Options

- `--host`: Host to bind to (default: 0.0.0.0)
- `--port`: Port to bind to (default: 8000)  
- `--model-cache-dir`: Directory containing models (default: models)
- `--log-level`: Log level (DEBUG, INFO, WARNING, ERROR, default: INFO)
- `--reload`: Enable auto-reload for development

### Configuration

Configure the server using environment variables:

```bash
export MLX_SERVER_HOST=0.0.0.0
export MLX_SERVER_PORT=8000
export MLX_MODEL_CACHE_DIR=./models
export MLX_MODEL_IDLE_TIMEOUT=300
export MLX_DEFAULT_MAX_TOKENS=512
export MLX_DEFAULT_TEMPERATURE=0.7
export MLX_LOG_LEVEL=INFO
```

### API Usage

The server provides OpenAI-compatible endpoints:

- `POST /v1/chat/completions` - Chat completions with tool calling support
- `GET /v1/models` - List available models
- `GET /health` - Health check

Example request:

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-30b-a3b-instruct-2507",
    "messages": [
      {"role": "user", "content": "Hello!"}
    ],
    "max_tokens": 100,
    "temperature": 0.7
  }'
```

## Model Support

Currently supports:
- **Qwen3** models with custom tool calling format
- **Devstral** models with bracket-based tool calling

Models should be placed in the configured cache directory (default: `./models/`).

## Tool Calling

Supports OpenAI-style function calling with model-specific implementations:

```json
{
  "model": "qwen3-30b-a3b-instruct-2507",
  "messages": [...],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get weather information",
        "parameters": {
          "type": "object",
          "properties": {
            "location": {"type": "string"}
          }
        }
      }
    }
  ]
}
```

## Development

```bash
# Install in development mode
uv pip install -e .

# Run the server
python -m mlx_server_nano.main
```