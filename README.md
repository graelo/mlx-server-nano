# MLX Server Nano

A lightweight, OpenAI-compatible API server for running language models on Apple Silicon using MLX.

## Features

- **OpenAI API Compatible**: Drop-in replacement for OpenAI's chat completions API
- **Apple Silicon Optimized**: Built specifically for Apple Silicon using MLX framework  
- **Hugging Face Hub Integration**: Automatically downloads models from HF Hub with local caching
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
- `--log-level`: Log level (DEBUG, INFO, WARNING, ERROR, default: INFO)
- `--reload`: Enable auto-reload for development

### Environment Variables

Configure Hugging Face cache behavior using standard HF environment variables:

```bash
export HF_HOME=/path/to/hf/cache          # Set HF cache directory
export HUGGINGFACE_HUB_CACHE=/path/cache  # Alternative cache location
export HF_HUB_OFFLINE=1                   # Use offline mode (cache only)
export HF_TOKEN=your_token                # For private/gated models
```

Configure the MLX server:

```bash
export MLX_SERVER_HOST=0.0.0.0
export MLX_SERVER_PORT=8000
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
    "model": "mlx-community/Qwen2.5-7B-Instruct-4bit",
    "messages": [
      {"role": "user", "content": "Hello!"}
    ],
    "max_tokens": 100,
    "temperature": 0.7
  }'
```

## Model Support

The server uses **Hugging Face Hub exclusively** with standard HF caching:

1. **Direct HF Hub Access**: Models are loaded directly using their HF Hub names
2. **Standard HF Cache**: Uses Hugging Face's standard cache system (respects HF_HOME, etc.)
3. **No Custom Cache**: No additional local caching beyond what HF provides
4. **MLX-Optimized Models**: Works best with models from the `mlx-community` organization
5. **On-Demand Loading**: Any MLX-compatible model can be loaded on-demand by specifying its name

**You can use any MLX-compatible model from Hugging Face Hub** by specifying its full name in your requests. Popular choices include:
- `mlx-community/Qwen2.5-7B-Instruct-4bit`
- `mlx-community/Qwen2.5-14B-Instruct-4bit`
- `mlx-community/Meta-Llama-3.1-8B-Instruct-4bit`
- `mlx-community/Mistral-7B-Instruct-v0.3-4bit`
- `mlx-community/CodeLlama-7b-Instruct-hf-4bit`

The MLX library will automatically download models from HF Hub on first use and cache them using Hugging Face's standard caching mechanism.

**Note**: The `/v1/models` endpoint returns an empty list since models are loaded on-demand. Simply specify any MLX-compatible model name in your chat completion requests.

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