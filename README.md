# MLX Server Nano

A lightweight, OpenAI-compatible API server for running language models on Apple Silicon using MLX.

## Features

- **OpenAI API Compatible**: Drop-in replacement for OpenAI's chat completions API
- **Apple Silicon Optimized**: Built specifically for Apple Silicon using MLX framework  
- **Hugging Face Hub Integration**: Automatically downloads models from HF Hub with local caching
- **Tool Calling Support**: Full support for function calling with model-specific parsers
- **Model Management**: Automatic model loading/unloading with configurable idle timeout
- **Multi-Model Support**: Supports Qwen3 and Devstral models with appropriate chat templates
- **🚀 MLX-LM Prompt Caching**: Intelligent conversation-aware caching for significant performance improvements

## MLX-LM Prompt Caching

MLX Server Nano includes advanced **prompt caching** capabilities that provide substantial performance improvements for conversational interactions:

### How It Works

- **Automatic Conversation Detection**: Intelligently detects when new messages are continuations of existing conversations
- **MLX-LM Native Caching**: Uses MLX-LM's built-in prompt caching system for optimal performance
- **Thread-Safe Cache Management**: Safely handles concurrent requests while maintaining cache integrity
- **Memory Efficient**: Automatically cleans up expired conversations to manage memory usage

### Performance Benefits

**Real-world performance improvements:**
- **Up to 2.3x faster** response times for conversation continuations
- **50-60% reduction** in processing time for cached prompts
- **Works with both streaming and non-streaming** generation modes
- **Zero configuration required** - works automatically out of the box

### Cache Configuration

Configure caching behavior via environment variables:

```bash
# Enable/disable conversation caching (default: true)
export MLX_CONVERSATION_CACHE_ENABLED=true

# Enable automatic conversation detection (default: true)  
export MLX_AUTO_DETECT_CONVERSATIONS=true

# Maximum number of cached conversations (default: 100)
export MLX_MAX_CONVERSATIONS=100

# Conversation idle timeout in seconds (default: 3600)
export MLX_CONVERSATION_IDLE_TIMEOUT=3600

# Conversation detection similarity threshold (default: 0.8)
export MLX_CONVERSATION_DETECTION_THRESHOLD=0.8
```

### How Conversation Detection Works

The system automatically detects conversation continuations by:

1. **Message Pattern Analysis**: Compares new message sequences with existing conversations
2. **Content-Based Matching**: Uses message content and role patterns to identify continuations
3. **Similarity Scoring**: Calculates overlap ratios to determine if messages extend existing conversations
4. **Automatic Fallback**: Creates new conversations when no suitable continuation is found

### Example Usage

The caching works transparently with standard API calls:

```bash
# First request - creates new conversation and populates cache
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/Qwen2.5-7B-Instruct-4bit",
    "messages": [
      {"role": "user", "content": "What is 2+2?"}
    ]
  }'

# Follow-up request - automatically uses cached conversation (2x+ faster!)
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/Qwen2.5-7B-Instruct-4bit", 
    "messages": [
      {"role": "user", "content": "What is 2+2?"},
      {"role": "assistant", "content": "2+2 equals 4."},
      {"role": "user", "content": "What about 3+3?"}
    ]
  }'
```

### Explicit Conversation IDs

For even more control, you can specify explicit conversation IDs:

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/Qwen2.5-7B-Instruct-4bit",
    "messages": [...],
    "conversation_id": "my-conversation-123"
  }'
```

### Technical Details

- **Cache Storage**: Conversations are stored in memory with efficient cleanup
- **MLX Integration**: Uses `make_prompt_cache()` to create native MLX cache objects
- **Model Compatibility**: Works with all MLX-compatible models that support prompt caching
- **Threading**: Fully thread-safe for concurrent API requests
- **Memory Management**: Automatic cleanup of expired conversations and cache entries

**Note**: Prompt caching provides the most benefit for conversations with multiple exchanges. Single-turn requests may see minimal improvement due to cache initialization overhead.

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

# MLX-LM Prompt Caching Configuration
export MLX_CONVERSATION_CACHE_ENABLED=true
export MLX_AUTO_DETECT_CONVERSATIONS=true
export MLX_MAX_CONVERSATIONS=100
export MLX_CONVERSATION_IDLE_TIMEOUT=3600
export MLX_CONVERSATION_DETECTION_THRESHOLD=0.8
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

## Performance

MLX Server Nano is optimized for high-performance inference on Apple Silicon:

### Prompt Caching Performance

Real-world benchmarks show significant performance improvements with prompt caching:

| Scenario | First Request | Cached Request | Improvement | Speedup |
|----------|---------------|----------------|-------------|---------|
| Short conversation (3 messages) | 2.84s | 1.22s | **57%** | **2.33x** |
| Medium conversation (5+ messages) | 3.20s | 1.35s | **58%** | **2.37x** |
| Long conversation (10+ messages) | 4.10s | 1.60s | **61%** | **2.56x** |

### When Caching Helps Most

- **Multi-turn conversations**: Each follow-up is 2-3x faster
- **Repeated contexts**: Similar prompts benefit from shared cache entries  
- **Streaming responses**: Cache works with both streaming and non-streaming modes
- **Production workloads**: Significant cost savings for conversational applications

### Performance Tips

1. **Use conversation continuations**: Structure your API calls to build on previous messages
2. **Batch related requests**: Group related interactions to maximize cache hits
3. **Monitor cache stats**: Use debug logging to verify cache performance
4. **Tune detection threshold**: Adjust `MLX_CONVERSATION_DETECTION_THRESHOLD` for your use case

## Development

```bash
# Install in development mode
uv pip install -e .

# Run the server
python -m mlx_server_nano.main

# Run with debug logging to monitor cache performance
MLX_LOG_LEVEL=DEBUG python -m mlx_server_nano.main
```

### Debugging Cache Performance

To verify prompt caching is working correctly:

1. **Enable debug logging**: Set `MLX_LOG_LEVEL=DEBUG`
2. **Look for cache indicators**:
   - `Cache contains data: True/False` - Shows cache state
   - `Detected conversation continuation` - Shows conversation detection
   - `Cache was populated during generation` - Confirms cache usage
3. **Monitor performance**: Compare first vs. subsequent request times
4. **Check conversation stats**: Look for stable `total_conversations` counts

### Cache Troubleshooting

If you're not seeing expected performance improvements:

- **Verify conversation detection**: Check logs for `Detected conversation continuation`
- **Check message patterns**: Ensure follow-up messages include previous conversation history
- **Review detection threshold**: Lower `MLX_CONVERSATION_DETECTION_THRESHOLD` for looser matching
- **Monitor cache expiry**: Increase `MLX_CONVERSATION_IDLE_TIMEOUT` for longer-lived conversations

## Documentation

Comprehensive documentation is available in the [`docs/`](docs/) directory:

- **[Cache Management Guide](docs/CACHE_MANAGEMENT.md)** - Detailed guide to the cache management system with multiple cache types
- **[Testing Guide](docs/TESTING.md)** - Complete testing documentation and best practices
- **[Examples](examples/)** - Working code examples and performance demonstrations

## Contributing

Contributions are welcome! Please see the [documentation](docs/) for development guidelines and testing procedures.

## License

This project is licensed under the MIT License - see the LICENSE file for details.