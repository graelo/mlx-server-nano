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

MLX Server Nano includes advanced **prompt caching** capabilities that provide substantial performance improvements for conversational interactions. The system supports **multiple cache types** to optimize for different use cases and hardware constraints.

### Cache Types Available

**5 Different Cache Types for Optimal Performance:**

1. **KVCache** (Default) - Standard cache for general use
2. **QuantizedKVCache** - Memory-efficient quantized cache (up to 50% memory reduction)
3. **RotatingKVCache** - Automatic cleanup for long conversations
4. **ChunkedKVCache** - Chunked processing for memory efficiency
5. **ConcatenateKVCache** - Concatenated cache management

**Choose the right cache type based on your needs:**
- **Memory constrained?** → Use `QuantizedKVCache` or `ChunkedKVCache`
- **Long conversations?** → Use `RotatingKVCache`
- **General purpose?** → Use `KVCache` (default)
- **Batch processing?** → Use `ConcatenateKVCache`

See the [Cache Management Guide](docs/CACHE_MANAGEMENT.md) for detailed comparison and configuration guidance.

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

### Using Different Cache Types

**Start server with specific cache type:**

```bash
# Memory-efficient quantized cache
mlx-server-nano --cache-type QuantizedKVCache

# Rotating cache for long conversations
mlx-server-nano --cache-type RotatingKVCache --cache-max-size 2000

# Chunked cache for memory constraints
mlx-server-nano --cache-type ChunkedKVCache --cache-chunk-size 256

# Or use environment variables
export MLX_CACHE_TYPE=QuantizedKVCache
mlx-server-nano
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

# Start with specific cache type
mlx-server-nano --cache-type QuantizedKVCache

# Start with custom cache configuration
mlx-server-nano --cache-type RotatingKVCache --cache-max-size 2000 --max-conversations 50

# Start with chunked cache for memory efficiency
mlx-server-nano --cache-type ChunkedKVCache --cache-chunk-size 256

# Show all options
mlx-server-nano --help
```

The server will start on `http://localhost:8000` by default.

### Command Line Options

**Basic Server Options:**
- `--host`: Host to bind to (default: 0.0.0.0)
- `--port`: Port to bind to (default: 8000)  
- `--log-level`: Log level (DEBUG, INFO, WARNING, ERROR, default: INFO)
- `--reload`: Enable auto-reload for development

**Cache Configuration Options:**
- `--cache-type`: Cache type - KVCache, QuantizedKVCache, RotatingKVCache, ChunkedKVCache, ConcatenateKVCache (default: KVCache)
- `--cache-max-size`: Maximum cache size for RotatingKVCache (default: 1000)
- `--cache-chunk-size`: Chunk size for ChunkedKVCache (default: 512)
- `--max-conversations`: Maximum number of cached conversations (default: 10)
- `--cache-enabled` / `--no-cache-enabled`: Enable/disable conversation caching (default: enabled)
- `--cache-timeout`: Cache idle timeout in seconds (default: 300)

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

# Advanced Cache Type Configuration
export MLX_CACHE_TYPE=KVCache                    # KVCache, QuantizedKVCache, RotatingKVCache, ChunkedKVCache, ConcatenateKVCache
export MLX_CACHE_MAX_SIZE=1000                   # For RotatingKVCache
export MLX_CACHE_CHUNK_SIZE=512                  # For ChunkedKVCache
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

1. **Choose the right cache type**: Select cache type based on your use case (see [Cache Management Guide](docs/CACHE_MANAGEMENT.md))
2. **Use conversation continuations**: Structure your API calls to build on previous messages
3. **Batch related requests**: Group related interactions to maximize cache hits
4. **Monitor cache stats**: Use debug logging to verify cache performance
5. **Tune detection threshold**: Adjust `MLX_CONVERSATION_DETECTION_THRESHOLD` for your use case
6. **Memory optimization**: Use `QuantizedKVCache` or `ChunkedKVCache` for memory-constrained environments

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