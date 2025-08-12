# MLX Server Nano Examples

This directory contains example scripts that demonstrate the key features of MLX Server Nano, particularly the advanced prompt caching capabilities.

## Examples

### 🚀 `cache_performance_example.py`
**Demonstrates prompt caching performance improvements**

Shows how MLX Server Nano's conversation-aware caching provides significant speedups for follow-up requests in conversations.

```bash
cd examples
python cache_performance_example.py
```

**What it shows:**
- First request (cold start) vs. cached follow-up request
- Real performance metrics and speedup calculations
- Cache statistics and configuration
- Automatic conversation detection in action

### 🔍 `conversation_detection_example.py`
**Shows how automatic conversation detection works**

Demonstrates the underlying conversation detection logic that enables efficient caching.

```bash
cd examples
python conversation_detection_example.py
```

**What it shows:**
- How the system identifies conversation continuations
- Explicit vs. automatic conversation ID management
- Conversation hash generation and matching
- Cache management and cleanup

### 🌊 `streaming_cache_example.py`
**Demonstrates caching with streaming responses**

Shows how prompt caching works with streaming generation to speed up conversational streaming.

```bash
cd examples
python streaming_cache_example.py
```

**What it shows:**
- Streaming response caching in action
- Performance improvements for streaming conversations
- Real-time cache utilization during streaming
- How to combine streaming and caching for optimal performance

## Requirements

To run these examples, you need:

1. **MLX Server Nano installed:**
   ```bash
   pip install -e .
   ```

2. **An MLX-compatible model available** (examples use `mlx-community/Qwen2.5-7B-Instruct-4bit` by default, but you can change this in the code)

3. **Apple Silicon Mac** with sufficient memory for your chosen model

## Configuration

The examples work with default settings, but you can customize caching behavior via environment variables:

```bash
# Enable/disable caching
export MLX_CONVERSATION_CACHE_ENABLED=true

# Conversation detection threshold (0.0-1.0)
export MLX_CONVERSATION_DETECTION_THRESHOLD=0.8

# Maximum cached conversations
export MLX_MAX_CONVERSATIONS=100

# Cache idle timeout (seconds)
export MLX_CONVERSATION_IDLE_TIMEOUT=3600
```

## Expected Results

With a typical 7B model, you should see:

- **Cache Performance Example**: 1.5-3x speedup for follow-up requests
- **Conversation Detection**: 100% accuracy for simple continuations  
- **Streaming Cache**: Reduced latency for conversation streaming

## Troubleshooting

### No performance improvement?
- Ensure you're using conversation continuations (include previous messages)
- Try longer conversations (3+ exchanges) for bigger benefits
- Check that `MLX_CONVERSATION_CACHE_ENABLED=true`

### Model loading errors?
- Verify the model name exists on Hugging Face
- Ensure you have enough memory for the model
- Try a smaller model like `mlx-community/Qwen2.5-1.5B-Instruct-4bit`

### Import errors?
- Run examples from the `examples/` directory
- Ensure MLX Server Nano is installed: `pip install -e .`
- Check that you're using Python 3.9+

## Learning More

- See the main [README.md](../README.md) for complete documentation
- Check [TESTING.md](../TESTING.md) for running the full test suite
- Review the source code in `src/mlx_server_nano/` for implementation details

These examples are designed to be educational and can be modified to test with your specific models and use cases!
