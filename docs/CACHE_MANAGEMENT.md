# Cache Management in MLX Server Nano

This document explains the comprehensive cache management system implemented in MLX Server Nano, including the different cache types available, their use cases, and how to configure them.

## Overview

MLX Server Nano implements a sophisticated cache management system that supports multiple cache types from the MLX framework. This system is designed to optimize memory usage, improve response times, and provide flexibility for different use cases and hardware constraints.

## Why Multiple Cache Types?

Different use cases require different caching strategies:

1. **Memory Constraints**: Some environments have limited memory and need efficient caching
2. **Performance Requirements**: Different cache types offer varying performance characteristics
3. **Conversation Length**: Long conversations may benefit from rotating or chunked caches
4. **Hardware Optimization**: Different cache types work better on different hardware configurations
5. **Use Case Specific**: Streaming, batch processing, and interactive chat have different needs

## Available Cache Types

The MLX Server Nano implements five different cache types, each optimized for specific scenarios:

### 1. KVCache (Standard Cache)
**Best for: General purpose, most conversations**

```python
cache_type = "kv"
```

- **Description**: The standard key-value cache implementation from MLX
- **Memory Usage**: Standard memory footprint
- **Performance**: Excellent for most use cases
- **Use Cases**: 
  - General chat conversations
  - Most production deployments
  - Default choice for balanced performance

**Characteristics:**
- Linear memory growth with conversation length
- Fast access times
- No memory optimization features
- Reliable and well-tested

### 2. QuantizedKVCache (Memory-Optimized Cache)
**Best for: Memory-constrained environments, long conversations**

```python
cache_type = "quantized_kv"
```

- **Description**: A quantized version of the KV cache that reduces memory usage
- **Memory Usage**: Significantly reduced (typically 50-75% less memory)
- **Performance**: Slight computation overhead, major memory savings
- **Use Cases**:
  - Devices with limited RAM
  - Very long conversations
  - Batch processing multiple conversations
  - Cost-sensitive deployments

**Characteristics:**
- Reduced precision for memory efficiency
- Automatic compression/decompression
- Ideal for memory-bound scenarios
- Minimal impact on output quality

### 3. RotatingKVCache (Fixed Memory Cache)
**Best for: Predictable memory usage, streaming applications**

```python
cache_type = "rotating_kv" 
# Configuration: max_size parameter controls rotation
```

- **Description**: A fixed-size cache that rotates old entries when full
- **Memory Usage**: Fixed maximum size, predictable memory footprint
- **Performance**: Consistent performance, no memory growth
- **Use Cases**:
  - Streaming applications
  - Long-running conversations
  - Memory-predictable deployments
  - Systems with strict memory limits

**Characteristics:**
- Fixed memory ceiling
- FIFO (First In, First Out) rotation
- Prevents memory leaks in long conversations
- May lose early conversation context

### 4. ChunkedKVCache (Segmented Processing Cache)
**Best for: Large batches, parallel processing**

```python
cache_type = "chunked_kv"
# Configuration: chunk_size parameter controls chunking
```

- **Description**: Processes cache in chunks for better memory locality
- **Memory Usage**: Optimized for batch processing
- **Performance**: Excellent for large batches and parallel processing
- **Use Cases**:
  - Batch processing multiple requests
  - Large document processing
  - Parallel conversation handling
  - High-throughput scenarios

**Characteristics:**
- Segmented memory allocation
- Better cache locality
- Optimized for vectorized operations
- Efficient parallel processing

### 5. ConcatenateKVCache (Unified Context Cache)
**Best for: Context preservation, multi-turn conversations**

```python
cache_type = "concatenate_kv"
```

- **Description**: Concatenates cache entries for unified context handling
- **Memory Usage**: Optimized for context preservation
- **Performance**: Excellent context continuity
- **Use Cases**:
  - Multi-turn conversations requiring full context
  - Document Q&A with context preservation
  - Complex reasoning tasks
  - Context-sensitive applications

**Characteristics:**
- Maintains full conversation context
- Optimized concatenation operations
- Better context understanding
- Ideal for reasoning tasks

## Configuration

### Setting Cache Type

Cache type can be configured through environment variables or configuration:

```bash
# Environment variable
export MLX_CACHE_TYPE="quantized_kv"

# Or in your configuration
MLX_CACHE_TYPE=rotating_kv mlx-server-nano
```

### Cache-Specific Parameters

Some cache types support additional configuration:

```python
# Rotating cache with custom size
cache_config = {
    "type": "rotating_kv",
    "max_size": 1000  # Maximum number of cached tokens
}

# Chunked cache with custom chunk size
cache_config = {
    "type": "chunked_kv", 
    "chunk_size": 512  # Process in chunks of 512 tokens
}
```

## Cache Management Operations

### Basic Operations

```python
from mlx_server_nano.model_manager.cache_manager import PromptCacheManager

# Initialize cache manager
cache_manager = PromptCacheManager()

# Create a specific cache type
cache = cache_manager.create_cache("quantized_kv")

# Save cache to disk
cache_manager.save_cache(cache, "conversation_001.cache")

# Load cache from disk
loaded_cache = cache_manager.load_cache("conversation_001.cache")

# Optimize cache (memory cleanup)
cache_manager.optimize_cache(cache)

# Get cache statistics
stats = cache_manager.get_cache_stats(cache)
print(f"Cache size: {stats['size']}, Memory usage: {stats['memory_mb']}MB")
```

### Cache Persistence

All cache types support persistence for conversation continuity:

```python
# Save conversation state
cache_manager.save_cache(current_cache, "user_session_123.cache")

# Restore conversation state
restored_cache = cache_manager.load_cache("user_session_123.cache")
```

### Cache Optimization

Regular cache optimization helps maintain performance:

```python
# Optimize memory usage
optimized_cache = cache_manager.optimize_cache(cache)

# Clear unused cache entries
cache_manager.clear_unused_entries(cache)

# Get optimization recommendations
recommendations = cache_manager.get_optimization_recommendations(cache)
```

## Performance Characteristics

| Cache Type | Memory Usage | Speed | Context Preservation | Best For |
|------------|--------------|--------|---------------------|----------|
| KVCache | Standard | Fastest | Full | General use |
| QuantizedKVCache | Low (50-75% reduction) | Fast | Full | Memory-constrained |
| RotatingKVCache | Fixed/Predictable | Fast | Partial | Streaming |
| ChunkedKVCache | Optimized | Very Fast | Full | Batch processing |
| ConcatenateKVCache | Optimized | Fast | Excellent | Context-heavy |

## Use Case Recommendations

### Production Web Service
**Recommended: `KVCache` or `QuantizedKVCache`**
- Balanced performance and memory usage
- Reliable and well-tested
- Good for most conversation lengths

### Memory-Constrained Environment
**Recommended: `QuantizedKVCache`**
- Significant memory savings
- Minimal performance impact
- Maintains conversation quality

### Streaming Applications
**Recommended: `RotatingKVCache`**
- Predictable memory usage
- No memory growth over time
- Consistent performance

### Batch Processing
**Recommended: `ChunkedKVCache`**
- Optimized for parallel processing
- Excellent throughput
- Memory-efficient for large batches

### Context-Heavy Applications
**Recommended: `ConcatenateKVCache`**
- Superior context preservation
- Optimized for reasoning tasks
- Best for complex conversations

## Monitoring and Debugging

### Cache Statistics

Monitor cache performance with built-in statistics:

```python
stats = cache_manager.get_cache_stats(cache)
print(f"""
Cache Statistics:
- Type: {stats['type']}
- Size: {stats['size']} entries
- Memory: {stats['memory_mb']:.2f} MB
- Hit Rate: {stats['hit_rate']:.2%}
- Last Optimized: {stats['last_optimized']}
""")
```

### Performance Monitoring

```python
# Monitor cache performance over time
performance = cache_manager.monitor_performance(cache, duration=60)
print(f"Average response time: {performance['avg_response_ms']:.2f}ms")
print(f"Memory efficiency: {performance['memory_efficiency']:.2%}")
```

### Health Checks

```python
# Verify cache health
health = cache_manager.check_cache_health(cache)
if not health['healthy']:
    print(f"Cache issues detected: {health['issues']}")
    # Automatic optimization
    cache = cache_manager.optimize_cache(cache)
```

## Best Practices

### 1. Choose the Right Cache Type
- Start with `KVCache` for general use
- Use `QuantizedKVCache` for memory constraints
- Use `RotatingKVCache` for streaming or long-running services
- Use `ChunkedKVCache` for batch processing
- Use `ConcatenateKVCache` for context-heavy applications

### 2. Monitor Memory Usage
```python
# Regular memory monitoring
if cache_manager.get_memory_usage(cache) > memory_threshold:
    cache = cache_manager.optimize_cache(cache)
```

### 3. Implement Cache Persistence
```python
# Save important conversations
if conversation_important:
    cache_manager.save_cache(cache, f"important_{conversation_id}.cache")
```

### 4. Regular Optimization
```python
# Periodic optimization (e.g., every 100 requests)
if request_count % 100 == 0:
    cache = cache_manager.optimize_cache(cache)
```

### 5. Handle Cache Errors Gracefully
```python
try:
    cache = cache_manager.create_cache(cache_type)
except CacheCreationError:
    # Fallback to standard cache
    cache = cache_manager.create_cache("kv")
    logger.warning(f"Failed to create {cache_type}, using standard KVCache")
```

## Migration Guide

### Upgrading from Basic Cache

If you're currently using a basic cache implementation:

1. **Assess your use case** using the recommendations above
2. **Start with QuantizedKVCache** for memory benefits with minimal changes
3. **Test performance** in your environment
4. **Gradually migrate** to specialized cache types as needed

### Configuration Migration

```python
# Old configuration
cache = basic_cache_create()

# New configuration
cache_manager = PromptCacheManager()
cache = cache_manager.create_cache("quantized_kv")
```

## Troubleshooting

### Common Issues

1. **Memory Usage Too High**
   - Switch to `QuantizedKVCache` or `RotatingKVCache`
   - Implement regular cache optimization
   - Monitor memory usage patterns

2. **Performance Degradation**
   - Check cache hit rates
   - Consider `ChunkedKVCache` for batch processing
   - Optimize cache regularly

3. **Context Loss**
   - Use `ConcatenateKVCache` for better context preservation
   - Avoid `RotatingKVCache` for context-sensitive applications
   - Save important conversation states

4. **Cache Creation Failures**
   - Check MLX installation
   - Verify system memory availability
   - Use fallback cache types

### Debug Mode

Enable debug logging for detailed cache operations:

```python
import logging
logging.getLogger('mlx_server_nano.cache_manager').setLevel(logging.DEBUG)
```

## Advanced Features

### Custom Cache Configuration

```python
# Advanced cache configuration
advanced_config = {
    "type": "quantized_kv",
    "quantization_bits": 8,
    "compression_ratio": 0.5,
    "optimization_interval": 1000
}

cache = cache_manager.create_cache_with_config(advanced_config)
```

### Cache Composition

```python
# Use different cache types for different conversation phases
initial_cache = cache_manager.create_cache("kv")  # Fast initial responses
long_cache = cache_manager.create_cache("rotating_kv")  # Memory-bounded long conversations
```

## Future Enhancements

The cache management system is designed for extensibility:

- **Custom cache implementations** can be added
- **Dynamic cache switching** based on conversation characteristics
- **Machine learning-based cache optimization**
- **Distributed caching** for multi-instance deployments

## Conclusion

The MLX Server Nano cache management system provides powerful, flexible caching solutions for various use cases. By choosing the appropriate cache type and following best practices, you can optimize both performance and memory usage for your specific requirements.

For more information, see:
- [Testing Guide](TESTING.md) - How to test cache implementations
- [MLX Documentation](https://ml-explore.github.io/mlx/) - MLX framework details
- [Configuration Guide](../README.md) - Server configuration options
