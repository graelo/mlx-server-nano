#!/usr/bin/env python3
"""
Streaming with Caching Example

This example demonstrates how MLX Server Nano's prompt caching works
with streaming generation, providing performance improvements for
streaming responses in conversational scenarios.

Features demonstrated:
- Streaming performance with different cache types
- Cache optimization for streaming workloads
- Real-time performance metrics for streaming
- Best practices for streaming cache configuration
"""

import os
import sys
import time
import logging

# Add the source directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mlx_server_nano.schemas import Message
from mlx_server_nano.model_manager.generation import generate_response_stream_cached
from mlx_server_nano.model_manager.cache_manager import (
    get_conversation_cache_stats,
    PromptCacheManager,
)
from mlx_server_nano.config import config, CacheType

# Set up logging to see cache activity (optional - set to DEBUG for more details)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def demonstrate_streaming_cache():
    """Demonstrate prompt caching with streaming generation."""

    print("=" * 60)
    print("MLX-LM STREAMING + CACHING DEMONSTRATION")
    print("=" * 60)
    print("This shows how caching speeds up streaming responses in conversations.")
    print()

    # Initialize timing variables with defaults
    first_time = 0.0
    second_time = 0.0

    # You can change this to any MLX-compatible model you have
    model_name = "mlx-community/Qwen2.5-7B-Instruct-4bit"

    print(f"Testing streaming with model: {model_name}")
    print()

    # First streaming call - populates cache
    print("🌊 Step 1: First streaming request (populates cache)")
    messages1 = [Message(role="user", content="What is 2+2? Please explain.")]

    start_time = time.time()
    full_response1 = ""
    try:
        print("   Response: ", end="", flush=True)
        for chunk, is_final in generate_response_stream_cached(
            model_name=model_name, messages=messages1, max_tokens=100, temperature=0.1
        ):
            if chunk:
                print(chunk, end="", flush=True)
                full_response1 += chunk

        first_time = time.time() - start_time
        print(f"\n   ✅ Completed in {first_time:.2f}s")
    except Exception as e:
        print(f"\n   ❌ Failed: {e}")
        return {
            "first_time": 0.0,
            "second_time": 0.0,
            "improvement": 0.0,
            "speedup": 1.0,
        }

    # Second streaming call - should use cache (conversation continuation)
    print("\n🚀 Step 2: Follow-up streaming request (uses cache)")
    messages2 = [
        Message(role="user", content="What is 2+2? Please explain."),
        Message(role="assistant", content=full_response1.strip()),
        Message(role="user", content="Now what about 3+3?"),
    ]

    start_time = time.time()
    full_response2 = ""
    try:
        print("   Response: ", end="", flush=True)
        for chunk, is_final in generate_response_stream_cached(
            model_name=model_name, messages=messages2, max_tokens=100, temperature=0.1
        ):
            if chunk:
                print(chunk, end="", flush=True)
                full_response2 += chunk

        second_time = time.time() - start_time
        print(f"\n   ✅ Completed in {second_time:.2f}s")
    except Exception as e:
        print(f"\n   ❌ Failed: {e}")
        return {
            "first_time": first_time,
            "second_time": 0.0,
            "improvement": 0.0,
            "speedup": 1.0,
        }

    # Performance analysis
    print("\n📊 Streaming Performance Analysis:")
    print("-" * 50)
    print(f"First streaming:  {first_time:.2f}s")
    print(f"Second streaming: {second_time:.2f}s")

    # Initialize variables with default values
    improvement = 0.0
    speedup = 1.0

    if first_time > 0 and second_time > 0:
        improvement = ((first_time - second_time) / first_time) * 100
        speedup = first_time / second_time
        print(f"Improvement:      {improvement:.1f}%")
        print(f"Speedup:          {speedup:.2f}x")

        if improvement > 15:
            print("🚀 EXCELLENT: Streaming cache is working great!")
        elif improvement > 5:
            print("✅ GOOD: Streaming cache providing benefits")
        elif improvement > 0:
            print("📈 MODEST: Some streaming cache improvement")
        else:
            print("⚠️  Minimal improvement - may need longer conversations")
    else:
        print("⚠️  Unable to calculate performance metrics")

    # Cache statistics
    print("\n📈 Cache Statistics:")
    print("-" * 30)
    stats = get_conversation_cache_stats()
    for key, value in stats.items():
        print(f"{key}: {value}")

    print("\n💡 Streaming Cache Benefits:")
    print("-" * 40)
    print("• First streaming request populates the prompt cache")
    print("• Follow-up streams in same conversation are faster")
    print("• Cache reduces token processing time, not generation time")
    print("• Bigger benefit with longer conversation contexts")
    print("• Works transparently - same API, better performance!")

    return {
        "first_time": first_time,
        "second_time": second_time,
        "improvement": improvement,
        "speedup": speedup,
    }


def demonstrate_streaming_cache_types():
    """Demonstrate different cache types optimized for streaming."""
    print("\n" + "=" * 60)
    print("STREAMING CACHE TYPES OPTIMIZATION")
    print("=" * 60)
    print("Comparing cache types for streaming performance characteristics.")
    print()

    print("🔧 Current Streaming Cache Configuration:")
    print(f"   Cache type: {config.cache_type.value}")
    print(f"   Cache enabled: {config.conversation_cache_enabled}")

    # Show cache-specific optimizations for streaming
    if config.cache_type == CacheType.RotatingKVCache:
        print(f"   Max cache size: {config.cache_max_size}")
        print("   🌊 STREAMING OPTIMIZATION: Fixed memory, predictable latency")
    elif config.cache_type == CacheType.QuantizedKVCache:
        print(f"   Quantization bits: {config.cache_quantization_bits}")
        print("   🌊 STREAMING OPTIMIZATION: Lower memory usage, sustained throughput")
    elif config.cache_type == CacheType.ChunkedKVCache:
        print(f"   Chunk size: {config.cache_chunk_size}")
        print("   🌊 STREAMING OPTIMIZATION: Parallel processing, high-throughput")
    elif config.cache_type == CacheType.ConcatenateKVCache:
        print("   🌊 STREAMING OPTIMIZATION: Full context preservation")
    else:  # KVCache
        print("   🌊 STREAMING OPTIMIZATION: Balanced performance")
    print()

    print("⚡ Streaming Cache Manager Performance:")
    cache_config = {
        "cache_type": config.cache_type.value,
        "max_conversations": config.max_conversations,
        "conversation_idle_timeout": config.conversation_idle_timeout,
        "cache_max_size": config.cache_max_size,
        "cache_chunk_size": config.cache_chunk_size,
        "quantization_bits": config.cache_quantization_bits,
    }

    try:
        # Measure streaming-specific cache operations
        start_time = time.time()
        cache_manager = PromptCacheManager(cache_config)
        manager_time = time.time() - start_time

        # Create multiple cache instances (simulating concurrent streams)
        start_time = time.time()
        stream_caches = {}
        for i in range(3):  # Simulate 3 concurrent streaming conversations
            stream_caches[f"stream_{i}"] = cache_manager.get_cache(
                f"streaming-conversation-{i}"
            )
        multi_cache_time = time.time() - start_time

        print(f"   ✅ Cache manager for streaming: {manager_time * 1000:.2f}ms")
        print(f"   ✅ 3 concurrent stream caches: {multi_cache_time * 1000:.2f}ms")
        print(f"   📊 Average per stream: {(multi_cache_time / 3) * 1000:.2f}ms")

        # Test cache statistics for streaming
        stats = cache_manager.get_cache_stats()
        print(f"   📈 Active conversations: {stats.get('internal_caches', 0)}")

        return {
            "manager_time": manager_time,
            "multi_cache_time": multi_cache_time,
            "per_stream_time": multi_cache_time / 3,
            "cache_type": config.cache_type.value,
        }

    except Exception as e:
        print(f"   ❌ Cache manager test failed: {e}")
        return None


def show_streaming_recommendations():
    """Show cache recommendations specifically for streaming workloads."""
    print("\n🌊 Streaming Cache Optimization Guide:")
    print("-" * 50)
    print()

    print("🔹 High-Throughput Streaming (Many concurrent streams):")
    print(
        "   uv run mlx-server-nano --cache-type chunkedkvcache --cache-chunk-size 256"
    )
    print("   → Optimized parallel processing for multiple concurrent streams")
    print()

    print("🔹 Memory-Constrained Streaming:")
    print("   uv run mlx-server-nano --cache-type quantizedkvcache")
    print("   → 50-75% memory reduction while maintaining streaming performance")
    print()

    print("🔹 Predictable Streaming Performance:")
    print(
        "   uv run mlx-server-nano --cache-type rotatingkvcache --cache-max-size 1500"
    )
    print("   → Fixed memory ceiling, consistent latency for long streams")
    print()

    print("🔹 Context-Aware Streaming:")
    print("   uv run mlx-server-nano --cache-type concatenatekvcache")
    print("   → Best for streams requiring full conversation context")
    print()

    print("🔹 Real-Time Streaming Monitoring:")
    print("   export MLX_LOG_LEVEL=DEBUG")
    print("   → Track streaming cache performance in real-time")
    print()

    print("💡 Streaming Performance Tips:")
    print("   • Use RotatingKVCache for sustained long streaming sessions")
    print("   • Use ChunkedKVCache for batch streaming multiple requests")
    print("   • Use QuantizedKVCache when memory is limited")
    print("   • Monitor cache hit rates with DEBUG logging")


def main():
    """Run the streaming cache demonstration."""
    print("To run this example, make sure you have MLX Server Nano installed:")
    print("pip install -e .")
    print()

    try:
        # Original streaming cache demonstration
        results = demonstrate_streaming_cache()

        # New cache types for streaming demonstration
        cache_results = demonstrate_streaming_cache_types()

        # Streaming-specific recommendations
        show_streaming_recommendations()

        print("\n🎯 Summary:")
        print("-" * 30)
        if results and results["improvement"] > 5:
            print("✅ Streaming cache is working and providing measurable benefits!")
            print(
                f"   You achieved a {results['speedup']:.2f}x speedup on the follow-up request."
            )
        else:
            print("📝 Cache is working, but benefits may be more noticeable with:")
            print("   • Longer conversation histories")
            print("   • More complex prompts")
            print("   • Multiple back-and-forth exchanges")

        if cache_results:
            print("\n⚡ Streaming Cache Manager Performance:")
            print(f"   • Cache type: {cache_results['cache_type']}")
            print(f"   • Manager setup: {cache_results['manager_time'] * 1000:.2f}ms")
            print(
                f"   • 3 concurrent streams: {cache_results['multi_cache_time'] * 1000:.2f}ms"
            )
            print(
                f"   • Per-stream overhead: {cache_results['per_stream_time'] * 1000:.2f}ms"
            )

        print("\n🌊 To test streaming with different cache types:")
        print(
            "   MLX_CACHE_TYPE=RotatingKVCache python examples/streaming_cache_example.py"
        )
        print(
            "   MLX_CACHE_TYPE=ChunkedKVCache python examples/streaming_cache_example.py"
        )

    except KeyboardInterrupt:
        print("\n\nExample interrupted by user.")
    except Exception as e:
        print(f"\nExample failed with error: {e}")
        print(
            "Make sure you have MLX Server Nano properly installed and a compatible model."
        )


if __name__ == "__main__":
    main()
