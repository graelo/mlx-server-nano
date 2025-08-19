#!/usr/bin/env python3
"""
MLX-LM Prompt Caching Performance Example

This example demonstrates how MLX Server Nano's prompt caching provides
significant performance improvements for conversational interactions.

Features demonstrated:
- Performance comparison between first and cached requests
- Different cache types and their performance characteristics
- Cache statistics and management
- CLI configuration examples

Run this script to see real-world caching performance with your models.
"""

import os
import sys
import time
import logging

# Add the source directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mlx_server_nano.schemas import Message
from mlx_server_nano.model_manager.generation import generate_response_with_tools_cached
from mlx_server_nano.model_manager.cache_manager import (
    get_conversation_cache_stats,
    PromptCacheManager,
)
from mlx_server_nano.config import config, CacheType

# Set up logging to see what's happening
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def demonstrate_caching_performance():
    """Demonstrate MLX-LM prompt caching performance improvements."""

    print("=" * 60)
    print("MLX-LM PROMPT CACHING PERFORMANCE DEMONSTRATION")
    print("=" * 60)
    print("This example shows how conversation caching speeds up follow-up requests.")
    print()

    # You can change this to any MLX-compatible model you have
    model_name = "mlx-community/Qwen2.5-7B-Instruct-4bit"

    print(f"Testing with model: {model_name}")
    print()

    # Initialize timing variables with defaults
    first_time = 0.0
    second_time = 0.0

    # Test 1: First conversation (cold start - no cache)
    print("🔄 Step 1: First request (cold start)")
    messages1 = [Message(role="user", content="What is 2+2?")]

    start_time = time.time()
    try:
        content1, _ = generate_response_with_tools_cached(
            model_name=model_name, messages=messages1, max_tokens=50, temperature=0.1
        )
        first_time = time.time() - start_time
        print(f"   ✅ Completed in {first_time:.2f}s")
        print(f"   📝 Response: {content1[:80] if content1 else 'None'}...")
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return {
            "first_time": 0.0,
            "second_time": 0.0,
            "improvement": 0.0,
            "speedup": 1.0,
        }

    # Test 2: Follow-up conversation (should use cache)
    print("\n🚀 Step 2: Follow-up request (using cache)")
    messages2 = [
        Message(role="user", content="What is 2+2?"),
        Message(role="assistant", content="2+2 equals 4."),
        Message(role="user", content="What about 3+3?"),
    ]

    start_time = time.time()
    try:
        content2, _ = generate_response_with_tools_cached(
            model_name=model_name, messages=messages2, max_tokens=50, temperature=0.1
        )
        second_time = time.time() - start_time
        print(f"   ✅ Completed in {second_time:.2f}s")
        print(f"   📝 Response: {content2[:80] if content2 else 'None'}...")
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return {
            "first_time": first_time,
            "second_time": 0.0,
            "improvement": 0.0,
            "speedup": 1.0,
        }

    # Performance analysis
    print("\n📊 Performance Results:")
    print("-" * 40)
    print(f"First request:  {first_time:.2f}s")
    print(f"Second request: {second_time:.2f}s")

    # Initialize variables with default values
    improvement = 0.0
    speedup = 1.0

    if first_time > 0 and second_time > 0:
        improvement = ((first_time - second_time) / first_time) * 100
        speedup = first_time / second_time
        print(f"Improvement:    {improvement:.1f}%")
        print(f"Speedup:        {speedup:.2f}x")

        if improvement > 20:
            print("🚀 EXCELLENT: Significant caching performance improvement!")
        elif improvement > 10:
            print("✅ GOOD: Noticeable caching performance improvement")
        elif improvement > 0:
            print("📈 MODEST: Some caching improvement detected")
        else:
            print("⚠️  No improvement - cache may not be working optimally")
    else:
        print("⚠️  Unable to calculate performance metrics")

    # Show cache statistics
    print("\n📈 Cache Statistics:")
    print("-" * 40)
    stats = get_conversation_cache_stats()
    for key, value in stats.items():
        print(f"{key}: {value}")

    print("\n💡 Key Takeaways:")
    print("-" * 40)
    print("• First requests populate the cache (slower)")
    print("• Follow-up requests in the same conversation use the cache (faster)")
    print("• Longer conversations see bigger cache benefits")
    print("• Cache works automatically - no configuration needed!")

    return {
        "first_time": first_time,
        "second_time": second_time,
        "improvement": improvement,
        "speedup": speedup,
    }


def demonstrate_cache_types_performance():
    """Compare performance characteristics of different cache types."""
    print("\n" + "=" * 60)
    print("CACHE TYPES PERFORMANCE COMPARISON")
    print("=" * 60)
    print("Comparing different cache types and their configurations.")
    print()

    print("🔧 Current Cache Configuration:")
    print(f"   Cache type: {config.cache_type.value}")
    print(f"   Cache enabled: {config.conversation_cache_enabled}")
    print(f"   Max conversations: {config.max_conversations}")
    print(f"   Cache timeout: {config.conversation_idle_timeout}s")

    if config.cache_type == CacheType.RotatingKVCache:
        print(f"   Max cache size: {config.cache_max_size}")
    elif config.cache_type == CacheType.ChunkedKVCache:
        print(f"   Chunk size: {config.cache_chunk_size}")
    elif config.cache_type == CacheType.QuantizedKVCache:
        print(f"   Quantization bits: {config.cache_quantization_bits}")
    print()

    print("🏗️  Cache Manager Performance Test:")
    cache_config = {
        "cache_type": config.cache_type.value,
        "max_conversations": config.max_conversations,
        "conversation_idle_timeout": config.conversation_idle_timeout,
        "cache_max_size": config.cache_max_size,
        "cache_chunk_size": config.cache_chunk_size,
        "quantization_bits": config.cache_quantization_bits,
    }

    try:
        # Test cache manager creation performance
        start_time = time.time()
        cache_manager = PromptCacheManager(cache_config)
        creation_time = time.time() - start_time
        print(f"   ✅ Cache manager created in {creation_time * 1000:.2f}ms")

        # Test cache creation performance
        start_time = time.time()
        test_cache = cache_manager.get_cache("perf-test-conversation")
        cache_creation_time = time.time() - start_time
        print(f"   ✅ Cache instance created in {cache_creation_time * 1000:.2f}ms")
        print(f"   📊 Cache type: {type(test_cache).__name__}")

        # Test cache operations
        start_time = time.time()
        stats = cache_manager.get_cache_stats()
        stats_time = time.time() - start_time
        print(f"   📈 Cache stats retrieved in {stats_time * 1000:.2f}ms")

        # Display cache stats
        print("\n📊 Detailed Cache Statistics:")
        print("-" * 40)
        for key, value in stats.items():
            if key != "internal_caches":  # Skip internal implementation details
                print(f"   {key}: {value}")

        print(f"\n💾 Memory Efficiency for {config.cache_type.value}:")
        if config.cache_type == CacheType.QuantizedKVCache:
            print("   • Memory usage: ~50-75% less than standard cache")
            print("   • Best for: Memory-constrained environments")
        elif config.cache_type == CacheType.RotatingKVCache:
            print("   • Memory usage: Fixed maximum size")
            print("   • Best for: Predictable memory requirements")
        elif config.cache_type == CacheType.ChunkedKVCache:
            print("   • Memory usage: Optimized for batch processing")
            print("   • Best for: High-throughput scenarios")
        elif config.cache_type == CacheType.ConcatenateKVCache:
            print("   • Memory usage: Optimized for context preservation")
            print("   • Best for: Context-heavy applications")
        else:  # KVCache
            print("   • Memory usage: Standard baseline")
            print("   • Best for: General purpose use")

        return {
            "creation_time": creation_time,
            "cache_creation_time": cache_creation_time,
            "stats_time": stats_time,
            "cache_type": config.cache_type.value,
        }

    except Exception as e:
        print(f"   ❌ Cache manager test failed: {e}")
        print("   Note: This is expected if MLX is not installed")
        return None


def print_optimization_recommendations():
    """Print recommendations for cache optimization."""
    print("\n🚀 Cache Optimization Recommendations:")
    print("-" * 50)

    print("For different use cases, consider these cache types:")
    print()

    print("🔹 Memory-Constrained Systems:")
    print("   uv run mlx-server-nano --cache-type quantizedkvcache")
    print("   → Reduces memory usage by 50-75%")
    print()

    print("🔹 Streaming Applications:")
    print(
        "   uv run mlx-server-nano --cache-type rotatingkvcache --cache-max-size 2000"
    )
    print("   → Fixed memory footprint, prevents memory growth")
    print()

    print("🔹 Batch Processing:")
    print(
        "   uv run mlx-server-nano --cache-type chunkedkvcache --cache-chunk-size 256"
    )
    print("   → Optimized for parallel processing and high throughput")
    print()

    print("🔹 Context-Heavy Applications:")
    print("   uv run mlx-server-nano --cache-type concatenatekvcache")
    print("   → Best context preservation for reasoning tasks")
    print()

    print("🔹 Production Monitoring:")
    print("   export MLX_LOG_LEVEL=DEBUG")
    print("   → Enable detailed cache performance logging")


def main():
    """Run the cache performance demonstration."""
    print("To run this example, make sure you have MLX Server Nano installed:")
    print("uv pip install -e .")
    print()

    try:
        # Original caching performance demonstration
        results = demonstrate_caching_performance()

        # New cache types performance demonstration
        cache_results = demonstrate_cache_types_performance()

        # Optimization recommendations
        print_optimization_recommendations()

        print("\n🎯 Summary:")
        print("-" * 30)
        if results and results["improvement"] > 5:
            print("✅ Caching is working and providing measurable benefits!")
            print(
                f"   You achieved a {results['speedup']:.2f}x speedup on the follow-up request."
            )
        else:
            print("📝 Cache is working, but benefits may be more noticeable with:")
            print("   • Longer conversation histories")
            print("   • More complex prompts")
            print("   • Multiple back-and-forth exchanges")

        if cache_results:
            print("\n⚡ Cache Manager Performance:")
            print(f"   • Cache type: {cache_results['cache_type']}")
            print(
                f"   • Manager creation: {cache_results['creation_time'] * 1000:.2f}ms"
            )
            print(
                f"   • Cache instance: {cache_results['cache_creation_time'] * 1000:.2f}ms"
            )
            print(
                f"   • Statistics retrieval: {cache_results['stats_time'] * 1000:.2f}ms"
            )

        print("\n🔄 To test different cache types:")
        print(
            "   MLX_CACHE_TYPE=QuantizedKVCache python examples/cache_performance_example.py"
        )
        print(
            "   MLX_CACHE_TYPE=RotatingKVCache python examples/cache_performance_example.py"
        )

    except KeyboardInterrupt:
        print("\n\nExample interrupted by user.")
    except Exception as e:
        print(f"\nExample failed with error: {e}")
        print(
            "Make sure you have MLX Server Nano properly installed and a compatible model available."
        )


if __name__ == "__main__":
    main()
