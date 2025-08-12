#!/usr/bin/env python3
"""
MLX-LM Prompt Caching Performance Example

This example demonstrates how MLX Server Nano's prompt caching provides
significant performance improvements for conversational interactions.

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
from mlx_server_nano.model_manager.cache import get_conversation_cache_stats

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


def main():
    """Run the cache performance demonstration."""
    print("To run this example, make sure you have MLX Server Nano installed:")
    print("uv pip install -e .")
    print()

    try:
        results = demonstrate_caching_performance()

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

    except KeyboardInterrupt:
        print("\n\nExample interrupted by user.")
    except Exception as e:
        print(f"\nExample failed with error: {e}")
        print(
            "Make sure you have MLX Server Nano properly installed and a compatible model available."
        )


if __name__ == "__main__":
    main()
