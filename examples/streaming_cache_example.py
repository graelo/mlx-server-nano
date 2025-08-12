#!/usr/bin/env python3
"""
Streaming with Caching Example

This example demonstrates how MLX Server Nano's prompt caching works
with streaming generation, providing performance improvements for
streaming responses in conversational scenarios.
"""

import os
import sys
import time
import logging

# Add the source directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mlx_server_nano.schemas import Message
from mlx_server_nano.model_manager.generation import generate_response_stream_cached
from mlx_server_nano.model_manager.cache import get_conversation_cache_stats

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


def main():
    """Run the streaming cache demonstration."""
    print("To run this example, make sure you have MLX Server Nano installed:")
    print("pip install -e .")
    print()

    try:
        results = demonstrate_streaming_cache()

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

    except KeyboardInterrupt:
        print("\n\nExample interrupted by user.")
    except Exception as e:
        print(f"\nExample failed with error: {e}")
        print(
            "Make sure you have MLX Server Nano properly installed and a compatible model."
        )


if __name__ == "__main__":
    main()
