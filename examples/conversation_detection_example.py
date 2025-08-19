#!/usr/bin/env python3
"""
Conversation Detection Example

This example demonstrates how MLX Server Nano automatically detects
conversation continuations and manages conversation state for caching.

This is useful for understanding how the automatic conversation detection
system works under the hood, and showcases the different cache types
available with the new cache manager architecture.

Features demonstrated:
- Automatic conversation detection
- Explicit conversation ID management
- Cache statistics and cleanup
- Different cache types (KVCache, QuantizedKVCache, RotatingKVCache, etc.)
- CLI configuration options for cache management
"""

import sys
import os

# Add the source directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mlx_server_nano.schemas import Message
from mlx_server_nano.model_manager.cache_manager import (
    get_or_create_conversation_state,
    generate_conversation_hash,
    get_conversation_cache_stats,
    cleanup_expired_conversations,
    PromptCacheManager,
)
from mlx_server_nano.config import config, CacheType


def demonstrate_conversation_detection():
    """Demonstrate automatic conversation detection."""
    print("=== Automatic Conversation Detection Demo ===")
    print()

    # Create some test messages - initial conversation
    messages1 = [
        Message(role="user", content="Hello, how are you?"),
        Message(role="assistant", content="I'm doing well, thank you!"),
        Message(role="user", content="What's the weather like?"),
    ]

    # Extended conversation - same messages plus new ones
    messages2 = [
        Message(role="user", content="Hello, how are you?"),
        Message(role="assistant", content="I'm doing well, thank you!"),
        Message(role="user", content="What's the weather like?"),
        Message(
            role="assistant", content="I don't have access to current weather data."
        ),
        Message(role="user", content="Can you help me with programming?"),
    ]

    print("🔍 Testing conversation hash generation:")
    hash1 = generate_conversation_hash(messages1)
    hash2 = generate_conversation_hash(messages2)
    print(f"   Hash for 3 messages: {hash1[:16]}...")
    print(f"   Hash for 5 messages: {hash2[:16]}...")
    print()

    print("🆕 Creating first conversation state:")
    conv_state1 = get_or_create_conversation_state(None, "test-model", messages1)
    print(f"   Conversation ID: {conv_state1.conversation_id}")
    print()

    print("🔗 Testing conversation continuation detection:")
    conv_state2 = get_or_create_conversation_state(None, "test-model", messages2)
    print(f"   Conversation ID: {conv_state2.conversation_id}")
    print()

    # Check if they're the same conversation
    if conv_state1.conversation_id == conv_state2.conversation_id:
        print("✅ SUCCESS: Conversation continuation detected!")
        print("   The system correctly identified that messages2 extends messages1")
    else:
        print("❌ FAILURE: Conversation continuation NOT detected")
        print("   The system treated these as separate conversations")
    print()


def demonstrate_explicit_conversation_ids():
    """Demonstrate explicit conversation ID management."""
    print("=== Explicit Conversation ID Demo ===")
    print()

    messages = [
        Message(role="user", content="Hello from explicit session"),
        Message(role="assistant", content="Hello! How can I help you?"),
    ]

    print("🏷️  Creating conversation with explicit ID:")
    conv_state = get_or_create_conversation_state("user-123", "test-model", messages)
    print(f"   Conversation ID: {conv_state.conversation_id}")
    print()

    print("🔄 Retrieving same conversation:")
    conv_state2 = get_or_create_conversation_state("user-123", "test-model", messages)
    print(f"   Conversation ID: {conv_state2.conversation_id}")
    print()

    if conv_state.conversation_id == conv_state2.conversation_id:
        print("✅ SUCCESS: Explicit conversation ID works correctly!")
        print("   Same conversation retrieved using the same ID")
    else:
        print("❌ FAILURE: Explicit conversation ID failed")
    print()


def show_cache_statistics():
    """Show current cache statistics and cleanup."""
    print("=== Cache Management Demo ===")
    print()

    print("📊 Current cache statistics:")
    stats_before = get_conversation_cache_stats()
    for key, value in stats_before.items():
        print(f"   {key}: {value}")
    print()

    print("🧹 Running cache cleanup:")
    cleanup_expired_conversations()

    stats_after = get_conversation_cache_stats()
    print("📊 Statistics after cleanup:")
    for key, value in stats_after.items():
        print(f"   {key}: {value}")
    print()


def demonstrate_cache_types():
    """Demonstrate different cache types available."""
    print("=== Cache Types Demo ===")
    print()

    print("🔧 Available cache types:")
    for cache_type in CacheType:
        print(f"   • {cache_type.value}")
    print()

    print("⚙️  Current server configuration:")
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

    print("🏗️  Creating cache manager with current config:")
    cache_config = {
        "cache_type": config.cache_type.value,
        "max_conversations": config.max_conversations,
        "conversation_idle_timeout": config.conversation_idle_timeout,
        "cache_max_size": config.cache_max_size,
        "cache_chunk_size": config.cache_chunk_size,
        "quantization_bits": config.cache_quantization_bits,
    }

    try:
        cache_manager = PromptCacheManager(cache_config)
        print(f"   ✅ Successfully created {config.cache_type.value} cache manager")

        # Demonstrate cache creation
        test_cache = cache_manager.get_cache("demo-conversation")
        print(f"   ✅ Created cache for demo conversation: {type(test_cache).__name__}")

        # Get cache statistics
        stats = cache_manager.get_cache_stats()
        print(f"   📊 Cache stats: {stats}")

    except Exception as e:
        print(f"   ❌ Failed to create cache manager: {e}")
        print("   Note: This is expected if MLX is not installed")
    print()


def main():
    """Run all conversation detection examples."""
    print("MLX Server Nano - Conversation Detection Examples")
    print("=" * 60)
    print("This demonstrates how the system automatically detects")
    print("conversation continuations for efficient caching.")
    print()

    try:
        demonstrate_conversation_detection()
        demonstrate_explicit_conversation_ids()
        show_cache_statistics()
        demonstrate_cache_types()

        print("💡 Key Points:")
        print(
            "• The system automatically detects when new messages extend existing conversations"
        )
        print("• You can use explicit conversation IDs for precise control")
        print("• Cache cleanup happens automatically based on idle timeouts")
        print("• Multiple cache types are available for different use cases:")
        print("  - KVCache: Standard cache for general use")
        print("  - QuantizedKVCache: Memory-efficient cache")
        print("  - RotatingKVCache: Fixed-size cache with predictable memory usage")
        print("  - ChunkedKVCache: Optimized for batch processing")
        print("  - ConcatenateKVCache: Best for context-heavy applications")
        print(
            "• All of this works transparently with the API - no configuration needed!"
        )
        print()
        print("🚀 To change cache types, use CLI options:")
        print("   uv run mlx-server-nano --cache-type quantizedkvcache")
        print(
            "   uv run mlx-server-nano --cache-type rotatingkvcache --cache-max-size 2000"
        )

    except Exception as e:
        print(f"Example failed with error: {e}")
        print("Make sure MLX Server Nano is properly installed.")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
