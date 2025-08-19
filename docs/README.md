# MLX Server Nano Documentation

Welcome to the MLX Server Nano documentation. This directory contains comprehensive guides for understanding, configuring, and testing the server.

## Documentation Index

### 📚 Core Documentation

- **[Cache Management Guide](CACHE_MANAGEMENT.md)** - Comprehensive guide to the cache management system
  - Multiple cache types (KVCache, QuantizedKVCache, RotatingKVCache, ChunkedKVCache, ConcatenateKVCache)
  - CLI and environment variable configuration options
  - Performance characteristics and use case recommendations
  - Configuration, monitoring, and troubleshooting
  - Best practices and advanced usage patterns

- **[Testing Guide](TESTING.md)** - Complete testing documentation
  - Test structure and categories
  - Running tests (unit, integration, memory, API)
  - Coverage reporting and debugging
  - Best practices for test development

## Quick Links

### Getting Started
- [Main README](../README.md) - Project overview and basic setup
- [Configuration](../README.md#configuration) - Environment variables, CLI options, and cache types
- [Examples](../examples/) - Working code examples and cache performance demonstrations

### Development
- [Testing Guide](TESTING.md) - How to run and write tests
- [Cache Management](CACHE_MANAGEMENT.md) - Understanding the cache system
- [Contributing Guidelines](../README.md#contributing) - How to contribute

### Advanced Topics
- [Cache Types Comparison](CACHE_MANAGEMENT.md#performance-characteristics) - Detailed performance comparison of all 5 cache types
- [CLI Cache Configuration](../README.md#command-line-options) - Complete CLI options for cache management
- [Memory Management](CACHE_MANAGEMENT.md#monitoring-and-debugging) - Monitoring and optimization
- [Troubleshooting](CACHE_MANAGEMENT.md#troubleshooting) - Common issues and solutions

## Documentation Structure

```
docs/
├── README.md                 # This index file
├── CACHE_MANAGEMENT.md       # Cache system documentation
└── TESTING.md               # Testing guide and best practices
```

## Contributing to Documentation

When contributing to the documentation:

1. **Keep it practical** - Include working code examples
2. **Update cross-references** - Maintain links between documents
3. **Test examples** - Ensure code samples work
4. **Be comprehensive** - Cover both basic and advanced use cases
5. **Update this index** - Add new documentation files here

## Getting Help

- **Issues**: Report problems or request features on [GitHub Issues](https://github.com/graelo/mlx-server-nano/issues)
- **Discussions**: Join community discussions on [GitHub Discussions](https://github.com/graelo/mlx-server-nano/discussions)
- **Documentation**: This documentation is continuously updated - check back for the latest information

---

*Documentation last updated: August 2025*
