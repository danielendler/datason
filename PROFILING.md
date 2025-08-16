# 📊 DataSON Profiling System

## Overview

DataSON includes a built-in profiling system that provides detailed performance insights with minimal overhead. This system is designed for both development debugging and CI/CD performance monitoring.

## Quick Start

### Enable Profiling

```python
import os
import datason

# Enable profiling
os.environ['DATASON_PROFILE'] = '1'

# Set up profile sink to capture events
datason.profile_sink = []

# Use DataSON normally
data = {"users": [{"id": i, "name": f"user_{i}"} for i in range(100)]}
json_str = datason.save_string(data)
loaded = datason.load_basic(json_str)

# View captured profiling events
for event in datason.profile_sink:
    duration_ms = event['duration'] / 1_000_000  # Convert nanoseconds to milliseconds
    print(f"{event['stage']}: {duration_ms:.3f}ms")
```

### Expected Output

```
eligibility_check: 0.234ms
limits_prepare: 0.012ms  
serialize_inner_python: 2.456ms
save_string: 2.701ms
```

## Profiling APIs

### Environment Variables

- `DATASON_PROFILE=1` - Enable profiling (disabled by default)
- `DATASON_RUST=auto|1|0` - Control Rust acceleration (when available)

### Python APIs

```python
import datason

# List-based profiling (for external tools like datason-benchmarks)
datason.profile_sink = []  # Events automatically appended here

# Function-based profiling (for custom handlers)
def my_profile_handler(timing_dict):
    print(f"Operation completed: {timing_dict}")

datason.set_profile_sink(my_profile_handler)

# Core benchmark APIs
result = datason.save_string(data)      # Primary serialization API
loaded = datason.load_basic(result)     # Primary deserialization API

# Check Rust availability
if datason.RUST_AVAILABLE:
    print("Rust acceleration is available")
```

## Profiling Stages

The profiling system captures timing for these key stages:

### Serialization Stages (`save_string`)
- `eligibility_check` - Determines if data needs special handling
- `limits_prepare` - Sets up security limits and validation
- `serialize_inner_python` - Core Python serialization logic
- `serialize_rust_fast` - Rust serialization (when available)
- `save_string` - Complete operation timing

### Deserialization Stages (`load_basic`)
- `eligibility_check` - Input validation and type detection
- `smart_scalars` - Intelligent scalar type handling
- `postprocess` - Final data transformation
- `load_basic_json` - Complete JSON parsing operation

## CI Integration

### Automatic Performance Testing

DataSON includes GitHub Actions workflows that automatically:

1. **Test profiling system** on every PR
2. **Capture performance metrics** for different data scenarios  
3. **Generate detailed reports** with stage-by-stage timing
4. **Post analysis to PR comments** for team visibility
5. **Upload artifacts** with comprehensive profiling data

### Workflow Example

```yaml
name: 📊 PR Performance Analysis
on:
  pull_request:
    branches: [ main ]

jobs:
  benchmark:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v5
    - name: Run performance test
      run: |
        DATASON_PROFILE=1 python test_performance.py
    # Automatically posts results to PR
```

## Performance Overhead

The profiling system is designed for minimal overhead:

- **When disabled** (`DATASON_PROFILE` not set): Near-zero overhead
- **When enabled**: Typically <5% overhead for most workloads
- **Nanosecond precision**: High-resolution timing using `time.perf_counter()`

## Integration with External Tools

### datason-benchmarks Integration

The profiling system is fully compatible with the [datason-benchmarks](https://github.com/danielendler/datason-benchmarks) repository:

```python
# The benchmarks automatically use these APIs:
datason.profile_sink = []           # Capture profiling events
result = datason.save_string(data)  # Primary benchmark API
loaded = datason.load_basic(result) # Primary benchmark API

# Events are automatically in the format:
# {"stage": "serialize_inner_python", "duration": 2456789}  # nanoseconds
```

### Custom Profiling Tools

```python
class CustomProfiler:
    def __init__(self):
        self.events = []

    def handle_event(self, timing_dict):
        self.events.append(timing_dict)

    def analyze(self):
        for stage, duration in timing_dict.items():
            print(f"{stage}: {duration/1_000_000:.3f}ms")

profiler = CustomProfiler()
datason.set_profile_sink(profiler.handle_event)

# Use DataSON - events will be sent to your custom profiler
datason.save_string(data)
profiler.analyze()
```

## Rust Core Integration

When Rust core is available (`datason.RUST_AVAILABLE = True`):

### Environment Control
```bash
DATASON_RUST=1    # Force Rust acceleration  
DATASON_RUST=0    # Force Python implementation
DATASON_RUST=auto # Auto-detect (default)
```

### Profiling Stages
```python
# With Rust enabled, you'll see stages like:
# serialize_rust_fast: 0.123ms    (Rust path)
# serialize_inner_python: 2.456ms (Python fallback)

# Profiling shows which path was taken for each operation
```

### Performance Comparison
The CI system automatically compares Rust vs Python performance:

| Implementation | Time | Speedup |
|----------------|------|---------|
| Python | 2.456ms | 1x |
| Rust | 0.123ms | 20x |

## Best Practices

### Development
- Enable profiling during development to identify bottlenecks
- Use `datason.profile_sink = []` for simple event collection
- Check `datason.RUST_AVAILABLE` to understand which implementation is active

### Production
- Keep profiling disabled in production (`DATASON_PROFILE` unset)
- Use profiling for performance debugging of specific issues
- Monitor CI profiling results to track performance over time

### CI/CD
- Include performance testing in your PR workflow
- Set up regression detection to catch performance degradations
- Use profiling artifacts to debug performance issues

## Troubleshooting

### Profiling Not Working
```python
import os
import datason

# Verify setup
print(f"DATASON_PROFILE: {os.environ.get('DATASON_PROFILE')}")
print(f"Profile sink available: {hasattr(datason, 'profile_sink')}")
print(f"Profile sink type: {type(datason.profile_sink)}")

# Enable and test
os.environ['DATASON_PROFILE'] = '1'
datason.profile_sink = []

# Test operation
datason.save_string({"test": "data"})
print(f"Events captured: {len(datason.profile_sink)}")
```

### Common Issues
- **No events captured**: Ensure `DATASON_PROFILE=1` is set before importing datason
- **Module reload needed**: In some environments, you may need to restart Python after setting environment variables
- **Missing profile_sink**: Ensure you're using the latest version with profiling support

## Example: Complete Profiling Session

```python
#!/usr/bin/env python3
import os
import time
import datason

# Enable profiling
os.environ['DATASON_PROFILE'] = '1'
datason.profile_sink = []

# Test data
test_scenarios = [
    {"name": "Simple", "data": {"id": 123, "name": "test"}},
    {"name": "Complex", "data": {"users": [{"id": i} for i in range(1000)]}},
]

for scenario in test_scenarios:
    print(f"\n=== {scenario['name']} ===")
    datason.profile_sink.clear()

    # Measure complete operation
    start = time.perf_counter()
    json_str = datason.save_string(scenario['data'])
    loaded = datason.load_basic(json_str)
    total_time = time.perf_counter() - start

    print(f"Total time: {total_time*1000:.2f}ms")
    print(f"JSON size: {len(json_str):,} chars")
    print("Profiling breakdown:")

    for event in datason.profile_sink:
        duration_ms = event['duration'] / 1_000_000
        print(f"  {event['stage']}: {duration_ms:.3f}ms")
```

This profiling system provides comprehensive performance insights while maintaining the simplicity and reliability that DataSON is known for.
