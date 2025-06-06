# 📋 API Reference

Complete API documentation for datason with examples and auto-generated documentation from source code.

## Modules

- [Core Functions](core.md)
- [Configuration System](config.md)
- [ML Library Integration](ml.md)
- [Redaction & Privacy](redaction.md)
- [Utilities](utils.md)

## Quick Reference

### Common Usage Patterns

```python
import datason as ds
import pandas as pd
import numpy as np
from datetime import datetime

# Basic serialization
data = {"values": [1, 2, 3], "timestamp": datetime.now()}
serialized = ds.serialize(data)
restored = ds.deserialize(serialized)

# With configuration
config = ds.get_ml_config()
ml_data = {"model": model, "features": pd.DataFrame(data)}
result = ds.serialize(ml_data, config=config)

# Chunked processing for large data
large_data = {"arrays": [np.random.random((1000, 1000)) for _ in range(100)]}
chunked = ds.serialize_chunked(large_data, chunk_size=10*1024*1024)

# Template enforcement
template = ds.infer_template_from_data(sample_data)
validated = ds.deserialize_with_template(new_data, template)

# Privacy protection
engine = ds.create_financial_redaction_engine()
safe_data = engine.process_object(sensitive_data)
```

### Error Handling

```python
try:
    result = ds.serialize(complex_data)
except ds.SecurityError as e:
    print(f"Security violation: {e}")
except MemoryError as e:
    # Fall back to chunked processing
    result = ds.serialize_chunked(complex_data)
except Exception as e:
    # Generic error handling
    result = ds.safe_serialize(complex_data)
```

### Performance Tips

```python
# For repeated operations, reuse configuration
config = ds.get_ml_config()
for batch in data_batches:
    result = ds.serialize(batch, config=config)

# Estimate memory before processing
memory_estimate = ds.estimate_memory_usage(large_data)
if memory_estimate > threshold:
    use_chunked_processing()

# Monitor performance
import time
start = time.time()
result = ds.serialize(data)
duration = time.time() - start
```
