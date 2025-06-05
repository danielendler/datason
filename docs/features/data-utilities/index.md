# Data Utilities with Security Patterns (v0.5.5)

The enhanced data utilities module provides comprehensive data analysis, transformation, and comparison tools with the same security patterns developed for the core serialization engine. This ensures consistent protection against resource exhaustion attacks while offering powerful data processing capabilities.

## 🛡️ Security-First Design

### Consistent Security Model
The utils module leverages the same security constants and patterns as `core.py`:

```python
from datason.utils import UtilityConfig, UtilitySecurityError

# Same security constants from core.py
config = UtilityConfig()
print(f"Max Depth: {config.max_depth}")           # 50 (from core.py)
print(f"Max Object Size: {config.max_object_size}") # 10,000,000 (from core.py)
print(f"Max String Length: {config.max_string_length}") # 1,000,000 (from core.py)
```

### Environment-Specific Security Configurations

```python
from datason.utils import UtilityConfig

# Development environment (permissive for testing)
dev_config = UtilityConfig(
    max_depth=100,
    max_object_size=1_000_000,
    max_string_length=100_000,
    enable_circular_reference_detection=False  # For performance
)

# Production environment (balanced security and performance)
prod_config = UtilityConfig(
    max_depth=25,
    max_object_size=100_000,
    max_string_length=1_000_000,
    enable_circular_reference_detection=True
)

# Public API environment (strict security)
api_config = UtilityConfig(
    max_depth=10,
    max_object_size=10_000,
    max_string_length=100_000,
    enable_circular_reference_detection=True
)
```

## 🔍 Data Analysis Utilities

### Deep Object Comparison

Advanced comparison with security limits and detailed reporting:

```python
from datason.utils import deep_compare, UtilityConfig

# Basic comparison
obj1 = {"users": [{"id": 1, "score": 85.5}], "meta": {"version": "1.0"}}
obj2 = {"users": [{"id": 1, "score": 85.6}], "meta": {"version": "1.0"}}

result = deep_compare(obj1, obj2, tolerance=1e-6)
print(f"Are equal: {result['are_equal']}")
print(f"Differences: {result['differences']}")
print(f"Summary: {result['summary']}")

# With security configuration
config = UtilityConfig(max_depth=5, max_object_size=1000)
try:
    result = deep_compare(very_deep_nested_obj, obj2, config=config)
except UtilitySecurityError as e:
    print(f"Security violation: {e}")
```

### Data Anomaly Detection

Identify potential issues in data structures with configurable rules:

```python
from datason.utils import find_data_anomalies

data = {
    "normal_field": "short text",
    "large_text": "x" * 50000,  # Large string
    "big_list": list(range(5000)),  # Large collection
    "user_input": "<script>alert('xss')</script>",  # Suspicious pattern
    "nested": {"deep": {"structure": "value"}}
}

# Custom anomaly detection rules
rules = {
    "max_string_length": 10000,  # Flag strings over 10KB
    "max_list_length": 1000,     # Flag lists over 1000 items
    "suspicious_patterns": [r"<script", r"javascript:", r"eval\("],
    "detect_suspicious_patterns": True,
}

anomalies = find_data_anomalies(data, rules=rules)
print(f"Large strings: {len(anomalies['large_strings'])}")
print(f"Large collections: {len(anomalies['large_collections'])}")
print(f"Suspicious patterns: {len(anomalies['suspicious_patterns'])}")
print(f"Security violations: {len(anomalies['security_violations'])}")
```

## 🔄 Data Transformation Utilities

### Smart Type Enhancement

Intelligent type inference and conversion with security:

```python
from datason.utils import enhance_data_types

# Raw data with string representations
messy_data = {
    "user": {
        "id": "123",           # String number
        "score": "85.5",       # String float  
        "active": "true",      # String boolean
        "joined": "2023-01-15", # String date
        "name": "  John Doe  " # Whitespace
    },
    "metadata": {
        "count": "42",
        "ratio": "0.95"
    }
}

# Enhancement with custom rules
rules = {
    "parse_numbers": True,
    "parse_dates": True,
    "clean_whitespace": True,
    "normalize_booleans": True,
}

enhanced_data, report = enhance_data_types(messy_data, rules)

print("Enhanced data:")
print(f"  User ID: {enhanced_data['user']['id']} (type: {type(enhanced_data['user']['id'])})")
print(f"  Score: {enhanced_data['user']['score']} (type: {type(enhanced_data['user']['score'])})")
print(f"  Active: {enhanced_data['user']['active']} (type: {type(enhanced_data['user']['active'])})")
print(f"  Joined: {enhanced_data['user']['joined']} (type: {type(enhanced_data['user']['joined'])})")

print(f"\nEnhancement report:")
print(f"  Type conversions: {len(report['type_conversions'])}")
print(f"  Cleaned values: {len(report['cleaned_values'])}")
```

### Data Structure Normalization

Transform nested structures into flat or normalized formats:

```python
from datason.utils import normalize_data_structure

# Nested data structure
nested_data = {
    "user": {
        "profile": {
            "name": "Alice",
            "age": 30
        },
        "settings": {
            "theme": "dark",
            "notifications": True
        }
    },
    "posts": [
        {"id": 1, "title": "First Post"},
        {"id": 2, "title": "Second Post"}
    ]
}

# Flatten structure
flattened = normalize_data_structure(nested_data, target_structure="flat")
print("Flattened structure:")
for key, value in flattened.items():
    print(f"  {key}: {value}")

# Convert to records format
records_data = {
    "names": ["Alice", "Bob", "Charlie"],
    "ages": [25, 30, 35],
    "cities": ["NYC", "LA", "Chicago"]
}

records = normalize_data_structure(records_data, target_structure="records")
print("\nRecords format:")
for record in records:
    print(f"  {record}")
```

## 📅 Datetime Utilities

### Standardize Datetime Formats

Consistent datetime format conversion throughout data structures:

```python
from datason.utils import standardize_datetime_formats
from datetime import datetime

data_with_dates = {
    "events": [
        {"timestamp": datetime(2023, 12, 25, 10, 30, 0), "type": "login"},
        {"timestamp": datetime(2023, 12, 25, 11, 45, 0), "type": "logout"}
    ],
    "created": datetime(2023, 12, 20, 9, 0, 0),
    "metadata": {
        "last_modified": datetime(2023, 12, 25, 12, 0, 0)
    }
}

# Convert to ISO format
iso_data, log = standardize_datetime_formats(data_with_dates, target_format="iso")
print("ISO format conversion:")
print(f"  Created: {iso_data['created']}")
print(f"  Conversion log: {log}")

# Convert to Unix timestamp
unix_data, log = standardize_datetime_formats(data_with_dates, target_format="unix")
print("\nUnix timestamp conversion:")
print(f"  Created: {unix_data['created']}")
```

### Extract Temporal Features

Analyze temporal patterns in data:

```python
from datason.utils import extract_temporal_features
from datetime import datetime

temporal_data = {
    "user_events": [
        {"login_time": datetime(2023, 12, 25, 9, 0, 0)},
        {"logout_time": datetime(2023, 12, 25, 17, 30, 0)}
    ],
    "system": {
        "created": datetime(2023, 1, 1, 0, 0, 0),
        "updated": datetime(2023, 12, 25, 12, 0, 0)
    }
}

features = extract_temporal_features(temporal_data)
print("Temporal analysis:")
print(f"  Datetime fields found: {features['datetime_fields']}")
print(f"  Date ranges: {features['date_ranges']}")
print(f"  Timezones: {features['timezones']}")
```

## 🐼 Pandas/NumPy Integration

### Enhanced DataFrame Processing

Optimized DataFrame processing with security and performance reporting:

```python
from datason.utils import enhance_pandas_dataframe, UtilityConfig
import pandas as pd

# Create sample DataFrame with mixed types
df = pd.DataFrame({
    'user_id': ['1', '2', '3', '4'],           # String numbers
    'score': ['85.5', '92.0', 'invalid', '88.5'], # Mixed string/invalid
    'active': ['true', 'false', 'true', '1'],   # String booleans
    'category': ['A', 'B', 'A', 'C']            # Categories
})

print("Original DataFrame:")
print(df.dtypes)
print(f"Memory usage: {df.memory_usage(deep=True).sum()} bytes")

# Enhanced processing with security limits
config = UtilityConfig(max_object_size=10000)  # Reasonable limit
enhanced_df, report = enhance_pandas_dataframe(df, config=config)

print("\nEnhanced DataFrame:")
print(enhanced_df.dtypes)
print(f"Memory usage: {enhanced_df.memory_usage(deep=True).sum()} bytes")

print(f"\nEnhancement report:")
print(f"  Columns processed: {len(report['columns_processed'])}")
print(f"  Type conversions: {len(report['type_conversions'])}")
print(f"  Memory saved: {report['memory_saved']} bytes")

for conversion in report['type_conversions']:
    print(f"    {conversion['column']}: {conversion['from']} -> {conversion['to']}")
```

### Enhanced NumPy Array Processing

Array optimization with dtype downcasting and cleaning:

```python
from datason.utils import enhance_numpy_array
import numpy as np

# Create sample array with suboptimal dtype
arr = np.array([1.0, 2.0, 3.0, np.inf, 5.0], dtype=np.float64)

print("Original array:")
print(f"  Shape: {arr.shape}, Dtype: {arr.dtype}")
print(f"  Memory usage: {arr.nbytes} bytes")
print(f"  Contains inf: {np.any(np.isinf(arr))}")

# Enhancement with custom rules
rules = {
    "optimize_dtype": True,
    "remove_inf": True,
    "remove_nan": True,
    "normalize_range": False,
}

enhanced_arr, report = enhance_numpy_array(arr, rules)

print("\nEnhanced array:")
print(f"  Shape: {enhanced_arr.shape}, Dtype: {enhanced_arr.dtype}")
print(f"  Memory usage: {enhanced_arr.nbytes} bytes")
print(f"  Contains inf: {np.any(np.isinf(enhanced_arr))}")

print(f"\nOptimizations applied: {report['optimizations_applied']}")
print(f"Memory saved: {report['memory_saved']} bytes")
```

## 🔧 Utility Discovery

### Programmatic Feature Discovery

```python
from datason.utils import get_available_utilities

utilities = get_available_utilities()
print("Available utility categories:")
for category, functions in utilities.items():
    print(f"  {category}: {', '.join(functions)}")
```

## ⚠️ Security Features in Action

### Protection Against Resource Exhaustion

```python
from datason.utils import deep_compare, UtilityConfig, UtilitySecurityError

# Create a deeply nested structure that could cause stack overflow
def create_deep_structure(depth):
    if depth == 0:
        return "bottom"
    return {"level": depth, "next": create_deep_structure(depth - 1)}

dangerous_data = create_deep_structure(100)  # 100 levels deep

# Configure strict security limits
strict_config = UtilityConfig(max_depth=10)

try:
    result = deep_compare(dangerous_data, dangerous_data, config=strict_config)
    print("Comparison completed - this shouldn't happen!")
except UtilitySecurityError as e:
    print(f"✓ Security protection triggered: {e}")
```

### Circular Reference Protection

```python
from datason.utils import enhance_data_types, UtilityConfig, UtilitySecurityError

# Create circular reference that could cause infinite loop
circular_data = {"name": "parent"}
circular_data["self_reference"] = circular_data

# Security config with circular reference detection enabled
config = UtilityConfig(enable_circular_reference_detection=True)

try:
    enhanced, report = enhance_data_types(circular_data, config=config)
    print("Enhancement completed - this shouldn't happen!")
except UtilitySecurityError as e:
    print(f"✓ Circular reference protection: {e}")
```

### Object Size Limits

```python
from datason.utils import find_data_anomalies, UtilityConfig

# Create large object that could exhaust memory
huge_object = {f"field_{i}": f"value_{i}" for i in range(100000)}

# Configure size limits
config = UtilityConfig(max_object_size=50000)

anomalies = find_data_anomalies(huge_object, config=config)
if anomalies["security_violations"]:
    violation = anomalies["security_violations"][0]
    print(f"✓ Object size protection: {violation['violation']}")
    print(f"  Details: {violation['details']}")
```

## 📊 Performance Considerations

### Configurable Performance vs Security Trade-offs

```python
from datason.utils import UtilityConfig

# High-performance config for trusted data
performance_config = UtilityConfig(
    max_depth=1000,           # Deep nesting allowed
    max_object_size=100_000, # Large objects allowed
    enable_circular_reference_detection=False  # Skip circular checks
)

# High-security config for untrusted data
security_config = UtilityConfig(
    max_depth=10,             # Shallow nesting only
    max_object_size=1000,     # Small objects only
    enable_circular_reference_detection=True  # Full protection
)

# Use appropriate config based on data source trust level
def process_data(data, trusted=False):
    config = performance_config if trusted else security_config
    return enhance_data_types(data, config=config)
```

## 🧪 Testing & Quality Assurance

The utils module includes comprehensive testing:

- **40+ test cases** covering all security features
- **Security violation testing** for each protection mechanism
- **Edge case handling** for circular references and malformed data
- **Configuration testing** for different environment scenarios
- **Mock testing** for optional pandas/numpy dependencies
- **73% code coverage** with full type safety

## 🎯 Best Practices

### 1. Choose Appropriate Security Levels
```python
# For user-submitted data (strict)
user_config = UtilityConfig(max_depth=5, max_object_size=10000)

# For internal system data (balanced)
system_config = UtilityConfig(max_depth=25, max_object_size=100000)

# For trusted analytical data (permissive)
analysis_config = UtilityConfig(max_depth=100, max_object_size=1000000)
```

### 2. Handle Security Violations Gracefully
```python
from datason.utils import UtilitySecurityError

try:
    result = process_untrusted_data(data)
except UtilitySecurityError as e:
    logger.warning(f"Security limit exceeded: {e}")
    # Fallback to safer processing or reject data
    result = None
```

### 3. Monitor Security Violations
```python
def analyze_data_safely(data, config):
    anomalies = find_data_anomalies(data, config=config)

    if anomalies["security_violations"]:
        for violation in anomalies["security_violations"]:
            logger.warning(f"Security violation at {violation['path']}: {violation['violation']}")

    return anomalies
```

### 4. Use Utility Discovery for Dynamic Processing
```python
def get_processing_capabilities():
    utilities = get_available_utilities()
    return {
        "comparison": "deep_compare" in utilities["data_comparison"],
        "enhancement": "enhance_data_types" in utilities["data_enhancement"],
        "pandas_support": "enhance_pandas_dataframe" in utilities.get("pandas_integration", []),
    }
```

The enhanced data utilities module provides a comprehensive toolkit for data analysis and transformation while maintaining the same security standards as the core datason serialization engine. This ensures consistent protection across all datason functionality while offering powerful, configurable processing capabilities.
