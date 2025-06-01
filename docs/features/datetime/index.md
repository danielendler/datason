# Date/Time Handling

Comprehensive support for temporal data with timezone awareness, multiple format options, and seamless integration with pandas datetime objects.

## 🎯 Overview

datason provides intelligent handling of all Python datetime types:

- **Core Types**: `datetime`, `date`, `time`, `timedelta`
- **Pandas Integration**: `pd.Timestamp`, `pd.NaT`, `pd.DatetimeIndex`
- **Format Options**: ISO, Unix timestamp, Unix milliseconds, custom patterns
- **Timezone Support**: Aware and naive datetime handling

## 📅 Supported Date/Time Types

### Standard Library Types
| Type | Example | Default Serialization |
|------|---------|----------------------|
| `datetime` | `datetime(2024, 6, 1, 12, 30)` | ISO format string |
| `date` | `date(2024, 6, 1)` | ISO date string |
| `time` | `time(12, 30, 45)` | ISO time string |
| `timedelta` | `timedelta(days=7)` | Total seconds |

### Pandas Types
| Type | Example | Default Serialization |
|------|---------|----------------------|
| `pd.Timestamp` | `pd.Timestamp('2024-06-01')` | ISO format string |
| `pd.NaT` | `pd.NaT` | `null` |
| `pd.DatetimeIndex` | `pd.date_range('2024-01-01', periods=3)` | List of ISO strings |

## 🔧 Date Format Options

Configure how dates and times are serialized:

```python
from datason import SerializationConfig, DateFormat

# ISO format (default) - human readable
config = SerializationConfig(date_format=DateFormat.ISO)
result = datason.serialize(datetime.now(), config=config)
# Output: "2024-06-01T12:30:45.123456"

# Unix timestamp - compact, fast parsing
config = SerializationConfig(date_format=DateFormat.UNIX)
result = datason.serialize(datetime.now(), config=config)
# Output: 1717243845.123456

# Unix milliseconds - JavaScript compatibility
config = SerializationConfig(date_format=DateFormat.UNIX_MS)
result = datason.serialize(datetime.now(), config=config)
# Output: 1717243845123

# String format - readable
config = SerializationConfig(date_format=DateFormat.STRING)
result = datason.serialize(datetime.now(), config=config)
# Output: "2024-06-01 12:30:45"

# Custom format
config = SerializationConfig(date_format=DateFormat.CUSTOM)
config.custom_date_format = "%Y-%m-%d %H:%M"
result = datason.serialize(datetime.now(), config=config)
# Output: "2024-06-01 12:30"
```

## 🌍 Timezone Handling

### Timezone-Aware Datetimes

```python
from datetime import datetime, timezone, timedelta

# UTC timezone
utc_time = datetime.now(timezone.utc)
result = datason.serialize(utc_time)
# Output: "2024-06-01T12:30:45.123456+00:00"

# Custom timezone
custom_tz = timezone(timedelta(hours=5))
local_time = datetime.now(custom_tz)
result = datason.serialize(local_time)
# Output: "2024-06-01T17:30:45.123456+05:00"
```

### Timezone-Naive Datetimes

```python
# Naive datetime (no timezone info)
naive_time = datetime(2024, 6, 1, 12, 30, 45)
result = datason.serialize(naive_time)
# Output: "2024-06-01T12:30:45"
```

## 📊 Pandas Integration

### Timestamp Handling

```python
import pandas as pd
import datason

# Single timestamp
timestamp = pd.Timestamp('2024-06-01 12:30:45')
result = datason.serialize(timestamp)
# Output: "2024-06-01T12:30:45"

# Timestamp with timezone
timestamp_tz = pd.Timestamp('2024-06-01 12:30:45', tz='UTC')
result = datason.serialize(timestamp_tz)
# Output: "2024-06-01T12:30:45+00:00"

# NaT (Not a Time) handling
nat_value = pd.NaT
result = datason.serialize(nat_value)
# Output: null
```

### DatetimeIndex Serialization

```python
# Date range
date_range = pd.date_range('2024-01-01', periods=3, freq='D')
result = datason.serialize(date_range)
# Output: ["2024-01-01T00:00:00", "2024-01-02T00:00:00", "2024-01-03T00:00:00"]

# With timezone
date_range_tz = pd.date_range('2024-01-01', periods=3, freq='D', tz='UTC')
result = datason.serialize(date_range_tz)
# Output: ["2024-01-01T00:00:00+00:00", "2024-01-02T00:00:00+00:00", "2024-01-03T00:00:00+00:00"]
```

## 🔄 Deserialization

Converting JSON back to datetime objects:

```python
from datason import deserialize

# From ISO string
iso_string = "2024-06-01T12:30:45.123456"
result = deserialize(iso_string, target_type=datetime)
# Output: datetime(2024, 6, 1, 12, 30, 45, 123456)

# From Unix timestamp
unix_timestamp = 1717243845.123456
result = deserialize(unix_timestamp, target_type=datetime)
# Output: datetime(2024, 6, 1, 12, 30, 45, 123456)

# Automatic detection
datetime_data = {"created": "2024-06-01T12:30:45", "updated": 1717243845}
result = deserialize(datetime_data)
# Automatically converts string and timestamp to datetime objects
```

## ⚡ Performance Optimization

### Format Performance Comparison

Based on benchmarking results:

| Format | Performance | Best For |
|--------|-------------|----------|
| **Unix Timestamp** | **3.11ms ± 0.06ms** | Compact, fast parsing |
| Unix Milliseconds | 3.26ms ± 0.05ms | JavaScript compatibility |
| String Format | 3.41ms ± 0.06ms | Human readability |
| ISO Format | 3.46ms ± 0.17ms | Standards compliance |
| Custom Format | 5.16ms ± 0.19ms | Specific requirements |

### Optimization Tips

```python
# For high-performance scenarios
config = datason.get_performance_config()
# Uses Unix timestamps by default

# For API compatibility
config = datason.get_api_config()
# Uses ISO format for standards compliance

# For ML workflows
config = datason.get_ml_config()
# Optimized for numeric data processing
```

## 🛠️ Advanced Usage

### Custom Date Formatting

```python
from datason import SerializationConfig, DateFormat

# Custom strftime format
config = SerializationConfig(
    date_format=DateFormat.CUSTOM,
    custom_date_format="%d/%m/%Y %H:%M"
)

data = {"event_time": datetime(2024, 6, 1, 12, 30)}
result = datason.serialize(data, config=config)
# Output: {"event_time": "01/06/2024 12:30"}
```

### Handling Mixed Temporal Data

```python
# Mixed datetime types in one structure
mixed_data = {
    "created": datetime.now(),
    "updated": pd.Timestamp('2024-06-01'),
    "duration": timedelta(hours=2, minutes=30),
    "schedule": [time(9, 0), time(17, 30)],
    "dates": [date(2024, 6, 1), date(2024, 6, 2)]
}

result = datason.serialize(mixed_data)
# All datetime types handled appropriately
```

### Timezone Conversion

```python
from datetime import timezone, timedelta

# Convert to specific timezone during serialization
def convert_to_utc(dt):
    if dt.tzinfo is None:
        # Assume local timezone
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)

# Custom serializer for timezone normalization
config = SerializationConfig(
    custom_serializers={
        datetime: lambda dt: convert_to_utc(dt).isoformat()
    }
)
```

## 🔧 Configuration Examples

### API-Friendly Configuration

```python
# ISO format for API responses
config = datason.get_api_config()
# Uses ISO format: "2024-06-01T12:30:45.123456Z"

api_response = {
    "user_id": 12345,
    "created_at": datetime.now(),
    "last_login": pd.Timestamp('2024-06-01 10:30'),
    "session_duration": timedelta(minutes=45)
}

result = datason.serialize(api_response, config=config)
```

### Database-Friendly Configuration

```python
# Unix timestamps for database storage
config = SerializationConfig(date_format=DateFormat.UNIX)

database_record = {
    "id": 1,
    "created_at": datetime.now(),
    "updated_at": pd.Timestamp.now(),
    "deleted_at": None
}

result = datason.serialize(database_record, config=config)
# Timestamps as numbers for efficient database storage
```

### Frontend-Friendly Configuration

```python
# JavaScript-compatible timestamps
config = SerializationConfig(date_format=DateFormat.UNIX_MS)

frontend_data = {
    "events": [
        {"timestamp": datetime.now(), "type": "click"},
        {"timestamp": datetime.now() + timedelta(seconds=5), "type": "scroll"}
    ]
}

result = datason.serialize(frontend_data, config=config)
# Timestamps in milliseconds for JavaScript Date() constructor
```

## 🚀 Best Practices

### 1. Choose the Right Format

```python
# For APIs and data exchange
config = SerializationConfig(date_format=DateFormat.ISO)

# For performance-critical applications
config = SerializationConfig(date_format=DateFormat.UNIX)

# For JavaScript frontend
config = SerializationConfig(date_format=DateFormat.UNIX_MS)
```

### 2. Handle Timezones Consistently

```python
# Always be explicit about timezone handling
data = {
    "utc_time": datetime.now(timezone.utc),
    "local_time": datetime.now(),  # Document timezone assumption
}

# Or normalize to UTC
def ensure_utc(dt):
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)
```

### 3. Validate Date Ranges

```python
# Add validation for reasonable date ranges
def validate_datetime(dt):
    min_date = datetime(1970, 1, 1)
    max_date = datetime(2100, 1, 1)

    if not (min_date <= dt <= max_date):
        raise ValueError(f"Date out of range: {dt}")

    return dt
```

## 🔗 Related Features

- **[Configuration System](../configuration/index.md)** - Control date format options
- **[Pandas Integration](../pandas/index.md)** - DataFrame with datetime columns
- **[Core Serialization](../core/index.md)** - Basic serialization principles
- **[Performance](../performance/index.md)** - Optimization strategies

## 🚀 Next Steps

- **[Configuration →](../configuration/index.md)** - Customize date/time serialization
- **[Pandas Integration →](../pandas/index.md)** - Work with DataFrame datetime columns
- **[Performance →](../performance/index.md)** - Optimize datetime handling
