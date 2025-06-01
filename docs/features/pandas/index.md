# Pandas Integration

Deep integration with the pandas ecosystem for data science workflows, including DataFrames, Series, Index types, and specialized pandas objects.

## 🎯 Overview

datason provides comprehensive support for pandas objects:

- **DataFrames**: Configurable orientation (records, split, index, columns, values, table)
- **Series**: Index preservation and metadata handling
- **Index Types**: RangeIndex, DatetimeIndex, MultiIndex, CategoricalIndex
- **Categorical**: Category metadata and ordering preservation
- **NaN Handling**: Configurable strategies for missing data
- **Performance**: Optimized serialization for large datasets

## 📊 DataFrame Serialization

### Orientation Options

DataFrames can be serialized in different orientations based on your use case:

```python
import pandas as pd
import datason
from datason import SerializationConfig, DataFrameOrient

# Sample DataFrame
df = pd.DataFrame({
    'A': [1, 2, 3],
    'B': ['x', 'y', 'z'],
    'C': [10.1, 20.2, 30.3]
})

# Records orientation (default) - intuitive for APIs
config = SerializationConfig(dataframe_orient=DataFrameOrient.RECORDS)
result = datason.serialize(df, config=config)
# Output: [{"A": 1, "B": "x", "C": 10.1}, {"A": 2, "B": "y", "C": 20.2}, ...]

# Split orientation - efficient for large datasets
config = SerializationConfig(dataframe_orient=DataFrameOrient.SPLIT)
result = datason.serialize(df, config=config)
# Output: {"columns": ["A", "B", "C"], "data": [[1, "x", 10.1], [2, "y", 20.2], ...]}

# Values orientation - array-like data only
config = SerializationConfig(dataframe_orient=DataFrameOrient.VALUES)
result = datason.serialize(df, config=config)
# Output: [[1, "x", 10.1], [2, "y", 20.2], [3, "z", 30.3]]

# Index orientation - preserves row labels
config = SerializationConfig(dataframe_orient=DataFrameOrient.INDEX)
result = datason.serialize(df, config=config)
# Output: {"0": {"A": 1, "B": "x", "C": 10.1}, "1": {"A": 2, "B": "y", "C": 20.2}, ...}

# Columns orientation - column-wise data
config = SerializationConfig(dataframe_orient=DataFrameOrient.COLUMNS)
result = datason.serialize(df, config=config)
# Output: {"A": [1, 2, 3], "B": ["x", "y", "z"], "C": [10.1, 20.2, 30.3]}
```

### Performance Comparison

Based on benchmarking results:

| Orientation | Small DataFrames (<1K rows) | Large DataFrames (>1K rows) | Best For |
|-------------|----------------------------|----------------------------|----------|
| **Values** | 0.07ms (fastest) | 2.42ms | Array-like processing |
| **Split** | 0.21ms | **1.63ms (fastest)** | Large datasets, efficiency |
| **Records** | 0.24ms | 2.48ms | APIs, intuitive structure |
| **Columns** | 0.25ms | 2.45ms | Column-wise operations |
| **Index** | 0.30ms | 2.80ms | Row-labeled data |

## 📈 Series Handling

### Basic Series Serialization

```python
import pandas as pd
import datason

# Simple series
series = pd.Series([1, 2, 3, 4, 5], name='values')
result = datason.serialize(series)
# Output: {"type": "pandas.Series", "data": [1, 2, 3, 4, 5], "name": "values", "index": [0, 1, 2, 3, 4]}

# Series with custom index
series = pd.Series([10, 20, 30], index=['a', 'b', 'c'], name='measurements')
result = datason.serialize(series)
# Output: {"type": "pandas.Series", "data": [10, 20, 30], "name": "measurements", "index": ["a", "b", "c"]}
```

### Series with DatetimeIndex

```python
# Time series data
dates = pd.date_range('2024-01-01', periods=3, freq='D')
ts = pd.Series([100, 105, 110], index=dates, name='stock_price')
result = datason.serialize(ts)
# Preserves datetime index information
```

## 🏷️ Index Types

### RangeIndex
```python
# Default integer index
df = pd.DataFrame({'values': [1, 2, 3]})
result = datason.serialize(df)
# RangeIndex is efficiently represented
```

### DatetimeIndex
```python
# Time-based index
dates = pd.date_range('2024-01-01', periods=5, freq='D')
df = pd.DataFrame({'values': range(5)}, index=dates)
result = datason.serialize(df)
# Datetime index preserved with timezone information
```

### MultiIndex
```python
# Hierarchical index
arrays = [['A', 'A', 'B', 'B'], [1, 2, 1, 2]]
index = pd.MultiIndex.from_arrays(arrays, names=['first', 'second'])
df = pd.DataFrame({'values': [10, 20, 30, 40]}, index=index)
result = datason.serialize(df)
# MultiIndex structure and names preserved
```

### CategoricalIndex
```python
# Categorical index
categories = pd.CategoricalIndex(['small', 'medium', 'large'], ordered=True)
df = pd.DataFrame({'count': [10, 20, 30]}, index=categories)
result = datason.serialize(df)
# Category order and metadata preserved
```

## 🔢 Categorical Data

### Basic Categorical Handling
```python
# Categorical columns
df = pd.DataFrame({
    'grade': pd.Categorical(['A', 'B', 'A', 'C'], ordered=True),
    'size': pd.Categorical(['S', 'M', 'L', 'M'], categories=['S', 'M', 'L', 'XL'])
})
result = datason.serialize(df)
# Categories, ordering, and unused categories preserved
```

### Categorical with Custom Ordering
```python
# Custom category order
priority = pd.Categorical(['high', 'low', 'medium', 'high'],
                         categories=['low', 'medium', 'high'],
                         ordered=True)
df = pd.DataFrame({'priority': priority, 'task_id': range(4)})
result = datason.serialize(df)
# Custom ordering preserved for proper reconstruction
```

## 🗃️ Missing Data Handling

### NaN Strategies
```python
from datason import SerializationConfig, NanHandling

# DataFrame with missing data
df = pd.DataFrame({
    'A': [1, 2, None, 4],
    'B': [1.1, None, 3.3, 4.4],
    'C': ['x', 'y', None, 'z']
})

# Convert to null (default)
config = SerializationConfig(nan_handling=NanHandling.NULL)
result = datason.serialize(df, config=config)
# None/NaN → null in JSON

# Convert to string representation
config = SerializationConfig(nan_handling=NanHandling.STRING)
result = datason.serialize(df, config=config)
# None/NaN → "NaN" string

# Drop missing values
config = SerializationConfig(nan_handling=NanHandling.DROP)
result = datason.serialize(df, config=config)
# Rows/columns with missing data removed

# Keep original (preserve NaN as special value)
config = SerializationConfig(nan_handling=NanHandling.PRESERVE)
result = datason.serialize(df, config=config)
# Special encoding to preserve exact NaN semantics
```

## ⚡ Performance Optimization

### Large DataFrame Handling
```python
# For large DataFrames, use split orientation
large_df = pd.DataFrame(np.random.randn(10000, 50))

config = datason.get_performance_config()
# Automatically uses split orientation for large DataFrames
result = datason.serialize(large_df, config=config)
```

### Memory-Efficient Serialization
```python
# Chunked processing for very large datasets
def serialize_large_dataframe(df, chunk_size=1000):
    """Serialize large DataFrame in chunks."""
    chunks = []
    for i in range(0, len(df), chunk_size):
        chunk = df.iloc[i:i+chunk_size]
        chunks.append(datason.serialize(chunk))
    return {"chunks": chunks, "total_rows": len(df)}
```

### Optimized Data Types
```python
# Use efficient data types before serialization
df = df.astype({
    'category_col': 'category',  # Use categorical for repeated strings
    'int_col': 'int32',  # Use smaller int types when possible
    'float_col': 'float32'  # Use float32 instead of float64 when precision allows
})
result = datason.serialize(df)
```

## 🔄 Deserialization

### Reconstructing DataFrames
```python
from datason import deserialize

# Deserialize back to DataFrame
serialized_df = datason.serialize(df)
reconstructed_df = deserialize(serialized_df)

# Verify reconstruction
assert df.equals(reconstructed_df)
assert df.dtypes.equals(reconstructed_df.dtypes)
assert df.index.equals(reconstructed_df.index)
```

### Handling Type Information
```python
# Automatic type reconstruction
original_df = pd.DataFrame({
    'dates': pd.date_range('2024-01-01', periods=3),
    'categories': pd.Categorical(['A', 'B', 'A']),
    'numbers': [1, 2, 3],
    'floats': [1.1, 2.2, 3.3]
})

# Serialize and deserialize
result = datason.serialize(original_df)
restored_df = datason.deserialize(result)

# All dtypes preserved automatically
print(original_df.dtypes)
print(restored_df.dtypes)
# Output should be identical
```

## 🧪 Advanced Use Cases

### Mixed Data Types
```python
# DataFrame with complex mixed types
complex_df = pd.DataFrame({
    'id': range(3),
    'timestamp': pd.date_range('2024-01-01', periods=3),
    'data': [{'key': 'value'}, [1, 2, 3], 'simple_string'],
    'numbers': np.array([1.1, 2.2, 3.3]),
    'categories': pd.Categorical(['type1', 'type2', 'type1'])
})

result = datason.serialize(complex_df)
# All types handled appropriately
```

### Sparse DataFrames
```python
# Sparse data handling
sparse_df = pd.DataFrame({
    'A': pd.arrays.SparseArray([0, 0, 1, 0, 1]),
    'B': pd.arrays.SparseArray([0, 1, 0, 0, 0]),
    'C': range(5)
})
result = datason.serialize(sparse_df)
# Sparse arrays converted to dense for JSON compatibility
```

### DataFrame with Custom Objects
```python
# DataFrame containing custom objects
class CustomObject:
    def __init__(self, value):
        self.value = value

    def __repr__(self):
        return f"CustomObject({self.value})"

df = pd.DataFrame({
    'id': [1, 2, 3],
    'custom': [CustomObject(10), CustomObject(20), CustomObject(30)]
})

result = datason.serialize(df)
# Custom objects handled with fallback serialization
```

## 🔧 Configuration Examples

### API Response Format
```python
# Optimized for API responses
config = datason.get_api_config()
api_df = pd.DataFrame({
    'user_id': [1, 2, 3],
    'created_at': pd.date_range('2024-01-01', periods=3),
    'score': [85.5, 92.1, 78.3]
})
result = datason.serialize(api_df, config=config)
# Uses records orientation with ISO date format
```

### Database Export Format
```python
# Optimized for database import
config = SerializationConfig(
    dataframe_orient=DataFrameOrient.SPLIT,
    date_format=DateFormat.UNIX,
    nan_handling=NanHandling.NULL
)
result = datason.serialize(df, config=config)
# Efficient format with numeric timestamps
```

### Analytics Format
```python
# Optimized for analytics tools
config = SerializationConfig(
    dataframe_orient=DataFrameOrient.COLUMNS,
    preserve_categorical_ordering=True,
    include_dtype_info=True
)
result = datason.serialize(analytics_df, config=config)
# Column-wise data with full type preservation
```

## 🚀 Best Practices

### 1. Choose the Right Orientation
```python
# For APIs with row-based processing
config = SerializationConfig(dataframe_orient=DataFrameOrient.RECORDS)

# For large datasets and performance
config = SerializationConfig(dataframe_orient=DataFrameOrient.SPLIT)

# For column-wise analytics
config = SerializationConfig(dataframe_orient=DataFrameOrient.COLUMNS)
```

### 2. Handle Missing Data Consistently
```python
# Be explicit about NaN handling
config = SerializationConfig(nan_handling=NanHandling.NULL)
# Ensures consistent behavior across different data sources
```

### 3. Preserve Important Metadata
```python
# Include index information when important
df_with_meaningful_index = pd.DataFrame(
    {'values': [1, 2, 3]},
    index=['row1', 'row2', 'row3']
)
# Index will be preserved automatically
```

### 4. Optimize Data Types
```python
# Use appropriate data types for efficiency
df = df.astype({
    'category_column': 'category',
    'small_int_column': 'int8',
    'date_column': 'datetime64[ns]'
})
```

## 🔗 Related Features

- **[Configuration System](../configuration/index.md)** - DataFrame-specific configurations
- **[Date/Time Handling](../datetime/index.md)** - Temporal data in DataFrames
- **[Advanced Types](../advanced-types/index.md)** - Complex data within DataFrames
- **[Performance](../performance/index.md)** - Large DataFrame optimization

## 🚀 Next Steps

- **[Configuration →](../configuration/index.md)** - Customize DataFrame serialization
- **[Performance →](../performance/index.md)** - Optimize for large datasets
- **[ML/AI Integration →](../ml-ai/index.md)** - Use DataFrames in ML workflows
