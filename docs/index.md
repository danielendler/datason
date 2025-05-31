# Welcome to datason

**A comprehensive Python package for intelligent serialization that handles complex data types with ease**

datason transforms complex Python objects into JSON-serializable formats and back with intelligence. Perfect for ML/AI workflows, data science, and any application dealing with complex nested data structures.

```python
import datason as ds
import pandas as pd
from datetime import datetime

# Works with complex data out of the box
data = {
    'dataframe': pd.DataFrame({'A': [1, 2, 3]}),
    'timestamp': datetime.now(),
    'nested': {'values': [1, 2, 3]}
}

json_data = ds.serialize(data)
# Works perfectly! 🎉
```

## Features

- **🧠 Intelligent Type Detection**: Automatically handles pandas DataFrames, NumPy arrays, datetime objects, and more
- **🔄 Bidirectional**: Serialize to JSON and deserialize back to original objects  
- **🚀 ML/AI Optimized**: Special support for PyTorch tensors, TensorFlow objects, and scikit-learn models
- **🛡️ Type Safety**: Preserves data types and structure integrity
- **⚡ High Performance**: Optimized for speed with minimal overhead
- **🔌 Extensible**: Easy to add custom serializers for your own types
- **📦 Zero Dependencies**: Core functionality works without additional packages

## Quick Start

```bash
pip install datason
```

## Documentation

- [API Reference](api/)
- [Examples](examples/)
- [Contributing](contributing/)

## License

MIT License - see [LICENSE](https://github.com/danielendler/datason/blob/main/LICENSE) file for details.
