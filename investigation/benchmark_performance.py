import datason
import json
import datetime as dt
import numpy as np
import timeit
from decimal import Decimal
import uuid

# Sample data for benchmarking
basic_data = {
    "str": "hello" * 10,
    "int": 12345,
    "float": 3.14159,
    "bool": True,
    "none": None,
    "list": list(range(100)),
    "dict": {f"key_{i}": i for i in range(100)}
}

complex_data = {
    "dt": dt.datetime.now(),
    "uuid": uuid.uuid4(),
    "decimal": Decimal("123.456"),
    "numpy": np.random.rand(10, 10),
    "list": [dt.datetime.now() for _ in range(10)],
    "nested": {"a": [Decimal("1.1"), uuid.uuid4()]}
}

def json_default(obj):
    if isinstance(obj, dt.datetime):
        return obj.isoformat()
    if isinstance(obj, uuid.UUID):
        return str(obj)
    if isinstance(obj, Decimal):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

def run_benchmarks():
    print("--- Performance Benchmarks ---")
    
    # 1. Basic Data Serialization
    t_json_basic = timeit.timeit(lambda: json.dumps(basic_data), number=1000)
    t_datason_basic = timeit.timeit(lambda: datason.dumps(basic_data), number=1000)
    
    print(f"Basic Data (1000 iterations):")
    print(f"  json.dumps:    {t_json_basic:.4f}s")
    print(f"  datason.dumps: {t_datason_basic:.4f}s ({(t_datason_basic/t_json_basic):.1f}x slower)")

    # 2. Complex Data Serialization
    t_json_complex = timeit.timeit(lambda: json.dumps(complex_data, default=json_default), number=1000)
    t_datason_complex = timeit.timeit(lambda: datason.dumps(complex_data), number=1000)
    
    print(f"\nComplex Data (1000 iterations):")
    print(f"  json.dumps (w/ default): {t_json_complex:.4f}s")
    print(f"  datason.dumps:           {t_datason_complex:.4f}s ({(t_datason_complex/t_json_complex):.1f}x slower)")

    # 3. Deserialization
    json_str_basic = json.dumps(basic_data)
    datason_str_basic = datason.dumps(basic_data)
    
    t_json_load_basic = timeit.timeit(lambda: json.loads(json_str_basic), number=1000)
    t_datason_load_basic = timeit.timeit(lambda: datason.loads(datason_str_basic), number=1000)
    
    print(f"\nBasic Deserialization (1000 iterations):")
    print(f"  json.loads:    {t_json_load_basic:.4f}s")
    print(f"  datason.loads: {t_datason_load_basic:.4f}s ({(t_datason_load_basic/t_json_load_basic):.1f}x slower)")

    # Complex Deserialization (Round-trip)
    datason_str_complex = datason.dumps(complex_data)
    t_datason_load_complex = timeit.timeit(lambda: datason.loads(datason_str_complex), number=1000)
    
    print(f"\nComplex Deserialization (1000 iterations):")
    print(f"  datason.loads: {t_datason_load_complex:.4f}s (Full type reconstruction)")

if __name__ == "__main__":
    run_benchmarks()