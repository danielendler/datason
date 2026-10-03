import datason
import json
import datetime as dt
import time

def test_large_dataset_scaling():
    # 10,000 complex objects
    large_data = [{"id": i, "ts": dt.datetime.now(), "meta": {"val": i * 1.1}} for i in range(10000)]
    
    start = time.time()
    d_str = datason.dumps(large_data)
    d_time = time.time() - start
    
    def default(obj):
        if isinstance(obj, dt.datetime): return obj.isoformat()
        return str(obj)
        
    start = time.time()
    j_str = json.dumps(large_data, default=default)
    j_time = time.time() - start
    
    print(f"Large Dataset (10k items) Serialization:")
    print(f"  datason: {d_time:.4f}s")
    print(f"  json:    {j_time:.4f}s")
    print(f"  Ratio:   {d_time/j_time:.1f}x slower")

    start = time.time()
    datason.loads(d_str)
    dl_time = time.time() - start
    print(f"Large Dataset (10k items) Deserialization:")
    print(f"  datason: {dl_time:.4f}s")

if __name__ == "__main__":
    test_large_dataset_scaling()
