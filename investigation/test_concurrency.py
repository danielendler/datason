import datason
import threading
import time
import pytest
import json

def test_thread_safety_of_config():
    # Test if datason.config is truly thread-local
    results = {}
    
    def worker(name, sort_keys_val):
        with datason.config(sort_keys=sort_keys_val):
            # Sleep a bit to allow other threads to potentially overwrite
            time.sleep(0.1)
            data = {"z": 1, "a": 2}
            results[name] = datason.dumps(data)
            
    t1 = threading.Thread(target=worker, args=("t1", True))
    t2 = threading.Thread(target=worker, args=("t2", False))
    
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    
    assert results["t1"] == '{"a": 2, "z": 1}'
    # For t2, we can't be sure of the order if sort_keys is False, 
    # but it shouldn't be guaranteed sorted if it was set to False.
    # Actually, if it leaked from t1, it WOULD be sorted.
    # Standard dict order in Python 3.7+ is insertion order, so {"z": 1, "a": 2} should be '{"z": 1, "a": 2}'
    assert results["t2"] == '{"z": 1, "a": 2}'

if __name__ == "__main__":
    pytest.main([__file__])
