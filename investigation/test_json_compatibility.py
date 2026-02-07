import json
import datason
import pytest
import io

def test_json_arguments_compatibility():
    data = {"a": 1, "b": [2, 3]}
    
    # Test common arguments
    assert datason.dumps(data, indent=4) == json.dumps(data, indent=4)
    assert datason.dumps(data, separators=(',', ':')) == json.dumps(data, separators=(',', ':'))
    assert datason.dumps(data, sort_keys=True) == json.dumps(data, sort_keys=True)
    assert datason.dumps(data, ensure_ascii=False) == json.dumps(data, ensure_ascii=False)

def test_json_dump_to_file():
    data = {"a": 1}
    f1 = io.StringIO()
    f2 = io.StringIO()
    
    datason.dump(data, f1)
    json.dump(data, f2)
    
    assert f1.getvalue() == f2.getvalue()

def test_json_loads_arguments():
    s = '{"a": 1}'
    assert datason.loads(s) == json.loads(s)
    
    # Test parse_float, parse_int, etc.
    assert datason.loads('1.5', parse_float=lambda x: "float") == json.loads('1.5', parse_float=lambda x: "float")

def test_unsupported_json_arguments():
    # json.dumps has skipkeys, allow_nan, check_circular, cls, default
    data = {tuple([1,2]): "oops"} # tuple key is not allowed in JSON
    
    try:
        json.dumps(data, skipkeys=True)
    except Exception as e:
        print(f"JSON error with skipkeys=False: {e}")

    # Check if datason supports skipkeys
    try:
        res = datason.dumps(data, skipkeys=True)
        assert res == "{}"
    except TypeError as e:
        pytest.fail(f"datason.dumps does not support skipkeys: {e}")

if __name__ == "__main__":
    pytest.main([__file__])
