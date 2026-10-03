import datason
import json
import datetime as dt
from datason import DateFormat, NanHandling

def test_config_ergonomics():
    data = {"ts": dt.datetime(2024, 1, 1), "val": float('nan')}
    
    # Datason approach
    with datason.config(date_format=DateFormat.UNIX, nan_handling=NanHandling.STRING):
        d_res = datason.dumps(data)
    
    # Standard JSON approach
    def my_default(obj):
        if isinstance(obj, dt.datetime):
            return obj.timestamp()
        return str(obj)
    
    j_res = json.dumps(data, default=my_default).replace('NaN', 'nan') # Hacky
    
    print(f"Datason result: {d_res}")
    print(f"JSON result:    {j_res}")

def test_api_preset():
    # Test if presets are actually useful
    from datason import api_config
    data = {"b": 2, "a": 1, "ts": dt.datetime.now()}
    
    # api_config: ISO dates, sorted keys, no type hints
    res = datason.dumps(data, **api_config().__dict__)
    print(f"API Preset result: {res}")
    assert '"a": 1, "b": 2' in res
    assert "__datason_type__" not in res

if __name__ == "__main__":
    test_config_ergonomics()
    test_api_preset()
