import datason
import pytest
from datason._errors import DeserializationError
import datetime as dt

def test_untrusted_type_hint():
    # If the attacker knows 'pathlib.Path' is handled:
    payload = '{"__datason_type__": "pathlib.Path", "__datason_value__": "/etc/passwd"}'
    res = datason.loads(payload)
    from pathlib import Path
    assert isinstance(res, Path)
    assert str(res) == "/etc/passwd"

def test_untrusted_type_hint_unknown():
    # Test strict mode with unknown type hint
    payload = '{"__datason_type__": "nonexistent_type", "__datason_value__": "some_value"}'
    
    # By default strict is True
    with pytest.raises(DeserializationError):
        datason.loads(payload)
        
    # With strict=False, it should return the dict as-is
    res = datason.loads(payload, strict=False)
    assert isinstance(res, dict)
    assert res["__datason_type__"] == "nonexistent_type"

def test_untrusted_datetime_malformed():
    # Attacker sends a non-string value for a datetime which expects string or number
    payload = '{"__datason_type__": "datetime", "__datason_value__": [1, 2, 3]}'
    # This should probably raise a PluginError which is then caught and warned about, 
    # but since it's the only plugin matching that type name, it might fail.
    # Actually registry says it warns and continues. If no more plugins, it fails if strict.
    with pytest.raises(DeserializationError):
        datason.loads(payload)

def test_untrusted_huge_int():
    # JSON handles arbitrarily large ints, but let's see if datason has issues
    huge_int = 10**10000
    payload = f'{{"val": {huge_int}}}'
    res = datason.loads(payload)
    assert res["val"] == huge_int

if __name__ == "__main__":
    pytest.main([__file__])
