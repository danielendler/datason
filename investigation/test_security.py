import datason
import pytest
import json
from datason._errors import SecurityError

def test_max_depth_limit():
    # Create a deeply nested object
    data = {}
    curr = data
    for _ in range(55): # Default is 50
        curr['a'] = {}
        curr = curr['a']
    
    with pytest.raises(SecurityError) as excinfo:
        datason.dumps(data)
    assert "depth" in str(excinfo.value).lower()

def test_max_size_limit():
    # Create a large dictionary
    data = {str(i): i for i in range(100_005)} # Default is 100,000
    
    with pytest.raises(SecurityError) as excinfo:
        datason.dumps(data)
    assert "size" in str(excinfo.value).lower()

def test_circular_reference_detection():
    a = {}
    b = {'a': a}
    a['b'] = b
    
    # Standard json raises RecursionError or similar, datason should handle it
    with pytest.raises((SecurityError, RecursionError)):
        datason.dumps(a)

def test_malicious_type_hint_deserialization():
    # Attempt to inject a type hint that might cause issues
    malicious_json = '{"__datason_type__": "os.system", "__datason_value__": "echo pwned"}'
    
    # By default it should probably just return the dict if it's not registered, 
    # OR raise an error if strict=True (default is True)
    try:
        # If strict=True, it should raise DeserializationError for unknown type hints
        from datason._errors import DeserializationError
        with pytest.raises(DeserializationError):
             datason.loads(malicious_json, strict=True)
    except ImportError:
        res = datason.loads(malicious_json)
        assert isinstance(res, dict)

def test_type_hint_with_large_value():
    # What if the value inside a type hint is huge?
    malicious_json = '{"__datason_type__": "datetime", "__datason_value__": "' + "A" * 1_000_000 + '"}'
    # This might cause a crash in the datetime plugin if it doesn't validate input length
    # or just a normal error during parsing.
    with pytest.raises(Exception):
        datason.loads(malicious_json)

def test_redaction_bypass_attempt():
    # Test if we can bypass redaction by using different cases or nested keys
    data = {"Password": "secret123", "user": {"password": "nested_secret"}}
    
    # If redact_fields is case-insensitive substring match as per README
    redacted = datason.dumps(data, redact_fields=("password",))
    
    assert "secret123" not in redacted
    assert "nested_secret" not in redacted
    assert "[REDACTED]" in redacted

def test_regex_redaction_dos():
    # Potential ReDoS if patterns are not handled carefully
    # Use a slow regex if possible, though built-in ones are used.
    # Let's see if we can pass a malicious regex.
    malicious_regex = "(a+)+$"
    payload = "a" * 30 + "!"
    
    # The README says: datason.dumps(data, redact_patterns=("email", "ssn"))
    # Can we pass a raw regex string?
    try:
        datason.dumps({"input": payload}, redact_patterns=(malicious_regex,))
    except Exception as e:
        print(f"Regex error: {e}")

if __name__ == "__main__":
    pytest.main([__file__])
