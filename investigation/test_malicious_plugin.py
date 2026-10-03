import datason
from datason._protocols import TypePlugin, SerializeContext, DeserializeContext
from datason._registry import default_registry
from datason._types import TYPE_METADATA_KEY, VALUE_METADATA_KEY
import pytest
import os

class Exploit: pass

class MaliciousPlugin:
    name = "malicious"
    priority = 500

    def can_handle(self, obj):
        return isinstance(obj, Exploit)

    def serialize(self, obj, ctx):
        return {TYPE_METADATA_KEY: "Exploit", VALUE_METADATA_KEY: "pwned"}

    def can_deserialize(self, data):
        return data.get(TYPE_METADATA_KEY) == "Exploit"

    def deserialize(self, data, ctx):
        # Arbitrary code execution during loads()
        print("PLUGIN EXPLOIT EXECUTED")
        return "ExploitExecuted"

def test_plugin_rce():
    default_registry.register(MaliciousPlugin())
    
    e = Exploit()
    # Serialize
    json_str = datason.dumps(e)
    assert "Exploit" in json_str
    
    # Deserialize triggers the exploit
    res = datason.loads(json_str)
    assert res == "ExploitExecuted"

if __name__ == "__main__":
    pytest.main([__file__])

if __name__ == "__main__":
    pytest.main([__file__])
