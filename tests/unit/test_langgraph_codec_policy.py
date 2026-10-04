"""Optional import failures must not conceal broken installed dependencies."""

import pytest

from datason.integrations import _langgraph_types as codec


@pytest.mark.parametrize("missing", ["langgraph", "langgraph.types", "langchain_core"])
def test_codec_registration_only_suppresses_missing_optional_sdk(monkeypatch, missing):
    failure = ModuleNotFoundError("Missing fixture dependency", name=missing)

    def missing_dependency(name):
        assert name == "langgraph.types"
        raise failure

    monkeypatch.setattr(codec.importlib, "import_module", missing_dependency)
    monkeypatch.setattr(codec.default_registry, "register_once", lambda plugin: pytest.fail("broken codec registered"))
    if missing == "langchain_core":
        with pytest.raises(ModuleNotFoundError) as caught:
            codec.register_langgraph_types()
        assert caught.value is failure
    else:
        codec.register_langgraph_types()
