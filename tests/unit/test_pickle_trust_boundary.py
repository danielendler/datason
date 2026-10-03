"""Pickle migration must fail before executing or reading untrusted input."""

import pickle
from pathlib import Path

import pytest

from datason._errors import SecurityError
from datason.security.pickle_bridge import pickle_file_to_json, pickle_to_json, validate_pickle_safety


class EvaluationPayload:
    def __reduce__(self):
        return eval, ("40 + 2",)


def test_allowed_module_does_not_authorize_execution() -> None:
    payload = pickle.dumps(EvaluationPayload())
    assert validate_pickle_safety(payload)[0]
    with pytest.raises(SecurityError, match="trusted=True"):
        pickle_to_json(payload)


@pytest.mark.parametrize("trusted", [False, None, 1, "yes"])
def test_trust_must_be_explicit(trusted, monkeypatch) -> None:
    def forbidden_load(*args, **kwargs):
        pytest.fail("untrusted pickle was executed")

    monkeypatch.setattr(pickle, "loads", forbidden_load)
    with pytest.raises(SecurityError, match="trusted=True"):
        pickle_to_json(b"not even a pickle", trusted=trusted)


def test_untrusted_file_is_not_opened(tmp_path: Path) -> None:
    with pytest.raises(SecurityError, match="trusted=True"):
        pickle_file_to_json(str(tmp_path / "missing.pkl"))


def test_explicitly_trusted_migration_remains_available() -> None:
    assert pickle_to_json(pickle.dumps({"answer": 42}), trusted=True) == '{"answer": 42}'
