"""Primitive fast traversal must retain budgets and normal value policies."""

import json

import pytest

import datason
from datason._errors import SecurityError


@pytest.mark.parametrize("value", [None, True, 7, 1.25, "text"])
def test_root_primitive_consumes_node_budget(value):
    with pytest.raises(SecurityError, match="node"):
        datason.dumps(value, max_nodes=0)
    assert datason.loads(datason.dumps(value, max_nodes=1)) == value


@pytest.mark.parametrize("value", [None, True, 7, 1.25, "text"])
def test_nested_primitive_obeys_depth_budget(value):
    with pytest.raises(SecurityError, match="depth"):
        datason.dumps([value], max_depth=0)


def test_shared_containers_and_repeated_primitives_are_not_cycles():
    shared = ["same", 1, True, None]
    assert json.loads(datason.dumps([shared, shared])) == [shared, shared]
    shared.append(shared)
    with pytest.raises(SecurityError, match="Circular"):
        datason.dumps(shared)
