"""_sanitize_param: scalar lists -> Postgres array (not JSON). Fixes KG-2.9 mirror bug
'malformed array literal "["freshrss"]"' on TEXT[] columns."""

from __future__ import annotations

import json

from agent_utilities.knowledge_graph.backends.cypher_transpiler import _sanitize_param


def test_scalar_list_passed_through_for_pg_array():
    # the bug: ["freshrss"] was json.dumps'd to '["freshrss"]' which PG rejects on TEXT[]
    assert _sanitize_param(["freshrss"]) == ["freshrss"]
    assert _sanitize_param(["a", "b", "c"]) == ["a", "b", "c"]
    assert _sanitize_param([1, 2, 3]) == [1, 2, 3]
    assert _sanitize_param([]) == []


def test_dict_still_json_encoded():
    assert _sanitize_param({"a": 1}) == json.dumps({"a": 1}, default=str)


def test_list_of_nested_json_encoded():
    # can't be a PG array of scalars → JSON for a TEXT column
    v = [{"k": 1}, {"k": 2}]
    assert _sanitize_param(v) == json.dumps(v, default=str)
    assert _sanitize_param([["x"]]) == json.dumps([["x"]], default=str)


def test_scalars_unchanged_and_nul_stripped():
    assert _sanitize_param("plain") == "plain"
    assert _sanitize_param("a\x00b") == "ab"
    assert _sanitize_param(42) == 42
