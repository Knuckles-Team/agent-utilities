"""CONCEPT:AU-P0-2 — a supported Cypher shape routes to the native engine query
client (not a client-side regex interpreter), and an unsupported/unroutable
shape FAILS EXPLICITLY rather than silently returning ``[]``.

``EpistemicGraphBackend.execute`` splits dispatch three ways:

  1. A handful of AU-specific shapes with no native-Cypher equivalent (the
     virtual ``id`` node-identity accessor, relationship traversal/merge — see
     the ``execute`` docstring) stay on typed engine methods.
  2. Everything else that starts with ``MATCH`` (a real WHERE/inline-property
     predicate on a single-node pattern) is handed, params inlined as Cypher
     literals, to ``GraphComputeEngine.query_cypher`` — the engine's own
     parser/executor. This module asserts that hand-off happens (mocking the
     client, not a live engine) and that the query text is NOT regex-parsed
     into a hand-rolled scan-and-filter first.
  3. A query the native engine rejects, or a parameter value its literal
     grammar cannot express (``None``, a negative number), raises a named
     exception (:class:`CypherEngineError` / ``NotImplementedError``) — never a
     silently empty result.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from agent_utilities.knowledge_graph.backends.epistemic_graph_backend import (
    CypherEngineError,
    EpistemicGraphBackend,
    _cypher_literal,
)


def _backend(graph: Any) -> EpistemicGraphBackend:
    """Construct the backend bypassing ``__init__`` (no live engine connect)."""
    b = EpistemicGraphBackend.__new__(EpistemicGraphBackend)
    b._graph = graph
    b._embeddings = {}
    return b


# --- 1. supported queries route to the native client, not a regex scan -------


def test_label_where_query_routes_to_native_client_verbatim() -> None:
    """A label+WHERE MATCH (a real predicate, no id anchor) must be handed to
    ``query_cypher`` with every ``$param`` inlined as a literal — not scanned
    and filtered client-side (the old regex-interpreter behaviour)."""
    graph = MagicMock()
    graph.query_cypher.return_value = [{"id": "n1"}]
    b = _backend(graph)

    rows = b.execute(
        "MATCH (n:Doc) WHERE n.status = $status RETURN n.id AS id",
        {"status": "hot"},
    )

    # ``n.id`` is rewritten to a bare ``n`` first (this backend's virtual ``id``
    # accessor has no reliable native-property mapping — see
    # ``_rewrite_virtual_id_accessors``); everything else passes through as-is.
    graph.query_cypher.assert_called_once_with(
        "MATCH (n:Doc) WHERE n.status = 'hot' RETURN n AS id"
    )
    assert rows == [{"id": "n1"}]
    # The regex label/full-scan typed methods were never touched for this shape.
    graph.get_nodes_by_label.assert_not_called()
    graph._get_all_nodes_with_properties.assert_not_called()


def test_or_disjunction_routes_to_native_client() -> None:
    """An OR disjunction across properties — a shape the old DNF regex parser
    had to hand-roll — is handed straight to the native engine instead."""
    graph = MagicMock()
    graph.query_cypher.return_value = [{"id": "a"}, {"id": "b"}]
    b = _backend(graph)

    rows = b.execute(
        "MATCH (t:Task) WHERE t.status = $a OR t.status = $b RETURN t.id AS id",
        {"a": "pending", "b": "running"},
    )

    graph.query_cypher.assert_called_once_with(
        "MATCH (t:Task) WHERE t.status = 'pending' OR t.status = 'running' "
        "RETURN t AS id"
    )
    assert rows == [{"id": "a"}, {"id": "b"}]


def test_aggregate_with_in_filter_routes_to_native_client() -> None:
    """``count(...)`` with an ``IN`` filter — aggregation is native-engine work,
    not a client-side group-by simulation."""
    graph = MagicMock()
    graph.query_cypher.return_value = [{"cnt": 2}]
    b = _backend(graph)

    rows = b.execute(
        "MATCH (t:Task) WHERE t.status IN ['pending', 'running'] RETURN count(t) AS cnt"
    )

    graph.query_cypher.assert_called_once_with(
        "MATCH (t:Task) WHERE t.status IN ['pending', 'running'] RETURN count(t) AS cnt"
    )
    assert rows == [{"cnt": 2}]


def test_id_anchored_match_does_not_call_native_client() -> None:
    """The virtual ``id`` accessor has no native-Cypher equivalent (this backend
    does not guarantee ``id`` is a stored property) — it must stay on the O(1)
    typed engine method, never reach ``query_cypher``."""
    graph = MagicMock()
    graph.has_node.return_value = True
    graph._get_node_properties.return_value = {"type": "Task", "status": "pending"}
    b = _backend(graph)

    rows = b.execute("MATCH (t:Task {id: $id}) RETURN t.status AS s", {"id": "job-1"})

    graph.query_cypher.assert_not_called()
    assert rows == [{"s": "pending"}]


# --- 2. unsupported / unroutable shapes fail explicitly, never [] -----------


def test_engine_runtime_error_is_wrapped_and_named() -> None:
    """When the native engine itself rejects the query, the backend must raise
    a named error naming the query — not swallow it into ``[]``."""
    graph = MagicMock()
    graph.query_cypher.side_effect = RuntimeError("Cypher error: unknown procedure")
    b = _backend(graph)

    with pytest.raises(CypherEngineError) as exc_info:
        b.execute("MATCH (n:Doc) WHERE n.status = $s RETURN n", {"s": "x"})

    err = exc_info.value
    assert err.query == "MATCH (n:Doc) WHERE n.status = $s RETURN n"
    assert "unknown procedure" in str(err)


def test_negative_number_param_raises_not_implemented_not_empty_list() -> None:
    """The native engine's literal grammar has no negative-number literal — this
    must raise ``NotImplementedError`` naming the gap, not silently drop the
    predicate or return ``[]``."""
    graph = MagicMock()
    b = _backend(graph)

    with pytest.raises(NotImplementedError, match="negative"):
        b.execute("MATCH (n:Metric) WHERE n.delta = $delta RETURN n", {"delta": -1})
    graph.query_cypher.assert_not_called()


def test_none_param_raises_not_implemented_not_empty_list() -> None:
    """The native engine has no NULL literal outside ``IS [NOT] NULL`` — a
    ``None`` parameter used as an equality/SET literal must raise, not silently
    inline as a broken/empty comparison."""
    graph = MagicMock()
    b = _backend(graph)

    with pytest.raises(NotImplementedError, match="NULL"):
        b.execute("MATCH (n:Task) WHERE n.priority = $p RETURN n", {"p": None})
    graph.query_cypher.assert_not_called()


def test_missing_param_raises_value_error_not_empty_list() -> None:
    """A ``$param`` the caller never supplied must raise — the old interpreter
    silently resolved a missing param to ``None``, which is exactly the
    silent-wrong behaviour this workstream removes."""
    graph = MagicMock()
    b = _backend(graph)

    with pytest.raises(ValueError, match="missing parameter"):
        b.execute("MATCH (n:Task) WHERE n.status = $missing RETURN n", {})
    graph.query_cypher.assert_not_called()


# --- 3. literal-inlining helper -----------------------------------------------


def test_cypher_literal_quotes_and_escapes_strings() -> None:
    assert _cypher_literal("hot") == "'hot'"
    assert _cypher_literal("a'b") == "'a\\'b'"


def test_cypher_literal_renders_bool_and_number() -> None:
    assert _cypher_literal(True) == "true"
    assert _cypher_literal(False) == "false"
    assert _cypher_literal(3) == "3"


def test_cypher_literal_renders_list_for_in_clause() -> None:
    assert _cypher_literal(["a", "b"]) == "['a', 'b']"
