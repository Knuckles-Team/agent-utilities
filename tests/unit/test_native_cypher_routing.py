"""Native Cypher authority contracts (roadmap item 3)."""

from __future__ import annotations

import ast
import inspect
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from agent_utilities.knowledge_graph.backends.epistemic_graph_backend import (
    CypherEngineError,
    EpistemicGraphBackend,
    _cypher_literal,
)


def _backend(graph: Any) -> EpistemicGraphBackend:
    backend = EpistemicGraphBackend.__new__(EpistemicGraphBackend)
    backend._graph = graph
    backend.graph_name = "test-graph"
    return backend


def test_read_delegates_to_native_read_mode_without_client_interpretation() -> None:
    graph = MagicMock()
    graph.query_cypher.return_value = [{"id": "node-1"}]
    backend = _backend(graph)

    rows = backend.execute_read(
        "MATCH (n:Record) WHERE n.status = $status RETURN n.id AS id",
        {"status": "ready"},
    )

    assert rows == [{"id": "node-1"}]
    graph.query_cypher.assert_called_once_with(
        "MATCH (n:Record) WHERE n.status = 'ready' RETURN n.id AS id"
    )
    graph.query_cypher_write.assert_not_called()


def test_internal_write_delegates_to_native_mutation_mode() -> None:
    graph = MagicMock()
    graph.query_cypher_write.return_value = []
    backend = _backend(graph)

    assert (
        backend.execute(
            "MATCH (n:Record {id: $id}) SET n.status = $status",
            {"id": "node-1", "status": "done"},
        )
        == []
    )
    graph.query_cypher_write.assert_called_once_with(
        "MATCH (n:Record {id: 'node-1'}) SET n.status = 'done'"
    )
    graph.query_cypher.assert_not_called()


def test_native_mode_mismatch_is_sanitized() -> None:
    graph = MagicMock()
    graph.query_cypher.side_effect = RuntimeError(
        "backend details containing a query, endpoint, or credential"
    )
    backend = _backend(graph)

    with pytest.raises(CypherEngineError) as caught:
        backend.execute_read("MATCH (n:Record) RETURN n")

    error = caught.value
    assert error.mode == "read"
    assert error.error_type == "RuntimeError"
    assert len(error.query_reference) == 16
    assert "backend details" not in str(error)
    assert "MATCH" not in str(error)
    assert error.__cause__ is None


def test_missing_parameter_fails_before_native_dispatch() -> None:
    graph = MagicMock()
    backend = _backend(graph)

    with pytest.raises(ValueError, match="missing a referenced parameter"):
        backend.execute_read("MATCH (n {id: $missing}) RETURN n")

    graph.query_cypher.assert_not_called()


@pytest.mark.parametrize("value", [None, -1, {"not": "scalar"}])
def test_unrepresentable_parameter_fails_closed(value: Any) -> None:
    graph = MagicMock()
    backend = _backend(graph)

    with pytest.raises((TypeError, ValueError, NotImplementedError)):
        backend.execute_read(
            "MATCH (n:Record) WHERE n.value = $value RETURN n",
            {"value": value},
        )

    graph.query_cypher.assert_not_called()


def test_raw_cypher_batch_translation_is_removed() -> None:
    graph = MagicMock()
    backend = _backend(graph)

    with pytest.raises(RuntimeError, match="ChangeEnvelope"):
        backend.execute_batch(
            "UNWIND $batch AS row MERGE (n:Record {id: row.id})",
            [{"id": "node-1"}],
        )

    graph.query_cypher_write.assert_not_called()


def test_backend_contains_no_alternate_cypher_engine() -> None:
    source = "\n".join(
        inspect.getsource(method)
        for method in (
            EpistemicGraphBackend.execute,
            EpistemicGraphBackend.execute_read,
            EpistemicGraphBackend.execute_write,
            EpistemicGraphBackend.execute_batch,
        )
    )
    retired = (
        "_exec_node_match",
        "_exec_rel_match",
        "_exec_rel_merge",
        "_exec_merge_node",
        "_exec_var_length_match",
        "_parse_where",
        "_project(",
        "_khop",
        "_unwind_to_per_row",
        "_get_all_nodes",
        "get_successors",
        "get_predecessors",
    )
    assert [name for name in retired if name in source] == []


def test_graph_compute_exposes_separate_native_read_and_write_calls() -> None:
    from agent_utilities.knowledge_graph.core.graph_compute import GraphComputeEngine

    read_source = inspect.getsource(GraphComputeEngine.query_cypher)
    write_source = inspect.getsource(GraphComputeEngine.query_cypher_write)
    assert ".query.cypher_read(" in read_source
    assert ".query.cypher_write(" in write_source
    assert "backend" not in read_source
    assert "backend" not in write_source


def test_public_query_modules_do_not_call_backend_or_graphcore() -> None:
    root = Path(__file__).resolve().parents[2] / "agent_utilities"
    public_files = (
        root / "gateway" / "graph_api.py",
        root / "mcp" / "kg_server.py",
        root / "mcp" / "tools" / "query_tools.py",
    )
    offenders: list[str] = []
    for path in public_files:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id == "GraphCore":
                offenders.append(f"{path.name}:{node.lineno}:GraphCore")
            if not isinstance(node, ast.Call) or not isinstance(
                node.func, ast.Attribute
            ):
                continue
            receiver = node.func.value
            if (
                isinstance(receiver, ast.Attribute)
                and receiver.attr == "backend"
                and node.func.attr.startswith("execute")
            ):
                offenders.append(f"{path.name}:{node.lineno}:backend")
    assert offenders == []


# --- literal-inlining helper (pure rendering — unaffected by the read/write/
# batch dispatch split above) ------------------------------------------------


def test_cypher_literal_quotes_and_escapes_strings() -> None:
    assert _cypher_literal("hot") == "'hot'"
    assert _cypher_literal("a'b") == "'a\\'b'"


def test_cypher_literal_renders_bool_and_number() -> None:
    assert _cypher_literal(True) == "true"
    assert _cypher_literal(False) == "false"
    assert _cypher_literal(3) == "3"


def test_cypher_literal_renders_list_for_in_clause() -> None:
    assert _cypher_literal(["a", "b"]) == "['a', 'b']"
