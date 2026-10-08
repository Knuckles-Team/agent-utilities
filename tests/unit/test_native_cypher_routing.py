"""Native Cypher authority contracts (roadmap item 3)."""

from __future__ import annotations

import ast
import inspect
import json
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


@pytest.fixture
def native_error_contract(tmp_path, monkeypatch):
    from agent_utilities.security import error_surface

    (tmp_path / "errors.json").write_text(
        json.dumps({"contract_version": 1, "errors": [{"code": "ACCESS_DENIED"}]}),
        encoding="utf-8",
    )
    monkeypatch.setattr(error_surface, "files", lambda package: tmp_path)
    error_surface._engine_error_codes.cache_clear()
    yield tmp_path
    error_surface._engine_error_codes.cache_clear()


@pytest.mark.parametrize(
    "wire_code", ["ACCESS_DENIED", "UNKNOWN_PRIVATE_CODE", None, 42, ["ACCESS_DENIED"]]
)
def test_native_refusal_retains_only_declared_code(
    native_error_contract, wire_code, caplog
) -> None:
    from agent_utilities.security.error_surface import public_error_payload

    class EngineResponseError(RuntimeError):
        def __init__(self, code):
            self.code = code
            self.detail = "private native diagnostic"
            super().__init__(self.detail)

    graph = MagicMock()
    graph.query_cypher.side_effect = EngineResponseError(wire_code)
    with pytest.raises(CypherEngineError) as caught:
        _backend(graph).execute_read("MATCH (w:WorkItem) RETURN w.id AS id LIMIT 1")
    payload = public_error_payload(caught.value)
    expected = "ACCESS_DENIED" if wire_code == "ACCESS_DENIED" else None
    assert caught.value.engine_error_code == expected
    assert payload.get("engine_error_code") == expected
    assert payload["status"] == "failed"
    assert payload["error"]["code"] == "operation_failed"
    assert payload["error"]["retryable"] is False
    assert caught.value.__cause__ is None
    output = json.dumps(payload) + str(caught.value) + caplog.text
    assert "private native diagnostic" not in output
    assert "UNKNOWN_PRIVATE_CODE" not in output
    assert "MATCH" not in output


@pytest.mark.parametrize(
    "broken_contract, cause",
    [
        ("missing", "No such file or directory"),
        ("malformed", "Expecting value"),
        ("wrong_shape", "not iterable"),
    ],
)
def test_native_refusal_omits_code_without_valid_contract(
    native_error_contract, broken_contract, cause, caplog
) -> None:
    from agent_utilities.security.error_surface import validated_engine_error_code

    contract = native_error_contract / "errors.json"
    if broken_contract == "missing":
        contract.unlink()
    else:
        content = (
            "invalid contract" if broken_contract == "malformed" else '{"errors":null}'
        )
        contract.write_text(content, encoding="utf-8")
    assert validated_engine_error_code("ACCESS_DENIED") is None
    records = [
        record
        for record in caplog.records
        if record.name == "agent_utilities.security.error_surface"
    ]
    assert len(records) == 1
    assert records[0].levelname == "WARNING"
    assert cause in records[0].getMessage()
