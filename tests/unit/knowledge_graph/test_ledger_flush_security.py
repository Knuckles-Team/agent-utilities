"""Fail-closed mutation-ledger replay contracts."""

from __future__ import annotations

import json
from unittest.mock import Mock

import pytest

from agent_utilities.knowledge_graph.core.graph_compute import GraphComputeEngine
from agent_utilities.security.identifiers import CYPHER_IDENTIFIER_RE
from tests.unit.knowledge_graph.test_identifier_validation import MALICIOUS_IDENTIFIERS


def _engine(entries: list[str]) -> GraphComputeEngine:
    engine = object.__new__(GraphComputeEngine)
    engine.get_ledger = Mock(return_value=entries)  # type: ignore[method-assign]
    engine.clear_ledger = Mock()  # type: ignore[method-assign]
    return engine


def test_ledger_replay_rejects_cypher_identifier_injection_without_clearing() -> None:
    # NOTE: the payload key the engine actually reads is "node_type" (see
    # ``GraphComputeEngine.flush_ledger_to_backend``'s
    # ``props.get("node_type", "Entity")``) — this test previously used the
    # wrong key ("type"), which the code silently ignores in favor of the safe
    # "Entity" default, so it never actually exercised the rejection path it
    # claimed to cover (it passed vacuously on every run, main included).
    payload = json.dumps({"node_type": "Entity) DETACH DELETE n //"})
    engine = _engine([f"AddNode|opaque-id|{payload}"])
    backend = Mock()

    with pytest.raises(ValueError, match="unsafe node type"):
        engine.flush_ledger_to_backend(backend)

    backend.execute_write.assert_not_called()
    engine.clear_ledger.assert_not_called()  # type: ignore[attr-defined]


@pytest.mark.parametrize("bad_node_type", MALICIOUS_IDENTIFIERS)
def test_ledger_replay_rejects_malicious_node_type_corpus(bad_node_type) -> None:
    if not bad_node_type:
        pytest.skip("empty node_type falls back to the safe 'Entity' default")
    payload = json.dumps({"node_type": bad_node_type})
    engine = _engine([f"AddNode|opaque-id|{payload}"])
    backend = Mock()

    with pytest.raises(ValueError, match="unsafe node type"):
        engine.flush_ledger_to_backend(backend)

    backend.execute_write.assert_not_called()
    engine.clear_ledger.assert_not_called()  # type: ignore[attr-defined]


@pytest.mark.parametrize("bad_rel_type", MALICIOUS_IDENTIFIERS)
def test_ledger_replay_rejects_malicious_edge_type_corpus(bad_rel_type) -> None:
    if not bad_rel_type:
        pytest.skip("empty relationship falls back to the safe 'RELATED_TO' default")
    if CYPHER_IDENTIFIER_RE.fullmatch(bad_rel_type.replace(" ", "_").upper()):
        pytest.skip(
            "edge types are space-normalized (' '->'_') and upper-cased before "
            "validation, by design (unlike node types) — this corpus entry no "
            "longer looks malicious once normalized the same way the production "
            "code normalizes it"
        )
    payload = json.dumps({"relationship": bad_rel_type})
    engine = _engine([f"AddEdge|src-id|tgt-id|{payload}"])
    backend = Mock()

    with pytest.raises(ValueError, match="unsafe edge type"):
        engine.flush_ledger_to_backend(backend)

    backend.execute_write.assert_not_called()
    engine.clear_ledger.assert_not_called()  # type: ignore[attr-defined]


def test_ledger_replay_preserves_json_delimiters_and_clears_after_success() -> None:
    payload = json.dumps({"type": "Entity", "description": "left|right"})
    engine = _engine([f"AddNode|opaque-id|{payload}"])
    backend = Mock()

    assert engine.flush_ledger_to_backend(backend) == 1

    parameters = backend.execute_write.call_args.kwargs["parameters"]
    assert json.loads(parameters["meta"])["description"] == "left|right"
    engine.clear_ledger.assert_called_once_with()  # type: ignore[attr-defined]


def test_ledger_backend_failure_is_generic_and_preserves_retry_state() -> None:
    payload = json.dumps({"type": "Entity"})
    engine = _engine([f"AddNode|opaque-id|{payload}"])
    backend = Mock()
    backend.execute_write.side_effect = OSError("environment-specific detail")

    with pytest.raises(RuntimeError) as captured:
        engine.flush_ledger_to_backend(backend)

    assert "environment-specific detail" not in str(captured.value)
    assert "OSError" in str(captured.value)
    engine.clear_ledger.assert_not_called()  # type: ignore[attr-defined]


def test_unknown_ledger_operation_is_never_acknowledged_or_discarded() -> None:
    engine = _engine(["UnknownMutation|opaque"])
    backend = Mock()

    with pytest.raises(ValueError, match="unsupported mutation"):
        engine.flush_ledger_to_backend(backend)

    backend.execute_write.assert_not_called()
    engine.clear_ledger.assert_not_called()  # type: ignore[attr-defined]
