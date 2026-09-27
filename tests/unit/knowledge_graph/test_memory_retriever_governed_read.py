"""Self-model reads use AU's verified graph query path."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from agent_utilities.knowledge_graph.retrieval.memory_retriever import MemoryRetriever
from agent_utilities.models.knowledge_graph import MemoryRetrieverNode


def _row(column: str, node_id: str, version: int) -> dict:
    node = MemoryRetrieverNode(id=node_id, name="Self Model", version=version)
    return {column: node.to_graph_properties()}


def test_current_read_uses_governed_query_and_verified_session() -> None:
    session = object()
    query_cypher = MagicMock(return_value=[_row("sm", "sm:2", 2)])
    engine = SimpleNamespace(backend=object(), query_cypher=query_cypher)

    current = MemoryRetriever(engine).get_current(session=session)

    assert current is not None and current.id == "sm:2"
    statement, params = query_cypher.call_args.args
    assert "CURRENT_SELF_MODEL" in statement
    assert params == {"anchor_id": "self:agent-model"}
    assert query_cypher.call_args.kwargs == {"session": session}


def test_predecessor_read_uses_governed_query_and_verified_session() -> None:
    session = object()
    query_cypher = MagicMock(return_value=[_row("prev", "sm:1", 1)])
    engine = SimpleNamespace(backend=object(), query_cypher=query_cypher)

    predecessor = MemoryRetriever(engine)._predecessor_via_supersedes(
        "sm:2", session=session
    )

    assert predecessor is not None and predecessor.id == "sm:1"
    statement, params = query_cypher.call_args.args
    assert "SUPERSEDES" in statement
    assert params == {"node_id": "sm:2"}
    assert query_cypher.call_args.kwargs == {"session": session}


def test_policy_denial_is_not_downgraded_to_local_graph_fallback() -> None:
    engine = SimpleNamespace(
        backend=object(),
        query_cypher=MagicMock(side_effect=PermissionError("denied")),
        graph=MagicMock(),
    )
    with pytest.raises(PermissionError, match="denied"):
        MemoryRetriever(engine).get_current(session=object())
    engine.graph.__contains__.assert_not_called()


def test_temporal_trend_keeps_one_session_across_pointer_reads() -> None:
    session = object()
    first = _row("sm", "sm:2", 2)
    first["sm"]["domain_success_rates"] = {"gitlab": 0.8}
    previous = _row("prev", "sm:1", 1)
    previous["prev"]["domain_success_rates"] = {"gitlab": 0.6}
    query_cypher = MagicMock(side_effect=[[first], [previous], []])
    engine = SimpleNamespace(backend=object(), query_cypher=query_cypher)

    trend = MemoryRetriever(engine).temporal_trend(
        "gitlab", lookback=3, session=session
    )

    assert trend == [0.6, 0.8]
    assert query_cypher.call_count == 3
    assert all(call.kwargs == {"session": session} for call in query_cypher.call_args_list)
