"""The AU latent caller binds EG reads to the verified session."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from agent_utilities.knowledge_graph.retrieval.latent_topology_rag import (
    LatentTopologicalRAG,
)


def test_uses_governed_query_with_same_session_for_each_relation() -> None:
    session = object()
    query_cypher = MagicMock(
        return_value=[{"n": {"id": "n1", "importance_score": 0.8}}]
    )
    engine = SimpleNamespace(backend=object(), query_cypher=query_cypher)

    rows = LatentTopologicalRAG(engine).retrieve("query", top_k=3, session=session)

    assert rows == [{"id": "n1", "name": "", "score": 0.8}]
    assert query_cypher.call_count == 2
    assert all(call.kwargs == {"session": session} for call in query_cypher.call_args_list)
    assert all("LIMIT 3" in call.args[0] for call in query_cypher.call_args_list)


def test_authorization_failure_is_not_downgraded_to_empty_result() -> None:
    engine = SimpleNamespace(
        backend=object(), query_cypher=MagicMock(side_effect=PermissionError("denied"))
    )
    with pytest.raises(PermissionError, match="denied"):
        LatentTopologicalRAG(engine).retrieve("query")
