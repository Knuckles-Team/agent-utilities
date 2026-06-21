"""compare_and_set_node_fields plumbing through the backend stack (CONCEPT:KG-2.141).

The engine ships one atomic ``nodes.compare_and_set``; we surface it as
``compare_and_set_node_fields`` at each tier so the :Task claim can use it
backend-agnostically:

* ``GraphComputeEngine`` → ``self._client.nodes.compare_and_set`` (mirrors add_node)
* ``EpistemicGraphBackend`` (L1) → delegates to the GraphComputeEngine
* ``TieredGraphBackend`` → runs on L1 (authoritative, returns its bool) and
  mirrors the won state to L3 best-effort; a lost CAS leaves L3 untouched.

The engine/loop is mocked — no live daemon required.
"""

from unittest.mock import MagicMock

from agent_utilities.knowledge_graph.backends.tiered_backend import (
    TieredGraphBackend,
)
from agent_utilities.knowledge_graph.core.graph_compute import GraphComputeEngine


def test_graph_compute_engine_delegates_to_client_nodes():
    eng = GraphComputeEngine.__new__(GraphComputeEngine)  # skip connect()
    eng._client = MagicMock()
    eng._client.nodes.compare_and_set.return_value = True

    won = eng.compare_and_set_node_fields(
        "n1", {"status": "pending"}, {"status": "running"}
    )

    assert won is True
    eng._client.nodes.compare_and_set.assert_called_once_with(
        "n1", {"status": "pending"}, {"status": "running"}
    )


def test_graph_compute_engine_coerces_to_bool():
    eng = GraphComputeEngine.__new__(GraphComputeEngine)
    eng._client = MagicMock()
    eng._client.nodes.compare_and_set.return_value = 0  # falsy non-bool

    assert eng.compare_and_set_node_fields("n1", {}, {"x": 1}) is False


def test_l1_backend_delegates_to_compute_engine():
    from agent_utilities.knowledge_graph.backends.epistemic_graph_backend import (
        EpistemicGraphBackend,
    )

    b = EpistemicGraphBackend.__new__(EpistemicGraphBackend)  # skip connect()
    b._graph = MagicMock()
    b._graph.compare_and_set_node_fields.return_value = True

    won = b.compare_and_set_node_fields(
        "n1", {"status": "pending"}, {"status": "running"}
    )

    assert won is True
    b._graph.compare_and_set_node_fields.assert_called_once_with(
        "n1", {"status": "pending"}, {"status": "running"}
    )


def _tiered_with_mocks(l1_result: bool) -> TieredGraphBackend:
    t = TieredGraphBackend.__new__(TieredGraphBackend)  # skip __init__/daemons
    t.l1 = MagicMock()
    t.l3 = MagicMock()
    t.l1.compare_and_set_node_fields.return_value = l1_result
    # _mirror runs the L3 closure inline (no write-behind queue configured).
    t._write_behind = False
    t._wb_queue = None
    t._l3_writes = 0
    t._l3_failures = 0
    t._wb_inline = 0
    return t


def test_tiered_runs_on_l1_and_mirrors_to_l3_when_won():
    t = _tiered_with_mocks(l1_result=True)

    won = t.compare_and_set_node_fields(
        "job-1", {"status": "pending"}, {"status": "running", "metadata": "enc"}
    )

    assert won is True
    t.l1.compare_and_set_node_fields.assert_called_once_with(
        "job-1", {"status": "pending"}, {"status": "running", "metadata": "enc"}
    )
    # L3 mirror applied the won updates keyed on the node id.
    t.l3.execute.assert_called_once()
    cypher, params = t.l3.execute.call_args.args
    assert "SET" in cypher
    assert params["_casid"] == "job-1"
    assert params["status"] == "running"
    assert params["metadata"] == "enc"


def test_tiered_does_not_mirror_when_cas_lost():
    t = _tiered_with_mocks(l1_result=False)

    won = t.compare_and_set_node_fields(
        "job-1", {"status": "pending"}, {"status": "running"}
    )

    assert won is False
    t.l3.execute.assert_not_called()  # loser leaves L3 untouched


def test_tiered_mirror_never_raises():
    t = _tiered_with_mocks(l1_result=True)
    t.l3.execute.side_effect = RuntimeError("L3 down")

    # Best-effort: a failing L3 mirror must not break the (won) claim.
    won = t.compare_and_set_node_fields(
        "job-1", {"status": "pending"}, {"status": "running"}
    )

    assert won is True
