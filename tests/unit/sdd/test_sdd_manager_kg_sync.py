"""Tests for ``SDDManager.record_sdd_outcome`` — CONCEPT:AU-AHE.sdd.spec-driven-development.

D-W2C-3: record_sdd_outcome used to issue a MERGE+SET(function-call-value)
followed by a two-MATCH edge-MERGE, both outside the native engine's write
subset (ONE leading MATCH, MERGE on a single bare node only —
epistemic-graph/crates/eg-query/src/cypher/parser.rs:1184). Fixed to dispatch
through the typed engine API (add_node + link_nodes), matching the pattern
already established for sdd/watcher.py's identical shape.
"""

from unittest.mock import MagicMock, patch

from agent_utilities.models import ProjectConstitution
from agent_utilities.sdd import SDDManager


def _mock_engine():
    engine = MagicMock()
    engine.backend = MagicMock()
    # No Project node resolvable by default; individual tests override.
    engine.backend.execute.return_value = []
    return engine


def test_record_sdd_outcome_writes_a_typed_node_not_raw_cypher():
    mgr = SDDManager(workspace_path=".tmp/does-not-matter")
    model = ProjectConstitution(metadata={"project_name": "demo"})
    engine = _mock_engine()

    with patch(
        "agent_utilities.knowledge_graph.core.engine.IntelligenceGraphEngine.get_active",
        return_value=engine,
    ):
        mgr.record_sdd_outcome(model, feature_id=None)

    assert engine.add_node.call_count == 1
    (node_id,), kwargs = engine.add_node.call_args
    assert node_id == "sdd:ProjectConstitution:Global"
    assert kwargs["node_type"] == "ProjectConstitution"
    assert kwargs["properties"]["name"] == "Global"
    assert isinstance(kwargs["properties"]["last_updated"], int)
    # The project-lookup read is the only raw Cypher left, and it is a
    # single-MATCH read (unrestricted MATCH count) -- not a write.
    engine.backend.execute.assert_called_once()
    read_query = engine.backend.execute.call_args[0][0]
    assert read_query.count("MATCH") == 1
    assert "MERGE" not in read_query


def test_record_sdd_outcome_links_to_project_when_one_exists():
    mgr = SDDManager(workspace_path=".tmp/does-not-matter")
    model = ProjectConstitution(metadata={"project_name": "demo"})
    engine = _mock_engine()
    engine.backend.execute.return_value = [{"id": "proj:current"}]

    with patch(
        "agent_utilities.knowledge_graph.core.engine.IntelligenceGraphEngine.get_active",
        return_value=engine,
    ):
        mgr.record_sdd_outcome(model, feature_id=None)

    engine.link_nodes.assert_called_once_with(
        "proj:current", "sdd:ProjectConstitution:Global", "HAS_ARTIFACT"
    )


def test_record_sdd_outcome_is_best_effort_on_engine_failure():
    mgr = SDDManager(workspace_path=".tmp/does-not-matter")
    model = ProjectConstitution(metadata={"project_name": "demo"})
    engine = _mock_engine()
    engine.add_node.side_effect = RuntimeError("engine unavailable")

    with patch(
        "agent_utilities.knowledge_graph.core.engine.IntelligenceGraphEngine.get_active",
        return_value=engine,
    ):
        # Must not raise -- KG sync is best-effort, never blocks the save().
        mgr.record_sdd_outcome(model, feature_id=None)
