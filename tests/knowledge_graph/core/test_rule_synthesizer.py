"""CONCEPT:AU-KG.research.research-pipeline-runner

Unit tests for RuleSynthesizerDaemon rule compilation and conflict resolution.
"""

from unittest.mock import MagicMock

import pytest

from agent_utilities.knowledge_graph.core.rule_synthesizer import RuleSynthesizerDaemon


@pytest.mark.asyncio
async def test_rule_synthesizer_daemon_resolved_conflicts():
    # 1. Setup mock engine
    engine = MagicMock()

    # Mock return values for query_cypher
    mock_conflict = {
        "id": "conflict_123",
        "resolved_by": "analyst:alice",
        "resolved_value": "high",
        "losing_source_system": "legacy_crm",
    }
    mock_node = {"id": "customer_456"}

    queries = []
    params_list = []

    def mock_query_cypher(cypher, params=None):
        queries.append(cypher)
        params_list.append(params)
        if "ConflictRecord" in cypher and "AFFECTS" in cypher:
            return [{"c": mock_conflict, "n": mock_node}]
        elif "TrustHierarchyEntry" in cypher and "SET" not in cypher:
            return [
                {
                    "t": {
                        "source_system": "legacy_crm",
                        "authority_level": 0.8,
                        "conflict_penalty": 0.1,
                    }
                }
            ]
        return []

    engine.query_cypher.side_effect = mock_query_cypher

    # 2. Instantiate and run daemon processing
    daemon = RuleSynthesizerDaemon(engine)
    await daemon.process_resolved_conflicts()

    # 3. Assertions
    assert len(queries) >= 3

    # Check rule synthesis query
    synthesis_query = [q for q in queries if "RuleNode" in q]
    assert len(synthesis_query) == 1
    assert "MATCH (c:ConflictRecord {id: $conflict_id})" in synthesis_query[0]
    assert "MATCH (n {id: $node_id})" in synthesis_query[0]
    assert "MERGE (r:RuleNode {id: $rule_id})" in synthesis_query[0]

    # Check rule synthesis parameters
    synthesis_params = [p for p in params_list if p and "conflict_id" in p]
    assert len(synthesis_params) == 1
    assert synthesis_params[0]["conflict_id"] == "conflict_123"
    assert synthesis_params[0]["node_id"] == "customer_456"
    assert synthesis_params[0]["rule_id"] == "rule_conflict_123"
    assert "analyst:alice" in synthesis_params[0]["desc"]
    assert "high" in synthesis_params[0]["desc"]
    assert synthesis_params[0]["human"] == "analyst:alice"

    # Check trust penalty update query
    penalty_query = [q for q in queries if "new_auth" in q]
    assert len(penalty_query) == 1
    penalty_params = [p for p in params_list if p and "new_auth" in p]
    assert len(penalty_params) == 1
    assert penalty_params[0]["source"] == "legacy_crm"
    assert pytest.approx(penalty_params[0]["new_auth"]) == 0.7
