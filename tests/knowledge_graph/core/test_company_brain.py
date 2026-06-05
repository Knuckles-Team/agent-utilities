"""CONCEPT:KG-2.6

Unit tests for CompanyBrain pre-commit ontological consistency verification.
"""

import json
from pathlib import Path
import pytest

# The compiled Rust core (epistemic_graph._epistemic_graph) must be built for these tests;
# skip the whole module cleanly when it isn't, rather than erroring out collection.
EpistemicGraph = pytest.importorskip(
    "epistemic_graph._epistemic_graph"
).EpistemicGraph
from agent_utilities.knowledge_graph.core.company_brain import CompanyBrain


def test_company_brain_initialization():
    brain = CompanyBrain()
    status = brain.status()
    assert "concurrency" in status
    assert "provenance" in status
    assert "permissions" in status


def test_pre_commit_validate_conforming_node():
    brain = CompanyBrain()
    base_graph = EpistemicGraph()

    # Define a conforming Agent node (AgentShape requires name)
    proposed_node = (
        "agent:test-runner",
        {"type": "Agent", "name": "Test Runner Agent"},
    )

    report = brain.pre_commit_validate(
        base_graph=base_graph,
        proposed_node=proposed_node,
    )

    assert report["conforms"] is True
    assert len(report["violations"]) == 0


def test_pre_commit_validate_violating_node():
    brain = CompanyBrain()
    base_graph = EpistemicGraph()

    # Define a violating ADR node (ADRShape requires context, decision, authority)
    proposed_node = (
        "adr:001",
        {
            "type": "ArchitectureDecisionRecord",
            "title": "Use Rust Datalog",
            # Missing context, decision, authority
        },
    )

    report = brain.pre_commit_validate(
        base_graph=base_graph,
        proposed_node=proposed_node,
    )

    # SHACL validator should catch the violations
    assert report["conforms"] is False
    assert len(report["violations"]) > 0

    messages = [v["message"] for v in report["violations"]]
    assert any("context" in msg for msg in messages)
    assert any("decision" in msg for msg in messages)
    assert any("authority" in msg for msg in messages)
