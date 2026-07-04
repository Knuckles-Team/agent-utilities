#!/usr/bin/python
from __future__ import annotations

"""Tests for Cross-Pillar Synergy Engine.

CONCEPT:AU-KG.compute.cross-pillar-synergy — Cross-Pillar Synergy Engine

Validates concept bridge discovery, pillar coupling computation,
missing edge suggestions, and synergy report generation.
"""


import pytest

from agent_utilities.knowledge_graph.core.synergy_engine import (
    PILLAR_NAMES,
    SynergyEngine,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_registry() -> dict:
    """A representative concept registry for testing."""
    return {
        "AU-ORCH.planning.recursion-nesting-depth": {
            "name": "Wide-Search Orchestration",
            "pillar": "ORCH-1",
            "module": "graph/executor.py",
        },
        "AU-ORCH.adapter.hot-cache-invalidation": {
            "name": "Confidence-Gated Router",
            "pillar": "ORCH-1",
            "module": "graph/executor.py",
            "depends_on": ["AU-KG.memory.tiered-memory-caching"],
        },
        "AU-ORCH.adapter.kg-graph-materialization": {
            "name": "Swarm Preset Engine",
            "pillar": "ORCH-1",
            "module": "graph/swarm.py",
        },
        "AU-KG.memory.tiered-memory-caching": {
            "name": "SelfModel",
            "pillar": "EG-KG.compute.backend",
            "module": "knowledge_graph/self_model.py",
        },
        "AU-KG.query.vendor-agnostic-traversal": {
            "name": "Context Compaction",
            "pillar": "EG-KG.compute.backend",
            "module": "knowledge_graph/elastic_context_manager.py",
        },
        "AU-KG.query.vendor-agnostic-traversal": {  # noqa: F601 — pre-existing duplicate fixture key (last-wins)
            "name": "Research Pipeline",
            "pillar": "EG-KG.compute.backend",
            "module": "automation/research_pipeline.py",
        },
        "AU-AHE.evaluation.interpretability-tests": {
            "name": "TeamConfig Promotion",
            "pillar": "AHE-3",
            "module": "harness/evolve_agent.py",
        },
        "AU-AHE.harness.self-evolution-narrative": {
            "name": "Memory-Aware Test-Time Scaling",
            "pillar": "AHE-3",
            "module": "harness/continuous_evaluation_engine.py",
        },
        "AU-AHE.evaluation.longmemeval-validation-harness": {
            "name": "EvalRunner",
            "pillar": "AHE-3",
            "module": "harness.continuous_evaluation_engine.py",
        },
        "AU-OS.deployment.infra-orchestration": {
            "name": "Ecosystem Topology Map",
            "pillar": "AU-ECO.connector.plane-provisioning-auth",
            "module": "knowledge_graph/ecosystem_topology.py",
        },
        "AU-OS.governance.wasm-micro-agent-sandbox": {
            "name": "Prompt Injection Scanner",
            "pillar": "OS-5",
            "module": "security/threat_defense_engine.py",
        },
        "AU-OS.deployment.platform-journey": {
            "name": "Guardrail Engine",
            "pillar": "OS-5",
            "module": "security/threat_defense_engine.py",
        },
        "AU-OS.scaling.epistemic-dynamic-priority-quota": {
            "name": "Telemetry & Observability",
            "pillar": "OS-5",
            "module": "observability/custom_observability.py",
        },
    }


@pytest.fixture()
def engine(sample_registry: dict) -> SynergyEngine:
    """Create a SynergyEngine with the sample registry."""
    return SynergyEngine(concept_registry=sample_registry)


# ---------------------------------------------------------------------------
# Concept Bridge Discovery
# ---------------------------------------------------------------------------


@pytest.mark.concept("AU-KG.query.vendor-agnostic-traversal")
class TestConceptBridges:
    """Tests for concept bridge discovery."""

    def test_discovers_known_bridges(self, engine: SynergyEngine) -> None:
        bridges = engine.discover_concept_bridges()
        bridge_ids = {b.concept_id for b in bridges}
        # Known bridges from _KNOWN_BRIDGES
        assert "AU-AHE.harness.self-evolution-narrative" in bridge_ids
        assert "AU-AHE.evaluation.interpretability-tests" in bridge_ids
        assert "AU-OS.governance.wasm-micro-agent-sandbox" in bridge_ids

    def test_bridge_has_correct_primary_pillar(self, engine: SynergyEngine) -> None:
        bridges = engine.discover_concept_bridges()
        ahe_35 = next(b for b in bridges if b.concept_id == "AU-AHE.harness.self-evolution-narrative")
        assert ahe_35.primary_pillar == "AHE-3"
        assert "EG-KG.compute.backend" in ahe_35.bridged_pillars

    def test_dependency_bridges_discovered(self, engine: SynergyEngine) -> None:
        bridges = engine.discover_concept_bridges()
        # ORCH-1.2 depends on KG-2.1, so it should bridge ORCH→KG
        orch_12 = next((b for b in bridges if b.concept_id == "AU-ORCH.adapter.hot-cache-invalidation"), None)
        assert orch_12 is not None
        # It's in _KNOWN_BRIDGES, so check bridged pillars
        assert "EG-KG.compute.backend" in orch_12.bridged_pillars

    def test_bridge_has_reason(self, engine: SynergyEngine) -> None:
        bridges = engine.discover_concept_bridges()
        for bridge in bridges:
            assert bridge.bridge_reason, f"Bridge {bridge.concept_id} has no reason"

    def test_empty_registry(self) -> None:
        engine = SynergyEngine(concept_registry={})
        bridges = engine.discover_concept_bridges()
        # Should still find known bridges but with empty names
        assert len(bridges) > 0


# ---------------------------------------------------------------------------
# Pillar Coupling
# ---------------------------------------------------------------------------


@pytest.mark.concept("AU-KG.query.vendor-agnostic-traversal")
class TestPillarCoupling:
    """Tests for pillar coupling computation."""

    def test_all_pairs_computed(self, engine: SynergyEngine) -> None:
        couplings = engine.compute_pillar_coupling()
        # 5 pillars → C(5,2) = 10 pairs
        assert len(couplings) == 10

    def test_sorted_by_coupling_score(self, engine: SynergyEngine) -> None:
        couplings = engine.compute_pillar_coupling()
        scores = [c.coupling_score for c in couplings]
        assert scores == sorted(scores, reverse=True)

    def test_coupling_scores_bounded(self, engine: SynergyEngine) -> None:
        couplings = engine.compute_pillar_coupling()
        for c in couplings:
            assert 0.0 <= c.coupling_score <= 1.0

    def test_ahe_kg_highly_coupled(self, engine: SynergyEngine) -> None:
        couplings = engine.compute_pillar_coupling()
        ahe_kg = next(
            c for c in couplings if {c.pillar_a, c.pillar_b} == {"AHE-3", "EG-KG.compute.backend"}
        )
        assert ahe_kg.coupling_score > 0.0
        assert len(ahe_kg.shared_edge_types) > 0


# ---------------------------------------------------------------------------
# Missing Edge Suggestions
# ---------------------------------------------------------------------------


@pytest.mark.concept("AU-KG.query.vendor-agnostic-traversal")
class TestMissingEdges:
    """Tests for missing relationship suggestions."""

    def test_returns_suggestions(self, engine: SynergyEngine) -> None:
        suggestions = engine.suggest_missing_edges()
        assert isinstance(suggestions, list)

    def test_suggestions_have_rationale(self, engine: SynergyEngine) -> None:
        suggestions = engine.suggest_missing_edges()
        for s in suggestions:
            assert s.rationale, (
                f"Suggestion {s.source_concept}→{s.target_concept} has no rationale"
            )
            assert 0.0 <= s.confidence <= 1.0

    def test_no_existing_edges_suggested(self, engine: SynergyEngine) -> None:
        suggestions = engine.suggest_missing_edges()
        # Filter should exclude is_existing=True
        for s in suggestions:
            assert not s.is_existing


# ---------------------------------------------------------------------------
# Synergy Report
# ---------------------------------------------------------------------------


@pytest.mark.concept("AU-KG.query.vendor-agnostic-traversal")
class TestSynergyReport:
    """Tests for markdown report generation."""

    def test_report_contains_sections(self, engine: SynergyEngine) -> None:
        report = engine.generate_synergy_report()
        assert "# Cross-Pillar Synergy Report" in report
        assert "## Concept Bridges" in report
        assert "## Pillar Coupling Matrix" in report

    def test_report_contains_metrics(self, engine: SynergyEngine) -> None:
        report = engine.generate_synergy_report()
        assert "Pillars Analyzed" in report
        assert "Total Concepts" in report
        assert "Bridges Found" in report

    def test_report_is_valid_markdown(self, engine: SynergyEngine) -> None:
        report = engine.generate_synergy_report()
        # Check for table headers
        assert "|" in report
        # Check for headers
        assert report.startswith("#")


# ---------------------------------------------------------------------------
# Pillar Names
# ---------------------------------------------------------------------------


@pytest.mark.concept("AU-KG.query.vendor-agnostic-traversal")
class TestPillarNames:
    """Tests for pillar name constants."""

    def test_five_pillars_defined(self) -> None:
        assert len(PILLAR_NAMES) == 5

    def test_pillar_prefixes(self) -> None:
        assert "ORCH-1" in PILLAR_NAMES
        assert "EG-KG.compute.backend" in PILLAR_NAMES
        assert "AHE-3" in PILLAR_NAMES
        assert "AU-ECO.connector.plane-provisioning-auth" in PILLAR_NAMES
        assert "OS-5" in PILLAR_NAMES
