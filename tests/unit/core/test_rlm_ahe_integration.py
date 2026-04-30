"""Tests for RLM × AHE integration.

CONCEPT:AU-007 × AU-012 — RLM for AHE Evolution

Tests that the TraceDistiller and EvolveAgent correctly trigger RLM
for large trace sets and evidence corpora.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_utilities.rlm.config import RLMConfig


class TestTraceDistillerRLMIntegration:
    """Verify TraceDistiller routes to RLM for large trace sets."""

    @pytest.mark.asyncio
    async def test_cluster_failures_uses_rlm_above_threshold(self):
        """When failure count exceeds ahe_trace_threshold, RLM is used."""
        from agent_utilities.harness.evidence_corpus import EvidenceEntry, EvidenceLayer
        from agent_utilities.harness.trace_backend import TraceBackend
        from agent_utilities.harness.trace_distiller import TraceDistiller

        mock_backend = MagicMock(spec=TraceBackend)
        distiller = TraceDistiller(backend=mock_backend)

        # Create 600 failures (above default 500 threshold)
        failures = [
            EvidenceEntry(
                task_id=f"task_{i}",
                pass_fail=False,
                root_cause=f"Error type {i % 5}",
                score=0.2,
                evidence_layer=EvidenceLayer.PER_TASK_REPORT,
                content=f"Task {i} failed",
            )
            for i in range(600)
        ]

        # Mock the RLM to return structured clusters
        mock_clusters = [
            {
                "label": "error_type_0",
                "root_cause_summary": "Type 0 errors",
                "task_ids": [f"task_{i}" for i in range(0, 600, 5)],
                "component": "tool_implementation",
                "frequency": 120,
                "severity": 0.8,
            }
        ]

        with patch(
            "agent_utilities.rlm.repl.RLMEnvironment"
        ) as MockEnv:
            mock_env_instance = MagicMock()
            mock_env_instance.run_full_rlm = AsyncMock(
                return_value=json.dumps(mock_clusters)
            )
            MockEnv.return_value = mock_env_instance

            clusters = await distiller._cluster_failures(failures)

            assert len(clusters) >= 1
            MockEnv.assert_called_once()
            mock_env_instance.run_full_rlm.assert_called_once()

    @pytest.mark.asyncio
    async def test_cluster_failures_keyword_below_threshold(self):
        """When failure count is below threshold, keyword clustering is used."""
        from agent_utilities.harness.evidence_corpus import EvidenceEntry, EvidenceLayer
        from agent_utilities.harness.trace_backend import TraceBackend
        from agent_utilities.harness.trace_distiller import TraceDistiller

        mock_backend = MagicMock(spec=TraceBackend)
        distiller = TraceDistiller(backend=mock_backend)

        # Create 10 failures (below 500 threshold)
        failures = [
            EvidenceEntry(
                task_id=f"task_{i}",
                pass_fail=False,
                root_cause="timeout error",
                score=0.1,
                evidence_layer=EvidenceLayer.PER_TASK_REPORT,
                content=f"Task {i} failed",
            )
            for i in range(10)
        ]

        # Should use keyword clustering, not RLM
        with patch(
            "agent_utilities.rlm.config.RLMConfig"
        ) as MockConfig:
            mock_config = RLMConfig(enabled=False, ahe_trace_threshold=500)
            MockConfig.return_value = mock_config

            clusters = await distiller._cluster_failures(failures)

            # Keyword clustering should group all 10 by the same root cause
            assert len(clusters) >= 1


class TestEvolveAgentRLMIntegration:
    """Verify EvolveAgent uses RLM for deep evidence analysis."""

    @pytest.mark.asyncio
    async def test_deep_analyze_evidence_triggers_on_large_corpus(self):
        """When evidence JSON exceeds threshold, RLM is invoked."""
        from agent_utilities.harness.evidence_corpus import EvidenceCorpus
        from agent_utilities.harness.evolve_agent import EvolveAgent

        agent = EvolveAgent(workspace_path="/tmp/test")

        # Create large evidence corpus
        evidence = EvidenceCorpus(
            round_id="test_round",
            overview="x" * 60_000,  # Exceed 50K threshold
            total_tasks=1000,
        )

        mock_edits = [
            {
                "component_type": "system_prompt",
                "file_path": "prompts/router.md",
                "edit_summary": "Improve routing instructions",
                "predicted_fixes": ["task_1", "task_2"],
                "predicted_regressions": [],
            }
        ]

        with patch(
            "agent_utilities.rlm.repl.RLMEnvironment"
        ) as MockEnv:
            mock_env_instance = MagicMock()
            mock_env_instance.run_full_rlm = AsyncMock(
                return_value=json.dumps(mock_edits)
            )
            MockEnv.return_value = mock_env_instance

            edits = await agent._deep_analyze_evidence(evidence)

            assert len(edits) == 1
            assert edits[0].edit_summary == "Improve routing instructions"
            MockEnv.assert_called_once()

    @pytest.mark.asyncio
    async def test_deep_analyze_evidence_skips_small_corpus(self):
        """When evidence is small, returns empty list (no RLM)."""
        from agent_utilities.harness.evidence_corpus import EvidenceCorpus
        from agent_utilities.harness.evolve_agent import EvolveAgent

        agent = EvolveAgent(workspace_path="/tmp/test")

        evidence = EvidenceCorpus(
            round_id="small_round",
            overview="Small overview",
            total_tasks=5,
        )

        edits = await agent._deep_analyze_evidence(evidence)
        assert edits == []


class TestRLMTriggerThresholds:
    """Test that RLM trigger thresholds are configurable."""

    def test_ahe_threshold_configurable(self):
        config = RLMConfig(ahe_trace_threshold=100)
        assert config.should_trigger(trace_count=50) is False
        assert config.should_trigger(trace_count=150) is True

    def test_kg_threshold_configurable(self):
        config = RLMConfig(kg_bulk_threshold=200)
        assert config.should_trigger(kg_node_count=100) is False
        assert config.should_trigger(kg_node_count=300) is True
