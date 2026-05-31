#!/usr/bin/python
"""Tests for Comparative Analysis Integration Modules.

Tests for CONCEPT:KG-2.6 through KG-2.37 and OS-5.18:
- ResearchSubagent (KG-2.33)
- SpectralClusterNavigator (KG-2.34)
- BlastRadiusAnalyzer (KG-2.35)
- AutoSimilarityLinker (KG-2.36)
- HybridSearchScorer (KG-2.37)
- DoomLoopDetector (OS-5.18)
"""

import time
import uuid

import numpy as np

# ── KG-2.34: Spectral Cluster Navigator ──────────────────────────────


class TestSpectralClusterNavigator:
    """Tests for CONCEPT:KG-2.5 — Spectral Cluster Navigator."""

    def test_two_clear_clusters(self):
        """Two well-separated clusters should be discovered."""
        from agent_utilities.knowledge_graph.core.spectral_navigator import (
            SpectralClusterNavigator,
        )

        nav = SpectralClusterNavigator()
        np.random.seed(42)
        c1 = np.random.randn(10, 5) + np.array([5, 0, 0, 0, 0])
        c2 = np.random.randn(10, 5) + np.array([0, 5, 0, 0, 0])
        vectors = np.vstack([c1, c2]).tolist()

        clusters = nav.cluster(vectors, max_k=5)
        assert len(clusters) == 2
        assert len(clusters[0].indices) == 10
        assert len(clusters[1].indices) == 10

    def test_singleton_input(self):
        """A single vector should produce a single cluster."""
        from agent_utilities.knowledge_graph.core.spectral_navigator import (
            SpectralClusterNavigator,
        )

        nav = SpectralClusterNavigator()
        clusters = nav.cluster([[1.0, 2.0, 3.0]])
        assert len(clusters) == 1
        assert len(clusters[0].indices) == 1

    def test_cluster_coherence_range(self):
        """Coherence scores should be in [0, 1]."""
        from agent_utilities.knowledge_graph.core.spectral_navigator import (
            SpectralClusterNavigator,
        )

        nav = SpectralClusterNavigator()
        np.random.seed(42)
        vectors = np.random.randn(20, 4).tolist()
        clusters = nav.cluster(vectors, max_k=4)
        for c in clusters:
            assert 0.0 <= c.coherence <= 1.0

    def test_kg_node_conversion(self):
        """Clusters should convert to valid SpectralClusterNode models."""
        from agent_utilities.knowledge_graph.core.spectral_navigator import (
            SpectralClusterNavigator,
        )
        from agent_utilities.models.knowledge_graph import RegistryNodeType

        nav = SpectralClusterNavigator()
        np.random.seed(42)
        vectors = (np.random.randn(20, 3) + np.array([5, 0, 0])).tolist()
        clusters = nav.cluster(vectors, max_k=3)

        nodes = nav.cluster_to_kg_nodes(clusters, domain="research")
        assert len(nodes) > 0
        for node in nodes:
            assert node.type == RegistryNodeType.SPECTRAL_CLUSTER
            assert node.member_count > 0

    def test_financial_regime_detection(self):
        """Financial regime detection should discover distinct regimes."""
        from agent_utilities.knowledge_graph.core.spectral_navigator import (
            SpectralClusterNavigator,
        )

        nav = SpectralClusterNavigator()
        np.random.seed(42)
        bull = np.random.randn(15, 3) + np.array([3, 0, 0])
        bear = np.random.randn(15, 3) + np.array([0, 3, 0])
        data = np.vstack([bull, bear]).tolist()

        regimes = nav.detect_financial_regimes(data, max_regimes=3)
        assert len(regimes) >= 2
        assert any("financial" in r.label for r in regimes)


# ── KG-2.35: Blast Radius Analyzer ──────────────────────────────────


class TestBlastRadiusAnalyzer:
    """Tests for CONCEPT:KG-2.5 — Symbol Blast Radius Analyzer."""

    def test_analyze_known_symbol(self):
        """RegistryNode should have many usages."""
        from agent_utilities.knowledge_graph.core.blast_radius import (
            BlastRadiusAnalyzer,
        )

        analyzer = BlastRadiusAnalyzer(".")
        result = analyzer.analyze(
            "RegistryNode",
            definition_file="agent_utilities/models/knowledge_graph.py",
        )
        assert result.usage_count > 10
        assert result.file_count > 3
        assert not result.is_low_usage
        assert result.impact_score > 0.0

    def test_analyze_nonexistent_symbol(self):
        """A nonexistent symbol should have zero or near-zero usages."""
        from agent_utilities.knowledge_graph.core.blast_radius import (
            BlastRadiusAnalyzer,
        )

        # Use uuid to ensure the symbol doesn't appear anywhere
        unique = uuid.uuid4().hex
        analyzer = BlastRadiusAnalyzer(".")
        result = analyzer.analyze(f"zz_{unique}")
        assert result.usage_count == 0
        assert result.file_count == 0
        assert result.is_low_usage
        assert result.impact_score == 0.0

    def test_blast_radius_node_type(self):
        """Result should be a valid BlastRadiusNode."""
        from agent_utilities.knowledge_graph.core.blast_radius import (
            BlastRadiusAnalyzer,
        )
        from agent_utilities.models.knowledge_graph import RegistryNodeType

        analyzer = BlastRadiusAnalyzer(".")
        result = analyzer.analyze("BaseModel")
        assert result.type == RegistryNodeType.BLAST_RADIUS_REPORT
        assert result.symbol_name == "BaseModel"


# ── OS-5.18: Enhanced Doom-Loop Detector ─────────────────────────────


class TestDoomLoopDetector:
    """Tests for CONCEPT:OS-5.0 — Enhanced Doom-Loop Detector."""

    def test_consecutive_detection(self):
        """Three identical calls should trigger detection."""
        from agent_utilities.security.execution_stability_engine import DoomLoopDetector

        det = DoomLoopDetector(consecutive_threshold=3)
        det.record_call("shell", {"cmd": "ls"}, "file1")
        det.record_call("shell", {"cmd": "ls"}, "file1")
        assert det.check() is None

        det.record_call("shell", {"cmd": "ls"}, "file1")
        incident = det.check()
        assert incident is not None
        assert incident.pattern_type == "consecutive"

    def test_different_results_no_loop(self):
        """Same args but different results should NOT trigger."""
        from agent_utilities.security.execution_stability_engine import DoomLoopDetector

        det = DoomLoopDetector(consecutive_threshold=3)
        det.record_call("poll", {"id": "1"}, "pending")
        det.record_call("poll", {"id": "1"}, "pending")
        det.record_call("poll", {"id": "1"}, "complete")  # Different result
        incident = det.check()
        assert incident is None

    def test_sequence_detection(self):
        """Repeating [A,B,A,B] should trigger sequence detection."""
        from agent_utilities.security.execution_stability_engine import DoomLoopDetector

        det = DoomLoopDetector(
            consecutive_threshold=5
        )  # High threshold to skip consecutive
        for _ in range(3):
            det.record_call("search", {"q": "test"}, "res")
            det.record_call("read", {"url": "a.com"}, "page")

        incident = det.check()
        assert incident is not None
        assert incident.pattern_type == "sequence"

    def test_corrective_prompt_generated(self):
        """Detected loops should produce corrective prompts."""
        from agent_utilities.security.execution_stability_engine import DoomLoopDetector

        det = DoomLoopDetector(consecutive_threshold=2)
        det.record_call("x", {}, "r")
        det.record_call("x", {}, "r")
        incident = det.check()
        assert incident is not None
        assert "DOOM-LOOP GUARD" in incident.corrective_prompt

    def test_reset(self):
        """Reset should clear all signatures."""
        from agent_utilities.security.execution_stability_engine import DoomLoopDetector

        det = DoomLoopDetector()
        det.record_call("a", {}, "r")
        det.record_call("a", {}, "r")
        assert det.signature_count == 2
        det.reset()
        assert det.signature_count == 0


# ── KG-2.36: Auto-Similarity Memory Graph ───────────────────────────


class TestAutoSimilarityLinker:
    """Tests for CONCEPT:KG-2.3 — Auto-Similarity Memory Graph."""

    def test_similar_nodes_linked(self):
        """Nodes above threshold should be linked."""
        from agent_utilities.knowledge_graph.memory import (
            AutoSimilarityLinker,
        )
        from agent_utilities.models.knowledge_graph import (
            MemoryDecayConfig,
            RegistryNode,
            RegistryNodeType,
        )

        config = MemoryDecayConfig(similarity_threshold=0.5)
        linker = AutoSimilarityLinker(config=config)

        n1 = RegistryNode(
            id="n1", type=RegistryNodeType.MEMORY, name="t1", embedding=[1.0, 0.0, 0.0]
        )
        n2 = RegistryNode(
            id="n2", type=RegistryNodeType.MEMORY, name="t2", embedding=[0.9, 0.1, 0.0]
        )

        edges = linker.link_new_node(n2, [n1])
        assert len(edges) == 1
        assert edges[0].cosine_similarity > 0.5

    def test_dissimilar_nodes_not_linked(self):
        """Orthogonal nodes should NOT be linked."""
        from agent_utilities.knowledge_graph.memory import (
            AutoSimilarityLinker,
        )
        from agent_utilities.models.knowledge_graph import (
            RegistryNode,
            RegistryNodeType,
        )

        linker = AutoSimilarityLinker()  # Default threshold=0.72
        n1 = RegistryNode(
            id="n1", type=RegistryNodeType.MEMORY, name="t1", embedding=[1.0, 0.0, 0.0]
        )
        n2 = RegistryNode(
            id="n2", type=RegistryNodeType.MEMORY, name="t2", embedding=[0.0, 1.0, 0.0]
        )

        edges = linker.link_new_node(n2, [n1])
        assert len(edges) == 0

    def test_decay_reduces_weight(self):
        """Weights should decay over time."""
        from agent_utilities.knowledge_graph.memory import (
            AutoSimilarityLinker,
        )
        from agent_utilities.models.knowledge_graph import SimilarityEdgeNode

        linker = AutoSimilarityLinker()
        edge = SimilarityEdgeNode(
            id="e1",
            name="test",
            cosine_similarity=0.9,
            decay_lambda=0.01,
            creation_epoch=time.time() - 86400 * 100,  # 100 days ago
        )

        weight = linker.decay_weight(edge)
        assert weight < 0.9
        assert weight > 0.0

    def test_hub_control(self):
        """Max edges per node should be enforced."""
        from agent_utilities.knowledge_graph.memory import (
            AutoSimilarityLinker,
        )
        from agent_utilities.models.knowledge_graph import (
            MemoryDecayConfig,
            RegistryNode,
            RegistryNodeType,
        )

        config = MemoryDecayConfig(similarity_threshold=0.3, max_edges_per_node=3)
        linker = AutoSimilarityLinker(config=config)

        # Create many similar nodes
        existing = [
            RegistryNode(
                id=f"n{i}",
                type=RegistryNodeType.MEMORY,
                name=f"t{i}",
                embedding=[0.9 + i * 0.01, 0.1, 0.0],
            )
            for i in range(10)
        ]
        new_node = RegistryNode(
            id="new",
            type=RegistryNodeType.MEMORY,
            name="new",
            embedding=[0.95, 0.1, 0.0],
        )

        edges = linker.link_new_node(new_node, existing)
        assert len(edges) <= 3


# ── KG-2.37: Hybrid Search Scorer ───────────────────────────────────


class TestHybridSearchScorer:
    """Tests for CONCEPT:KG-2.3 — Hybrid Search Index."""

    def test_relevant_doc_ranked_first(self):
        """Document matching query should rank above irrelevant one."""
        from agent_utilities.knowledge_graph.retrieval.semantic_retrieval_engine import (
            HybridSearchScorer,
        )

        scorer = HybridSearchScorer()
        docs = [
            {
                "id": "d1",
                "text": "spectral clustering eigenvalue decomposition",
                "embedding": [0.9, 0.1, 0.0, 0.0],
            },
            {
                "id": "d2",
                "text": "cooking recipe for pasta sauce",
                "embedding": [0.0, 0.0, 0.9, 0.1],
            },
        ]
        results = scorer.score_documents(
            "spectral clustering", [0.95, 0.05, 0.0, 0.0], docs
        )
        assert len(results) >= 1
        assert results[0]["id"] == "d1"

    def test_compound_name_splitting(self):
        """camelCase and snake_case should be split correctly."""
        from agent_utilities.knowledge_graph.retrieval.semantic_retrieval_engine import (
            _split_compound_name,
        )

        assert "function" in _split_compound_name("myFunctionName")
        assert "class" in _split_compound_name("MyClass_with_method")
        assert "registry" in _split_compound_name("RegistryNode")

    def test_custom_config(self):
        """Custom config should be respected."""
        from agent_utilities.knowledge_graph.retrieval.semantic_retrieval_engine import (
            HybridSearchScorer,
        )
        from agent_utilities.models.knowledge_graph import HybridSearchConfig

        config = HybridSearchConfig(
            semantic_weight=0.5,
            keyword_weight=0.5,
            top_k=1,
        )
        scorer = HybridSearchScorer(config=config)
        docs = [
            {"id": "d1", "text": "alpha beta", "embedding": [0.5, 0.5]},
            {"id": "d2", "text": "gamma delta", "embedding": [0.5, 0.5]},
        ]
        results = scorer.score_documents("alpha", [0.5, 0.5], docs)
        assert len(results) <= 1  # top_k=1


# ── KG-2.33: Research Subagent ──────────────────────────────────────


class TestResearchSubagent:
    """Tests for CONCEPT:KG-2.6 — Research Intelligence Sub-Agent."""

    def test_session_lifecycle(self):
        """Session should transition from active to completed."""
        from agent_utilities.knowledge_graph.orchestration.research_subagent import (
            ResearchSubagent,
        )

        sa = ResearchSubagent(query="test query")
        assert sa.status == "active"

        sa.add_finding("Finding 1", confidence=0.9)
        sa.add_paper("p1", "Paper 1", ["Author"], 2025)
        session = sa.finalize()

        assert session.status == "completed"
        assert session.findings_count == 1
        assert session.papers_discovered == 1

    def test_token_budget_warning(self):
        """Exceeding warn threshold should return warning."""
        from agent_utilities.knowledge_graph.orchestration.research_subagent import (
            ResearchSubagent,
        )

        sa = ResearchSubagent(query="test", token_budget_warn=100, token_budget_max=200)
        result = sa.add_tokens(50)
        assert result is None

        result = sa.add_tokens(60)
        assert isinstance(result, str) and "WARNING" in result

    def test_token_budget_exceeded(self):
        """Exceeding max budget should change status."""
        from agent_utilities.knowledge_graph.orchestration.research_subagent import (
            ResearchSubagent,
        )

        sa = ResearchSubagent(query="test", token_budget_max=100)
        result = sa.add_tokens(150)
        assert isinstance(result, str) and "EXCEEDED" in result
        assert sa.status == "budget_exceeded"

    def test_provenance_edges(self):
        """Provenance edges should link findings and papers to session."""
        from agent_utilities.knowledge_graph.orchestration.research_subagent import (
            ResearchSubagent,
        )

        sa = ResearchSubagent(query="test")
        sa.add_finding("F1")
        sa.add_paper("p1", "P1")
        edges = sa.get_provenance_edges()
        assert len(edges) >= 2
