#!/usr/bin/python
"""Tests for CONCEPT:AHE-3.4 — Heavy Thinking Orchestration.

Covers:
- HeavyThinkingConfig validation and defaults
- ComplexityEstimator tiered classification
- MemoryCache serialization, pruning, shuffling, augmentation
- TrajectoryPruner thinking token removal
- TrajectoryShuffler randomization
- TrajectoryNode and DeliberationNode KG models
- WorkspaceAttention deliberation scoring
- HeavyThinkingOrchestrator pipeline integration
- KG persistence of trajectories and deliberation nodes
"""

import hashlib

import pytest

from agent_utilities.graph.heavy_thinking import (
    ComplexityEstimator,
    HeavyThinkingConfig,
    HeavyThinkingOrchestrator,
    HeavyThinkingPlanner,
)
from agent_utilities.graph.memory_cache import (
    MemoryCache,
    TrajectoryEntry,
    TrajectoryPruner,
    TrajectoryShuffler,
)
from agent_utilities.graph.workspace_attention import Proposal, WorkspaceAttention
from agent_utilities.models.knowledge_graph import (
    DeliberationNode,
    RegistryEdgeType,
    RegistryNodeType,
    TrajectoryNode,
)

# ── HeavyThinkingConfig Tests ────────────────────────────────────────


@pytest.mark.concept("AHE-3.7")
@pytest.mark.timeout(30)
class TestHeavyThinkingConfig:
    """Test configuration model for Heavy Thinking."""

    def test_default_values(self):
        """Config should have sensible production defaults."""
        config = HeavyThinkingConfig()
        assert config.k == 4
        assert config.k_summary == 4
        assert config.max_iterations == 1
        assert config.complexity_threshold == 0.6
        assert config.prune_thinking_tokens is True
        assert config.shuffle_trajectories is True
        assert config.persist_to_kg is True
        assert config.thinker_timeout == 120.0

    def test_k_configurable(self):
        """K value should be configurable from 1 to 32."""
        config = HeavyThinkingConfig(k=8)
        assert config.k == 8

        config = HeavyThinkingConfig(k=1)
        assert config.k == 1

        config = HeavyThinkingConfig(k=32)
        assert config.k == 32

    def test_k_validation(self):
        """K value must be within bounds."""
        with pytest.raises(Exception):
            HeavyThinkingConfig(k=0)
        with pytest.raises(Exception):
            HeavyThinkingConfig(k=33)

    def test_complexity_threshold_bounds(self):
        """Complexity threshold must be [0.0, 1.0]."""
        config = HeavyThinkingConfig(complexity_threshold=0.0)
        assert config.complexity_threshold == 0.0

        config = HeavyThinkingConfig(complexity_threshold=1.0)
        assert config.complexity_threshold == 1.0

        with pytest.raises(Exception):
            HeavyThinkingConfig(complexity_threshold=-0.1)
        with pytest.raises(Exception):
            HeavyThinkingConfig(complexity_threshold=1.1)

    def test_max_iterations_bounds(self):
        """Max iterations must be [0, 5]."""
        config = HeavyThinkingConfig(max_iterations=0)
        assert config.max_iterations == 0

        config = HeavyThinkingConfig(max_iterations=5)
        assert config.max_iterations == 5

        with pytest.raises(Exception):
            HeavyThinkingConfig(max_iterations=6)


# ── ComplexityEstimator Tests ─────────────────────────────────────────


@pytest.mark.concept("AHE-3.7")
@pytest.mark.timeout(30)
class TestComplexityEstimator:
    """Test tiered hybrid complexity estimation."""

    def test_simple_query_low_complexity(self):
        """Short simple queries should have low complexity."""
        score = ComplexityEstimator.estimate("What time is it?")
        assert score < 0.4

    def test_complex_query_high_complexity(self):
        """Complex multi-step queries should have high complexity."""
        query = (
            "Analyze the performance regression in the authentication module, "
            "compare the current implementation with the previous version, "
            "and then debug the failing integration tests. First, review the "
            "git log. Then, investigate the test output. Finally, propose a fix."
        )
        score = ComplexityEstimator.estimate(query)
        assert score >= 0.4

    def test_code_block_increases_complexity(self):
        """Queries with code blocks should score higher."""
        base_query = "Fix this function"
        code_query = "Fix this function:\n```python\ndef broken():\n    pass\n```"

        base_score = ComplexityEstimator.estimate(base_query)
        code_score = ComplexityEstimator.estimate(code_query)
        assert code_score > base_score

    def test_keyword_markers_increase_complexity(self):
        """Complexity keywords should increase score."""
        simple = "get the list of files"
        complex_q = "analyze and evaluate the architectural design"

        simple_score = ComplexityEstimator.estimate(simple)
        complex_score = ComplexityEstimator.estimate(complex_q)
        assert complex_score > simple_score

    def test_tier2_low_confidence_increases_complexity(self):
        """Low specialist confidence should increase complexity."""
        query = "What is the answer?"
        score_no_signal = ComplexityEstimator.estimate(query)
        score_low_conf = ComplexityEstimator.estimate(query, specialist_confidence=0.2)
        assert score_low_conf > score_no_signal

    def test_tier2_high_diversity_increases_complexity(self):
        """High specialist diversity should increase complexity."""
        query = "What is the answer?"
        score_no_signal = ComplexityEstimator.estimate(query)
        score_high_div = ComplexityEstimator.estimate(query, specialist_diversity=5)
        assert score_high_div > score_no_signal

    def test_tier2_high_confidence_decreases_complexity(self):
        """High specialist confidence should moderate complexity."""
        query = "analyze the performance"
        score_no_signal = ComplexityEstimator.estimate(query)
        score_high_conf = ComplexityEstimator.estimate(
            query, specialist_confidence=0.95
        )
        # High confidence should reduce or at least not increase
        assert score_high_conf <= score_no_signal + 0.1

    def test_multi_step_patterns(self):
        """Numbered steps should increase complexity."""
        query = (
            "1. First check the logs. 2. Then review the code. 3. Finally fix the bug."
        )
        score = ComplexityEstimator.estimate(query)
        assert score > 0.3


# ── TrajectoryPruner Tests ────────────────────────────────────────────


@pytest.mark.concept("AHE-3.7")
@pytest.mark.timeout(30)
class TestTrajectoryPruner:
    """Test thinking token pruning."""

    def test_prune_think_tags(self):
        """Should remove <think>...</think> tags."""
        text = "Hello <think>internal reasoning here</think> World"
        result = TrajectoryPruner.prune(text)
        assert "<think>" not in result
        assert "internal reasoning" not in result
        assert "Hello" in result
        assert "World" in result

    def test_prune_thinking_tags(self):
        """Should remove <thinking>...</thinking> tags."""
        text = "Start <thinking>deep thought process</thinking> End"
        result = TrajectoryPruner.prune(text)
        assert "deep thought" not in result
        assert "Start" in result
        assert "End" in result

    def test_prune_code_thinking_blocks(self):
        """Should remove ```thinking blocks."""
        text = "Before\n```thinking\ninternal process\n```\nAfter"
        result = TrajectoryPruner.prune(text)
        assert "internal process" not in result
        assert "Before" in result
        assert "After" in result

    def test_prune_preserves_non_thinking_content(self):
        """Should preserve content outside thinking tags."""
        text = "The answer is 42. The reasoning is sound."
        result = TrajectoryPruner.prune(text)
        assert result == text

    def test_prune_multiple_thinking_blocks(self):
        """Should remove multiple thinking blocks."""
        text = "<think>first</think> middle <think>second</think> end"
        result = TrajectoryPruner.prune(text)
        assert "first" not in result
        assert "second" not in result
        assert "middle" in result
        assert "end" in result

    def test_extract_boxed_answer(self):
        """Should extract \\boxed{} answers."""
        text = "The calculation gives us \\boxed{42}"
        answer = TrajectoryPruner.extract_answer(text)
        assert answer == "42"

    def test_extract_bold_answer(self):
        """Should extract **Answer**: patterns."""
        text = "After analysis:\n**Answer**: The solution is X=5"
        answer = TrajectoryPruner.extract_answer(text)
        assert "X=5" in answer

    def test_extract_answer_plain_text(self):
        """Should extract 'The answer is...' patterns."""
        text = "Considering all factors, the answer is 7"
        answer = TrajectoryPruner.extract_answer(text)
        assert "7" in answer

    def test_extract_answer_fallback(self):
        """Should fall back to last line when no patterns match."""
        text = "Some reasoning\nMore reasoning\nFinal conclusion"
        answer = TrajectoryPruner.extract_answer(text)
        assert answer == "Final conclusion"


# ── TrajectoryShuffler Tests ──────────────────────────────────────────


@pytest.mark.concept("AHE-3.7")
@pytest.mark.timeout(30)
class TestTrajectoryShuffler:
    """Test trajectory randomization."""

    def test_shuffle_preserves_all_entries(self):
        """Shuffling should preserve all entries."""
        entries = [
            TrajectoryEntry(thinker_id=f"t{i}", raw_output=f"output {i}")
            for i in range(10)
        ]
        shuffled = TrajectoryShuffler.shuffle(entries)
        assert len(shuffled) == len(entries)
        assert set(e.thinker_id for e in shuffled) == set(e.thinker_id for e in entries)

    def test_shuffle_returns_new_list(self):
        """Shuffling should return a new list, not modify in-place."""
        entries = [
            TrajectoryEntry(thinker_id=f"t{i}", raw_output=f"output {i}")
            for i in range(5)
        ]
        original_ids = [e.thinker_id for e in entries]
        TrajectoryShuffler.shuffle(entries)

        # Original should be unchanged
        assert [e.thinker_id for e in entries] == original_ids

    def test_shuffle_empty_list(self):
        """Shuffling empty list should return empty list."""
        assert TrajectoryShuffler.shuffle([]) == []


# ── MemoryCache Tests ─────────────────────────────────────────────────


@pytest.mark.concept("AHE-3.7")
@pytest.mark.timeout(30)
class TestMemoryCache:
    """Test the Serialized Memory Cache."""

    def test_from_query(self):
        """Should create a cache with query hash."""
        cache = MemoryCache.from_query("What is 2+2?")
        assert cache.query == "What is 2+2?"
        assert cache.query_hash == hashlib.sha256(b"What is 2+2?").hexdigest()
        assert len(cache.trajectories) == 0

    def test_add_trajectory(self):
        """Should add trajectories correctly."""
        cache = MemoryCache.from_query("test query")
        cache.add_trajectory("t1", "The answer is 42", model_id="gpt-4")
        cache.add_trajectory("t2", "I think it's 43", model_id="claude")

        assert len(cache.trajectories) == 2
        assert cache.trajectories[0].thinker_id == "t1"
        assert cache.trajectories[1].model_id == "claude"

    def test_serialize_includes_header(self):
        """Serialized output should include header with metadata."""
        cache = MemoryCache.from_query("test query")
        cache.add_trajectory("t1", "Answer is 42")

        result = cache.serialize(prune=False, shuffle=False)
        assert "Serialized Memory Cache" in result
        assert "test query" in result
        assert "**Trajectories**: 1" in result

    def test_serialize_with_pruning(self):
        """Serialized output should use pruned reasoning."""
        cache = MemoryCache.from_query("test")
        cache.add_trajectory("t1", "Hello <think>secret thought</think> World")

        result = cache.serialize(prune=True, shuffle=False)
        assert "secret thought" not in result
        assert "Thinker 1" in result

    def test_serialize_without_pruning(self):
        """Serialize without pruning should include raw output."""
        cache = MemoryCache.from_query("test")
        cache.add_trajectory("t1", "Hello <think>internal</think> World")

        result = cache.serialize(prune=False, shuffle=False)
        assert "Hello <think>internal</think> World" in result

    def test_augment_adds_deliberation(self):
        """Augmenting should add deliberation result."""
        cache = MemoryCache.from_query("test")
        cache.augment("First deliberation result")
        cache.augment("Second deliberation result")

        assert len(cache.deliberation_results) == 2
        assert "First deliberation result" in cache.deliberation_results[0]

    def test_serialize_includes_prior_deliberations(self):
        """Serialized output should include prior deliberation results."""
        cache = MemoryCache.from_query("test")
        cache.add_trajectory("t1", "Answer 1")
        cache.augment("Prior deliberation output")

        result = cache.serialize(prune=False, shuffle=False)
        assert "Prior Deliberation Results" in result
        assert "Prior deliberation output" in result

    def test_serialize_iteration_counter(self):
        """Iteration counter should reflect augmentation count."""
        cache = MemoryCache.from_query("test")
        cache.add_trajectory("t1", "Answer")

        # First serialize: iteration 1
        result1 = cache.serialize()
        assert "**Iteration**: 1" in result1

        cache.augment("delib 1")
        result2 = cache.serialize()
        assert "**Iteration**: 2" in result2


# ── KG Model Tests ────────────────────────────────────────────────────


@pytest.mark.concept("AHE-3.7")
@pytest.mark.timeout(30)
class TestKGModels:
    """Test TrajectoryNode and DeliberationNode models."""

    def test_trajectory_node_creation(self):
        """Should create a valid TrajectoryNode."""
        node = TrajectoryNode(
            id="traj:test001",
            name="Test Trajectory",
            thinker_id="thinker_0",
            query_hash="abc123",
            answer="42",
            reasoning_summary="The answer is derived from...",
            score=0.85,
            is_correct=True,
            model_id="gpt-4",
        )
        assert node.type == RegistryNodeType.TRAJECTORY
        assert node.thinker_id == "thinker_0"
        assert node.score == 0.85
        assert node.is_correct is True

    def test_trajectory_node_defaults(self):
        """TrajectoryNode should have sensible defaults."""
        node = TrajectoryNode(id="traj:test", name="Test")
        assert node.type == RegistryNodeType.TRAJECTORY
        assert node.thinker_id == ""
        assert node.score == 0.0
        assert node.is_correct is None

    def test_trajectory_score_bounds(self):
        """Score must be [0.0, 1.0]."""
        node = TrajectoryNode(id="t", name="t", score=0.0)
        assert node.score == 0.0

        node = TrajectoryNode(id="t", name="t", score=1.0)
        assert node.score == 1.0

        with pytest.raises(Exception):
            TrajectoryNode(id="t", name="t", score=1.5)

    def test_deliberation_node_creation(self):
        """Should create a valid DeliberationNode."""
        node = DeliberationNode(
            id="delib:test001",
            name="Test Deliberation",
            trajectories_analyzed=4,
            consensus_answer="The answer is 42",
            confidence=0.92,
            critical_analysis="All 4 trajectories agreed on 42",
            iteration=1,
            model_id="claude-3",
        )
        assert node.type == RegistryNodeType.DELIBERATION
        assert node.trajectories_analyzed == 4
        assert node.confidence == 0.92
        assert node.iteration == 1

    def test_deliberation_node_defaults(self):
        """DeliberationNode should have sensible defaults."""
        node = DeliberationNode(id="d", name="d")
        assert node.trajectories_analyzed == 0
        assert node.confidence == 0.0
        assert node.iteration == 0

    def test_edge_type_trajectory_of(self):
        """TRAJECTORY_OF edge type should be defined."""
        assert RegistryEdgeType.TRAJECTORY_OF == "trajectory_of"

    def test_edge_type_deliberated_by(self):
        """DELIBERATED_BY edge type should be defined."""
        assert RegistryEdgeType.DELIBERATED_BY == "deliberated_by"

    def test_edge_type_agrees_with(self):
        """AGREES_WITH edge type should be defined."""
        assert RegistryEdgeType.AGREES_WITH == "agrees_with"

    def test_edge_type_disagrees_with(self):
        """DISAGREES_WITH edge type should be defined."""
        assert RegistryEdgeType.DISAGREES_WITH == "disagrees_with"


# ── WorkspaceAttention Deliberation Score Tests ───────────────────────


@pytest.mark.concept("AHE-3.7")
@pytest.mark.timeout(30)
class TestDeliberationScore:
    """Test cross-trajectory deliberation scoring."""

    def _make_proposals(self, outputs: list[str]) -> list[Proposal]:
        """Helper to create proposals from output strings."""
        return [
            Proposal(
                specialist_id=f"s{i}",
                output=out,
                confidence_score=0.5,
            )
            for i, out in enumerate(outputs)
        ]

    def test_empty_proposals(self):
        """Empty proposals should return zero scores."""
        wa = WorkspaceAttention()
        result = wa.deliberation_score([])
        assert result["consensus"] == 0.0
        assert result["deliberation_needed"] == 0.0

    def test_full_consensus(self):
        """All identical outputs should show full consensus."""
        wa = WorkspaceAttention()
        proposals = self._make_proposals(["same answer"] * 4)
        result = wa.deliberation_score(proposals)
        assert result["consensus"] == 1.0
        assert result["diversity"] == pytest.approx(0.25, abs=0.01)

    def test_no_consensus(self):
        """All different outputs should show no consensus."""
        wa = WorkspaceAttention()
        proposals = self._make_proposals(
            ["answer A", "answer B", "answer C", "answer D"]
        )
        result = wa.deliberation_score(proposals)
        assert result["consensus"] == 0.25  # Each answer has 1/4
        assert result["diversity"] == 1.0  # All unique

    def test_deliberation_needed_high_diversity(self):
        """High diversity with moderate confidence should need deliberation."""
        wa = WorkspaceAttention()
        proposals = [
            Proposal(
                specialist_id=f"s{i}",
                output=f"answer {i}",
                confidence_score=0.5,
            )
            for i in range(4)
        ]
        result = wa.deliberation_score(proposals)
        assert result["deliberation_needed"] > 0.5

    def test_deliberation_not_needed_high_confidence(self):
        """High confidence with consensus should not need deliberation."""
        wa = WorkspaceAttention()
        proposals = [
            Proposal(
                specialist_id=f"s{i}",
                output="consensus answer",
                confidence_score=0.95,
            )
            for i in range(4)
        ]
        result = wa.deliberation_score(proposals)
        # High confidence reduces deliberation need
        assert result["deliberation_needed"] < 0.5


# ── HeavyThinkingOrchestrator Tests ───────────────────────────────────


@pytest.mark.concept("AHE-3.7")
@pytest.mark.timeout(30)
class TestHeavyThinkingOrchestrator:
    """Test the orchestrator initialization and configuration."""

    def test_default_config(self):
        """Orchestrator should initialize with default config."""
        orchestrator = HeavyThinkingOrchestrator()
        assert orchestrator.config.k == 4
        assert orchestrator.config.max_iterations == 1

    def test_custom_config(self):
        """Orchestrator should accept custom config."""
        config = HeavyThinkingConfig(k=8, max_iterations=3)
        orchestrator = HeavyThinkingOrchestrator(config)
        assert orchestrator.config.k == 8
        assert orchestrator.config.max_iterations == 3

    def test_majority_vote_fallback(self):
        """Majority vote should select most common answer."""
        orchestrator = HeavyThinkingOrchestrator()
        trajectories = [
            TrajectoryEntry(thinker_id="t0", answer="42", success=True),
            TrajectoryEntry(thinker_id="t1", answer="42", success=True),
            TrajectoryEntry(thinker_id="t2", answer="43", success=True),
            TrajectoryEntry(thinker_id="t3", answer="42", success=True),
        ]
        result = orchestrator._majority_vote_fallback(trajectories)
        assert result["answer"] == "42"
        assert result["confidence"] == pytest.approx(0.75, abs=0.01)

    def test_majority_vote_no_trajectories(self):
        """Majority vote with no trajectories should return empty."""
        orchestrator = HeavyThinkingOrchestrator()
        result = orchestrator._majority_vote_fallback([])
        assert result["answer"] == ""
        assert result["confidence"] == 0.0

    def test_majority_vote_skips_failures(self):
        """Majority vote should skip failed trajectories."""
        orchestrator = HeavyThinkingOrchestrator()
        trajectories = [
            TrajectoryEntry(thinker_id="t0", answer="42", success=True),
            TrajectoryEntry(thinker_id="t1", answer="error", success=False),
            TrajectoryEntry(thinker_id="t2", answer="42", success=True),
        ]
        result = orchestrator._majority_vote_fallback(trajectories)
        assert result["answer"] == "42"


# ── MemoryCache KG Persistence Tests ──────────────────────────────────


@pytest.mark.concept("AHE-3.7")
@pytest.mark.timeout(30)
class TestMemoryCacheKGPersistence:
    """Test KG persistence of trajectories."""

    def test_to_kg_nodes_creates_trajectory_nodes(self):
        """to_kg_nodes should create TrajectoryNode instances in the graph."""
        from agent_utilities.knowledge_graph.core.graph_compute import GraphComputeEngine

        from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine

        engine = IntelligenceGraphEngine(graph=GraphComputeEngine(backend_type="rust"))
        cache = MemoryCache.from_query("What is 2+2?")
        cache.add_trajectory("t1", "The answer is \\boxed{4}", model_id="gpt-4")
        cache.add_trajectory("t2", "2+2 = 4. **Answer**: 4", model_id="claude")

        node_ids = cache.to_kg_nodes(engine)

        assert len(node_ids) == 2
        for nid in node_ids:
            assert nid.startswith("traj:")
            assert nid in engine.graph

    def test_to_kg_nodes_creates_query_anchor(self):
        """Should create a query anchor node."""
        from agent_utilities.knowledge_graph.core.graph_compute import GraphComputeEngine

        from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine

        engine = IntelligenceGraphEngine(graph=GraphComputeEngine(backend_type="rust"))
        cache = MemoryCache.from_query("test query")
        cache.add_trajectory("t1", "answer")

        cache.to_kg_nodes(engine)

        query_hash = cache.query_hash[:12]
        anchor_id = f"query:{query_hash}"
        assert anchor_id in engine.graph

    def test_to_kg_nodes_includes_enc_pi(self):
        """Trajectory nodes should include EncPI metadata."""
        from agent_utilities.knowledge_graph.core.graph_compute import GraphComputeEngine

        from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine

        engine = IntelligenceGraphEngine(graph=GraphComputeEngine(backend_type="rust"))
        cache = MemoryCache.from_query("test")
        cache.add_trajectory("t1", "answer 1")
        cache.add_trajectory("t2", "answer 2")

        node_ids = cache.to_kg_nodes(engine)

        for nid in node_ids:
            node_data = engine.graph.nodes[nid]
            metadata = node_data.get("metadata", {})
            assert "enc_pi" in metadata
            assert "source" in metadata
            assert metadata["source"] == "heavy_thinking"


# ── Integration: HeavyThinkingPlanner Tests ───────────────────────────


@pytest.mark.concept("AHE-3.7")
@pytest.mark.timeout(30)
class TestHeavyThinkingPlannerConfig:
    """Test the HeavyThinkingPlanner configuration."""

    def test_planner_wraps_orchestrator(self):
        """HeavyThinkingPlanner should wrap the orchestrator."""
        from unittest.mock import MagicMock

        deps = MagicMock()
        deps.agent_model = "test-model"

        planner = HeavyThinkingPlanner(
            context="test context",
            deps=deps,
            model="test-model",
        )
        assert planner.orchestrator is not None
        assert planner.config.k == 4

    def test_planner_accepts_custom_config(self):
        """HeavyThinkingPlanner should accept custom config."""
        from unittest.mock import MagicMock

        deps = MagicMock()
        config = HeavyThinkingConfig(k=16, max_iterations=3)

        planner = HeavyThinkingPlanner(
            context="test",
            deps=deps,
            model="model",
            config=config,
        )
        assert planner.config.k == 16
        assert planner.config.max_iterations == 3
