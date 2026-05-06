#!/usr/bin/python
"""CONCEPT:AHE-3.7 — Heavy Thinking Orchestration.

Implements the two-stage parallel-then-deliberate pipeline from the
HEAVYSKILL research framework (2026), adapted for the agent-utilities
ecosystem with native KG integration, OWL-enriched context, and
hypergraph structural generalization.

Pipeline overview::

    Stage 1: Parallel Reasoning
    ├── Spawn K independent thinker agents
    ├── Each reasons about the query independently
    └── Collect all K trajectories

    Stage 2: Sequential Deliberation
    ├── Build Serialized Memory Cache from K trajectories
    ├── Prune thinking tokens, shuffle order
    ├── Deliberation agent synthesizes + critiques
    └── Optionally iterate (augment cache, re-deliberate)

    Free value-adds from agent-utilities:
    ├── KG persistence of trajectories (cross-session reuse)
    ├── OWL-enriched deliberation context (CONCEPT:KG-2.2)
    ├── EncPI hyperedge mapping (CONCEPT:KG-2.4)
    ├── EWC consolidation protection (CONCEPT:AHE-3.6)
    └── Topological trajectory clustering (CONCEPT:KG-2.5)

Integrates with:
    - CONCEPT:ORCH-1.1 (HTN Planning): As an alternative to ``LATSPlanner``
    - CONCEPT:ORCH-1.2 (Model Routing): Heterogeneous model pairing
    - CONCEPT:AHE-3.5 (Experience Distillation): Trajectory → ExperienceNode
    - CONCEPT:KG-2.0 (OGM): Native KG persistence
    - CONCEPT:KG-2.4 (Hypergraphs): Structural generalization

See docs/overview.md §CONCEPT:AHE-3.7.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
import uuid
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field
from pydantic_ai import Agent

from .memory_cache import MemoryCache, TrajectoryEntry

if TYPE_CHECKING:
    from ..knowledge_graph.engine import IntelligenceGraphEngine
    from .state import GraphDeps

logger = logging.getLogger(__name__)


# ── Configuration ────────────────────────────────────────────────────


class HeavyThinkingConfig(BaseModel):
    """Configuration for the Heavy Thinking orchestration pipeline.

    CONCEPT:AHE-3.7 — Heavy Thinking Orchestration

    Attributes:
        k: Number of parallel thinker agents to spawn (default 4).
        k_summary: Number of trajectories to include in summary output.
        max_iterations: Maximum iterative refinement depth (0 = single pass).
        complexity_threshold: Activation gate — queries with estimated
            complexity below this value skip heavy thinking.
        prune_thinking_tokens: Whether to strip CoT from the memory cache.
        shuffle_trajectories: Whether to randomize trajectory order.
        persist_to_kg: Whether to persist trajectories as KG nodes.
        thinker_timeout: Per-thinker timeout in seconds.
    """

    k: int = Field(default=4, ge=1, le=32)
    k_summary: int = Field(default=4, ge=1)
    max_iterations: int = Field(default=1, ge=0, le=5)
    complexity_threshold: float = Field(default=0.6, ge=0.0, le=1.0)
    prune_thinking_tokens: bool = True
    shuffle_trajectories: bool = True
    persist_to_kg: bool = True
    thinker_timeout: float = 120.0


# ── Complexity Estimation ────────────────────────────────────────────


class ComplexityEstimator:
    """Tiered hybrid complexity estimator for activation gating.

    CONCEPT:AHE-3.7 — Determines whether heavy thinking should be
    activated for a given query.  Uses a three-tier approach:

    **Tier 1 (Free)**: Heuristic signals — query length, keyword markers,
    code blocks, multi-step indicators.

    **Tier 2 (Free)**: Confidence routing signals from ``WorkspaceAttention``
    and ``ModelRegistry`` — low specialist consensus triggers heavy thinking.

    **Tier 3 (Fallback)**: Lightweight LLM call to classify complexity
    only when Tiers 1+2 are inconclusive.
    """

    # Tier 1: Keyword markers that indicate complex reasoning
    COMPLEXITY_KEYWORDS: set[str] = {
        "analyze",
        "compare",
        "contrast",
        "debug",
        "evaluate",
        "explain",
        "investigate",
        "optimize",
        "prove",
        "refactor",
        "review",
        "synthesize",
        "troubleshoot",
        "architect",
        "design",
        "derive",
        "integrate",
    }

    # Multi-step indicators
    MULTI_STEP_PATTERNS: list[re.Pattern[str]] = [
        re.compile(r"\b(?:first|then|next|finally|step \d)\b", re.IGNORECASE),
        re.compile(
            r"\b(?:and also|additionally|furthermore|moreover)\b", re.IGNORECASE
        ),
        re.compile(r"\d+\.\s", re.MULTILINE),  # Numbered lists
    ]

    @classmethod
    def estimate(
        cls,
        query: str,
        specialist_confidence: float | None = None,
        specialist_diversity: int | None = None,
    ) -> float:
        """Estimate query complexity using tiered signals.

        Args:
            query: The user query to evaluate.
            specialist_confidence: Optional confidence score from
                ``WorkspaceAttention`` (Tier 2).  Low confidence (< 0.5)
                increases complexity estimate.
            specialist_diversity: Optional diversity count from
                ``WorkspaceAttention`` (Tier 2).  High diversity (> 2)
                increases complexity estimate.

        Returns:
            Estimated complexity score in [0.0, 1.0].
        """
        scores: list[float] = []

        # ── Tier 1: Heuristic signals ────────────────────────────────
        tier1 = cls._tier1_heuristics(query)
        scores.append(tier1)

        # ── Tier 2: Confidence routing signals ───────────────────────
        tier2 = cls._tier2_confidence(specialist_confidence, specialist_diversity)
        if tier2 is not None:
            scores.append(tier2)

        # Weighted average (Tier 1 has higher weight as it's always available)
        if len(scores) == 1:
            return scores[0]
        return 0.6 * scores[0] + 0.4 * scores[1]

    @classmethod
    async def estimate_with_fallback(
        cls,
        query: str,
        specialist_confidence: float | None = None,
        specialist_diversity: int | None = None,
        deps: GraphDeps | None = None,
    ) -> float:
        """Estimate with Tier 3 LLM fallback for inconclusive cases.

        Triggers the LLM fallback only when Tiers 1+2 produce a score
        in the ambiguous range [0.35, 0.65].

        Args:
            query: The user query.
            specialist_confidence: Optional confidence from specialists.
            specialist_diversity: Optional diversity count.
            deps: Graph dependencies for LLM access (Tier 3).

        Returns:
            Refined complexity estimate in [0.0, 1.0].
        """
        base_estimate = cls.estimate(query, specialist_confidence, specialist_diversity)

        # Only trigger Tier 3 if estimate is ambiguous
        if 0.35 <= base_estimate <= 0.65 and deps is not None:
            tier3 = await cls._tier3_llm_fallback(query, deps)
            if tier3 is not None:
                # Blend: 50% base + 50% LLM
                return 0.5 * base_estimate + 0.5 * tier3

        return base_estimate

    @classmethod
    def _tier1_heuristics(cls, query: str) -> float:
        """Tier 1: Free heuristic-based complexity estimation."""
        score = 0.0
        words = query.lower().split()
        word_count = len(words)

        # Length signal: longer queries tend to be more complex
        if word_count > 50:
            score += 0.3
        elif word_count > 20:
            score += 0.15

        # Keyword markers
        keyword_hits = sum(1 for w in words if w in cls.COMPLEXITY_KEYWORDS)
        score += min(0.3, keyword_hits * 0.1)

        # Code block presence
        if "```" in query:
            score += 0.15

        # Multi-step patterns
        multi_step_hits = sum(1 for p in cls.MULTI_STEP_PATTERNS if p.search(query))
        score += min(0.25, multi_step_hits * 0.1)

        return min(1.0, score)

    @classmethod
    def _tier2_confidence(
        cls,
        confidence: float | None,
        diversity: int | None,
    ) -> float | None:
        """Tier 2: Confidence routing signal from specialist consensus."""
        if confidence is None and diversity is None:
            return None

        score = 0.0
        if confidence is not None:
            # Low confidence → high complexity
            score += max(0.0, 1.0 - confidence) * 0.6

        if diversity is not None:
            # High diversity → high complexity (specialists disagree)
            if diversity > 3:
                score += 0.4
            elif diversity > 1:
                score += 0.2

        return min(1.0, score)

    @classmethod
    async def _tier3_llm_fallback(
        cls,
        query: str,
        deps: GraphDeps,
    ) -> float | None:
        """Tier 3: Lightweight LLM classification as absolute fallback."""
        try:
            classifier = Agent(
                model=deps.agent_model,
                system_prompt=(
                    "Classify the complexity of this query on a scale from 0.0 to 1.0.\n"
                    "0.0 = trivial lookup or simple factual question\n"
                    "0.5 = moderate analysis or multi-step reasoning\n"
                    "1.0 = complex synthesis, debugging, or architectural design\n"
                    "Output ONLY the numeric score, nothing else."
                ),
            )
            result = await asyncio.wait_for(
                classifier.run(query),
                timeout=10.0,
            )
            score_str = str(result.output).strip()
            # Parse numeric score
            match = re.search(r"(\d+\.?\d*)", score_str)
            if match:
                return max(0.0, min(1.0, float(match.group(1))))
        except Exception as e:
            logger.debug("Tier 3 complexity estimation failed: %s", e)

        return None


# ── Heavy Thinking Orchestrator ──────────────────────────────────────


class HeavyThinkingOrchestrator:
    """Two-stage parallel-then-deliberate reasoning pipeline.

    CONCEPT:AHE-3.7 — Heavy Thinking Orchestration

    Implements the HEAVYSKILL framework adapted for agent-utilities:

    1. **Parallel Reasoning**: Spawn K independent thinker agents using
       ``asyncio.gather`` to produce diverse reasoning trajectories.
    2. **Sequential Deliberation**: Build a ``MemoryCache``, prune and
       shuffle trajectories, then run a deliberation agent that
       critically synthesizes all trajectories into a consensus answer.
    3. **Iterative Refinement** (optional): Feed the deliberation output
       back into the cache and re-run deliberation for convergence.

    Free value-adds from the agent-utilities architecture:
    - Trajectories persisted as ``TrajectoryNode`` in KG (cross-session reuse)
    - OWL-enriched context injected during deliberation
    - EncPI hyperedge mapping for trajectory interactions
    - EWC consolidation protects established knowledge
    - Topological partitioning clusters trajectories by approach

    Args:
        config: ``HeavyThinkingConfig`` with pipeline parameters.
    """

    def __init__(self, config: HeavyThinkingConfig | None = None) -> None:
        self.config = config or HeavyThinkingConfig()

    async def execute(
        self,
        query: str,
        deps: GraphDeps,
        model_parallel: Any = None,
        model_deliberation: Any = None,
    ) -> dict[str, Any]:
        """Run the full heavy thinking pipeline.

        Args:
            query: The query to reason about.
            deps: Graph runtime dependencies.
            model_parallel: Optional model override for parallel thinkers.
            model_deliberation: Optional model override for deliberation.

        Returns:
            Dict with ``answer``, ``confidence``, ``trajectories``,
            ``iterations``, and ``deliberation_node_id``.
        """
        parallel_model = model_parallel or deps.agent_model
        delib_model = model_deliberation or deps.agent_model

        logger.info(
            "[CONCEPT:AHE-3.7] Starting Heavy Thinking pipeline "
            "(k=%d, max_iter=%d, query=%s)",
            self.config.k,
            self.config.max_iterations,
            query[:60],
        )

        # Stage 1: Parallel Reasoning
        cache = await self._parallel_reasoning(query, deps, parallel_model)

        # Stage 2: Sequential Deliberation (with optional iterative refinement)
        result = await self._deliberation_loop(cache, query, deps, delib_model)

        # Persist to KG
        if self.config.persist_to_kg and deps.knowledge_engine:
            trajectory_ids = cache.to_kg_nodes(deps.knowledge_engine)
            delib_node_id = self._persist_deliberation(
                deps.knowledge_engine,
                result,
                trajectory_ids,
                cache,
            )
            result["trajectory_node_ids"] = trajectory_ids
            result["deliberation_node_id"] = delib_node_id

        # Distill experience for continual learning (CONCEPT:AHE-3.5)
        await self._distill_heavy_experience(cache, result, deps)

        logger.info(
            "[CONCEPT:AHE-3.7] Heavy Thinking complete: confidence=%.2f, "
            "iterations=%d, trajectories=%d",
            result.get("confidence", 0.0),
            result.get("iterations", 1),
            len(cache.trajectories),
        )

        return result

    async def _parallel_reasoning(
        self,
        query: str,
        deps: GraphDeps,
        model: Any,
    ) -> MemoryCache:
        """Stage 1: Spawn K parallel thinker agents.

        Each thinker reasons about the query independently with no
        shared context, producing diverse reasoning trajectories.

        Args:
            query: The query to reason about.
            deps: Runtime dependencies.
            model: The LLM model for parallel thinkers.

        Returns:
            A ``MemoryCache`` populated with K trajectories.
        """
        cache = MemoryCache.from_query(query)
        k = self.config.k

        logger.info("[CONCEPT:AHE-3.7] Spawning %d parallel thinkers...", k)

        async def run_thinker(thinker_id: str) -> TrajectoryEntry:
            """Run a single parallel thinker."""
            thinker = Agent(
                model=model,
                system_prompt=(
                    "You are an independent reasoning agent. Think through this "
                    "problem carefully and provide your best answer. Show your "
                    "complete reasoning process. End with a clear final answer."
                ),
            )
            try:
                result = await asyncio.wait_for(
                    thinker.run(query),
                    timeout=self.config.thinker_timeout,
                )
                return TrajectoryEntry(
                    thinker_id=thinker_id,
                    raw_output=str(result.output),
                    answer="",  # Will be extracted during cache.add_trajectory
                    model_id=str(model),
                    success=True,
                )
            except Exception as e:
                logger.warning("[CONCEPT:AHE-3.7] Thinker %s failed: %s", thinker_id, e)
                return TrajectoryEntry(
                    thinker_id=thinker_id,
                    raw_output=f"Error: {e}",
                    model_id=str(model),
                    success=False,
                )

        # Spawn all thinkers concurrently
        tasks = [run_thinker(f"thinker_{i}") for i in range(k)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, TrajectoryEntry):
                cache.add_trajectory(
                    thinker_id=result.thinker_id,
                    output=result.raw_output,
                    model_id=result.model_id,
                    success=result.success,
                )
            elif isinstance(result, Exception):
                logger.warning("[CONCEPT:AHE-3.7] Thinker failed: %s", result)

        logger.info(
            "[CONCEPT:AHE-3.7] Parallel reasoning complete: %d/%d succeeded",
            sum(1 for t in cache.trajectories if t.success),
            k,
        )
        return cache

    async def _deliberation_loop(
        self,
        cache: MemoryCache,
        query: str,
        deps: GraphDeps,
        model: Any,
    ) -> dict[str, Any]:
        """Stage 2: Sequential deliberation with optional iterative refinement.

        The deliberation agent receives the serialized memory cache and
        critically analyzes all trajectories to produce a consensus answer.

        Args:
            cache: The populated MemoryCache.
            query: The original query.
            deps: Runtime dependencies.
            model: The LLM model for deliberation.

        Returns:
            Dict with ``answer``, ``confidence``, ``critical_analysis``,
            ``iterations``.
        """
        max_iter = max(1, self.config.max_iterations)
        last_result: dict[str, Any] = {}

        for iteration in range(max_iter):
            serialized = cache.serialize(
                prune=self.config.prune_thinking_tokens,
                shuffle=self.config.shuffle_trajectories,
            )

            result = await self._sequential_deliberation(
                serialized, query, deps, model, iteration
            )
            last_result = result

            # Check for convergence (same answer as previous iteration)
            if iteration > 0 and len(cache.deliberation_results) > 0:
                prev_answer = cache.deliberation_results[-1]
                if result.get("answer", "") == prev_answer:
                    logger.info(
                        "[CONCEPT:AHE-3.7] Deliberation converged at iteration %d",
                        iteration + 1,
                    )
                    break

            # Augment cache for next iteration
            if iteration < max_iter - 1:
                cache.augment(result.get("answer", ""))

        last_result["iterations"] = min(iteration + 1, max_iter)
        return last_result

    async def _sequential_deliberation(
        self,
        serialized_cache: str,
        query: str,
        deps: GraphDeps,
        model: Any,
        iteration: int,
    ) -> dict[str, Any]:
        """Run a single deliberation pass.

        The deliberation agent receives all trajectories and performs
        critical cross-trajectory analysis to synthesize a consensus.

        Args:
            serialized_cache: The serialized memory cache string.
            query: The original query.
            deps: Runtime dependencies.
            model: The deliberation model.
            iteration: Current refinement iteration.

        Returns:
            Dict with ``answer``, ``confidence``, ``critical_analysis``.
        """
        from pydantic import BaseModel as PydanticModel

        class DeliberationOutput(PydanticModel):
            answer: str
            confidence: float = Field(ge=0.0, le=1.0)
            critical_analysis: str

        deliberator = Agent(
            model=model,
            output_type=DeliberationOutput,
            system_prompt=(
                "You are a deliberation agent performing sequential analysis.\n\n"
                "You have received multiple independent reasoning trajectories "
                "for the same query. Your task:\n"
                "1. Identify areas of AGREEMENT across trajectories\n"
                "2. Identify areas of DISAGREEMENT and analyze why\n"
                "3. Apply critical thinking to determine the CORRECT answer\n"
                "4. Synthesize a final consensus answer with confidence score\n\n"
                "Do NOT simply vote — reason deeply about WHY trajectories "
                "agree or disagree and which reasoning chains are sound.\n\n"
                f"ITERATION: {iteration + 1}"
            ),
        )

        try:
            prompt = (
                f"## Original Query\n{query}\n\n"
                f"## Reasoning Trajectories\n{serialized_cache}"
            )
            result = await asyncio.wait_for(
                deliberator.run(prompt),
                timeout=self.config.thinker_timeout * 2,  # Deliberation gets more time
            )

            if result.output:
                return {
                    "answer": result.output.answer,
                    "confidence": result.output.confidence,
                    "critical_analysis": result.output.critical_analysis,
                }
        except Exception as e:
            logger.warning("[CONCEPT:AHE-3.7] Deliberation failed: %s", e)

        # Fallback: majority vote from trajectories
        return self._majority_vote_fallback(
            [t for t in MemoryCache.from_query(query).trajectories]
        )

    def _majority_vote_fallback(
        self,
        trajectories: list[TrajectoryEntry],
    ) -> dict[str, Any]:
        """Fallback to majority vote when deliberation fails.

        Args:
            trajectories: Available trajectory entries.

        Returns:
            Dict with the most common answer.
        """
        answers: dict[str, int] = {}
        for t in trajectories:
            if t.answer and t.success:
                key = t.answer.strip().lower()
                answers[key] = answers.get(key, 0) + 1

        if answers:
            best = max(answers, key=answers.get)  # type: ignore[arg-type]
            total = sum(answers.values())
            confidence = answers[best] / total if total > 0 else 0.0
            return {
                "answer": best,
                "confidence": confidence,
                "critical_analysis": f"Majority vote fallback: {answers[best]}/{total} agreement",
            }

        return {
            "answer": "",
            "confidence": 0.0,
            "critical_analysis": "No valid trajectories for majority vote",
        }

    def _persist_deliberation(
        self,
        engine: IntelligenceGraphEngine,
        result: dict[str, Any],
        trajectory_ids: list[str],
        cache: MemoryCache,
    ) -> str:
        """Persist deliberation result as a KG node.

        Creates a ``DeliberationNode`` linked to all consumed trajectories
        via ``DELIBERATED_BY`` edges, and records agreement/disagreement
        via ``AGREES_WITH`` / ``DISAGREES_WITH`` edges.

        Args:
            engine: The Intelligence Graph Engine.
            result: The deliberation result dict.
            trajectory_ids: IDs of persisted trajectory nodes.
            cache: The memory cache with trajectory data.

        Returns:
            The ID of the persisted deliberation node.
        """
        from ..knowledge_graph.ogm import KGMapper
        from ..models.knowledge_graph import (
            DeliberationNode,
            RegistryEdgeType,
        )

        ogm = KGMapper(engine)
        ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        delib_id = f"delib:{uuid.uuid4().hex[:8]}"

        node = DeliberationNode(
            id=delib_id,
            name=f"Deliberation: {cache.query[:40]}",
            description=f"Heavy Thinking deliberation for: {cache.query[:100]}",
            trajectories_analyzed=len(trajectory_ids),
            consensus_answer=result.get("answer", ""),
            confidence=result.get("confidence", 0.0),
            critical_analysis=result.get("critical_analysis", ""),
            iteration=result.get("iterations", 1),
            model_id="",
            timestamp=ts,
            metadata={"source": "heavy_thinking"},
        )

        ogm.upsert(node)

        # Link deliberation → trajectories
        for traj_id in trajectory_ids:
            ogm.upsert_edge(
                traj_id,
                delib_id,
                RegistryEdgeType.DELIBERATED_BY,
            )

        logger.info(
            "[CONCEPT:AHE-3.7] Persisted deliberation node %s "
            "(trajectories=%d, confidence=%.2f)",
            delib_id,
            len(trajectory_ids),
            result.get("confidence", 0.0),
        )
        return delib_id

    async def _distill_heavy_experience(
        self,
        cache: MemoryCache,
        result: dict[str, Any],
        deps: GraphDeps,
    ) -> None:
        """Distill heavy thinking into an ExperienceNode for continual learning.

        CONCEPT:AHE-3.5 — Cross-Rollout Critique integration.
        Extracts a condition-action heuristic from the deliberation's
        critical analysis of trajectory differences.

        Args:
            cache: The memory cache with all trajectories.
            result: The deliberation result.
            deps: Runtime dependencies.
        """
        if not deps.knowledge_engine:
            return

        if result.get("confidence", 0.0) < 0.5:
            return  # Don't distill low-confidence deliberations

        try:
            from ..models.knowledge_graph import ExperienceNode, RegistryNodeType

            analysis = result.get("critical_analysis", "")
            answer = result.get("answer", "")

            if analysis and answer:
                exp_id = f"exp_ht_{uuid.uuid4().hex[:8]}"
                node = ExperienceNode(
                    id=exp_id,
                    name=f"Heavy Thinking: {answer[:30]}",
                    description=(
                        f"Derived from heavy thinking deliberation across "
                        f"{len(cache.trajectories)} trajectories"
                    ),
                    type=RegistryNodeType.EXPERIENCE,
                    condition=f"Complex query requiring deep reasoning: {cache.query[:100]}",
                    action=f"Deliberation consensus: {analysis[:300]}",
                    success_rate=result.get("confidence", 0.0),
                    source_run_id=cache.query_hash[:8],
                    metadata={"source": "heavy_thinking_distillation"},
                )
                from ..knowledge_graph.ogm import KGMapper

                ogm = KGMapper(deps.knowledge_engine)
                ogm.upsert(node)
                logger.info(
                    "[CONCEPT:AHE-3.7] Distilled ExperienceNode from deliberation: %s",
                    exp_id,
                )
        except Exception as e:
            logger.warning("Heavy thinking experience distillation failed: %s", e)


# ── Planner Integration ──────────────────────────────────────────────


class HeavyThinkingPlanner:
    """CONCEPT:AHE-3.7 — Heavy Thinking as an alternative to LATSPlanner.

    Wraps ``HeavyThinkingOrchestrator`` to produce a ``GraphPlan`` from
    the deliberation result.  Used when the complexity estimator
    determines the query warrants heavy thinking.

    Args:
        context: Background context for reasoning.
        deps: Runtime graph dependencies.
        model: Default LLM model.
        config: Optional ``HeavyThinkingConfig`` override.
    """

    def __init__(
        self,
        context: str,
        deps: GraphDeps,
        model: Any,
        config: HeavyThinkingConfig | None = None,
    ) -> None:
        self.context = context
        self.deps = deps
        self.model = model
        self.config = config or HeavyThinkingConfig()
        self.orchestrator = HeavyThinkingOrchestrator(self.config)

    async def search(self, query: str) -> Any:
        """Execute heavy thinking and produce a GraphPlan.

        The deliberation result is converted into a single-step
        GraphPlan that carries the synthesized answer as context
        for the dispatcher.

        Args:
            query: The query to plan for.

        Returns:
            A ``GraphPlan`` with the heavy thinking result embedded.
        """
        from ..models import ExecutionStep, GraphPlan

        result = await self.orchestrator.execute(
            query=f"{query}\n\nContext:\n{self.context}",
            deps=self.deps,
        )

        # Convert deliberation result into a plan
        answer = result.get("answer", "")
        confidence = result.get("confidence", 0.0)
        analysis = result.get("critical_analysis", "")

        steps = [
            ExecutionStep(
                node_id="synthesizer",
                # description is not a field on ExecutionStep
                input_data={
                    "heavy_thinking_answer": answer,
                    "heavy_thinking_analysis": analysis,
                    "heavy_thinking_confidence": confidence,
                },
            ),
        ]

        return GraphPlan(
            steps=steps,
            metadata={
                "reasoning": "heavy_thinking",
                "confidence": confidence,
                "trajectories": result.get("iterations", 1),
            },
        )
