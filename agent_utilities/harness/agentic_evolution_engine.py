"""Agentic Evolution Engine — Synthesized Evolution Facade.

CONCEPT:AU-AHE.harness.evolutionary-aggregation — Agentic Evolution Engine

Provides a single entry point for all evolutionary capabilities:
- Variant pool management (AHE-3.2 via ``VariantPool``)
- Skill neologism detection and evolution (ECO-4.1 via ``SkillEvolver``)

The facade enables a unified evolution cycle:
    detect_skill_gap() → create_skill() → register_variant()
    → evaluate_fitness() → tournament_select() → promote_winner()
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..knowledge_graph.core.engine import IntelligenceGraphEngine

logger = logging.getLogger(__name__)


class AgenticEvolutionEngine:
    """Synthesized evolution engine.

    CONCEPT:AU-AHE.harness.evolutionary-aggregation — Agentic Evolution Engine

    Combines variant pool management (AHE-3.2) with skill neologism
    detection and evolution (ECO-4.1) into a unified lifecycle.

    Usage::

        engine = AgenticEvolutionEngine(kg_engine)

        # Detect skill gaps
        gap = engine.detect_skill_gap("deploy kubernetes pods")
        if gap:
            skill = engine.create_skill_from_gap(gap)

        # Manage prompt/config variants
        variant_id = engine.register_variant("base_prompt_1", variant_node)
        fitness = engine.evaluate_fitness(variant_id)
        winners = engine.tournament_select("base_prompt_1")

    Args:
        engine: The IntelligenceGraphEngine for KG access.
        gap_threshold: Similarity threshold for skill gap detection.
        merge_threshold: Overlap threshold for skill merging.
    """

    def __init__(
        self,
        engine: IntelligenceGraphEngine | None = None,
        gap_threshold: float = 0.3,
        merge_threshold: float = 0.7,
    ) -> None:
        self._engine = engine
        self._variant_pool: Any = None
        self._skill_detector: Any = None
        self._skill_factory: Any = None
        self._skill_merger: Any = None
        self._memory_store: Any = None
        self._decentralized_memory: Any = None
        self._self_play: Any = None
        self._fast_slow: Any = None
        self._substrate_trainer: Any = None
        self._replay_buffer: Any = None
        self._gap_threshold = gap_threshold
        self._merge_threshold = merge_threshold
        self._initialized = False

    def _lazy_init(self) -> None:
        """Lazily initialize sub-components."""
        if self._initialized:
            return
        self._initialized = True

        # Variant pool (AHE-3.2)
        if self._engine:
            try:
                from .variant_pool import VariantPool

                self._variant_pool = VariantPool(self._engine)
            except Exception as e:
                logger.debug("VariantPool not available: %s", e)

        # Unified evolving-memory store (KG-2.1) — captures per-cycle insights.
        try:
            from .evolving_memory import EvolvingMemoryStore

            self._memory_store = EvolvingMemoryStore(engine=self._engine)
        except Exception as e:  # pragma: no cover - defensive
            logger.debug("EvolvingMemoryStore not available: %s", e)
            self._memory_store = None

        # Decentralized per-agent memory + exploit/explore bandit (KG-2.82 /
        # AHE-3.33): each evolving component keeps its OWN exploit (proven winners)
        # and explore (fresh variants) pools, and a per-agent bandit converges its
        # exploit/explore balance from cycle outcomes (DecentMem, arXiv:2605.22721).
        try:
            from .decentralized_memory import DecentralizedMemory

            self._decentralized_memory = DecentralizedMemory(engine=self._engine)
        except Exception as e:  # pragma: no cover - defensive
            logger.debug("DecentralizedMemory not available: %s", e)
            self._decentralized_memory = None

        # Prioritized replay buffer (AHE-3.0, b4-03 F4) — decisive cycles resurface.
        try:
            from .replay_buffer import PrioritizedReplayBuffer

            self._replay_buffer = PrioritizedReplayBuffer()
        except Exception as e:  # pragma: no cover - defensive
            logger.debug("PrioritizedReplayBuffer not available: %s", e)
            self._replay_buffer = None

        # Self-guided self-play (CONCEPT:AU-AHE.harness.when-task-is-scope, SGS arXiv:2604.20209): a
        # Conjecturer proposes difficulty-matched tasks, the Guide gatekeeps quality
        # (rejecting gamed/illogical conjectures — the anti-plateau mechanism), and a
        # Solver attempts them, raising a curriculum. Deterministic defaults run with
        # zero infra; an LLM-backed conjecture/solve pair can be injected later.
        try:
            from .self_guided_play import SelfGuidedSelfPlay

            def _conjecture(target: str, difficulty: float) -> str:
                return f"{target} [difficulty={difficulty:.2f}]"

            def _solve(task: str) -> tuple[str, bool]:
                # Parse the curriculum difficulty; the solver succeeds while the task
                # stays within reach, then starts failing as difficulty climbs — a
                # realistic curriculum signal for the plateau breaker.
                diff = 0.0
                if "[difficulty=" in task:
                    try:
                        diff = float(task.rsplit("[difficulty=", 1)[1].rstrip("]"))
                    except ValueError:
                        diff = 0.0
                return ("solved", diff <= 0.7)

            self._self_play = SelfGuidedSelfPlay(_conjecture, _solve)
        except Exception as e:  # pragma: no cover - defensive
            logger.debug("SelfGuidedSelfPlay not available: %s", e)
            self._self_play = None

        # Fast-Slow learning controller (FST arXiv:2605.12484, CONCEPT:AU-ORCH.execution.feed-cycle-outcome-fast).
        # Each cycle's outcome is a trace; the FAST loop updates the harness now and
        # the SLOW loop absorbs what RECURS across bases via the REAL SubstrateTrainer
        # (CONCEPT:AU-ORCH.execution.substrate-training-job-emission).
        # It builds a GRPO corpus from the recurring group and emits a training-job
        # spec to the gradient substrate (DSM/GPU); the gradient step runs in
        # data-science-mcp and is GPU-gated; jobs are recorded (queued) when no
        # substrate is reachable, never lost.
        try:
            from .fast_slow_controller import FastSlowController
            from .substrate_trainer import SubstrateTrainer

            def _harness_update(traces: Any) -> str:
                return f"harness:{len(traces)}"

            self._substrate_trainer = SubstrateTrainer()
            self._fast_slow = FastSlowController(
                _harness_update, trainer_fn=self._substrate_trainer.as_trainer_fn()
            )
        except Exception as e:  # pragma: no cover - defensive
            logger.debug("FastSlowController not available: %s", e)
            self._fast_slow = None
            self._substrate_trainer = None

        # Skill evolution (ECO-4.1)
        try:
            from ..knowledge_graph.adaptation.skill_evolver import (
                SkillFactory,
                SkillMerger,
                SkillNeologismDetector,
            )

            self._skill_detector = SkillNeologismDetector(
                gap_threshold=self._gap_threshold
            )
            self._skill_factory = SkillFactory()
            self._skill_merger = SkillMerger(merge_threshold=self._merge_threshold)
        except Exception as e:
            logger.debug("SkillEvolver not available: %s", e)

    # --- Variant Pool API (AHE-3.2) ---

    def register_variant(
        self,
        base_id: str,
        variant: Any,
        generation: int = 0,
        strategy: str = "llm",
    ) -> str:
        """Register a new variant derived from a base component."""
        self._lazy_init()
        if not self._variant_pool:
            raise RuntimeError("VariantPool requires an IntelligenceGraphEngine")
        return self._variant_pool.register_variant(
            base_id, variant, generation, strategy
        )

    def evaluate_fitness(self, variant_id: str) -> float:
        """Compute fitness score for a variant from linked outcome evaluations."""
        self._lazy_init()
        if not self._variant_pool:
            return 0.0
        return self._variant_pool.evaluate_fitness(variant_id)

    def tournament_select(
        self, base_id: str, top_k: int = 3, tournament_size: int = 2
    ) -> list[str]:
        """Select the fittest variants using tournament selection."""
        self._lazy_init()
        if not self._variant_pool:
            return []
        return self._variant_pool.tournament_select(base_id, top_k, tournament_size)

    def promote_winner(self, variant_id: str, base_id: str) -> None:
        """Promote a variant to become the new baseline."""
        self._lazy_init()
        if not self._variant_pool:
            raise RuntimeError("VariantPool requires an IntelligenceGraphEngine")
        self._variant_pool.promote_winner(variant_id, base_id)

    def prune_losers(self, base_id: str, keep: int = 3) -> int:
        """Remove underperforming variants, keeping only the top ``keep``."""
        self._lazy_init()
        if not self._variant_pool:
            return 0
        return self._variant_pool.prune_losers(base_id, keep)

    # --- Skill Evolution API (ECO-4.1) ---

    def detect_skill_gap(self, task_text: str) -> Any | None:
        """Check if the task reveals a skill gap.

        Returns:
            A SkillGap if no existing skill covers this task, else None.
        """
        self._lazy_init()
        if not self._skill_detector:
            return None
        return self._skill_detector.detect_gap(task_text)

    def create_skill_from_gap(self, gap: Any, trace_id: str = "") -> Any:
        """Create a new skill node from a detected gap."""
        self._lazy_init()
        if not self._skill_factory:
            raise RuntimeError("SkillFactory not available")
        skill = self._skill_factory.create_from_gap(gap, trace_id)
        self._record_skill(skill, "gap")
        return skill

    def create_skill_from_execution(
        self,
        task_text: str,
        result_summary: str,
        success: bool = True,
        trace_id: str = "",
    ) -> Any:
        """Create a skill from a successful execution trace."""
        self._lazy_init()
        if not self._skill_factory:
            raise RuntimeError("SkillFactory not available")
        skill = self._skill_factory.create_from_execution(
            task_text, result_summary, success, trace_id
        )
        self._record_skill(skill, "execution")
        return skill

    def find_merge_candidates(self, skills: list[Any]) -> list[Any]:
        """Find pairs of skills that may overlap."""
        self._lazy_init()
        if not self._skill_merger:
            return []
        return self._skill_merger.find_merge_candidates(skills)

    def merge_skills(self, skill_a: Any, skill_b: Any) -> Any:
        """Merge two overlapping skills into one."""
        self._lazy_init()
        if not self._skill_merger:
            raise RuntimeError("SkillMerger not available")
        merged = self._skill_merger.merge(skill_a, skill_b)
        self._record_skill(merged, "merge")
        return merged

    def _record_skill(self, skill: Any, source: str) -> None:
        """Mirror a created/merged skill into the unified EvolvingMemoryStore.

        CONCEPT:AU-KG.memory.tiered-memory-caching — routes skill evolution through the single graph-native
        SKILL bank (dedup + resolve shared with insights/workspace banks). Best-effort.
        """
        if self._memory_store is None or skill is None:
            return
        name = getattr(skill, "name", None) or getattr(skill, "title", None) or ""
        desc = getattr(skill, "description", "") or getattr(skill, "content", "")
        content = f"{name}: {desc}".strip(": ").strip()
        if not content:
            return
        try:
            from .evolving_memory import MemoryBank

            self._memory_store.add(
                MemoryBank.SKILL,
                content,
                importance=0.55,
                metadata={"source": source, "skill_id": str(getattr(skill, "id", ""))},
            )
        except Exception as e:  # pragma: no cover - best-effort
            logger.debug("skill memory record failed: %s", e)

    # --- Unified Evolution Cycle ---

    def run_evolution_cycle(
        self,
        base_id: str,
        task_text: str = "",
        top_k: int = 3,
    ) -> dict[str, Any]:
        """Run a complete evolution cycle: evaluate → select → prune.

        Optionally checks for skill gaps if task_text is provided.

        Args:
            base_id: The base component to evolve.
            task_text: Optional task to check for skill gaps.
            top_k: Number of top variants to keep.

        Returns:
            Cycle report dict with winners, pruned count, and skill gap info.
        """
        self._lazy_init()
        report: dict[str, Any] = {
            "base_id": base_id,
            "winners": [],
            "pruned": 0,
            "skill_gap": None,
        }

        # Variant evolution
        if self._variant_pool:
            winners = self.tournament_select(base_id, top_k=top_k)
            report["winners"] = winners
            report["pruned"] = self.prune_losers(base_id, keep=top_k)
            # CONCEPT:AU-AHE.harness.evolutionary-aggregation — population-drift / diversity-collapse signal
            health = self._variant_pool.population_health(base_id)
            report["population_health"] = health
            report["early_stop_recommended"] = bool(health.get("collapsed"))

            # CONCEPT:AU-KG.memory.tiered-memory-caching — capture the cycle outcome as a reusable INSIGHT.
            if self._memory_store is not None and winners:
                from .evolving_memory import MemoryBank

                insight = self._memory_store.add(
                    MemoryBank.INSIGHT,
                    f"Evolution cycle for {base_id}: promoted {winners[:3]} "
                    f"(spread={health.get('spread')}, collapsed={health.get('collapsed')})",
                    importance=0.6,
                    metadata={
                        "base_id": base_id,
                        "winners": winners,
                        "collapsed": bool(health.get("collapsed")),
                    },
                )
                report["insight_id"] = insight.id
                # b4-03 merge-generalize: converge paraphrased cycle insights.
                generalized = self._memory_store.reconcile_similar(MemoryBank.INSIGHT)
                if generalized:
                    report["insights_generalized"] = generalized

            # CONCEPT:AU-KG.memory.ahe-record-this-base / AHE-3.33 — record this base's winners as reusable
            # trajectories in ITS OWN exploitation pool, and feed the cycle outcome
            # back to its bandit: a healthy spread rewards exploitation, a collapse
            # rewards exploration so the next cycle diversifies (DecentMem routing).
            if self._decentralized_memory is not None and winners:
                from .decentralized_memory import MemoryPool

                collapsed = bool(health.get("collapsed"))
                for winner in winners[:top_k]:
                    self._decentralized_memory.record_trajectory(
                        base_id,
                        f"Promoted variant {winner} for {base_id}",
                        metadata={"cycle_base": base_id, "variant": winner},
                    )
                if collapsed:
                    self._decentralized_memory.reward(base_id, MemoryPool.EXPLORE, 1.0)
                else:
                    self._decentralized_memory.reward(base_id, MemoryPool.EXPLOIT, 1.0)
                report["decentralized_router"] = (
                    self._decentralized_memory.router_stats(base_id)
                )

            # b4-03 F4: push the cycle as a replay state, keyed by base_id so rare
            # (decisive) bases resurface preferentially for re-evaluation.
            if self._replay_buffer is not None:
                self._replay_buffer.add(
                    {
                        "base_id": base_id,
                        "winners": winners,
                        "spread": health.get("spread"),
                        "collapsed": bool(health.get("collapsed")),
                    },
                    key=base_id,
                )
                report["replay_buffer_size"] = len(self._replay_buffer)

        # CONCEPT:AU-AHE.harness.when-task-is-scope — when a task is in scope, run a short self-guided
        # self-play curriculum for it: the Guide rejects gamed conjectures so the
        # solver trains on quality tasks, and the plateau-breaker fires if progress
        # stalls. Surfaces accept/solve rates + plateau flag for the loop.
        if task_text and self._self_play is not None:
            play = self._self_play.run(task_text, rounds=6)
            report["self_play"] = {
                "accept_rate": play.accept_rate,
                "solve_rate": play.solve_rate,
                "plateaued": play.plateaued,
                "rounds": len(play.rounds),
            }

        # CONCEPT:AU-ORCH.execution.feed-cycle-outcome-fast — feed the cycle outcome to the Fast-Slow controller:
        # observe it as a trace (keyed by base so recurrence is detectable), run the
        # FAST harness update now, then a SLOW step that absorbs recurring bases (the
        # GRPO advantage spine runs; real weight training is the deferred trainer).
        if self._fast_slow is not None and self._variant_pool:
            from .fast_slow_controller import Trace

            self._fast_slow.observe(
                Trace(task_key=base_id, reward=float(health.get("spread", 0.0) or 0.0))
            )
            report["fast_harness_id"] = self._fast_slow.fast_step()
            slow = self._fast_slow.slow_step()
            if slow:
                report["slow_updates"] = [u.task_key for u in slow]

        # Skill gap detection
        if task_text and self._skill_detector:
            gap = self.detect_skill_gap(task_text)
            if gap:
                report["skill_gap"] = {
                    "task": task_text,
                    "closest_skill": gap.closest_skill,
                    "similarity": gap.similarity_score,
                    "suggested_name": gap.suggested_name,
                }

        return report

    def evolve_via_graph_search(
        self,
        task_text: str,
        *,
        num_branches: int = 3,
        num_steps: int = 10,
        coder_fn: Any = None,
        evaluate_fn: Any = None,
    ) -> dict[str, Any]:
        """Evolve a solution by Monte-Carlo GRAPH search (CONCEPT:AU-KG.retrieval.monte-carlo-graph-search, MLEvolve).

        Unlike tournament evolution, this searches a GRAPH of candidate solutions
        with cross-branch fusion edges + a global code memory. ``coder_fn`` /
        ``evaluate_fn`` default to deterministic, zero-infra implementations (used by
        tests and offline runs); the MCP ``evolve_code`` action injects an LLM-backed
        coder (CONCEPT:AU-ORCH.execution.drop-rlm-completion-client ``RLM``) for real production code evolution.
        """
        from .graph_search_evolution import GraphSearchEvolver

        def _coder(plan: str, prior_code: str | None) -> tuple[str, str]:
            base = prior_code or ""
            return (plan, f"{base}\n# step for: {plan}".strip())

        def _evaluate(code: str) -> tuple[float, bool]:
            # Deterministic proxy: a longer, more-refined solution scores marginally
            # higher. Replace evaluate_fn with a real sandboxed executor in production.
            return (float(code.count("# step") + code.count("\n")), False)

        evolver = GraphSearchEvolver(
            coder_fn or _coder,
            evaluate_fn or _evaluate,
            num_branches=num_branches,
            num_steps=num_steps,
        )
        best = evolver.run(task_text)
        return {
            "best_metric": best.metric,
            "stage": str(best.stage),
            "branch_id": best.branch_id,
            "reference_ids": best.reference_ids,
            "plan": best.plan,
        }

    def sample_replay(self, n: int = 1, *, seed: int | None = None) -> list[Any]:
        """Sample decisive past cycles to re-evaluate (CONCEPT:AU-AHE.harness.harness-evolution, b4-03 F4).

        Returns prioritized (rare/decisive) cycle states from the replay buffer —
        the daemon can re-run these instead of only fresh bases. Empty when no
        cycle has been recorded yet.
        """
        self._lazy_init()
        if self._replay_buffer is None:
            return []
        return self._replay_buffer.sample(n, seed=seed)
