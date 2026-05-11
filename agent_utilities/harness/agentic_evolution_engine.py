"""Agentic Evolution Engine — Consolidated Evolution Facade.

CONCEPT:AHE-3.2 — Agentic Evolution Engine

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
    """Consolidated evolution engine.

    CONCEPT:AHE-3.2 — Agentic Evolution Engine

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
        return self._skill_factory.create_from_gap(gap, trace_id)

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
        return self._skill_factory.create_from_execution(
            task_text, result_summary, success, trace_id
        )

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
        return self._skill_merger.merge(skill_a, skill_b)

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
