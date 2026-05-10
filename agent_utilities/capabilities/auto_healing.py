"""Auto-Healing Evolutionary Engine.

Combines Dynamic Skill Evolution (ECO-4.8), Structured Retry Manager (AHE-3.11),
and Ontological Fallback Chains (ORCH-1.14) to permanently patch capability gaps.

Configurable and disabled by default.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


class AutoHealingEngine:
    """Intercepts tool failures and synthesizes new skills to patch them."""

    def __init__(self, skill_evolver: Any, fallback_router: Any, enabled: bool = False):
        self.enabled = enabled
        self.skill_evolver = skill_evolver
        self.fallback_router = fallback_router
        self._failure_registry: dict[str, int] = {}
        self.failure_threshold = 3

    def report_failure(self, task_name: str, error_context: str) -> None:
        """Called by the Retry Manager when a task fails repeatedly."""
        if not self.enabled:
            return

        count = self._failure_registry.get(task_name, 0) + 1
        self._failure_registry[task_name] = count

        logger.warning(f"Task {task_name} failed {count} times.")

        if count >= self.failure_threshold:
            logger.error(
                f"Threshold reached for {task_name}. Triggering auto-heal sequence."
            )
            self._trigger_evolution(task_name, error_context)
            # Reset after triggering evolution
            self._failure_registry[task_name] = 0

    def _trigger_evolution(self, task_name: str, error_context: str) -> None:
        """Synthesize a new skill or route a new fallback chain."""
        logger.info(f"Synthesizing new skill to patch gap in {task_name}...")

        # In a real scenario, this delegates to the AHE Harness to run a background agent
        new_skill_id = None
        if hasattr(self.skill_evolver, "evolve_skill"):
            new_skill_id = self.skill_evolver.evolve_skill(task_name, error_context)

        if new_skill_id and hasattr(self.fallback_router, "register_fallback"):
            logger.info(
                f"Successfully evolved {new_skill_id}. Registering fallback chain."
            )
            self.fallback_router.register_fallback(task_name, new_skill_id)
        else:
            logger.warning(f"Failed to evolve a patch for {task_name}.")
