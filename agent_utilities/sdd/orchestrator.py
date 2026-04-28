#!/usr/bin/python
"""SDD Orchestration Layer.

This module provides high-level orchestration for Spec-Driven Development,
integrating TDD, manual testing, and documentation patterns.
"""

import logging
from collections.abc import Callable
from typing import Any

from ..models import AgentDeps, Spec
from . import SDDManager

logger = logging.getLogger(__name__)


class SDDOrchestrator:
    """Orchestrates the SDD workflow with integrated agentic patterns."""

    def __init__(
        self, deps: AgentDeps, spec_generator: Callable[[str], Any] | None = None
    ):
        self.deps = deps
        self.manager = SDDManager(deps.workspace_path)
        self.spec_generator = spec_generator

    async def run_sdd(self, goal: str):
        """Execute the full SDD workflow for a given goal.

        Phases:
        1. First Run Tests: Check baseline.
        2. Spec Generation: Define what to build.
        3. TDD RED: Write failing tests.
        4. Implementation (existing plan/execute flow).
        5. TDD GREEN: Pass the tests.
        6. TDD REFACTOR: Clean up.
        7. Optional: Documentation & Explanations.
        """
        logger.info(f"Starting SDD workflow for goal: {goal}")

        # 1. Baseline Tests
        if self.deps.patterns:
            logger.info("SDD: Running baseline tests...")
            await self.deps.patterns.first_run_tests()

        # 2. Spec Generation
        if self.spec_generator is None:
            logger.warning(
                "No spec_generator provided to SDDOrchestrator. Skipping spec phase."
            )
            spec = Spec(feature_id="default", title=goal, user_stories=[])
        else:
            logger.info("SDD: Generating specification...")
            spec = await self.spec_generator(goal)

        # 3. TDD RED Phase
        red_tests = ""
        if self.deps.patterns:
            logger.info("SDD: TDD RED Phase - writing failing tests...")
            red_tests = await self.deps.patterns.tdd_red_phase(spec)

        # 4. Implementation (Plan & Execute)
        # Note: In a full implementation, this would involve the Planner/Executor agents.
        # Here we bridge to the TDD GREEN phase which handles implementation.

        # 5. TDD GREEN Phase
        green_impl = ""
        if self.deps.patterns:
            logger.info("SDD: TDD GREEN Phase - implementing minimal code...")
            green_impl = await self.deps.patterns.tdd_green_phase(spec, red_tests)

        # 6. TDD REFACTOR Phase
        refactored = green_impl
        if self.deps.patterns:
            logger.info("SDD: TDD REFACTOR Phase - cleaning up...")
            refactored = await self.deps.patterns.tdd_refactor_phase(
                spec, green_impl, red_tests
            )

        # 7. Optional Phases
        if self.deps.patterns:
            if "explain" in goal.lower():
                logger.info("SDD: Generating interactive explanation...")
                await self.deps.patterns.interactive_explain(goal, refactored)

            if "document" in goal.lower() or "walkthrough" in goal.lower():
                logger.info("SDD: Generating codebase walkthrough...")
                await self.deps.patterns.generate_walkthrough(
                    str(self.deps.workspace_path)
                )

        logger.info(f"SDD workflow completed for '{spec.feature_id}'.")
        return refactored
