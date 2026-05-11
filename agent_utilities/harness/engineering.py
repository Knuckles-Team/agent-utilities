#!/usr/bin/python
from __future__ import annotations

"""Agentic Engineering Patterns — Facade Module.

CONCEPT:AHE-3.2 — Agentic Engineering Patterns

This module provides a unified entry point for all agentic engineering
patterns that are implemented across the ``patterns/`` sub-package.
It consolidates TDD cycles, first-run tests, manual testing, code
walkthroughs, and interactive explanations into a single importable
facade::

    from agent_utilities.harness.engineering import (
        EngineeringPatternOrchestrator,
        PatternType,
    )

    orchestrator = EngineeringPatternOrchestrator(workspace_path="/path")
    result = await orchestrator.execute(PatternType.TDD, spec_id="feature-1")

Patterns
--------
- **TDD Cycle** — Red/Green/Refactor via :mod:`agent_utilities.patterns.tdd`
- **First Run Tests** — Baseline establishment via :mod:`agent_utilities.patterns.first_run_tests`
- **Manual Testing** — Exploratory verification via :mod:`agent_utilities.patterns.manual_testing`
- **Code Walkthroughs** — Linear documentation via :mod:`agent_utilities.patterns.walkthroughs`
- **Interactive Explanations** — HTML/JS artifact generation via :mod:`agent_utilities.patterns.interactive_explanations`

See Also:
    - :mod:`agent_utilities.patterns` — Individual pattern implementations
    - :mod:`agent_utilities.patterns.tdd` — Red/Green/Refactor cycle
"""


import logging
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

logger = logging.getLogger(__name__)


class PatternType(StrEnum):
    """Enumeration of supported agentic engineering patterns.

    CONCEPT:AHE-3.2 — Agentic Engineering Patterns
    """

    TDD = "tdd"
    FIRST_RUN_TESTS = "first_run_tests"
    MANUAL_TESTING = "manual_testing"
    CODE_WALKTHROUGH = "code_walkthrough"
    INTERACTIVE_EXPLANATION = "interactive_explanation"


@dataclass
class PatternResult:
    """Result of executing an engineering pattern.

    CONCEPT:AHE-3.2 — Agentic Engineering Patterns

    Attributes:
        pattern: The pattern type that was executed.
        success: Whether the pattern completed successfully.
        output: The primary output (test results, walkthrough, etc.).
        artifacts: Paths to any generated artifact files.
        metadata: Additional pattern-specific metadata.
        error: Error message if the pattern failed.
    """

    pattern: PatternType
    success: bool = True
    output: str = ""
    artifacts: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


class EngineeringPatternOrchestrator:
    """Orchestrates agentic engineering patterns for a workspace.

    CONCEPT:AHE-3.2 — Agentic Engineering Patterns

    Provides a unified interface to execute TDD cycles, first-run tests,
    manual testing sessions, code walkthroughs, and interactive explanations
    within a given workspace.

    The orchestrator delegates to the individual pattern modules in
    :mod:`agent_utilities.patterns` and enriches results with KG
    persistence when a Knowledge Graph engine is available.

    Args:
        workspace_path: Absolute path to the project workspace root.

    Example::

        orchestrator = EngineeringPatternOrchestrator("/home/user/project")
        result = await orchestrator.execute(
            PatternType.TDD,
            spec_id="feature-auth-login",
        )
        print(f"TDD {'passed' if result.success else 'failed'}: {result.output}")
    """

    def __init__(self, workspace_path: str) -> None:
        self.workspace_path = workspace_path
        self._pattern_registry: dict[PatternType, str] = {
            PatternType.TDD: "agent_utilities.patterns.tdd",
            PatternType.FIRST_RUN_TESTS: "agent_utilities.patterns.first_run_tests",
            PatternType.MANUAL_TESTING: "agent_utilities.patterns.manual_testing",
            PatternType.CODE_WALKTHROUGH: "agent_utilities.patterns.walkthroughs",
            PatternType.INTERACTIVE_EXPLANATION: "agent_utilities.patterns.interactive_explanations",
        }

    async def execute(
        self,
        pattern: PatternType,
        *,
        spec_id: str | None = None,
        target_path: str | None = None,
        deps: Any | None = None,
        **kwargs: Any,
    ) -> PatternResult:
        """Execute an engineering pattern.

        CONCEPT:AHE-3.2 — Agentic Engineering Patterns

        Dispatches to the appropriate pattern module and returns a
        structured result.

        Args:
            pattern: The pattern type to execute.
            spec_id: Feature/spec identifier (required for TDD).
            target_path: Target file or directory (for walkthroughs).
            deps: ``AgentDeps`` instance for agent-backed patterns.
            **kwargs: Additional pattern-specific arguments.

        Returns:
            :class:`PatternResult` with execution outcome.
        """
        logger.info(
            "Executing engineering pattern %s for workspace %s",
            pattern,
            self.workspace_path,
        )

        try:
            if pattern == PatternType.TDD:
                return await self._execute_tdd(spec_id=spec_id, deps=deps, **kwargs)
            elif pattern == PatternType.FIRST_RUN_TESTS:
                return await self._execute_first_run(target_path=target_path, **kwargs)
            elif pattern == PatternType.MANUAL_TESTING:
                return await self._execute_manual_test(
                    target_path=target_path, **kwargs
                )
            elif pattern == PatternType.CODE_WALKTHROUGH:
                return await self._execute_walkthrough(
                    target_path=target_path, **kwargs
                )
            elif pattern == PatternType.INTERACTIVE_EXPLANATION:
                return await self._execute_interactive(
                    target_path=target_path, **kwargs
                )
            else:
                return PatternResult(
                    pattern=pattern,
                    success=False,
                    error=f"Unknown pattern type: {pattern}",
                )
        except Exception as exc:
            logger.error("Pattern %s failed: %s", pattern, exc)
            return PatternResult(
                pattern=pattern,
                success=False,
                error=str(exc),
            )

    async def _execute_tdd(
        self,
        spec_id: str | None = None,
        deps: Any | None = None,
        **kwargs: Any,
    ) -> PatternResult:
        """Execute a Red/Green/Refactor TDD cycle."""
        if not spec_id:
            return PatternResult(
                pattern=PatternType.TDD,
                success=False,
                error="spec_id is required for TDD pattern",
            )

        from agent_utilities.patterns.tdd import run_tdd_cycle

        result = await run_tdd_cycle(
            feature_id=spec_id,
            deps=deps,  # type: ignore
            **kwargs,
        )
        return PatternResult(
            pattern=PatternType.TDD,
            success=True,
            output=result,
            metadata={"spec_id": spec_id},
        )

    async def _execute_first_run(
        self,
        target_path: str | None = None,
        **kwargs: Any,
    ) -> PatternResult:
        """Execute a first-run test baseline."""
        from pathlib import Path

        from agent_utilities.patterns.first_run_tests import (
            TestResult,
            run_first_tests,
        )

        path = target_path or self.workspace_path
        result: TestResult = await run_first_tests(Path(path), **kwargs)
        return PatternResult(
            pattern=PatternType.FIRST_RUN_TESTS,
            success=result.success,
            output=result.output,
            metadata={
                "exit_code": result.exit_code,
                "command": result.command,
            },
        )

    async def _execute_manual_test(
        self,
        target_path: str | None = None,
        **kwargs: Any,
    ) -> PatternResult:
        """Execute a manual testing session."""
        from agent_utilities.patterns.manual_testing import (
            run_manual_test_cycle,
        )

        goal = kwargs.get("goal", f"Test {target_path}")
        deps = kwargs.get("deps")
        result = await run_manual_test_cycle(goal, deps)
        return PatternResult(
            pattern=PatternType.MANUAL_TESTING,
            success=True,
            output=result,
            metadata={"target_path": target_path},
        )

    async def _execute_walkthrough(
        self,
        target_path: str | None = None,
        **kwargs: Any,
    ) -> PatternResult:
        """Generate a code walkthrough."""
        from agent_utilities.patterns.walkthroughs import generate_linear_walkthrough

        path = target_path or self.workspace_path
        deps = kwargs.get("deps")
        output = await generate_linear_walkthrough(path, deps)
        return PatternResult(
            pattern=PatternType.CODE_WALKTHROUGH,
            success=True,
            output=output,
            metadata={"target_path": path},
        )

    async def _execute_interactive(
        self,
        target_path: str | None = None,
        **kwargs: Any,
    ) -> PatternResult:
        """Generate an interactive explanation artifact."""
        from agent_utilities.patterns.interactive_explanations import (
            generate_interactive_explanation,
        )

        path = target_path or self.workspace_path
        goal = kwargs.get("goal", f"Explain {path}")
        deps = kwargs.get("deps")
        output = await generate_interactive_explanation(
            explanation_goal=goal, content_to_explain=path, deps=deps
        )
        return PatternResult(
            pattern=PatternType.INTERACTIVE_EXPLANATION,
            success=True,
            output=output,
            artifacts=[],
            metadata={"target_path": path},
        )

    def list_available_patterns(self) -> list[dict[str, str]]:
        """List all available engineering patterns.

        Returns:
            List of dicts with ``name``, ``module``, and ``description`` keys.
        """
        descriptions = {
            PatternType.TDD: "Red/Green/Refactor TDD cycle with SDD integration",
            PatternType.FIRST_RUN_TESTS: "Baseline test suite establishment",
            PatternType.MANUAL_TESTING: "Structured exploratory verification",
            PatternType.CODE_WALKTHROUGH: "Linear code documentation generation",
            PatternType.INTERACTIVE_EXPLANATION: "Interactive HTML/JS artifact generation",
        }
        return [
            {
                "name": p.value,
                "module": self._pattern_registry[p],
                "description": descriptions.get(p, ""),
            }
            for p in PatternType
        ]


__all__ = [
    "EngineeringPatternOrchestrator",
    "PatternResult",
    "PatternType",
]
