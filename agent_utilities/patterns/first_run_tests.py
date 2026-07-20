#!/usr/bin/python
"""First Run Tests Orchestrator.

CONCEPT:AU-AHE.harness.evolutionary-aggregation — Agentic Engineering Patterns

This module provides tools to run existing tests in a workspace and feed the
results back to the agent (Planner/Verifier).
"""

from dataclasses import dataclass
from typing import Any

from agent_utilities.runtime.events import ErrorObservation, TestRunAction


@dataclass
class TestResult:
    """Represents the result of a test run."""

    success: bool
    output: str
    exit_code: int
    selector: str | None
    framework: str


async def run_first_tests(
    workspace: Any,
    *,
    selector: str | None = None,
    framework: str = "pytest",
    cwd: str | None = None,
) -> TestResult:
    """Run tests through the governed developer workspace.

    Args:
        workspace: A started or startable :class:`DevWorkspace`.
        selector: Optional framework-native test selector.
        framework: Registered test framework name.
        cwd: Optional workspace-relative working directory.

    Returns:
        A TestResult object containing the outcome.
    """
    if workspace is None or not callable(getattr(workspace, "act", None)):
        return TestResult(
            success=False,
            output="Error: governed developer workspace is unavailable.",
            exit_code=1,
            selector=selector,
            framework=framework,
        )

    result = await workspace.act(
        TestRunAction(
            selector=selector,
            framework=framework,
            cwd=cwd,
            timeout=600.0,
        )
    )
    if isinstance(result, ErrorObservation):
        return TestResult(
            success=False,
            output=result.message,
            exit_code=1,
            selector=selector,
            framework=framework,
        )

    exit_code = int(getattr(result, "exit_code", 1))
    report = str(getattr(result, "report", ""))
    raw = str(getattr(result, "raw", ""))
    output = report if not raw else f"{report}\n{raw}" if report else raw
    return TestResult(
        success=exit_code == 0,
        output=output,
        exit_code=exit_code,
        selector=selector,
        framework=framework,
    )
