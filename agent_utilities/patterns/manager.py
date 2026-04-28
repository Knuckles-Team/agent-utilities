#!/usr/bin/python
"""Unified Agentic Pattern Manager.

This module provides a centralized manager for all agentic patterns (TDD,
Manual Testing, Walkthroughs, Explanations), allowing easy access from AgentDeps.
"""

import logging
from typing import Any

from ..graph.config_helpers import emit_graph_event
from ..models import Spec
from .first_run_tests import run_first_tests
from .interactive_explanations import generate_interactive_explanation
from .manual_testing import run_manual_test_cycle
from .tdd import (
    run_tdd_cycle,
    tdd_green_phase,
    tdd_red_phase,
    tdd_refactor_phase,
)
from .walkthroughs import generate_linear_walkthrough

logger = logging.getLogger(__name__)


class PatternManager:
    """Manager for orchestrating advanced agentic patterns."""

    def __init__(self, deps: Any):
        self.deps = deps

    async def first_run_tests(self, test_command: str = "uv run pytest"):
        """Run initial tests in the workspace."""
        emit_graph_event(
            self.deps.graph_event_queue, "FIRST_TESTS_RUN", command=test_command
        )
        return await run_first_tests(self.deps.workspace_path, test_command)

    async def tdd_cycle(self, feature_id: str, goal: str | None = None):
        """Run a full Red/Green/Refactor TDD cycle."""
        return await run_tdd_cycle(feature_id, self.deps, goal)

    async def tdd_red_phase(self, spec: Spec):
        """Run the RED phase of TDD."""
        return await tdd_red_phase(spec, self.deps)

    async def tdd_green_phase(self, spec: Spec, red_result: str):
        """Run the GREEN phase of TDD."""
        return await tdd_green_phase(spec, red_result, self.deps)

    async def tdd_refactor_phase(self, spec: Spec, green_result: str, red_result: str):
        """Run the REFACTOR phase of TDD."""
        return await tdd_refactor_phase(spec, green_result, red_result, self.deps)

    async def manual_test(self, goal: str):
        """Run a manual testing/verification cycle."""
        emit_graph_event(
            self.deps.graph_event_queue, "SHOWBOAT_ARTIFACT_CREATED", goal=goal
        )
        return await run_manual_test_cycle(goal, self.deps)

    async def generate_walkthrough(self, path_or_query: str):
        """Generate a linear codebase walkthrough."""
        emit_graph_event(
            self.deps.graph_event_queue, "WALKTHROUGH_STARTED", path=path_or_query
        )
        return await generate_linear_walkthrough(path_or_query, self.deps)

    async def interactive_explain(self, goal: str, content: str):
        """Generate an interactive HTML explanation."""
        emit_graph_event(
            self.deps.graph_event_queue, "INTERACTIVE_HTML_READY", goal=goal
        )
        return await generate_interactive_explanation(goal, content, self.deps)
