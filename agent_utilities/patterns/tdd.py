#!/usr/bin/python
"""Red/Green/Refactor TDD Orchestration Pattern.

This module provides the core logic for agentic TDD, deeply integrated with
Spec-Driven Development (SDD) artifacts.
"""

import logging

from ..models import AgentDeps, Spec
from ..sdd import SDDManager
from .subagents import dispatch_subagent

logger = logging.getLogger(__name__)


async def run_tdd_cycle(
    feature_id: str,
    deps: AgentDeps,
    goal: str | None = None,
) -> str:
    """Orchestrate a full Red/Green/Refactor cycle for a feature.

    Args:
        feature_id: The ID of the feature/spec to implement.
        deps: The agent dependencies (including workspace and KG).
        goal: Optional specific goal override.

    Returns:
        A summary of the TDD cycle outcome.
    """
    manager = SDDManager(deps.workspace_path)
    spec = manager.load(Spec, feature_id=feature_id)
    if not spec:
        return (
            f"Error: Spec for '{feature_id}' not found. SDD must be initialized first."
        )

    # 1. RED Phase: Write Failing Tests
    logger.info(f"TDD [RED]: Generating failing tests for {feature_id}")
    red_result = await dispatch_subagent(
        goal=f"Write comprehensive FAILING tests for the following spec: {spec.model_dump_json()}\n"
        f"Ensure tests fail for the right reasons. Language: python.",
        deps=deps,
        name="TDD-Red-Agent",
        skill_types=["universal", "tdd-methodology"],
        system_prompt_suffix="You are an expert in writing failing tests (Red phase of TDD).",
    )

    # 2. GREEN Phase: Minimal Implementation
    logger.info(f"TDD [GREEN]: Implementing minimal code for {feature_id}")
    green_result = await dispatch_subagent(
        goal=f"Implement the minimal code to make these tests pass: {red_result}\n"
        f"Spec: {spec.model_dump_json()}",
        deps=deps,
        name="TDD-Green-Agent",
        skill_types=["universal", "tdd-methodology"],
        system_prompt_suffix="You are an expert in minimal implementation to pass tests (Green phase of TDD).",
    )

    # 3. REFACTOR Phase: Clean up
    logger.info(f"TDD [REFACTOR]: Cleaning up implementation for {feature_id}")
    refactor_result = await dispatch_subagent(
        goal=f"Refactor the following implementation while keeping the tests green: {green_result}\n"
        f"Tests: {red_result}\nSpec: {spec.model_dump_json()}",
        deps=deps,
        name="TDD-Refactor-Agent",
        skill_types=["universal", "tdd-methodology"],
        system_prompt_suffix="You are an expert in code refactoring (Refactor phase of TDD).",
    )

    # Hoard the successful cycle in the Knowledge Graph
    if deps.knowledge_engine:
        deps.knowledge_engine.store_pattern_template(
            name=f"TDD Cycle: {feature_id}",
            pattern_type="tdd_cycle",
            content=refactor_result,
            success_rate=1.0,
            tags=["tdd", "sdd", feature_id],
            metadata={"red_tests": red_result, "green_impl": green_result},
        )

    return f"TDD Cycle for '{feature_id}' completed successfully.\n\nFinal Result:\n{refactor_result}"
