#!/usr/bin/python
"""Red/Green/Refactor TDD Orchestration Pattern.

This module provides the core logic for agentic TDD, deeply integrated with
Spec-Driven Development (SDD) artifacts.
"""

import logging
from typing import Any

from ..models import AgentDeps, Spec
from .subagents import dispatch_subagent

logger = logging.getLogger(__name__)


async def tdd_red_phase(spec: Spec, deps: AgentDeps) -> str:
    """RED Phase: Write Failing Tests."""
    logger.info(f"TDD [RED]: Generating failing tests for {spec.feature_id}")
    return await dispatch_subagent(
        goal=f"Write comprehensive FAILING tests for the following spec: {spec.model_dump_json()}\n"
        f"Ensure tests fail for the right reasons. Language: python.",
        deps=deps,
        name="TDD-Red-Agent",
        skill_types=["universal", "tdd-methodology"],
        system_prompt_suffix="You are an expert in writing failing tests (Red phase of TDD).",
    )


async def tdd_green_phase(spec: Spec, red_result: str, deps: AgentDeps) -> str:
    """GREEN Phase: Minimal Implementation."""
    logger.info(f"TDD [GREEN]: Implementing minimal code for {spec.feature_id}")
    return await dispatch_subagent(
        goal=f"Implement the minimal code to make these tests pass: {red_result}\n"
        f"Spec: {spec.model_dump_json()}",
        deps=deps,
        name="TDD-Green-Agent",
        skill_types=["universal", "tdd-methodology"],
        system_prompt_suffix="You are an expert in minimal implementation to pass tests (Green phase of TDD).",
    )


async def tdd_refactor_phase(spec: Spec, green_result: str, red_result: str, deps: AgentDeps) -> str:
    """REFACTOR Phase: Clean up."""
    logger.info(f"TDD [REFACTOR]: Cleaning up implementation for {spec.feature_id}")
    return await dispatch_subagent(
        goal=f"Refactor the following implementation while keeping the tests green: {green_result}\n"
        f"Tests: {red_result}\nSpec: {spec.model_dump_json()}",
        deps=deps,
        name="TDD-Refactor-Agent",
        skill_types=["universal", "tdd-methodology"],
        system_prompt_suffix="You are an expert in code refactoring (Refactor phase of TDD).",
    )


async def run_tdd_cycle(
    feature_id: str,
    deps: AgentDeps,
    goal: str | None = None,
) -> str:
    """Orchestrate a full Red/Green/Refactor cycle for a feature."""
    from ..sdd import SDDManager
    manager = SDDManager(deps.workspace_path)
    spec = manager.load(Spec, feature_id=feature_id)
    if not spec:
        return f"Error: Spec for '{feature_id}' not found."

    red_result = await tdd_red_phase(spec, deps)
    green_result = await tdd_green_phase(spec, red_result, deps)
    refactor_result = await tdd_refactor_phase(spec, green_result, red_result, deps)

    if deps.knowledge_engine:
        deps.knowledge_engine.store_pattern_template(
            name=f"TDD Cycle: {feature_id}",
            pattern_type="tdd_cycle",
            content=refactor_result,
            success_rate=1.0,
            tags=["tdd", "sdd", feature_id],
            metadata={"red_tests": red_result, "green_impl": green_result},
        )

    return f"TDD Cycle for '{feature_id}' completed.\nFinal Result:\n{refactor_result}"
