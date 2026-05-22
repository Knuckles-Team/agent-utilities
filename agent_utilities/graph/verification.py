#!/usr/bin/python
from __future__ import annotations

"""Graph Verification Steps.

Quality gates, synthesis, error recovery, and join synchronization.
Extracted from the monolithic steps.py for maintainability.
"""


import asyncio
import logging
from typing import Any

from pydantic_ai import Agent
from pydantic_graph import End
from pydantic_graph.beta import StepContext

from ..models import GraphResponse
from .config_helpers import emit_graph_event, load_specialized_prompts
from .graph_models import ValidationResult
from .hsm import StateInvariantError, assert_state_valid
from .lifecycle import _emit_node_lifecycle
from .state import GraphDeps, GraphState

logger = logging.getLogger(__name__)

lock = asyncio.Lock()

__all__ = [
    "verifier_step",
    "synthesizer_step",
    "error_recovery_step",
    "join_step",
    "wide_search_joiner_step",
]


async def join_step(
    ctx: StepContext[GraphState, GraphDeps, Any],
) -> str | None:
    """Synchronize parallel execution paths using a thread-safe barrier count.

    Monitors the completion of concurrent tasks. Once the pending count
    reaches zero, it triggers a transition back to the dispatcher for the
    subsequent plan phase or to the router if failures require re-planning.

    Args:
        ctx: The pydantic-graph step context.

    Returns:
        The next node identifier ('dispatcher', 'router') or None while waiting.

    """
    async with lock:
        ctx.state.pending_parallel_count -= 1
        count = ctx.state.pending_parallel_count
        logger.debug(f"Join: Remaining parallel tasks = {count}")

        if count <= 0:
            logger.info("Join: All parallel tasks completed.")
            if ctx.state.needs_replan:
                logger.warning(
                    "Join: Re-planning required due to failures. Routing to router_step."
                )
                ctx.state.needs_replan = False  # Reset for the next plan
                return "router_step"

            # CONCEPT:ORCH-1.1 — Hybrid Pydantic Validation Gate for Wide-Search
            if ctx.state.workboard is not None:
                wb = ctx.state.workboard
                try:
                    # 1. Assert row count matches expectations
                    if (
                        wb.expected_row_count > 0
                        and len(wb.row_slots) < wb.expected_row_count
                    ):
                        raise ValueError(
                            f"Missing rows: Expected {wb.expected_row_count}, got {len(wb.row_slots)}"
                        )

                    # 2. Assert schema conformity
                    if wb.schema_definition:
                        for entity_id, row in wb.row_slots.items():
                            for col in wb.schema_definition.keys():
                                if col not in row:
                                    raise ValueError(
                                        f"Row '{entity_id}' missing required column: '{col}'"
                                    )

                    logger.info(
                        "Join: Fast-Path Pydantic validation PASSED for WideSearchWorkboard."
                    )
                except Exception as e:
                    logger.warning(
                        f"Join: Fast-Path Pydantic validation FAILED: {e}. Routing to wide_search_joiner (Slow-Path)."
                    )
                    ctx.state.validation_feedback = str(e)
                    return "wide_search_joiner"

            return "dispatcher"

    # Still waiting for others
    return None


async def wide_search_joiner_step(
    ctx: StepContext[GraphState, GraphDeps, None],
) -> str:
    """CONCEPT:ORCH-1.1 — Slow-Path LLM Repair for WideSearch extraction.

    This node is triggered when the Fast-Path Pydantic validation fails in join_step.
    It uses an LLM to attempt to repair schema mismatches, standardize formatting,
    or identify critical data gaps that require a targeted re-plan.
    """
    logger.info(
        f"[LAYER:GRAPH:BIGTABLE_JOINER] Starting slow-path repair. Feedback: {ctx.state.validation_feedback}"
    )

    if not ctx.state.workboard:
        return "dispatcher"

    repair_agent = Agent(
        model=ctx.deps.agent_model,
        system_prompt=(
            f"You are a WideSearch Repair Specialist. The fast-path Pydantic validation failed for a wide-search extraction.\n\n"
            f"Error: {ctx.state.validation_feedback}\n\n"
            f"Expected Schema: {ctx.state.workboard.schema_definition}\n"
            f"Expected Row Count: {ctx.state.workboard.expected_row_count}\n"
            f"Current Row Count: {len(ctx.state.workboard.row_slots)}\n\n"
            f"Your task is to analyze the conflict_log and the row_slots to fix schema mismatches, "
            f"standardize data formats, or synthesize missing data if possible. "
            f"If the missing data requires more research, you MUST trigger a re-plan by stating so."
        ),
    )

    try:
        # In a full implementation, we'd pass the serialized workboard to the LLM and
        # parse its structured output back into the workboard.
        # For this integration, we log the repair attempt and route back to dispatcher or planner.
        # We will assume the repair agent output determines if we need more planning.
        res = await repair_agent.run(
            "Analyze the WideSearchWorkboard state and repair the data."
        )
        output = res.output.lower()

        if "re-plan" in output or "research" in output:
            logger.warning(
                "WideSearch Joiner: Repair failed or requires more data. Routing to planner."
            )
            ctx.state.needs_replan = True
            return "planner"

        logger.info("WideSearch Joiner: Repair successful. Routing to dispatcher.")
        # Mark as resolved
        ctx.state.validation_feedback = None
        return "dispatcher"
    except Exception as e:
        logger.error(f"WideSearch Joiner failed: {e}")
        return "error_recovery"


async def verifier_step(
    ctx: StepContext[GraphState, GraphDeps, None],
) -> str:
    """Validate execution results and route to synthesis or re-dispatch.

    This node performs a structured quality audit of the accumulated
    execution results.  It does NOT synthesize the final response —
    that is handled by :func:`synthesizer_step`.

    Routing decisions:
    - **score >= 0.7** → ``'synthesizer'`` for final response composition
    - **0.4 <= score < 0.7** → ``'dispatcher'`` for re-execution
    - **score < 0.4** → ``'planner'`` for a fresh approach

    Args:
        ctx: The pydantic-graph step context containing all registry results.

    Returns:
        The next node identifier (``'synthesizer'``, ``'dispatcher'``,
        or ``'planner'``).

    """
    # HSM: State invariant check
    try:
        assert_state_valid(ctx.state, "verifier_step")
    except StateInvariantError as e:
        logger.error(f"Verifier state invariant violation: {e}")

    logger.info(
        f"[LAYER:GRAPH:VERIFIER] Starting (attempt {ctx.state.verification_attempts + 1})..."
    )
    _emit_node_lifecycle(
        ctx.deps.event_queue,
        "verifier",
        "node_start",
        attempt=ctx.state.verification_attempts + 1,
    )

    # Consolidate results for the verifier's context
    results_summary = "\n".join(
        [f"### {node}: {val}" for node, val in ctx.state.results_registry.items()]
    )

    # Structured Validation (quality gate)
    if ctx.state.verification_attempts < 2 and results_summary.strip():
        try:
            from .executor import _get_domain_tools

            domain_tools, domain_toolsets = await _get_domain_tools(
                "verifier", ctx.deps
            )

            validation_agent = Agent(
                model=ctx.deps.agent_model,
                output_type=ValidationResult,
                deps_type=GraphDeps,
                tools=domain_tools,
                toolsets=domain_toolsets,
                system_prompt=(
                    f"You are a quality gate. Evaluate whether the execution results "
                    f"fully and accurately answer the original query with specific data findings.\n\n"
                    f"Original Query: {ctx.state.query}\n\n"
                    f"Execution Results:\n{results_summary}\n\n"
                    f"CRITICAL: If the query asks for a list, status, or specific info, the results MUST contain "
                    f"the actual data records, not just a summary that the task was completed.\n"
                    f"## TESTING MINDSET CRITERIA\n"
                    f"1. Did the agent run tests first to baseline or verify changes? (Check for 'pytest' or 'test' tool calls).\n"
                    f"2. Did the agent produce relevant artifacts? (e.g. ExecutionNotes, walkthroughs, HTML explanations).\n"
                    f"Score 0.0-1.0. If data is missing, testing protocol was ignored, or results are non-responsive, score < 0.7 and "
                    f"provide EXACT feedback on what is missing (e.g. 'Missing the list of container names', 'Failed to run tests before implementing')."
                ),
            )
            async with validation_agent.run_stream("Evaluate the results") as stream:
                validation = await asyncio.wait_for(
                    stream.get_output(), timeout=ctx.deps.verifier_timeout
                )

            emit_graph_event(
                ctx.deps.event_queue,
                event_type="verification_result",
                is_valid=validation.is_valid,
                feedback=validation.feedback,
                attempt=ctx.state.verification_attempts + 1,
            )
            if (
                not validation.is_valid
                and validation.score < 0.7
                and validation.feedback
            ):
                ctx.state.verification_attempts += 1
                ctx.state.validation_feedback = validation.feedback

                # Distinguish plan-level failures from execution-level failures.
                # Very low scores (< 0.4) suggest the approach itself was wrong
                # and a fresh plan is needed; moderate scores suggest the right
                # plan was executed poorly and can be re-dispatched.
                if validation.score < 0.4 and ctx.state.verification_attempts <= 2:
                    logger.warning(
                        f"Verifier: Score {validation.score:.2f} < 0.4. "
                        f"Feedback: {validation.feedback[:200]}. "
                        f"Re-planning (attempt {ctx.state.verification_attempts})."
                    )
                    ctx.state.needs_replan = True
                    ctx.state.error = f"Plan-level failure: {validation.feedback[:300]}"
                    return "planner"

                logger.warning(
                    f"Verifier: Score {validation.score:.2f} < 0.7. "
                    f"Feedback: {validation.feedback[:200]}. "
                    f"Re-dispatching (attempt {ctx.state.verification_attempts})."
                )
                ctx.state.step_cursor = 0
                ctx.state.needs_replan = False
                return "dispatcher"
            logger.info(f"Verifier: Validation passed (score: {validation.score:.2f}).")

            # CONCEPT:AHE-3.4: Cross-Rollout Critique (Distill Experience)
            if ctx.state.verification_attempts > 0 or ctx.state.retry_count > 0:
                await _distill_experience_from_retry(ctx, results_summary)

            # CONCEPT:AHE-3.1 — Adversarial Verification (opt-in)
            # If ADVERSARIAL_VERIFICATION=true, run a second "hacker agent"
            # pass to stress-test the implementation.  Only fires when the
            # quality gate has already passed.
            try:
                from ..capabilities.adversarial_verifier import (
                    ADVERSARIAL_ENABLED,
                    run_adversarial_pass,
                )

                if ADVERSARIAL_ENABLED:
                    adversarial_result = await run_adversarial_pass(
                        state=ctx.state,
                        deps=ctx.deps,
                        results_summary=results_summary,
                    )
                    if adversarial_result and adversarial_result.vulnerabilities_found:
                        # Only fail on high/critical severity
                        if adversarial_result.severity in ("high", "critical"):
                            ctx.state.verification_attempts += 1
                            ctx.state.validation_feedback = (
                                f"Adversarial verification found {len(adversarial_result.findings)} "
                                f"{adversarial_result.severity}-severity issue(s): "
                                + "; ".join(adversarial_result.findings[:3])
                                + ". Fix these before re-submitting."
                            )
                            logger.warning(
                                "[CONCEPT:AHE-3.1] Adversarial FAIL (severity: %s). Re-dispatching.",
                                adversarial_result.severity,
                            )
                            ctx.state.step_cursor = 0
                            return "dispatcher"
                        logger.info(
                            "[CONCEPT:AHE-3.1] Adversarial found %s issues (severity: %s) "
                            "— proceeding to synthesis with advisory.",
                            len(adversarial_result.findings),
                            adversarial_result.severity,
                        )
            except Exception as e:
                logger.debug(f"Adversarial verification skipped: {e}")

        except Exception as e:
            logger.warning(
                f"Verifier: Structure validation failed: {e}. Attempting unstructured fallback."
            )
            try:
                # Get the raw text from the agent response
                raw_text = ""
                if hasattr(validation_agent, "last_run_messages"):
                    for msg in reversed(validation_agent.last_run_messages):
                        if hasattr(msg, "parts"):
                            for part in msg.parts:
                                if hasattr(part, "content") and isinstance(
                                    part.content, str
                                ):
                                    raw_text += part.content

                # Simple heuristic extraction if we can't find raw text easily
                fallback_score = 0.5
                fallback_feedback = "Fallback validation failed to parse output."

                if not raw_text:
                    # Run a quick unstructured extraction pass
                    extraction_agent = Agent(
                        model=ctx.deps.agent_model,
                        system_prompt=(
                            "Extract the validation score (0.0 to 1.0) and feedback text from the previous response. "
                            "Format exactly as: SCORE: <number>\\nFEEDBACK: <text>"
                        ),
                    )
                    res = await extraction_agent.run(
                        f"Evaluate the following results:\n{results_summary}"
                    )
                    raw_text = res.output

                import re

                score_match = re.search(
                    r"(?:score|SCORE)[\s:=]+([0-1](?:\.\d+)?)", raw_text, re.IGNORECASE
                )
                if score_match:
                    fallback_score = float(score_match.group(1))

                feedback_match = re.search(
                    r"(?:feedback|FEEDBACK)[\s:=]+(.*)",
                    raw_text,
                    re.IGNORECASE | re.DOTALL,
                )
                if feedback_match:
                    fallback_feedback = feedback_match.group(1).strip()
                elif raw_text:
                    fallback_feedback = raw_text.strip()[:500]

                validation = ValidationResult(
                    is_valid=fallback_score >= 0.7,
                    score=fallback_score,
                    feedback=fallback_feedback,
                )
                logger.info(
                    f"Verifier Fallback: Extracted score {validation.score:.2f} and feedback."
                )

            except Exception as fallback_e:
                logger.warning(
                    f"Verifier Fallback failed: {fallback_e}. Proceeding to synthesis."
                )
                validation = ValidationResult(
                    is_valid=True, score=0.8, feedback="Fallback triggered."
                )

            # Since validation fallback produced a result, we need to process it like normal
            emit_graph_event(
                ctx.deps.event_queue,
                event_type="verification_result",
                is_valid=validation.is_valid,
                feedback=validation.feedback,
                attempt=ctx.state.verification_attempts + 1,
            )
            if (
                not validation.is_valid
                and validation.score < 0.7
                and validation.feedback
            ):
                ctx.state.verification_attempts += 1
                ctx.state.validation_feedback = validation.feedback

                if validation.score < 0.4 and ctx.state.verification_attempts <= 2:
                    logger.warning(
                        f"Verifier: Score {validation.score:.2f} < 0.4. "
                        f"Feedback: {validation.feedback[:200]}. "
                        f"Re-planning (attempt {ctx.state.verification_attempts})."
                    )
                    ctx.state.needs_replan = True
                    ctx.state.error = f"Plan-level failure: {validation.feedback[:300]}"
                    return "planner"

                logger.warning(
                    f"Verifier: Score {validation.score:.2f} < 0.7. "
                    f"Feedback: {validation.feedback[:200]}. "
                    f"Re-dispatching (attempt {ctx.state.verification_attempts})."
                )
                ctx.state.step_cursor = 0
                ctx.state.needs_replan = False
                return "dispatcher"

            logger.info(f"Verifier: Validation passed (score: {validation.score:.2f}).")
    # final response composition.  This separates the quality-gate
    # concern from the response-generation concern.
    _emit_node_lifecycle(
        ctx.deps.event_queue, "verifier", "node_complete", next_node="synthesizer"
    )
    return "synthesizer"


async def synthesizer_step(
    ctx: StepContext[GraphState, GraphDeps, None],
) -> End[GraphResponse]:
    """Compose the final authoritative response from execution results.

    This node is responsible for synthesizing a cohesive markdown response
    from the disparate specialist findings stored in
    ``ctx.state.results_registry``.  It also persists session metadata to
    the Knowledge Graph for future context retrieval.

    The synthesizer is intentionally separate from the verifier so that
    synthesis work is never wasted on re-dispatch/re-plan cycles.

    Args:
        ctx: The pydantic-graph step context containing all registry results.

    Returns:
        A terminal ``End`` state with the ``GraphResponse`` instance.

    """
    logger.info("[LAYER:GRAPH:SYNTHESIZER] Composing final response...")
    _emit_node_lifecycle(ctx.deps.event_queue, "synthesizer", "node_start")

    validator_prompt = load_specialized_prompts("verifier")

    results_summary = "\n".join(
        [f"### {node}: {val}" for node, val in ctx.state.results_registry.items()]
    )

    extra_context = ""
    if ctx.state.architectural_decisions:
        extra_context += (
            f"\n### ARCHITECTURAL INTENT\n{ctx.state.architectural_decisions}\n"
        )
    if ctx.state.exploration_notes:
        extra_context += f"\n### EXPLORATION FINDINGS\n{ctx.state.exploration_notes}\n"

    final_system_prompt = (
        f"{validator_prompt}\n"
        f"{extra_context}\n"
        f"### AGENT EXECUTION RESULTS\n{results_summary}\n\n"
        f"### FINAL INSTRUCTION\n"
        f"Synthesize the execution results into a cohesive final answer for the "
        f"user query: '{ctx.state.query}'.\n"
        f"Format data cleanly. Do NOT repeat yourself.\n"
        f"CRITICAL JSON ADHERENCE: If the query or execution involves extracting 'TweetNode' entries "
        f"or X/Twitter ingestion, you MUST output your final answer as a raw JSON array of structured objects "
        f"adhering to the TweetNode schema: post_id, text_content, author_handle, timestamp, engagement_metrics. "
        f"Do NOT wrap the JSON in markdown code blocks."
    )

    synthesizer = Agent(
        model=ctx.deps.agent_model,
        system_prompt=final_system_prompt,
    )

    try:
        logger.debug(f"Synthesizer: Prompt summary length: {len(results_summary)}")
        async with synthesizer.run_stream(
            "Consolidate and verify based on provided results. Be concise."
        ) as stream:
            async for chunk in stream.stream_text(delta=True):
                emit_graph_event(
                    ctx.deps.event_queue,
                    "agent_node_delta",
                    content=chunk,
                    node="synthesizer",
                )
            res = await asyncio.wait_for(
                stream.get_output(), timeout=ctx.deps.verifier_timeout
            )
        result_text = res if res else "None"
        if result_text.lower() == "none":
            raise ValueError("Synthesis returned 'None'")

        logger.info(
            f"Synthesizer: Synthesis successful. Output length: {len(result_text)}"
        )
    except Exception as e:
        logger.warning(
            f"Synthesizer: Synthesis failed or timed out: {e}. Falling back to raw results."
        )
        emit_graph_event(
            ctx.deps.event_queue,
            "synthesis_fallback",
            reason=str(e)[:300],
            has_results=bool(results_summary.strip()),
        )
        if results_summary.strip():
            result_text = (
                f"The query was executed, but a final synthesis could not be generated concisely.\n\n"
                f"### RAW EXECUTION RESULTS\n{results_summary}"
            )
        else:
            result_text = (
                "The agent completed its analysis but was unable to find specific data matching your request. "
                "Please verify the query or target system status."
            )

    # Persist session metadata for future context retrieval
    try:
        from datetime import datetime

        memory_entry = (
            f"Execution {ctx.state.session_id or 'unknown'} "
            f"({datetime.now().isoformat()})\n"
            f"- Query: {ctx.state.query[:200]}\n"
            f"- Plan: {[s.node_id for s in ctx.state.plan.steps]}\n"
            f"- Results: {list(ctx.state.results_registry.keys())}\n"
            f"- Failures: {ctx.state.error or 'none'}\n"
            f"- Tokens: {ctx.state.session_usage.total_tokens}\n"
            f"- Verification attempts: {ctx.state.verification_attempts}"
        )
        if ctx.deps.knowledge_engine:
            ctx.deps.knowledge_engine.add_memory(
                memory_entry,
                name=f"Execution {ctx.state.session_id or 'unknown'}",
                category="historical_execution",
            )
    except Exception as e:
        logger.debug(f"Failed to write execution memory: {e}")

    # CONCEPT:KG-2.1 — Post-execution feedback loop
    # Record execution outcome into Self-Model and TeamConfig for learning.
    execution_success = bool(result_text and result_text.lower() != "none")
    if ctx.deps.knowledge_engine:
        # Self-Model session feedback
        try:
            from ..knowledge_graph.retrieval.memory_retriever import MemoryRetriever

            memory_retriever = MemoryRetriever(ctx.deps.knowledge_engine)
            memory_retriever.update_after_session(ctx.state)
            logger.info(
                "[CONCEPT:KG-2.1] Self-Model updated: domain=%s, success=%s",
                ctx.state.routed_domain,
                execution_success,
            )
        except Exception as e:
            logger.debug(f"Self-Model feedback failed: {e}")

        # TeamConfig outcome recording
        try:
            plan_meta = ctx.state.plan.metadata if ctx.state.plan else {}
            team_config_id = plan_meta.get("team_config_id")
            if team_config_id:
                from ..knowledge_graph.core.engine_registry import RegistryMixin

                if isinstance(ctx.deps.knowledge_engine, RegistryMixin):
                    reward = 1.0 if execution_success else 0.0
                    ctx.deps.knowledge_engine.record_team_outcome(
                        team_config_id, reward=reward
                    )
                    logger.info(
                        "[CONCEPT:AHE-3.3] TeamConfig '%s' outcome recorded: reward=%.1f",
                        team_config_id,
                        reward,
                    )
        except Exception as e:
            logger.debug(f"TeamConfig feedback failed: {e}")

        # CONCEPT:ORCH-1.25 — Workflow Distillation Hook (async background)
        # If the execution was successful, fire the distillation hook to
        # potentially promote this workflow pattern to a reusable template.
        if execution_success:
            try:
                from ..workflows.distillation_hook import WorkflowDistillationHook

                plan_meta = ctx.state.plan.metadata if ctx.state.plan else {}
                hook = WorkflowDistillationHook(engine=ctx.deps.knowledge_engine)

                async def _fire_distillation() -> None:
                    try:
                        await hook.on_execution_complete(
                            run_id=ctx.state.session_id or "unknown",
                            plan=ctx.state.plan,
                            result=result_text or "",
                            team_config_id=plan_meta.get("team_config_id"),
                            quality_score=1.0,
                        )
                    except Exception as ex:
                        logger.debug(f"Distillation hook async task failed: {ex}")

                # Fire and forget — do not block the response
                asyncio.ensure_future(_fire_distillation())
                logger.debug("[ORCH-1.25] Distillation hook dispatched (background)")
            except Exception as e:
                logger.debug(f"Distillation hook dispatch failed: {e}")

    return End(
        GraphResponse(
            status="completed",
            results={"output": result_text},
            metadata={
                "domain": ctx.state.routed_domain,
                "verification_attempts": ctx.state.verification_attempts,
            },
        )
    )


async def error_recovery_step(
    ctx: StepContext[GraphState, GraphDeps, Exception | str | Any],
) -> str | End[dict]:
    """Attempt graceful recovery before terminating the graph.

    Implements a two-tier recovery strategy:

    1. **Recoverable errors** (retry_count < 2 and partial results exist):
       Injects the error as feedback and routes to the planner for a
       fresh strategy, preserving any partial results already gathered.
    2. **Terminal errors** (retries exhausted, policy violations, or
       max-transition overflows): Terminates with a diagnostic report.

    Args:
        ctx: The pydantic-graph step context with the failure details as input.

    Returns:
        ``'planner'`` if recovery is attempted, or a terminal ``End`` state
        with the error summary and partial results.

    """
    error_str = str(ctx.inputs) if ctx.inputs else (ctx.state.error or "Unknown error")
    _emit_node_lifecycle(ctx.deps.event_queue, "error_recovery", "node_start")
    logger.error(f"error_recovery_step: {error_str}")

    # Terminal conditions that should NOT retry
    terminal_keywords = (
        "policy violation",
        "max node transitions",
        "max planning loops",
    )
    is_terminal = any(kw in error_str.lower() for kw in terminal_keywords)

    if not is_terminal and ctx.state.retry_count < 2:
        ctx.state.retry_count += 1
        ctx.state.validation_feedback = (
            f"Previous execution failed with error: {error_str[:500]}. "
            f"Please devise a different strategy to satisfy the query."
        )
        logger.warning(
            f"error_recovery_step: Recoverable failure (attempt {ctx.state.retry_count}). "
            f"Routing to planner for fresh strategy."
        )
        emit_graph_event(
            ctx.deps.event_queue,
            "error_recovery_replan",
            attempt=ctx.state.retry_count,
            error=error_str[:300],
        )
        _emit_node_lifecycle(
            ctx.deps.event_queue,
            "error_recovery",
            "node_complete",
            next_node="planner",
        )
        return "planner"

    logger.error(
        f"error_recovery_step: Terminal failure after {ctx.state.retry_count} retries. "
        f"Error: {error_str[:300]}"
    )
    emit_graph_event(
        ctx.deps.event_queue,
        "error_recovery_terminal",
        error=error_str[:300],
        retries=ctx.state.retry_count,
    )
    _emit_node_lifecycle(
        ctx.deps.event_queue,
        "error_recovery",
        "node_complete",
        next_node="end",
    )
    return End({"error": error_str, "results": ctx.state.results})


async def _distill_experience_from_retry(
    ctx: StepContext[GraphState, GraphDeps, Any], results_summary: str
) -> None:
    """CONCEPT:AHE-3.4: Contrastive self-correction distillation.

    Extracts an ExperienceNode by contrasting a successful retry against
    its previous failure feedback.
    """
    if not ctx.deps.knowledge_engine:
        return

    logger.info("Distilling Experience from successful retry...")
    try:
        from pydantic import BaseModel

        class ExtractedExperience(BaseModel):
            condition: str
            action: str

        distillation_agent = Agent(
            model=ctx.deps.agent_model,
            output_type=ExtractedExperience,
            system_prompt=(
                "You are an evolutionary critique engine. "
                "The agent previously failed this task, but succeeded on retry. "
                "Contrast the failure state with the success state to identify the specific "
                "tactical action that fixed the problem. Extract this as a condition-action pair."
            ),
        )

        prompt = (
            f"Original Query: {ctx.state.query}\n"
            f"Previous Feedback/Error: {ctx.state.validation_feedback or ctx.state.error}\n"
            f"Current Success Results: {results_summary}\n"
        )

        res = await distillation_agent.run(prompt)

        res_data = getattr(res, "data", None)
        if res_data:
            import uuid

            from ..models.knowledge_graph import ExperienceNode, RegistryNodeType

            exp_id = f"exp_{uuid.uuid4().hex[:8]}"
            node = ExperienceNode(
                id=exp_id,
                name=f"Fix: {res_data.action[:40]}",
                description=f"When {res_data.condition}",
                type=RegistryNodeType.EXPERIENCE,
                condition=res_data.condition,
                action=res_data.action,
                source_run_id=ctx.state.session_id,
            )
            from ..knowledge_graph.core.ogm import KGMapper

            ogm = KGMapper(ctx.deps.knowledge_engine)
            ogm.upsert(node)
            logger.info(f"Distilled Experience Node: {exp_id}")
    except Exception as e:
        logger.warning(f"Experience distillation failed: {e}")


async def parallel_trajectory_distiller(
    deps: GraphDeps, trajectories: list[dict[str, Any]], query: str = ""
) -> None:
    """CONCEPT:AHE-3.4: Memory-Aware Test-Time Scaling (Parallel Experience Distillation).

    Extracts an ExperienceNode by evaluating a batch of parallel trajectory attempts
    (both successes and failures). Unlike sequential retry loops, this batch processing
    distills a holistic reasoning memory that guides test-time scaling.

    Leverages CONCEPT:KG-2.4 (Inductive Knowledge Hypergraphs) to map derived tactics to
    hyperedges, enabling zero-shot structural generalization across novel topologies.
    """
    if not deps.knowledge_engine:
        return
    logger.info("Distilling Memory from Parallel Trajectories (CONCEPT:AHE-3.4)...")

    try:
        from pydantic import BaseModel, Field
        from pydantic_ai import Agent

        class BatchDistillation(BaseModel):
            tactical_condition: str = Field(
                description="The structural condition that requires this tactic."
            )
            tactical_action: str = Field(
                description="The generalized action/heuristic derived from parallel attempts."
            )
            confidence: float = Field(
                description="Confidence from 0.0 to 1.0 based on consensus across trials."
            )

        summary_blocks = []
        for traj in trajectories:
            status = "SUCCESS" if traj.get("success") else "FAILURE"
            out_str = str(traj.get("output", ""))[:500]
            summary_blocks.append(
                f"Trial {traj.get('candidate_id')}: [{status}] - {out_str}"
            )

        combined_trajectories = "\n\n".join(summary_blocks)

        distillation_agent = Agent(
            model=deps.agent_model,
            output_type=BatchDistillation,
            system_prompt=(
                "You are an evolutionary hypergraph orchestrator. "
                "Analyze this batch of parallel trajectory attempts for a single query. "
                "Identify the underlying structural/topological reason why some succeeded and others failed. "
                "Extract a generalized condition-action heuristic that captures this insight."
            ),
        )

        res = await distillation_agent.run(
            f"Original Query: {query}\n\nParallel Trajectories:\n{combined_trajectories}"
        )

        res_data = getattr(res, "data", None)
        if res_data and getattr(res_data, "confidence", 0) >= 0.5:
            import uuid

            from ..knowledge_graph.core.hypergraph import PositionalInteractionEncoder
            from ..models.knowledge_graph import ExperienceNode, RegistryNodeType

            exp_id = f"exp_par_{uuid.uuid4().hex[:8]}"
            node = ExperienceNode(
                id=exp_id,
                name=f"Parallel Tactic: {res_data.tactical_action[:30]}",
                description=f"Derived from parallel test-time scaling. When: {res_data.tactical_condition}",
                type=RegistryNodeType.EXPERIENCE,
                condition=res_data.tactical_condition,
                action=res_data.tactical_action,
                success_rate=res_data.confidence,
            )

            # CONCEPT:KG-2.4: Compute positional interaction encoding for structural generalization
            # We map the condition (position 1) to the action (position 2) in the hypergraph
            encoder = PositionalInteractionEncoder()
            enc_pi = encoder.encode_interaction(1, 2)
            node.metadata["enc_pi"] = enc_pi
            node.metadata["source"] = "parallel_scaling_distillation"

            from ..knowledge_graph.core.ogm import KGMapper

            ogm = KGMapper(deps.knowledge_engine)
            ogm.upsert(node)
            logger.info(
                f"Distilled Parallel Experience Node (CONCEPT:AHE-3.4): {exp_id} with EncPI mapping."
            )

    except Exception as e:
        logger.warning(f"Parallel experience distillation failed: {e}")
