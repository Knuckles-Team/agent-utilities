from __future__ import annotations

"""ORCH-1.1 Hierarchical Planner.

Consolidates HTN, LATS, Conductor, and Recursive Executor logic.
"""


# === From planning.py ===

#!/usr/bin/python
"""Graph Planning Steps.

Research, architecture, planning, and memory selection nodes.
Extracted from the monolithic steps.py for maintainability.

- ``researcher_step``: Deep workspace/web/KG context triangulation.
- ``planner_step``: Re-planning after verification failures.
- ``architect_step``: High-level system design generation.
- ``memory_selection_step``: Workspace memory filtering and gap detection.
- ``fetch_epistemic_context``: Aggregate workspace metadata helper.
"""
# CONCEPT:ECO-4.0 — Planner Step

import logging
import os
import re
import subprocess
from pathlib import Path
from typing import Any

from pydantic_ai import Agent
from pydantic_graph import End
from pydantic_graph.beta import StepContext

from ..models import ExecutionStep, GraphPlan
from ..tools.knowledge_tools import knowledge_tools
from .config_helpers import (
    emit_graph_event,
    get_discovery_registry,
    load_specialized_prompts,
)
from .lifecycle import _emit_node_lifecycle
from .state import GraphDeps, GraphState

logger = logging.getLogger(__name__)

__all__ = [
    "researcher_step",
    "planner_step",
    "architect_step",
    "memory_selection_step",
    "fetch_epistemic_context",
]


async def researcher_step(
    ctx: StepContext[GraphState, GraphDeps, ExecutionStep | str],
) -> str:
    """Execute deep discovery across the workspace, codebase, and web.

    This node acts as a unified researcher that gathers the technical
    context required for system architecture or task planning.

    Args:
        ctx: The pydantic-graph step context, potentially containing
            a specific structured research question.

    Returns:
        The ID of the next node to execute ('execution_joiner').

    """
    logger.info("Researcher: Triangulating context...")
    unified_context = await fetch_epistemic_context()

    # If the dispatcher sent a specific question, use it as the prompt
    step_input = ctx.inputs
    research_query = ctx.state.query
    if isinstance(step_input, ExecutionStep) and step_input.input_data:
        if isinstance(step_input.input_data, dict):
            research_query = step_input.input_data.get("question", research_query)
        elif isinstance(step_input.input_data, str):
            research_query = step_input.input_data

    # Intelligent Researcher Sub-Agents (for internal reasoning)
    researcher_prompt = load_specialized_prompts("researcher")

    from ..tools.developer_tools import project_search
    from ..tools.workspace_tools import read_workspace_file

    researcher = Agent(
        model=ctx.deps.agent_model,
        system_prompt=(f"{researcher_prompt}\n\nWorkspace Context: {unified_context}"),
        tools=[project_search, read_workspace_file] + knowledge_tools,
    )

    try:
        logger.info(
            f"Researcher: Starting execution. Query length: {len(research_query)}"
        )
        res = await researcher.run(research_query)
        logger.info("Researcher: Execution successfully completed.")
        ctx.state._update_usage(getattr(res, "usage", None))

        # Save to registry for other agents to consume
        node_uid = f"researcher_{ctx.state.step_cursor}"
        ctx.state.results_registry[node_uid] = str(res.output)

        return "execution_joiner"
    except Exception as e:
        logger.error(f"Researcher failed: {e}")
        return "error_recovery"


async def planner_step(
    ctx: StepContext[GraphState, GraphDeps, None],
) -> GraphPlan | str:
    """Re-plan execution after a verification failure.

    This node is the dedicated re-planning entry point.  It is invoked by
    the verifier when a validation score is very low (< 0.4), indicating
    the *approach* — not just the execution — was wrong.

    Unlike the router (which generates an initial plan from scratch), the
    planner incorporates verification feedback, previous execution results,
    and architectural decisions to craft a corrected plan.

    Args:
        ctx: The pydantic-graph step context.

    Returns:
        ``'dispatcher'`` on successful re-plan, or ``'error_recovery'``
        on failure.

    """
    logger.info(
        f"Planner: Re-planning (attempt {ctx.state.verification_attempts}). "
        f"Feedback: {(ctx.state.validation_feedback or 'none')[:100]}"
    )

    emit_graph_event(
        ctx.deps.event_queue,
        "replanning_started",
        attempt=ctx.state.verification_attempts,
        feedback=ctx.state.validation_feedback or "",
    )

    planner_prompt = load_specialized_prompts("planner")
    unified_context = await fetch_epistemic_context()

    # Build a rich re-planning context from previous execution
    previous_results = "\n".join(
        f"- {node}: {str(val)[:300]}"
        for node, val in ctx.state.results_registry.items()
    )

    feedback_section = ""
    if ctx.state.validation_feedback:
        feedback_section = (
            f"### VERIFICATION FEEDBACK (CRITICAL)\n"
            f"The previous plan was rejected by the verifier. Address this:\n"
            f"{ctx.state.validation_feedback}\n\n"
        )

    error_section = ""
    if ctx.state.error:
        error_section = f"### PREVIOUS ERROR\n{ctx.state.error}\n\n"

    results_section = ""
    if previous_results:
        results_section = (
            f"### PREVIOUS EXECUTION RESULTS (for context)\n{previous_results}\n\n"
        )

    from .executor import _get_domain_tools
    from .nodes import find_best_matching_process_flow_via_kg

    domain_tools, domain_toolsets = await _get_domain_tools("planner", ctx.deps)

    # 0. Discover Processes and Policies from Knowledge Graph
    policies_context = ""
    process_context = ""
    if ctx.deps.knowledge_engine:
        logger.info("Planner: Discovering relevant policies and processes from KG...")
        relevant_policies = ctx.deps.knowledge_engine.find_relevant_policies(
            ctx.state.query
        )
        if relevant_policies:
            policies_context = "### APPLICABLE POLICIES\n" + "\n".join(
                [f"- {p['name']}: {p['description']}" for p in relevant_policies]
            )

        best_flow = await find_best_matching_process_flow_via_kg(ctx.state.query)
        if best_flow:
            process_context = (
                f"### RECOMMENDED PROCESS FLOW\n"
                f"A matching process flow '{best_flow.name}' was found in the Knowledge Graph:\n"
                f"Goal: {best_flow.goal}\n"
                f"You should strongly consider following this established process."
            )
            ctx.state.current_flow_id = best_flow.flow_id

    planner = Agent(
        model=ctx.deps.agent_model,
        output_type=GraphPlan,
        deps_type=GraphDeps,
        tools=domain_tools,
        toolsets=domain_toolsets,
        system_prompt=(
            f"{planner_prompt}\n\n"
            f"### WIDE-SEARCH ORCHESTRATION (CONCEPT:ORCH-1.1)\n"
            f"If the query requests extracting a large table of data across many entities "
            f"(e.g., 'Web2WideSearch'), decompose the extraction into discrete batches. Emit "
            f"multiple parallel ExecutionSteps assigned to an extraction specialist, each targeting "
            f"a specific partition of the data in its 'refined_subtask'. Set 'is_parallel=True' "
            f"and configure 'access_list' to share the shared workboard context.\n\n"
            f"### MULTI-LEVEL ABSTRACTION LAYERING (CONCEPT:ORCH-1.3)\n"
            f"Do not attempt to plan every micro-step. Instead, emit coarse, high-level steps "
            f"and delegate the detailed refinement to the executing adaptive_agent_router. This saves "
            f"upfront planning tokens and allows adaptive_agent_router to adapt dynamically.\n\n"
            f"{feedback_section}"
            f"{error_section}"
            f"{results_section}"
            f"{policies_context}\n\n"
            f"{process_context}\n\n"
            f"### ARCHITECTURAL DECISIONS\n{ctx.state.architectural_decisions}\n\n"
            f"### WORKSPACE CONTEXT\n{unified_context}"
        ),
    )

    from ..rlm.config import RLMConfig
    from ..rlm.repl import RLMEnvironment

    rlm_config = RLMConfig()

    try:
        # CONCEPT:AHE-3.4 — Heavy Thinking activation gate
        # When the complexity estimator determines the query warrants
        # deep multi-trajectory reasoning, use the Heavy Thinking pipeline
        # instead of standard LATS or single-shot planning.
        use_heavy_thinking = getattr(ctx.state, "use_heavy_thinking", False)
        if not use_heavy_thinking and ctx.state.verification_attempts > 2:
            # Auto-escalate to heavy thinking after multiple failures
            try:
                from .heavy_thinking import ComplexityEstimator

                complexity = ComplexityEstimator.estimate(ctx.state.query)
                if complexity >= 0.6:
                    use_heavy_thinking = True
                    logger.info(
                        "Planner: Auto-escalating to Heavy Thinking "
                        "(complexity=%.2f, attempts=%d)",
                        complexity,
                        ctx.state.verification_attempts,
                    )
            except Exception as e:
                logger.debug("Complexity estimation failed: %s", e)

        if use_heavy_thinking:
            logger.info("Planner: Running in Heavy Thinking (CONCEPT:AHE-3.4) mode.")
            from .heavy_thinking import HeavyThinkingPlanner as HTP

            ht_planner = HTP(
                context=f"{feedback_section}\n{error_section}\n{results_section}\n{policies_context}\n{process_context}\n{unified_context}",
                deps=ctx.deps,
                model=ctx.deps.agent_model,
            )
            ctx.state.plan = await ht_planner.search(ctx.state.query)
        # CONCEPT:ORCH-1.1 — LATS implementation fallback for complex failures
        elif (
            getattr(ctx.state, "use_lats", False) or ctx.state.verification_attempts > 1
        ):
            logger.info("Planner: Running in LATS (Language Agent Tree Search) mode.")
            lats_env = LATSPlanner(
                context=f"{feedback_section}\n{error_section}\n{results_section}\n{policies_context}\n{process_context}\n{unified_context}",
                deps=ctx.deps,
                model=ctx.deps.agent_model,
            )
            ctx.state.plan = await lats_env.search(ctx.state.query)
        elif rlm_config.enabled or getattr(ctx.state, "requires_long_horizon", False):
            logger.info("Planner: Running in RLM (Recursive Language Model) mode.")
            env = RLMEnvironment(
                context=f"{feedback_section}\n{error_section}\n{results_section}\n{policies_context}\n{process_context}\n{unified_context}",
                config=rlm_config,
                graph_deps=ctx.deps,
            )
            rlm_result = await env.run_full_rlm(
                f"Create a CORRECTED execution plan for: {ctx.state.query}\n\n"
                f"The previous approach failed. You MUST use a different strategy. "
                f"Use the REPL to deeply analyze the context. "
                f"You MUST output a JSON representation of a GraphPlan using FINAL_VAR('plan', <json_string>)."
            )

            import json

            try:
                plan_data = json.loads(rlm_result)
                ctx.state.plan = GraphPlan.model_validate(plan_data)
            except Exception as parse_e:
                logger.warning(
                    f"RLM output was not valid GraphPlan JSON: {parse_e}. Running fallback parser."
                )
                res = await planner.run(
                    f"Parse this into a GraphPlan:\n{rlm_result}", deps=ctx.deps
                )
                ctx.state.plan = res.output
        else:
            res = await planner.run(
                f"Create a CORRECTED execution plan for: {ctx.state.query}\n\n"
                f"The previous approach failed. You MUST use a different strategy.",
                deps=ctx.deps,
            )
            ctx.state.plan = res.output

        ctx.state.step_cursor = 0
        ctx.state.needs_replan = False
        ctx.state.error = None
        logger.info(
            f"Planner: Re-plan generated with {len(ctx.state.plan.steps)} steps."
        )

        emit_graph_event(
            ctx.deps.event_queue,
            "replanning_completed",
            step_count=len(ctx.state.plan.steps),
        )

        return "dispatcher"
    except Exception as e:
        logger.warning(f"Planning failed: {e}. Attempting unstructured fallback.")
        try:
            fallback_prompt = (
                f"{planner_prompt}\n\nCRITICAL: You failed JSON validation. "
                "Please reply ONLY with a simple text list of the exact agent names you want to use from the available specialists "
                "(separated by commas). DO NOT output conversational text, just the comma-separated agent names."
            )
            fallback_agent = Agent(
                model=ctx.deps.agent_model, system_prompt=fallback_prompt
            )
            fallback_res = await fallback_agent.run(ctx.state.query)

            raw_text = str(
                getattr(fallback_res, "data", getattr(fallback_res, "output", ""))
            )
            available = (
                list(ctx.deps.tag_prompts.keys())
                if ctx.deps and hasattr(ctx.deps, "tag_prompts")
                else []
            )

            steps = []
            for spec in available:
                if spec.lower() in raw_text.lower():
                    steps.append(
                        ExecutionStep(node_id=spec, input_data=ctx.state.query)
                    )

            if steps:
                logger.info(
                    f"Planner Fallback: Extracted {len(steps)} steps from text: {[s.node_id for s in steps]}"
                )
                ctx.state.plan = GraphPlan(
                    steps=steps,
                    metadata={"reasoning": "Fallback natural language extraction"},
                )
                ctx.state.step_cursor = 0
                return "dispatcher"
            else:
                logger.warning(
                    f"Planner Fallback: No known specialists found in text. Available: {available}. Raw text: {raw_text}"
                )
        except Exception as fallback_e:
            logger.error(f"Planner fallback also failed: {fallback_e}")

        logger.error(f"Re-planning failed fully: {e}")
        return "error_recovery"


async def fetch_epistemic_context() -> str:
    """Aggregate essential workspace metadata for agent situational awareness.

    Collects agent registries from Knowledge Graph, historical memory, VCS state
    (git status), and KG-based semantic retrieval to provide agents with a
    holistic view of the current repository state without overwhelming the
    context window.

    Returns:
        A formatted markdown string containing truncated registry previews,
        recent memory, git status, and KG retrieval results.

    """
    # 1. Fetch agents from Knowledge Graph
    try:
        registry = get_discovery_registry()
        agents_info = []
        for agent in registry.agents:
            agents_info.append(f"- {agent.name}: {agent.description}")

        mcp_agents = "\n".join(agents_info)
        if len(mcp_agents.splitlines()) > 500:
            mcp_agents = (
                "\n".join(mcp_agents.splitlines()[:500]) + "\n\n... (truncated)"
            )
    except Exception as e:
        logger.debug(f"Failed to fetch agents for context: {e}")
        mcp_agents = "(empty or graph unavailable)"

    # 2. Run git status
    try:
        git_status = subprocess.check_output(
            ["git", "status", "--short"], text=True
        ).strip()
    except Exception:
        git_status = "Not a git repository or git not installed."

    # 3. KG-based context from IntelligenceGraphEngine (CONCEPT:KG-2.3)
    kg_context = ""
    try:
        import networkx as nx

        from ..knowledge_graph.core.engine import IntelligenceGraphEngine

        engine = IntelligenceGraphEngine(graph=nx.MultiDiGraph())
        if engine.graph and engine.graph.number_of_nodes() > 0:
            # Sample top nodes by degree centrality for awareness
            nodes_by_degree = sorted(
                engine.graph.degree(), key=lambda x: x[1], reverse=True
            )
            kg_entries = []
            for node_id, degree in nodes_by_degree[:5]:
                data = engine.graph.nodes.get(node_id, {})
                name = data.get("name", node_id)
                desc = str(data.get("description", ""))[:120]
                kg_entries.append(f"- {name} (degree={degree}): {desc}")
            kg_context = "\n".join(kg_entries)
    except Exception as e:
        logger.debug("KG context unavailable: %s", e)

    sections = [
        "### PROJECT CONTEXT (Agent OS)\n",
        f"**Registered Agents (Knowledge Graph):**\n{mcp_agents or '(empty)'}\n",
        f"**Git Status:**\n{git_status or '(clean)'}",
    ]
    if kg_context:
        sections.append(f"\n**KG Context (top nodes):**\n{kg_context}")

    return "\n".join(sections)


async def architect_step(
    ctx: StepContext[GraphState, GraphDeps, None],
) -> str:
    """Analyze system requirements and propose high-level design decisions.

    This node uses gathered research findings to define the overall structure,
    technical stack, and implementation approach. It bridges the gap
    between raw research and concrete planning.

    Args:
        ctx: The pydantic-graph step context containing shared state.

    Returns:
        The next node identifier ('planner') or 'error_recovery'.

    """
    logger.info("Architect: Designing system changes...")
    architect_prompt = load_specialized_prompts("architect")

    architect = Agent(
        model=ctx.deps.agent_model,
        system_prompt=architect_prompt + f"\n\nContext:\n{ctx.state.exploration_notes}",
        tools=knowledge_tools,
    )

    try:
        res = await architect.run(
            f"Design the architecture for: {ctx.state.query}", deps=ctx.deps
        )
        ctx.state.architectural_decisions = str(res.output)
        return "planner"
    except Exception as e:
        logger.error(f"Architect failed: {e}")
        return "error_recovery"


async def memory_selection_step(
    ctx: StepContext[GraphState, GraphDeps, None],
) -> str | End[Any]:
    """Filter and select long-term project memories relevant to the session.

    Scans the local workspace for relevant documentation and memory files,
    using an LLM-based selector to identify which files contain essential
    context for the current user query.

    When the selector finds no relevant memories and the query appears to
    require background research (more than a simple conversational turn),
    this node routes to the researcher for web/codebase discovery before
    returning to the dispatcher.

    Args:
        ctx: The pydantic-graph step context.

    Returns:
        ``'dispatcher'`` with updated exploration notes, or ``'researcher'``
        when a context gap is detected.

    """
    logger.info("Memory Selection: Identifying relevant context...")
    if not ctx.state.exploration_notes:
        ctx.state.exploration_notes = (
            "### KNOWLEDGE EXPLORATION\nInitialized memory discovery phase.\n"
        )

    _emit_node_lifecycle(ctx.deps.event_queue, "memory_selection", "node_start")
    prompt_content = load_specialized_prompts("memory_selection")
    root = ctx.state.project_root or os.getcwd()
    memories = []

    # Phase 1: Scan Workspace Documentation (Markdown files)
    for p in Path(root).rglob("*.md"):
        if ".gemini" in str(p) or "node_modules" in str(p):
            continue
        try:
            content = p.read_text(encoding="utf-8", errors="ignore")
            description = "General project documentation"
            if content.startswith("---"):
                match = re.search(r"description:\s*(.*)", content)
                if match:
                    description = match.group(1).strip()
            memories.append(f"- [Doc] {p.name}: {description}")
        except Exception as e:
            logger.warning(f"Context document scanning failed: {e}")

    # Phase 2: Knowledge Graph Memory Lookup (Unified Layer)
    kg_memories = []
    if ctx.deps.knowledge_engine:
        logger.info(
            "Memory Selection: Querying Knowledge Graph for contextual memories..."
        )
        # Simple extraction of potential memory keywords
        words = re.findall(r"\b[a-z0-9_]{4,}\b", ctx.state.query.lower())
        seen_mem_ids = set()
        for word in words:
            for node_id, data in ctx.deps.knowledge_engine.graph.nodes(data=True):
                if node_id in seen_mem_ids:
                    continue
                if data.get("type") == "memory" and (
                    word in data.get("description", "").lower()
                    or word in data.get("name", "").lower()
                ):
                    kg_memories.append(
                        f"- [KnowledgeGraph Memory] {data['name']}: {data['description'][:300]}"
                    )
                    seen_mem_ids.add(node_id)

        if kg_memories:
            logger.info(
                f"Memory Selection: Retrieved {len(kg_memories)} memories from Knowledge Graph."
            )
            memories.extend(kg_memories)

    if not memories:
        logger.info("Memory Selection: No workspace memories found. Skipping.")
        # Context gap detection: non-trivial queries with no memories
        # benefit from a research pass before execution.
        word_count = len(ctx.state.query.split())
        already_researched = ctx.state.global_research_loops > 1
        if (
            word_count > 8
            and not already_researched
            and not ctx.state.exploration_notes
        ):
            logger.info(
                "Memory Selection: Context gap detected — routing to researcher."
            )
            emit_graph_event(
                ctx.deps.event_queue,
                "context_gap_detected",
                reason="no_workspace_memories",
                query_words=word_count,
            )
            _emit_node_lifecycle(
                ctx.deps.event_queue,
                "memory_selection",
                "node_complete",
                next_node="researcher",
            )
            return "researcher"
        _emit_node_lifecycle(
            ctx.deps.event_queue,
            "memory_selection",
            "node_complete",
            next_node="dispatcher",
        )
        return "dispatcher"

    selectors = Agent(
        model=ctx.deps.agent_model,
        system_prompt=prompt_content,
        output_type=dict,
    )
    try:
        res = await selectors.run(
            f"Query: {ctx.state.query}\n\nAvailable memories:\n"
            + "\n".join(memories[:20])
        )
        selected = res.output.get("selected_memories", [])
        logger.info(
            f"Memory Selection: Selected {len(selected)} relevant files: {selected}"
        )
    except Exception as e:
        logger.warning(
            f"Memory Selection structured output failed: {e}. Attempting unstructured fallback."
        )
        try:
            fallback = Agent(model=ctx.deps.agent_model, system_prompt=prompt_content)
            res = await fallback.run(
                f"Query: {ctx.state.query}\n\nAvailable memories:\n"
                + "\n".join(memories[:20])
                + "\n\nCRITICAL: Just output the exact file names you need, separated by commas. DO NOT output conversational text."
            )

            selected = []
            raw_text = str(getattr(res, "data", getattr(res, "output", "")))
            for mem_line in memories[:20]:
                filename = (
                    mem_line.split(":")[0]
                    .replace("- [Doc] ", "")
                    .replace("- [KnowledgeGraph Memory] ", "")
                    .strip()
                )
                if filename.lower() in raw_text.lower():
                    selected.append(filename)

            logger.info(
                f"Memory Selection Fallback: Extracted {len(selected)} memories from text: {selected}"
            )
        except Exception as fallback_e:
            logger.error(f"Memory selection fallback also failed: {fallback_e}")
            selected = []

        loaded_context = []
        for filename in selected:
            for p in Path(root).rglob(filename):
                loaded_context.append(
                    f"### {filename}\n{p.read_text(encoding='utf-8', errors='ignore')}"
                )
                break

        ctx.state.exploration_notes += "\n\n### SELECTED MEMORIES\n" + "\n\n".join(
            loaded_context
        )

        # Context gap: memories exist but none are relevant to this query
        already_researched = ctx.state.global_research_loops > 1
        if (
            not selected
            and not already_researched
            and not ctx.state.exploration_notes.strip()
        ):
            logger.info(
                "Memory Selection: No relevant memories matched — routing to researcher."
            )
            emit_graph_event(
                ctx.deps.event_queue,
                "context_gap_detected",
                reason="no_relevant_memories",
                available=len(memories),
                selected=0,
            )
            _emit_node_lifecycle(
                ctx.deps.event_queue,
                "memory_selection",
                "node_complete",
                next_node="researcher",
            )
            return "researcher"

        _emit_node_lifecycle(
            ctx.deps.event_queue,
            "memory_selection",
            "node_complete",
            next_node="dispatcher",
        )
        return "dispatcher"

    return "dispatcher"


# === From lats.py ===

import asyncio
import logging

logger = logging.getLogger(__name__)


class LATSPlanner:
    """CONCEPT:ORCH-1.1 — Language Agent Tree Search (LATS) Planner.
    Uses Monte Carlo Tree Search to explore multiple execution plans,
    simulate their outcomes, and select the highest-scoring trajectory
    when the standard single-shot planner fails.
    """

    def __init__(self, context: str, deps: GraphDeps, model: Any):
        self.context = context
        self.deps = deps
        self.model = model
        self.agent = Agent(
            model=model,
            system_prompt=(
                f"You are a LATS Planning Agent. Generate candidate execution plans "
                f"and evaluate them based on the context.\n\nContext:\n{context}"
            ),
            output_type=GraphPlan,
        )
        self.evaluator = Agent(
            model=model,
            system_prompt="Evaluate the feasibility of the following plan out of 10. Output only the integer score.",
        )

    async def search(self, query: str, num_simulations: int = 3) -> GraphPlan:
        # MCTS Simulation Phase (simplified breadth-first for now)
        tasks = []
        for i in range(num_simulations):
            prompt = f"Generate a unique candidate plan for: {query}\nCandidate {i + 1}: Focus on a different approach."
            tasks.append(self.agent.run(prompt, deps=self.deps))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # CONCEPT:AHE-3.4: Memory-Aware Test-Time Scaling
        # Distill memory from parallel scaling trajectories before evaluation
        try:
            from .verification import parallel_trajectory_distiller

            trajectories = []
            for i, res in enumerate(results):
                success = not isinstance(res, Exception)
                output_data = (
                    res.output.model_dump()
                    if success
                    and hasattr(res, "output")
                    and hasattr(res.output, "model_dump")
                    else str(res)
                )
                trajectories.append(
                    {"candidate_id": i, "success": success, "output": output_data}
                )

            # Fire-and-forget or await parallel distillation
            await parallel_trajectory_distiller(self.deps, trajectories, query=query)
        except Exception as e:
            logger.warning(f"LATSPlanner: Parallel trajectory distillation failed: {e}")

        best_plan = None
        best_score = -1

        for res in results:
            if isinstance(res, Exception):
                logger.warning(f"LATSPlanner: Simulation failed: {res}")
                continue

            plan = res.output  # type: ignore[union-attr]
            # Evaluate plan
            try:
                eval_res = await self.evaluator.run(
                    f"Query: {query}\nPlan: {plan.model_dump_json()}\nEvaluate from 1-10:"
                )
                score = int(eval_res.output.strip())
                logger.info(f"LATSPlanner: Candidate scored {score}/10")
                if score > best_score:
                    best_score = score
                    best_plan = plan
            except Exception as e:
                logger.warning(f"LATSPlanner: Evaluation failed: {e}")

        if best_plan:
            logger.info(f"LATSPlanner: Selected best plan with score {best_score}")
            return best_plan

        logger.warning(
            "LATSPlanner: All simulations failed, falling back to empty plan."
        )
        return GraphPlan(steps=[], metadata={"reasoning": "LATS failed"})


# === From recursive_executor.py ===

#!/usr/bin/python
"""CONCEPT:ORCH-1.1 — Recursive Graph Orchestration.

Enables the graph orchestrator to spawn a nested ``run_graph()`` call as a
specialist step. The inner graph receives the parent's ``GraphState``
(including failed results and error context) and can independently plan
and execute a corrected strategy.

Inspired by the RL Conductor's self-referential recursive topologies
(Nielsen et al., ICLR 2026), where the Conductor can specify itself as
a worker LLM and adaptively revise its coordination strategy at test-time.

This composes naturally with:
    - CONCEPT:ORCH-1.1 (RLM): Both provide recursive execution, but RLM is
      sub-shell-level while this is graph-level.
    - CONCEPT:ORCH-1.0 (Graph Orchestration): Reuses the existing ``run_graph`` pipeline.
    - CONCEPT:ORCH-1.1 (Conductor Workflow): The recursive call gets a refined
      subtask explaining what the parent tried and what needs correction.

Controlled by:
    - ``MAX_RECURSION_DEPTH`` (env var, default 2): Hard ceiling on nesting.
    - Inherits parent's timeout budget minus elapsed time.

Usage::

    from agent_utilities.graph.hierarchical_planner import (
        execute_recursive_graph,
        RecursiveContext,
    )

    ctx = RecursiveContext(
        parent_query="build a REST API",
        parent_plan_summary="[researcher, python_programmer]",
        parent_error="Researcher found no relevant context",
        parent_results={"researcher": "No results found"},
        recursion_depth=1,
    )
    result = await execute_recursive_graph(ctx, graph_deps)

See docs/pillars/1_graph_orchestration.md §CONCEPT:ORCH-1.1
"""


import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .state import GraphDeps

logger = logging.getLogger(__name__)

from agent_utilities.core.config import config

MAX_RECURSION_DEPTH = int(config.max_recursion_depth or "2")
"""Hard ceiling on recursive graph nesting.  Default 2, configurable."""


@dataclass
class RecursiveContext:
    """Parent state snapshot for recursive graph orchestration.

    CONCEPT:ORCH-1.1 — Recursive Graph Orchestration

    Carries the parent graph's context into the inner graph so it can
    understand what was tried, what failed, and devise a corrected strategy.

    Attributes:
        parent_query: The original user query from the parent graph.
        parent_plan_summary: String summary of the parent's ``GraphPlan`` steps.
        parent_error: Error message from the parent's failed execution.
        parent_results: Partial results from the parent's ``results_registry``.
        recursion_depth: Current nesting depth (parent's depth + 1).
    """

    parent_query: str = ""
    parent_plan_summary: str = ""
    parent_error: str = ""
    parent_results: dict[str, Any] = field(default_factory=dict)
    recursion_depth: int = 1


class RecursionDepthExceeded(RuntimeError):
    """Raised when recursive orchestration exceeds MAX_RECURSION_DEPTH."""


async def execute_recursive_graph(
    context: RecursiveContext,
    deps: GraphDeps,
) -> str:
    """Spawn a nested graph execution with parent context.

    CONCEPT:ORCH-1.1 — Recursive Graph Orchestration

    Creates a new ``GraphState`` seeded with the parent's failed context
    as ``exploration_notes``, then runs the full graph pipeline.  The inner
    graph independently plans and executes a corrected strategy.

    Args:
        context: The ``RecursiveContext`` carrying parent state.
        deps: The ``GraphDeps`` runtime configuration (shared with parent).

    Returns:
        The synthesized result text from the inner graph execution.

    Raises:
        RecursionDepthExceeded: If ``context.recursion_depth`` exceeds
            ``MAX_RECURSION_DEPTH``.
    """
    if context.recursion_depth > MAX_RECURSION_DEPTH:
        raise RecursionDepthExceeded(
            f"Recursive orchestration depth {context.recursion_depth} exceeds "
            f"MAX_RECURSION_DEPTH={MAX_RECURSION_DEPTH}. Terminating to prevent "
            f"infinite recursion."
        )

    logger.info(
        "[CONCEPT:ORCH-1.1] Spawning recursive graph execution (depth=%d/%d)",
        context.recursion_depth,
        MAX_RECURSION_DEPTH,
    )

    # Build exploration notes from parent context
    parent_context_notes = (
        f"### RECURSIVE ORCHESTRATION CONTEXT (Depth {context.recursion_depth})\n"
        f"This is a recursive re-orchestration attempt. The parent graph's strategy "
        f"failed and needs a fundamentally different approach.\n\n"
        f"**Original Query**: {context.parent_query}\n\n"
        f"**Parent's Plan**: {context.parent_plan_summary}\n\n"
        f"**Parent's Error**: {context.parent_error}\n\n"
    )

    # Include partial results if available
    if context.parent_results:
        parent_context_notes += "**Partial Results from Parent**:\n"
        for key, value in context.parent_results.items():
            # Truncate long results to prevent prompt bloat
            val_str = str(value)[:500]
            parent_context_notes += f"- {key}: {val_str}\n"

    # CONCEPT:ORCH-1.1 — Generate a refined subtask for the inner graph
    refined_query = (
        f"The previous orchestration attempt failed with error: "
        f"'{context.parent_error[:300]}'. "
        f"The original goal was: '{context.parent_query}'. "
        f"Devise and execute a DIFFERENT strategy to accomplish this goal. "
        f"Do NOT repeat the same approach that failed."
    )

    # Import here to avoid circular dependency
    from .dynamic_graph_orchestrator import run_graph

    result = await run_graph(  # type: ignore[call-arg]
        query=refined_query,
        deps=deps,
        exploration_notes=parent_context_notes,
        recursion_depth=context.recursion_depth,
    )

    # Extract result text
    if isinstance(result, dict):
        output = result.get("results", {}).get("output", "")
        if not output:
            output = str(result)
    else:
        output = str(result)

    logger.info(
        "[CONCEPT:ORCH-1.1] Recursive graph execution completed (depth=%d). Output length: %d",
        context.recursion_depth,
        len(output),
    )

    return output


# === From evolutionary_aggregation.py ===

#!/usr/bin/python
"""CONCEPT:AHE-3.2 — Evolutionary Aggregation Engine.

Group-level diversity scoring, three-tier aggregation, and convergence-
aware early stopping for specialist outputs.  Adapted from the Squeeze
Evolve multi-model orchestration framework (Maheswaran et al., 2026).

Core signals:
    - **Group Confidence (GC)**: Mean confidence across proposals in a
      group.  Eq. 4 from the paper: GC(g) = (1/K) Σ C(τ) for τ ∈ g.
    - **Group Diversity (D)**: Number of unique answer clusters in a
      group.  Eq. 5: D(g) = |{answer(τ) : τ ∈ g}|.

Three-tier aggregation strategy:
    - ``MAJORITY_VOTE``:  Free — no LLM call when all adaptive_agent_router agree.
    - ``LIGHT_MODEL``:    Cheap model synthesis for moderate-confidence.
    - ``HEAVY_MODEL``:    Deep aggregation for low-confidence, high-
                          diversity groups requiring reasoning.

Integrates with:
    - CONCEPT:ORCH-1.2 (Global Workspace Attention): Proposals and confidence scores.
    - CONCEPT:OS-5.2 (Cognitive Scheduler): ConvergenceMonitor for early stopping.
    - CONCEPT:ORCH-1.2 (Confidence-Gated Router): Feeds group confidence back as a
      routing signal.

See docs/pillars/1_graph_orchestration.md §CONCEPT:ORCH-1.2
"""


import logging
import time
from enum import StrEnum
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from .workspace_attention import Proposal

logger = logging.getLogger(__name__)


# ── Aggregation Strategy ─────────────────────────────────────────────


class AggregationStrategy(StrEnum):
    """Three-tier aggregation strategy from Squeeze Evolve §5.1."""

    MAJORITY_VOTE = "majority_vote"
    """Free — no LLM call needed when all adaptive_agent_router agree."""

    LIGHT_MODEL = "light_model"
    """Cheap model synthesis for moderate-confidence groups."""

    HEAVY_MODEL = "heavy_model"
    """Expensive model deep aggregation for low-confidence, high-diversity."""


# ── Group Fitness ─────────────────────────────────────────────────────


class GroupFitness(BaseModel):
    """Fitness metrics for a group of specialist proposals.

    CONCEPT:AHE-3.2 — Evolutionary Aggregation Engine

    Implements the group-level signals from Squeeze Evolve §5.1:
    - Group Confidence (GC): Mean confidence across proposals in the group
    - Group Diversity (D): Number of unique answer clusters

    The ``recommended_strategy`` is computed from these signals and
    determines which aggregation path the group should follow.
    """

    group_id: str = Field(description="Unique identifier for this group.")
    specialist_ids: list[str] = Field(
        default_factory=list,
        description="IDs of adaptive_agent_router whose proposals form this group.",
    )
    group_confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="GC(g) — mean confidence across group proposals. Eq. 4.",
    )
    group_diversity: int = Field(
        default=1,
        ge=0,
        description="D(g) — number of unique answer clusters. Eq. 5.",
    )
    recommended_strategy: AggregationStrategy = Field(
        default=AggregationStrategy.LIGHT_MODEL,
        description="The recommended aggregation tier for this group.",
    )
    timestamp: str = Field(
        default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )


# ── Convergence Monitor ──────────────────────────────────────────────


class ConvergenceMonitor:
    """Tracks diversity across iterations for early stopping.

    CONCEPT:AHE-3.2 — Evolutionary Aggregation Engine

    When diversity drops below ``convergence_threshold`` for
    ``patience`` consecutive iterations, signals that the evolutionary
    loop should terminate early to avoid diversity collapse — the
    central bottleneck identified in Squeeze Evolve §4.

    Args:
        convergence_threshold: Diversity score below which we consider
            the population converged.
        patience: Number of consecutive low-diversity readings before
            signalling convergence.
    """

    def __init__(
        self,
        convergence_threshold: float = 0.1,
        patience: int = 3,
    ) -> None:
        self.convergence_threshold = convergence_threshold
        self.patience = patience
        self._history: list[float] = []
        self._consecutive_low: int = 0

    def update(self, diversity_score: float) -> bool:
        """Record a diversity measurement.

        Args:
            diversity_score: Normalised diversity ``[0, 1]``.

        Returns:
            ``True`` if convergence has been detected (the loop should
            stop), ``False`` otherwise.
        """
        self._history.append(diversity_score)

        if diversity_score < self.convergence_threshold:
            self._consecutive_low += 1
        else:
            self._consecutive_low = 0

        converged = self._consecutive_low >= self.patience
        if converged:
            logger.info(
                "[CONCEPT:ORCH-1.2] Convergence detected: diversity %.3f < %.3f "
                "for %d consecutive iterations — recommending early stop.",
                diversity_score,
                self.convergence_threshold,
                self.patience,
            )
        return converged

    def reset(self) -> None:
        """Reset the monitor for a new evolutionary loop."""
        self._history.clear()
        self._consecutive_low = 0

    @property
    def history(self) -> list[float]:
        """Return the full diversity history."""
        return list(self._history)


# ── Evolutionary Aggregator ──────────────────────────────────────────


class EvolutionaryAggregator:
    """Evolutionary aggregation engine for specialist outputs.

    CONCEPT:AHE-3.2 — Evolutionary Aggregation Engine

    Implements the core Squeeze Evolve evolutionary loop adapted for
    agent orchestration:

    1. Collects scored proposals from WorkspaceAttention (CONCEPT:ORCH-1.2).
    2. Groups proposals and computes group fitness (GC, D).
    3. Routes each group to the appropriate aggregation strategy.
    4. Detects convergence (diversity collapse) for early stopping.

    The engine is lightweight by default (pure Python scoring, no LLM
    calls) and becomes opt-in for expensive re-aggregation when the
    recommended strategy is ``HEAVY_MODEL``.

    Args:
        confidence_threshold: GC above this → ``MAJORITY_VOTE`` or
            ``LIGHT_MODEL``.
        diversity_threshold: D above this AND low confidence →
            ``HEAVY_MODEL``.
        group_size: Number of proposals per group (K in the paper).
            Default 2 for lightweight operation.
        population_size: Maximum number of proposals to consider (N
            in the paper).  Default 4.
    """

    def __init__(
        self,
        confidence_threshold: float = 0.7,
        diversity_threshold: int = 2,
        group_size: int = 2,
        population_size: int = 4,
    ) -> None:
        self.confidence_threshold = confidence_threshold
        self.diversity_threshold = diversity_threshold
        self.group_size = max(1, group_size)
        self.population_size = max(1, population_size)

    def compute_group_fitness(
        self,
        proposals: list[Proposal],
        group_id: str = "",
    ) -> GroupFitness:
        """Compute fitness metrics for a group of proposals.

        Args:
            proposals: List of scored proposals forming the group.
            group_id: Optional identifier for the group.

        Returns:
            A ``GroupFitness`` instance with computed GC, D, and
            recommended strategy.
        """
        if not proposals:
            return GroupFitness(
                group_id=group_id or "empty",
                group_confidence=0.0,
                group_diversity=0,
                recommended_strategy=AggregationStrategy.HEAVY_MODEL,
            )

        # GC(g) = (1/K) * Σ C(τ)  — Eq. 4
        gc = sum(p.confidence_score for p in proposals) / len(proposals)

        # D(g) = |{answer(τ) : τ ∈ g}|  — Eq. 5
        # Use first 200 chars of output as a diversity fingerprint
        unique_answers = {p.output[:200].strip().lower() for p in proposals}
        diversity = len(unique_answers)

        # Route aggregation strategy
        strategy = self._route_strategy(gc, diversity)

        return GroupFitness(
            group_id=group_id or f"group_{len(proposals)}",
            specialist_ids=[p.specialist_id for p in proposals],
            group_confidence=gc,
            group_diversity=diversity,
            recommended_strategy=strategy,
        )

    def group_proposals(
        self,
        proposals: list[Proposal],
    ) -> list[list[Proposal]]:
        """Split proposals into groups of ``group_size``.

        Args:
            proposals: All proposals to group, typically sorted by
                composite score from WorkspaceAttention.

        Returns:
            A list of proposal groups, each up to ``group_size`` in
            length.  The last group may be smaller.
        """
        limited = proposals[: self.population_size]
        groups: list[list[Proposal]] = []
        for i in range(0, len(limited), self.group_size):
            groups.append(limited[i : i + self.group_size])
        return groups

    def route_aggregation(
        self,
        groups: list[GroupFitness],
    ) -> dict[str, AggregationStrategy]:
        """Map each group to its recommended aggregation strategy.

        Args:
            groups: List of computed group fitness instances.

        Returns:
            Dictionary mapping group_id → AggregationStrategy.
        """
        return {g.group_id: g.recommended_strategy for g in groups}

    def _route_strategy(
        self,
        group_confidence: float,
        group_diversity: int,
    ) -> AggregationStrategy:
        """Determine the aggregation strategy from GC and D.

        Decision tree:
        - High confidence + low diversity → MAJORITY_VOTE (free)
        - High confidence + some diversity → LIGHT_MODEL (cheap)
        - Low confidence + high diversity → HEAVY_MODEL (expensive)
        """
        if group_confidence >= self.confidence_threshold and group_diversity <= 1:
            return AggregationStrategy.MAJORITY_VOTE
        elif group_confidence >= self.confidence_threshold:
            return AggregationStrategy.LIGHT_MODEL
        elif group_diversity >= self.diversity_threshold:
            return AggregationStrategy.HEAVY_MODEL
        else:
            return AggregationStrategy.LIGHT_MODEL
