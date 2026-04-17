#!/usr/bin/python
# coding: utf-8
"""Graph Executor Module.

This module implements the core logic for executing specialized agent nodes
within a pydantic-graph orchestration. It handles dynamic MCP tool binding,
domain-specific specialist logic, circuit breaker health checks, and
automated fallback strategies for resilience in production workflows.
"""

from __future__ import annotations

import os
import logging
import asyncio
from typing import Any, Tuple, List

from pydantic_ai import Agent

from ..models import (
    MCPAgent,
    MCPServerHealth,
    ExecutionStep,
)
from ..agent_factory import create_agent
from ..tool_filtering import filter_tools_by_tag
from .config_helpers import (
    load_node_agents_registry,
    get_discovery_registry,
    emit_graph_event,
    load_specialized_prompts,
    DEFAULT_GRAPH_TIMEOUT,
)
from .hsm import on_enter_specialist, on_exit_specialist, check_specialist_preconditions
from pydantic_graph import End
from pydantic_graph.beta import StepContext
from .state import GraphState, GraphDeps
from .runner import run_graph

logger = logging.getLogger(__name__)


def agent_matches_node_id(agent: MCPAgent, node_id: str) -> bool:
    """Multi-strategy agent name matching for approximate node IDs from the router.

    Handles cases where the LLM-generated node_id doesn't exactly match registry
    entries.  Tries exact match, substring match, prefix/suffix match, and keyword
    intersection against the agent's tag, name, server, and description fields.

    Args:
        agent: The MCPAgent registry entry to match against.
        node_id: The node identifier emitted by the router / executor.

    Returns:
        True if the agent is a plausible match for the given node_id.

    """
    node_id_norm = node_id.lower().replace("-", "_")
    name = agent.name.lower().replace("-", "_").replace(" ", "_")
    mcp_tools = (agent.mcp_tools or "").lower().replace("-", "_").replace(" ", "_")
    server = (agent.mcp_server or "").lower().replace("-", "_").replace(" ", "_")
    desc = (agent.description or "").lower()
    capabilities = [c.lower() for c in agent.capabilities]

    if (
        name == node_id_norm
        or mcp_tools == node_id_norm
        or server == node_id_norm
        or node_id_norm in capabilities
    ):
        return True

    if (name and name in node_id_norm) or (server and server in node_id_norm):
        return True

    if name and (node_id_norm.startswith(name) or node_id_norm.endswith(name)):
        return True
    if server and (node_id_norm.startswith(server) or node_id_norm.endswith(server)):
        return True

    stop_words = {"researcher", "expert", "agent", "manager", "action"}
    node_keywords = {
        w for w in node_id_norm.split("_") if len(w) >= 3 and w not in stop_words
    }
    if node_keywords:
        for kw in node_keywords:
            if kw in name or kw in desc:
                logger.debug(
                    f"Keyword match: '{kw}' found in name/desc of '{agent.name}'"
                )
                return True

    return False


async def _get_domain_tools(
    node_id: str, deps: GraphDeps
) -> Tuple[List[Any], List[Any]]:
    """Dynamically discover and load toolsets specialized for a domain expert.

    Starts with universal developer tools and augments them with domain-specific
    These tools are resolved by matching the node identifier against the
    unified specialist registry (NODE_AGENTS.md) to discover assigned
    capability tags and MCP server associations.

    Returns:
        A tuple containing (list of developer tools, list of specialized skill toolsets).

    """
    from ..tools.developer_tools import developer_tools
    from ..tools.sdd_tools import sdd_tools
    from .config_helpers import get_discovery_registry

    tools: list[Any] = list(developer_tools) + list(sdd_tools)
    toolsets: list[Any] = []

    registry = get_discovery_registry()
    agent = next((a for a in registry.agents if a.name == node_id), None)
    skill_tags = agent.capabilities if agent else []
    if not skill_tags:
        return tools, toolsets

    logger.debug(
        f"Loading {len(skill_tags)} specialized skill tags for '{node_id}': {skill_tags}"
    )

    try:
        from pydantic_ai_skills import SkillsToolset
        from ..workspace import get_skills_path

        skill_dirs: list[str] = []
        if skills_path := get_skills_path():
            skill_dirs.append(skills_path)

        try:
            from universal_skills.skill_utilities import get_universal_skills_path

            skill_dirs.extend(get_universal_skills_path())
        except ImportError:
            pass

        try:
            from skill_graphs.skill_graph_utilities import get_skill_graphs_path

            skill_dirs.extend(get_skill_graphs_path(default_enabled=True))
        except ImportError:
            pass

        if skill_dirs:
            from ..tool_filtering import skill_matches_tags

            filtered_dirs = [d for d in skill_dirs if skill_matches_tags(d, skill_tags)]
            if filtered_dirs:
                skills_toolset = SkillsToolset(directories=filtered_dirs)
                toolsets.append(skills_toolset)
                logger.info(
                    f"Loaded {len(filtered_dirs)} skill directories for '{node_id}'"
                )
    except ImportError:
        logger.debug("pydantic-ai-skills not installed; skipping skill injection")

    return tools, toolsets


def get_step_descriptions() -> str:
    """Generate a formatted catalog of expert capabilities for the LLM planner.

    Combines static roles, discovered A2A peers, and registered MCP specialists
    into a cohesive markdown list used in system prompts.  Uses the unified
    :func:`discover_all_specialists` roster so that both MCP and A2A sources
    are enumerated through the same code path.

    Returns:
        A multi-line markdown string describing all available graph nodes.

    """
    from ..discovery import discover_all_specialists

    steps = {
        "researcher": "Multi-vector discovery expert. Trigger this when information is missing or assumptions need validation. Can be spawned in parallel for simultaneous Web, Code, and Workspace research.",
        "architect": "System design expert. Analyzes requirements and defines high-level structures. Performs 'Gap Analysis' to identify missing context.",
        "planner": "Task orchestration expert. Bridges the gap between architecture and execution. Assesses missing knowledge and spawns researchers to validate assumptions.",
        "python_programmer": "Specialized Python engineer for implementation, refactoring, and standalone scripts.",
        "typescript_programmer": "Frontend and Node.js expert specializing in TypeScript and React ecosystems.",
        "javascript_programmer": "General-purpose JavaScript and web development specialist.",
        "rust_programmer": "Systems programming and memory safety expert.",
        "golang_programmer": "Cloud-native and high-performance backend expert.",
        "java_programmer": "Java/JVM and PHP/Laravel enterprise application developer.",
        "security_auditor": "Expert in threat modeling, vulnerability scanning, and secure coding practices.",
        "qa_expert": "Quality assurance lead. Designs test plans and implements automated test suites.",
        "ui_ux_designer": "Frontend design, CSS, and user interface expert.",
        "devops_engineer": "CI/CD, Docker, and infrastructure expert.",
        "database_expert": "SQL/NoSQL design and query optimization expert.",
        "data_scientist": "ML/data expert. NumPy, Pandas, Matplotlib, Scikit-learn, PyTorch, TensorFlow, HuggingFace, LangChain.",
        "document_specialist": "Document processing. PDFs, Office docs, Markdown conversion, Marp presentations, GIF/video creation.",
        "mobile_developer": "React Native and Expo mobile development expert.",
        "agent_engineer": "Meta-tooling for building agents, MCP servers, skills, and agent packages. Pydantic AI, FastMCP, A2A.",
        "project_manager": "Jira, GitHub workflows, Google Workspace, sprint planning, and internal communications.",
        "systems_admin": "Systems administration and home-lab. OS ops, Home Assistant, Uptime Kuma, self-hosted services.",
        "debugger_expert": "Interpreting error logs and fixing complex bugs.",
        "verifier": "Final quality gate. Validates that the implementation meets the original query requirements.",
        "mcp_server": "General-purpose tool hub for any task not covered by specialized nodes.",
    }

    for specialist in discover_all_specialists():
        if specialist.tag in steps:
            continue
        if specialist.source == "a2a":
            steps[specialist.tag] = (
                f"Remote A2A Specialist '{specialist.name}': {specialist.description} "
                f"(Capabilities: {specialist.capabilities or 'N/A'})"
            )
        else:
            tool_preview = ", ".join(specialist.tools[:5])
            steps[specialist.tag] = (
                f"MCP Agent '{specialist.name}': {specialist.description}. "
                f"Targeted expertise for: {tool_preview}..."
            )

    return "\n".join([f"- {k}: {v}" for k, v in steps.items()])


async def _execute_dynamic_mcp_agent(
    ctx: StepContext[GraphState, GraphDeps, ExecutionStep | Any], agent_info: MCPAgent
) -> str:
    """Execute a dynamically generated specialist agent from an MCP server registry.

    This implements a resilient execution protocol including:
    1. Precondition checks and circuit breaker validation.
    2. Dynamic binding of tagged MCP tools for the specific domain.
    3. LLM execution with per-node timeouts and exponential backoff retries.
    4. Data synthesis from raw tool results in case of partial success.
    5. Sideband event emission for real-time UI monitoring and transparency.

    Args:
        ctx: The pydantic-graph step context containing state and deps.
        agent_info: Metadata for the specialist to be executed (Registry entry).

    Returns:
        The identifier of the next graph node to execute (usually 'execution_joiner').

    Raises:
        RuntimeError: If all retries are exhausted or preconditions fail.

    """
    server_name = agent_info.mcp_server
    agent_name = agent_info.name

    # HSM: Entry action
    await on_enter_specialist(
        ctx_deps=ctx.deps,
        ctx_state=ctx.state,
        agent_name=agent_name,
        server_name=server_name,
    )

    # BT: Precondition guard - Check before committing to this specialist
    can_proceed, reason = check_specialist_preconditions(agent_info, ctx.deps)
    if not can_proceed:
        logger.warning(
            f"Precondition failed for '{agent_name}': {reason}. Attempting fallback."
        )
        await on_exit_specialist(
            ctx_deps=ctx.deps,
            ctx_state=ctx.state,
            agent_name=agent_name,
            success=False,
            server_name=server_name,
        )
        fallback_result = await _attempt_specialist_fallback(
            ctx=ctx, failed_agent=agent_info
        )
        if fallback_result:
            return fallback_result
        ctx.state.error = f"Precondition failed for '{agent_name}': {reason}"
        raise RuntimeError(ctx.state.error)

    logger.info(f"[LAYER:GRAPH:EXPERT] Running dynamic MCP agent '{agent_name}'")

    # 1. Look up discovery metadata for this server to help the agent "know" what it has
    discovered_tools = []
    logger.debug(
        f"Expert Execution: discovery_metadata keys={list(ctx.deps.discovery_metadata.keys()) if ctx.deps.discovery_metadata else 'EMTPY'}, "
        f"agent_info.tools={agent_info.tools}"
    )
    if hasattr(ctx.deps, "discovery_metadata") and ctx.deps.discovery_metadata:
        target_server = agent_info.mcp_server.lower()
        for s_id, tools in ctx.deps.discovery_metadata.items():
            if (
                s_id.lower() == target_server
                or s_id.lower().startswith(f"{target_server}-")
                or s_id.lower().startswith(f"{target_server}_")
            ):
                discovered_tools.extend(tools)

    # Merge with registry tools (fallback/augmentation)
    registry_tools = agent_info.tools or []
    total_tools = list(set(discovered_tools) | set(registry_tools))

    tool_list_str = ", ".join(total_tools) if total_tools else "NONE"

    # Build agent
    agent_sys_prompt = (
        f"{agent_info.system_prompt}\n\n"
        f"### STRICT DOMAIN EXPERT PROTOCOL\n"
        f"You are the SOLE authoritative expert for the '{agent_info.name}' domain. "
        f"You have access to the '{agent_info.mcp_server}' server tools.\n\n"
        f"## DATA SOURCE MANDATE (CRITICAL)\n"
        f"1. You MUST retrieve data ONLY from your available tools: [{tool_list_str}]\n"
        f"2. If the tool call returns an empty list, your answer MUST be: 'The tool returned no data for this query.'\n"
        f"3. If the tool call fails, you MUST report the exact error.\n"
        f"4. **NEVER** invent data (names, IDs, statuses, URLs). Hallucination is a SEVERE protocol violation.\n"
        f"5. **ALWAYS** include a detailed table or list of the RAW findings. Downstream automated systems (Verifiers) REQUIRE this data to pass your work.\n\n"
        f"IMPORTANT: You are currently asked to: {agent_info.description}\n"
        f"Query: {ctx.state.query}"
    )

    # Include validation feedback if this is a re-dispatch
    if ctx.state.validation_feedback:
        agent_sys_prompt += (
            f"\n\n### PREVIOUS FEEDBACK\n"
            f"Your previous output was reviewed and needs improvement:\n"
            f"{ctx.state.validation_feedback}\n"
            f"Address this feedback in your response by being more thorough or providing the missing data."
        )

    # Emit startup event with detailed metadata for UI transparency
    emit_graph_event(
        ctx.deps.event_queue,
        "expert_metadata",
        domain=agent_info.tag or "unknown",
        expert=agent_info.name,
        target_server=agent_info.mcp_server,
        domain_tag=agent_info.tag,
        expected_tools=total_tools,
        node_id=getattr(ctx, "node_id", "unknown"),
    )

    agent = Agent(
        model=ctx.deps.agent_model,
        system_prompt=agent_sys_prompt,
        deps_type=GraphDeps,
    )

    # Bind the specific subset of MCP tools (with deduplication)
    bound_tool_count = 0
    actually_bound_tools: list[str] = []
    matched_toolsets: list[Any] = []
    _seen_toolset_ids: set[int] = set()

    for toolset in ctx.deps.mcp_toolsets:
        ts_identity = id(toolset)
        if ts_identity in _seen_toolset_ids:
            continue

        server_id = getattr(toolset, "id", getattr(toolset, "name", None))
        if server_id:
            target = agent_info.mcp_server.lower().replace("-", "_")
            current = server_id.lower().replace("-", "_")

            if (
                current == target
                or current.startswith(f"{target}_")
                or target.startswith(f"{current}_")
            ):
                _seen_toolset_ids.add(ts_identity)
                matched_toolsets.append(toolset)

                if hasattr(toolset, "tools"):
                    for t_name in toolset.tools.keys():
                        if t_name not in actually_bound_tools:
                            actually_bound_tools.append(t_name)
                    bound_tool_count += len(toolset.tools)
                else:
                    bound_tool_count += len(total_tools)

                logger.info(
                    f"[LAYER:GRAPH:EXPERT] Bound toolset '{server_id}' to expert '{agent_info.name}'"
                )

    # Wrap MCP toolsets with the tool guard so sensitive tools are flagged
    # as 'unapproved' and trigger the DeferredToolRequests flow.
    from ..tool_guard import flag_mcp_tool_definitions, build_sensitive_tool_names

    guarded_toolsets = flag_mcp_tool_definitions(
        matched_toolsets, build_sensitive_tool_names()
    )

    # Include DeferredToolRequests in output type so the agent can defer
    # sensitive tool calls instead of failing.
    from pydantic_ai import DeferredToolRequests
    from typing import Union

    agent = Agent(
        model=ctx.deps.agent_model,
        system_prompt=agent_sys_prompt,
        deps_type=GraphDeps,
        toolset=guarded_toolsets,
        output_type=Union[str, DeferredToolRequests],
        end_strategy="early",
        output_retries=2,
    )

    # Tool-count telemetry: surfaces blind or overloaded specialists
    emit_graph_event(
        ctx.deps.event_queue,
        "tools_bound",
        expert=agent_info.name,
        count=bound_tool_count,
        tools=actually_bound_tools,
        toolset_count=len(matched_toolsets),
    )

    if bound_tool_count == 0:
        logger.warning(
            f"[TELEMETRY] Specialist '{agent_info.name}' has ZERO tools bound "
            f"(server='{agent_info.mcp_server}'). Agent will run blind."
        )
        emit_graph_event(
            ctx.deps.event_queue,
            "expert_warning",
            message=f"No tools bound for server '{agent_info.mcp_server}'. Agent may be blind.",
        )
    elif bound_tool_count > 50:
        logger.warning(
            f"[TELEMETRY] Specialist '{agent_info.name}' has {bound_tool_count} tools "
            f"bound — consider partitioning to reduce context overhead."
        )
    else:
        logger.info(
            f"[TELEMETRY] Specialist '{agent_info.name}': "
            f"{bound_tool_count} tools across {len(matched_toolsets)} toolset(s)"
        )

    # Build Query
    sub_query = ctx.state.query
    step_input = ctx.inputs
    if isinstance(step_input, ExecutionStep) and step_input.input_data:
        if isinstance(step_input.input_data, dict):
            sub_query = step_input.input_data.get("question", sub_query)
        elif isinstance(step_input.input_data, str):
            sub_query = step_input.input_data

    # Execute with Per-Node Timeout and Retries
    node_timeout = 120.0
    if isinstance(step_input, ExecutionStep):
        node_timeout = step_input.timeout

    # Retrieve cached message history for re-dispatch context
    cache_key = agent_info.name.lower().replace(" ", "_")
    prev_messages = ctx.deps.message_history_cache.get(cache_key)

    max_attempts = 3
    last_error = None

    for attempt in range(max_attempts):
        emit_graph_event(
            ctx.deps.event_queue,
            "expert_thinking",
            expert=agent_info.name,
            attempt=attempt + 1,
        )
        try:
            logger.info(
                f"[LAYER:GRAPH:EXPERT] '{agent_info.name}' LLM Call Starting (attempt {attempt+1}). Prompt length: {len(agent_sys_prompt)}"
            )
            run_input = (
                ctx.state.query_parts
                if ctx.state.query_parts and sub_query == ctx.state.query
                else sub_query
            )
            res = await asyncio.wait_for(
                agent.run(run_input, deps=ctx.deps, message_history=prev_messages),
                timeout=node_timeout,
            )
            logger.info(f"[LAYER:GRAPH:EXPERT] '{agent_info.name}' LLM Call Completed.")
            ctx.state._update_usage(getattr(res, "usage", None))

            # Cache message history for potential re-dispatch
            try:
                ctx.deps.message_history_cache[cache_key] = res.all_messages()
            except Exception as e:
                logger.debug(f"Failed to update cache for '{cache_key}': {e}")
                pass

            # Record success on circuit breaker
            if server_name not in ctx.deps.server_health:
                ctx.deps.server_health[server_name] = MCPServerHealth(
                    server_name=server_name,
                )
            ctx.deps.server_health[server_name].record_success()

            # Stream events to WebUI
            if ctx.deps.event_queue:
                from pydantic_ai.messages import (
                    ModelRequest,
                    ModelResponse,
                    ToolCallPart,
                    ToolReturnPart,
                )

                for msg in res.all_messages():
                    if isinstance(msg, ModelResponse):
                        for part in msg.parts:
                            if isinstance(part, ToolCallPart):
                                emit_graph_event(
                                    ctx.deps.event_queue,
                                    "expert_tool_call",
                                    domain=agent_info.tag or "unknown",
                                    tool_name=part.tool_name,
                                    args=part.args,
                                )
                            elif hasattr(part, "content") and part.content:
                                emit_graph_event(
                                    ctx.deps.event_queue,
                                    "expert_text",
                                    domain=agent_info.tag or "unknown",
                                    content=part.content,
                                )
                    elif isinstance(msg, ModelRequest):
                        for part in msg.parts:
                            if isinstance(part, ToolReturnPart):
                                emit_graph_event(
                                    ctx.deps.event_queue,
                                    event_type="tool_result",
                                    agent=agent_info.name,
                                    tool=part.tool_name,
                                    result=str(part.content)[:500],
                                )
            # Unified result storage: write to results_registry (primary, read by dispatcher/verifier)
            # and mirror to results (keyed by domain tag, for backwards compatibility).
            result_str = str(res.output)

            # Data Enhancement Synthesizer: Synthesize response from real output tool data captured
            if "no data" in result_str.lower() or "returned no" in result_str.lower():
                from pydantic_ai.messages import ModelRequest, ToolReturnPart

                tool_returns = []
                for msg in res.all_messages():
                    if isinstance(msg, ModelRequest):
                        for part in msg.parts:
                            if isinstance(part, ToolReturnPart) and part.content:
                                content_str = str(part.content)
                                if content_str and content_str not in (
                                    "[]",
                                    "None",
                                    "null",
                                    "",
                                ):
                                    tool_returns.append(
                                        f"**{part.tool_name}**: {content_str}"
                                    )
                if tool_returns:
                    logger.warning(
                        f"Expert '{agent_info.name}': LLM dismissed tool response data. "
                        f"Injection {len(tool_returns)} raw tool return(s) into result."
                    )
                    result_str = (
                        "### Tool Execution Results\n"
                        + "\n".join(tool_returns)
                        + f"\n\n### Agent Summary\n{result_str}"
                    )
            node_uid = f"{cache_key}_{ctx.state.step_cursor}"
            ctx.state.results_registry[node_uid] = result_str
            result_key = agent_info.tag or cache_key
            ctx.state.results[result_key] = result_str
            ctx.state.routed_domain = result_key
            logger.info(
                f"Expert: '{agent_info.name}' succeeded (attempt {attempt + 1}). "
                f"Result: {len(result_str)} chars. Registry key: '{node_uid}'"
            )
            # Emit completion event
            emit_graph_event(
                ctx.deps.event_queue,
                "subagent_completed",
                domain=agent_info.tag or "unknown",
                status="success",
            )
            # HSM: Exit action (success)
            await on_exit_specialist(
                ctx_deps=ctx.deps,
                ctx_state=ctx.state,
                agent_name=agent_name,
                success=True,
                server_name=server_name,
            )
            return "execution_joiner"

        except asyncio.TimeoutError:
            last_error = f"Timeout after {node_timeout}s"
            logger.warning(
                f"Expert '{agent_name}' timed out (attempt {attempt + 1}/{max_attempts})"
            )
            emit_graph_event(
                ctx.deps.event_queue,
                "expert_complete",
                expert=agent_info.name,
                status="timeout",
            )
        except Exception as e:
            last_error = str(e)
            logger.warning(
                f"Expert '{agent_name}' failed (attempt {attempt + 1}/{max_attempts}): {e}"
            )
            emit_graph_event(
                ctx.deps.event_queue,
                "subagent_tool_call",
                domain=agent_info.tag or "unknown",
                tool_name=getattr(e, "tool_name", "unknown"),
                args=getattr(e, "args", {}),
            )
            emit_graph_event(
                ctx.deps.event_queue,
                "expert_complete",
                expert=agent_info.name,
                status="error",
                error=str(e),
            )

        # Exponential backoff between retries
        if attempt < max_attempts - 1:
            backoff = min(2**attempt, 10)
            await asyncio.sleep(backoff)

    # All retries exhausted
    # HSM: Exit action (failure)
    await on_exit_specialist(
        ctx_deps=ctx.deps,
        ctx_state=ctx.state,
        agent_name=agent_name,
        success=False,
        server_name=server_name,
    )

    # Try fallback specialist from same server
    fallback_result = await _attempt_specialist_fallback(ctx, agent_info)
    if fallback_result:
        return fallback_result

    ctx.state.error = (
        f"Agent '{agent_name}' failed after {max_attempts} attempts: {last_error}"
    )
    raise RuntimeError(ctx.state.error)


async def _attempt_specialist_fallback(
    ctx: StepContext[GraphState, GraphDeps, ExecutionStep | Any],
    failed_agent: MCPAgent,
) -> str | None:
    """Implement a resilience strategy by falling back to sibling specialists.

    If a targeted expert fails or is unavailable, this helper searches for
    other specialists from the same MCP server. It scores candidate siblings
    using keyword intersection between the query and the specialist tags.

    Args:
        ctx: The pydantic-graph step context containing state and deps.
        failed_agent: The metadata of the expert that failed.

    Returns:
        The identifier of the joiner node if a fallback succeeded, or None if
        no suitable fallback could be identified or executed.

    """
    registry = load_node_agents_registry()
    siblings = [
        a
        for a in registry.agents
        if a.mcp_server == failed_agent.mcp_server
        and a.tag != failed_agent.tag
        and a.name != failed_agent.name
    ]

    if not siblings:
        return None

    # Score siblings by keyword overlap with the query
    query_words = set(ctx.state.query.lower().split())
    best_sibling = None
    best_score = 0

    for sibling in siblings:
        tag_words = set(sibling.tag.lower().replace("-", " ").replace("_", " ").split())
        score = len(query_words & tag_words)
        if score > best_score:
            best_score = score
            best_sibling = sibling

    if best_sibling and best_score > 0:
        logger.info(
            f"Fallback: Trying sibling '{best_sibling.name}' fallback for '{failed_agent.name}'.\nScore: {best_score}"
        )
        emit_graph_event(
            ctx.deps.event_queue,
            event_type="specialist_fallback",
            failed=failed_agent.name,
            fallback=best_sibling.name,
        )
        try:
            return await _execute_dynamic_mcp_agent(ctx, best_sibling)
        except Exception as e:
            logger.warning(f"Fallback '{best_sibling.name}' also failed: {e}")

    return None


async def _execute_agent_package_logic(
    ctx: StepContext[GraphState, GraphDeps, ExecutionStep | Any],
    node_id: str,
    meta: dict,
) -> str:
    """Execute specialized logic for a discovered agent package.

    This function handles the dispatch logic for two primary agent types:
    - Remote A2A Agents: Delegated via HTTP/SSE using the A2AClient.
    - Local Dynamic MCP Agents: Managed via the node agent registry and
      dynamic tool binding.

    Args:
        ctx: The pydantic-graph step context containing the execution state.
        node_id: The identifier of the agent package (e.g., 'github').
        meta: Discovery metadata for the agent (type, URL, description).

    Returns:
        The identifier of the joiner node ('execution_joiner') after completion.

    """
    deps = ctx.deps

    if meta.get("type") == "remote_a2a":
        # Remote A2A Execution
        from ..a2a import A2AClient

        peer_url = meta["url"]
        logger.info(
            f"Expert Execution: Calling remote A2A agent '{node_id}' at {peer_url}"
        )
        client = A2AClient(
            timeout=deps.approval_timeout or 300.0, ssl_verify=deps.ssl_verify
        )

        # Use the expert's specific question or the original query
        sub_query = ctx.state.query
        step_input = ctx.inputs
        if isinstance(step_input, ExecutionStep) and step_input.input_data:
            if isinstance(step_input.input_data, dict):
                sub_query = step_input.input_data.get("question", sub_query)
            elif isinstance(step_input.input_data, str):
                sub_query = step_input.input_data

        res_content = await client.execute_task(peer_url, sub_query)
        result_str = str(res_content)
        # Unified result storage
        node_uid = f"{node_id}_{ctx.state.step_cursor}"
        ctx.state.results_registry[node_uid] = result_str
        ctx.state.results[node_id] = result_str
    elif meta.get("type") == "prompt":
        # Execute Prompt-based logic
        return await _execute_specialized_step(ctx, node_id)
    else:
        registry = load_node_agents_registry()
        mcp_agent = next(
            (a for a in registry.agents if agent_matches_node_id(a, node_id)),
            None,
        )

        if mcp_agent:
            await _execute_dynamic_mcp_agent(ctx, mcp_agent)
        else:
            # Fallback to generic expert node with all tools if metadata is missing
            logger.warning(
                f"Expert Execution: Node '{node_id}' fallback. No NODE_AGENTS.md metadata found."
            )
            await _execute_domain_logic(ctx, node_id)

    return "execution_joiner"


async def agent_package_step(
    ctx: StepContext[GraphState, GraphDeps, ExecutionStep | Any],
    node_id: str,
) -> str:
    """Graph node step wrapper for agent package execution.

    This acts as the standardized entry point for all specialist agent nodes
    discovered during the bootstrap phase.

    Args:
        ctx: The pydantic-graph step context.
        node_id: The identifier of the package to execute.

    Returns:
        The next node identifier (usually 'execution_joiner' or 'Error').

    """
    from ..discovery import discover_agents

    discovered = discover_agents()
    if node_id not in discovered:
        logger.error(f"Agent package node '{node_id}' not found in discovery.")
        return "Error"

    meta = discovered[node_id]
    return await _execute_agent_package_logic(ctx, node_id, meta)


async def _execute_specialized_step(
    ctx: StepContext[GraphState, GraphDeps, None], prompt_name: str
) -> str | End[Any]:
    """Execute a specialized expert role using structured prompts and tool injection.

    This implements core functional layers (e.g., 'Programmers', 'Security',
    'QA') by loading persona-specific prompts and binding matching MCP
    toolsets based on tag compatibility.

    Args:
        ctx: The pydantic-graph step context containing shared state.
        prompt_name: The name of the specialized role/prompt to load from
            the prompts directory.

    Returns:
        The next node identifier (usually 'execution_joiner') or a terminal
        End state with a GraphResponse.

    """
    from ..models import GraphResponse

    # HSM: Entry action
    await on_enter_specialist(
        ctx_deps=ctx.deps, ctx_state=ctx.state, agent_name=prompt_name
    )

    prompt = load_specialized_prompts(prompt_name)

    # Dynamic Skill Distribution
    custom_tools, skill_toolsets = await _get_domain_tools(prompt_name, ctx.deps)
    logger.info(
        f"[LAYER:GRAPH:EXPERT] Specialized step '{prompt_name}' started. Tools loaded: {len(custom_tools)}, Toolsets: {len(skill_toolsets)}"
    )

    memory_instruction = load_specialized_prompts("memory_instruction")
    tool_guidance = load_specialized_prompts("tool_guidance")

    # Include validation feedback if this is a re-dispatch from verifier
    feedback_section = ""
    if ctx.state.validation_feedback:
        feedback_section = (
            f"\n\n###PREVIOUS FEEDBACK\n"
            f"Your previous output was reviewed and needs improvement:\n"
            f"{ctx.state.validation_feedback}\n"
            f"Address this feedback in your response."
        )

    # Filter MCP toolsets by domain tag AND node_id (prompt_name) with deduplication.
    # Wrap each with the tool guard so sensitive tools trigger approval.
    from ..tool_guard import flag_mcp_tool_definitions, build_sensitive_tool_names

    _sensitive_names = build_sensitive_tool_names()

    # Resolve tags from the unified registry instead of the deprecated NODE_SKILL_MAP
    registry = get_discovery_registry()
    agent_info = next((a for a in registry.agents if a.name == prompt_name), None)
    tool_tags = [prompt_name]
    if agent_info and agent_info.capabilities:
        # Capabilities field in NODE_AGENTS.md corresponds to skill tags
        tool_tags.extend(agent_info.capabilities)

    tool_tags = list(set(tool_tags))
    _seen_ts: set[int] = set()
    mcp_tool_count = 0
    collected_mcp_toolsets: list[Any] = []
    for toolset in ctx.deps.mcp_toolsets:
        ts_identity = id(toolset)
        if ts_identity in _seen_ts:
            continue

        server_id = (
            getattr(toolset, "id", getattr(toolset, "name", "unknown"))
            .lower()
            .replace("-", "_")
        )
        target = prompt_name.lower().replace("-", "_")

        if server_id == target or any(
            t.lower().replace("-", "_") == target for t in tool_tags
        ):
            _seen_ts.add(ts_identity)
            guarded = flag_mcp_tool_definitions([toolset], _sensitive_names)
            collected_mcp_toolsets.append(guarded[0])
            mcp_tool_count += len(getattr(toolset, "tools", {})) or 1
        else:
            filtered = filter_tools_by_tag(toolset, tool_tags)
            if filtered:
                fid = id(filtered)
                if fid not in _seen_ts:
                    _seen_ts.add(fid)
                    guarded = flag_mcp_tool_definitions([filtered], _sensitive_names)
                    collected_mcp_toolsets.append(guarded[0])
                    mcp_tool_count += len(getattr(filtered, "tools", {})) or 1

    # Build the agent with ALL toolsets at construction time.
    # agent.toolsets is a read-only property — appending after construction
    # is a silent no-op.
    from pydantic_ai import DeferredToolRequests
    from typing import Union

    agent = Agent(
        model=ctx.deps.agent_model,
        system_prompt=(
            f"{memory_instruction}\n\n"
            f"{prompt}\n\n"
            f"### TOOL USAGE GUIDANCE\n{tool_guidance}\n\n"
            f"### CONTEXT\n{ctx.state.exploration_notes}"
            f"{feedback_section}"
        ),
        tools=custom_tools,
        toolsets=collected_mcp_toolsets + skill_toolsets,
        output_type=Union[str, DeferredToolRequests],
    )

    # Tool-count telemetry for specialized steps
    total_tool_count = len(custom_tools) + mcp_tool_count
    logger.info(
        f"[TELEMETRY] Specialist '{prompt_name}': "
        f"{len(custom_tools)} dev/skill tools + {mcp_tool_count} MCP tools "
        f"= {total_tool_count} total"
    )

    emit_graph_event(
        ctx.deps.event_queue,
        "tools_bound",
        expert=prompt_name,
        count=total_tool_count,
        dev_tools=len(custom_tools),
        mcp_tools=mcp_tool_count,
    )
    if total_tool_count == 0:
        logger.warning(
            f"[TELEMETRY] Specialist '{prompt_name}' has ZERO tools. Agent will run blind."
        )
    elif total_tool_count > 50:
        logger.warning(
            f"[TELEMETRY] Specialist '{prompt_name}' has {total_tool_count} tools "
            f"— consider partitioning to reduce context overhead."
        )

    # Retrieve cached message history for re-dispatch context
    prev_messages = ctx.deps.message_history_cache.get(prompt_name)

    try:
        run_input = ctx.state.query_parts if ctx.state.query_parts else ctx.state.query
        async with agent.run_stream(
            run_input,
            message_history=prev_messages,
        ) as stream:
            async for chunk in stream.stream_text(delta=True):
                emit_graph_event(
                    ctx.deps.event_queue,
                    "agent_node_delta",
                    content=chunk,
                    node=prompt_name,
                )
            res = await stream.get_output()
        usage = stream.usage()
        if asyncio.iscoroutine(usage):
            usage = await usage
        ctx.state._update_usage(usage)
        result_str = str(res)

        # Unified result storage: write to results_registry (primary, read by dispatcher/verifier)
        # and mirror to results (keyed by domain, for backwards compatibility).
        node_uid = f"{prompt_name}_{ctx.state.step_cursor}"
        ctx.state.results_registry[node_uid] = result_str
        ctx.state.results[prompt_name] = result_str

        logger.info(
            f"Specialized step '{prompt_name}': stored result ({len(result_str)} chars) "
            f"at registry key '{node_uid}'"
        )

        # Cache message history for potential re-dispatch
        try:
            ctx.deps.message_history_cache[prompt_name] = stream.all_messages()
        except Exception as e:
            logger.debug(f"Unable to cache: {e}")

        # HSM: Exit action (success)
        await on_exit_specialist(
            ctx_deps=ctx.deps,
            ctx_state=ctx.state,
            agent_name=prompt_name,
            success=True,
        )

        # In Dynamic Plan mode, return to execution_joiner for barrier synchronization
        if ctx.state.plan and ctx.state.plan.steps:
            return "execution_joiner"

        # Standalone mode (no plan): wrap and terminate
        return End(
            GraphResponse(
                status="completed",
                results={"output": result_str},
                metadata={"domain": prompt_name},
            )
        )

    except Exception as e:
        # HSM: Exit action (failure)
        await on_exit_specialist(
            ctx_deps=ctx.deps,
            ctx_state=ctx.state,
            agent_name=prompt_name,
            success=False,
        )
        logger.error(f"Specialized step '{prompt_name}' failed: {e}")
        return "error_recovery"


async def _execute_domain_logic(
    ctx: StepContext[GraphState, GraphDeps, str], domain: str
):
    """Core logic to execute a domain-specific agent or sub-graph.

    This implements the 'Data & Lifestyle' and 'Media & HomeLab' layers of
    the ecosystem. It handles environment-based tool activation, local
    delegation to agent packages, and automated fallback to generic
    expert agents for unspecified domains.

    Args:
        ctx: The pydantic-graph step context.
        domain: The domain identifier to execute (e.g., 'home_assistant').

    Returns:
        The identifier of the next node (usually 'execution_joiner'), a
        terminal End state if approval is required, or 'error_recovery'.

    """
    deps = ctx.deps
    domain_prompt = deps.tag_prompts.get(
        domain, f"You are a specialized assistant for the '{domain}' domain."
    )

    logger.info(f"domain_step executing logic for domain='{domain}'")

    original_env = {}
    for tag, env_var in deps.tag_env_vars.items():
        original_env[env_var] = os.environ.get(env_var)
        os.environ[env_var] = "True" if tag == domain else "False"

    try:
        domain_mcp_toolsets = []
        for toolset in deps.mcp_toolsets:
            if toolset is None:
                continue
            filtered = filter_tools_by_tag(toolset, domain)
            domain_mcp_toolsets.append(filtered)

        sub_agent_target = deps.sub_agents.get(domain)

        if sub_agent_target:
            try:
                target = sub_agent_target
                if isinstance(target, dict) and "tags" in target:
                    target = create_agent(
                        name=domain,
                        system_prompt=target.get(
                            "description", f"Specialized assistant for {domain}"
                        ),
                        enable_skills=True,
                        load_universal_skills=True,
                        load_skill_graphs=True,
                        tool_tags=target["tags"],
                        tool_guard_mode="off",
                    )
                elif isinstance(target, str):
                    # Legacy package-based delegation is deprecated
                    raise RuntimeError(
                        f"Legacy delegation to package '{target}' is deprecated. "
                        "Use the MCPAgent pattern or provide an Agent instance."
                    )

                if isinstance(target, tuple) and len(target) == 2:
                    sub_graph, sub_config = target
                    res = await run_graph(
                        graph=sub_graph,
                        config=sub_config,
                        query=ctx.state.query,
                        eq=deps.event_queue,
                    )
                    output = res.get("results") or res.get("error")
                else:
                    emit_graph_event(
                        deps.event_queue, "subagent_started", domain=domain, type="flat"
                    )
                    run_input = (
                        ctx.state.query_parts
                        if ctx.state.query_parts
                        else ctx.state.query
                    )
                    async with target.run_stream(run_input) as stream:
                        async for message, last in stream.stream_messages():
                            emit_graph_event(
                                deps.event_queue,
                                "subagent_thought",
                                domain=domain,
                                message=str(message),
                            )
                        res = await stream.get_output()
                    output = res

                result_str = str(output)
                # Unified result storage
                node_uid = f"{domain}_{ctx.state.step_cursor}"
                ctx.state.results_registry[node_uid] = result_str
                ctx.state.results[domain] = result_str
            except Exception as e:
                logger.error(f"domain_step delegation error for '{domain}': {e}")
                ctx.state.results[domain] = f"Delegation Error: {e}"
        else:
            query = ctx.state.query
            if ctx.state.validation_feedback:
                query = f"{query}\n\n[SELF-CORRECTION FEEDBACK]: {ctx.state.validation_feedback}"

            sub_agent = create_agent(
                provider=deps.provider,
                model_id=deps.agent_model,
                base_url=deps.base_url,
                api_key=deps.api_key,
                mcp_toolsets=deps.mcp_toolsets,
                tool_tags=[domain],
                name=f"Graph-{domain}",
                system_prompt=domain_prompt,
                ssl_verify=deps.ssl_verify,
                tool_guard_mode="off",
            )

            emit_graph_event(deps.event_queue, "subagent_started", domain=domain)

            run_input = (
                ctx.state.query_parts
                if ctx.state.query_parts and query == ctx.state.query
                else query
            )

            # If an approval manager is available, use the transparent
            # approval loop that pauses the graph and waits for user
            # decisions.  Otherwise, fall back to the legacy behaviour
            # that terminates the graph on DeferredToolRequests.
            if deps.approval_manager is not None:
                from ..approval_manager import run_with_approvals

                result = await run_with_approvals(
                    sub_agent,
                    run_input,
                    approval_manager=deps.approval_manager,
                    event_queue=deps.event_queue,
                    request_id_prefix=f"{domain}_",
                    approval_timeout=deps.approval_timeout,
                )
                output = getattr(result, "output", None) or getattr(
                    result, "data", result
                )
            else:
                result = await asyncio.wait_for(
                    sub_agent.run(run_input),
                    timeout=DEFAULT_GRAPH_TIMEOUT / 1000.0,
                )
                output = getattr(result, "output", None) or getattr(
                    result, "data", result
                )
                from pydantic_ai import DeferredToolRequests

                if isinstance(output, DeferredToolRequests):
                    ctx.state.human_approval_required = True
                    ctx.state.results[domain] = output
                    emit_graph_event(
                        deps.event_queue,
                        event_type="approval_required",
                        domain=domain,
                        tool_calls=[
                            (tc.model_dump() if hasattr(tc, "model_dump") else str(tc))
                            for tc in (getattr(output, "calls", []) or [])
                        ],
                    )
                    return End(output)

            result_str = str(output)
            node_uid = f"{domain}_{ctx.state.step_cursor}"
            ctx.state.results_registry[node_uid] = result_str
            ctx.state.results[domain] = result_str
            emit_graph_event(deps.event_queue, "subagent_completed", domain=domain)

    except Exception as e:
        logger.error(f"domain_step error for '{domain}': {e}")
        ctx.state.error = f"Domain failed: {e}"
        ctx.state.results[domain] = f"Error: {e}"
        return "error_recovery"
    finally:
        for env_var, value in original_env.items():
            if value is None:
                os.environ.pop(env_var, None)
            else:
                os.environ[env_var] = value
    return None
