#!/usr/bin/python
from __future__ import annotations

"""Graph Executor Module.

This module implements the core logic for executing specialized agent nodes
within a pydantic-graph orchestration. It handles dynamic MCP tool binding,
domain-specific specialist logic, circuit breaker health checks, and
automated fallback strategies for resilience in production workflows.
"""


import asyncio
import logging
import os
from typing import Any, cast

from pydantic_ai import Agent, DeferredToolRequests
from pydantic_graph import End

from agent_utilities.core.config import setting

try:
    from pydantic_graph.step import StepContext
except ImportError:
    from pydantic_graph.beta import StepContext


from agent_utilities.core.config import (
    DEFAULT_GRAPH_TIMEOUT,
    emit_graph_event,
    get_discovery_registry,
    load_node_agents_registry,
    load_specialized_prompts,
)
from agent_utilities.tools.tool_filtering import filter_tools_by_tag

from ..models import (
    ExecutionStep,
    MCPAgent,
    MCPServerHealth,
)
from ..orchestration.resilience import (
    DEFAULT_RETRYABLE,
    ResiliencePolicy,
    run_with_resilience,
)
from .hsm import check_specialist_preconditions, on_enter_specialist, on_exit_specialist
from .protocol_agnostic_execution import execute_graph
from .state import (  # noqa: F401 — GraphState re-exported for tests
    GraphDeps,
    GraphState,
)

logger = logging.getLogger(__name__)


# Simple per-node-id tier heuristic used when the Knowledge Graph registry
# does not carry an explicit ``default_tier``. Keeps parity with the
# description in AGENTS.md: cheap/fast models for discovery adaptive_agent_router,
# heavy/reasoning models for the planner and synthesizer.
_SPECIALIST_TIER_HINTS: dict[str, str] = {
    "researcher": "light",
    "web_researcher": "light",
    "code_researcher": "light",
    "workspace_researcher": "light",
    "simple_tool": "light",
    "planner": "heavy",
    "architect": "heavy",
    "synthesizer": "heavy",
    "verifier": "reasoning",
    "recursive_orchestrator": "heavy",
}


def _specialist_resilience_policy(node_timeout: float) -> ResiliencePolicy:
    """CONCEPT:AU-ORCH.execution.retry-predicate-raised-treating — Declarative Resilience Policy for specialist runs.

    Builds the default policy applied to a single specialist LLM call on the
    live execution path (:func:`_execute_dynamic_mcp_agent`). The policy retries
    only transient model/tool errors (``TimeoutError``/``ConnectionError`` —
    never ``ValueError``/permission errors), uses short exponential backoff with
    jitter, and enforces ``node_timeout`` as the per-attempt timeout. It composes
    with the outer attempt loop, the per-server circuit breaker, and the existing
    sibling-specialist fallback — it does not replace them.
    """
    return ResiliencePolicy(
        max_attempts=2,
        backoff_base_s=0.5,
        backoff_factor=2.0,
        max_backoff_s=5.0,
        jitter=True,
        retry_on=DEFAULT_RETRYABLE,
        timeout_s=node_timeout,
        name="specialist_execution",
    )


def _resolve_access_context(
    step: ExecutionStep,
    results_registry: dict[str, Any],
) -> str:
    """CONCEPT:AU-ORCH.execution.visibility-allow-list — Build context string from access_list.

    Filters the results_registry to only include outputs from steps
    specified in the ExecutionStep's access_list.  This prevents
    context pollution and reduces prompt bloat.

    Args:
        step: The current execution step with its access_list.
        results_registry: The full results registry from GraphState.

    Returns:
        A formatted context string with only the permitted results.
    """
    if not step.access_list:
        return ""  # No prior context shared

    if "all" in step.access_list:
        # Full access — inject everything
        if not results_registry:
            return ""
        return "\n".join(
            f"### Prior result from '{k}':\n{v}" for k, v in results_registry.items()
        )

    # Selective access — only specified steps
    sections: list[str] = []
    for key in step.access_list:
        for reg_key, value in results_registry.items():
            if key.lower() in reg_key.lower():
                sections.append(f"### Prior result from '{reg_key}':\n{value}")
    return "\n".join(sections)


def _default_tier_for(node_id: str) -> str:
    """Infer the default routing tier for a specialist by name.

    The heuristic is intentionally small; richer tiers come from
    ``MCPAgent.default_tier`` when the registry is populated.
    """
    return _SPECIALIST_TIER_HINTS.get(node_id, "medium")


def pick_specialist_model(
    ctx_deps: Any, node_id: str, step_model_id: str | None = None
) -> Any:
    """Pick the model to use when spawning the specialist ``node_id``.

    Resolution order:

    1. If ``ctx_deps.requested_model_id`` is set AND the id resolves
       inside ``ctx_deps.model_registry``, use that model verbatim — this
       is the per-turn override sourced from the ``x-agent-model-id``
       header and wins over tier-based routing.
    2. If ``ctx_deps.model_registry`` is populated, consult the discovery
       registry for a ``default_tier`` / ``required_tags`` hint on the
       specialist; fall back to the heuristic ``_default_tier_for`` map.
       Call :meth:`ModelRegistry.pick_for_task` and build a concrete
       pydantic-ai model via :func:`create_model`.
    3. Otherwise return ``ctx_deps.agent_model`` (the single graph-wide
       default) so behaviour is unchanged when no registry is configured.

    The function never raises on lookup problems: if anything goes wrong,
    it logs a warning and returns the default ``agent_model``.
    """
    registry = getattr(ctx_deps, "model_registry", None)
    if registry is None or not getattr(registry, "models", None):
        return ctx_deps.agent_model

    # CONCEPT:AU-ORCH.routing.conductor-per-step-model — a Conductor-assigned per-step model_id wins over both the
    # per-turn header override and tier routing (the Conductor explicitly chose it).
    if step_model_id:
        chosen = registry.get_by_id(step_model_id)
        if chosen is not None:
            try:
                from agent_utilities.core.model_factory import create_model

                api_key = setting(chosen.api_key_env) if chosen.api_key_env else None
                logger.info(
                    "Spawning specialist '%s' with Conductor-assigned model '%s'",
                    node_id,
                    chosen.id,
                )
                return create_model(
                    provider=chosen.provider,
                    model_id=chosen.model_id,
                    base_url=chosen.base_url,
                    api_key=api_key,
                )
            except Exception as e:
                logger.warning(
                    "Conductor model '%s' failed to build; falling back: %s",
                    step_model_id,
                    e,
                )
        else:
            logger.debug(
                "Conductor model id '%s' not in registry; using override/tier routing",
                step_model_id,
            )

    requested_id = getattr(ctx_deps, "requested_model_id", None)
    if requested_id:
        chosen = registry.get_by_id(requested_id)
        if chosen is not None:
            try:
                from agent_utilities.core.model_factory import create_model

                api_key = setting(chosen.api_key_env) if chosen.api_key_env else None
                logger.info(
                    "Spawning specialist '%s' with user-requested model '%s'",
                    node_id,
                    chosen.id,
                )
                return create_model(
                    provider=chosen.provider,
                    model_id=chosen.model_id,
                    base_url=chosen.base_url,
                    api_key=api_key,
                )
            except Exception as e:
                logger.warning(
                    "Requested model '%s' failed to build; falling back: %s",
                    requested_id,
                    e,
                )
        else:
            logger.debug(
                "Requested model id '%s' not found in registry; using tier routing",
                requested_id,
            )

    tier = _default_tier_for(node_id)
    required_tags: list[str] = []
    try:
        reg = get_discovery_registry()
        agent_info = next((a for a in reg.agents if a.name == node_id), None)
        if agent_info is not None:
            tier = getattr(agent_info, "default_tier", tier) or tier
            required_tags = list(getattr(agent_info, "required_tags", []) or [])
    except Exception as e:
        logger.debug(f"Registry tier lookup failed for '{node_id}': {e}")

    # CONCEPT:AU-OS.state.homeostatic-model-downgrade — Homeostatic Model Downgrade
    # When the ResourceOptimizer detects budget pressure, autonomously
    # downgrade the tier to reduce cost — the system's "blood pressure"
    # regulation.  The optimizer's select_model_for_step() already knows
    # how to map remaining_pct → effective_complexity; we just need to
    # ask it and let it override our heuristic tier.
    resource_optimizer = getattr(ctx_deps, "resource_optimizer", None)
    if resource_optimizer is not None:
        try:
            optimized = resource_optimizer.select_model_for_step(
                complexity=tier,
                required_tags=required_tags or None,
            )
            if optimized is not None:
                # The optimizer returned a model dict — it handled tier
                # adjustment itself.  We log the homeostatic event.
                effective_tier = optimized.get("tier", tier)
                if effective_tier != tier:
                    logger.info(
                        "[CONCEPT:AU-OS.state.homeostatic-model-downgrade] Homeostatic downgrade: '%s' tier %s → %s "
                        "(budget %.0f%% remaining)",
                        node_id,
                        tier,
                        effective_tier,
                        resource_optimizer.budget.cost_remaining
                        / max(resource_optimizer.budget.total_cost_budget_usd, 0.01)
                        * 100,
                    )
                    tier = effective_tier
        except Exception as e:
            logger.debug(f"CONCEPT:AU-OS.state.homeostatic-model-downgrade homeostatic check skipped: {e}")

    # CONCEPT:AU-ORCH.adapter.hot-cache-invalidation — Confidence-Gated Model Router
    # Compute a confidence signal from upstream scoring to adaptively
    # select cheaper or more expensive models.  Composes with CONCEPT:AU-OS.state.homeostatic-model-downgrade:
    # budget pressure adjusts the tier first, then confidence further
    # refines within the budget-allowed range.
    confidence_signal: float | None = None
    routing_percentile = getattr(ctx_deps, "routing_percentile", 50.0)

    # Source 1: WorkspaceAttention attention score (runtime signal)
    knowledge_engine = getattr(ctx_deps, "knowledge_engine", None)
    runtime_confidence = 0.5
    if knowledge_engine is not None:
        try:
            from .workspace_attention import WorkspaceAttention as _WA

            _wa = _WA(knowledge_engine)
            _score = _wa.get_attention_score(node_id)
            if _score is not None:
                runtime_confidence = _score
        except Exception:
            pass  # nosec B110

    # Source 2: MemoryRetriever historical proficiency (soft dependency on CONCEPT:AU-KG.compute.workspace-attention-scoring)
    historical_confidence = 0.5
    if knowledge_engine is not None:
        try:
            from ..knowledge_graph.retrieval.memory_retriever import MemoryRetriever

            sm = MemoryRetriever(knowledge_engine)
            current = sm.get_current()
            if current:
                historical_confidence = current.tool_proficiency.get(node_id, 0.5)
        except Exception:
            pass  # nosec B110

    # Blend: 70% runtime + 30% historical (degrades gracefully when
    # no MemoryRetriever is present — both default to 0.5 neutral)
    confidence_signal = 0.7 * runtime_confidence + 0.3 * historical_confidence

    try:
        from agent_utilities.core.model_factory import create_model

        original_tier = tier
        chosen = registry.pick_for_task_adaptive(
            complexity=tier,
            confidence_signal=confidence_signal,
            routing_percentile=routing_percentile,
            required_tags=required_tags,
        )
        effective_tier = chosen.tier

        if effective_tier != original_tier:
            logger.info(
                "[CONCEPT:AU-ORCH.adapter.hot-cache-invalidation] Confidence-gated routing: '%s' tier %s → %s "
                "(confidence=%.2f, percentile=%.0f)",
                node_id,
                original_tier,
                effective_tier,
                confidence_signal,
                routing_percentile,
            )

        api_key = setting(chosen.api_key_env) if chosen.api_key_env else None
        logger.debug(
            "Spawning specialist '%s' with model '%s' (tier=%s, tags=%s, confidence=%.2f)",
            node_id,
            chosen.id,
            effective_tier,
            required_tags,
            confidence_signal,
        )
        return create_model(
            provider=chosen.provider,
            model_id=chosen.model_id,
            base_url=chosen.base_url,
            api_key=api_key,
        )
    except Exception as e:
        logger.warning(f"Model selection for '{node_id}' fell back to default: {e}")
        return ctx_deps.agent_model


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
    capabilities = [c.lower().replace("-", "_") for c in agent.capabilities]
    # Also keep originals to be safe
    capabilities.extend([c.lower() for c in agent.capabilities])

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
) -> tuple[list[Any], list[Any]]:
    """Dynamically discover and load toolsets specialized for a domain expert.

    Starts with universal developer tools and augments them with domain-specific
    These tools are resolved by matching the node identifier against the
    Knowledge Graph specialist registry to discover assigned
    capability tags and MCP server associations.

    Returns:
        A tuple containing (list of developer tools, list of specialized skill toolsets).

    """
    from agent_utilities.core.config import get_discovery_registry

    from ..tools.developer_tools import developer_tools
    from ..tools.sdd_tools import sdd_tools

    registry = get_discovery_registry()
    agent = next((a for a in registry.agents if a.name == node_id), None)
    skill_tags = agent.capabilities if agent else []

    # CONCEPT:AU-ORCH.execution.orchestration-flow-mermaid (perf) — capability-gated generic-tool injection.
    # Previously the 13 developer_tools + 10 sdd_tools were dumped onto EVERY node
    # unconditionally. For an MCP-server node (e.g. "repository-manager-mcp") that drowns
    # the server's real tools in 23 irrelevant ones → wrong-tool selection (rg / SDD) and
    # ~3.5–9K wasted context tokens per call. Only inject the generic toolkits when the
    # node's name/capabilities indicate it does code/shell work (dev) or spec/planning (sdd).
    _DEV_TOOL_TAGS = {
        "code",
        "coding",
        "filesystem",
        "shell",
        "git",
        "devops",
        "python",
        "typescript",
        "programmer",
        "developer",
        "engineer",
        "refactor",
        "debug",
    }
    _SDD_TOOL_TAGS = {
        "sdd",
        "spec",
        "plan",
        "planner",
        "architect",
        "tdd",
        "requirements",
    }
    _tag_blob = " ".join([node_id, *skill_tags]).lower()
    tools: list[Any] = []
    if any(t in _tag_blob for t in _DEV_TOOL_TAGS):
        tools += list(developer_tools)
    if any(t in _tag_blob for t in _SDD_TOOL_TAGS):
        tools += list(sdd_tools)

    toolsets: list[Any] = []
    if not skill_tags:
        return tools, toolsets

    logger.debug(
        f"Loading {len(skill_tags)} specialized skill tags for '{node_id}': {skill_tags}"
    )

    try:
        from pydantic_ai_skills import SkillsToolset

        from agent_utilities.core.workspace import get_skills_path

        skill_dirs: list[str] = []
        if skills_path := get_skills_path():
            skill_dirs.extend(skills_path)

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
            from agent_utilities.tools.tool_filtering import skill_matches_tags

            filtered_dirs = [d for d in skill_dirs if skill_matches_tags(d, skill_tags)]
            if filtered_dirs:
                skills_toolset = SkillsToolset(
                    directories=cast("list[Any]", filtered_dirs)
                )
                toolsets.append(skills_toolset)
                logger.info(
                    f"Loaded {len(filtered_dirs)} skill directories for '{node_id}'"
                )
    except ImportError:
        logger.debug("pydantic-ai-skills not installed; skipping skill injection")

    return tools, toolsets


def _resolve_invoker_cred(state: Any, deps: GraphDeps) -> str | None:
    """CONCEPT:AU-ORCH.session.invoker-agent-handoff (Phase 4) — resolve the invoker's credential REFERENCE to a raw token.

    The reference (``GraphState.invoker_cred_ref``) names a secret the invoker stored in the
    secrets backend; the raw value is resolved here (deps-build time) and lives only on the
    transient AgentDeps — never persisted to GraphState/graph/logs. Returns None on miss.
    """
    ref = getattr(state, "invoker_cred_ref", None) if state is not None else None
    if not ref:
        return None
    try:
        client = getattr(deps, "secrets_client", None)
        if client is None:
            from ..security.secrets_client import create_secrets_client

            client = create_secrets_client()
        return client.get(ref)
    except Exception:  # noqa: BLE001 — a missing/failed secret must not block the spawn
        return None


def agent_deps_from_graph(
    deps: GraphDeps, toolsets: list[Any] | None = None, state: Any = None
) -> Any:
    """Build an ``AgentDeps`` from the graph-level ``GraphDeps``.

    Dynamic agents spawned inside graph nodes inject ``developer_tools``/``sdd_tools``
    which are ``RunContext[AgentDeps]``-typed and read ``ctx.deps.workspace_path`` —
    a field ``GraphDeps`` does not have. Running such an agent without ``deps`` (or with
    raw ``GraphDeps``) raises ``'NoneType'/'GraphDeps' object has no attribute
    'workspace_path'``. This adapts the graph context into a valid ``AgentDeps`` so both
    the injected tools and the MCP toolsets work. (Wire-First: ORCH-1.21 execution path.)

    CONCEPT:AU-ORCH.session.invoker-agent-handoff (Phase 4) — when ``state`` carries an invoker credential reference, the
    raw token is resolved here onto the transient AgentDeps.auth_token (never into GraphState).
    """
    from pathlib import Path

    from ..core.workspace import get_agent_workspace
    from ..models.agent import AgentDeps

    ws = (
        Path(deps.project_root)
        if getattr(deps, "project_root", "")
        else get_agent_workspace()
    )
    return AgentDeps(
        workspace_path=ws,
        knowledge_engine=getattr(deps, "knowledge_engine", None),
        mcp_toolsets=toolsets if toolsets is not None else list(deps.mcp_toolsets),
        ssl_verify=deps.ssl_verify,
        provider=deps.provider,
        base_url=deps.base_url,
        api_key=deps.api_key,
        request_id=deps.request_id,
        approval_timeout=deps.approval_timeout,
        graph_event_queue=deps.event_queue,
        auth_token=_resolve_invoker_cred(state, deps),
        message_channel_id=getattr(
            state, "invoker_channel_id", None
        ),  # CONCEPT:AU-ORCH.session.session-anchored-collections-native
    )


def invoker_context_section(state: Any, *, window_tokens: int = 32768) -> str:
    """CONCEPT:AU-ORCH.session.invoker-agent-handoff — render the invoker's curated context as a budgeted prompt section.

    Returns an empty string when no invoker context was provided, otherwise an
    ``### INVOKER CONTEXT`` block trimmed to a fraction (~15%) of the target model's context
    window. Defaults to the smaller (32K) window so the section fits BOTH the 9B (64K) and the
    lite (32K) models without overflow; a future pass can resolve the exact per-model window.
    """
    text = (getattr(state, "invoker_context", "") or "").strip()
    if not text:
        return ""
    budget_chars = max(2000, int(window_tokens * 0.15) * 4)  # ~4 chars/token
    if len(text) > budget_chars:
        text = (
            text[:budget_chars] + "\n…[invoker context truncated to fit model window]"
        )
    return (
        "\n\n### INVOKER CONTEXT (curated by the invoking agent — treat as authoritative "
        "background for this task)\n" + text + "\n"
    )


def apply_tool_scope(
    state: Any, tools: list[Any], toolsets: list[Any]
) -> tuple[list[Any], list[Any]]:
    """CONCEPT:AU-ORCH.session.invoker-agent-handoff — enforce the invoker's least-privilege tool allow-list.

    When ``GraphState.invoker_allowed_tools`` is set, function tools are filtered by name and
    MCP/skill toolsets are wrapped with a pydantic-ai ``.filtered()`` predicate so the spawned
    agent can ONLY call the allowed tools. Empty/None allow-list = no restriction.
    """
    allowed = getattr(state, "invoker_allowed_tools", None)
    if not allowed:
        return tools, toolsets
    allowed_set = {str(a) for a in allowed}
    scoped_tools = [t for t in tools if getattr(t, "__name__", None) in allowed_set]
    scoped_toolsets = []
    for ts in toolsets:
        flt = getattr(ts, "filtered", None)
        # A toolset that can't be filtered must NOT pass through unrestricted — that
        # silently violates the invoker's least-privilege allow-list. Mirror the
        # single-server path (agent_runner) and fail loudly.
        if not callable(flt):
            raise RuntimeError(
                f"toolset {type(ts).__name__!r} does not support tool filtering; "
                f"cannot enforce allowed_tools={sorted(allowed_set)[:8]}"
            )
        scoped_toolsets.append(flt(lambda ctx, td: td.name in allowed_set))
    # If the allow-list eliminated every function tool AND left no toolset to invoke,
    # the spawned agent would have nothing to call and would fabricate a tool call —
    # surface that clearly instead of producing a tool-less hallucinator. (A toolset
    # that is present but whose tools don't intersect the allow-list — e.g. a tool
    # name passed with the wrong server prefix — can only be detected once its tools
    # are enumerated at run time, not here.)
    if not scoped_tools and not scoped_toolsets:
        raise RuntimeError(
            f"allowed_tools={sorted(allowed_set)[:8]} eliminated every bound tool; "
            "the scoped agent would have nothing to invoke"
        )
    logger.info(
        "[ORCH-1.39] Tool scope enforced: %d→%d function tools; allow-list=%s",
        len(tools),
        len(scoped_tools),
        sorted(allowed_set)[:8],
    )
    return scoped_tools, scoped_toolsets


def spawn_usage_limits(state: Any, *, request_limit: int = 8) -> Any:
    """CONCEPT:AU-ORCH.execution.orchestration-flow-mermaid/1.38 — UsageLimits for a spawned task agent.

    Always bounds requests (default pydantic-ai cap is 50). When the invoking agent granted a
    token budget (``GraphState.invoker_budget_tokens``), also enforce it as
    ``total_tokens_limit`` so the spawned agent cannot exceed the budget the invoker allotted.
    """
    from pydantic_ai.usage import UsageLimits

    req = setting("AGENT_REQUEST_LIMIT", request_limit)
    budget = getattr(state, "invoker_budget_tokens", None)
    if budget and int(budget) > 0:
        return UsageLimits(request_limit=req, total_tokens_limit=int(budget))
    return UsageLimits(request_limit=req)


def get_step_descriptions() -> str:
    """Generate a formatted catalog of expert capabilities for the LLM planner.

    Combines static roles, discovered A2A peers, and registered MCP adaptive_agent_router
    into a cohesive markdown list used in system prompts.  Uses the unified
    :func:`discover_all_specialists` roster so that both MCP and A2A sources
    are enumerated through the same code path.

    Returns:
        A multi-line markdown string describing all available graph nodes.

    """
    from agent_utilities.agent.discovery import discover_all_specialists

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
        "recursive_orchestrator": (
            "CONCEPT:AU-ORCH.planning.recursion-nesting-depth — Recursive graph re-orchestration. Use when the "
            "current plan has failed and needs a fundamentally different approach. "
            "Spawns a nested graph execution with the parent's context and errors "
            "to devise a corrected strategy. Only use for complex multi-step failures."
        ),
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


async def _execute_dynamic_mcp_agent(ctx: StepContext, agent_info: MCPAgent) -> str:
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
        server_name=server_name or "unknown",
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
            server_name=server_name or "unknown",
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
        target_server = (agent_info.mcp_server or "").lower()
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
    agent_sys_prompt += invoker_context_section(ctx.state)  # CONCEPT:AU-ORCH.session.invoker-agent-handoff

    # Include validation feedback if this is a re-dispatch
    if ctx.state.validation_feedback:
        agent_sys_prompt += (
            f"\n\n### PREVIOUS FEEDBACK\n"
            f"Your previous output was reviewed and needs improvement:\n"
            f"{ctx.state.validation_feedback}\n"
            f"Address this feedback in your response by being more thorough or providing the missing data."
        )

    # CONCEPT:AU-ORCH.execution.inject-signal-board-observations — Inject signal board observations from prior adaptive_agent_router
    if ctx.state.signal_board:
        signal_lines = []
        for sig_type, messages in ctx.state.signal_board.items():
            for msg in messages[:3]:  # Limit injection to avoid prompt bloat
                signal_lines.append(f"- [{sig_type}] {msg}")
        if signal_lines:
            agent_sys_prompt += (
                "\n\n### OBSERVATIONS FROM PRIOR SPECIALISTS\n"
                "Other adaptive_agent_router have flagged the following for your awareness:\n"
                + "\n".join(signal_lines[:10])
                + "\nConsider these signals when performing your task."
            )

    # CONCEPT:AU-ORCH.execution.visibility-allow-list — Inject access-list-filtered prior results
    step_input_for_access = ctx.inputs
    if (
        isinstance(step_input_for_access, ExecutionStep)
        and step_input_for_access.access_list
    ):
        access_context = _resolve_access_context(
            step_input_for_access, ctx.state.results_registry
        )
        if access_context:
            agent_sys_prompt += (
                f"\n\n### PRIOR STEP RESULTS (Access-List Filtered)\n"
                f"{access_context}\n"
                f"Use these results as context for your task."
            )
            logger.info(
                "[CONCEPT:AU-ORCH.planning.recursion-nesting-depth] Injected %d access-list results for '%s'",
                len(step_input_for_access.access_list),
                agent_name,
            )

    # Emit startup event with detailed metadata for UI transparency
    emit_graph_event(
        ctx.deps.event_queue,
        "expert_metadata",
        domain=agent_info.name or "unknown",
        expert=agent_info.name,
        target_server=agent_info.mcp_server or "unknown",
        domain_tag=agent_info.name,
        expected_tools=total_tools,
        id=getattr(ctx, "node_id", "unknown"),
    )

    # CONCEPT:AU-ORCH.adapter.hot-cache-invalidation — Capability Auto-Activation
    # Check if this specialist has registered capabilities (e.g., RLM, critic)
    # and activate them before execution.
    activated_capabilities: list[str] = []
    if ctx.deps.knowledge_engine:
        try:
            cap_rows = []
            if ctx.deps.knowledge_engine.backend:
                cap_rows = ctx.deps.knowledge_engine.backend.execute(
                    "MATCH (a {name: $name})-[:has_capability]->(c:AgentCapability) "
                    "WHERE c.auto_activate = true RETURN c",
                    {"name": agent_name},
                )
            for row in cap_rows:
                cap_data = row.get("c", row)
                cap_type = cap_data.get("capability_type", "unknown")
                handler_module = cap_data.get("handler_module")
                handler_fn = cap_data.get("handler_function")
                if handler_module and handler_fn:
                    # Check trigger conditions
                    triggers = cap_data.get("trigger_conditions", {})
                    should_activate = True
                    if "input_chars_gt" in triggers:
                        should_activate = (
                            len(ctx.state.query) > triggers["input_chars_gt"]
                        )

                    if should_activate:
                        activated_capabilities.append(cap_type)
                        logger.info(
                            f"[CONCEPT:AU-ORCH.adapter.hot-cache-invalidation] Auto-activated capability '{cap_type}' for specialist '{agent_name}' "
                            f"(handler={handler_module}.{handler_fn})"
                        )
                        emit_graph_event(
                            ctx.deps.event_queue,
                            "capability_activated",
                            specialist=agent_name,
                            capability=cap_type,
                        )
        except Exception as e:
            logger.debug(f"Capability auto-activation lookup failed: {e}")

    # CONCEPT:AU-KG.compute.workspace-attention-scoring — WorkspaceAttention scoring for specialist priority
    attention_score: float | None = None
    if ctx.deps.knowledge_engine:
        try:
            from .workspace_attention import WorkspaceAttention

            wa = WorkspaceAttention(ctx.deps.knowledge_engine)
            attention_score = wa.get_attention_score(agent_name)
            if attention_score is not None:
                logger.info(
                    f"[GWT] Specialist '{agent_name}' attention score: {attention_score:.2f}"
                )
        except Exception as e:
            logger.debug(f"WorkspaceAttention scoring failed for '{agent_name}': {e}")

    agent = Agent(
        model=ctx.deps.agent_model,
        system_prompt=agent_sys_prompt,
        deps_type=GraphDeps,
    )

    from contextlib import AsyncExitStack

    async with AsyncExitStack() as stack:
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
                target = (agent_info.mcp_server or "").lower().replace("-", "_")
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

        target_server_name = (agent_info.mcp_server or "").lower().replace("-", "_")
        if target_server_name and not matched_toolsets:
            try:
                from pydantic_ai.mcp import load_mcp_toolsets

                from agent_utilities.core.workspace import resolve_mcp_config_path

                mcp_path = resolve_mcp_config_path(None)
                if mcp_path and mcp_path.exists():
                    all_servers = load_mcp_toolsets(mcp_path)
                    for srv in all_servers:
                        srv_id = getattr(srv, "id", getattr(srv, "name", str(srv)))
                        current = srv_id.lower().replace("-", "_")
                        if (
                            current == target_server_name
                            or current.startswith(f"{target_server_name}_")
                            or target_server_name.startswith(f"{current}_")
                        ):
                            logger.info(
                                f"[LAYER:GRAPH:EXPERT] Lazy loading MCP server '{srv_id}' for expert '{agent_info.name}'"
                            )
                            await stack.enter_async_context(srv)
                            matched_toolsets.append(srv)
                            _tools = getattr(srv, "tools", {})
                            for t_name in (
                                _tools.keys() if hasattr(_tools, "keys") else []
                            ):
                                if t_name not in actually_bound_tools:
                                    actually_bound_tools.append(t_name)
                            bound_tool_count += len(_tools)
                            break
            except Exception as e:
                logger.warning(
                    f"Failed to lazy load MCP server '{target_server_name}': {e}"
                )

        # Wrap MCP toolsets with the tool guard so sensitive tools are flagged
        # as 'unapproved' and trigger the DeferredToolRequests flow.
        from agent_utilities.security.tool_guard import (
            build_sensitive_tool_names,
            flag_mcp_tool_definitions,
        )

        guarded_toolsets = flag_mcp_tool_definitions(
            matched_toolsets, build_sensitive_tool_names()
        )

        # Include DeferredToolRequests in output type so the agent can defer
        # sensitive tool calls instead of failing.

        from pydantic_ai import DeferredToolRequests

        # CONCEPT:AU-ORCH.session.invoker-agent-handoff — enforce the invoker's least-privilege tool allow-list (if any).
        _scoped_tools, guarded_toolsets = apply_tool_scope(
            ctx.state, [], guarded_toolsets
        )

        agent = Agent(
            model=ctx.deps.agent_model,
            system_prompt=agent_sys_prompt,
            deps_type=GraphDeps,
            toolsets=guarded_toolsets,
            output_type=[str, DeferredToolRequests],
            end_strategy="early",
        )

        # Tool-count telemetry: surfaces blind or overloaded adaptive_agent_router
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
        # CONCEPT:AU-ORCH.planning.recursion-nesting-depth — Prefer refined subtask over raw query
        if isinstance(step_input, ExecutionStep) and step_input.refined_subtask:
            sub_query = step_input.refined_subtask
            logger.info(
                "[CONCEPT:AU-ORCH.planning.recursion-nesting-depth] Using refined subtask for '%s': '%s'",
                agent_info.name,
                sub_query[:100],
            )
        elif isinstance(step_input, ExecutionStep) and step_input.description:
            if isinstance(step_input.description, dict):
                sub_query = step_input.description.get("question", sub_query)
            elif isinstance(step_input.description, str):
                sub_query = step_input.description

        # Execute with Per-Node Timeout and Retries
        node_timeout = 120.0
        if isinstance(step_input, ExecutionStep):
            node_timeout = step_input.timeout

        # Retrieve cached message history for re-dispatch context
        cache_key = agent_info.name.lower().replace(" ", "_")
        prev_messages = ctx.deps.message_history_cache.get(cache_key)

        max_attempts = 3
        last_error = None
        attempt_no = 0

        async def _dispatch_specialist_once() -> str:
            nonlocal last_error, attempt_no
            attempt_no += 1
            emit_graph_event(
                ctx.deps.event_queue,
                "expert_thinking",
                expert=agent_info.name,
                attempt=attempt_no,
            )
            try:
                logger.info(
                    f"[LAYER:GRAPH:EXPERT] '{agent_info.name}' LLM Call Starting (attempt {attempt_no}). Prompt length: {len(agent_sys_prompt)}"
                )
                # Wrap user query in XML tags to protect against prompt injection
                # and provide clear boundaries for the model.
                raw_input = (
                    ctx.state.query_parts
                    if ctx.state.query_parts and sub_query == ctx.state.query
                    else sub_query
                )
                run_input = (
                    f"<user_query>\n{raw_input}\n</user_query>"
                    if isinstance(raw_input, str)
                    else raw_input
                )
                # CONCEPT:AU-ORCH.execution.retry-predicate-raised-treating — Declarative Resilience Policy.
                # The single per-node-timeout LLM call is wrapped in the
                # declarative retry/backoff/timeout policy so transient model
                # or tool errors (TimeoutError/ConnectionError) are retried with
                # exponential backoff *before* surfacing to the outer attempt
                # loop. This composes with — and does not replace — the existing
                # per-server circuit breaker (``ctx.deps.server_health``) and the
                # ``node_timeout`` second wait, which is now enforced per attempt
                # by the policy's ``timeout_s``.
                _policy = _specialist_resilience_policy(node_timeout)

                async def _run_agent_once(
                    _agent: Any = agent,
                    _input: Any = run_input,
                    _hist: Any = prev_messages,
                ) -> Any:
                    return await _agent.run(
                        _input, deps=ctx.deps, message_history=_hist
                    )

                res = await run_with_resilience(_run_agent_once, _policy)
                logger.info(
                    f"[LAYER:GRAPH:EXPERT] '{agent_info.name}' LLM Call Completed."
                )
                ctx.state._update_usage(getattr(res, "usage", None))

                # Cache message history for potential re-dispatch
                try:
                    ctx.deps.message_history_cache[cache_key] = res.all_messages()
                except Exception as e:
                    logger.debug(f"Failed to update cache for '{cache_key}': {e}")
                    pass

                # Record success on circuit breaker
                srv_name = server_name or "unknown"
                if srv_name not in ctx.deps.server_health:
                    ctx.deps.server_health[srv_name] = MCPServerHealth(
                        server_name=srv_name,
                    )
                ctx.deps.server_health[srv_name].record_success()

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
                                        domain=agent_info.name or "unknown",
                                        tool_name=part.tool_name,
                                        args=part.args,
                                    )
                                elif hasattr(part, "content") and part.content:
                                    emit_graph_event(
                                        ctx.deps.event_queue,
                                        "expert_text",
                                        domain=agent_info.name or "unknown",
                                        content=part.content,
                                    )
                        elif isinstance(msg, ModelRequest):
                            for req_part in msg.parts:
                                if isinstance(req_part, ToolReturnPart):
                                    emit_graph_event(
                                        ctx.deps.event_queue,
                                        event_type="tool_result",
                                        agent=agent_info.name,
                                        tool=req_part.tool_name,
                                        result=str(req_part.content)[:500],
                                    )
                # Unified result storage: write to results_registry (primary, read by dispatcher/verifier)
                # and mirror to results (keyed by domain tag, for backwards compatibility).
                result_str = str(res.output)

                # RLM Large Result Summarization
                from ..rlm.config import RLMConfig

                rlm_config = RLMConfig()
                if len(result_str) > rlm_config.max_context_threshold:
                    logger.warning(
                        f"Expert '{agent_info.name}' result ({len(result_str)} chars) exceeds threshold. "
                        "Routing to RLM for summarization."
                    )
                    from ..rlm.specialist import recursive_reasoner_tool

                    try:
                        summary = await recursive_reasoner_tool(
                            ctx,
                            prompt=f"The specialist '{agent_info.name}' returned a massive output. Summarize the key findings relevant to the user's query: {ctx.state.query}",
                            context_data=result_str,
                        )
                        result_str = (
                            f"[RLM Synthesized Summary of Massive Data]\n{summary}"
                        )
                    except Exception as rlm_err:
                        logger.error(f"RLM summarization failed: {rlm_err}")
                        result_str = (
                            result_str[: rlm_config.max_context_threshold]
                            + "... [TRUNCATED DUE TO SIZE & RLM FAILURE]"
                        )

                # Data Enhancement Synthesizer: Synthesize response from real output tool data captured
                if (
                    "no data" in result_str.lower()
                    or "returned no" in result_str.lower()
                ):
                    from pydantic_ai.messages import ModelRequest, ToolReturnPart

                    tool_returns: list[str] = []
                    for msg in res.all_messages():
                        if isinstance(msg, ModelRequest):
                            for ret_part in msg.parts:
                                if (
                                    isinstance(ret_part, ToolReturnPart)
                                    and ret_part.content
                                ):
                                    content_str = str(ret_part.content)
                                    if content_str and content_str not in (
                                        "[]",
                                        "None",
                                        "null",
                                        "",
                                    ):
                                        tool_returns.append(
                                            f"**{ret_part.tool_name}**: {content_str}"
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

                result_key = agent_info.name or cache_key
                ctx.state.results[result_key] = result_str
                ctx.state.routed_domain = result_key
                logger.info(
                    f"Expert: '{agent_info.name}' succeeded (attempt {attempt_no}). "
                    f"Result: {len(result_str)} chars. Registry key: '{node_uid}'"
                )
                # Emit completion event
                emit_graph_event(
                    ctx.deps.event_queue,
                    "subagent_completed",
                    domain=agent_info.name or "unknown",
                    status="success",
                )
                # HSM: Exit action (success)
                await on_exit_specialist(
                    ctx_deps=ctx.deps,
                    ctx_state=ctx.state,
                    agent_name=agent_name,
                    success=True,
                    server_name=server_name or "unknown",
                )
                return "execution_joiner"

            except TimeoutError:
                last_error = f"Timeout after {node_timeout}s"
                logger.warning(
                    f"Expert '{agent_name}' timed out (attempt {attempt_no}/{max_attempts})"
                )
                emit_graph_event(
                    ctx.deps.event_queue,
                    "expert_complete",
                    expert=agent_info.name,
                    status="timeout",
                )
                raise
            except Exception as e:
                last_error = str(e)
                logger.warning(
                    f"Expert '{agent_name}' failed (attempt {attempt_no}/{max_attempts}): {e}"
                )
                emit_graph_event(
                    ctx.deps.event_queue,
                    "subagent_tool_call",
                    domain=agent_info.name or "unknown",
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
                raise

        # Historical outer dispatch backoff min(2**n, 10)s, declaratively
        # (CONCEPT:AU-ORCH.execution.retry-predicate-raised-treating). This OUTER policy retries the whole specialist
        # dispatch (events + run + result handling); it composes with the
        # INNER per-LLM-call policy inside the attempt body.
        dispatch_policy = ResiliencePolicy(
            max_attempts=max_attempts,
            backoff_base_s=1.0,
            backoff_factor=2.0,
            max_backoff_s=10.0,
            jitter=False,
            retry_on=lambda exc: isinstance(exc, Exception),
            name=f"specialist-dispatch:{agent_info.name or 'unknown'}",
        )
        try:
            return await run_with_resilience(_dispatch_specialist_once, dispatch_policy)
        except Exception:  # noqa: BLE001 - exhausted; fall through to HSM exit + fallback
            pass

        # All retries exhausted
        # HSM: Exit action (failure)
        await on_exit_specialist(
            ctx_deps=ctx.deps,
            ctx_state=ctx.state,
            agent_name=agent_name,
            success=False,
            server_name=server_name or "unknown",
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
    ctx: StepContext,
    failed_agent: MCPAgent,
) -> str | None:
    """Implement a resilience strategy by falling back to sibling adaptive_agent_router.

    If a targeted expert fails or is unavailable, this helper searches for
    other adaptive_agent_router from the same MCP server. It scores candidate siblings
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
        and a.name != failed_agent.name
        and a.name != failed_agent.name
    ]

    if not siblings:
        return None

    # Score siblings by keyword overlap with the query
    query_words = set(ctx.state.query.lower().split())
    best_sibling = None
    best_score = 0

    for sibling in siblings:
        tag_words = set(
            sibling.name.lower().replace("-", " ").replace("_", " ").split()
        )
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
    ctx: StepContext,
    node_id: str,
    meta: dict,
) -> str | End[Any]:
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
        from agent_utilities.protocols.a2a import A2AClient

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
        if isinstance(step_input, ExecutionStep) and step_input.description:
            if isinstance(step_input.description, dict):
                sub_query = step_input.description.get("question", sub_query)
            elif isinstance(step_input.description, str):
                sub_query = step_input.description

        res_content = await client.execute_task(peer_url, sub_query)
        result_str = str(res_content)
        # Unified result storage
        node_uid = f"{node_id}_{ctx.state.step_cursor}"
        ctx.state.results_registry[node_uid] = result_str
        ctx.state.results[node_id] = result_str
    else:
        # CONCEPT:AU-ORCH.adapter.hot-cache-invalidation: Unified specialist execution
        # Try specialized prompt-based execution first (loads persona, injects tools + skills)
        registry = load_node_agents_registry()
        mcp_agent = next(
            (a for a in registry.agents if agent_matches_node_id(a, node_id)),
            None,
        )

        if mcp_agent and mcp_agent.mcp_server:
            # MCP-bound specialist — execute with bound tools
            await _execute_dynamic_mcp_agent(ctx, mcp_agent)
        elif mcp_agent and mcp_agent.json_blueprint:
            # Prompt-based specialist — execute with persona + injected tools
            return await _execute_specialized_step(ctx, node_id)
        else:
            # Fallback: try specialized step (prompt lookup by name), then generic
            try:
                return await _execute_specialized_step(ctx, node_id)
            except Exception:
                logger.warning(
                    f"Expert Execution: Node '{node_id}' fallback. "
                    f"No specialist metadata found in the Knowledge Graph."
                )
                await _execute_domain_logic(ctx, node_id)

    return "execution_joiner"


async def agent_package_step(
    ctx: StepContext,
    node_id: str,
) -> str | End[Any]:
    """Graph node step wrapper for agent package execution.

    This acts as the standardized entry point for all specialist agent nodes
    discovered during the bootstrap phase.

    Args:
        ctx: The pydantic-graph step context.
        node_id: The identifier of the package to execute.

    Returns:
        The next node identifier (usually 'execution_joiner' or 'Error').

    """
    from agent_utilities.agent.discovery import discover_agents

    discovered = discover_agents()
    if node_id not in discovered:
        logger.error(f"Agent package node '{node_id}' not found in discovery.")
        return "Error"

    meta = discovered[node_id]
    return await _execute_agent_package_logic(ctx, node_id, meta)


async def _execute_specialized_step(
    ctx: StepContext, prompt_name: str
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
    from agent_utilities.security.tool_guard import (
        build_sensitive_tool_names,
        flag_mcp_tool_definitions,
    )

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

    # CONCEPT:AU-ORCH.routing.conductor-per-step-model — honor a Conductor-assigned per-step model_id (ctx.inputs is
    # the current ExecutionStep/Task); falls back to override/tier routing when unset.
    specialist_model = pick_specialist_model(
        ctx.deps, prompt_name, step_model_id=getattr(ctx.inputs, "model_id", None)
    )

    # CONCEPT:AU-ORCH.session.invoker-agent-handoff — enforce the invoker's least-privilege tool allow-list (if any).
    custom_tools, _scoped_toolsets = apply_tool_scope(
        ctx.state, custom_tools, collected_mcp_toolsets + skill_toolsets
    )

    agent = Agent(
        model=specialist_model,
        system_prompt=(
            f"{memory_instruction}\n\n"
            f"{prompt}\n\n"
            f"### TOOL USAGE GUIDANCE\n{tool_guidance}\n\n"
            f"### CONTEXT\n{ctx.state.exploration_notes}"
            f"{feedback_section}"
            f"{invoker_context_section(ctx.state)}"  # CONCEPT:AU-ORCH.session.invoker-agent-handoff
        ),
        tools=custom_tools,
        toolsets=_scoped_toolsets,
        output_type=[str, DeferredToolRequests],
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

    # Injected dev/sdd tools are RunContext[AgentDeps]-typed (read ctx.deps.workspace_path);
    # adapt the graph context so specialist tool calls don't NoneType on missing deps.
    _agent_deps = agent_deps_from_graph(
        ctx.deps, collected_mcp_toolsets + skill_toolsets, state=ctx.state
    )

    # CONCEPT:AU-ORCH.execution.orchestration-flow-mermaid (perf) — bound per-agent requests. Without this, pydantic-ai's
    # default request_limit=50 lets a confused agent burn 50 model calls before failing
    # (we observed this twice). A specialist answering one question needs only a few.
    try:
        run_input = ctx.state.query_parts if ctx.state.query_parts else ctx.state.query
        async with agent.run_stream(
            run_input,
            message_history=prev_messages,
            deps=_agent_deps,
            usage_limits=spawn_usage_limits(ctx.state),  # CONCEPT:AU-ORCH.session.invoker-agent-handoff budget
        ) as stream:
            async for chunk in stream.stream_text(delta=True):
                emit_graph_event(
                    ctx.deps.event_queue,
                    "agent_node_delta",
                    content=chunk,
                    node=prompt_name,
                )
            res = await stream.get_output()
        usage = stream.usage  # v2: property (was a method in v1)
        if callable(usage):
            usage = usage()
        if asyncio.iscoroutine(usage):
            usage = await usage
        ctx.state._update_usage(usage)
        result_str = str(res)

        # RLM Large Result Summarization
        from ..rlm.config import RLMConfig

        rlm_config = RLMConfig()
        if len(result_str) > rlm_config.max_context_threshold:
            logger.warning(
                f"Specialist '{prompt_name}' result ({len(result_str)} chars) exceeds threshold. "
                "Routing to RLM for summarization."
            )
            from ..rlm.specialist import recursive_reasoner_tool

            try:
                summary = await recursive_reasoner_tool(
                    ctx,
                    prompt=f"The specialist '{prompt_name}' returned a massive output. Summarize the key findings relevant to the user's query: {ctx.state.query}",
                    context_data=result_str,
                )
                result_str = f"[RLM Synthesized Summary of Massive Data]\n{summary}"
            except Exception as rlm_err:
                logger.error(f"RLM summarization failed: {rlm_err}")
                result_str = (
                    result_str[: rlm_config.max_context_threshold]
                    + "... [TRUNCATED DUE TO SIZE & RLM FAILURE]"
                )

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
            history = stream.all_messages()
            if asyncio.iscoroutine(history):
                history = await history
            ctx.deps.message_history_cache[prompt_name] = history
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


async def _execute_domain_logic(ctx: StepContext, domain: str):
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
        original_env[env_var] = setting(env_var)
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
                    from agent_utilities.agent.factory import create_agent

                    target, _ = create_agent(
                        name=domain,
                        system_prompt=target.get(
                            "description", f"Specialized assistant for {domain}"
                        ),
                        enable_skills=True,
                        skill_types=["universal", "graphs"],
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
                    res = await execute_graph(
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

            from agent_utilities.agent.factory import create_agent

            sub_agent, _ = create_agent(
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
                from agent_utilities.observability.approval_manager import (
                    run_with_approvals,
                )

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


# implements core.execution.ExecutionEngine
class GraphExecutorEngine:
    """Additive engine wrapper conforming to the unified ExecutionEngine contract.

    Plan 03 Step 5 — ``graph.executor`` is historically a *module* of
    step-execution functions rather than an engine object. This thin wrapper
    binds a pydantic ``graph`` + ``config`` and exposes the shared
    ``run(manifest) -> ExecutionResult`` contract by delegating to the
    existing module entrypoint :func:`execute_graph`. It is **purely
    additive**: no existing public function or class in this module is
    renamed, removed, or behaviourally changed.
    """

    def __init__(self, graph: Any, config: dict[str, Any] | None = None):
        self.graph = graph
        self.config = config or {}

    async def run(self, manifest: Any) -> Any:
        """Unified ExecutionEngine contract entrypoint.

        Normalises ``manifest`` to a query string and runs the graph via
        :func:`execute_graph`, wrapping the result into a canonical
        ``ExecutionResult``.
        """
        from agent_utilities.core.execution.models import ExecutionResult

        if isinstance(manifest, str):
            query = manifest
            manifest_id = ""
        else:
            query = getattr(manifest, "query", "") or ""
            manifest_id = getattr(manifest, "manifest_id", "") or ""

        result = await execute_graph(self.graph, self.config, query)

        synthesis_output = ""
        success = True
        if isinstance(result, dict):
            synthesis_output = str(
                result.get("output")
                or result.get("response")
                or result.get("result")
                or ""
            )
            if "success" in result:
                success = bool(result["success"])
            elif "error" in result and result["error"]:
                success = False
        else:
            synthesis_output = str(result)

        return ExecutionResult(
            manifest_id=manifest_id,
            synthesis_output=synthesis_output,
            success=success,
        )
