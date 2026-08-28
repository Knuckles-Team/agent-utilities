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
from dataclasses import dataclass
from typing import Any, NamedTuple, cast

from pydantic_ai import DeferredToolRequests
from pydantic_graph import End
from pydantic_graph.step import StepContext

from agent_utilities.core.config import (
    DEFAULT_GRAPH_TIMEOUT,
    emit_graph_event,
    get_discovery_registry,
    load_specialized_prompts,
    setting,
)
from agent_utilities.core.contextual_model import create_context_agent
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


async def _maybe_rlm_summarize(
    ctx: StepContext, role_label: str, name: str, result_str: str
) -> str:
    """RLM Large Result Summarization.

    If ``result_str`` exceeds the RLM config's ``max_context_threshold``,
    summarize it via :func:`recursive_reasoner_tool`, falling back to a
    truncated string if that also fails. Shared by
    :func:`_execute_specialized_step` and :func:`_execute_dynamic_mcp_agent`'s
    dispatch loop -- ``role_label`` carries their differing log prefixes
    ("Specialist" / "Expert").
    """
    from ..rlm.config import RLMConfig

    rlm_config = RLMConfig()
    if len(result_str) <= rlm_config.max_context_threshold:
        return result_str

    logger.warning(
        f"{role_label} '{name}' result ({len(result_str)} chars) exceeds threshold. "
        "Routing to RLM for summarization."
    )
    from ..rlm.specialist import recursive_reasoner_tool

    try:
        summary = await recursive_reasoner_tool(
            ctx,
            prompt=f"The specialist '{name}' returned a massive output. Summarize the key findings relevant to the user's query: {ctx.state.query}",
            context_data=result_str,
        )
        return f"[RLM Synthesized Summary of Massive Data]\n{summary}"
    except Exception as rlm_err:
        logger.error(f"RLM summarization failed: {rlm_err}")
        return (
            result_str[: rlm_config.max_context_threshold]
            + "... [TRUNCATED DUE TO SIZE & RLM FAILURE]"
        )


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


def _resolve_explicit_model(
    registry: Any, model_id: str, node_id: str, label: str
) -> Any | None:
    """CONCEPT:AU-ORCH.routing.conductor-per-step-model — resolve one explicit model_id override.

    Shared by the Conductor-assigned ``step_model_id`` path and the per-turn
    ``requested_model_id`` (``x-agent-model-id`` header) path in
    :func:`pick_specialist_model`. Both look the id up in ``registry``, build
    a concrete pydantic-ai model via :func:`create_model` on a hit, and log +
    return ``None`` on any lookup/build failure so the caller falls through to
    tier-based routing.
    """
    chosen = registry.get_by_id(model_id)
    if chosen is None:
        logger.debug(
            "%s model id '%s' not in registry; using override/tier routing",
            label,
            model_id,
        )
        return None
    try:
        from agent_utilities.core.model_factory import create_model

        api_key = setting(chosen.api_key_env) if chosen.api_key_env else None
        logger.info(
            "Spawning specialist '%s' with %s model '%s'",
            node_id,
            label,
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
            "%s model '%s' failed to build; falling back: %s",
            label,
            model_id,
            e,
        )
        return None


def _resolve_tier_and_tags(node_id: str) -> tuple[str, list[str]]:
    """Resolve the heuristic tier + required_tags for ``node_id`` from the discovery registry.

    Falls back to :func:`_default_tier_for` and an empty tag list on any
    registry-read failure — identical behaviour to a specialist with no
    registry entry.
    """
    tier = _default_tier_for(node_id)
    required_tags: list[str] = []
    try:
        reg = get_discovery_registry()
        agent_info = next((a for a in reg.agents if a.name == node_id), None)
        if agent_info is not None:
            tier = getattr(agent_info, "default_tier", tier) or tier
            required_tags = list(getattr(agent_info, "required_tags", []) or [])
    except Exception as e:  # noqa: BLE001 — tier/required_tags already hold safe pre-lookup defaults; a registry read failure just leaves those defaults in place, identical to a specialist with no registry entry
        logger.debug(f"Registry tier lookup failed for '{node_id}': {e}")
    return tier, required_tags


def _apply_homeostatic_downgrade(
    resource_optimizer: Any, tier: str, node_id: str, required_tags: list[str]
) -> str:
    """CONCEPT:AU-OS.state.homeostatic-model-downgrade — Homeostatic Model Downgrade.

    When the ResourceOptimizer detects budget pressure, autonomously
    downgrade the tier to reduce cost — the system's "blood pressure"
    regulation.  The optimizer's select_model_for_step() already knows
    how to map remaining_pct → effective_complexity; we just need to
    ask it and let it override our heuristic tier. Any failure leaves
    ``tier`` unchanged — this is an optional override, not a requirement.
    """
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
    except Exception as e:  # noqa: BLE001 — resource_optimizer.select_model_for_step() is an optional budget-pressure override; tier already holds the heuristic/registry value on entry, so a failure here just skips the optional downgrade
        logger.debug(
            f"CONCEPT:AU-OS.state.homeostatic-model-downgrade homeostatic check skipped: {e}"
        )
    return tier


def _compute_confidence_signal(ctx_deps: Any, node_id: str) -> float:
    """CONCEPT:AU-ORCH.adapter.hot-cache-invalidation — Confidence-Gated Model Router.

    Blends a runtime WorkspaceAttention score (70%) with historical
    MemoryRetriever tool proficiency (30%) into one confidence signal used
    to adaptively select cheaper or more expensive models. Both sources
    degrade gracefully to a neutral 0.5 when unavailable.
    """
    knowledge_engine = getattr(ctx_deps, "knowledge_engine", None)

    # Source 1: WorkspaceAttention attention score (runtime signal)
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
    return 0.7 * runtime_confidence + 0.3 * historical_confidence


def _pick_adaptive_model(
    registry: Any,
    tier: str,
    confidence_signal: float,
    routing_percentile: float,
    required_tags: list[str],
    node_id: str,
    ctx_deps: Any,
) -> Any:
    """Final confidence-gated adaptive pick + concrete model build.

    Composes with :func:`_apply_homeostatic_downgrade`: budget pressure
    adjusts the tier first, then confidence further refines within the
    budget-allowed range. Falls back to ``ctx_deps.agent_model`` on any
    failure — the function never raises.
    """
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
        chosen = _resolve_explicit_model(
            registry, step_model_id, node_id, "Conductor-assigned"
        )
        if chosen is not None:
            return chosen

    requested_id = getattr(ctx_deps, "requested_model_id", None)
    if requested_id:
        chosen = _resolve_explicit_model(
            registry, requested_id, node_id, "user-requested"
        )
        if chosen is not None:
            return chosen

    tier, required_tags = _resolve_tier_and_tags(node_id)

    resource_optimizer = getattr(ctx_deps, "resource_optimizer", None)
    if resource_optimizer is not None:
        tier = _apply_homeostatic_downgrade(
            resource_optimizer, tier, node_id, required_tags
        )

    confidence_signal = _compute_confidence_signal(ctx_deps, node_id)
    routing_percentile = getattr(ctx_deps, "routing_percentile", 50.0)

    return _pick_adaptive_model(
        registry,
        tier,
        confidence_signal,
        routing_percentile,
        required_tags,
        node_id,
        ctx_deps,
    )


class _NormalizedAgentFields(NamedTuple):
    """Normalized, lower-cased fields used by :func:`agent_matches_node_id`'s match strategies."""

    name: str
    mcp_tools: str
    server: str
    desc: str
    capabilities: list[str]


def _normalize_agent_fields(agent: MCPAgent) -> _NormalizedAgentFields:
    name = agent.name.lower().replace("-", "_").replace(" ", "_")
    mcp_tools = (agent.mcp_tools or "").lower().replace("-", "_").replace(" ", "_")
    server = (agent.mcp_server or "").lower().replace("-", "_").replace(" ", "_")
    desc = (agent.description or "").lower()
    capabilities = [c.lower().replace("-", "_") for c in agent.capabilities]
    # Also keep originals to be safe
    capabilities.extend([c.lower() for c in agent.capabilities])
    return _NormalizedAgentFields(name, mcp_tools, server, desc, capabilities)


def _matches_exact(node_id_norm: str, f: _NormalizedAgentFields) -> bool:
    return (
        f.name == node_id_norm
        or f.mcp_tools == node_id_norm
        or f.server == node_id_norm
        or node_id_norm in f.capabilities
    )


def _matches_substring(node_id_norm: str, f: _NormalizedAgentFields) -> bool:
    return (bool(f.name) and f.name in node_id_norm) or (
        bool(f.server) and f.server in node_id_norm
    )


def _matches_affix(node_id_norm: str, f: _NormalizedAgentFields) -> bool:
    if f.name and (node_id_norm.startswith(f.name) or node_id_norm.endswith(f.name)):
        return True
    return bool(
        f.server
        and (node_id_norm.startswith(f.server) or node_id_norm.endswith(f.server))
    )


def _matches_keyword(
    node_id_norm: str, f: _NormalizedAgentFields, agent_name: str
) -> bool:
    stop_words = {"researcher", "expert", "agent", "manager", "action"}
    node_keywords = {
        w for w in node_id_norm.split("_") if len(w) >= 3 and w not in stop_words
    }
    for kw in node_keywords:
        if kw in f.name or kw in f.desc:
            logger.debug(f"Keyword match: '{kw}' found in name/desc of '{agent_name}'")
            return True
    return False


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
    fields = _normalize_agent_fields(agent)

    return (
        _matches_exact(node_id_norm, fields)
        or _matches_substring(node_id_norm, fields)
        or _matches_affix(node_id_norm, fields)
        or _matches_keyword(node_id_norm, fields, agent.name)
    )


def _inject_generic_toolkits(node_id: str, skill_tags: list[str]) -> list[Any]:
    """CONCEPT:AU-ORCH.execution.orchestration-flow-mermaid (perf) — capability-gated generic-tool injection.

    Previously the 13 developer_tools + 10 sdd_tools were dumped onto EVERY node
    unconditionally. For an MCP-server node (e.g. "repository-manager-mcp") that drowns
    the server's real tools in 23 irrelevant ones → wrong-tool selection (rg / SDD) and
    ~3.5–9K wasted context tokens per call. Only inject the generic toolkits when the
    node's name/capabilities indicate it does code/shell work (dev) or spec/planning (sdd).
    """
    from ..tools.developer_tools import developer_tools
    from ..tools.sdd_tools import sdd_tools

    _dev_tool_tags = {
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
    _sdd_tool_tags = {
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
    if any(t in _tag_blob for t in _dev_tool_tags):
        tools += list(developer_tools)
    if any(t in _tag_blob for t in _sdd_tool_tags):
        tools += list(sdd_tools)
    return tools


def _collect_skill_toolsets(node_id: str, skill_tags: list[str]) -> list[Any]:
    """Build the specialized-skill toolset list for one specialist, given its capability tags.

    Returns an empty list if ``pydantic-ai-skills`` is not installed, or if no
    skill directory matches ``skill_tags``.
    """
    toolsets: list[Any] = []
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

    return toolsets


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

    # CONCEPT:AU-ORCH.routing.offload-sync-roundtrip — registry hydration is a synchronous
    # backend round-trip; keep it off the event loop.
    registry = await asyncio.to_thread(get_discovery_registry)
    agent = next((a for a in registry.agents if a.name == node_id), None)
    skill_tags = agent.capabilities if agent else []

    tools = _inject_generic_toolkits(node_id, skill_tags)

    if not skill_tags:
        return tools, []

    logger.debug(
        f"Loading {len(skill_tags)} specialized skill tags for '{node_id}': {skill_tags}"
    )
    toolsets = _collect_skill_toolsets(node_id, skill_tags)
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


def _intersect_principal_ceiling(
    state: Any, allowed: list[str] | None, tools: list[Any], toolsets: list[Any]
) -> list[str] | None:
    """Intersect the invoker's tool allow-list with the ultimate principal's capability ceiling.

    CONCEPT:AU-OS.identity.per-agent-on-behalf-delegation (decision 4). A spawn can never exceed
    its ultimate human/service principal: the resolved ``invoker_allowed_tools`` is intersected
    with ``invoker_capability_ceiling`` (the principal's ``base_capabilities()``). Delegated
    ``on`` mode returns the narrowed set (tools exceeding the ceiling are dropped — surfaced by
    the caller's existing empty-allow-list guard as a loud ceiling-violation denial); ``warn``
    logs the would-be denials and returns the list unchanged; ``off``/no-ceiling is a no-op.
    The narrowing is applied only to an explicit least-privilege list, so an unrestricted spawn
    (governed by task-aware selection) is never accidentally emptied here — its principal ceiling
    is still enforced on the wire envelope and the run-token scope.
    """
    ceiling = getattr(state, "invoker_capability_ceiling", None)
    if not allowed or not ceiling:
        return allowed
    try:
        from agent_utilities.security.delegation import (
            current_delegation,
            delegation_mode,
            enforce_ceiling,
        )
    except Exception:  # noqa: BLE001 — delegation layer optional
        return allowed
    delegation = current_delegation()
    mode = delegation.mode if delegation is not None else delegation_mode()
    decision = enforce_ceiling(allowed, ceiling, mode=mode, context="apply_tool_scope")
    # A full ceiling denial (the spawn requested ONLY tools outside its principal's ceiling)
    # must FAIL CLOSED — an empty list here would otherwise read as "no restriction" in the
    # caller's ``if not allowed`` guard and silently open every tool. Raise the ceiling-violation
    # loudly instead (only in enforcing ``on`` mode; ``warn`` leaves the request unchanged).
    if decision.enforced and not decision.effective:
        raise RuntimeError(
            "delegation ceiling denied EVERY requested tool: the spawn requested "
            f"{sorted(str(t) for t in allowed)[:8]} but its principal ceiling permits none "
            "of them (a spawn can never exceed its ultimate principal's capabilities)"
        )
    return list(decision.effective)


def apply_tool_scope(
    state: Any, tools: list[Any], toolsets: list[Any]
) -> tuple[list[Any], list[Any]]:
    """CONCEPT:AU-ORCH.session.invoker-agent-handoff — enforce the invoker's least-privilege tool allow-list.

    When ``GraphState.invoker_allowed_tools`` is set, function tools are filtered by name and
    MCP/skill toolsets are wrapped with a pydantic-ai ``.filtered()`` predicate so the spawned
    agent can ONLY call the allowed tools. Empty/None allow-list = no restriction.
    """
    allowed = getattr(state, "invoker_allowed_tools", None)
    allowed = _intersect_principal_ceiling(state, allowed, tools, toolsets)
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
    Every spawn also gets ``per_request_input_tokens_limit`` (CONCEPT:AU-ORCH.execution.execution-budget-caps)
    so an oversized tool result terminates the spawn instead of compounding across
    its remaining requests — this is exactly the ServiceNow production failure
    class: a fleet tool that silently ignored an unknown ``limit`` argument
    returned 212 KB from one call. Note the cap is checked against the
    provider-reported ``input_tokens`` of the RESPONSE (pydantic-ai's
    ``count_tokens_before_request`` stays at its default ``False``), so that one
    oversized request is still sent and billed; what the cap prevents is carrying
    it forward. See D-W15-12 in ``reports/deferred/waves1-5-gate.md``.
    """
    from pydantic_ai.usage import UsageLimits

    from agent_utilities.orchestration.loop_guards import (
        DEFAULT_PER_REQUEST_INPUT_TOKENS_LIMIT,
    )

    req = setting("AGENT_REQUEST_LIMIT", request_limit)
    budget = getattr(state, "invoker_budget_tokens", None)
    if budget and int(budget) > 0:
        return UsageLimits(
            request_limit=req,
            total_tokens_limit=int(budget),
            per_request_input_tokens_limit=DEFAULT_PER_REQUEST_INPUT_TOKENS_LIMIT,
        )
    return UsageLimits(
        request_limit=req,
        per_request_input_tokens_limit=DEFAULT_PER_REQUEST_INPUT_TOKENS_LIMIT,
    )


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


async def _check_specialist_precondition_or_fallback(
    ctx: StepContext, agent_info: MCPAgent, agent_name: str, server_name: str | None
) -> str | None:
    """Precondition guard for a dynamic MCP specialist.

    Returns a fallback result to return immediately, or None if the
    specialist may proceed.
    """
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
    return None


def _resolve_specialist_tool_list(ctx: StepContext, agent_info: MCPAgent) -> list[str]:
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

    return total_tools


def _build_specialist_system_prompt(
    ctx: StepContext, agent_info: MCPAgent, agent_name: str, total_tools: list[str]
) -> str:
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
    agent_sys_prompt += invoker_context_section(
        ctx.state
    )  # CONCEPT:AU-ORCH.session.invoker-agent-handoff

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
    return agent_sys_prompt


def _emit_specialist_startup_event(
    ctx: StepContext, agent_info: MCPAgent, total_tools: list[str]
) -> None:
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


async def _fetch_auto_activate_capability_rows(
    ctx: StepContext, agent_name: str
) -> list[Any]:
    """Fetch AgentCapability rows registered with ``auto_activate = true`` for ``agent_name``."""
    if not ctx.deps.knowledge_engine or not ctx.deps.knowledge_engine.backend:
        return []
    return await asyncio.to_thread(
        ctx.deps.knowledge_engine.backend.execute,
        "MATCH (a {name: $name})-[:has_capability]->(c:AgentCapability) "
        "WHERE c.auto_activate = true RETURN c",
        {"name": agent_name},
    )


def _capability_should_activate(ctx: StepContext, cap_data: dict[str, Any]) -> bool:
    """Evaluate a capability's ``trigger_conditions`` against the current step."""
    triggers = cap_data.get("trigger_conditions", {})
    if "input_chars_gt" in triggers:
        return len(ctx.state.query) > triggers["input_chars_gt"]
    return True


def _activate_one_capability(ctx: StepContext, agent_name: str, row: Any) -> str | None:
    """Activate a single capability row if it has a handler and its triggers pass.

    Returns the ``capability_type`` on activation, else ``None``.
    """
    cap_data = row.get("c", row)
    cap_type = cap_data.get("capability_type", "unknown")
    handler_module = cap_data.get("handler_module")
    handler_fn = cap_data.get("handler_function")
    if not (handler_module and handler_fn):
        return None
    if not _capability_should_activate(ctx, cap_data):
        return None

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
    return cap_type


async def _activate_specialist_capabilities(ctx: StepContext, agent_name: str) -> None:
    """Auto-activate any registered specialist capabilities (write-only
    telemetry -- see the noqa comment below; no downstream read in the
    caller either before or after this extraction)."""
    # CONCEPT:AU-ORCH.adapter.hot-cache-invalidation — Capability Auto-Activation
    # Check if this specialist has registered capabilities (e.g., RLM, critic)
    # and activate them before execution.
    if not ctx.deps.knowledge_engine:
        return

    activated_capabilities: list[str] = []
    try:
        cap_rows = await _fetch_auto_activate_capability_rows(ctx, agent_name)
        for row in cap_rows:
            cap_type = _activate_one_capability(ctx, agent_name, row)
            if cap_type is not None:
                activated_capabilities.append(cap_type)
    except Exception as e:  # noqa: BLE001 — activated_capabilities is write-only telemetry (no downstream read of the list in this function); a lookup failure means zero capabilities auto-activate/log for this step, degrading to pre-feature behavior
        logger.debug(f"Capability auto-activation lookup failed: {e}")


async def _score_specialist_attention(ctx: StepContext, agent_name: str) -> None:
    """Best-effort WorkspaceAttention scoring (write-only priority signal --
    see the noqa comment below; the result is logged only, never read by the
    caller before or after this extraction)."""
    attention_score: float | None = None
    # CONCEPT:AU-KG.compute.workspace-attention-scoring — WorkspaceAttention scoring for specialist priority
    if ctx.deps.knowledge_engine:
        try:
            from .workspace_attention import WorkspaceAttention

            wa = WorkspaceAttention(ctx.deps.knowledge_engine)
            attention_score = await asyncio.to_thread(
                wa.get_attention_score, agent_name
            )
            if attention_score is not None:
                logger.info(
                    f"[GWT] Specialist '{agent_name}' attention score: {attention_score:.2f}"
                )
        except Exception as e:  # noqa: BLE001 — attention_score is a best-effort priority signal (stays None on failure); the specialist dispatch below does not gate on it
            logger.debug(f"WorkspaceAttention scoring failed for '{agent_name}': {e}")


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
    precondition_result = await _check_specialist_precondition_or_fallback(
        ctx, agent_info, agent_name, server_name
    )
    if precondition_result is not None:
        return precondition_result

    logger.info(f"[LAYER:GRAPH:EXPERT] Running dynamic MCP agent '{agent_name}'")

    total_tools = _resolve_specialist_tool_list(ctx, agent_info)

    agent_sys_prompt = _build_specialist_system_prompt(
        ctx, agent_info, agent_name, total_tools
    )

    _emit_specialist_startup_event(ctx, agent_info, total_tools)

    await _activate_specialist_capabilities(ctx, agent_name)

    await _score_specialist_attention(ctx, agent_name)

    agent = create_context_agent(
        model=ctx.deps.agent_model,
        system_prompt=agent_sys_prompt,
        deps_type=GraphDeps,
    )

    from contextlib import AsyncExitStack

    async with AsyncExitStack() as stack:
        (
            matched_toolsets,
            bound_tool_count,
            actually_bound_tools,
        ) = await _bind_specialist_toolsets(ctx, stack, agent_info, total_tools)

        agent = _build_guarded_specialist_agent(ctx, agent_sys_prompt, matched_toolsets)

        _emit_tool_binding_telemetry(
            ctx, agent_info, bound_tool_count, actually_bound_tools, matched_toolsets
        )

        sub_query, node_timeout = _resolve_specialist_query_and_timeout(ctx, agent_info)

        # Retrieve cached message history for re-dispatch context
        cache_key = agent_info.name.lower().replace(" ", "_")
        prev_messages = ctx.deps.message_history_cache.get(cache_key)

        max_attempts = 3
        state = _SpecialistDispatchState(
            ctx=ctx,
            agent_info=agent_info,
            agent=agent,
            agent_sys_prompt=agent_sys_prompt,
            sub_query=sub_query,
            node_timeout=node_timeout,
            cache_key=cache_key,
            prev_messages=prev_messages,
            server_name=server_name,
            agent_name=agent_name,
            max_attempts=max_attempts,
        )

        async def _dispatch_once() -> str:
            return await _dispatch_specialist_once(state)

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
            return await run_with_resilience(_dispatch_once, dispatch_policy)
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

        ctx.state.error = f"Agent '{agent_name}' failed after {max_attempts} attempts: {state.last_error}"
        raise RuntimeError(ctx.state.error)


def _toolset_matches_target(toolset: Any, target: str) -> str | None:
    """Return the toolset's normalized server_id if it matches ``target``, else ``None``."""
    server_id = getattr(toolset, "id", getattr(toolset, "name", None))
    if not server_id:
        return None
    current = server_id.lower().replace("-", "_")
    if (
        current == target
        or current.startswith(f"{target}_")
        or target.startswith(f"{current}_")
    ):
        return server_id
    return None


def _bind_matched_toolset(
    toolset: Any,
    total_tools: list[Any],
    actually_bound_tools: list[str],
) -> int:
    """Merge one matched toolset's tool names into ``actually_bound_tools`` (in place);
    returns its tool-count contribution."""
    if hasattr(toolset, "tools"):
        for t_name in toolset.tools.keys():
            if t_name not in actually_bound_tools:
                actually_bound_tools.append(t_name)
        return len(toolset.tools)
    return len(total_tools)


def _match_bound_toolsets(
    ctx: StepContext, agent_info: MCPAgent, total_tools: list[Any]
) -> tuple[list[Any], int, list[str]]:
    """Match already-live MCP toolsets against ``agent_info.mcp_server`` (deduplicated)."""
    bound_tool_count = 0
    actually_bound_tools: list[str] = []
    matched_toolsets: list[Any] = []
    seen_toolset_ids: set[int] = set()
    target = (agent_info.mcp_server or "").lower().replace("-", "_")

    for toolset in ctx.deps.mcp_toolsets:
        if id(toolset) in seen_toolset_ids:
            continue
        server_id = _toolset_matches_target(toolset, target)
        if server_id is None:
            continue

        seen_toolset_ids.add(id(toolset))
        matched_toolsets.append(toolset)
        bound_tool_count += _bind_matched_toolset(
            toolset, total_tools, actually_bound_tools
        )
        logger.info(
            f"[LAYER:GRAPH:EXPERT] Bound toolset '{server_id}' to expert '{agent_info.name}'"
        )
    return matched_toolsets, bound_tool_count, actually_bound_tools


async def _lazy_load_specialist_toolset(
    ctx: StepContext,
    stack: Any,
    agent_info: MCPAgent,
    matched_toolsets: list[Any],
    bound_tool_count: int,
    actually_bound_tools: list[str],
) -> tuple[list[Any], int, list[str]]:
    """When no live toolset matched, lazily load + enter the target MCP server from config."""
    target_server_name = (agent_info.mcp_server or "").lower().replace("-", "_")
    if not target_server_name or matched_toolsets:
        return matched_toolsets, bound_tool_count, actually_bound_tools
    try:
        from pydantic_ai.mcp import load_mcp_toolsets

        from agent_utilities.core.workspace import resolve_mcp_config_path
        from agent_utilities.mcp.protocol_compat import (
            force_legacy_protocol_mode,
            install_mcp_v2_bridge,
        )

        install_mcp_v2_bridge()

        mcp_path = resolve_mcp_config_path(None)
        if mcp_path and mcp_path.exists():
            all_servers = force_legacy_protocol_mode(load_mcp_toolsets(mcp_path))
            for srv in all_servers:
                srv_id = getattr(srv, "id", getattr(srv, "name", str(srv)))
                current = srv_id.lower().replace("-", "_")
                if not (
                    current == target_server_name
                    or current.startswith(f"{target_server_name}_")
                    or target_server_name.startswith(f"{current}_")
                ):
                    continue
                logger.info(
                    f"[LAYER:GRAPH:EXPERT] Lazy loading MCP server '{srv_id}' for expert '{agent_info.name}'"
                )
                await stack.enter_async_context(srv)
                matched_toolsets.append(srv)
                _tools = getattr(srv, "tools", {})
                for t_name in _tools.keys() if hasattr(_tools, "keys") else []:
                    if t_name not in actually_bound_tools:
                        actually_bound_tools.append(t_name)
                bound_tool_count += len(_tools)
                break
    except Exception as e:
        logger.warning(f"Failed to lazy load MCP server '{target_server_name}': {e}")
    return matched_toolsets, bound_tool_count, actually_bound_tools


async def _bind_specialist_toolsets(
    ctx: StepContext, stack: Any, agent_info: MCPAgent, total_tools: list[Any]
) -> tuple[list[Any], int, list[str]]:
    """Bind the specific subset of MCP tools for this specialist (dedup + lazy-load fallback)."""
    matched_toolsets, bound_tool_count, actually_bound_tools = _match_bound_toolsets(
        ctx, agent_info, total_tools
    )
    return await _lazy_load_specialist_toolset(
        ctx, stack, agent_info, matched_toolsets, bound_tool_count, actually_bound_tools
    )


def _build_guarded_specialist_agent(
    ctx: StepContext, agent_sys_prompt: str, matched_toolsets: list[Any]
) -> Any:
    """Guard matched toolsets under caller-identity policy + invoker tool-scope, build the real agent."""
    # Bind MCP toolsets to the mandatory caller identity policy. Approval
    # decisions trigger DeferredToolRequests; policy denials fail closed.
    from agent_utilities.security.tool_guard import flag_mcp_tool_definitions

    guarded_toolsets = flag_mcp_tool_definitions(
        matched_toolsets,
        permissions_kernel=ctx.deps.permissions_kernel,
        agent_identity=ctx.deps.agent_identity,
        engine=ctx.deps.knowledge_engine,
    )

    # Include DeferredToolRequests in output type so the agent can defer
    # sensitive tool calls instead of failing.
    from pydantic_ai import DeferredToolRequests

    # CONCEPT:AU-ORCH.session.invoker-agent-handoff — enforce the invoker's least-privilege tool allow-list (if any).
    _scoped_tools, guarded_toolsets = apply_tool_scope(ctx.state, [], guarded_toolsets)

    # `ctx.deps.agent_model` resolves to `Any` (GraphDeps.agent_model, like the
    # rest of this generic pydantic-graph state — see the module-level mypy
    # override for this file), which combined with the `output_type=[...]` list
    # form and the generic AgentDepsT/OutputDataT inference doesn't land on any
    # single Agent() overload. Tracked with the file's other pydantic-graph
    # generic-typing debt (mypy-remediation-plan.md Phase 2), not a runtime bug.
    return create_context_agent(
        model=ctx.deps.agent_model,
        permissions_kernel=ctx.deps.permissions_kernel,
        agent_identity=ctx.deps.agent_identity,
        permission_engine=ctx.deps.knowledge_engine,
        system_prompt=agent_sys_prompt,
        deps_type=GraphDeps,
        toolsets=guarded_toolsets,
        output_type=[str, DeferredToolRequests],
        end_strategy="early",
    )


def _emit_tool_binding_telemetry(
    ctx: StepContext,
    agent_info: MCPAgent,
    bound_tool_count: int,
    actually_bound_tools: list[str],
    matched_toolsets: list[Any],
) -> None:
    """Tool-count telemetry: surfaces blind or overloaded adaptive_agent_router."""
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


def _resolve_specialist_query_and_timeout(
    ctx: StepContext, agent_info: MCPAgent
) -> tuple[str, float]:
    """Prefer the Conductor's refined subtask over the raw query; resolve the per-node timeout."""
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

    return sub_query, node_timeout


@dataclass
class _SpecialistDispatchState:
    """Mutable per-dispatch state shared across retry attempts of one specialist call.

    ``last_error``/``attempt_no`` are mutated in place across attempts (the same
    instance is reused by every retry of :func:`_dispatch_specialist_once`, exactly
    as the closure's ``nonlocal last_error, attempt_no`` did before extraction).
    """

    ctx: StepContext
    agent_info: MCPAgent
    agent: Any
    agent_sys_prompt: str
    sub_query: str
    node_timeout: float
    cache_key: str
    prev_messages: Any
    server_name: str | None
    agent_name: str
    max_attempts: int
    last_error: str | None = None
    attempt_no: int = 0


async def _run_specialist_llm_call(
    state: _SpecialistDispatchState, run_input: Any
) -> Any:
    """CONCEPT:AU-ORCH.execution.retry-predicate-raised-treating — the per-attempt LLM call, policy-wrapped.

    The single per-node-timeout LLM call is wrapped in the declarative
    retry/backoff/timeout policy so transient model or tool errors
    (TimeoutError/ConnectionError) are retried with exponential backoff *before*
    surfacing to the outer attempt loop. This composes with — and does not
    replace — the existing per-server circuit breaker (``server_health``) and the
    ``node_timeout`` second wait, which is now enforced per attempt by the
    policy's ``timeout_s``.
    """
    _policy = _specialist_resilience_policy(state.node_timeout)

    async def _run_agent_once(
        _agent: Any = state.agent,
        _input: Any = run_input,
        _hist: Any = state.prev_messages,
    ) -> Any:
        return await _agent.run(_input, deps=state.ctx.deps, message_history=_hist)

    return await run_with_resilience(_run_agent_once, _policy)


def _record_specialist_call_provenance(
    ctx: StepContext, res: Any, cache_key: str
) -> None:
    """Accumulate :ToolCall provenance and cache message history (both best-effort)."""
    # Accumulate this expert's tool calls for :ToolCall provenance on the
    # graph path (CONCEPT:AU-KG.temporal.message-history-read). Unconditional — the WebUI
    # event block below is gated on ``event_queue`` and skipped for headless
    # (MCP/telegram) delegations, which is exactly where provenance was lost.
    try:
        from ..orchestration.tool_provenance import extract_tool_calls

        ctx.state.tool_calls.extend(extract_tool_calls(res))
    except Exception as _tc_exc:  # noqa: BLE001 — never break a run
        logger.debug("expert tool-call provenance skipped: %s", _tc_exc)

    # Cache message history for potential re-dispatch
    try:
        ctx.deps.message_history_cache[cache_key] = res.all_messages()
    except Exception as e:  # noqa: BLE001 — message_history_cache is a best-effort re-dispatch optimization; a failed write just means a future re-dispatch re-fetches/regenerates history instead of reusing a cached copy, it does not lose the run's actual result (already handled above)
        logger.debug(f"Failed to update cache for '{cache_key}': {e}")


def _record_specialist_circuit_success(
    ctx: StepContext, server_name: str | None
) -> None:
    """Record a successful call on the per-server circuit breaker."""
    srv_name = server_name or "unknown"
    if srv_name not in ctx.deps.server_health:
        ctx.deps.server_health[srv_name] = MCPServerHealth(
            server_name=srv_name,
        )
    ctx.deps.server_health[srv_name].record_success()


def _stream_response_parts(ctx: StepContext, agent_info: MCPAgent, msg: Any) -> None:
    """Stream one ModelResponse's tool-call/text parts to the WebUI event queue."""
    from pydantic_ai.messages import ToolCallPart

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


def _stream_request_parts(ctx: StepContext, agent_info: MCPAgent, msg: Any) -> None:
    """Stream one ModelRequest's tool-return parts to the WebUI event queue."""
    from pydantic_ai.messages import ToolReturnPart

    for req_part in msg.parts:
        if isinstance(req_part, ToolReturnPart):
            emit_graph_event(
                ctx.deps.event_queue,
                event_type="tool_result",
                agent=agent_info.name,
                tool=req_part.tool_name,
                result=str(req_part.content)[:500],
            )


def _stream_specialist_events(ctx: StepContext, agent_info: MCPAgent, res: Any) -> None:
    """Stream this call's tool-call/text/tool-result events to the WebUI event queue."""
    if not ctx.deps.event_queue:
        return
    from pydantic_ai.messages import ModelRequest, ModelResponse

    for msg in res.all_messages():
        if isinstance(msg, ModelResponse):
            _stream_response_parts(ctx, agent_info, msg)
        elif isinstance(msg, ModelRequest):
            _stream_request_parts(ctx, agent_info, msg)


async def _summarize_oversized_result(
    ctx: StepContext, agent_info: MCPAgent, result_str: str
) -> str:
    """RLM Large Result Summarization: shrink an oversized specialist result via RLM, else truncate."""
    from ..rlm.config import RLMConfig

    rlm_config = RLMConfig()
    if len(result_str) <= rlm_config.max_context_threshold:
        return result_str

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
        return f"[RLM Synthesized Summary of Massive Data]\n{summary}"
    except Exception as rlm_err:
        logger.error(f"RLM summarization failed: {rlm_err}")
        return (
            result_str[: rlm_config.max_context_threshold]
            + "... [TRUNCATED DUE TO SIZE & RLM FAILURE]"
        )


def _synthesize_from_tool_returns(
    agent_info: MCPAgent, res: Any, result_str: str
) -> str:
    """Data Enhancement Synthesizer: inject raw tool-return data when the LLM claims 'no data'."""
    if "no data" not in result_str.lower() and "returned no" not in result_str.lower():
        return result_str

    from pydantic_ai.messages import ModelRequest, ToolReturnPart

    tool_returns: list[str] = []
    for msg in res.all_messages():
        if isinstance(msg, ModelRequest):
            for ret_part in msg.parts:
                if isinstance(ret_part, ToolReturnPart) and ret_part.content:
                    content_str = str(ret_part.content)
                    if content_str and content_str not in ("[]", "None", "null", ""):
                        tool_returns.append(f"**{ret_part.tool_name}**: {content_str}")
    if not tool_returns:
        return result_str

    logger.warning(
        f"Expert '{agent_info.name}': LLM dismissed tool response data. "
        f"Injection {len(tool_returns)} raw tool return(s) into result."
    )
    return (
        "### Tool Execution Results\n"
        + "\n".join(tool_returns)
        + f"\n\n### Agent Summary\n{result_str}"
    )


async def _finalize_specialist_dispatch_success(
    state: _SpecialistDispatchState, result_str: str
) -> str:
    """Write the result to the registry, mark routed_domain, and run the HSM success exit."""
    node_uid = f"{state.cache_key}_{state.ctx.state.step_cursor}"
    state.ctx.state.results_registry[node_uid] = result_str

    result_key = state.agent_info.name or state.cache_key
    state.ctx.state.routed_domain = result_key
    logger.info(
        f"Expert: '{state.agent_info.name}' succeeded (attempt {state.attempt_no}). "
        f"Result: {len(result_str)} chars. Registry key: '{node_uid}'"
    )
    # Emit completion event
    emit_graph_event(
        state.ctx.deps.event_queue,
        "subagent_completed",
        domain=state.agent_info.name or "unknown",
        status="success",
    )
    # HSM: Exit action (success)
    await on_exit_specialist(
        ctx_deps=state.ctx.deps,
        ctx_state=state.ctx.state,
        agent_name=state.agent_name,
        success=True,
        server_name=state.server_name or "unknown",
    )
    return "execution_joiner"


def _handle_specialist_timeout(state: _SpecialistDispatchState) -> None:
    """Record + emit a per-attempt timeout (the caller re-raises)."""
    state.last_error = f"Timeout after {state.node_timeout}s"
    logger.warning(
        f"Expert '{state.agent_name}' timed out (attempt {state.attempt_no}/{state.max_attempts})"
    )
    emit_graph_event(
        state.ctx.deps.event_queue,
        "expert_complete",
        expert=state.agent_info.name,
        status="timeout",
    )


def _handle_specialist_error(state: _SpecialistDispatchState, e: Exception) -> None:
    """Record + emit a per-attempt failure (the caller re-raises)."""
    state.last_error = str(e)
    logger.warning(
        f"Expert '{state.agent_name}' failed (attempt {state.attempt_no}/{state.max_attempts}): {e}"
    )
    emit_graph_event(
        state.ctx.deps.event_queue,
        "subagent_tool_call",
        domain=state.agent_info.name or "unknown",
        tool_name=getattr(e, "tool_name", "unknown"),
        args=getattr(e, "args", {}),
    )
    emit_graph_event(
        state.ctx.deps.event_queue,
        "expert_complete",
        expert=state.agent_info.name,
        status="error",
        error=str(e),
    )


async def _dispatch_specialist_once(state: _SpecialistDispatchState) -> str:
    """One attempt of a specialist LLM dispatch: call, record, stream, synthesize, finalize."""
    state.attempt_no += 1
    emit_graph_event(
        state.ctx.deps.event_queue,
        "expert_thinking",
        expert=state.agent_info.name,
        attempt=state.attempt_no,
    )
    try:
        logger.info(
            f"[LAYER:GRAPH:EXPERT] '{state.agent_info.name}' LLM Call Starting (attempt {state.attempt_no}). Prompt length: {len(state.agent_sys_prompt)}"
        )
        # Wrap user query in XML tags to protect against prompt injection
        # and provide clear boundaries for the model.
        raw_input = (
            state.ctx.state.query_parts
            if state.ctx.state.query_parts and state.sub_query == state.ctx.state.query
            else state.sub_query
        )
        run_input = (
            f"<user_query>\n{raw_input}\n</user_query>"
            if isinstance(raw_input, str)
            else raw_input
        )
        res = await _run_specialist_llm_call(state, run_input)
        logger.info(
            f"[LAYER:GRAPH:EXPERT] '{state.agent_info.name}' LLM Call Completed."
        )
        state.ctx.state._update_usage(getattr(res, "usage", None))

        _record_specialist_call_provenance(state.ctx, res, state.cache_key)
        _record_specialist_circuit_success(state.ctx, state.server_name)
        _stream_specialist_events(state.ctx, state.agent_info, res)

        result_str = str(res.output)
        result_str = await _summarize_oversized_result(
            state.ctx, state.agent_info, result_str
        )
        result_str = _synthesize_from_tool_returns(state.agent_info, res, result_str)

        return await _finalize_specialist_dispatch_success(state, result_str)

    except TimeoutError:
        _handle_specialist_timeout(state)
        raise
    except Exception as e:
        _handle_specialist_error(state, e)
        raise


async def _find_fallback_siblings(failed_agent: MCPAgent) -> list[MCPAgent]:
    """Find candidate sibling specialists on the same MCP server as ``failed_agent``."""
    # CONCEPT:AU-ORCH.routing.offload-sync-roundtrip — registry hydration is a synchronous
    # backend round-trip; keep it off the event loop.
    registry = await asyncio.to_thread(get_discovery_registry)
    return [
        a
        for a in registry.agents
        if a.mcp_server == failed_agent.mcp_server
        and a.name != failed_agent.name
        and a.name != failed_agent.name
    ]


def _best_fallback_sibling(
    siblings: list[MCPAgent], query: str
) -> tuple[MCPAgent | None, int]:
    """Score candidate siblings by keyword overlap with the query and return the best."""
    query_words = set(query.lower().split())
    best_sibling: MCPAgent | None = None
    best_score = 0
    for sibling in siblings:
        tag_words = set(
            sibling.name.lower().replace("-", " ").replace("_", " ").split()
        )
        score = len(query_words & tag_words)
        if score > best_score:
            best_score = score
            best_sibling = sibling
    return best_sibling, best_score


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
    siblings = await _find_fallback_siblings(failed_agent)
    if not siblings:
        return None

    best_sibling, best_score = _best_fallback_sibling(siblings, ctx.state.query)
    if not (best_sibling and best_score > 0):
        return None

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


def _resolve_a2a_sub_query(ctx: StepContext) -> Any:
    """Use the expert's specific question (from step_input.description) or the original query."""
    sub_query = ctx.state.query
    step_input = ctx.inputs
    if isinstance(step_input, ExecutionStep) and step_input.description:
        if isinstance(step_input.description, dict):
            sub_query = step_input.description.get("question", sub_query)
        elif isinstance(step_input.description, str):
            sub_query = step_input.description
    return sub_query


async def _annotate_a2a_epistemic(envelope: dict[str, Any], node_id: str) -> None:
    """Forward an A2A envelope's optional epistemic metadata to telemetry (best-effort)."""
    epistemic = envelope.get("epistemic") or {}
    if not epistemic:
        return
    try:
        from agent_utilities.observability import get_telemetry_engine

        get_telemetry_engine().annotate_epistemic(
            confidence=epistemic.get("confidence"),
            status=epistemic.get("status"),
            contradiction_count=epistemic.get("contradiction_count"),
            policy_labels=epistemic.get("policy_labels"),
            model=node_id,
        )
    except (  # noqa: BLE001 — result_str is already computed and written to ctx.state.results_registry above before this block; the try only forwards optional epistemic metadata to telemetry (comment: "tracing must never break the graph")
        Exception
    ) as exc:  # pragma: no cover - tracing must never break the graph
        logger.debug("A2A epistemic span annotation skipped for %s: %s", node_id, exc)


async def _execute_remote_a2a_agent(ctx: StepContext, node_id: str, meta: dict) -> None:
    """Handle the ``remote_a2a`` branch of :func:`_execute_agent_package_logic`.

    Calls the peer over HTTP/SSE via :class:`A2AClient`, stores the
    byte-identical result in ``ctx.state.results_registry``, and forwards any
    epistemic metadata to telemetry (best-effort).
    """
    from agent_utilities.protocols.a2a import A2AClient

    peer_url = meta["url"]
    logger.info(f"Expert Execution: Calling remote A2A agent '{node_id}' at {peer_url}")
    client = A2AClient(timeout=ctx.deps.approval_timeout or 300.0)

    sub_query = _resolve_a2a_sub_query(ctx)

    # CONCEPT:AU-KB-CURRENCY (A2A projection) — use the envelope variant
    # so a peer's epistemic metadata (confidence/status/
    # contradiction_count/policy_labels/source_refs, when it sends any)
    # is visible, while `result_str` stays BYTE-IDENTICAL to what plain
    # `execute_task` would have returned (content on success, the same
    # "Error: ..."/"A2A Error: ..." string on failure) — no behavior
    # change to the existing result-registry path.
    envelope = await client.execute_task_with_epistemic(peer_url, sub_query)
    result_str = envelope.get("content") or envelope.get("error") or ""
    # Unified result storage
    node_uid = f"{node_id}_{ctx.state.step_cursor}"
    ctx.state.results_registry[node_uid] = result_str

    await _annotate_a2a_epistemic(envelope, node_id)


async def _execute_local_agent_package(
    ctx: StepContext, node_id: str
) -> str | End[Any] | None:
    """Handle the local (non-A2A) branch of :func:`_execute_agent_package_logic`.

    Returns an early result (``str``/``End``) when the prompt-based specialist
    path short-circuits the graph, else ``None`` to signal the caller should
    fall through to ``"execution_joiner"``.
    """
    # CONCEPT:AU-ORCH.adapter.hot-cache-invalidation: Unified specialist execution
    # Try specialized prompt-based execution first (loads persona, injects tools + skills)
    # CONCEPT:AU-ORCH.routing.offload-sync-roundtrip — registry hydration is a synchronous
    # backend round-trip; keep it off the event loop.
    registry = await asyncio.to_thread(get_discovery_registry)
    mcp_agent = next(
        (a for a in registry.agents if agent_matches_node_id(a, node_id)),
        None,
    )

    if mcp_agent and mcp_agent.mcp_server:
        # MCP-bound specialist — execute with bound tools
        await _execute_dynamic_mcp_agent(ctx, mcp_agent)
        return None
    if mcp_agent and mcp_agent.json_blueprint:
        # Prompt-based specialist — execute with persona + injected tools
        return await _execute_specialized_step(ctx, node_id)

    # Fallback: try specialized step (prompt lookup by name), then generic
    try:
        return await _execute_specialized_step(ctx, node_id)
    except Exception:
        logger.warning(
            f"Expert Execution: Node '{node_id}' fallback. "
            f"No specialist metadata found in the Knowledge Graph."
        )
        await _execute_domain_logic(ctx, node_id)
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
    if meta.get("type") == "remote_a2a":
        await _execute_remote_a2a_agent(ctx, node_id, meta)
    else:
        early = await _execute_local_agent_package(ctx, node_id)
        if early is not None:
            return early

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

    # CONCEPT:AU-ORCH.routing.offload-sync-roundtrip — discovery reads the KG registry
    # synchronously; keep it off the event loop.
    discovered = await asyncio.to_thread(discover_agents)
    if node_id not in discovered:
        logger.error(f"Agent package node '{node_id}' not found in discovery.")
        return "Error"

    meta = discovered[node_id]
    return await _execute_agent_package_logic(ctx, node_id, meta)


def _resolve_tool_tags(registry: Any, prompt_name: str) -> list[str]:
    """Resolve skill tags for ``prompt_name`` from the unified registry
    (replaces the deprecated NODE_SKILL_MAP)."""
    agent_info = next((a for a in registry.agents if a.name == prompt_name), None)
    tool_tags = [prompt_name]
    if agent_info and agent_info.capabilities:
        # Capabilities field in NODE_AGENTS.md corresponds to skill tags
        tool_tags.extend(agent_info.capabilities)
    return list(set(tool_tags))


def _guard_single_toolset(ctx: StepContext, toolset: Any) -> Any:
    """Wrap one MCP toolset with the mandatory caller-identity guard policy."""
    from agent_utilities.security.tool_guard import flag_mcp_tool_definitions

    guarded = flag_mcp_tool_definitions(
        [toolset],
        permissions_kernel=ctx.deps.permissions_kernel,
        agent_identity=ctx.deps.agent_identity,
        engine=ctx.deps.knowledge_engine,
    )
    return guarded[0]


def _bind_native_toolset(
    ctx: StepContext, toolset: Any, seen: set[int]
) -> tuple[Any, int] | None:
    """Bind a GraphOS-native toolset unconditionally, else return ``None``.

    A native GraphOS toolset is already run-scoped to the exact skill and
    caller allow-list. It is not a fleet server and must not be discarded by
    server/domain tag matching.
    """
    metadata = getattr(toolset, "metadata", None)
    if not (isinstance(metadata, dict) and metadata.get("graphos_native") is True):
        return None
    seen.add(id(toolset))
    guarded = _guard_single_toolset(ctx, toolset)
    tool_count = len(getattr(toolset, "tools", {})) or 1
    return guarded, tool_count


def _bind_matching_toolset(
    ctx: StepContext,
    toolset: Any,
    prompt_name: str,
    tool_tags: list[str],
    seen: set[int],
) -> tuple[Any, int] | None:
    """Bind a toolset whose server id or tags match ``prompt_name``, else return ``None``."""
    server_id = (
        getattr(toolset, "id", getattr(toolset, "name", "unknown"))
        .lower()
        .replace("-", "_")
    )
    target = prompt_name.lower().replace("-", "_")
    if server_id != target and not any(
        t.lower().replace("-", "_") == target for t in tool_tags
    ):
        return None
    seen.add(id(toolset))
    guarded = _guard_single_toolset(ctx, toolset)
    tool_count = len(getattr(toolset, "tools", {})) or 1
    return guarded, tool_count


def _bind_filtered_toolset(
    ctx: StepContext, toolset: Any, tool_tags: list[str], seen: set[int]
) -> tuple[Any, int] | None:
    """Bind the tag-filtered subset of a toolset that didn't match directly, else ``None``."""
    filtered = filter_tools_by_tag(toolset, tool_tags)
    if not filtered or id(filtered) in seen:
        return None
    seen.add(id(filtered))
    guarded = _guard_single_toolset(ctx, filtered)
    tool_count = len(getattr(filtered, "tools", {})) or 1
    return guarded, tool_count


def _collect_mcp_toolsets_for_step(
    ctx: StepContext, prompt_name: str, tool_tags: list[str]
) -> tuple[list[Any], int]:
    """Filter+bind ``ctx.deps.mcp_toolsets`` by domain tag AND node_id (``prompt_name``),
    with deduplication, applying the mandatory caller-identity guard to each bound toolset."""
    seen: set[int] = set()
    mcp_tool_count = 0
    collected: list[Any] = []
    for toolset in ctx.deps.mcp_toolsets:
        if id(toolset) in seen:
            continue

        bound = _bind_native_toolset(ctx, toolset, seen)
        if bound is None:
            bound = _bind_matching_toolset(ctx, toolset, prompt_name, tool_tags, seen)
        if bound is None:
            bound = _bind_filtered_toolset(ctx, toolset, tool_tags, seen)

        if bound is not None:
            guarded, tool_count = bound
            collected.append(guarded)
            mcp_tool_count += tool_count

    return collected, mcp_tool_count


def _emit_tool_count_telemetry(
    ctx: StepContext, prompt_name: str, custom_tool_count: int, mcp_tool_count: int
) -> int:
    """Log + emit ``tools_bound`` telemetry for a specialized step; returns the total count."""
    total_tool_count = custom_tool_count + mcp_tool_count
    logger.info(
        f"[TELEMETRY] Specialist '{prompt_name}': "
        f"{custom_tool_count} dev/skill tools + {mcp_tool_count} MCP tools "
        f"= {total_tool_count} total"
    )
    emit_graph_event(
        ctx.deps.event_queue,
        "tools_bound",
        expert=prompt_name,
        count=total_tool_count,
        dev_tools=custom_tool_count,
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
    return total_tool_count


async def _cache_specialist_history(
    ctx: StepContext, prompt_name: str, stream: Any
) -> None:
    """Cache message history for potential re-dispatch (best-effort)."""
    try:
        history = stream.all_messages()
        if asyncio.iscoroutine(history):
            history = await history
        ctx.deps.message_history_cache[prompt_name] = history
    except Exception as e:  # noqa: BLE001 — same best-effort re-dispatch cache as the expert-dispatch path above; result_str is already stored in ctx.state.results_registry unconditionally above this block, so the step's actual output is unaffected
        logger.debug(f"Unable to cache: {e}")


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

    # CONCEPT:AU-ORCH.routing.offload-sync-roundtrip — the persona/memory/tool-guidance prompt
    # loads (file I/O + registry lookup) and the registry hydration below are all SYNCHRONOUS
    # and independent of each other; batch them into ONE thread hop instead of four.
    def _read_specialist_bindings() -> tuple[str, str, str, Any]:
        return (
            load_specialized_prompts(prompt_name),
            load_specialized_prompts("memory_instruction"),
            load_specialized_prompts("tool_guidance"),
            get_discovery_registry(),
        )

    (
        prompt,
        memory_instruction,
        tool_guidance,
        registry,
    ) = await asyncio.to_thread(_read_specialist_bindings)

    # Dynamic Skill Distribution
    custom_tools, skill_toolsets = await _get_domain_tools(prompt_name, ctx.deps)
    logger.info(
        f"[LAYER:GRAPH:EXPERT] Specialized step '{prompt_name}' started. Tools loaded: {len(custom_tools)}, Toolsets: {len(skill_toolsets)}"
    )

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
    # Bind each MCP toolset to the mandatory caller identity policy.
    tool_tags = _resolve_tool_tags(registry, prompt_name)
    collected_mcp_toolsets, mcp_tool_count = _collect_mcp_toolsets_for_step(
        ctx, prompt_name, tool_tags
    )

    # Build the agent with ALL toolsets at construction time.
    # agent.toolsets is a read-only property — appending after construction
    # is a silent no-op.

    from pydantic_ai import DeferredToolRequests

    # CONCEPT:AU-ORCH.routing.conductor-per-step-model — honor a Conductor-assigned per-step model_id (ctx.inputs is
    # the current ExecutionStep/Task); falls back to override/tier routing when unset.
    # CONCEPT:AU-ORCH.routing.offload-sync-roundtrip — internally does registry hydration +
    # WorkspaceAttention/MemoryRetriever KG round-trips; keep it off the event loop.
    specialist_model = await asyncio.to_thread(
        pick_specialist_model,
        ctx.deps,
        prompt_name,
        step_model_id=getattr(ctx.inputs, "model_id", None),
    )

    # CONCEPT:AU-ORCH.session.invoker-agent-handoff — enforce the invoker's least-privilege tool allow-list (if any).
    custom_tools, _scoped_toolsets = apply_tool_scope(
        ctx.state, custom_tools, collected_mcp_toolsets + skill_toolsets
    )

    agent = create_context_agent(
        model=specialist_model,
        permissions_kernel=ctx.deps.permissions_kernel,
        agent_identity=ctx.deps.agent_identity,
        permission_engine=ctx.deps.knowledge_engine,
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
    # Dynamic function tools must pass through the same fail-closed approval
    # policy as top-level agents and MCP toolsets. Without this step a local
    # specialist could invoke an ungoverned shell/file tool directly.
    from agent_utilities.security.tool_guard import apply_tool_guard_approvals

    apply_tool_guard_approvals(agent)

    # Tool-count telemetry for specialized steps
    _emit_tool_count_telemetry(ctx, prompt_name, len(custom_tools), mcp_tool_count)

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
            usage_limits=spawn_usage_limits(
                ctx.state
            ),  # CONCEPT:AU-ORCH.session.invoker-agent-handoff budget
        ) as stream:
            async for chunk in stream.stream_text(delta=True):
                emit_graph_event(
                    ctx.deps.event_queue,
                    "agent_node_delta",
                    content=chunk,
                    node=prompt_name,
                )
            res = await stream.get_output()
        ctx.state._update_usage(stream.usage)
        result_str = str(res)

        # RLM Large Result Summarization
        result_str = await _maybe_rlm_summarize(
            ctx, "Specialist", prompt_name, result_str
        )

        node_uid = f"{prompt_name}_{ctx.state.step_cursor}"
        ctx.state.results_registry[node_uid] = result_str

        logger.info(
            f"Specialized step '{prompt_name}': stored result ({len(result_str)} chars) "
            f"at registry key '{node_uid}'"
        )

        # Cache message history for potential re-dispatch
        await _cache_specialist_history(ctx, prompt_name, stream)

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

    original_env = _activate_domain_env(deps, domain)

    try:
        domain_mcp_toolsets = []
        for toolset in deps.mcp_toolsets:
            if toolset is None:
                continue
            filtered = filter_tools_by_tag(toolset, domain)
            domain_mcp_toolsets.append(filtered)

        sub_agent_target = deps.sub_agents.get(domain)

        if sub_agent_target:
            await _execute_domain_sub_agent(ctx, domain, deps, sub_agent_target)
        else:
            early_end = await _execute_domain_fallback_agent(
                ctx, domain, deps, domain_prompt
            )
            if early_end is not None:
                return early_end

    except Exception as e:
        logger.error(f"domain_step error for '{domain}': {e}")
        ctx.state.error = f"Domain failed: {e}"
        node_uid = f"{domain}_{ctx.state.step_cursor}"
        ctx.state.results_registry[node_uid] = f"Error: {e}"
        return "error_recovery"
    finally:
        _restore_domain_env(original_env)
    return None


def _activate_domain_env(deps: Any, domain: str) -> dict[str, str | None]:
    """Set each domain-tag env var True for the active domain, False otherwise.

    Returns the prior values so :func:`_restore_domain_env` can undo it.
    """
    original_env: dict[str, str | None] = {}
    for tag, env_var in deps.tag_env_vars.items():
        original_env[env_var] = setting(env_var)
        os.environ[env_var] = "True" if tag == domain else "False"
    return original_env


def _restore_domain_env(original_env: dict[str, str | None]) -> None:
    """Restore the env vars :func:`_activate_domain_env` overrode."""
    for env_var, value in original_env.items():
        if value is None:
            os.environ.pop(env_var, None)
        else:
            os.environ[env_var] = value


async def _execute_domain_sub_agent(
    ctx: StepContext, domain: str, deps: Any, sub_agent_target: Any
) -> None:
    """Delegate to an already-registered sub_agent (tag-spec dict, sub-graph tuple, or flat agent)."""
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
                permissions_kernel=ctx.deps.permissions_kernel,
                agent_identity=ctx.deps.agent_identity,
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
                ctx.state.query_parts if ctx.state.query_parts else ctx.state.query
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
    except Exception as e:
        logger.error(f"domain_step delegation error for '{domain}': {e}")
        node_uid = f"{domain}_{ctx.state.step_cursor}"
        ctx.state.results_registry[node_uid] = f"Delegation Error: {e}"


async def _execute_domain_fallback_agent(
    ctx: StepContext, domain: str, deps: Any, domain_prompt: str
) -> Any:
    """No registered sub_agent: build a generic per-domain agent and run it.

    Returns an ``End`` when a deferred-tool approval is required and there is no
    approval manager (the graph must terminate here); otherwise ``None`` and the
    caller continues normally.
    """
    sub_agent, run_input = _build_domain_fallback_agent(
        ctx, domain, deps, domain_prompt
    )
    output, early_end = await _run_domain_fallback_agent(
        ctx, domain, deps, sub_agent, run_input
    )
    if early_end is not None:
        return early_end

    result_str = str(output)
    node_uid = f"{domain}_{ctx.state.step_cursor}"
    ctx.state.results_registry[node_uid] = result_str
    emit_graph_event(deps.event_queue, "subagent_completed", domain=domain)
    return None


def _build_domain_fallback_agent(
    ctx: StepContext, domain: str, deps: Any, domain_prompt: str
) -> tuple[Any, Any]:
    """Build the generic per-domain agent and resolve its run_input."""
    query = ctx.state.query
    if ctx.state.validation_feedback:
        query = (
            f"{query}\n\n[SELF-CORRECTION FEEDBACK]: {ctx.state.validation_feedback}"
        )

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
        permissions_kernel=ctx.deps.permissions_kernel,
        agent_identity=ctx.deps.agent_identity,
    )

    emit_graph_event(deps.event_queue, "subagent_started", domain=domain)

    run_input = (
        ctx.state.query_parts
        if ctx.state.query_parts and query == ctx.state.query
        else query
    )
    return sub_agent, run_input


async def _run_domain_fallback_agent(
    ctx: StepContext, domain: str, deps: Any, sub_agent: Any, run_input: Any
) -> tuple[Any, Any]:
    """Run the fallback agent (approval-manager loop, or a bare timeout).

    Returns ``(output, early_end)``: ``early_end`` is an ``End`` when a
    deferred-tool approval fires with no approval manager configured (the graph
    must terminate); otherwise ``None`` and ``output`` is the agent's result.
    """
    # If an approval manager is available, use the transparent
    # approval loop that pauses the graph and waits for user
    # decisions. Without a manager, deferred requests terminate the graph.
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
        output = getattr(result, "output", None) or getattr(result, "data", result)
        return output, None

    result = await asyncio.wait_for(
        sub_agent.run(run_input),
        timeout=DEFAULT_GRAPH_TIMEOUT / 1000.0,
    )
    output = getattr(result, "output", None) or getattr(result, "data", result)

    if isinstance(output, DeferredToolRequests):
        ctx.state.human_approval_required = True
        node_uid = f"{domain}_{ctx.state.step_cursor}"
        ctx.state.results_registry[node_uid] = output
        emit_graph_event(
            deps.event_queue,
            event_type="approval_required",
            domain=domain,
            tool_calls=[
                (tc.model_dump() if hasattr(tc, "model_dump") else str(tc))
                for tc in (getattr(output, "calls", []) or [])
            ],
        )
        return output, End(output)

    return output, None


# implements core.execution.ExecutionEngine
def _normalize_manifest(manifest: Any) -> tuple[str, str]:
    """Normalize an ExecutionEngine ``manifest`` (a plain query string or a manifest object) to ``(query, manifest_id)``."""
    if isinstance(manifest, str):
        return manifest, ""
    query = getattr(manifest, "query", "") or ""
    manifest_id = getattr(manifest, "manifest_id", "") or ""
    return query, manifest_id


def _result_to_output(result: Any) -> tuple[str, bool]:
    """Extract ``(synthesis_output, success)`` from an :func:`execute_graph` result."""
    if not isinstance(result, dict):
        return str(result), True

    synthesis_output = str(
        result.get("output") or result.get("response") or result.get("result") or ""
    )
    success = True
    if "success" in result:
        success = bool(result["success"])
    elif "error" in result and result["error"]:
        success = False
    return synthesis_output, success


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

        query, manifest_id = _normalize_manifest(manifest)
        result = await execute_graph(self.graph, self.config, query)
        synthesis_output, success = _result_to_output(result)

        return ExecutionResult(
            manifest_id=manifest_id,
            synthesis_output=synthesis_output,
            success=success,
        )
