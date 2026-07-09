#!/usr/bin/python
from __future__ import annotations

"""Agent Factory Module.

This module provides factory functions for creating and configuring Pydantic AI
agents. It handles CLI argument parsing, MCP server initialization, toolset
registration, and system prompt construction.

CONCEPT:AU-OS.safety.doom-loop-detection Agent Creation
"""


import argparse
import logging
import os
from typing import Any, Literal, cast

import httpx
from pydantic_ai import Agent, DeferredToolRequests, ModelSettings
from pydantic_ai.mcp import MCPToolset

from agent_utilities.core.config import setting
from agent_utilities.mcp.toolset_factory import build_http_toolset


def load_mcp_servers(config_path: str) -> Any:
    """Load MCP servers from configuration using the centralized coordinator."""
    from agent_utilities.mcp_utilities import load_mcp_config

    return load_mcp_config(config_path)


from agent_utilities.base_utilities import (
    is_loopback_url,
    to_boolean,
)
from agent_utilities.capabilities import (
    CheckpointMiddleware,
    ContextLimitWarner,
    HooksCapability,
    InMemoryCheckpointStore,
    MementoCompaction,
    StuckLoopDetection,
    TeamCapability,
    ToolOutputEviction,
)
from agent_utilities.core.config import (
    DEFAULT_AGENT_NAME,
    DEFAULT_AGENT_SYSTEM_PROMPT,
    DEFAULT_CUSTOM_SKILLS_DIRECTORY,
    DEFAULT_DEBUG,
    DEFAULT_ENABLE_TERMINAL_UI,
    DEFAULT_ENABLE_WEB_LOGS,
    DEFAULT_EXTRA_BODY,
    DEFAULT_EXTRA_HEADERS,
    DEFAULT_FREQUENCY_PENALTY,
    DEFAULT_HOST,
    DEFAULT_LLM_API_KEY,
    DEFAULT_LLM_BASE_URL,
    DEFAULT_LLM_MODEL_ID,
    DEFAULT_LLM_PROVIDER,
    DEFAULT_LOGIT_BIAS,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MCP_CONFIG,
    DEFAULT_MCP_URL,
    DEFAULT_OTEL_EXPORTER_OTLP_ENDPOINT,
    DEFAULT_OTEL_EXPORTER_OTLP_HEADERS,
    DEFAULT_OTEL_EXPORTER_OTLP_PROTOCOL,
    DEFAULT_OTEL_EXPORTER_OTLP_PUBLIC_KEY,
    DEFAULT_OTEL_EXPORTER_OTLP_SECRET_KEY,
    DEFAULT_PARALLEL_TOOL_CALLS,
    DEFAULT_PORT,
    DEFAULT_PRESENCE_PENALTY,
    DEFAULT_SEED,
    DEFAULT_SSL_VERIFY,
    DEFAULT_STOP_SEQUENCES,
    DEFAULT_TEMPERATURE,
    DEFAULT_TIMEOUT,
    DEFAULT_TOOL_TIMEOUT,
    DEFAULT_TOP_P,
    DEFAULT_VALIDATION_MODE,
)
from agent_utilities.core.model_factory import create_model
from agent_utilities.core.workspace import (
    get_skills_path,
)
from agent_utilities.models import AgentDeps
from agent_utilities.security.tool_guard import apply_tool_guard_approvals
from agent_utilities.tools.tool_filtering import (
    filter_tools_by_tag,
    skill_matches_tags,
)

logger = logging.getLogger(__name__)


def create_agent_parser() -> argparse.ArgumentParser:
    """Create an ArgumentParser with standard agent CLI options.

    Returns:
        A configured ArgumentParser instance containing options for host, port,
        model provider, MCP settings, workspace paths, and observability.

    """
    parser = argparse.ArgumentParser(
        add_help=False, description=f"Run the {DEFAULT_AGENT_NAME} A2A + AG-UI Server"
    )
    parser.add_argument(
        "--host", default=DEFAULT_HOST, help="Host to bind the server to"
    )
    parser.add_argument(
        "--port", type=int, default=DEFAULT_PORT, help="Port to bind the server to"
    )
    parser.add_argument(
        "--debug",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_DEBUG,
        help="Enable/disable debug mode",
    )
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")

    parser.add_argument(
        "--provider",
        default=DEFAULT_LLM_PROVIDER,
        choices=[
            "openai",
            "anthropic",
            "google",
            "huggingface",
            "groq",
            "mistral",
            "ollama",
            "deepseek",
        ],
        help="LLM Provider",
    )
    parser.add_argument("--model-id", default=DEFAULT_LLM_MODEL_ID, help="LLM Model ID")
    parser.add_argument(
        "--base-url",
        default=DEFAULT_LLM_BASE_URL,
        help="LLM Base URL (for OpenAI compatible providers)",
    )
    parser.add_argument("--api-key", default=DEFAULT_LLM_API_KEY, help="LLM API Key")
    parser.add_argument("--mcp-url", default=DEFAULT_MCP_URL, help="MCP Server URL")
    parser.add_argument(
        "--mcp-config", default=DEFAULT_MCP_CONFIG, help="MCP Server Config"
    )
    parser.add_argument(
        "--custom-skills-directory",
        default=DEFAULT_CUSTOM_SKILLS_DIRECTORY,
        help="Directory containing additional custom agent skills",
    )
    parser.add_argument(
        "--workspace",
        help=(
            "Explicit path to the agent workspace directory "
            "(contains main_agent.json, mcp_config.json, etc.)"
        ),
    )
    parser.add_argument(
        "--web",
        action=argparse.BooleanOptionalAction,
        default=to_boolean(setting("ENABLE_WEB_UI", "False")),
        help="Enable/Disable Agent Web UI",
    )

    parser.add_argument(
        "--terminal",
        "--tui",
        dest="terminal",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_ENABLE_TERMINAL_UI,
        help="Enable/Disable Agent Terminal UI (TUI)",
    )
    parser.add_argument(
        "--web-logs",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_ENABLE_WEB_LOGS,
        help="Enable/Disable log streaming to TUI",
    )

    parser.add_argument(
        "--evolve",
        action="store_true",
        help="Run as an Evolutionary Vector (indefinitely runs SelfImprovementCycle)",
    )

    parser.add_argument(
        "--insecure",
        action="store_true",
        help="Disable SSL verification for LLM requests (Use with caution)",
    )

    parser.add_argument(
        "--otel",
        action=argparse.BooleanOptionalAction,
        default=to_boolean(setting("ENABLE_OTEL", "False")),
        help="Enable/Disable OpenTelemetry tracing",
    )
    parser.add_argument(
        "--otel-endpoint",
        default=DEFAULT_OTEL_EXPORTER_OTLP_ENDPOINT,
        help="OpenTelemetry OTLP endpoint",
    )
    parser.add_argument(
        "--otel-headers",
        default=DEFAULT_OTEL_EXPORTER_OTLP_HEADERS,
        help="OpenTelemetry OTLP headers",
    )
    parser.add_argument(
        "--otel-public-key",
        default=DEFAULT_OTEL_EXPORTER_OTLP_PUBLIC_KEY,
        help="OpenTelemetry OTLP public key (for Basic Auth)",
    )
    parser.add_argument(
        "--otel-secret-key",
        default=DEFAULT_OTEL_EXPORTER_OTLP_SECRET_KEY,
        help="OpenTelemetry OTLP secret key (for Basic Auth)",
    )
    parser.add_argument(
        "--otel-protocol",
        default=DEFAULT_OTEL_EXPORTER_OTLP_PROTOCOL,
        help="OpenTelemetry OTLP protocol",
    )

    parser.add_argument("--help", action="help", help="Show usage")
    return parser


def create_agent(
    provider: str | None = DEFAULT_LLM_PROVIDER,
    model_id: str | None = DEFAULT_LLM_MODEL_ID,
    base_url: str | None = DEFAULT_LLM_BASE_URL,
    api_key: str | None = DEFAULT_LLM_API_KEY,
    mcp_url: str | None = DEFAULT_MCP_URL,
    mcp_config: str | None = DEFAULT_MCP_CONFIG,
    mcp_toolsets: list[Any] | None = None,
    custom_skills_directory: str | None = DEFAULT_CUSTOM_SKILLS_DIRECTORY,
    ssl_verify: bool = DEFAULT_SSL_VERIFY,
    enable_skills: bool = True,
    enable_universal_tools: bool = True,
    name: str = DEFAULT_AGENT_NAME,
    system_prompt: str | None = DEFAULT_AGENT_SYSTEM_PROMPT,
    debug: bool | None = False,
    skill_types: list[str] | None = None,
    tool_tags: list[str] | None = None,
    capabilities: list[str] | None = None,
    graph_bundle: tuple[Any, ...] | None = None,
    output_type: Any | None = None,
    current_host: str | None = None,
    current_port: int | None = None,
    tool_guard_mode: str = "strict",
    isolate_mcp: bool = False,
    # Reliability & Capabilities
    stuck_loop_detection: bool = True,
    stuck_loop_max_repeated: int = 3,
    stuck_loop_action: Literal["warn", "error"] = "warn",
    hooks: list[Any] | None = None,
    auto_graph_trace: bool = True,
    context_warnings: bool = True,
    max_context_tokens: int | None = None,
    use_rlm: bool = False,
    rlm_config: Any | None = None,
    include_checkpoints: bool = False,
    checkpoint_store: Any | None = None,
    checkpoint_frequency: Literal[
        "every_tool", "every_turn", "manual_only"
    ] = "every_tool",
    include_teams: bool = False,
    output_style: str | None = None,
    output_eviction: bool = True,
    eviction_threshold_chars: int = 80_000,
    memento_compaction: bool = True,
    # v2 synergy (opt-in — both are expensive/behavior-changing):
    #   thinking_effort: native provider extended thinking ('low'|'medium'|'high');
    #     falls back to the AGENT_THINKING_EFFORT config setting. None/"" = off.
    #   defer_tool_loading: keep agent-local toolsets out of the prompt as a compact
    #     catalog, loaded on demand by the model.
    thinking_effort: str | None = None,
    defer_tool_loading: bool = False,
) -> tuple[Agent[Any, Any], list[Any]]:
    """Initialize a Pydantic AI Agent with requested capabilities.

    CONCEPT:AU-OS.safety.doom-loop-detection Agent Creation

    Args:
        provider: LLM provider name.
        model_id: Specific model identifier.
        base_url: Optional override for the LLM base URL.
        api_key: Optional API key for the provider.
        mcp_url: Optional URL of a single MCP server.
        mcp_config: Optional path to an MCP configuration file.
        mcp_toolsets: Optional list of pre-initialized MCP toolsets.
        custom_skills_directory: Optional path to additional skills.
        ssl_verify: Whether to verify SSL certificates.
        enable_skills: Whether to load skills from the workspace.
        enable_universal_tools: Whether to register universal agent tools.
        name: Name of the agent.
        system_prompt: Optional custom system prompt.
        debug: Whether to enable debug mode.
        skill_types: Optional list of skill types to load ('universal', 'graphs', 'tdd-methodology', 'manual_testing', 'walkthroughs').
        tool_tags: Optional tags to filter tools by.
        graph_bundle: Optional state machine bundle for the orchestrator.
        output_type: Optional Pydantic-compatible output type schema.
        current_host: Hostname of the current process for loopback detection.
        current_port: Port of the current process for loopback detection.
        tool_guard_mode: Mode for tool approval ('on', 'off', 'dry-run').
        isolate_mcp: Whether to isolate MCP tools from the main agent.

    Returns:
        A tuple containing the initialized Agent and a list of all initialized toolsets.

    """

    # When the database-traversal tools are enabled for this agent, default RLM
    # on so a db_query returning millions of rows is recursively processed by the
    # rlm_large_output_hook instead of overflowing the context window (CONCEPT:
    # ORCH-1.1 + ECO-4.33). An explicit ``use_rlm=`` always wins.
    if not use_rlm and to_boolean(setting("DB_TOOLS", "False")):
        use_rlm = True

    # Centralized Knowledge Graph Coordination
    import sys

    is_testing = (
        setting("AGENT_UTILITIES_TESTING") == "true"
        or "pytest" in sys.modules
        or setting("PYTEST_CURRENT_TEST") is not None
    )
    if not DEFAULT_VALIDATION_MODE and not is_testing:
        try:
            from agent_utilities.mcp.kg_coordinator import KGCoordinator

            kg_host = setting("KG_SERVER_HOST", "127.0.0.1")
            kg_port = setting("KG_SERVER_PORT", 8100)
            KGCoordinator.get_kg_client(host=kg_host, port=kg_port)
        except Exception as e:
            logger.debug(f"Auto-coordinating KG failed in create_agent: {e}")

    agent_toolsets: list[Any] = []
    initialized_mcp_toolsets: list[Any] = []

    if mcp_url:
        if DEFAULT_VALIDATION_MODE:
            logger.info(f"VALIDATION_MODE: Skipping MCP connection to {mcp_url}")
        elif is_loopback_url(mcp_url, current_host, current_port):
            logger.warning(
                f"Loopback Guard: Skipping self-referential MCP connection to {mcp_url}"
            )
        else:
            try:
                server = build_http_toolset(
                    mcp_url, verify=ssl_verify, timeout=DEFAULT_TIMEOUT
                )
                initialized_mcp_toolsets.append(
                    filter_tools_by_tag(server, tool_tags) if tool_tags else server
                )
                if not isolate_mcp:
                    agent_toolsets.append(initialized_mcp_toolsets[-1])
                logger.info(f"Connected to MCP Server: {mcp_url}")
            except Exception as e:
                logger.error(f"Failed to connect to MCP Server {mcp_url}: {e}")

    if mcp_config:
        if DEFAULT_VALIDATION_MODE:
            logger.info(
                f"VALIDATION_MODE: Skipping MCP config loading from {mcp_config}"
            )
        else:
            try:
                from agent_utilities.core.workspace import resolve_mcp_config_path

                mcp_path = resolve_mcp_config_path(mcp_config)
                if mcp_path:
                    mcp_config = str(mcp_path)
                    logger.info(f"Resolved MCP config: {mcp_config}")

                mcp_toolset = load_mcp_servers(mcp_config)
                for server in mcp_toolset:
                    if hasattr(server, "http_client") and not getattr(
                        server, "http_client", None
                    ):
                        server.http_client = httpx.AsyncClient(
                            verify=ssl_verify, timeout=DEFAULT_TIMEOUT
                        )

                if tool_tags:
                    mcp_toolset = [
                        filter_tools_by_tag(s, tool_tags) for s in mcp_toolset
                    ]

                initialized_mcp_toolsets.extend(mcp_toolset)
                if not isolate_mcp:
                    agent_toolsets.extend(mcp_toolset)
                logger.info(f"Connected to MCP Config: {mcp_config}")
            except Exception as e:
                logger.warning(f"Could not load MCP config {mcp_config}: {e}")

    if mcp_toolsets:
        if DEFAULT_VALIDATION_MODE:
            logger.info("VALIDATION_MODE: Skipping external mcp_toolsets connection")
        else:
            for server in mcp_toolsets:
                if server is None:
                    continue
                if hasattr(server, "http_client") and not getattr(
                    server, "http_client", None
                ):
                    server.http_client = httpx.AsyncClient(
                        verify=ssl_verify, timeout=DEFAULT_TIMEOUT
                    )
            for server in mcp_toolsets:
                if server is None:
                    continue

                ts = None
                if type(server).__name__ == "FastMCP":
                    # v2: a FastMCP server instance is wrapped directly by MCPToolset
                    ts = MCPToolset(server)
                else:
                    ts = server

                initialized_mcp_toolsets.append(ts)
                if not isolate_mcp:
                    agent_toolsets.append(ts)

    model = create_model(
        provider=provider,
        model_id=model_id,
        base_url=base_url,
        api_key=api_key,
        ssl_verify=ssl_verify,
        timeout=DEFAULT_TIMEOUT,
    )

    settings = ModelSettings(
        max_tokens=DEFAULT_MAX_TOKENS,
        temperature=DEFAULT_TEMPERATURE,
        top_p=DEFAULT_TOP_P,
        timeout=DEFAULT_TIMEOUT,
        parallel_tool_calls=DEFAULT_PARALLEL_TOOL_CALLS,
        seed=DEFAULT_SEED,
        presence_penalty=DEFAULT_PRESENCE_PENALTY,
        frequency_penalty=DEFAULT_FREQUENCY_PENALTY,
        logit_bias=DEFAULT_LOGIT_BIAS,
        stop_sequences=DEFAULT_STOP_SEQUENCES,
        extra_headers=DEFAULT_EXTRA_HEADERS,
        extra_body=DEFAULT_EXTRA_BODY,
    )

    from pydantic_ai_skills import SkillsToolset

    if enable_skills and not DEFAULT_VALIDATION_MODE:
        skill_dirs = []
        _skill_types = (
            skill_types
            if skill_types is not None
            else [
                "universal",
                "graphs",
                "tdd-methodology",
                "manual_testing",
                "walkthroughs",
            ]
        )

        if skills_path := get_skills_path():
            skill_dirs.extend(skills_path)

        if "universal" in _skill_types:
            from universal_skills.skill_utilities import get_universal_skills_path

            skill_dirs.extend(get_universal_skills_path())

        if "graphs" in _skill_types:
            try:
                from skill_graphs.skill_graph_utilities import get_skill_graphs_path

                skill_dirs.extend(get_skill_graphs_path(default_enabled=True))
            except ImportError:
                pass

        if tool_tags:
            skill_dirs = [d for d in skill_dirs if skill_matches_tags(d, tool_tags)]

        if custom_skills_directory:
            if isinstance(custom_skills_directory, list | tuple):
                for d in custom_skills_directory:
                    if d and os.path.exists(d):
                        skill_dirs.append(str(d))
                        logger.info(f"Loaded Custom Skills at {d}")
            elif os.path.exists(custom_skills_directory):
                logger.debug(f"Loading custom skills {custom_skills_directory}")
                skill_dirs.append(str(custom_skills_directory))
                logger.info(f"Loaded Custom Skills at {custom_skills_directory}")

        # Always load the XDG skills drop-in (~/.local/share/agent-utilities/skills,
        # or AGENT_UTILITIES_SKILLS_DIR) — where skill-installer puts user skills — so
        # dropping a SKILL.md there gives the agent that skill with no extra config.
        try:
            from agent_utilities.core.paths import skills_dir

            xdg_skills = str(skills_dir())
            if os.path.isdir(xdg_skills) and xdg_skills not in skill_dirs:
                skill_dirs.append(xdg_skills)
                logger.info(f"Loaded XDG skills drop-in at {xdg_skills}")
        except Exception as exc:  # noqa: BLE001 — best-effort, never block agent creation
            logger.debug(f"XDG skills dir unavailable: {exc}")

        # CONCEPT:AU-ORCH.dispatch.warm-skills-share — warm-share the SkillsToolset across the fan-out cohort: the
        # directory scan + SKILL.md parse is deterministic per skill-dir set, so build it once
        # and reuse it (pydantic-ai toolsets attach to many agents). Falls back to a fresh build.
        from agent_utilities.agent.warm_skills import get_or_build_skills_toolset

        skills = get_or_build_skills_toolset(
            skill_dirs, lambda: SkillsToolset(directories=skill_dirs)
        )
        agent_toolsets.append(skills)
        logger.info(f"Loaded {len(skill_dirs)} Skills")

    if system_prompt is None:
        logger.info(
            "No system_prompt provided to create_agent. Building from workspace..."
        )
        from agent_utilities.prompting.builder import build_system_prompt_from_workspace

        system_prompt_str = build_system_prompt_from_workspace()
    else:
        logger.debug(f"Custom Agent System Prompt provided: {system_prompt[:100]}...")
        system_prompt_str = system_prompt

    # CONCEPT:AU-ECO.bus.agent-bus-awareness — weave AgentBus awareness into EVERY agent's prompt at the one
    # choke point, so the orchestrator and every spawned swarm/sub-agent natively know they
    # can coordinate with peers (the bus_* tools are registered just below when universal
    # tools are on). Native-by-default: not a separate persona, just how agents work together.
    if enable_universal_tools:
        from agent_utilities.messaging.bus import bus_capability_prompt

        if "AgentBus" not in system_prompt_str:
            system_prompt_str = f"{system_prompt_str}\n\n{bus_capability_prompt()}"

    # Initialize Capabilities
    agent_capabilities: list[Any] = []

    if stuck_loop_detection:
        agent_capabilities.append(
            StuckLoopDetection(
                max_repeated=stuck_loop_max_repeated,
                action=stuck_loop_action,
            )
        )

    if context_warnings:
        agent_capabilities.append(
            ContextLimitWarner(
                max_tokens=max_context_tokens,
            )
        )

    if output_eviction:
        agent_capabilities.append(
            ToolOutputEviction(
                threshold_chars=eviction_threshold_chars,
            )
        )

    # CONCEPT:AU-KG.memory.memento-compress-evict — live Memento block-compress-evict sawtooth (default ON). Where
    # ContextLimitWarner only *warns* and ToolOutputEviction only handles oversized tool results,
    # this evicts old completed reasoning blocks from the message history, replacing each with a
    # dense memento, when the running context exceeds budget.
    if memento_compaction:
        agent_capabilities.append(
            MementoCompaction(
                max_tokens=max_context_tokens,
            )
        )

    if include_checkpoints:
        store = checkpoint_store or InMemoryCheckpointStore()
        agent_capabilities.append(
            CheckpointMiddleware(
                store=store,
                frequency=checkpoint_frequency,
            )
        )

    if include_teams:
        agent_capabilities.append(TeamCapability())

    # CONCEPT:AU-ORCH.routing.sampling-profile-selection (v2 synergy) — native provider-side extended thinking. Opt-in
    # because reasoning is expensive: enabled via the thinking_effort arg or the
    # AGENT_THINKING_EFFORT config setting. It runs natively where the provider supports
    # reasoning and no-ops elsewhere, and composes with the per-call sampling profile
    # (which still threads vLLM enable_thinking via extra_body) attached below.
    _ThinkingEffort = Literal["minimal", "low", "medium", "high", "xhigh"]
    _VALID_THINKING_EFFORTS: tuple[_ThinkingEffort, ...] = (
        "minimal",
        "low",
        "medium",
        "high",
        "xhigh",
    )
    _thinking_effort = str(thinking_effort or setting("AGENT_THINKING_EFFORT", ""))
    if _thinking_effort in _VALID_THINKING_EFFORTS:
        from pydantic_ai.capabilities import Thinking

        agent_capabilities.append(
            Thinking(effort=cast(_ThinkingEffort, _thinking_effort))
        )
    elif _thinking_effort:
        logger.warning(
            "Ignoring invalid thinking_effort %r (must be one of %s)",
            _thinking_effort,
            _VALID_THINKING_EFFORTS,
        )

    # Unified Hooks
    all_hooks = hooks or []
    if auto_graph_trace:
        # HooksCapability handles auto_graph_trace in its before/after tool hooks
        pass

    agent_capabilities.append(
        HooksCapability(
            hooks=all_hooks,
            auto_graph_trace=auto_graph_trace,
        )
    )

    if use_rlm or (skill_types and "recursive_reasoner" in skill_types):
        try:
            from agent_utilities.rlm.hook import rlm_large_output_hook

            all_hooks.append(rlm_large_output_hook)
        except ImportError:
            pass

    # CONCEPT (v2 synergy) — on-demand tool loading: keep agent-local toolsets out of
    # the prompt as a one-line catalog until the model loads them, cutting prompt bloat
    # for tool-heavy agents. Opt-in (default off) because it changes tool visibility.
    # Orthogonal to the cross-process multiplexer (find_tools/load_tools).
    if defer_tool_loading and agent_toolsets:
        agent_toolsets = [
            ts.defer_loading() if hasattr(ts, "defer_loading") else ts
            for ts in agent_toolsets
        ]

    agent = Agent(
        model=model,
        model_settings=settings,
        name=name,
        output_type=(
            [str, DeferredToolRequests] if output_type is None else output_type
        ),
        toolsets=agent_toolsets,
        tool_timeout=DEFAULT_TOOL_TIMEOUT,
        deps_type=AgentDeps,
        capabilities=agent_capabilities,
        # pydantic-ai v2 default; set explicitly. Function tools requested
        # alongside an output/deferred tool now run — side-effecting tools are
        # kept safe by the tool_guard ApprovalRequiredToolset (they become
        # DeferredToolRequests and never auto-run before human approval).
        end_strategy="graceful",
    )

    if output_style:
        from agent_utilities.tools.style_tools import BUILTIN_STYLES

        style_instr = BUILTIN_STYLES.get(output_style.lower(), "")
        if style_instr:

            @agent.instructions
            def inject_style_prompt() -> str:
                return f"\n\nOUTPUT STYLE: {style_instr}"

    if use_rlm or (skill_types and "recursive_reasoner" in skill_types):
        from pydantic_ai import Tool

        from agent_utilities.rlm.specialist import recursive_reasoner_tool

        agent._function_toolset.add_tool(
            Tool(
                recursive_reasoner_tool,
                takes_ctx=True,
                description="Recursive reasoning specialist for massive contexts",
                name="recursive_reasoner_tool",
            )
        )
        logger.info("Enabled RLM Recursive Reasoner Tool")

    @agent.instructions
    def inject_system_prompt() -> str:
        """Instruction handler to inject the dynamically built system prompt."""
        return system_prompt_str

    if enable_universal_tools:
        from agent_utilities.tools.tool_registry import register_agent_tools

        register_agent_tools(agent, graph_bundle=graph_bundle)

    # CONCEPT:AU-ECO.toolkit.capability-tool-binding — bind tools from declared capability intents (rename/prefix-proof),
    # never from hard-coded names. Blueprints/callers pass `capabilities`; the KG resolves the
    # long tail when an engine is available.
    if capabilities:
        from agent_utilities.agent.capability_resolver import register_capability_tools
        from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine

        register_capability_tools(
            agent, list(capabilities), kg=IntelligenceGraphEngine.get_active()
        )

    if tool_guard_mode != "off":
        apply_tool_guard_approvals(agent)

    # CONCEPT:AU-ORCH.routing.sampling-profile-selection — task-aware per-call sampling. The static `settings` above is
    # the base floor; this wraps the agent's run path so each call threads a profile
    # (deterministic low-temp for code/extraction, exploratory for brainstorm) as a
    # per-call model_settings override, unless a caller passes one explicitly.
    from agent_utilities.agent.sampling_profile import attach_profile_resolver

    attach_profile_resolver(agent, settings, system_prompt=system_prompt_str)

    return agent, initialized_mcp_toolsets
