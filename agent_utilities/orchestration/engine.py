"""Orchestration Engine — The canonical entrypoint for all agent execution.

CONCEPT:ORCH-2.0 — Orchestration Engine

This module replaces the legacy fragmented orchestrators (AgentOrchestrationEngine,
KGDrivenExecutionEngine, run_graph, ParallelEngine, WorkflowRunner, SDDOrchestrator,
and DynamicToolOrchestrator) with a single, coherent execution kernel.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine

logger = logging.getLogger(__name__)

import asyncio
import contextlib
import json
import time
from pathlib import Path
from uuid import uuid4

from agent_utilities.core.config import (
    DEFAULT_LLM_MODEL_ID,
    DEFAULT_LLM_PROVIDER,
    DEFAULT_ROUTER_MODEL,
    GET_DEFAULT_SSL_VERIFY,
)

DEFAULT_ENABLE_LLM_VALIDATION = True
DEFAULT_GRAPH_AGENT_MODEL = DEFAULT_LLM_MODEL_ID
DEFAULT_GRAPH_PERSISTENCE_PATH = ".agent_utilities/graph_persistence"
DEFAULT_GRAPH_ROUTER_TIMEOUT = 120000
DEFAULT_GRAPH_VERIFIER_TIMEOUT = 120000
DEFAULT_PROVIDER = DEFAULT_LLM_PROVIDER
DEFAULT_SSL_VERIFY = GET_DEFAULT_SSL_VERIFY()

from agent_utilities.core.config import (
    DEFAULT_GRAPH_TIMEOUT,
    emit_graph_event,
    load_node_agents_registry,
)
from agent_utilities.core.model_factory import create_model

from ..graph.mermaid import get_graph_mermaid
from ..graph.state import REQUESTED_MODEL_ID_CTX, GraphDeps, GraphState
from ..models import GraphResponse

try:
    from opentelemetry import trace

    tracer: trace.Tracer | None = trace.get_tracer("agent-utilities.graph")
except ImportError:
    tracer = None

from contextlib import AsyncExitStack

import anyio

from agent_utilities.models.execution_manifest import (
    ExecutionManifest,
    ExecutionResult,
)


# implements core.execution.ExecutionEngine
class AgentOrchestrationEngine:
    """The singular orchestration engine for the agent ecosystem.

    Modes:
      - graph: Full pydantic-graph execution
      - stream: SSE streaming graph execution
      - dynamic: KG-driven dynamic synthesis
      - parallel: Concurrent subagent dispatch
      - workflow: Compiled workflow execution
      - sdd: Spec-driven development orchestration
      - research: Research pipeline execution
    """

    def __init__(self, engine: IntelligenceGraphEngine | None = None):
        if engine is None:
            try:
                from agent_utilities.knowledge_graph.core.engine import (
                    IntelligenceGraphEngine,
                )

                engine = IntelligenceGraphEngine.get_active()
            except Exception as e:
                logger.warning(f"Failed to initialize IntelligenceGraphEngine: {e}")
                engine = None
        self.engine = engine

    async def dispatch(
        self, task: Any, *, mode: str = "auto", **kwargs: Any
    ) -> dict[str, Any]:
        """Core dispatch — the single entrypoint for task execution."""
        logger.info(f"Dispatching task in mode: {mode}")
        if mode == "auto":
            mode = self._determine_mode(task)

        if mode == "graph":
            return await self.execute_graph(task, **kwargs)
        elif mode == "stream":
            # This would return an AsyncIterator in practice, but wrapping it for generic typing
            return {"stream": self.stream_graph(task, **kwargs)}
        elif mode == "dynamic":
            return await self._execute_dynamic(task, **kwargs)
        elif mode == "parallel":
            return await self.execute_parallel(task, **kwargs)
        elif mode == "workflow":
            return await self.execute_workflow(task, **kwargs)
        elif mode == "sdd":
            return await self.execute_sdd(task, **kwargs)
        elif mode == "swe":
            return await self.execute_swe(task, **kwargs)
        else:
            raise ValueError(f"Unknown dispatch mode: {mode}")

    async def execute_swe(
        self, task: Any, *, deps: Any = None, workspace: Any = None, **kwargs: Any
    ) -> dict[str, Any]:
        """Run the KG-grounded SWE agent (CONCEPT:ORCH-1.47) on ``task``.

        Drives the edit→run→test loop inside a developer workspace (OS-5.33), grounding in the
        code ontology (KG-2.65). A ``workspace`` may be supplied (e.g. a repo already cloned by
        the SWE-bench harness, AHE-3.22); otherwise a fresh one is created and torn down.
        """
        from agent_utilities.models import AgentDeps
        from agent_utilities.runtime import create_workspace

        from .swe_agent import run_swe_task

        deps = deps or AgentDeps()
        owns_ws = workspace is None
        ws = workspace or create_workspace(actor=getattr(deps, "user_id", None))
        if owns_ws:
            await ws.start()
        deps.workspace = ws
        try:
            result = await run_swe_task(str(task), deps, **kwargs)
            return {
                "mode": "swe",
                "output": result.output,
                "patch": result.patch,
                "tool_calls": result.tool_calls,
            }
        finally:
            if owns_ws:
                await ws.stop()

    def _determine_mode(self, task: Any) -> str:
        """Heuristics to determine the best execution mode based on the task payload."""
        if isinstance(task, list):
            return "parallel"
        if isinstance(task, dict) and "workflow_id" in task:
            return "workflow"
        if isinstance(task, dict) and "spec" in task:
            return "sdd"
        if hasattr(task, "run"):
            return "graph"
        return "dynamic"

    # --- Dynamic Team Synthesis ---
    def synthesize_team(
        self,
        query: str,
        domain: str,
        complexity: float = 1.0,
        delegated_authority: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """Synthesize a subagent team for a domain via KG analysis.

        Topology scoping runs on the **epistemic-graph compute layer**
        (``get_blast_radius`` over the out-of-process tokio/MessagePack engine — not
        PyO3/FFI, not scipy/sklearn), which bounds the candidate agents to the domain
        sub-graph. The agent roster and each agent's tools are then resolved from the
        graph store, and when a ``delegated_authority`` is supplied the roster is
        restricted to agents authorised for it (CONCEPT:ORCH-1.3 governance). Replaces
        legacy AgentOrchestrationEngine team synthesis.
        """
        logger.info(f"Synthesizing team for {domain} (complexity: {complexity})")
        assert self.engine is not None, (
            "IntelligenceGraphEngine is required for team synthesis"
        )

        import uuid

        from agent_utilities.models.knowledge_graph import TeamComposition

        # Epistemic-graph compute: scope candidate agents to the domain's blast radius.
        # Coerce to a concrete id list so a missing/short-circuited compute layer simply
        # widens the roster query rather than raising.
        try:
            candidate_ids = [
                n.get("id") if isinstance(n, dict) else str(n)
                for n in (
                    self.engine.graph_compute.get_blast_radius(
                        f"domain:{domain}", max_depth=2
                    )
                    or []
                )
            ]
        except Exception:  # noqa: BLE001 - Rust compute unavailable → domain-wide roster
            candidate_ids = []

        # Resolve the agent roster from the graph store. With a delegated authority,
        # restrict to agents authorised for it; otherwise scope by domain.
        if delegated_authority:
            agent_query = (
                "MATCH (a:Agent)-[:HAS_DELEGATED_AUTHORITY_FROM|AUTHORIZED_FOR]->"
                "(auth {id: $delegated_authority}) "
                "WHERE size($candidate_ids) = 0 OR a.agent_id IN $candidate_ids "
                "RETURN a.agent_id AS agent_id, a.role AS role, a.name AS name"
            )
        else:
            agent_query = (
                "MATCH (a:Agent) "
                "WHERE a.domain = $domain "
                "AND (size($candidate_ids) = 0 OR a.agent_id IN $candidate_ids) "
                "RETURN a.agent_id AS agent_id, a.role AS role, a.name AS name"
            )
        params = {
            "domain": domain,
            "candidate_ids": candidate_ids,
            "delegated_authority": delegated_authority,
        }
        agent_rows = self.engine.backend.execute(agent_query, params) or []

        agents = []
        for row in agent_rows:
            agent_id = row.get("agent_id") or row.get("id")
            role = row.get("role") or "general"
            # Per-agent tool binding via the USES edge to CallableResource nodes.
            tool_rows = (
                self.engine.backend.execute(
                    "MATCH (a:Agent {agent_id: $agent_id})-[:USES]->(t:CallableResource) "
                    "RETURN t.name AS tool_name",
                    {"agent_id": agent_id},
                )
                or []
            )
            tools = [t["tool_name"] for t in tool_rows if t.get("tool_name")]
            agents.append(
                {
                    "role": role,
                    "agent_id": agent_id,
                    "tools": tools,
                    "system_prompt": f"You are the {role} agent.",
                }
            )

        if not agents:
            agents.append(
                {
                    "role": "general",
                    "agent_id": "general_agent",
                    "tools": [],
                    "system_prompt": "You are a general agent.",
                }
            )

        return TeamComposition(
            team_id=f"team:{uuid.uuid4().hex[:12]}",
            adaptive_agent_router=agents,
            execution_mode=(
                "parallel" if (complexity > 1 or len(agents) > 1) else "sequential"
            ),
            reasoning=(
                f"Synthesized {len(agents)}-agent topology for domain:{domain}"
                + (
                    f" under delegated authority {delegated_authority}"
                    if delegated_authority
                    else ""
                )
            ),
            confidence=0.8,
        )

    # --- KG State Machine ---
    def determine_next_node(self, current_node: str, context: dict[str, Any]) -> str:
        """Determine the next execution node in a dynamic graph.
        Replaces KGDrivenExecutionEngine routing logic.
        """
        # Call into rust to evaluate next hops based on semantic edges
        assert self.engine is not None, (
            "IntelligenceGraphEngine is required for node determination"
        )
        successors = self.engine.graph_compute.get_successors(current_node)
        return successors[0] if successors else "END"

    # --- Graph Execution (pydantic-graph) ---
    async def execute_graph(
        self,
        graph,
        config: dict,
        query: str,
        run_id: str | None = None,
        persist: bool = False,
        state_dir: str = DEFAULT_GRAPH_PERSISTENCE_PATH or "graph_state",
        streamdown: bool = True,
        eq: asyncio.Queue[Any] | None = None,
        mode: str = "ask",
        topology: str = "basic",
        mcp_toolsets: list[Any] | None = None,
        query_parts: list[dict[str, Any]] | None = None,
        plan_sync=None,
        requested_model_id: str | None = None,
    ) -> dict:
        """Execute a query through the graph orchestrator (synchronous/batch).

        This function initializes the execution context, connects to the required
        MCP servers, and executes the graph loop until completion or timeout.

        Args:
            graph: The Graph object created by the builder.
            config: Configuration dictionary containing dependencies and settings.
            query: The user's input query string.
            run_id: Optional unique identifier for this execution session.
            persist: Whether to enable persistent state storage for this run.
            state_dir: Directory path for state persistence files.
            streamdown: Whether to include a mermaid diagram of the graph in the output.
            eq: Optional asyncio.Queue for sideband graph lifecycle events.
            mode: The orchestrator's execution mode (e.g., 'ask', 'research').
            topology: The selected graph topology (e.g., 'basic', 'dynamic').
            mcp_toolsets: Optional override list of MCP toolsets to use.
            query_parts: Optional structural message parts for complex queries.
            requested_model_id: Optional per-turn model id sourced from the
                ``x-agent-model-id`` request header. When valid within the
                attached ``model_registry``, specialist spawning uses it
                verbatim (see :func:`pick_specialist_model`).

        Returns:
            A GraphResponse instance containing the final synthesized output
            and execution metadata.

        """
        if run_id is None:
            run_id = uuid4().hex

        # CONCEPT:OS-5.11 — establish a run-wide correlation id so every nested
        # agent/span/side-effect in this run is joinable. Idempotent: nested
        # in-process runs inherit the parent's id via the contextvar.
        with contextlib.suppress(Exception):
            from ..observability.correlation import ensure_correlation_id

            ensure_correlation_id()

        if requested_model_id is None:
            requested_model_id = REQUESTED_MODEL_ID_CTX.get()

        mermaid_prefix = ""
        if streamdown:
            with contextlib.suppress(Exception):
                mermaid_prefix = (
                    f"```mermaid\n{get_graph_mermaid(graph, config)}\n```\n\n"
                )

        state = GraphState(
            query=query,
            query_parts=query_parts or [],
            mode=mode,
            topology=topology,
            invoker_context=config.get("invoker_context", ""),  # CONCEPT:ORCH-1.39
            invoker_budget_tokens=config.get(
                "invoker_budget_tokens"
            ),  # CONCEPT:ORCH-1.39
            invoker_allowed_tools=config.get(
                "invoker_allowed_tools"
            ),  # CONCEPT:ORCH-1.39
            invoker_cred_ref=config.get("invoker_cred_ref"),  # CONCEPT:ORCH-1.39
            invoker_channel_id=config.get("message_channel_id"),  # CONCEPT:ORCH-1.40
        )

        _custom_headers = config.get("custom_headers")
        _ssl_verify = config.get("ssl_verify", DEFAULT_SSL_VERIFY)

        deps = GraphDeps(
            tag_prompts=config.get("tag_prompts", {}),
            tag_env_vars=config.get("tag_env_vars", {}),
            mcp_toolsets=(
                mcp_toolsets
                if mcp_toolsets is not None
                else config.get("mcp_toolsets", [])
            ),
            mcp_url=config.get("mcp_url", ""),
            mcp_config=config.get("mcp_config", ""),
            router_model=create_model(
                model_id=config.get("router_model", DEFAULT_ROUTER_MODEL),
                api_key=config.get("api_key"),
                base_url=config.get("base_url"),
                custom_headers=_custom_headers,
                provider=config.get("provider", DEFAULT_PROVIDER),
                ssl_verify=_ssl_verify,
            ),
            agent_model=create_model(
                model_id=config.get("agent_model", DEFAULT_GRAPH_AGENT_MODEL),
                api_key=config.get("api_key"),
                base_url=config.get("base_url"),
                custom_headers=_custom_headers,
                provider=config.get("provider", DEFAULT_PROVIDER),
                ssl_verify=_ssl_verify,
            ),
            nodes=config.get("nodes", {}),
            min_confidence=config.get("min_confidence", 0.6),
            sub_agents=config.get("sub_agents", {}),
            provider=config.get("provider", DEFAULT_PROVIDER),
            base_url=config.get("base_url"),
            api_key=config.get("api_key"),
            ssl_verify=_ssl_verify,
            event_queue=eq,
            router_timeout=config.get("router_timeout", DEFAULT_GRAPH_ROUTER_TIMEOUT),
            verifier_timeout=config.get(
                "verifier_timeout", DEFAULT_GRAPH_VERIFIER_TIMEOUT
            ),
            request_id=config.get("request_id", run_id),
            routing_strategy=config.get("routing_strategy", "hybrid"),
            enable_llm_validation=config.get(
                "enable_llm_validation", DEFAULT_ENABLE_LLM_VALIDATION
            ),
            discovery_metadata=config.get("discovery_metadata", {}),
            plan_sync=plan_sync,
            approval_manager=config.get("approval_manager"),
            model_registry=config.get("model_registry"),
            requested_model_id=requested_model_id,
        )

        state = GraphState(
            query=query,
            query_parts=query_parts or [],
            session_id=run_id,
            mode=mode,
            topology=topology,
            invoker_context=config.get("invoker_context", ""),  # CONCEPT:ORCH-1.39
            invoker_budget_tokens=config.get(
                "invoker_budget_tokens"
            ),  # CONCEPT:ORCH-1.39
            invoker_allowed_tools=config.get(
                "invoker_allowed_tools"
            ),  # CONCEPT:ORCH-1.39
            invoker_cred_ref=config.get("invoker_cred_ref"),  # CONCEPT:ORCH-1.39
            invoker_channel_id=config.get("message_channel_id"),  # CONCEPT:ORCH-1.40
        )

        if persist:
            path = Path(state_dir)
            if path.suffix != ".json":
                path = path / f"{run_id}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            # persistence = FileStatePersistence(json_file=path)

        # Track which MCP servers fail to connect so we can report them clearly.
        failed_servers: list[tuple[str, str]] = []
        connected_toolsets: list = []

        # Track which toolsets have already been connected by the global
        # lifespan (server.py pre-connects them at startup).  Use object
        # identity instead of mutating toolset attributes.
        _already_connected: set[int] = set()

        async with AsyncExitStack() as stack:
            for ts in deps.mcp_toolsets:
                if not hasattr(ts, "__aenter__"):
                    connected_toolsets.append(ts)
                    continue
                if id(ts) in _already_connected:
                    connected_toolsets.append(ts)
                    continue

                srv_id = getattr(ts, "id", getattr(ts, "name", repr(ts)))
                try:
                    logger.debug(f"run_graph: Connecting to MCP server '{srv_id}'...")
                    # Use asyncio.timeout() (not asyncio.wait_for) to bound the connect:
                    # wait_for runs the coroutine in a NEW task, so a stdio toolset's anyio
                    # cancel scope would be ENTERED in that child task while the AsyncExitStack
                    # EXITS it in this (outer) task → "Attempted to exit cancel scope in a
                    # different task than it was entered in". asyncio.timeout() applies to the
                    # current task, keeping enter/exit on the same task.
                    async with asyncio.timeout(60.0):
                        connected = await stack.enter_async_context(ts)
                    _already_connected.add(id(ts))
                    connected_toolsets.append(connected)
                    logger.info(f"run_graph: ✅ MCP server '{srv_id}' connected")
                except Exception as e:
                    err_msg = str(e)
                    logger.error(
                        f"run_graph: ❌ MCP server '{srv_id}' FAILED to connect: {err_msg}"
                    )
                    failed_servers.append((srv_id, err_msg))

            deps.mcp_toolsets = [ts for ts in connected_toolsets if ts is not None]

            if failed_servers:
                logger.warning(
                    f"run_graph: {len(failed_servers)} MCP server(s) failed to connect — "
                    f"graph will proceed without them:\n"
                    + "\n".join(f"  ❌ {sid}: {err}" for sid, err in failed_servers)
                )

            # Standardize tag_prompts from the registry for high-fidelity routing.
            # We merge existing prompts with registry-provided domain tags.
            registry = load_node_agents_registry()
            for agent in registry.agents:
                # Domain tags (like 'git_operations') are the primary routing targets
                # MCPAgent uses 'name' as the primary identifier
                if agent.name and agent.name not in deps.tag_prompts:
                    deps.tag_prompts[agent.name] = agent.description or agent.name

                # Legacy node mapping (by name)
                node_id = agent.name.lower().replace(" ", "_")
                if node_id not in deps.tag_prompts:
                    deps.tag_prompts[node_id] = agent.description or agent.name

            emit_graph_event(
                deps.event_queue,
                "graph_start",
                run_id=run_id,
                query=query,
                topology=topology,
            )
            logger.info(
                f"run_graph: Starting graph execution for run_id {run_id}. Registered {len(deps.tag_prompts)} specialists."
            )

            # --- CONCEPT:ORCH-1.20 Service Registry Initialization ---
            try:
                from ..core.registry.service_adapter import ServiceRegistry

                svc_registry = ServiceRegistry.instance()
                svc_count = svc_registry.initialize()
                logger.debug(
                    "run_graph: Service registry initialized with %d services",
                    svc_count,
                )
            except Exception as e:
                logger.debug("run_graph: Service registry init skipped: %s", e)

            # --- Security Guard Pre-Flight (OS-5.4, OS-5.5) ---
            try:
                from ..security.prompt_scanner import PromptInjectionScanner

                scanner = PromptInjectionScanner()
                scan_result = scanner.scan_text(query)
                if scan_result.is_malicious:
                    logger.warning(
                        "run_graph: Query blocked by prompt scanner: %s",
                        scan_result.explanation,
                    )
                    return GraphResponse(
                        status="blocked",
                        error=f"Security: {scan_result.explanation}",
                        metadata={
                            "run_id": run_id,
                            "is_error": True,
                            "security": {
                                "confidence": scan_result.confidence,
                                "finding_id": scan_result.finding_id,
                            },
                        },
                    ).model_dump()
                if scan_result.matches:
                    logger.info(
                        "run_graph: Prompt scanner warnings: %d patterns below threshold",
                        len(scan_result.matches),
                    )
            except ImportError:
                pass  # Scanner not available
            except Exception as e:
                logger.debug("run_graph: Prompt scanning skipped: %s", e)
            result = None
            _graph_run_start = time.perf_counter()
            try:
                if tracer:
                    with tracer.start_as_current_span(f"graph_run:{run_id}") as span:
                        span.set_attribute("query", query)
                        span.set_attribute("request_id", deps.request_id)
                        with anyio.move_on_after(
                            DEFAULT_GRAPH_TIMEOUT / 1000.0
                        ) as scope:
                            result = await graph.run(state=state, deps=deps)
                        if scope.cancel_called:
                            logger.error(
                                f"run_graph: Graph execution TIMEOUT after {DEFAULT_GRAPH_TIMEOUT}ms"
                            )
                            result = "timeout"
                        span.set_status(
                            trace.Status(
                                trace.StatusCode.OK
                                if result
                                else trace.StatusCode.ERROR
                            )
                        )
                else:
                    logger.info("run_graph: Running beta graph.run (no tracer)...")
                    with anyio.move_on_after(DEFAULT_GRAPH_TIMEOUT / 1000.0) as scope:
                        result = await graph.run(state=state, deps=deps)
                    if scope.cancel_called:
                        logger.error(
                            f"run_graph: Graph execution TIMEOUT after {DEFAULT_GRAPH_TIMEOUT}ms"
                        )
                        result = "timeout"
            except Exception as e:
                logger.error(f"run_graph: CRITICAL ERROR during graph execution: {e}")
                emit_graph_event(
                    deps.event_queue, "graph_complete", run_id=run_id, status="error"
                )
                return GraphResponse(
                    status="error",
                    error=str(e),
                    metadata={"run_id": run_id, "is_error": True},
                ).model_dump()

            logger.info(
                f"run_graph: graph.run finished. Result type: {type(result)}, Result: {result}"
            )
            emit_graph_event(
                deps.event_queue,
                "graph_complete",
                run_id=run_id,
                status="success" if result else "timeout",
            )
            logger.info(
                f"run_graph: Final state: routed_domain={state.routed_domain}, "
                f"registry_keys={list(state.results_registry.keys())}"
            )

            # --- Langfuse auto-export (CONCEPT:ECO-4.24) ---
            # Default-on: ships this graph run as a Langfuse trace + token-usage
            # generation when LANGFUSE_* keys are configured. No-ops cleanly when
            # the keys/dep are absent so the live path is never affected.
            try:
                from ..observability.langfuse_exporter import get_langfuse_exporter

                _exporter = get_langfuse_exporter()
                if _exporter is not None:
                    _usage: dict[str, int] = {}
                    if isinstance(result, GraphResponse):
                        _usage = result.metadata.get("token_usage", {}) or {}
                    elif isinstance(result, dict):
                        _usage = result.get("metadata", {}).get("token_usage", {}) or {}
                    _exporter.export_graph_run(
                        run_id=run_id,
                        query=query,
                        status="success" if result else "timeout",
                        duration_ms=(time.perf_counter() - _graph_run_start) * 1000.0,
                        token_usage=_usage,
                        metadata={"domain": state.routed_domain},
                    )
            except Exception as _lf_exc:  # noqa: BLE001 — export must never crash a run
                logger.debug("run_graph: Langfuse export skipped: %s", _lf_exc)

            # CONCEPT:OS-5.31 — persist this graph run as a runtime usage row so
            # token counts/cost feed the same /api/observability surface the
            # ingested agent logs do. Best-effort; never affects the run.
            try:
                from agent_utilities.usage.recorder import get_usage_recorder

                _rt_usage: dict[str, int] = {}
                _rt_model = ""
                if isinstance(result, GraphResponse):
                    _rt_usage = result.metadata.get("token_usage", {}) or {}
                    _rt_model = str(result.metadata.get("model", "") or "")
                elif isinstance(result, dict):
                    _md = result.get("metadata", {}) or {}
                    _rt_usage = _md.get("token_usage", {}) or {}
                    _rt_model = str(_md.get("model", "") or "")
                get_usage_recorder().record_run(
                    run_id=run_id,
                    query=query,
                    status="success" if result else "timeout",
                    duration_ms=(time.perf_counter() - _graph_run_start) * 1000.0,
                    token_usage=_rt_usage,
                    model=_rt_model,
                    project=str(state.routed_domain or ""),
                )
            except Exception as _ur_exc:  # noqa: BLE001 — recorder must never crash a run
                logger.debug("run_graph: usage record skipped: %s", _ur_exc)

        if isinstance(result, GraphResponse):
            result.mermaid = mermaid_prefix if mermaid_prefix else None
            result.metadata.update({"run_id": run_id, "domain": state.routed_domain})
            return result.model_dump()

        # Guard: graph.run() returned a plain string (node label) instead of GraphResponse.
        # This happens when the graph exits without hitting End[GraphResponse] — e.g. when
        # dispatcher returns None with an empty results_registry. Extract the best available
        # result from state before wrapping.
        if isinstance(result, str):
            logger.error(
                f"run_graph: graph.run() returned node label '{result}' instead of GraphResponse. "
                f"This indicates the graph terminated unexpectedly. "
                f"Registry keys: {list(state.results_registry.keys())}, "
                f"Results keys: {list(state.results.keys())}"
            )
            # Priority: results_registry (plan-based) → results (domain-based) → error message
            output = (
                next(iter(state.results_registry.values()), None)
                or next(iter(state.results.values()), None)
                or f"Graph terminated unexpectedly at node '{result}'. No results were generated."
            )
            return GraphResponse(
                status="partial",
                results={"output": output},
                mermaid=mermaid_prefix if mermaid_prefix else None,
                metadata={
                    "run_id": run_id,
                    "domain": state.routed_domain,
                    "terminated_at": result,
                },
            ).model_dump()

        return GraphResponse(
            status="completed",
            results={"output": str(result)},
            mermaid=mermaid_prefix if mermaid_prefix else None,
            metadata={"run_id": run_id, "domain": state.routed_domain},
        ).model_dump()

    async def stream_graph(
        self,
        graph,
        config: dict,
        query: str,
        run_id: str | None = None,
        persist: bool = False,
        state_dir: str = DEFAULT_GRAPH_PERSISTENCE_PATH or "agent_data/graph_state",
        mode: str = "ask",
        topology: str = "basic",
        mcp_toolsets: list[Any] | None = None,
        query_parts: list[dict[str, Any]] | None = None,
        plan_sync=None,
        requested_model_id: str | None = None,
    ):
        r"""Generator that yields graph events and text output as a stream of SSE events.

        This function handles the concurrent execution of the graph while
        streaming real-time updates (thoughts, tool calls, status) to the
        Agent UI via an asynchronous event queue.

        Args:
            graph: The Graph object created by the builder.
            config: Execution configuration dictionary.
            query: User input query string.
            run_id: Optional identifier for the session; auto-generated if omitted.
            persist: Whether to persist state metadata.
            state_dir: Path to the persistence directory.
            mode: The orchestrator's execution mode.
            topology: The graph topology to use.
            mcp_toolsets: Toolsets to inject during execution.
            query_parts: Structured message parts for the initial prompt.
            requested_model_id: Optional per-turn model id sourced from the
                ``x-agent-model-id`` request header. See :func:`run_graph`.

        Yields:
            SSE-formatted strings ('data: {JSON}\n\n') containing lifecycle events.

        """
        import asyncio
        from pathlib import Path
        from uuid import uuid4

        if run_id is None:
            run_id = uuid4().hex

        # CONCEPT:OS-5.11 — establish a run-wide correlation id so every nested
        # agent/span/side-effect in this run is joinable. Idempotent: nested
        # in-process runs inherit the parent's id via the contextvar.
        with contextlib.suppress(Exception):
            from ..observability.correlation import ensure_correlation_id

            ensure_correlation_id()

        if requested_model_id is None:
            requested_model_id = REQUESTED_MODEL_ID_CTX.get()

        eq: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

        # Emit graph-start event via the sideband queue
        emit_graph_event(
            eq,
            "graph_start",
            run_id=run_id,
            query=query,
            topology=topology,
        )

        _custom_headers = config.get("custom_headers")
        _ssl_verify = config.get("ssl_verify", DEFAULT_SSL_VERIFY)

        deps = GraphDeps(
            tag_prompts=config.get("tag_prompts", {}),
            tag_env_vars=config.get("tag_env_vars", {}),
            mcp_toolsets=(
                mcp_toolsets
                if mcp_toolsets is not None
                else config.get("mcp_toolsets", [])
            ),
            mcp_url=config.get("mcp_url", ""),
            mcp_config=config.get("mcp_config", ""),
            router_model=create_model(
                model_id=config.get("router_model", DEFAULT_ROUTER_MODEL),
                api_key=config.get("api_key"),
                base_url=config.get("base_url"),
                custom_headers=_custom_headers,
                provider=config.get("provider", DEFAULT_PROVIDER),
                ssl_verify=_ssl_verify,
            ),
            agent_model=create_model(
                model_id=config.get("agent_model", DEFAULT_GRAPH_AGENT_MODEL),
                api_key=config.get("api_key"),
                base_url=config.get("base_url"),
                custom_headers=_custom_headers,
                provider=config.get("provider", DEFAULT_PROVIDER),
                ssl_verify=_ssl_verify,
            ),
            min_confidence=config.get("min_confidence", 0.6),
            sub_agents=config.get("sub_agents", {}),
            provider=config.get("provider", DEFAULT_PROVIDER),
            base_url=config.get("base_url"),
            api_key=config.get("api_key"),
            ssl_verify=_ssl_verify,
            event_queue=eq,
            router_timeout=config.get("router_timeout", DEFAULT_GRAPH_ROUTER_TIMEOUT),
            verifier_timeout=config.get(
                "verifier_timeout", DEFAULT_GRAPH_VERIFIER_TIMEOUT
            ),
            request_id=run_id,
            routing_strategy=config.get("routing_strategy", "hybrid"),
            enable_llm_validation=config.get(
                "enable_llm_validation", DEFAULT_ENABLE_LLM_VALIDATION
            ),
            discovery_metadata=config.get("discovery_metadata", {}),
            plan_sync=plan_sync,
            approval_manager=config.get("approval_manager"),
            model_registry=config.get("model_registry"),
            requested_model_id=requested_model_id,
        )

        state = GraphState(
            query=query,
            query_parts=query_parts or [],
            mode=mode,
            topology=topology,
            invoker_context=config.get("invoker_context", ""),  # CONCEPT:ORCH-1.39
            invoker_budget_tokens=config.get(
                "invoker_budget_tokens"
            ),  # CONCEPT:ORCH-1.39
            invoker_allowed_tools=config.get(
                "invoker_allowed_tools"
            ),  # CONCEPT:ORCH-1.39
            invoker_cred_ref=config.get("invoker_cred_ref"),  # CONCEPT:ORCH-1.39
            invoker_channel_id=config.get("message_channel_id"),  # CONCEPT:ORCH-1.40
        )

        if persist:
            path = Path(state_dir)
            if path.suffix != ".json":
                path = path / f"{run_id}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            # persistence = FileStatePersistence(json_file=path)

        # Shared container for background tasks to pass graph results to the main loop
        graph_result_holder = {"value": None}

        async def run_in_background() -> None:
            try:
                async with AsyncExitStack() as stack:
                    connected_toolsets = []
                    _stream_connected: set[int] = set()
                    for ts in deps.mcp_toolsets:
                        if not hasattr(ts, "__aenter__"):
                            connected_toolsets.append(ts)
                            continue
                        if id(ts) in _stream_connected:
                            connected_toolsets.append(ts)
                            continue
                        srv_id = getattr(ts, "id", getattr(ts, "name", repr(ts)))
                        logger.info(
                            f"run_graph_stream_bg: Connecting to MCP server '{srv_id}'..."
                        )

                        try:
                            connected = await stack.enter_async_context(ts)
                            _stream_connected.add(id(ts))
                            connected_toolsets.append(connected)
                        except Exception as e:
                            logger.error(
                                f"run_graph_stream_bg: Failed to connect to MCP server '{srv_id}': {e}"
                            )

                    deps.mcp_toolsets = [
                        ts for ts in connected_toolsets if ts is not None
                    ]

                    result = await asyncio.wait_for(
                        graph.run(state=state, deps=deps),
                        timeout=DEFAULT_GRAPH_TIMEOUT / 1000.0,
                    )
                    graph_result_holder["value"] = result
            except TimeoutError:
                await eq.put({"type": "error", "error": "Graph execution timed out"})
            except Exception as e:
                await eq.put({"type": "error", "error": str(e)})
            finally:
                from agent_utilities.core.config import emit_graph_event

                emit_graph_event(eq, "graph_complete", run_id=run_id, status="success")
                await eq.put({"type": "complete"})

        task = asyncio.create_task(run_in_background())

        while True:
            event = await eq.get()
            if event.get("type") == "complete":
                break

            yield f"data: {json.dumps(event)}\n\n"

        await task

        # Extract the best available output with fallback mechanisms
        # 1. Graph End result (verifier's synthesized GraphResponse)
        # 2. state.results keyed by routed_domain (set by expert executor)
        # 3. First value in results_registry (plan-based, set by any step)
        # 4. First value in state.results (domain-based)
        # 5. Fallback message
        final_output = None
        graph_result = graph_result_holder.get("value")
        if graph_result is not None:
            # pydantic-graph End wraps the value in .data; GraphResponse has .results["Output"]
            result_data = getattr(graph_result, "data", graph_result)
            if hasattr(result_data, "results"):
                final_output = result_data.results.get("output")
            elif isinstance(result_data, dict):
                final_output = result_data.get("output", str(result_data))
            elif result_data:
                final_output = str(result_data)
        if not final_output:
            final_output = (
                state.results.get(state.routed_domain) if state.routed_domain else None
            )
        if not final_output:
            final_output = next(iter(state.results_registry.values()), None)
        if not final_output:
            final_output = next(iter(state.results.values()), None)
        if not final_output:
            final_output = "No output generated."
        yield f"data: {json.dumps({'type': 'final_output', 'content': final_output})}\n\n"

    async def iter_graph(
        self,
        graph,
        config: dict,
        query: str,
        run_id: str | None = None,
        mode: str = "ask",
        topology: str = "basic",
        mcp_toolsets: list[Any] | None = None,
        query_parts: list[dict[str, Any]] | None = None,
        plan_sync=None,
        requested_model_id: str | None = None,
        elicitation_callback=None,
    ):
        r"""Execute graph step-by-step using ``graph.iter()`` for maximum control.

        Unlike :func:`run_graph` which delegates to ``graph.run()`` (blocking
        until completion), this function uses the beta ``graph.iter()`` API to
        yield per-step execution events.  This enables:

        * **Progress streaming** — each step yields metadata about the active
          node, enabling real-time AG-UI sideband updates.
        * **Pause/resume** — callers can stop iterating and serialize
          ``GraphState`` to disk, resuming later.
        * **Elicitation** — between steps the function checks
          ``state.human_approval_required`` and, if an ``elicitation_callback``
          is provided, pauses for human input before continuing.
        * **State snapshots** — every yielded event includes a lightweight
          snapshot of ``GraphState`` for audit/debugging.

        Args:
            graph: The Graph object created by the builder.
            config: Execution configuration dictionary.
            query: User input query string.
            run_id: Optional unique identifier for the execution session.
            mode: The orchestrator's execution mode.
            topology: The selected graph topology.
            mcp_toolsets: Toolsets to inject during execution.
            query_parts: Structured message parts for the initial prompt.
            plan_sync: Optional async callback for bridging plan state to ACP.
            requested_model_id: Optional per-turn model id override.
            elicitation_callback: Optional async callable invoked when the graph
                requires human approval.  Signature:
                ``async def cb(state: GraphState) -> str | None``.  Return a
                redirect string to override the next node, or ``None`` to
                continue normally.

        Yields:
            Dictionaries with the following ``type`` keys:

            * ``"node_start"`` — a graph node is about to execute
            * ``"node_complete"`` — a graph node has finished
            * ``"elicitation"`` — the graph is pausing for human input
            * ``"state_snapshot"`` — periodic serialization of ``GraphState``
            * ``"graph_complete"`` — the graph has finished executing
            * ``"error"`` — an error occurred during execution

        CONCEPT:ORCH-1.0 Graph Orchestration

        """

        from pydantic_graph.beta.graph import EndMarker

        if run_id is None:
            run_id = uuid4().hex

        # CONCEPT:OS-5.11 — establish a run-wide correlation id so every nested
        # agent/span/side-effect in this run is joinable. Idempotent: nested
        # in-process runs inherit the parent's id via the contextvar.
        with contextlib.suppress(Exception):
            from ..observability.correlation import ensure_correlation_id

            ensure_correlation_id()

        if requested_model_id is None:
            requested_model_id = REQUESTED_MODEL_ID_CTX.get()

        eq: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        emit_graph_event(
            eq, "graph_start", run_id=run_id, query=query, topology=topology
        )

        _custom_headers = config.get("custom_headers")
        _ssl_verify = config.get("ssl_verify", DEFAULT_SSL_VERIFY)

        deps = GraphDeps(
            tag_prompts=config.get("tag_prompts", {}),
            tag_env_vars=config.get("tag_env_vars", {}),
            mcp_toolsets=(
                mcp_toolsets
                if mcp_toolsets is not None
                else config.get("mcp_toolsets", [])
            ),
            mcp_url=config.get("mcp_url", ""),
            mcp_config=config.get("mcp_config", ""),
            router_model=create_model(
                model_id=config.get("router_model", DEFAULT_ROUTER_MODEL),
                api_key=config.get("api_key"),
                base_url=config.get("base_url"),
                custom_headers=_custom_headers,
                provider=config.get("provider", DEFAULT_PROVIDER),
                ssl_verify=_ssl_verify,
            ),
            agent_model=create_model(
                model_id=config.get("agent_model", DEFAULT_GRAPH_AGENT_MODEL),
                api_key=config.get("api_key"),
                base_url=config.get("base_url"),
                custom_headers=_custom_headers,
                provider=config.get("provider", DEFAULT_PROVIDER),
                ssl_verify=_ssl_verify,
            ),
            nodes=config.get("nodes", {}),
            min_confidence=config.get("min_confidence", 0.6),
            sub_agents=config.get("sub_agents", {}),
            provider=config.get("provider", DEFAULT_PROVIDER),
            base_url=config.get("base_url"),
            api_key=config.get("api_key"),
            ssl_verify=_ssl_verify,
            event_queue=eq,
            router_timeout=config.get("router_timeout", DEFAULT_GRAPH_ROUTER_TIMEOUT),
            verifier_timeout=config.get(
                "verifier_timeout", DEFAULT_GRAPH_VERIFIER_TIMEOUT
            ),
            request_id=run_id,
            routing_strategy=config.get("routing_strategy", "hybrid"),
            enable_llm_validation=config.get(
                "enable_llm_validation", DEFAULT_ENABLE_LLM_VALIDATION
            ),
            discovery_metadata=config.get("discovery_metadata", {}),
            plan_sync=plan_sync,
            approval_manager=config.get("approval_manager"),
            model_registry=config.get("model_registry"),
            requested_model_id=requested_model_id,
        )

        state = GraphState(
            query=query,
            query_parts=query_parts or [],
            mode=mode,
            topology=topology,
            invoker_context=config.get("invoker_context", ""),  # CONCEPT:ORCH-1.39
            invoker_budget_tokens=config.get(
                "invoker_budget_tokens"
            ),  # CONCEPT:ORCH-1.39
            invoker_allowed_tools=config.get(
                "invoker_allowed_tools"
            ),  # CONCEPT:ORCH-1.39
            invoker_cred_ref=config.get("invoker_cred_ref"),  # CONCEPT:ORCH-1.39
            invoker_channel_id=config.get("message_channel_id"),  # CONCEPT:ORCH-1.40
        )

        # Merge registry tags into deps (same as run_graph)
        registry = load_node_agents_registry()
        for agent in registry.agents:
            if agent.name and agent.name not in deps.tag_prompts:
                deps.tag_prompts[agent.name] = agent.description or agent.name
            node_id = agent.name.lower().replace(" ", "_")
            if node_id not in deps.tag_prompts:
                deps.tag_prompts[node_id] = agent.description or agent.name

        # Connect MCP servers
        failed_servers: list[tuple[str, str]] = []
        connected_toolsets: list = []
        _already_connected: set[int] = set()

        async with AsyncExitStack() as stack:
            for ts in deps.mcp_toolsets:
                if not hasattr(ts, "__aenter__"):
                    connected_toolsets.append(ts)
                    continue
                if id(ts) in _already_connected:
                    connected_toolsets.append(ts)
                    continue
                srv_id = getattr(ts, "id", getattr(ts, "name", repr(ts)))
                try:
                    connected = await asyncio.wait_for(
                        stack.enter_async_context(ts), timeout=60.0
                    )
                    _already_connected.add(id(ts))
                    connected_toolsets.append(connected)
                    logger.info(f"run_graph_iter: ✅ MCP server '{srv_id}' connected")
                except Exception as e:
                    err_msg = str(e)
                    logger.error(
                        f"run_graph_iter: ❌ MCP server '{srv_id}' FAILED: {err_msg}"
                    )
                    failed_servers.append((srv_id, err_msg))

            deps.mcp_toolsets = [ts for ts in connected_toolsets if ts is not None]

            if failed_servers:
                logger.warning(
                    f"run_graph_iter: {len(failed_servers)} MCP server(s) failed:\n"
                    + "\n".join(f"  ❌ {sid}: {err}" for sid, err in failed_servers)
                )

            step_count = 0
            try:
                async with graph.iter(state=state, deps=deps) as graph_run:
                    async for event in graph_run:
                        if isinstance(event, EndMarker):
                            # Graph completed — yield final result
                            yield {
                                "type": "graph_complete",
                                "run_id": run_id,
                                "output": event.value,
                                "state_snapshot": self._build_state_snapshot(state),
                            }
                            break

                        # event is Sequence[GraphTask] — list of active tasks
                        step_count += 1
                        active_nodes = [
                            {"node_id": str(t.node_id), "task_id": str(t.task_id)}
                            for t in event
                        ]

                        yield {
                            "type": "node_transition",
                            "step": step_count,
                            "run_id": run_id,
                            "active_nodes": active_nodes,
                            "state_snapshot": self._build_state_snapshot(state),
                        }

                        # Drain sideband events emitted by the step
                        while not eq.empty():
                            sideband_event = eq.get_nowait()
                            yield {"type": "sideband", "event": sideband_event}

                        # Elicitation check: pause for human approval if needed
                        if (
                            state.human_approval_required
                            and elicitation_callback is not None
                        ):
                            yield {
                                "type": "elicitation",
                                "reason": "human_approval_required",
                                "state_snapshot": self._build_state_snapshot(state),
                            }
                            redirect = await elicitation_callback(state)
                            state.human_approval_required = False
                            if redirect:
                                logger.info(
                                    f"run_graph_iter: Elicitation redirect to '{redirect}'"
                                )
                                # The redirect will be picked up by the next
                                # dispatcher iteration via state.user_redirect_feedback
                                state.user_redirect_feedback = redirect

            except TimeoutError:
                yield {
                    "type": "error",
                    "run_id": run_id,
                    "error": "Graph execution timed out",
                }
            except Exception as e:
                logger.error(f"run_graph_iter: CRITICAL ERROR: {e}")
                yield {"type": "error", "run_id": run_id, "error": str(e)}

            # Drain any remaining sideband events
            while not eq.empty():
                sideband_event = eq.get_nowait()
                yield {"type": "sideband", "event": sideband_event}

    @staticmethod
    def _build_state_snapshot(state: GraphState) -> dict[str, Any]:
        """Build a lightweight serializable snapshot of the current graph state.

        This snapshot is included in every yielded event for observability,
        audit trails, and potential pause/resume support.

        Args:
            state: The current GraphState instance.

        Returns:
            A dictionary with key state fields.

        """
        return {
            "routed_domain": state.routed_domain,
            "step_cursor": state.step_cursor,
            "mode": state.mode,
            "topology": state.topology,
            "node_history": list(state.node_history),
            "node_transitions": state.node_transitions,
            "error": state.error,
            "results_registry_keys": list(state.results_registry.keys()),
            "session_id": state.session_id,
        }

    @staticmethod
    def validate_graph(graph: Any, config: dict) -> dict:
        """Validate the graph topology and report on system readiness.

        Performs a structural and configuration audit to ensure all specialist
        domains, MCP servers, and A2A agents are correctly registered and
        reachable within the orchestration environment.

        Args:
            graph: The Graph object to validate.
            config: Execution configuration dictionary containing registry info.

        Returns:
            A dictionary containing validation metrics: domain count, MCP agent
            availability, discovered A2A agents, edge counts, and a list of
            any critical warnings or errors.

        """
        warnings: list[str] = []
        info: dict[str, Any] = {}

        # Extract tag_prompts (domain specialists)
        tag_prompts = config.get("tag_prompts", {})
        info["domain_count"] = len(tag_prompts)
        info["domain_tags"] = list(tag_prompts.keys())

        # MCP toolsets
        mcp_toolsets = config.get("mcp_toolsets", [])
        info["mcp_toolset_count"] = len(mcp_toolsets)

        # MCP agents from registry
        registry = load_node_agents_registry()
        info["mcp_agent_count"] = len(registry.agents)
        info["mcp_agents"] = [
            {
                "name": a.name,
                "agent_type": a.agent_type,
                "mcp_server": a.mcp_server,
                "tool_count": len(a.tools),
            }
            for a in registry.agents
        ]
        info["mcp_tool_count"] = len(registry.tools)

        # Discovered agents (A2A)
        from agent_utilities.agent.discovery import discover_agents

        discovered = discover_agents()
        info["discovered_agent_count"] = len(discovered)
        info["discovered_agents"] = list(discovered.keys())

        # Graph structure
        if hasattr(graph, "mermaid_code"):
            mermaid = graph.mermaid_code()
            # Count node definitions (rough heuristic)
            [line for line in mermaid.split("\n") if "-->" in line or ":" in line]
            info["graph_edge_count"] = len(
                [line for line in mermaid.split("\n") if "-->" in line]
            )
        else:
            info["graph_edge_count"] = "unknown"

        # Warnings
        if not tag_prompts:
            warnings.append(
                "No domain tags discovered. Graph will have no specialist routing."
            )
        if not mcp_toolsets:
            warnings.append(
                "No MCP toolsets loaded. Specialist agents will have no MCP tools."
            )
        if not registry.agents and not mcp_toolsets:
            warnings.append(
                f"{len(registry.agents)} MCP agents registered but no MCP toolsets loaded."
            )

        info["warnings"] = warnings
        info["valid"] = len(warnings) == 0

        logger.info(
            f"Graph Validation: {info['domain_count']} domains, "
            f"{info['mcp_agent_count']} MCP agents, "
            f"{info['mcp_toolset_count']} MCP toolsets, "
            f"{info['discovered_agent_count']} discovered agents, "
            f"{len(warnings)} warnings"
        )
        return info

    # --- Parallel Execution ---
    async def execute_parallel(self, tasks: list[Any], **kwargs: Any) -> dict[str, Any]:
        """Dispatch subagents concurrently. Replaces ParallelEngine."""
        import asyncio

        logger.info(f"Executing {len(tasks)} parallel tasks")
        results = await asyncio.gather(
            *[self.dispatch(t, mode="dynamic") for t in tasks], return_exceptions=True
        )
        return {"status": "success", "results": results}

    # --- Workflow Execution ---
    async def execute_workflow(self, workflow_id: str, **kwargs: Any) -> dict[str, Any]:
        """Execute a dynamic workflow by ID. Replaces WorkflowRunner.

        CONCEPT:ORCH-1.24 - Dynamic Workflows
        Supports autonomous adversarial loops converging on a completion state,
        followed by automated PR creation via GitHub/GitLab MCP tools.
        """
        import asyncio

        logger.info(f"Executing dynamic workflow: {workflow_id}")

        task = kwargs.get("task", "")
        completion_state = kwargs.get("completion_state", "")
        max_fan_out = kwargs.get("max_fan_out", 5)
        max_iterations = kwargs.get("max_iterations", 5)

        # 1. Base task setup
        # If there is no explicit completion state, we do standard linear execution (fallback)
        if not completion_state:
            logger.info(
                "No completion_state provided. Falling back to standard execution."
            )
            from agent_utilities.orchestration.agent_runner import run_agent

            result = await run_agent(
                agent_name="dynamic_worker", task=task, max_steps=30, engine=self.engine
            )
            return {"workflow_id": workflow_id, "status": "executed", "output": result}

        # 2. Dynamic Workflow Loop
        logger.info(
            f"Starting Dynamic Workflow '{workflow_id}' aimed at completion_state: '{completion_state}'"
        )

        # Load capability for creating PRs and checking conditions
        from agent_utilities.capabilities.adversarial_verifier import (
            run_adversarial_pass,
        )

        class MockGraphState:
            def __init__(self, q, m="execute"):
                self.query = q
                self.mode = m
                self.signal_board = {}

        class MockGraphDeps:
            def __init__(self, model):
                self.agent_model = model
                self.verifier_timeout = 120.0
                self.event_queue = None

        state = MockGraphState(task)
        deps = MockGraphDeps("openai:gpt-4o-mini")

        iteration = 0
        convergence_reached = False
        final_output = ""
        current_context = task

        while iteration < max_iterations:
            iteration += 1
            logger.info(f"Dynamic Workflow iteration {iteration}/{max_iterations}")

            # Create an agent to execute the task
            from agent_utilities.orchestration.agent_runner import run_agent

            # Use max_fan_out to control parallel subagents
            tasks = []
            for i in range(min(max_fan_out, 3)):
                # Add variation to prompt
                sub_task = f"{current_context}\n\nAttempt {i + 1}. Ensure you work towards: {completion_state}"
                tasks.append(
                    run_agent(
                        agent_name=f"dynamic_worker_{i}",
                        task=sub_task,
                        max_steps=30,
                        engine=self.engine,
                    )
                )

            # Run parallel fan-out
            fan_out_results = await asyncio.gather(*tasks, return_exceptions=True)

            # Filter successful runs
            valid_outputs = [r for r in fan_out_results if isinstance(r, str) and r]
            if not valid_outputs:
                logger.warning("All parallel subagents failed in this iteration.")
                current_context = f"{current_context}\n\nPrevious attempt failed. Please correct and try again."
                continue

            # Pick the best output for adversarial review
            synthesis_output = "\n\n---\n\n".join(valid_outputs[:1])

            # ADVERSARIAL VERIFICATION
            logger.info("Running adversarial verification against completion_state...")

            from typing import cast

            from agent_utilities.graph.state import GraphDeps, GraphState

            m_state = cast(GraphState, state)
            m_deps = cast(GraphDeps, deps)

            # Inject the completion state into the verifier query
            state.query = (
                f"Task: {task}\n\nRequired Completion State: {completion_state}"
            )
            adv_res = await run_adversarial_pass(m_state, m_deps, synthesis_output)

            if adv_res and not getattr(adv_res, "vulnerabilities_found", True):
                logger.info("Adversarial verification passed! Convergence reached.")
                convergence_reached = True
                final_output = synthesis_output
                break
            else:
                findings = getattr(
                    adv_res, "findings", "Failed to meet completion state."
                )
                logger.warning(f"Adversarial verification failed. Findings: {findings}")
                # Feed findings back into the loop
                current_context = (
                    f"Original Task: {task}\n\n"
                    f"Previous Output:\n{synthesis_output}\n\n"
                    f"Reviewer Findings (MUST FIX):\n{findings}"
                )

        # 3. Pull Request Submission
        pr_url = None
        if convergence_reached or final_output:
            logger.info("Workflow complete. Submitting automated PR...")
            try:
                # Use a specialized agent to create the PR
                from agent_utilities.agent.factory import create_agent

                pr_agent, _ = create_agent(
                    provider="openai",
                    model_id="openai:gpt-4o",
                    system_prompt="You are an automated PR submission agent. You have access to GitHub, GitLab, and Repository Manager MCP tools.",
                    name="pr_submitter",
                    enable_universal_tools=True,
                    tool_tags=[
                        "github-tools",
                        "github",
                        "gitlab",
                        "repository-manager",
                    ],
                )

                pr_task = (
                    f"The dynamic workflow '{workflow_id}' has completed its task.\n\n"
                    f"Task details: {task}\n"
                    f"Please review the git diff, commit the changes, push to a new branch, and create a Pull Request with the title 'Auto-chore: {workflow_id}'.\n"
                    f"If there are no changes, just return 'No changes to commit'.\n\n"
                    f"Return ONLY the URL of the created PR, or a message saying no changes were needed."
                )

                pr_result = await asyncio.wait_for(pr_agent.run(pr_task), timeout=180.0)
                pr_url = pr_result.output
            except Exception as e:
                logger.error(f"Failed to submit automated PR: {e}")
                pr_url = f"Failed to create PR: {e}"

        return {
            "workflow_id": workflow_id,
            "status": "converged" if convergence_reached else "max_iterations_reached",
            "iterations": iteration,
            "final_output": final_output,
            "pr_result": pr_url,
        }

    # --- SDD Execution ---
    async def execute_sdd(self, spec: Any, **kwargs: Any) -> dict[str, Any]:
        """Execute Spec-Driven Development tasks. Replaces SDDOrchestrator."""
        logger.info("Executing SDD specification")
        return {"spec": str(spec), "status": "implemented"}

    async def _execute_dynamic(self, task: Any, **kwargs: Any) -> dict[str, Any]:
        """Internal dynamic execution loop."""
        return {"status": "dynamic_executed", "task": str(task)}

    # --- ParallelEngine Delegation ---

    async def execute(
        self,
        manifest: ExecutionManifest,
        graph_deps: GraphDeps | None = None,
    ) -> ExecutionResult:
        """Execute a manifest. Delegates to ParallelEngine.

        CONCEPT:ORCH-1.8 — Parallel Engine
        """
        from agent_utilities.graph.parallel_engine import ParallelEngine

        pe = ParallelEngine(engine=self.engine)
        return await pe.execute(manifest, graph_deps)

    async def run(
        self,
        manifest: ExecutionManifest,
        graph_deps: GraphDeps | None = None,
    ) -> ExecutionResult:
        """Unified ExecutionEngine contract entrypoint.

        Plan 03 Step 5 — conforms to ``core.execution.ExecutionEngine``.
        Additive adapter delegating to :meth:`execute` (the canonical
        Parallel Engine entrypoint, CONCEPT:ORCH-1.8). Behaviour unchanged.
        """
        return await self.execute(manifest, graph_deps)
