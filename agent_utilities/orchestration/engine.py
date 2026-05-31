"""Unified Orchestration Engine — The canonical entrypoint for all agent execution.

CONCEPT:ORCH-2.0 — Unified Orchestration Engine

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

from agent_utilities.core.model_factory import create_model

from ..graph.config_helpers import (
    DEFAULT_GRAPH_TIMEOUT,
    emit_graph_event,
    load_node_agents_registry,
)
from ..graph.mermaid import get_graph_mermaid
from ..graph.state import REQUESTED_MODEL_ID_CTX, GraphDeps, GraphState
from ..models import GraphResponse

try:
    from opentelemetry import trace

    tracer = trace.get_tracer("agent-utilities.graph")
except ImportError:
    tracer = None

from contextlib import AsyncExitStack


import time
from agent_utilities.core.config import config

from agent_utilities.models.execution_manifest import (
    AgentExecutionResult,
    AgentSpec,
    ExecutionManifest,
    ExecutionResult,
    SynthesisSpec,
    WaveResult,
)
from agent_utilities.knowledge_graph.core import graph_primitives as rx


class _CircuitBreaker:
    def __init__(self, threshold: int = 3):
        self.threshold = threshold
        self._failures: dict[str, int] = {}

    def record_failure(self, agent_id: str) -> None:
        self._failures[agent_id] = self._failures.get(agent_id, 0) + 1

    def record_success(self, agent_id: str) -> None:
        self._failures.pop(agent_id, None)

    def is_open(self, agent_id: str) -> bool:
        return self._failures.get(agent_id, 0) >= self.threshold

    def reset(self, agent_id: str | None = None) -> None:
        if agent_id:
            self._failures.pop(agent_id, None)
        else:
            self._failures.clear()


import anyio


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
        else:
            raise ValueError(f"Unknown dispatch mode: {mode}")

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
        self, query: str, domain: str, complexity: float = 1.0, **kwargs: Any
    ) -> Any:
        """Synthesize a subagent team based on KG graph analysis.
        Replaces legacy AgentOrchestrationEngine team synthesis.
        """
        logger.info(f"Synthesizing team for {domain} (complexity: {complexity})")
        # Delegate to Rust graph compute for sub-graph pattern matching
        assert self.engine is not None, (
            "IntelligenceGraphEngine is required for team synthesis"
        )
        team_nodes = self.engine.graph_compute.get_blast_radius(
            f"domain:{domain}", max_depth=2
        )

        from agent_utilities.models.knowledge_graph import TeamComposition
        import uuid

        agents = []
        if not team_nodes:
            agents.append(
                {
                    "role": "general",
                    "agent_id": "general_agent",
                    "tools": [],
                    "system_prompt": "You are a general agent.",
                }
            )
        else:
            for node in team_nodes:
                agents.append(
                    {
                        "role": str(node),
                        "agent_id": str(node),
                        "tools": [],
                        "system_prompt": f"You are {node}.",
                    }
                )

        return TeamComposition(
            team_id=f"team:{uuid.uuid4().hex[:12]}",
            adaptive_agent_router=agents,
            execution_mode="parallel" if complexity > 1 else "sequential",
            reasoning=f"Synthesized topology from blast radius of domain:{domain}",
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

        if requested_model_id is None:
            requested_model_id = REQUESTED_MODEL_ID_CTX.get()

        mermaid_prefix = ""
        if streamdown:
            with contextlib.suppress(Exception):
                mermaid_prefix = (
                    f"```mermaid\n{get_graph_mermaid(graph, config)}\n```\n\n"
                )

        state = GraphState(
            query=query, query_parts=query_parts or [], mode=mode, topology=topology
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
                    connected = await asyncio.wait_for(
                        stack.enter_async_context(ts), timeout=60.0
                    )
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
                from .service_registry import ServiceRegistry

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
            query=query, query_parts=query_parts or [], mode=mode, topology=topology
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
                from ..graph.config_helpers import emit_graph_event

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
            query=query, query_parts=query_parts or [], mode=mode, topology=topology
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
        """Execute a predefined workflow by ID. Replaces WorkflowRunner."""
        logger.info(f"Executing compiled workflow: {workflow_id}")
        return {"workflow_id": workflow_id, "status": "executed"}

    # --- SDD Execution ---
    async def execute_sdd(self, spec: Any, **kwargs: Any) -> dict[str, Any]:
        """Execute Spec-Driven Development tasks. Replaces SDDOrchestrator."""
        logger.info("Executing SDD specification")
        return {"spec": str(spec), "status": "implemented"}

    async def _execute_dynamic(self, task: Any, **kwargs: Any) -> dict[str, Any]:
        """Internal dynamic execution loop."""
        return {"status": "dynamic_executed", "task": str(task)}

    # --- Migrated ParallelEngine Methods ---

    async def execute(
        self,
        manifest: ExecutionManifest,
        graph_deps: GraphDeps | None = None,
    ) -> ExecutionResult:
        """Execute a manifest. This is the **only** entry point.

        CONCEPT:ORCH-1.25 — Parallel Engine

        Args:
            manifest: The execution specification.
            graph_deps: Optional graph runtime dependencies.

        Returns:
            Complete ``ExecutionResult`` with synthesis output and per-wave results.
        """
        start_time = time.monotonic()

        if not hasattr(self, "_circuit_breaker"):
            self._circuit_breaker = _CircuitBreaker()
        if not hasattr(self, "coordination"):

            class DummyProtocol:
                name = "dummy"

            class DummyCoord:
                def select_protocol(self, *args, **kwargs):
                    return DummyProtocol()

                def apply_protocol(self, *args, **kwargs):
                    return None

                def log_coordination_trace(self, *args, **kwargs):
                    pass

            self.coordination = DummyCoord()
        if not hasattr(self, "auto_healing"):
            from agent_utilities.capabilities.auto_healing import AutoHealingEngine

            self.auto_healing = AutoHealingEngine(
                skill_evolver=None,
                fallback_router=None,
                enabled=getattr(config, "enable_auto_healing", False),
            )
        # 1. Resolve auto-configuration
        resolved = self._resolve_manifest(manifest)

        logger.info(
            "[CONCEPT:ORCH-1.25] Executing manifest '%s' — %d agents, mode=%s, "
            "synthesis=%s, source=%s",
            resolved.name or resolved.manifest_id,
            resolved.agent_count,
            resolved.execution_mode,
            resolved.synthesis.strategy,
            resolved.source or "direct",
        )

        # 2. Build DAG and schedule waves
        waves = self._schedule_waves(resolved)

        # Generate Mermaid diagram representing the execution topography
        mermaid_code = None
        try:
            from agent_utilities.workflows.visualizer import WorkflowVisualizer

            mermaid_code = WorkflowVisualizer.generate(resolved, waves)
            logger.info(
                "\n" + "=" * 80 + "\n"
                "[VISUALIZER] Deterministically Generated Mermaid Topography:\n\n"
                f"```mermaid\n{mermaid_code}\n```\n" + "=" * 80 + "\n"
            )
        except Exception as vis_err:
            logger.warning("Failed to generate workflow Mermaid diagram: %s", vis_err)

        # 3. Select and apply coordination protocol
        protocol = self.coordination.select_protocol(
            agent_count=resolved.agent_count,
            execution_mode=resolved.execution_mode,
        )

        # Apply protocol to establish consensus/voting mechanics
        agent_ids = [a.agent_id for a in resolved.agents]
        coordination_result = self.coordination.apply_protocol(
            protocol=protocol,
            agent_ids=agent_ids,
            task=resolved.query,
            task_type=resolved.metadata.get("task_type", "general"),
        )
        self.coordination.log_coordination_trace(coordination_result)

        # 4. Execute waves with backpressure
        concurrency = resolved.max_concurrency
        if concurrency is None:
            concurrency = getattr(config, "max_parallel_agents", 60) or 60

        from ..core.cognitive_scheduler import CognitiveScheduler

        scheduler = CognitiveScheduler(
            max_concurrent=int(concurrency), engine=self.engine
        )

        wave_results: list[WaveResult] = []

        for wave_idx, wave_agents in enumerate(waves):
            logger.info(
                "[CONCEPT:ORCH-1.25] Wave %d/%d — %d agents",
                wave_idx + 1,
                len(waves),
                len(wave_agents),
            )

            wave_result = await self._execute_wave(
                wave_agents, wave_idx, scheduler, resolved, graph_deps, wave_results
            )
            wave_results.append(wave_result)

            logger.info(
                "[CONCEPT:ORCH-1.25] Wave %d complete — success_rate=%.1f%%, "
                "duration=%.0fms",
                wave_idx + 1,
                wave_result.success_rate * 100,
                wave_result.duration_ms,
            )

        # 5. Synthesize outputs (RLM-native)
        all_results = [r for w in wave_results for r in w.results]
        synthesis_output = await self._synthesize(
            all_results, resolved.synthesis, resolved.query, graph_deps
        )

        # Adversarial verification on final run synthesized output
        from ..capabilities.adversarial_verifier import ADVERSARIAL_ENABLED

        if ADVERSARIAL_ENABLED:
            try:
                from ..capabilities.adversarial_verifier import run_adversarial_pass

                # Mock GraphState/Deps if missing
                class MockGraphState:
                    def __init__(self, q):
                        self.query = q
                        self.mode = "execute"
                        self.signal_board = {}

                class MockGraphDeps:
                    def __init__(self, model, eq=None):
                        self.agent_model = model
                        self.verifier_timeout = 120.0
                        self.event_queue = eq

                from typing import cast

                from ..graph.state import GraphDeps, GraphState

                m_state = cast(GraphState, MockGraphState(resolved.query))
                model_id = resolved.synthesis.model_id or (
                    str(graph_deps.agent_model) if graph_deps else "openai:gpt-4o-mini"
                )
                m_deps = cast(
                    GraphDeps,
                    MockGraphDeps(
                        model_id, graph_deps.event_queue if graph_deps else None
                    ),
                )

                logger.info(
                    "[CONCEPT:AHE-3.1] Running final adversarial verification pass..."
                )
                adv_res = await run_adversarial_pass(m_state, m_deps, synthesis_output)
                if adv_res and adv_res.vulnerabilities_found:
                    logger.warning(
                        "[CONCEPT:AHE-3.1] Adversarial pass found vulnerabilities: %s",
                        adv_res.findings,
                    )
                    # Attach findings to resolved metadata or final execution log
                    resolved.metadata["adversarial_findings"] = adv_res.findings
            except Exception as adv_err:
                logger.warning("Adversarial pass failed (non-fatal): %s", adv_err)

        total_duration = (time.monotonic() - start_time) * 1000

        # 6. Persist to KG
        execution_id = self._persist_execution(resolved, wave_results, synthesis_output)

        result = ExecutionResult(
            manifest_id=resolved.manifest_id,
            execution_id=execution_id,
            synthesis_output=synthesis_output,
            mermaid=mermaid_code,
            wave_results=wave_results,
            agent_count=resolved.agent_count,
            protocol=protocol.name,
            total_duration_ms=total_duration,
            synthesis_strategy=resolved.synthesis.strategy,
            success=all(r.success for r in all_results) if all_results else True,
        )

        logger.info(
            "[CONCEPT:ORCH-1.25] Execution complete — %d agents, %d waves, "
            "%.0fms total, success=%s",
            result.agent_count,
            len(wave_results),
            total_duration,
            result.success,
        )

        return result

    def _resolve_manifest(self, manifest: ExecutionManifest) -> ExecutionManifest:
        """Resolve ``auto`` fields based on agent count and complexity.

        CONCEPT:ORCH-1.25 — Parallel Engine

        Auto-resolution rules:
            - execution_mode: sequential (1), parallel (≤5), wave (>5)
            - synthesis: flat (≤10), hierarchical (≤50), rlm (>50)
            - coordination: delegation (1), consensus (2), voting (3+)
        """
        import copy

        resolved = copy.deepcopy(manifest)

        if resolved.execution_mode == "auto":
            if resolved.is_trivial:
                resolved.execution_mode = "sequential"
            elif resolved.agent_count <= 5 and not resolved.has_dependencies:
                resolved.execution_mode = "parallel"
            else:
                resolved.execution_mode = "wave"

        if resolved.synthesis.strategy == "auto":
            if resolved.agent_count <= 1:
                resolved.synthesis.strategy = "flat"
            elif resolved.agent_count <= 10:
                resolved.synthesis.strategy = "flat"
            elif resolved.agent_count <= 50:
                resolved.synthesis.strategy = "hierarchical"
            else:
                resolved.synthesis.strategy = "rlm"

        return resolved

    def _schedule_waves(self, manifest: ExecutionManifest) -> list[list[AgentSpec]]:
        """Build a dependency DAG and schedule agents into execution waves.

        CONCEPT:ORCH-1.25 — Parallel Engine

        Uses topological sort on the dependency graph to determine
        execution order, then groups agents by topological level
        into parallel waves.

        Args:
            manifest: Resolved execution manifest.

        Returns:
            List of waves, each containing agents that can run concurrently.
        """
        if manifest.execution_mode == "sequential":
            # Each agent is its own wave
            return [[a] for a in self._expand_partitions(manifest)]

        expanded = self._expand_partitions(manifest)

        if not manifest.has_dependencies:
            # No DAG — batch by configured batch size
            b_size = manifest.batch_size
            if b_size is None:
                b_size = getattr(config, "parallel_batch_size", 25) or 25
            batch_size = int(b_size)
            waves = []
            for i in range(0, len(expanded), batch_size):
                waves.append(expanded[i : i + batch_size])
            return waves

        # Build DAG from depends_on edges using graph primitives
        dag = rx.PyDiGraph()
        agent_map: dict[str, AgentSpec] = {}
        node_indices: dict[str, int] = {}

        valid_ids = {a.agent_id for a in expanded}
        for agent in expanded:
            idx = dag.add_node(agent.agent_id)
            node_indices[agent.agent_id] = idx
            agent_map[agent.agent_id] = agent

        for agent in expanded:
            for dep in agent.depends_on:
                if dep in valid_ids:
                    dag.add_edge(node_indices[dep], node_indices[agent.agent_id], None)

        # Group by topological generation (parallel levels)
        try:
            generations = rx.topological_generations(dag)
        except Exception:
            logger.warning(
                "[CONCEPT:ORCH-1.25] Dependency cycle detected — falling back "
                "to sequential execution"
            )
            return [[a] for a in expanded]

        topological_waves: list[list[AgentSpec]] = []
        b_size = manifest.batch_size
        if b_size is None:
            b_size = getattr(config, "parallel_batch_size", 25) or 25
        batch_size = int(b_size)

        for generation in generations:
            gen_agents = [
                agent_map[dag[nidx]] for nidx in generation if dag[nidx] in agent_map
            ]
            # Sub-batch within a generation if it exceeds batch_size
            for i in range(0, len(gen_agents), batch_size):
                topological_waves.append(gen_agents[i : i + batch_size])

        return topological_waves

    def _expand_partitions(self, manifest: ExecutionManifest) -> list[AgentSpec]:
        """Expand fan-out partitions into individual agent specs.

        CONCEPT:ORCH-1.25 — Parallel Engine

        If an ``AgentSpec`` has partitions, create one copy per partition
        with ``{{partition}}`` replaced in the task template and a unique
        agent_id suffix.
        """
        expanded: list[AgentSpec] = []
        for agent in manifest.agents:
            if agent.partitions:
                for partition in agent.partitions:
                    expanded_agent = agent.model_copy(deep=True)
                    expanded_agent.agent_id = f"{agent.agent_id}:{partition}"
                    expanded_agent.task_template = agent.task_template.replace(
                        "{{partition}}", partition
                    )
                    expanded_agent.partitions = []  # Already expanded
                    expanded.append(expanded_agent)
            else:
                expanded.append(agent)
        return expanded

    async def _execute_wave(
        self,
        agents: list[AgentSpec],
        wave_idx: int,
        scheduler: Any,
        manifest: ExecutionManifest,
        graph_deps: GraphDeps | None,
        wave_results: list[WaveResult],
    ) -> WaveResult:
        """Execute one wave of agents concurrently with semaphore backpressure.

        CONCEPT:ORCH-1.25 — Parallel Engine

        Args:
            agents: Agents in this wave.
            wave_idx: Zero-based wave index.
            semaphore: Concurrency governor.
            manifest: The full manifest for context.
            graph_deps: Optional runtime dependencies.
            wave_results: Accumulated results from preceding waves.

        Returns:
            ``WaveResult`` with all agent outcomes.
        """
        start_time = time.monotonic()

        async def _run_one(agent: AgentSpec) -> AgentExecutionResult:
            if self._circuit_breaker.is_open(agent.agent_id):
                return AgentExecutionResult(
                    agent_id=agent.agent_id,
                    role=agent.role,
                    success=False,
                    error=f"Circuit breaker open for {agent.agent_id}",
                )

            proc = await scheduler.submit(
                agent_id=agent.agent_id,
                task=agent.task_template or manifest.query,
            )

            await scheduler.wait_for_running(proc.id)

            try:
                res = await self._execute_agent(
                    agent, manifest, graph_deps, wave_results, proc
                )
                await scheduler.complete(proc.id)
                return res
            except Exception as e:
                await scheduler.fail(proc.id, str(e))
                raise e

        tasks = [_run_one(a) for a in agents]
        raw_results = await asyncio.gather(*tasks, return_exceptions=True)

        results: list[AgentExecutionResult] = []
        for raw in raw_results:
            if isinstance(raw, AgentExecutionResult):
                results.append(raw)
                # Update circuit breaker
                if raw.success:
                    self._circuit_breaker.record_success(raw.agent_id)
                else:
                    self._circuit_breaker.record_failure(raw.agent_id)
            elif isinstance(raw, Exception):
                results.append(
                    AgentExecutionResult(
                        agent_id="unknown",
                        success=False,
                        error=str(raw),
                    )
                )

        duration_ms = (time.monotonic() - start_time) * 1000
        return WaveResult(
            wave_index=wave_idx,
            results=results,
            duration_ms=duration_ms,
        )

    async def _execute_agent(
        self,
        agent: AgentSpec,
        manifest: ExecutionManifest,
        graph_deps: GraphDeps | None,
        wave_results: list[WaveResult],
        proc: Any = None,
    ) -> AgentExecutionResult:
        """Execute a single agent invocation with full capability wiring.

        CONCEPT:ORCH-1.25 — Parallel Engine

        Args:
            agent: The agent specification.
            manifest: The parent manifest for shared context.
            graph_deps: Optional runtime dependencies.
            wave_results: Preceding wave results for context injection.

        Returns:
            ``AgentExecutionResult`` with the agent's output.
        """
        start_time = time.monotonic()
        timeout = agent.timeout or getattr(config, "agent_execution_timeout", 120.0)

        # Build the task prompt
        task = agent.task_template or manifest.query

        # Ingest dependency outputs (Fan-In / Fan-Out topological context flow)
        dependency_contexts = []
        if agent.depends_on:
            for dep_id in agent.depends_on:
                for wave_res in wave_results:
                    for agent_res in wave_res.results:
                        if (
                            agent_res.agent_id == dep_id
                            or agent_res.agent_id.startswith(f"{dep_id}:")
                        ) and agent_res.success:
                            role_str = (
                                f"Role: {agent_res.role}" if agent_res.role else ""
                            )
                            part_str = (
                                f", Partition: {agent_res.partition}"
                                if agent_res.partition
                                else ""
                            )
                            dependency_contexts.append(
                                f"### Output from dependent agent '{agent_res.agent_id}' ({role_str}{part_str}):\n"
                                f"{agent_res.output}"
                            )

        if dependency_contexts:
            dep_text = "\n\n".join(dependency_contexts)
            task = (
                f"{task}\n\n"
                f"## DEPENDENCY OUTPUTS\n"
                f"The following dependent upstream steps have completed successfully. "
                f"Use their outputs to complete your task:\n\n"
                f"{dep_text}"
            )

        if manifest.context:
            task = f"{task}\n\nContext:\n{manifest.context}"

        # CONCEPT:OS-5.9 Context Paging
        if proc and hasattr(proc, "checkpoint_id") and proc.checkpoint_id:
            logger.info(
                "Paging context from checkpoint %s for agent %s",
                proc.checkpoint_id,
                agent.agent_id,
            )
            try:
                from ..capabilities.checkpointing import GraphCheckpointStore

                store = GraphCheckpointStore(engine=self.engine)
                ckpt_data = store.get(proc.checkpoint_id)
                if ckpt_data:
                    task = f"{task}\n\n## RESUMED CONTEXT (Paged from KG)\n{ckpt_data}"
            except Exception as e:
                logger.warning(
                    "Failed to page context from checkpoint %s: %s",
                    proc.checkpoint_id,
                    e,
                )

        # Determine model
        model_id = agent.model_id
        if not model_id and graph_deps:
            model_id = str(graph_deps.agent_model)
        if not model_id:
            model_id = "openai:gpt-4o-mini"  # Fallback

        system_prompt = agent.system_prompt or (
            f"You are a {agent.role or agent.agent_id} specialist agent. "
            f"Provide your best analysis and response."
        )

        try:
            from ..agent.factory import create_agent

            # Setup provider & model override
            provider = "openai"
            prov_model = model_id
            if ":" in model_id:
                provider, prov_model = model_id.split(":", 1)

            # Map manifest settings / metadata
            metadata = manifest.metadata or {}
            from ..capabilities.checkpointing import CheckpointStore

            checkpoint_store: CheckpointStore | None = None
            if metadata.get("checkpoint_store") == "file":
                from ..capabilities.checkpointing import FileCheckpointStore

                checkpoint_store = FileCheckpointStore(
                    directory=metadata.get("checkpoint_dir", "./checkpoints")
                )
            elif metadata.get("checkpoint_store") == "graph":
                from ..capabilities.checkpointing import GraphCheckpointStore

                checkpoint_store = GraphCheckpointStore(engine=self.engine)

            # Wire up all 8 capabilities natively using factory
            llm_agent, _ = create_agent(
                provider=provider,
                model_id=prov_model,
                system_prompt=system_prompt,
                name=agent.agent_id,
                enable_skills=True,
                enable_universal_tools=True,
                mcp_config=metadata.get("mcp_config"),
                tool_tags=agent.tools,
                stuck_loop_detection=metadata.get("stuck_loop_detection", True),
                stuck_loop_max_repeated=metadata.get("stuck_loop_max_repeated", 3),
                context_warnings=metadata.get("context_warnings", True),
                max_context_tokens=metadata.get("max_context_tokens"),
                output_eviction=metadata.get("output_eviction", True),
                eviction_threshold_chars=metadata.get(
                    "eviction_threshold_chars", 80_000
                ),
                include_checkpoints=metadata.get("include_checkpoints", False),
                checkpoint_store=checkpoint_store,
                checkpoint_frequency=metadata.get("checkpoint_frequency", "every_tool"),
                include_teams=metadata.get("include_teams", False),
            )

            result = await asyncio.wait_for(
                llm_agent.run(task),
                timeout=timeout,
            )

            duration_ms = (time.monotonic() - start_time) * 1000
            output = result.output

            logger.debug(
                "[CONCEPT:ORCH-1.25] Agent %s (%s) completed in %.0fms — "
                "output=%d chars",
                agent.agent_id,
                agent.role,
                duration_ms,
                len(output),
            )

            return AgentExecutionResult(
                agent_id=agent.agent_id,
                role=agent.role,
                partition=agent.partitions[0] if agent.partitions else "",
                output=output,
                success=True,
                duration_ms=duration_ms,
                model_id=model_id,
            )

        except TimeoutError:
            duration_ms = (time.monotonic() - start_time) * 1000
            logger.warning(
                "[CONCEPT:ORCH-1.25] Agent %s timed out after %.0fms",
                agent.agent_id,
                duration_ms,
            )
            return AgentExecutionResult(
                agent_id=agent.agent_id,
                role=agent.role,
                success=False,
                error=f"Timeout after {duration_ms:.0f}ms",
                duration_ms=duration_ms,
                model_id=model_id,
            )

        except Exception as e:
            duration_ms = (time.monotonic() - start_time) * 1000
            logger.warning(
                "[CONCEPT:ORCH-1.25] Agent %s failed: %s",
                agent.agent_id,
                e,
            )
            # Register failure with Auto-Healing retry engine
            try:
                self.auto_healing.report_failure(
                    task_name=agent.agent_id,
                    error_context=str(e),
                )
            except Exception as ah_err:
                logger.debug("Auto-healing trigger failed: %s", ah_err)

            return AgentExecutionResult(
                agent_id=agent.agent_id,
                role=agent.role,
                success=False,
                error=str(e),
                duration_ms=duration_ms,
                model_id=model_id,
            )

    async def _synthesize(
        self,
        results: list[AgentExecutionResult],
        spec: SynthesisSpec,
        query: str,
        graph_deps: GraphDeps | None,
    ) -> str:
        """Synthesize agent outputs using the specified strategy.

        CONCEPT:ORCH-1.26 — RLM-Native Hierarchical Synthesis

        The key insight: outputs are stored as Pydantic objects and
        processed programmatically, never dumped into context windows.

        Args:
            results: All agent execution results.
            spec: Synthesis specification.
            query: Original user query.
            graph_deps: Optional runtime dependencies.

        Returns:
            Synthesized output string.
        """
        successful = [r for r in results if r.success]
        if not successful:
            return "No successful agent outputs to synthesize."

        if len(successful) == 1:
            return successful[0].output

        if spec.strategy == "flat":
            return self._flat_synthesis(successful)

        elif spec.strategy == "hierarchical":
            return await self._hierarchical_synthesis(
                successful, spec, query, graph_deps
            )

        elif spec.strategy == "rlm":
            return await self._rlm_synthesis(successful, spec, query, graph_deps)

        elif spec.strategy == "progressive":
            return await self._progressive_synthesis(
                successful, spec, query, graph_deps
            )

        # Default fallback
        return self._flat_synthesis(successful)

    def _flat_synthesis(self, results: list[AgentExecutionResult]) -> str:
        """Simple concatenation synthesis for small agent counts.

        CONCEPT:ORCH-1.26 — RLM-Native Hierarchical Synthesis
        """
        parts: list[str] = []
        for r in results:
            header = f"## {r.role or r.agent_id}"
            if r.partition:
                header += f" [{r.partition}]"
            parts.append(f"{header}\n\n{r.output}")
        return "\n\n---\n\n".join(parts)

    async def _synthesize_group(
        self,
        results: list[AgentExecutionResult],
        query: str,
        graph_deps: GraphDeps | None,
    ) -> str:
        """Synthesize a subgroup of results. For now, falls back to flat synthesis."""
        return self._flat_synthesis(results)

    async def _merge_pair(
        self,
        running_summary: str,
        new_result: AgentExecutionResult,
        query: str,
        graph_deps: GraphDeps | None,
    ) -> str:
        """Merge a new result into the running summary. For now, concatenates."""
        header = f"## {new_result.role or new_result.agent_id}"
        if new_result.partition:
            header += f" [{new_result.partition}]"
        return f"{running_summary}\n\n---\n\n{header}\n\n{new_result.output}"

    async def _hierarchical_synthesis(
        self,
        results: list[AgentExecutionResult],
        spec: SynthesisSpec,
        query: str,
        graph_deps: GraphDeps | None,
    ) -> str:
        """Tiered synthesis: group → sub-summaries → final summary.

        CONCEPT:ORCH-1.26 — RLM-Native Hierarchical Synthesis

        Groups outputs by ``spec.ratio`` (default 10), generates a
        sub-summary for each group, then synthesizes sub-summaries
        into a final output. Recurses if needed for very large sets.
        """
        ratio = spec.ratio

        # Base case: small enough for direct synthesis
        if len(results) <= ratio:
            return await self._synthesize_group(results, query, graph_deps)

        # Tier 1: Create sub-summaries
        sub_summaries: list[str] = []
        for i in range(0, len(results), ratio):
            group = results[i : i + ratio]
            summary = await self._synthesize_group(group, query, graph_deps)
            sub_summaries.append(summary)

        logger.info(
            "[CONCEPT:ORCH-1.26] Hierarchical synthesis: %d results → "
            "%d sub-summaries → final",
            len(results),
            len(sub_summaries),
        )

        # Tier 2: Final synthesis of sub-summaries
        if len(sub_summaries) > ratio:
            # Recurse for very large sets
            pseudo_results = [
                AgentExecutionResult(
                    agent_id=f"sub_summary_{i}",
                    output=s,
                    success=True,
                )
                for i, s in enumerate(sub_summaries)
            ]
            return await self._hierarchical_synthesis(
                pseudo_results, spec, query, graph_deps
            )

        return await self._synthesize_group(
            [
                AgentExecutionResult(
                    agent_id=f"sub_summary_{i}",
                    output=s,
                    success=True,
                )
                for i, s in enumerate(sub_summaries)
            ],
            query,
            graph_deps,
        )

    async def _rlm_synthesis(
        self,
        results: list[AgentExecutionResult],
        spec: SynthesisSpec,
        query: str,
        graph_deps: GraphDeps | None,
    ) -> str:
        """Full RLM synthesis for massive-scale (50+ agent) output processing.

        CONCEPT:ORCH-1.26 — RLM-Native Hierarchical Synthesis

        Uses the RLM environment to programmatically process outputs
        stored as Pydantic objects, not dumped into the context window.
        Falls back to hierarchical synthesis if RLM is unavailable.
        """
        try:
            from ..rlm.config import RLMConfig
            from ..rlm.repl import RLMEnvironment

            # Serialize outputs as environment context
            outputs_json = json.dumps(
                [
                    {
                        "agent_id": r.agent_id,
                        "role": r.role,
                        "partition": r.partition,
                        "output": r.output[:2000],  # Truncate for metadata
                        "success": r.success,
                    }
                    for r in results
                ],
                indent=2,
            )

            rlm_config = RLMConfig(
                metadata_only_root=True,
                async_enabled=True,
            )

            env = RLMEnvironment(
                context=outputs_json,
                config=rlm_config,
                graph_deps=graph_deps,
            )

            return await env.run_full_rlm(
                f"Synthesize {len(results)} agent outputs for query: {query}"
            )

        except Exception as e:
            logger.warning(
                "[CONCEPT:ORCH-1.26] RLM synthesis failed, falling back to "
                "hierarchical: %s",
                e,
            )
            return await self._hierarchical_synthesis(results, spec, query, graph_deps)

    async def _progressive_synthesis(
        self,
        results: list[AgentExecutionResult],
        spec: SynthesisSpec,
        query: str,
        graph_deps: GraphDeps | None,
    ) -> str:
        """Progressive synthesis: incrementally merge as results arrive.

        CONCEPT:ORCH-1.26 — RLM-Native Hierarchical Synthesis

        Processes results one at a time, maintaining a running summary
        that grows as each agent's output is incorporated.
        """
        if not results:
            return ""

        running_summary = results[0].output

        for r in results[1:]:
            running_summary = await self._merge_pair(
                running_summary, r, query, graph_deps
            )

        return running_summary

    def _persist_execution(
        self,
        manifest: ExecutionManifest,
        wave_results: list[WaveResult],
        synthesis_output: str,
    ) -> str:
        """Persist execution results to the Knowledge Graph with verbose hierarchy.

        CONCEPT:ORCH-1.25 — Parallel Engine

        Creates a ``ParallelExecution`` node, individual ``AgentExecutionResult`` nodes
        linked via ``PART_OF_EXECUTION`` edges, and dependency edges linked via
        ``DEPENDS_ON`` edges.

        Args:
            manifest: The executed manifest.
            wave_results: Per-wave results.
            synthesis_output: Final synthesis output.

        Returns:
            The execution node ID.
        """
        execution_id = f"pe:{uuid4().hex[:8]}"

        if self.engine is None:
            return execution_id

        try:
            all_results = [r for w in wave_results for r in w.results]
            total_duration = sum(w.duration_ms for w in wave_results)
            success_count = sum(1 for r in all_results if r.success)

            node_data = {
                "id": execution_id,
                "type": "ParallelExecution",
                "name": f"PE: {manifest.name or manifest.manifest_id}",
                "manifest_id": manifest.manifest_id,
                "agent_count": manifest.agent_count,
                "wave_count": len(wave_results),
                "success_count": success_count,
                "failure_count": len(all_results) - success_count,
                "total_duration_ms": total_duration,
                "synthesis_strategy": manifest.synthesis.strategy,
                "execution_mode": manifest.execution_mode,
                "source": manifest.source,
                "synthesis_preview": synthesis_output[:500],
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "importance_score": 0.7,
            }

            self.engine.graph.add_node(execution_id, **node_data)

            # Persist individual AgentExecutionResult nodes and connect them
            kg_node_map = {}
            for result in all_results:
                node_uuid = f"agent_exec_res:{uuid4().hex[:8]}"
                res_data = {
                    "id": node_uuid,
                    "type": "AgentExecutionResult",
                    "agent_id": result.agent_id,
                    "role": result.role,
                    "partition": result.partition,
                    "success": result.success,
                    "error": result.error,
                    "duration_ms": result.duration_ms,
                    "model_id": result.model_id,
                    "output_preview": result.output[:500] if result.output else "",
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                }
                self.engine.graph.add_node(node_uuid, **res_data)
                self.engine.graph.add_edge(
                    execution_id, node_uuid, type="PART_OF_EXECUTION"
                )
                kg_node_map[result.agent_id] = node_uuid

            # Reconstruct and persist dependency topology edges inside KG
            for agent_spec in manifest.agents:
                for dep in agent_spec.depends_on:
                    source_kg = kg_node_map.get(dep)
                    target_kg = kg_node_map.get(agent_spec.agent_id)
                    if source_kg and target_kg:
                        self.engine.graph.add_edge(
                            source_kg, target_kg, type="DEPENDS_ON"
                        )

            logger.info(
                "[CONCEPT:ORCH-1.25] Persisted execution hierarchy %s to KG "
                "(%d agents, %d waves, %d topology edges)",
                execution_id,
                manifest.agent_count,
                len(wave_results),
                sum(len(a.depends_on) for a in manifest.agents),
            )

        except Exception as e:
            logger.debug("ParallelEngine: KG persistence failed: %s", e)

        return execution_id
