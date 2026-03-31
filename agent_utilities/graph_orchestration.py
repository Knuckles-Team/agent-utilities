#!/usr/bin/python

from __future__ import annotations

import os
import re
import json
import logging
import asyncio


from typing import Any, List, Optional, TYPE_CHECKING
from datetime import datetime

if TYPE_CHECKING:
    pass
from pathlib import Path


from pydantic_ai import Agent



from .config import *
from .workspace import *
from .base_utilities import (
    retrieve_package_name,
    is_loopback_url,
)


from .models import (
    PeriodicTask,
    TaskList,
    Task,
    TaskStatus,
    ProgressLog,
    ProgressEntry,
    SprintContract,
)

tasks: List[PeriodicTask] = []
lock = asyncio.Lock()




logger = logging.getLogger(__name__)


def load_mcp_config() -> dict:
    """Load MCP config from workspace."""
    path = get_workspace_path(CORE_FILES["MCP_CONFIG"])
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {"mcpServers": {}}
    return {"mcpServers": {}}


def save_mcp_config(config: dict):
    """Save MCP config to workspace."""
    path = get_workspace_path(CORE_FILES["MCP_CONFIG"])
    path.write_text(json.dumps(config, indent=2), encoding="utf-8")


from dataclasses import dataclass, field
from uuid import uuid4
from pydantic import BaseModel, Field

try:
    from pydantic_graph import BaseNode, End, Graph

    _PYDANTIC_GRAPH_AVAILABLE = True
except ImportError:
    _PYDANTIC_GRAPH_AVAILABLE = False

DEFAULT_ROUTER_MODEL = os.getenv(
    "GRAPH_ROUTER_MODEL", os.getenv("MODEL_ID", config.model_id)
)
DEFAULT_GRAPH_AGENT_MODEL = os.environ.get("GRAPH_AGENT_MODEL", config.model_id)
DEFAULT_GRAPH_TIMEOUT = int(os.environ.get("GRAPH_TIMEOUT", "30000"))
DEFAULT_ROUTER_PROVIDER = os.getenv(
    "GRAPH_ROUTER_PROVIDER", os.getenv("PROVIDER", "openai")
)
DEFAULT_ROUTER_BASE_URL = os.getenv("GRAPH_ROUTER_BASE_URL", os.getenv("LLM_BASE_URL"))
DEFAULT_ROUTER_API_KEY = os.getenv("GRAPH_ROUTER_API_KEY", os.getenv("LLM_API_KEY"))


def emit_graph_event(eq: Optional[asyncio.Queue], event_type: str, **kwargs):
    """Emit a standardized graph event to the sideband queue."""
    if not eq:
        return
    import time

    try:
        eq.put_nowait(
            {
                "type": "graph-event",
                "event": event_type,
                "timestamp": time.time(),
                **kwargs,
            }
        )
    except Exception as e:
        logger.warning(f"Failed to emit graph event '{event_type}': {e}")


from .tool_filtering import filter_tools_by_tag

from .model_factory import create_model
from .a2a import discover_agents

try:
    from opentelemetry import trace

    tracer = trace.get_tracer("agent-utilities.graph")
except ImportError:
    tracer = None


@dataclass
class GraphDeps:
    """Configuration dependencies passed to graph nodes at runtime."""

    tag_prompts: dict[str, str]
    tag_env_vars: dict[str, str]
    mcp_toolsets: list[Any]
    mcp_url: str = ""
    mcp_config: str = ""
    router_model: str = DEFAULT_ROUTER_MODEL
    agent_model: str = DEFAULT_GRAPH_AGENT_MODEL
    min_confidence: float = 0.6
    sub_agents: dict[str, str | Agent] = field(default_factory=dict)
    provider: str = DEFAULT_PROVIDER
    base_url: Optional[str] = DEFAULT_LLM_BASE_URL
    api_key: Optional[str] = DEFAULT_LLM_API_KEY
    ssl_verify: bool = DEFAULT_SSL_VERIFY
    event_queue: Optional[asyncio.Queue] = None
    request_id: str = ""
    routing_strategy: str = "hybrid"
    enable_llm_validation: bool = False
    project_root: str = ""
    max_parallel_agents: int = 3
    auto_approve_plan: bool = False
    auto_approve_tasks: bool = False
    approval_timeout: float = 0.0


@dataclass
class GraphState:
    """Universal graph state for all agent graph orchestrations."""

    query: str
    """The original user query."""

    validation_feedback: str | None = None
    """Feedback from ValidatorNode to DomainNode for self-correction."""

    routed_domain: str = ""
    """The domain tag this query was routed to."""

    results: dict[str, Any] = field(default_factory=dict)
    """Accumulated results keyed by domain."""

    error: str | None = None
    """Error message if something went wrong."""

    session_id: str = ""
    """Unique session identifier for checkpoint resumption."""

    checkpoint_ts: float = 0.0
    """Timestamp of the last checkpoint."""

    node_history: list[str] = field(default_factory=list)
    """History of executed nodes."""

    retry_count: int = 0
    """Number of retries attempted in ErrorRecoveryNode."""

    parallel_domains: list[str] = field(default_factory=list)
    """List of domains for parallel execution."""

    task_list: TaskList = field(default_factory=TaskList)
    """Phased task list for Project Mode."""

    progress_log: ProgressLog = field(default_factory=ProgressLog)
    """Historical progress log for Project Mode."""

    sprint_contract: SprintContract = field(default_factory=SprintContract)
    """Sprint goals and definition of done."""

    project_root: str = ""
    """Root directory for project artifacts."""

    current_batch_ids: list[str] = field(default_factory=list)
    """IDs of tasks currently executing in parallel."""

    last_git_commit: Optional[str] = None
    """The last recorded git commit hash for this session/project."""

    human_approval_required: bool = False
    """Flag to pause for human intervention."""

    def sync_to_disk(self, artifact_prefix: str = ""):
        """Helper to dump state artifacts for human-in-the-loop inspection."""
        root = self.project_root or os.getcwd()
        if not os.path.exists(root):
            try:
                os.makedirs(root, exist_ok=True)
            except Exception:
                return

        mappings = {
            "tasks.json": self.task_list,
            "progress.json": self.progress_log,
            "sprint.json": self.sprint_contract,
        }
        for filename, model in mappings.items():
            path = os.path.join(root, f"{artifact_prefix}{filename}")
            try:
                with open(path, "w") as f:
                    f.write(model.model_dump_json(indent=2))
            except Exception as e:
                logger.warning(f"Failed to sync artifact {filename}: {e}")

    def load_from_disk(self):
        """Loads existing JSON artifacts from project root."""
        root = self.project_root or os.getcwd()
        mappings = {
            "tasks.json": ("task_list", TaskList),
            "progress.json": ("progress_log", ProgressLog),
            "sprint.json": ("sprint_contract", SprintContract),
        }
        loaded = False
        for filename, (attr, model_cls) in mappings.items():
            path = os.path.join(root, filename)
            if os.path.exists(path):
                try:
                    with open(path, "r") as f:
                        data = f.read()
                        if data.strip():
                            setattr(self, attr, model_cls.model_validate_json(data))
                            loaded = True
                    logger.info(f"Loaded {filename} for project state.")
                except Exception as e:
                    logger.warning(f"Could not load {filename}: {e}")
        return loaded


class DomainChoice(BaseModel):
    """Structured output from the router LLM."""

    domain: str = Field(description="The domain tag to route to")
    confidence: float = Field(ge=0, le=1, description="Routing confidence 0-1")
    reasoning: str = Field(description="Brief reasoning for the classification")


class MultiDomainChoice(BaseModel):
    """Structured output for multi-domain routing."""

    domains: list[str] = Field(description="List of domain tags to route to")
    project_mode: bool = Field(
        False,
        description="Whether the query warrants a complex, multi-stage project with planning and decomposition",
    )
    reasoning: str = Field(
        description="Brief reasoning for the multi-classification and mode choice"
    )


class ValidationResult(BaseModel):
    """Structured output for result validation."""

    is_valid: bool = Field(
        description="True if the result is high quality and accurate"
    )
    feedback: Optional[str] = Field(
        None, description="Detailed feedback if invalid, explaining what to improve"
    )
    score: float = Field(ge=0, le=1, description="Quality score from 0 to 1")


_RouterNodeBase = (
    BaseNode[GraphState, GraphDeps, dict] if _PYDANTIC_GRAPH_AVAILABLE else object
)

_DomainNodeBase = (
    BaseNode[GraphState, GraphDeps, dict] if _PYDANTIC_GRAPH_AVAILABLE else object
)


@dataclass
class ErrorRecoveryNode(_RouterNodeBase):
    """Handles transient and permanent errors during graph execution.

    Implements a simple retry backoff for transient failures (e.g. rate limits, timeouts).
    """

    async def run(self, ctx) -> "RouterNode | End[dict]":
        error_msg = str(ctx.state.error).lower()

        is_transient = any(
            phrase in error_msg
            for phrase in [
                "timeout",
                "rate limit",
                "connection",
                "502",
                "503",
                "504",
                "unavailable",
                "network",
            ]
        )

        if is_transient and ctx.state.retry_count < 3:
            ctx.state.retry_count += 1
            logger.warning(
                f"Transient error detected ('{ctx.state.error}'). Retrying {ctx.state.retry_count}/3 in 2 seconds..."
            )
            await asyncio.sleep(2**ctx.state.retry_count)
            return RouterNode()

        logger.error(f"Permanent error or max retries reached: {ctx.state.error}")
        return End(
            {
                "error": ctx.state.error,
                "domain": ctx.state.routed_domain,
                "results": ctx.state.results,
            }
        )


@dataclass
class ResumeNode(_RouterNodeBase):
    """Entrypoint for resuming a graph from a checkpoint state.

    Determines next action based on node_history or current state variables.
    """

    async def run(self, ctx) -> "RouterNode | DomainNode | End[dict]":
        logger.info(f"Resuming workflow session: {ctx.state.session_id}")
        if ctx.state.error:
            return ErrorRecoveryNode()
        if ctx.state.routed_domain:
            return DomainNode()
        return RouterNode()


@dataclass
class RouterNode(_RouterNodeBase):
    """Classifies an incoming query into one of the valid domain tags.

    Uses a lightweight LLM for fast, cheap classification.
    Returns a DomainNode on success, or End with an error if unroutable.
    """

    min_confidence: float = 0.6
    """Minimum confidence threshold for routing. Kept for backwards compatibility."""

    def _rule_based_route_multi(self, query: str, labels: dict) -> list[str]:
        """Simple keyword-based routing for multiple matches with plural/singular awareness."""
        matches = []

        query_lower = query.lower()
        for label in labels:
            label_lower = label.lower()

            if re.search(rf"\b{label_lower}\b", query_lower):
                logger.debug(
                    f"_rule_based_route_multi: Exact boundary match for '{label}'"
                )
                matches.append(label)
                continue

            if label_lower.endswith("s"):
                alt = label_lower[:-1]
                if len(alt) > 3 and re.search(rf"\b{alt}\b", query_lower):
                    logger.debug(
                        f"_rule_based_route_multi: Singular match for plural label '{label}': {alt}"
                    )
                    matches.append(label)
                    continue
            else:
                alt = label_lower + "s"
                if re.search(rf"\b{alt}\b", query_lower):
                    logger.debug(
                        f"_rule_based_route_multi: Plural match for singular label '{label}': {alt}"
                    )
                    matches.append(label)
                    continue

            if label_lower.endswith("y"):
                alt = label_lower[:-1] + "ies"
                if re.search(rf"\b{alt}\b", query_lower):
                    logger.debug(
                        f"_rule_based_route_multi: Plural IES match for label '{label}': {alt}"
                    )
                    matches.append(label)
                    continue
            elif label_lower.endswith("ies"):
                alt = label_lower[:-3] + "y"
                if re.search(rf"\b{alt}\b", query_lower):
                    logger.debug(
                        f"_rule_based_route_multi: Singular Y match for label '{label}': {alt}"
                    )
                    matches.append(label)
                    continue

        return matches

    def _rule_based_route(self, query: str, labels: dict) -> str | None:
        matches = self._rule_based_route_multi(query, labels)
        return matches[0] if matches else None

    async def run(
        self, ctx
    ) -> "DomainNode | ParallelDomainNode | ErrorRecoveryNode | End[dict]":
        deps = ctx.deps
        import time

        logger.info(
            f"RouterNode classification started for query: '{ctx.state.query[:50]}'"
        )
        if deps.event_queue:
            try:
                deps.event_queue.put_nowait(
                    {
                        "type": "graph-event",
                        "event": "routing_started",
                        "query": ctx.state.query,
                        "timestamp": datetime.now().timestamp(),
                        "available_domains": list(ctx.deps.tag_prompts.keys()),
                    }
                )
            except Exception as e:
                logger.warning(f"Failed to put routing_started event: {e}")

        query_lower = ctx.state.query.lower()
        logger.debug(
            f"RouterNode: Available domain tags: {list(deps.tag_prompts.keys())}"
        )
        if hasattr(self, "_rule_based_route_multi"):
            matched = self._rule_based_route_multi(query_lower, deps.tag_prompts)
            if matched:
                logger.info(f"RouterNode: Rule-based routing matched: {matched}")
                if deps.event_queue:
                    try:
                        deps.event_queue.put_nowait(
                            {
                                "type": "graph-event",
                                "event": "routing_completed",
                                "domains": matched,
                                "reasoning": "Rule-based keyword match",
                                "timestamp": datetime.now().timestamp(),
                            }
                        )
                    except:
                        pass

                if len(matched) > 1:
                    ctx.state.parallel_domains = matched
                    return ParallelDomainNode()
                ctx.state.routed_domain = matched[0]
                return DomainNode()

        from .agent_utilities import create_model

        model = create_model(
            provider=deps.provider,
            model_id=(
                deps.router_model.split(":")[-1]
                if deps.router_model and ":" in deps.router_model
                else deps.router_model
            ),
            base_url=deps.base_url,
            api_key=deps.api_key,
            ssl_verify=deps.ssl_verify,
        )

        try:

            router_agent = Agent(
                model=model,
                output_type=MultiDomainChoice,
                instructions=(
                    f"You are a domain classifier. Classify the user query into ONE or MORE "
                    f"of these domains: {', '.join(deps.tag_prompts.keys())}.\n"
                    f"Return a list of domains, and brief reasoning.\n"
                    f"Set project_mode=True if the query is a complex request that warrants a multi-stage project "
                    f"(e.g. 'Implement a new feature', 'Refactor the system', 'Build a complex component').\n"
                    f"If the query spans multiple domains (e.g. 'Check the git logs AND search the web'), "
                    f"return all relevant domains."
                ),
            )
            logger.debug(
                f"RouterNode: Sending request to LLM (Model: {deps.router_model})..."
            )
            start_time = time.time()
            try:

                result = await asyncio.wait_for(
                    router_agent.run(ctx.state.query), timeout=45.0
                )
                end_time = time.time()
                choice = result.output
                logger.info(
                    f"RouterNode: LLM responded in {end_time - start_time:.2f}s with {len(choice.domains)} domains. Reasoning: {choice.reasoning}"
                )
            except (asyncio.TimeoutError, Exception) as run_error:
                error_type = (
                    "Timeout"
                    if isinstance(run_error, asyncio.TimeoutError)
                    else "Error"
                )
                logger.error(
                    f"RouterNode: router_agent.run {error_type.lower()}: {run_error}"
                )

                if deps.event_queue:
                    try:
                        deps.event_queue.put_nowait(
                            {
                                "type": "graph-event",
                                "event": "routing_failed",
                                "error": f"Classification {error_type.lower()}: {run_error}",
                                "timestamp": time.time(),
                            }
                        )
                    except:
                        pass

                ctx.state.error = f"Router {error_type.lower()}: {run_error}"
                return ErrorRecoveryNode()

            if deps.event_queue:
                try:
                    deps.event_queue.put_nowait(
                        {
                            "type": "graph-event",
                            "event": "routing_completed",
                            "domains": choice.domains,
                            "reasoning": choice.reasoning,
                            "timestamp": datetime.now().timestamp(),
                        }
                    )
                except Exception as e:
                    logger.warning(f"Failed to put routing_completed event: {e}")

            if choice.project_mode:
                logger.info(
                    f"RouterNode: Project Mode detected. Reasoning: {choice.reasoning}"
                )
                return PlannerNode()

            if len(choice.domains) > 1:
                ctx.state.parallel_domains = [
                    d for d in choice.domains if d in deps.tag_prompts
                ]
                return ParallelDomainNode()
            elif len(choice.domains) == 1:
                ctx.state.routed_domain = choice.domains[0]
                return DomainNode()
            else:
                ctx.state.error = "No domains matched"
                return ErrorRecoveryNode()

        except Exception as e:
            logger.error(f"Router classification failed: {e}. Falling back to rules.")

            query_lower = ctx.state.query.lower()
            if hasattr(self, "_rule_based_route_multi"):
                matched = self._rule_based_route_multi(query_lower, deps.tag_prompts)
                if matched:
                    logger.info(f"Fallback rules matched: {matched}")
                    if len(matched) > 1:
                        ctx.state.parallel_domains = matched
                        return ParallelDomainNode()
                    ctx.state.routed_domain = matched[0]
                    return DomainNode()

            ctx.state.error = f"Router failed and no rules matched: {e}"
            return ErrorRecoveryNode()


@dataclass
class DomainNode(_DomainNodeBase):
    """Executes a query against a specific domain's MCP tools.

    Uses env-var gating to restrict the MCP server to only register
    the tools belonging to the routed domain tag. Works with both
    stdio and HTTP-based MCP servers.
    """

    async def run(self, ctx) -> "ValidatorNode | ErrorRecoveryNode | End[Any]":
        domain = ctx.state.routed_domain
        result = await self.execute_domain(ctx, domain)

        if isinstance(result, (ErrorRecoveryNode, End)):
            return result
        return DomainValidatorNode()

    async def execute_domain(self, ctx, domain: str):
        deps = ctx.deps
        domain_prompt = deps.tag_prompts.get(
            domain, f"You are a specialized assistant for the '{domain}' domain."
        )

        logger.info(f"DomainNode executing for domain='{domain}'")

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
                logger.info(
                    f"DomainNode: Injected filtered toolset for domain '{domain}'"
                )

            sub_agent_target = deps.sub_agents.get(domain)

            if sub_agent_target:
                logger.info(
                    f"DomainNode: Delegating to sub-agent for domain '{domain}'"
                )
                try:
                    target = sub_agent_target

                    if isinstance(target, dict) and "tags" in target:
                        logger.info(
                            f"DomainNode: Creating dynamic skill agent for '{domain}'"
                        )
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

                        import importlib

                        module = importlib.import_module(f"{target}.agent_server")
                        if hasattr(module, "agent_template"):
                            target = module.agent_template()
                        else:
                            raise AttributeError(
                                f"Package {target} is missing agent_template()"
                            )

                    if isinstance(target, tuple) and len(target) == 2:

                        sub_graph, sub_config = target
                        logger.info(
                            f"DomainNode: Delegating to Sub-Graph for domain '{domain}'"
                        )
                        res = await run_graph(
                            graph=sub_graph,
                            config=sub_config,
                            query=ctx.state.query,
                            eq=deps.event_queue,
                        )
                        output = res.get("results") or res.get("error")
                        mermaid = res.get("mermaid")
                        if mermaid and isinstance(output, str):
                            output = f"{mermaid}\n\n{output}"
                    else:

                        logger.info(
                            f"DomainNode: Delegating to Flat Agent for domain '{domain}'"
                        )
                        emit_graph_event(
                            deps.event_queue,
                            "subagent_started",
                            domain=domain,
                            type="flat",
                        )

                        async with target.run_stream(ctx.state.query) as stream:

                            async for message, last in stream.stream_messages():

                                emit_graph_event(
                                    deps.event_queue,
                                    "subagent_thought",
                                    domain=domain,
                                    message=str(message),
                                )
                            res = await stream.get_output()

                        output = getattr(res, "output", None) or getattr(
                            res, "data", res
                        )

                    ctx.state.results[domain] = str(output)
                    logger.info(
                        f"DomainNode: Delegation completed for domain '{domain}'"
                    )
                except Exception as e:
                    logger.error(f"DomainNode delegation error for '{domain}': {e}")
                    ctx.state.results[domain] = f"Delegation Error: {e}"
            else:

                query = ctx.state.query
                if ctx.state.validation_feedback:
                    query = f"{query}\n\n[SELF-CORRECTION FEEDBACK]: {ctx.state.validation_feedback}"
                    logger.info(
                        f"DomainNode: Appending self-correction feedback for '{domain}'"
                    )

                logger.info(
                    f"DomainNode: Running standard MCP sub-agent for domain '{domain}'"
                )
                from .agent_utilities import create_agent

                sub_agent = create_agent(
                    provider=deps.provider,
                    model_id=deps.agent_model,
                    base_url=deps.base_url,
                    api_key=deps.api_key,
                    mcp_url=None,
                    mcp_config=None,
                    mcp_toolsets=deps.mcp_toolsets,
                    tool_tags=[domain],
                    name=f"Graph-{domain}",
                    system_prompt=domain_prompt,
                    enable_skills=False,
                    enable_universal_tools=False,
                    ssl_verify=deps.ssl_verify,
                    tool_guard_mode="off",
                )

                import time

                eq = deps.event_queue

                logger.info(f"DomainNode execution started for '{domain}'")
                if eq:
                    try:
                        eq.put_nowait(
                            {
                                "type": "graph-event",
                                "event": "subagent_started",
                                "domain": domain,
                                "timestamp": time.time(),
                            }
                        )
                    except Exception as e:
                        logger.warning(f"Failed to put subagent_started event: {e}")

                    from .models import AgentDeps

                    sub_deps = AgentDeps(
                        workspace_path=getattr(deps, "workspace_path", Path.cwd()),
                        graph_event_queue=eq,
                        elicitation_queue=getattr(deps, "elicitation_queue", None),
                    )

                    async def _run_stream():
                        try:

                            async with sub_agent.run_stream(
                                query, deps=sub_deps
                            ) as stream:
                                async for message, last in stream.stream_messages():

                                    emit_graph_event(
                                        eq,
                                        "subagent_event",
                                        domain=domain,
                                        message=str(message),
                                    )

                                    if hasattr(message, "content") and isinstance(
                                        message.content, str
                                    ):
                                        emit_graph_event(
                                            eq,
                                            "subagent_text",
                                            domain=domain,
                                            text=message.content,
                                        )

                                return await stream.get_output()
                        except asyncio.TimeoutError:
                            logger.error(
                                f"Subagent stream timed out for domain '{domain}'"
                            )
                            raise
                        except Exception as e:
                            logger.error(
                                f"Subagent stream failed for domain '{domain}': {e}"
                            )
                            raise

                    try:
                        result = await asyncio.wait_for(
                            _run_stream(),
                            timeout=DEFAULT_GRAPH_TIMEOUT,
                        )
                    except asyncio.TimeoutError:
                        logger.error(f"DomainNode: sub-agent timeout for '{domain}'")
                        ctx.state.results[domain] = "Error: Sub-agent timed out."
                        return ErrorRecoveryNode()
                else:
                    logger.info(
                        f"DomainNode: Running sub-agent for domain '{domain}' with query: {ctx.state.query}"
                    )
                    try:
                        result = await asyncio.wait_for(
                            sub_agent.run(ctx.state.query),
                            timeout=DEFAULT_GRAPH_TIMEOUT,
                        )
                    except asyncio.TimeoutError:
                        logger.error(f"DomainNode: sub-agent timeout for '{domain}'")
                        ctx.state.results[domain] = "Error: Sub-agent timed out."
                        return ErrorRecoveryNode()

                logger.info(
                    f"DomainNode: Sub-agent run completed for domain '{domain}'"
                )

                from pydantic_ai import DeferredToolRequests

                output = getattr(result, "output", None) or getattr(
                    result, "data", result
                )

                if isinstance(output, DeferredToolRequests):
                    logger.info(
                        f"DomainNode: Human approval required for domain '{domain}'"
                    )
                    ctx.state.human_approval_required = True

                    ctx.state.results[domain] = output

                    if eq:
                        try:

                            all_calls = (getattr(output, "calls", []) or []) + (
                                getattr(output, "approvals", []) or []
                            )
                            eq.put_nowait(
                                {
                                    "type": "graph-event",
                                    "event": "approval_required",
                                    "domain": domain,
                                    "tool_calls": [
                                        (
                                            tc.model_dump()
                                            if hasattr(tc, "model_dump")
                                            else str(tc)
                                        )
                                        for tc in all_calls
                                    ],
                                    "timestamp": time.time(),
                                }
                            )
                        except Exception as e:
                            logger.warning(
                                f"Failed to put approval_required event: {e}"
                            )

                    return End(output)
                else:
                    ctx.state.results[domain] = str(output)

                if eq:
                    try:
                        eq.put_nowait(
                            {
                                "type": "graph-event",
                                "event": "subagent_completed",
                                "domain": domain,
                                "has_result": bool(output),
                                "timestamp": time.time(),
                            }
                        )
                    except Exception as e:
                        logger.warning(f"Failed to put subagent_completed event: {e}")
            logger.info(f"DomainNode completed for '{domain}'")

        except Exception as e:
            import traceback

            logger.error(
                f"DomainNode error for '{domain}': {e}\n{traceback.format_exc()}"
            )
            ctx.state.error = f"Domain failed: {e}"
            ctx.state.results[domain] = f"Error: {e}"
            return ErrorRecoveryNode()

        finally:

            for env_var, value in original_env.items():
                if value is None:
                    os.environ.pop(env_var, None)
                else:
                    os.environ[env_var] = value

        return None


_routing_cache: dict[str, DomainChoice] = {}


@dataclass
class HybridRouterNode(RouterNode):
    """Classifies an incoming query into a domain tag, using rules/regex first, caching, then falling back to LLM."""

    async def run(
        self, ctx
    ) -> "DomainNode | ParallelDomainNode | ErrorRecoveryNode | End[dict]":
        query_normalized = ctx.state.query.strip().lower()

        import time

        strategy = ctx.deps.routing_strategy.lower()
        if strategy == "llm":

            return await super().run(ctx)

        if strategy == "rules":
            matched_domains = self._rule_based_route_multi(
                query_normalized, ctx.deps.tag_prompts
            )
            if matched_domains:

                pass

        if ctx.state.load_from_disk():
            if ctx.state.task_list.phases:
                logger.info("HybridRouterNode: Resuming active project from disk.")
                return ParallelExecutionNode()

        if strategy != "llm" and query_normalized in _routing_cache:
            if ctx.deps.event_queue:
                try:
                    ctx.deps.event_queue.put_nowait(
                        {
                            "type": "graph-event",
                            "event": "routing_started",
                            "timestamp": datetime.now().timestamp(),
                            "available_domains": list(ctx.deps.tag_prompts.keys()),
                        }
                    )
                except Exception as e:
                    logger.warning(f"Failed to put routing_started event: {e}")
            choice = _routing_cache[query_normalized]
            logger.info(f"Router cache hit for '{choice.domain}'")
            ctx.state.routed_domain = choice.domain
            if ctx.deps.event_queue:
                try:
                    ctx.deps.event_queue.put_nowait(
                        {
                            "type": "graph-event",
                            "event": "routing_completed",
                            "domain": choice.domain,
                            "confidence": choice.confidence,
                            "reasoning": choice.reasoning,
                            "timestamp": datetime.now().timestamp(),
                        }
                    )
                except Exception as e:
                    logger.warning(f"Failed to put routing_completed event: {e}")
            return DomainNode()

        matched_domains = self._rule_based_route_multi(
            query_normalized, ctx.deps.tag_prompts
        )
        if matched_domains:
            if ctx.deps.event_queue:
                try:
                    ctx.deps.event_queue.put_nowait(
                        {
                            "type": "graph-event",
                            "event": "routing_started",
                            "timestamp": time.time(),
                            "available_domains": list(ctx.deps.tag_prompts.keys()),
                        }
                    )
                except Exception as e:
                    logger.warning(f"Failed to put routing_started event: {e}")

            if len(matched_domains) > 1:
                logger.info(f"Multi-domain rule match: {matched_domains}")
                ctx.state.parallel_domains = matched_domains
                return ParallelDomainNode()

            domain = matched_domains[0]
            logger.info(f"Rule-based routing matched '{domain}'")
            ctx.state.routed_domain = domain
            _routing_cache[query_normalized] = DomainChoice(
                domain=domain, confidence=1.0, reasoning="Rule match"
            )
            if ctx.deps.event_queue:
                emit_graph_event(
                    ctx.deps.event_queue,
                    "routing_completed",
                    domain=domain,
                    confidence=1.0,
                    reasoning="Rule match",
                )
            return DomainNode()

        logger.info(f"HybridRouterNode fallback to LLM for: '{query_normalized[:50]}'")
        return await super().run(ctx)

    def _rule_based_route(self, query: str, labels: dict) -> str | None:
        matches = self._rule_based_route_multi(query, labels)
        return matches[0] if matches else None


@dataclass
class ParallelDomainNode(_RouterNodeBase):
    """Executes multiple DomainNodes in parallel."""

    async def run(self, ctx) -> "ResultMergerNode | ErrorRecoveryNode":

        domains = ctx.state.parallel_domains
        logger.info(
            f"🚀 ParallelDomainNode: Starting {len(domains)} tasks for {domains}"
        )

        emit_graph_event(
            ctx.deps.event_queue, "parallel_execution_started", domains=domains
        )

        async def run_single_domain(domain: str):
            node = DomainNode()
            try:
                result = await node.execute_domain(ctx, domain)

                if isinstance(result, End):
                    return result
                return True
            except Exception as e:
                logger.error(f"Parallel execution failed for domain '{domain}': {e}")
                ctx.state.results[domain] = f"Error: {e}"
                return False

        results = await asyncio.gather(*(run_single_domain(d) for d in domains))

        for res in results:
            if isinstance(res, End):
                logger.info(
                    "ParallelDomainNode: One or more sub-agents require human approval. Pausing."
                )
                return res

        emit_graph_event(ctx.deps.event_queue, "parallel_execution_completed")

        return ResultMergerNode()


@dataclass
class ResultMergerNode(_RouterNodeBase):
    """Merges parallel results into a final response."""

    async def run(self, ctx) -> "DomainValidatorNode":
        logger.info("Merging parallel results...")

        combined = {}
        for domain, result in ctx.state.results.items():
            if isinstance(result, str):
                try:

                    data = json.loads(result)
                    combined[domain] = data
                except (json.JSONDecodeError, TypeError):
                    combined[domain] = result
            else:
                combined[domain] = result

        ctx.state.results["combined_summary"] = combined
        return DomainValidatorNode()


@dataclass
class DomainValidatorNode(_DomainNodeBase):
    """Checks the quality of the 'DomainNode' execution before ending the graph.
    Allows for iterative refinement loops.
    """

    async def run(self, ctx) -> "DomainNode | End[dict] | ErrorRecoveryNode":
        domain = ctx.state.routed_domain
        result_text = ctx.state.results.get(domain, "")
        deps = ctx.deps

        if "Delegation Error:" in result_text or "Error:" in result_text:
            return End(
                {
                    "domain": domain,
                    "results": ctx.state.results,
                    "error": ctx.state.error,
                }
            )

        if deps.enable_llm_validation and ctx.state.retry_count < 2:
            logger.info(
                f"ValidatorNode: Performing LLM-based validation for '{domain}'"
            )

            validator_model = create_model(
                provider=deps.provider,
                model_id=deps.router_model,
                base_url=deps.base_url,
                api_key=deps.api_key,
                ssl_verify=deps.ssl_verify,
            )

            validator_agent = Agent(
                model=validator_model,
                output_type=ValidationResult,
                instructions=(
                    f"You are a quality assurance expert. Evaluate the output of the '{domain}' agent.\n"
                    f"Original Query: {ctx.state.query}\n"
                    f"Agent Result: {result_text}\n"
                    f"Determine if the result accurately and comprehensively addresses the query.\n"
                    f"If the result is incomplete, hallucinated, or has errors, set is_valid=False and provide feedback."
                ),
            )

            try:
                val_res = await validator_agent.run("Evaluate the result.")
                quality = val_res.output

                if not quality.is_valid:
                    logger.warning(
                        f"ValidatorNode: LLM rejected output for '{domain}' (Score: {quality.score}). Feedback: {quality.feedback}"
                    )
                    emit_graph_event(
                        ctx.deps.event_queue,
                        "validation_failed",
                        domain=domain,
                        score=quality.score,
                        feedback=quality.feedback,
                    )
                    ctx.state.retry_count += 1
                    ctx.state.validation_feedback = quality.feedback
                    return DomainNode()

                emit_graph_event(
                    ctx.deps.event_queue,
                    "validation_passed",
                    domain=domain,
                    score=quality.score,
                )
                logger.info(
                    f"ValidatorNode: LLM approved output for '{domain}' with score {quality.score}"
                )
                return End(
                    {
                        "domain": domain,
                        "results": ctx.state.results,
                        "error": ctx.state.error,
                        "validation_score": quality.score,
                    }
                )
            except Exception as e:
                logger.error(
                    f"ValidatorNode: LLM validation failed: {e}. Falling back to default heuristics."
                )

        if len(result_text) < 10 and ctx.state.retry_count < 2:
            feedback = "The output was too short. Please provide a more detailed and comprehensive response."
            logger.warning(
                f"ValidatorNode rejected output for '{domain}': {feedback}. Retrying..."
            )
            ctx.state.retry_count += 1
            ctx.state.validation_feedback = feedback
            return DomainNode()

        return End(
            {
                "domain": domain,
                "results": ctx.state.results,
                "error": ctx.state.error,
            }
        )


def build_tag_env_map(tag_names: list[str]) -> dict[str, str]:
    """Build a tag→env_var mapping following the standard convention.

    Standard convention: tag "incidents" → env var "INCIDENTSTOOL"
    (upper-cased tag + "TOOL" suffix).

    Args:
        tag_names: List of domain tag names.

    Returns:
        Dict mapping tag name → env var name.
    """
    result = {}
    for tag in tag_names:
        env_var = tag.upper().replace("-", "_") + "TOOL"
        result[tag] = env_var
    return result


def create_master_graph(
    name: str = "MasterGraph",
    include_agents: list[str] | None = None,
    exclude_agents: list[str] | None = None,
    skill_agents: dict[str, dict] | None = None,
    **kwargs,
) -> tuple["Graph", dict]:
    """Factory to create a master orchestrator graph that discovers and routes to sub-agents.

    Args:
        name: Name of the master graph.
        include_agents: Optional list of specific agent package names to include.
        exclude_agents: Optional list of agent package names to ignore.
        skill_agents: Optional dict of specialized skill agents (Tag -> Config).
        **kwargs: Forwarded to create_graph_agent.

    Returns:
        tuple: (Graph, config_dict)
    """
    agents = discover_agents(
        include_packages=include_agents, exclude_packages=exclude_agents
    )

    tag_prompts = {
        name: f"Specialized agent for {package_name}"
        for name, package_name in agents.items()
    }

    _skill_agents = skill_agents or {}
    for tag, config in _skill_agents.items():
        if tag not in tag_prompts:
            tag_prompts[tag] = config.get(
                "description", f"Specialized skill agent for {tag}"
            )

    sub_agents = {name: package_name for name, package_name in agents.items()}
    for tag in _skill_agents.keys():
        if tag not in sub_agents:
            sub_agents[tag] = tag

    return create_graph_agent(
        tag_prompts=tag_prompts,
        name=name,
        sub_agents=sub_agents,
        **kwargs,
    )


def create_graph_agent(
    tag_prompts: dict[str, str],
    tag_env_vars: dict[str, str] | None = None,
    mcp_url: str | None = None,
    mcp_config: str | None = None,
    name: str = "GraphAgent",
    router_model: str = DEFAULT_ROUTER_MODEL,
    agent_model: str = DEFAULT_GRAPH_AGENT_MODEL,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    sub_agents: dict[str, str | Agent] | None = None,
    mcp_toolsets: list[Any] | None = None,
    routing_strategy: str = DEFAULT_ROUTING_STRATEGY,
    custom_nodes: list[Any] | None = None,
    disabled_nodes: list[Any] | None = None,
    **kwargs,
) -> tuple["Graph", dict]:
    """Factory to create a router-led graph assistant.

    Args:
        tag_prompts: Dict of domain tags → intent description prompts.
        tag_env_vars: Dict of domain tags → env var gating names.
        mcp_url: Optional base MCP URL for all nodes.
        mcp_config: Optional path to JSON MCP config.
        name: Name of the graph.
        router_model: Model for the router node.
        agent_model: Model for the domain nodes.
        min_confidence: Confidence threshold for routing.
        sub_agents: Dict of domain tags → sub-agent package name or instance.
        mcp_toolsets: Optional list of pre-instantiated toolsets (e.g. FastMCP).
        routing_strategy: Strategy for routing (hybrid, rules, llm).

    Returns:
        Graph and config dict.
    """
    _sub_agents = sub_agents or {}
    """Create a pydantic-graph based agent from a tag→prompt mapping.

    This is the graph equivalent of create_agent(). Consumer packages
    provide a tag→prompt dict and an MCP URL; this function builds the
    full Graph with RouterNode and DomainNode.

    Args:
        tag_prompts: Maps domain tag → system prompt for that domain's sub-agent.
        tag_env_vars: Maps domain tag → env var name that toggles that tool category.
                      If None, auto-generated via build_tag_env_map().
        mcp_url: URL of the MCP server to connect domain nodes to.
        name: Name for the graph.
        router_model: Model for the router node (cheap/fast recommended).
        agent_model: Model for domain executor nodes.
        min_confidence: Minimum confidence threshold for routing.

    Returns:
        (Graph, config_dict) — the graph and its runtime configuration.
    """
    if not _PYDANTIC_GRAPH_AVAILABLE:
        raise ImportError(
            "pydantic-graph is required for graph agents. "
            "Install with: pip install 'agent-utilities[agent]'"
        )

    if tag_env_vars is None:
        tag_env_vars = build_tag_env_map(list(tag_prompts.keys()))

    default_nodes = {
        RouterNode,
        DomainNode,
        HybridRouterNode,
        DomainValidatorNode,
        ErrorRecoveryNode,
        ResumeNode,
        ParallelDomainNode,
        ResultMergerNode,
        PlannerNode,
        ParallelExecutionNode,
        ProjectValidatorNode,
    }

    if disabled_nodes:
        default_nodes = {n for n in default_nodes if n not in disabled_nodes}

    all_nodes = list(default_nodes)
    if custom_nodes:

        custom_names = {n.__name__ for n in custom_nodes}
        all_nodes = [n for n in all_nodes if n.__name__ not in custom_names]
        all_nodes.extend(custom_nodes)

    graph = Graph(
        nodes=tuple(all_nodes),
        name=name,
    )
    _mcp_toolsets = list(mcp_toolsets) if mcp_toolsets else []

    if DEFAULT_VALIDATION_MODE:
        if mcp_url:
            logger.info(f"VALIDATION_MODE: Skipping MCP URL connection to {mcp_url}")
        if mcp_config:
            logger.info(
                f"VALIDATION_MODE: Skipping MCP config loading from {mcp_config}"
            )
    else:
        if mcp_url:
            import httpx
            from pydantic_ai.mcp import MCPServerSSE, MCPServerStreamableHTTP

            if is_loopback_url(
                mcp_url, kwargs.get("current_host"), kwargs.get("current_port")
            ):
                logger.warning(
                    f"Loopback Guard: Skipping self-referential MCP connection to {mcp_url}"
                )
            elif mcp_url.lower().endswith("/sse"):
                server = MCPServerSSE(
                    mcp_url,
                    http_client=httpx.AsyncClient(
                        verify=kwargs.get("ssl_verify", DEFAULT_SSL_VERIFY), timeout=60
                    ),
                )
            else:
                server = MCPServerStreamableHTTP(
                    mcp_url,
                    http_client=httpx.AsyncClient(
                        verify=kwargs.get("ssl_verify", DEFAULT_SSL_VERIFY), timeout=60
                    ),
                )
            if not is_loopback_url(
                mcp_url, kwargs.get("current_host"), kwargs.get("current_port")
            ):
                _mcp_toolsets.append(server)

        if mcp_config:
            from pydantic_ai.mcp import load_mcp_servers

            try:

                if not os.path.isabs(mcp_config) and "/" not in mcp_config:
                    ws_config = get_workspace_path(mcp_config)
                    if ws_config.exists():
                        mcp_config = str(ws_config)
                    else:
                        local_config = Path.cwd() / mcp_config
                        if local_config.exists():
                            mcp_config = str(local_config)
                        else:
                            pkg = retrieve_package_name()
                            if pkg and pkg != "agent_utilities":
                                local_pkg_config = (
                                    Path.cwd() / pkg / "agent_data" / mcp_config
                                )
                                if local_pkg_config.exists():
                                    mcp_config = str(local_pkg_config)
                                else:
                                    local_pkg_root = Path.cwd() / pkg / mcp_config
                                    if local_pkg_root.exists():
                                        mcp_config = str(local_pkg_root)

                mcp_loaded = load_mcp_servers(mcp_config)
                for s in mcp_loaded:
                    if hasattr(s, "http_client"):
                        s.http_client = httpx.AsyncClient(
                            verify=kwargs.get("ssl_verify", DEFAULT_SSL_VERIFY),
                            timeout=60,
                        )
                _mcp_toolsets.extend(mcp_loaded)
            except Exception as e:
                logger.warning(f"Could not load MCP config {mcp_config}: {e}")

    config = {
        "tag_prompts": tag_prompts,
        "tag_env_vars": tag_env_vars,
        "mcp_url": mcp_url,
        "mcp_config": mcp_config,
        "mcp_toolsets": _mcp_toolsets,
        "router_model": router_model,
        "agent_model": agent_model,
        "min_confidence": min_confidence,
        "valid_domains": tuple(tag_prompts.keys()),
        "provider": kwargs.get("provider", DEFAULT_PROVIDER),
        "base_url": kwargs.get("base_url", DEFAULT_LLM_BASE_URL),
        "api_key": kwargs.get("api_key", DEFAULT_LLM_API_KEY),
        "ssl_verify": kwargs.get("ssl_verify", DEFAULT_SSL_VERIFY),
        "sub_agents": _sub_agents,
        "routing_strategy": routing_strategy,
    }

    logger.info(
        f"Created graph '{name}' with {len(tag_prompts)} domain nodes: "
        f"{', '.join(tag_prompts.keys())}"
    )

    return graph, config


async def run_graph(
    graph,
    config: dict,
    query: str,
    run_id: str | None = None,
    persist: bool = False,
    state_dir: str = DEFAULT_GRAPH_PERSISTENCE_PATH or "agent_data/graph_state",
    streamdown: bool = True,
    eq: Optional[asyncio.Queue] = None,
) -> dict:
    """Execute a query through the graph orchestrator.

    Args:
        graph: The Graph object from create_graph_agent().
        config: The config dict from create_graph_agent().
        query: The user's query string.
        run_id: Optional run ID for persistence. Auto-generated if None.
        persist: Whether to persist state to disk via FileStatePersistence.
        state_dir: Directory for state files when persist=True.
        streamdown: Whether to prepend the mermaid diagram to the output.
        eq: Optional event queue for SSE streaming of graph lifecycle events.

    Returns:
        Dict with run_id, domain, results, and any error.
    """
    if run_id is None:
        run_id = uuid4().hex

    mermaid_prefix = ""
    if streamdown:
        try:
            mermaid_prefix = f"```mermaid\n{get_graph_mermaid(graph, config)}\n```\n\n"
        except Exception:
            pass

    state = GraphState(query=query)

    deps = GraphDeps(
        tag_prompts=config.get("tag_prompts", {}),
        tag_env_vars=config.get("tag_env_vars", {}),
        mcp_toolsets=config.get("mcp_toolsets", []),
        mcp_url=config.get("mcp_url", ""),
        mcp_config=config.get("mcp_config", ""),
        router_model=config.get("router_model", DEFAULT_ROUTER_MODEL),
        agent_model=config.get("agent_model", DEFAULT_GRAPH_AGENT_MODEL),
        min_confidence=config.get("min_confidence", 0.6),
        sub_agents=config.get("sub_agents", {}),
        provider=config.get("provider", DEFAULT_PROVIDER),
        base_url=config.get("base_url"),
        api_key=config.get("api_key"),
        ssl_verify=config.get("ssl_verify", DEFAULT_SSL_VERIFY),
        event_queue=eq,
        request_id=config.get("request_id", run_id),
        routing_strategy=config.get("routing_strategy", "hybrid"),
        enable_llm_validation=config.get(
            "enable_llm_validation", DEFAULT_ENABLE_LLM_VALIDATION
        ),
    )

    start_node = HybridRouterNode()

    persistence = None
    if persist:
        from pydantic_graph.persistence.file import FileStatePersistence

        path = Path(state_dir)
        if not path.suffix:
            path = path / f"{run_id}.json"

        path.parent.mkdir(parents=True, exist_ok=True)
        persistence = FileStatePersistence(json_file=path)

        existing_state = await persistence.load(run_id)
        if existing_state:
            logger.info(f"run_graph: Resuming from existing state for run_id {run_id}")
            state = existing_state
            start_node = None
        else:
            logger.info(
                f"run_graph: No existing state found for run_id {run_id}. Starting fresh."
            )
            state = GraphState(query=query, session_id=run_id)
            start_node = HybridRouterNode()
    else:
        state = GraphState(query=query, session_id=run_id)
        start_node = HybridRouterNode()

    from contextlib import AsyncExitStack

    async with AsyncExitStack() as stack:

        connected_toolsets = []
        for server in deps.mcp_toolsets:
            if hasattr(server, "__aenter__"):
                logger.info(f"run_graph: Connecting to MCP server {server}")
                try:
                    connected_server = await stack.enter_async_context(server)
                    connected_toolsets.append(connected_server)
                except Exception as e:
                    logger.error(
                        f"run_graph: Failed to connect to MCP server {server}: {e}"
                    )
            else:
                connected_toolsets.append(server)

        deps.mcp_toolsets = connected_toolsets

        logger.info(f"run_graph: Starting graph execution for run_id {run_id}")
        if tracer:
            with tracer.start_as_current_span(f"graph_run:{run_id}") as span:
                span.set_attribute("query", query)
                span.set_attribute("request_id", deps.request_id)
                result = await asyncio.wait_for(
                    graph.run(
                        start_node, state=state, deps=deps, persistence=persistence
                    ),
                    timeout=DEFAULT_GRAPH_TIMEOUT,
                )
                span.set_status(trace.Status(trace.StatusCode.OK))
        else:
            logger.info("run_graph: Running graph.run (no tracer)...")
            result = await asyncio.wait_for(
                graph.run(start_node, state=state, deps=deps, persistence=persistence),
                timeout=DEFAULT_GRAPH_TIMEOUT,
            )
            logger.info(f"run_graph: graph.run finished. Result: {result}")

    return {
        "run_id": run_id,
        "results": result.output,
        "domain": state.routed_domain,
        "mermaid": mermaid_prefix if streamdown else None,
    }


async def run_graph_stream(
    graph,
    config: dict,
    query: str,
    run_id: str | None = None,
    persist: bool = False,
    state_dir: str = DEFAULT_GRAPH_PERSISTENCE_PATH or "agent_data/graph_state",
):
    """
    Generator that yields graph events and text output concurrently.
    Used for SSE streaming via /api/chat.
    """
    import asyncio
    import json
    from uuid import uuid4
    from pathlib import Path

    if run_id is None:
        run_id = uuid4().hex

    eq = asyncio.Queue()

    deps = GraphDeps(
        tag_prompts=config.get("tag_prompts", {}),
        tag_env_vars=config.get("tag_env_vars", {}),
        mcp_toolsets=config.get("mcp_toolsets", []),
        mcp_url=config.get("mcp_url", ""),
        mcp_config=config.get("mcp_config", ""),
        router_model=config.get("router_model", DEFAULT_ROUTER_MODEL),
        agent_model=config.get("agent_model", DEFAULT_GRAPH_AGENT_MODEL),
        min_confidence=config.get("min_confidence", 0.6),
        sub_agents=config.get("sub_agents", {}),
        provider=config.get("provider", DEFAULT_PROVIDER),
        base_url=config.get("base_url"),
        api_key=config.get("api_key"),
        ssl_verify=config.get("ssl_verify", DEFAULT_SSL_VERIFY),
        event_queue=eq,
        request_id=run_id,
        routing_strategy=config.get("routing_strategy", "hybrid"),
        enable_llm_validation=config.get(
            "enable_llm_validation", DEFAULT_ENABLE_LLM_VALIDATION
        ),
    )

    state = GraphState(query=query)
    start_node = HybridRouterNode()

    persistence = None
    if persist:
        from pydantic_graph.persistence.file import FileStatePersistence

        path = Path(state_dir)
        if path.suffix != ".json":
            path = path / f"{run_id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        persistence = FileStatePersistence(json_file=path)

    async def run_in_background():
        from contextlib import AsyncExitStack

        try:
            async with AsyncExitStack() as stack:

                connected_toolsets = []
                for server in deps.mcp_toolsets:
                    if hasattr(server, "__aenter__"):
                        logger.info(
                            f"run_graph_stream_bg: Connecting to MCP server {server}"
                        )
                        try:
                            connected_server = await stack.enter_async_context(server)
                            connected_toolsets.append(connected_server)
                        except Exception as e:
                            logger.error(
                                f"run_graph_stream_bg: Failed to connect to MCP server {server}: {e}"
                            )
                    else:
                        connected_toolsets.append(server)

                deps.mcp_toolsets = connected_toolsets

                await asyncio.wait_for(
                    graph.run(
                        start_node, state=state, deps=deps, persistence=persistence
                    ),
                    timeout=DEFAULT_GRAPH_TIMEOUT,
                )
        except asyncio.TimeoutError:
            await eq.put({"type": "error", "error": "Graph execution timed out"})
        except Exception as e:
            await eq.put({"type": "error", "error": str(e)})
        finally:
            await eq.put(None)

    task = asyncio.create_task(run_in_background())

    while True:
        event = await eq.get()
        if event is None:
            break

        yield f"data: {json.dumps(event)}\n\n"

    await task

    final_output = state.results.get(state.routed_domain, "No output generated.")
    yield f"data: {json.dumps({'type': 'final_output', 'content': final_output})}\n\n"


def get_graph_mermaid(
    graph, config: dict, title: str = "Graph", routed_domain: str | None = None
) -> str:
    """Generate a Mermaid diagram for the graph.

    Args:
        graph: The Graph object.
        config: The config dict from create_graph_agent().
        title: Optional title for the diagram.
        routed_domain: Optional domain tag that was routed to.

    Returns:
        Mermaid diagram string.
    """
    mermaid = graph.mermaid_code(start_node=RouterNode)

    if title:
        if "---" in mermaid:

            import re

            mermaid = re.sub(r"title: .*", f"title: {title}", mermaid)
        else:
            mermaid = f"---\ntitle: {title}\n---\n{mermaid}"

    router_model = config.get("router_model") or "Master Router"
    if ":" in router_model:
        router_model = router_model.split(":")[-1]

    router_label = f"Router ({router_model})"
    domain_label = f"Domain Node ({routed_domain})" if routed_domain else "Domain Node"

    if "RouterNode" in mermaid:
        mermaid += f"\n  RouterNode : {router_label}"
    if "DomainNode" in mermaid:
        mermaid += f"\n  DomainNode : {domain_label}"

    return mermaid


@dataclass
class ProjectState(GraphState):
    """
    Standard state for a multi-agent project with phased execution.
    """

    task_list: TaskList = field(default_factory=TaskList)
    progress_log: ProgressLog = field(default_factory=ProgressLog)
    sprint_contract: SprintContract = field(default_factory=SprintContract)
    project_root: str = ""
    current_batch_ids: List[str] = field(default_factory=list)
    human_approval_required: bool = False

    def sync_to_disk(self, artifact_prefix: str = ""):
        """Helper to dump state artifacts for human-in-the-loop inspection."""
        if not self.project_root or not os.path.exists(self.project_root):
            return

        mappings = {
            "tasks.json": self.task_list,
            "progress.json": self.progress_log,
            "sprint.json": self.sprint_contract,
        }
        for filename, model in mappings.items():
            path = os.path.join(self.project_root, f"{artifact_prefix}{filename}")
            try:
                with open(path, "w") as f:
                    f.write(model.model_dump_json(indent=2))
            except Exception as e:
                logger.warning(f"Failed to sync artifact {filename}: {e}")


@dataclass
class ProjectDeps(GraphDeps):
    """
    Standard dependencies for a Project-mode graph.
    """

    project_root: str = ""
    max_parallel_agents: int = 3
    auto_approve_plan: bool = False
    auto_approve_tasks: bool = False


@dataclass
class BaseProjectNode(_RouterNodeBase):
    """Abstract base for Project-related nodes."""

    pass


@dataclass
class BaseProjectInitializerNode(BaseProjectNode):
    """
    Initializer for Project mode. Loads existing JSON artifacts from disk.
    """

    async def run(self, ctx: Any) -> Optional[Any]:
        ctx.state.project_root = ctx.deps.project_root or os.getcwd()
        logger.info(f"Initializing Project mode at: {ctx.state.project_root}")

        mappings = {
            "tasks.json": ("task_list", TaskList),
            "progress.json": ("progress_log", ProgressLog),
            "sprint.json": ("sprint_contract", SprintContract),
        }
        for filename, (attr, model_cls) in mappings.items():
            path = os.path.join(ctx.state.project_root, filename)
            if os.path.exists(path):
                try:
                    with open(path, "r") as f:
                        data = f.read()
                        if data.strip():
                            setattr(
                                ctx.state, attr, model_cls.model_validate_json(data)
                            )
                    logger.info(f"Loaded {filename} for project state.")
                except Exception as e:
                    logger.warning(f"Could not load {filename}: {e}")

        if ctx.state.task_list.phases:
            logger.info(
                "Initializer: Active project detected. Resuming Project Lifecycle."
            )
            return True

        return False


@dataclass
class PlannerNode(_RouterNodeBase):
    """
    Decomposes the high-level enhancement request into a structured TaskList.
    """

    async def run(self, ctx: Any) -> "ParallelExecutionNode | End[dict]":
        logger.info("Planner: Decomposing request...")

        from pydantic_ai import Agent

        planner_prompt = ctx.deps.tag_prompts.get(
            "project_planner",
            """
You are a Project Planner. Your task is to decompose a high-level request into a structured, phased TaskList.
Each phase should contain tasks that can be executed in parallel if they have no dependencies.
Break the work into logical steps: Research, Implementation (multiple parts if needed), and Validation.
Return a TaskList object.
""",
        )

        planner_agent = Agent(
            model=ctx.deps.agent_model,
            result_type=TaskList,
            system_prompt=planner_prompt,
        )

        repo_info = f"Project root: {ctx.state.project_root}\n"
        readme_path = os.path.join(ctx.state.project_root, "README.md")
        if os.path.exists(readme_path):
            with open(readme_path, "r") as f:
                repo_info += f"README Context:\n{f.read()[:500]}\n"

        prompt = f"Goal: {ctx.state.query}\n\n{repo_info}"

        try:
            result = await planner_agent.run(prompt)
            ctx.state.task_list = result.data
            ctx.state.sync_to_disk()

            return ParallelExecutionNode()
        except Exception as e:
            logger.error(f"Planning failed: {e}")
            return End({"error": f"Planning failed: {e}"})


@dataclass
class ParallelExecutionNode(_RouterNodeBase):
    """
    Manages parallel execution of tasks in the current phase.
    """

    async def run(self, ctx: Any) -> "ProjectValidatorNode":
        phase_idx = ctx.state.task_list.current_phase_index
        if phase_idx >= len(ctx.state.task_list.phases):
            return ProjectValidatorNode()

        current_phase = ctx.state.task_list.phases[phase_idx]
        pending_tasks = [
            t for t in current_phase.tasks if t.status == TaskStatus.PENDING
        ]

        if not pending_tasks:

            return ProjectValidatorNode()

        execution_batch = pending_tasks[: ctx.deps.max_parallel_agents]
        ctx.state.current_batch_ids = [t.id for t in execution_batch]

        logger.info(
            f"ParallelCoding: Starting batch of {len(execution_batch)} tasks in phase '{current_phase.name}'."
        )

        sub_agent_tasks = []
        for task in execution_batch:
            sub_agent_tasks.append(self.execute_task(ctx, task))

        await asyncio.gather(*sub_agent_tasks)
        ctx.state.sync_to_disk()

        return ProjectValidatorNode()

    async def execute_task(self, ctx: Any, task: Task):
        from pydantic_ai import Agent

        task.status = TaskStatus.IN_PROGRESS

        coding_agent = Agent(
            model=ctx.deps.agent_model,
            system_prompt=f"Task Context: {ctx.state.query}\n\nTask: {task.title}\nDescription: {task.description}",
        )

        for toolset in ctx.deps.mcp_toolsets:
            coding_agent.toolsets.append(toolset)

        try:
            res = await coding_agent.run(f"Execute task: {task.title}")
            task.result = str(res.data) if hasattr(res, "data") else str(res.output)
            task.status = TaskStatus.COMPLETED
            ctx.state.progress_log.entries.append(
                ProgressEntry(message=f"Task {task.id} complete: {task.title}")
            )
        except Exception as e:
            logger.error(f"Task {task.id} failed: {e}")
            task.status = TaskStatus.FAILED
            task.result = str(e)
            ctx.state.progress_log.entries.append(
                ProgressEntry(message=f"Task {task.id} FAILED: {task.title} Error: {e}")
            )


@dataclass
class ProjectValidatorNode(_RouterNodeBase):
    """
    Validates phase completion and project health.
    """

    async def run(self, ctx: Any) -> "ParallelExecutionNode | End[dict]":
        logger.info("ProjectValidator: Checking phase health...")
        phase_idx = ctx.state.task_list.current_phase_index
        if phase_idx >= len(ctx.state.task_list.phases):
            return End(
                {
                    "status": "project_completed",
                    "task_list": ctx.state.task_list.model_dump(),
                }
            )

        current_phase = ctx.state.task_list.phases[phase_idx]

        all_complete = all(
            t.status == TaskStatus.COMPLETED for t in current_phase.tasks
        )
        has_failed = any(t.status == TaskStatus.FAILED for t in current_phase.tasks)

        if has_failed:
            logger.warning(f"Phase '{current_phase.name}' has failed tasks.")

            return End(
                {
                    "status": "failed",
                    "message": f"Phase '{current_phase.name}' failed.",
                    "task_list": ctx.state.task_list.model_dump(),
                }
            )

        if all_complete:
            logger.info(
                f"Phase '{current_phase.name}' completed. Moving to next phase."
            )
            ctx.state.task_list.current_phase_index += 1
            if ctx.state.task_list.current_phase_index >= len(
                ctx.state.task_list.phases
            ):
                return End(
                    {
                        "status": "project_completed",
                        "task_list": ctx.state.task_list.model_dump(),
                    }
                )
            return ParallelExecutionNode()

        return ParallelExecutionNode()


ValidatorNode = DomainValidatorNode
