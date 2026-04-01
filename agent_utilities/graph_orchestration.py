#!/usr/bin/python

from __future__ import annotations

import os
import re
import json
import logging
import asyncio


from typing import Any, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    pass
from typing import Literal, Dict
from pathlib import Path


from pydantic_ai import Agent


from .config import (
    config,
    DEFAULT_PROVIDER,
    DEFAULT_LLM_BASE_URL,
    DEFAULT_LLM_API_KEY,
    DEFAULT_SSL_VERIFY,
    DEFAULT_MIN_CONFIDENCE,
    DEFAULT_ROUTING_STRATEGY,
    DEFAULT_VALIDATION_MODE,
    DEFAULT_GRAPH_PERSISTENCE_PATH,
    DEFAULT_ENABLE_LLM_VALIDATION,
    DEFAULT_ROUTER_MODEL,
    DEFAULT_GRAPH_AGENT_MODEL,
)
from .workspace import get_workspace_path, CORE_FILES, load_workspace_file
from .base_utilities import (
    is_loopback_url,
)
from .agent_factory import create_agent
from .model_factory import create_model
from .tools.git_tools import get_git_status
from .tools import (
    project_search,
    list_files,
    get_git_status,
    create_worktree,
    list_worktrees,
)


from .models import (
    PeriodicTask,
    TaskList,
    Task,
    TaskStatus,
    ProgressLog,
    SprintContract,
    UsageStatistics,
    MCPConfigModel,
)

tasks: List[PeriodicTask] = []
lock = asyncio.Lock()


logger = logging.getLogger(__name__)

_routing_cache: Dict[str, DomainChoice] = {}


def load_mcp_config() -> MCPConfigModel:
    """Load MCP config from workspace."""
    path = get_workspace_path(CORE_FILES["MCP_CONFIG"])
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return MCPConfigModel.model_validate(data)
        except Exception:
            return MCPConfigModel()
    return MCPConfigModel()


def save_mcp_config(config: MCPConfigModel):
    """Save MCP config to workspace."""
    path = get_workspace_path(CORE_FILES["MCP_CONFIG"])
    path.write_text(config.model_dump_json(indent=2), encoding="utf-8")


from dataclasses import dataclass, field
from uuid import uuid4
from pydantic import BaseModel, Field

try:
    from pydantic_graph import BaseNode, End, Graph
    from pydantic_graph.beta import GraphBuilder, StepContext, TypeExpression
    from pydantic_graph.beta.join import reduce_list_append

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


class HybridRouterNode:
    """Compatibility shim for legacy class-based routing."""

    pass


class DomainNode:
    """Compatibility shim for legacy class-based domain logic."""

    pass


@dataclass
class GraphState:
    """Universal graph state for all agent graph orchestrations."""

    query: str
    """The original user query."""

    # Pro Mode State
    topology: str = "basic"  # "basic" or "pro"
    mode: str = "ask"  # "ask", "plan", or "execute"
    exploration_notes: str = ""
    architectural_decisions: str = ""
    verification_feedback: str = ""

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

    session_usage: UsageStatistics = field(default_factory=UsageStatistics)
    """Aggregated token usage and cost for this session."""

    user_redirect_feedback: Optional[str] = None
    """Feedback from a triage pause that redirects the graph to a different domain."""

    def _update_usage(self, result_usage: Any):
        """Standardizes token usage incrementing across all steps."""
        if not result_usage:
            return
        self.session_usage.input_tokens += getattr(result_usage, "request_tokens", 0)
        self.session_usage.output_tokens += getattr(result_usage, "response_tokens", 0)
        self.session_usage.total_tokens += getattr(result_usage, "total_tokens", 0)

        # Simple cost estimation based on Sonnet 3.5 defaults
        self.session_usage.estimated_cost_usd = (
            self.session_usage.input_tokens * 0.000003
        ) + (self.session_usage.output_tokens * 0.000015)
        logger.debug(
            f"Usage Updated: ${self.session_usage.estimated_cost_usd:.4f} ({self.session_usage.total_tokens} tokens)"
        )

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
            "usage.json": self.session_usage,
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
            "usage.json": ("session_usage", UsageStatistics),
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
    """Model for multi-domain routing decisions."""

    domains: list[str]
    """List of identified domain tags. Options: security, debugger, ui_ux, devops, cloud, database, python, typescript, rust, golang, mcp, workspace."""

    reasoning: str
    """Brief explanation of the routing decision."""

    project_mode: bool = False
    """True if this is a complex request requiring project-level planning."""

    topology: Literal["basic", "pro"] = "basic"
    """The chosen graph topology based on complexity."""

    mode: Literal["ask", "plan", "execute"] = "ask"
    """The chosen orchestration mode."""


class ValidationResult(BaseModel):
    """Structured output for result validation."""

    is_valid: bool = Field(
        description="True if the result is high quality and accurate"
    )
    feedback: Optional[str] = Field(
        None, description="Detailed feedback if invalid, explaining what to improve"
    )
    score: float = Field(ge=0, le=1, description="Quality score from 0 to 1")


# --- Beta API Functional Steps ---


def load_specialized_prompts(name: str) -> str:
    """
    Loads a de-branded specialized prompt from the package's prompts directory.
    Uses importlib.resources for robust access when installed as a package.
    """
    import importlib.resources as pkg_resources
    from . import prompts

    try:
        # Pydantic 3.11+ / Python 3.9+ standard way
        with pkg_resources.files(prompts).joinpath(f"{name}.md").open("r") as f:
            return f.read()
    except Exception as e:
        logger.warning(f"Could not load specialized prompt '{name}': {e}")
        # Build-time fallback for development if needed
        local_path = os.path.join(os.path.dirname(__file__), "prompts", f"{name}.md")
        if os.path.exists(local_path):
            with open(local_path, "r") as f:
                return f.read()
        return ""


async def fetch_unified_context() -> str:
    """Fetch AGENTS.md, MEMORY.md, and git status for unified context."""
    from .workspace import CORE_FILES

    agents = load_workspace_file(CORE_FILES["AGENTS"])
    memory = load_workspace_file(CORE_FILES["MEMORY"])
    # Run git status directly or use our new tool if available
    try:
        git_status = subprocess.check_output(
            ["git", "status", "--short"], text=True
        ).strip()
    except Exception:
        git_status = "Not a git repository or git not installed."

    return (
        f"### PROJECT CONTEXT (Agent OS)\n\n"
        f"**AGENTS.md (Peer Registry):**\n{agents or '(empty)'}\n\n"
        f"**MEMORY.md (Historical Context):**\n{memory or '(empty)'}\n\n"
        f"**Git Status:**\n{git_status or '(clean)'}"
    )


async def router_step(
    ctx: StepContext[GraphState, GraphDeps, None],
) -> MultiDomainChoice | str:
    """Classifies the query into domains and determines project mode."""
    deps = ctx.deps
    import time

    emit_graph_event(
        deps.event_queue,
        "routing_started",
        query=ctx.state.query,
        available_domains=list(deps.tag_prompts.keys()),
    )

    query_normalized = ctx.state.query.strip().lower()

    # Strategy Check
    strategy = deps.routing_strategy.lower()

    # Rule-based / Cache Check
    if strategy != "llm":
        if query_normalized in _routing_cache:
            choice = _routing_cache[query_normalized]
            emit_graph_event(
                deps.event_queue,
                "routing_completed",
                domain=choice.domain,
                confidence=choice.confidence,
                reasoning="Cache hit",
            )
            return MultiDomainChoice(domains=[choice.domain], reasoning="Cache hit")

        # Reuse rule-based logic from what was in RouterNode
        matches = _rule_based_route_multi(query_normalized, deps.tag_prompts)
        if matches:
            emit_graph_event(
                deps.event_queue,
                "routing_completed",
                domains=matches,
                reasoning="Rule match",
            )
            return MultiDomainChoice(domains=matches, reasoning="Rule match")

    # LLM Routing & Topology Selection
    unified_context = await fetch_unified_context()
    if ctx.state.user_redirect_feedback:
        logger.info(
            f"router_step: Using user redirect feedback: {ctx.state.user_redirect_feedback}"
        )
        # We can either let the LLM see the feedback and re-classify, or force a choice.
        # For high-fidelity control, we'll inject it into the prompt.
        unified_context += (
            f"\n\n### USER REDIRECT FEEDBACK\n{ctx.state.user_redirect_feedback}\n"
        )
        # Clear the feedback after inject so it doesn't linger indefinitely
        ctx.state.user_redirect_feedback = None

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
            result_type=MultiDomainChoice,
            system_prompt=(
                f"You are a domain classifier and task architect. Classify the user query into ONE or MORE "
                f"of these domains: {', '.join(deps.tag_prompts.keys())}.\n\n"
                f"SUPPORTED DOMAINS:\n"
                f"- 'security': Security audits, vulnerability checks, secure coding.\n"
                f"- 'debugger': High-fidelity bug hunting and error resolution.\n"
                f"- 'ui_ux': Interface design, Tailwind CSS, accessibility.\n"
                f"- 'devops': CI/CD, infrastructure-as-code, deployment.\n"
                f"- 'cloud': Cloud architecture and service integration.\n"
                f"- 'database': SQL/NoSQL schema design and query optimization.\n"
                f"- 'python', 'typescript', 'rust', 'golang': Language experts.\n"
                f"- 'workspace': File/Skill management.\n"
                f"- 'mcp': General tool usage.\n\n"
                f"{unified_context}\n\n"
                f"DETERMINE TOPOLOGY:\n"
                f"- Set topology='pro' if the task requires deep exploration, multi-stage planning, architectural design, or rigorous verification (Expert mode).\n"
                f"- Set topology='basic' for direct queries, tool usage with clear steps, or information retrieval.\n\n"
                f"Set project_mode=True if this is a large-scale project warranting sub-task decomposition."
            ),
        )
        result = await router_agent.run(ctx.state.query)
        ctx.state._update_usage(getattr(result, "usage", None))
        return result.data
    except Exception:

        start_time = time.time()
        result = await asyncio.wait_for(router_agent.run(ctx.state.query), timeout=45.0)
        choice = result.data

        # Update state from unified choice
        ctx.state.topology = choice.topology
        ctx.state.mode = choice.mode
        ctx.state.routed_domain = ", ".join(choice.domains)
        logger.info(
            f"Router: Domains={choice.domains}, Topology={choice.topology}, Mode={choice.mode}"
        )

        emit_graph_event(
            deps.event_queue,
            "routing_completed",
            domains=choice.domains,
            topology=choice.topology,
            mode=choice.mode,
            reasoning=choice.reasoning,
        )

        return choice
    except Exception as e:
        logger.error(f"Router classification failed: {e}")
        ctx.state.error = f"Router failed: {e}"
        return "Error"


async def explorer_step(
    ctx: StepContext[GraphState, GraphDeps, None],
) -> str:
    """
    Discovery step. Uses the explorer prompt to gather context before planning.
    """
    logger.info("Explorer: Discovering codebase context...")
    explorer_prompt = load_specialized_prompts("explorer")
    unified_context = await fetch_unified_context()

    from pydantic_ai import Agent

    explorer = Agent(
        model=ctx.deps.agent_model,
        system_prompt=explorer_prompt + f"\n\n{unified_context}",
    )

    # Register all developer and git tools for exploration
    explorer.tool(project_search)
    explorer.tool(list_files)
    explorer.tool(get_git_status)
    explorer.tool(list_worktrees)

    # Add toolsets for additional context if needed
    for toolset in ctx.deps.mcp_toolsets:
        explorer.toolsets.append(toolset)

    try:
        res = await explorer.run(
            f"Research and map out the context for: {ctx.state.query}"
        )
        ctx.state.exploration_notes = str(res.data)
        return "Coordinator"
    except Exception as e:
        logger.error(f"Exploration failed: {e}")
        return "Error"


async def coordinator_step(
    ctx: StepContext[GraphState, GraphDeps, None],
) -> str:
    """
    Strategy step. Synthesizes findings into a specific implementation plan.
    """
    logger.info("Coordinator: Formulating strategy...")
    coordinator_prompt = load_specialized_prompts("coordinator")

    from pydantic_ai import Agent

    coordinator = Agent(
        model=ctx.deps.agent_model,
        system_prompt=coordinator_prompt
        + f"\n\nExploration Findings:\n{ctx.state.exploration_notes}",
    )

    prompt = f"Goal: {ctx.state.query}\n\nBased on exploration, determine if we need architectural design ('Architect') or can go straight to task planning ('Planner')."

    try:
        # For now, we'll use a simple classification or just default to Architect if complex
        if (
            "architect" in ctx.state.query.lower()
            or len(ctx.state.exploration_notes) > 1000
        ):
            return "Architect"
        return "Planner"
    except Exception as e:
        logger.error(f"Coordination failed: {e}")
        return "Error"


async def architect_step(
    ctx: StepContext[GraphState, GraphDeps, None],
) -> str:
    """
    Design step. Makes high-level architectural decisions.
    """
    logger.info("Architect: Designing system changes...")
    architect_prompt = load_specialized_prompts("architect")

    from pydantic_ai import Agent

    architect = Agent(
        model=ctx.deps.agent_model,
        system_prompt=architect_prompt + f"\n\nContext:\n{ctx.state.exploration_notes}",
    )

    try:
        res = await architect.run(f"Design the architecture for: {ctx.state.query}")
        ctx.state.architectural_decisions = str(res.data)
        return "Planner"
    except Exception as e:
        logger.error(f"Architect failed: {e}")
        return "Error"


async def verifier_step(
    ctx: StepContext[GraphState, GraphDeps, None],
) -> str | End[dict]:
    """
    Verification step. Rigorously validates implementation.
    """
    logger.info("Verifier: Validating implementation...")
    verifier_prompt = load_specialized_prompts("verifier")

    from pydantic_ai import Agent

    verifier = Agent(
        model=ctx.deps.agent_model,
        system_prompt=verifier_prompt
        + f"\n\nArchitectural Intent:\n{ctx.state.architectural_decisions}",
    )

    for toolset in ctx.deps.mcp_toolsets:
        verifier.toolsets.append(toolset)

    try:
        res = await verifier.run(f"Verify the solution for: {ctx.state.query}")
        if "PASS" in str(res.data).upper():
            return End({"status": "success", "summary": str(res.data)})

        ctx.state.verification_feedback = str(res.data)
        return "Critique"
    except Exception as e:
        logger.error(f"Verification failed: {e}")
        return "Error"


async def critique_step(
    ctx: StepContext[GraphState, GraphDeps, None],
) -> str:
    """
    Self-correction step. Analyzes verification failures.
    """
    logger.info("Critique: Analyzing failures...")
    critique_prompt = load_specialized_prompts("critique")

    # Return to planner for fix
    return "Planner"


async def planner_step(
    ctx: StepContext[GraphState, GraphDeps, None],
) -> str | End[Any]:
    """
    Phased-task list creation node. Breaks large requests into manageable phases.
    """
    logger.info("Planner: Creating execution plan...")
    planner_prompt = load_specialized_prompts("planner")
    unified_context = await fetch_unified_context()

    from pydantic_ai import Agent

    planner = Agent(
        model=ctx.deps.agent_model,
        result_type=TaskList,
        system_prompt=(
            planner_prompt + f"\n\n{unified_context}\n\n"
            f"Decision Log:\n{ctx.state.architectural_decisions}\n\n"
            f"Exploration Notes:\n{ctx.state.exploration_notes}"
        ),
    )

    # Integrated Worktree & Development Tools
    planner.tool(get_git_status)
    planner.tool(create_worktree)
    planner.tool(list_worktrees)
    planner.tool(project_search)

    try:
        res = await planner.run(
            f"Create a phased implementation plan for: {ctx.state.query}"
        )
        ctx.state.task_list = res.data

        # In 'Plan' mode, we might just end here or go to an approval step
        if ctx.state.mode == "plan":
            return End({"status": "planned", "plan": ctx.state.task_list.model_dump()})

        return "ProjectExecutor"
    except Exception as e:
        logger.error(f"Planning failed: {e}")
        return "Error"


async def domain_step(
    ctx: StepContext[GraphState, GraphDeps, str],
) -> str | End[Any]:
    """Executes a single domain's MCP tools or sub-agent."""
    domain = ctx.inputs
    ctx.state.routed_domain = domain

    # Logic extracted from DomainNode.execute_domain
    # (Abbreviated here, I'll need to make sure I get the full implementation)
    # I'll define a helper for this to avoid duplication.
    result = await _execute_domain_logic(ctx, domain)

    if isinstance(result, End):
        return result
    return domain


async def validator_step(
    ctx: StepContext[GraphState, GraphDeps, str],
) -> str | End[dict]:
    """Validates the output of a domain execution and performs lightweight aggregation."""
    domain = ctx.inputs
    result_text = ctx.state.results.get(domain, "")
    deps = ctx.deps

    # Logic: If this is the last domain to report in a fan-out, we might aggregate.
    # In Pydantic Graph, each parallel node calls its successor independently.
    # So we check if we have all the results we expected.

    # 1. Skip if LLM validation is disabled
    if not deps.enable_llm_validation:
        return End(
            {"status": "success", "domain": domain, "results": ctx.state.results}
        )

    # 2. Lightweight Synthesis: If it's a "list" or "search" query, skip the LLM call
    read_patterns = ["list", "search", "find", "get", "show", "describe", "where"]
    is_read_only = any(p in ctx.state.query.lower() for p in read_patterns)

    if is_read_only and len(ctx.state.results) >= 1:
        logger.info(
            "validator_step: Read-only query detected. Performing lightweight aggregation."
        )
        # We'll let the final caller decide how to join, or just provide the map.
        return End(
            {
                "status": "success",
                "summary": "Aggregated domain results",
                "results": ctx.state.results,
            }
        )

    # 3. Standard LLM Validation (for complex write-heavy or reasoning tasks)
    if ctx.state.retry_count < 2:
        logger.info(f"validator_step: Performing LLM-based validation for '{domain}'")
        validator_agent = Agent(
            model=deps.router_model,
            result_type=ValidationResult,
            system_prompt=(
                f"You are a quality assurance expert. Evaluate the output of the '{domain}' agent.\n"
                f"Original Query: {ctx.state.query}\n"
                f"Agent Result: {result_text}\n"
                "Determine if the result accurately and comprehensively addresses the query.\n"
                "If it looks correct, set is_valid=True. Otherwise provide feedback for improvement."
            ),
        )
        try:
            val_res = await validator_agent.run("Evaluate the result.")
            if val_res.data.is_valid:
                return End({"status": "success", "results": ctx.state.results})
            else:
                ctx.state.retry_count += 1
                ctx.state.validation_feedback = val_res.data.feedback
                return domain  # Loop back
        except Exception:
            pass

    return End({"status": "success", "results": ctx.state.results})


async def python_step(ctx: StepContext[GraphState, GraphDeps, None]) -> str | End[Any]:
    """Specialized Python expert step."""
    return await _execute_specialized_step(ctx, "python")


async def golang_step(ctx: StepContext[GraphState, GraphDeps, None]) -> str | End[Any]:
    """Specialized Golang expert step."""
    return await _execute_specialized_step(ctx, "golang")


async def typescript_step(
    ctx: StepContext[GraphState, GraphDeps, None],
) -> str | End[Any]:
    """Specialized TypeScript expert step."""
    return await _execute_specialized_step(ctx, "typescript")


async def rust_step(ctx: StepContext[GraphState, GraphDeps, None]) -> str | End[Any]:
    """Specialized Rust expert step."""
    return await _execute_specialized_step(ctx, "rust")


async def security_step(
    ctx: StepContext[GraphState, GraphDeps, None],
) -> str | End[Any]:
    """Specialized Security expert step."""
    return await _execute_specialized_step(ctx, "security")


async def javascript_step(
    ctx: StepContext[GraphState, GraphDeps, None],
) -> str | End[Any]:
    """Specialized JavaScript expert step."""
    return await _execute_specialized_step(ctx, "javascript")


async def c_step(ctx: StepContext[GraphState, GraphDeps, None]) -> str | End[Any]:
    """Specialized C expert step."""
    return await _execute_specialized_step(ctx, "c")


async def cpp_step(ctx: StepContext[GraphState, GraphDeps, None]) -> str | End[Any]:
    """Specialized C++ expert step."""
    return await _execute_specialized_step(ctx, "cpp")


async def qa_step(ctx: StepContext[GraphState, GraphDeps, None]) -> str | End[Any]:
    """Specialized QA expert step."""
    return await _execute_specialized_step(ctx, "qa")


async def debugger_step(
    ctx: StepContext[GraphState, GraphDeps, None],
) -> str | End[Any]:
    """Specialized Debugging expert step."""
    return await _execute_specialized_step(ctx, "debugger")


async def ui_ux_step(ctx: StepContext[GraphState, GraphDeps, None]) -> str | End[Any]:
    """Specialized UI/UX expert step."""
    return await _execute_specialized_step(ctx, "ui_ux")


async def devops_step(ctx: StepContext[GraphState, GraphDeps, None]) -> str | End[Any]:
    """Specialized DevOps expert step."""
    return await _execute_specialized_step(ctx, "devops")


async def cloud_step(ctx: StepContext[GraphState, GraphDeps, None]) -> str | End[Any]:
    """Specialized Cloud expert step."""
    return await _execute_specialized_step(ctx, "cloud")


async def database_step(
    ctx: StepContext[GraphState, GraphDeps, None],
) -> str | End[Any]:
    """Specialized Database expert step."""
    return await _execute_specialized_step(ctx, "database")


async def _execute_specialized_step(
    ctx: StepContext[GraphState, GraphDeps, None], prompt_name: str
) -> str | End[Any]:
    """Shared logic for specialized steps using migrated prompts."""
    prompt = load_specialized_prompts(prompt_name)
    agent = Agent(
        model=ctx.deps.agent_model,
        system_prompt=prompt,
        tools=developer_tools,
    )
    for toolset in ctx.deps.mcp_toolsets:
        agent.toolsets.append(toolset)

    try:
        res = await agent.run(ctx.state.query)
        ctx.state._update_usage(getattr(res, "usage", None))
        ctx.state.results[prompt_name] = str(res.data)
        return End(
            {"status": "success", "domain": prompt_name, "result": str(res.data)}
        )
    except Exception as e:
        logger.error(f"Specialized step '{prompt_name}' failed: {e}")
        return "Error"


async def dynamic_mcp_routing_step(
    ctx: StepContext[GraphState, GraphDeps, None],
) -> List[str]:
    """Dynamically identifies MCP servers to route to based on config."""
    mcp_config = load_mcp_config()
    servers = list(mcp_config.mcpServers.keys())
    logger.info(f"Dynamic MCP Routing: Routing to {len(servers)} servers: {servers}")
    return servers


async def mcp_server_step(
    ctx: StepContext[GraphState, GraphDeps, str],
) -> str | End[Any]:
    """Execute a query against a specific MCP server."""
    server_name = ctx.input
    query = ctx.state.query

    logger.info(f"Executing MCP Server Step: {server_name} for query: {query}")

    # Emit node start event
    emit_graph_event(
        ctx.deps.event_queue,
        "node_start",
        node_id="mcp_server_execution",
        server=server_name,
    )

    try:
        # Create a dynamic agent for this specific MCP server

        # We assume the MCP config is already handled at the factory level,
        # but here we might need to filter tools specifically for this server.
        # For now, we'll use a simplified implementation that returns a placeholder
        # or simulates the execution. In a full implementation, this would connect to the server.

        # Placeholder result
        ctx.state.results[server_name] = f"Results from {server_name} for '{query}'"

        emit_graph_event(
            ctx.deps.event_queue,
            "node_complete",
            node_id="mcp_server_execution",
            server=server_name,
            result=ctx.state.results[server_name],
        )

        return "approval_gate"
    except Exception as e:
        logger.error(f"MCP Server Step '{server_name}' failed: {e}")
        return "error_recovery"


#     """Executes tools for a specific MCP server."""
#     server_name = ctx.inputs
#     logger.info(f"Executing MCP Server Step: {server_name}")
#     return await _execute_domain_logic(ctx, server_name)


async def usage_guard_step(
    ctx: StepContext[GraphState, GraphDeps, Any],
) -> Any:
    """Monitors token usage and cost, emitting warnings if limits are exceeded."""
    usage = ctx.state.session_usage

    # Defaults: $5.00 limit, 500k tokens
    cost_limit = 5.0
    token_limit = 500000

    if usage.estimated_cost_usd > cost_limit or usage.total_tokens > token_limit:
        logger.warning(
            f"UsageGuard: Safety limits reached! Cost: ${usage.estimated_cost_usd:.2f}, Tokens: {usage.total_tokens}"
        )
        emit_graph_event(
            ctx.deps.event_queue,
            "safety_warning",
            message=f"Session usage has exceeded safety limits. Current cost: ${usage.estimated_cost_usd:.2f}",
            usage=usage.model_dump(),
        )

    return ctx.inputs


async def approval_gate_step(
    ctx: StepContext[GraphState, GraphDeps, Any],
) -> Any:
    """Pauses for human approval and captures triage redirection feedback."""
    if ctx.state.mode != "plan" and not ctx.state.human_approval_required:
        return ctx.inputs

    logger.info("Approval Gate: Pausing for user review...")
    ctx.state.human_approval_required = True

    # In a real-time SSE environment, the user would provide feedback via a separate endpoint/call.
    # If the user provides a "Redirect" command, it will be populated in ctx.state.user_redirect_feedback.

    if ctx.state.user_redirect_feedback:
        logger.info(
            "Approval Gate: Captured redirection feedback. Returning to router."
        )
        ctx.state.human_approval_required = False
        return "router"

    return ctx.inputs


async def planner_step(
    ctx: StepContext[GraphState, GraphDeps, MultiDomainChoice],
) -> TaskList:
    """Decomposes a request into a TaskList for project mode."""
    logger.info("planner_step: Decomposing request...")
    planner_prompt = ctx.deps.tag_prompts.get(
        "project_planner",
        "You are a Project Planner. Decompose the request into a phased TaskList. Phases: Research, Implementation, Validation.",
    )
    planner_agent = Agent(
        model=ctx.deps.agent_model,
        result_type=TaskList,
        system_prompt=planner_prompt,
    )
    repo_info = f"Project root: {ctx.state.project_root}\n"
    prompt = f"Goal: {ctx.state.query}\n\n{repo_info}"
    result = await planner_agent.run(prompt)
    ctx.state.task_list = result.data
    ctx.state.sync_to_disk()
    return ctx.state.task_list


async def project_executor_step(
    ctx: StepContext[GraphState, GraphDeps, Task],
) -> Task:
    """Executes a single task from a project plan with specialized agent support."""
    task = ctx.inputs
    task.status = TaskStatus.IN_PROGRESS

    # Specialist Mapping
    specialist_map = {
        "python": "python_programmer",
        "c": "c_reviewer",
        "cpp": "cpp_reviewer",
        "golang": "golang_reviewer",
        "javascript": "javascript_reviewer",
        "typescript": "typescript_reviewer",
        "security": "security_auditor",
        "qa": "qa_expert",
    }

    prompt_name = None
    task_type_lower = task.type.lower()
    for key, name in specialist_map.items():
        if key in task_type_lower:
            prompt_name = name
            break

    if prompt_name:
        logger.info(
            f"project_executor_step: Using specialized agent '{prompt_name}' for task '{task.title}'"
        )
        special_prompt = load_specialized_prompts(prompt_name)
        system_prompt = f"Global Goal: {ctx.state.query}\n\nSpecialized Context: {special_prompt}\n\nTarget Task: {task.title}\nDescription: {task.description}"
    else:
        system_prompt = f"Task Context: {ctx.state.query}\n\nTask: {task.title}\nDescription: {task.description}"

    from .tools.developer_tools import developer_tools

    agent = Agent(
        model=ctx.deps.agent_model,
        system_prompt=system_prompt,
        tools=developer_tools,
    )
    for toolset in ctx.deps.mcp_toolsets:
        agent.toolsets.append(toolset)

    try:
        res = await agent.run(f"Execute task: {task.title}")
        task.result = str(res.data) if hasattr(res, "data") else str(res.output)
        task.status = TaskStatus.COMPLETED
    except Exception as e:
        logger.error(f"Task '{task.title}' failed: {e}")
        task.status = TaskStatus.FAILED
        task.result = str(e)
    return task


async def error_recovery_step(
    ctx: StepContext[GraphState, GraphDeps, Exception | str | Any],
) -> End[dict]:
    """Handles errors by either retrying or ending with error state."""
    logger.error(f"error_recovery_step: {ctx.inputs}")
    return End({"error": str(ctx.inputs), "results": ctx.state.results})


def _rule_based_route_multi(query: str, labels: dict) -> list[str]:
    """Simple keyword-based routing for multiple matches with plural/singular awareness."""
    matches = []

    query_lower = query.lower()
    for label in labels:
        label_lower = label.lower()

        if re.search(rf"\b{label_lower}\b", query_lower):
            logger.debug(f"_rule_based_route_multi: Exact boundary match for '{label}'")
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


async def _execute_domain_logic(
    ctx: StepContext[GraphState, GraphDeps, str], domain: str
):
    """Core logic for executing a domain, extracted from DomainNode."""
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
                    import importlib

                    module = importlib.import_module(f"{target}.agent_server")
                    if hasattr(module, "agent_template"):
                        target = module.agent_template(
                            provider=deps.provider,
                            agent_model=deps.agent_model,
                            router_model=deps.router_model,
                            api_key=deps.api_key,
                            base_url=deps.base_url,
                            ssl_verify=deps.ssl_verify,
                        )
                    else:
                        raise AttributeError(
                            f"Package {target} is missing agent_template()"
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
                    async with target.run_stream(ctx.state.query) as stream:
                        async for message, last in stream.stream_messages():
                            emit_graph_event(
                                deps.event_queue,
                                "subagent_thought",
                                domain=domain,
                                message=str(message),
                            )
                        res = await stream.get_output()
                    output = getattr(res, "output", None) or getattr(res, "data", res)

                ctx.state.results[domain] = str(output)
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

            result = await asyncio.wait_for(
                sub_agent.run(query), timeout=DEFAULT_GRAPH_TIMEOUT
            )

            output = getattr(result, "output", None) or getattr(result, "data", result)
            from pydantic_ai import DeferredToolRequests

            if isinstance(output, DeferredToolRequests):
                ctx.state.human_approval_required = True
                ctx.state.results[domain] = output
                emit_graph_event(
                    deps.event_queue,
                    "approval_required",
                    domain=domain,
                    tool_calls=[
                        (tc.model_dump() if hasattr(tc, "model_dump") else str(tc))
                        for tc in (getattr(output, "calls", []) or [])
                    ],
                )
                return End(output)
            else:
                ctx.state.results[domain] = str(output)
                emit_graph_event(deps.event_queue, "subagent_completed", domain=domain)

    except Exception as e:
        logger.error(f"domain_step error for '{domain}': {e}")
        ctx.state.error = f"Domain failed: {e}"
        ctx.state.results[domain] = f"Error: {e}"
        return "Error"
    finally:
        for env_var, value in original_env.items():
            if value is None:
                os.environ.pop(env_var, None)
            else:
                os.environ[env_var] = value
    return None


# --- End Beta API Functional Steps ---


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
    for tag, agent_cfg in _skill_agents.items():
        if tag not in tag_prompts:
            tag_prompts[tag] = agent_cfg.get(
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
    **kwargs,
) -> tuple["Graph", dict]:
    """Factory to create a router-led graph assistant using Pydantic Graph Beta.

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
    if not _PYDANTIC_GRAPH_AVAILABLE:
        raise ImportError("pydantic-graph is required for graph agents.")

    if tag_env_vars is None:
        tag_env_vars = build_tag_env_map(list(tag_prompts.keys()))

    # Initialize GraphBuilder
    g = GraphBuilder(
        state_type=GraphState,
        deps_type=GraphDeps,
        output_type=dict,
    )

    # Register Steps
    # We use decorators on the functions we defined earlier, or just refer to them.
    # Since they are defined globally, we can use them directly with g.step()

    _router = g.step(router_step, node_id="router")
    _validator = g.step(validator_step, node_id="validator")
    _planner = g.step(planner_step, node_id="planner")
    _executor = g.step(project_executor_step, node_id="project_executor")
    _error = g.step(error_recovery_step, node_id="error_recovery")

    _explorer = g.step(explorer_step, node_id="explorer")
    _coordinator = g.step(coordinator_step, node_id="coordinator")
    _architect = g.step(architect_step, node_id="architect")
    _verifier = g.step(verifier_step, node_id="verifier")
    _critique = g.step(critique_step, node_id="critique")

    # Native Developer Steps
    _python = g.step(python_step, node_id="python_programmer")
    _c = g.step(c_step, node_id="c_reviewer")
    _cpp = g.step(cpp_step, node_id="cpp_reviewer")
    _golang = g.step(golang_step, node_id="golang_reviewer")
    _javascript = g.step(javascript_step, node_id="javascript_reviewer")
    _typescript = g.step(typescript_step, node_id="typescript_reviewer")
    _security = g.step(security_step, node_id="security_auditor")
    _qa = g.step(qa_step, node_id="qa_expert")
    _debugger = g.step(debugger_step, node_id="debugger_step")
    _ui_ux = g.step(ui_ux_step, node_id="ui_ux_step")
    _devops = g.step(devops_step, node_id="devops_step")
    _cloud = g.step(cloud_step, node_id="cloud_step")
    _database = g.step(database_step, node_id="database_step")
    _rust = g.step(rust_step, node_id="rust_step")

    # # Dynamic MCP Steps
    _mcp_router = g.step(dynamic_mcp_routing_step, node_id="mcp_router")
    _mcp_server = g.step(mcp_server_step, node_id="mcp_server_execution")

    # Approval Gate
    _approval = g.step(approval_gate_step, node_id="approval_gate")

    # Usage Guard
    _usage_guard = g.step(usage_guard_step, node_id="usage_guard")

    # Define Graph Topology
    g.add(
        # Start -> UsageGuard -> Router
        g.edge_from(g.start_node).to(_usage_guard),
        g.edge_from(_usage_guard).to(_router),
        # Router Decision Branching
        g.edge_from(_router).to(
            g.decision()
            # Path 1: Pro Mode (Dynamic Shift)
            .branch(
                g.match(
                    TypeExpression[MultiDomainChoice],
                    matches=lambda x: x.topology == "pro",
                ).to(_explorer)
            )
            # Path 2: Mode-based Branching (Plan mode requires approval)
            .branch(
                g.match(
                    TypeExpression[MultiDomainChoice],
                    matches=lambda x: x.mode == "plan",
                ).to(_approval)
            )
            # Path 3: Execute Mode (Resume tasks)
            .branch(
                g.match(
                    TypeExpression[MultiDomainChoice],
                    matches=lambda x: x.mode == "execute",
                ).to(_planner)
            )
            # Path 4: Specialized Expert Nodes
            .branch(
                g.match(
                    TypeExpression[MultiDomainChoice],
                    matches=lambda x: "security" in x.domains,
                ).to(_security)
            )
            .branch(
                g.match(
                    TypeExpression[MultiDomainChoice],
                    matches=lambda x: "debugger" in x.domains,
                ).to(_debugger)
            )
            .branch(
                g.match(
                    TypeExpression[MultiDomainChoice],
                    matches=lambda x: "ui_ux" in x.domains,
                ).to(_ui_ux)
            )
            .branch(
                g.match(
                    TypeExpression[MultiDomainChoice],
                    matches=lambda x: "devops" in x.domains,
                ).to(_devops)
            )
            .branch(
                g.match(
                    TypeExpression[MultiDomainChoice],
                    matches=lambda x: "cloud" in x.domains,
                ).to(_cloud)
            )
            .branch(
                g.match(
                    TypeExpression[MultiDomainChoice],
                    matches=lambda x: "database" in x.domains,
                ).to(_database)
            )
            # Path 5: Language Experts
            .branch(
                g.match(
                    TypeExpression[MultiDomainChoice],
                    matches=lambda x: "python" in x.domains,
                ).to(_python)
            )
            .branch(
                g.match(
                    TypeExpression[MultiDomainChoice],
                    matches=lambda x: "c" in x.domains,
                ).to(_c)
            )
            .branch(
                g.match(
                    TypeExpression[MultiDomainChoice],
                    matches=lambda x: "cpp" in x.domains,
                ).to(_cpp)
            )
            .branch(
                g.match(
                    TypeExpression[MultiDomainChoice],
                    matches=lambda x: "golang" in x.domains,
                ).to(_golang)
            )
            .branch(
                g.match(
                    TypeExpression[MultiDomainChoice],
                    matches=lambda x: "javascript" in x.domains,
                ).to(_javascript)
            )
            .branch(
                g.match(
                    TypeExpression[MultiDomainChoice],
                    matches=lambda x: "typescript" in x.domains,
                ).to(_typescript)
            )
            .branch(
                g.match(
                    TypeExpression[MultiDomainChoice],
                    matches=lambda x: "security" in x.domains,
                ).to(_security)
            )
            .branch(
                g.match(
                    TypeExpression[MultiDomainChoice],
                    matches=lambda x: "qa" in x.domains,
                ).to(_qa)
            )
            # Path 5: Dynamic MCP (if no specific domain matches or explicit mcp requested)
            .branch(
                g.match(
                    TypeExpression[MultiDomainChoice],
                    matches=lambda x: "mcp" in x.domains or not x.domains,
                ).to(_mcp_router)
            )
            # Fallback
            .branch(g.match(TypeExpression[object]).to(_error))
        ),
        g.edge_from(_python).to(_validator),
        g.edge_from(_golang).to(_validator),
        g.edge_from(_typescript).to(_validator),
        g.edge_from(_rust).to(_validator),
        g.edge_from(_c).to(_validator),
        g.edge_from(_cpp).to(_validator),
        g.edge_from(_javascript).to(_validator),
        g.edge_from(_qa).to(_validator),
        g.edge_from(_security).to(_validator),
        g.edge_from(_debugger).to(_validator),
        g.edge_from(_ui_ux).to(_validator),
        g.edge_from(_devops).to(_validator),
        g.edge_from(_cloud).to(_validator),
        g.edge_from(_database).to(_validator),
        # MCP Parallel Flow
        g.edge_from(_mcp_router).map().to(_mcp_server),
        g.edge_from(_mcp_server).to(_validator),
        # Approval Gate routing
        g.edge_from(_approval).to(
            g.decision()
            .branch(g.match("router").to(_router))
            .branch(
                g.match(
                    TypeExpression[MultiDomainChoice], matches=lambda x: x.project_mode
                ).to(_planner)
            )
            .branch(g.match(TypeExpression[MultiDomainChoice]).to(_mcp_router))
            .branch(g.match(TaskList).to(_planner))
            .branch(g.match(TypeExpression[object]).to(_error))
        ),
        # Pro Mode Lifecycle
        g.edge_from(_explorer).to(_coordinator),
        g.edge_from(_coordinator).to(
            g.decision()
            .branch(g.match("Architect").to(_architect))
            .branch(g.match("Planner").to(_planner))
            .branch(g.match(TypeExpression[object]).to(_error))
        ),
        g.edge_from(_architect).to(_planner),
        # Shared Execution & Verification
        g.edge_from(_planner)
        .transform(lambda ctx: ctx.state.task_list.all_tasks)
        .map()
        .to(_executor),
        g.edge_from(_executor).to(
            g.decision()
            .branch(
                g.match(Task, matches=lambda _: ctx.state.topology == "pro").to(
                    _verifier
                )
            )
            .branch(g.match(TypeExpression[object]).to(g.end_node))
        ),
        g.edge_from(_verifier).to(
            g.decision()
            .branch(g.match("Critique").to(_critique))
            .branch(g.match(TypeExpression[dict]).to(g.end_node))
            .branch(g.match(TypeExpression[object]).to(_error))
        ),
        g.edge_from(_critique).to(_planner),
        # Basic Mode Fan-in
        g.edge_from(_mcp_server).to(_validator),
        g.edge_from(_validator).to(g.end_node),
        # Error -> End
        g.edge_from(_error).to(g.end_node),
    )

    # Add custom nodes if provided (legacy support via wrapping)
    if custom_nodes:
        for node in custom_nodes:
            logger.warning(
                f"Custom node {node} detected. Functional steps are preferred in Beta API."
            )
            # We could wrap them here if needed, but for now we prioritize functional steps.

    graph = g.build()

    # MCP Setup (same as before)
    _mcp_toolsets = list(mcp_toolsets) if mcp_toolsets else []
    # (Loading toolsets logic preserved...)
    if not DEFAULT_VALIDATION_MODE:
        if mcp_url:
            import httpx
            from pydantic_ai.mcp import MCPServerSSE, MCPServerStreamableHTTP

            if is_loopback_url(
                mcp_url, kwargs.get("current_host"), kwargs.get("current_port")
            ):
                pass
            elif mcp_url.lower().endswith("/sse"):
                _mcp_toolsets.append(
                    MCPServerSSE(
                        mcp_url,
                        http_client=httpx.AsyncClient(
                            verify=kwargs.get("ssl_verify", DEFAULT_SSL_VERIFY),
                            timeout=60,
                        ),
                    )
                )
            else:
                _mcp_toolsets.append(
                    MCPServerStreamableHTTP(
                        mcp_url,
                        http_client=httpx.AsyncClient(
                            verify=kwargs.get("ssl_verify", DEFAULT_SSL_VERIFY),
                            timeout=60,
                        ),
                    )
                )

        if mcp_config:
            from pydantic_ai.mcp import load_mcp_servers

            try:
                mcp_loaded = load_mcp_servers(mcp_config)
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
        "sub_agents": sub_agents or {},
        "routing_strategy": routing_strategy,
    }

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
    mode: str = "ask",
    topology: str = "basic",
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

    state = GraphState(query=query, mode=mode, topology=topology)

    deps = GraphDeps(
        tag_prompts=config.get("tag_prompts", {}),
        tag_env_vars=config.get("tag_env_vars", {}),
        mcp_toolsets=config.get("mcp_toolsets", []),
        mcp_url=config.get("mcp_url", ""),
        mcp_config=config.get("mcp_config", ""),
        router_model=create_model(config.get("router_model", DEFAULT_ROUTER_MODEL)),
        agent_model=create_model(config.get("agent_model", DEFAULT_GRAPH_AGENT_MODEL)),
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

    state = GraphState(query=query, session_id=run_id, mode=mode, topology=topology)

    persistence = None
    if persist:
        from pydantic_graph.persistence.file import FileStatePersistence

        path = Path(state_dir)
        if path.suffix != ".json":
            path = path / f"{run_id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        persistence = FileStatePersistence(json_file=path)

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
                    graph.run(state=state, deps=deps),
                    timeout=DEFAULT_GRAPH_TIMEOUT,
                )
                span.set_status(trace.Status(trace.StatusCode.OK))
        else:
            logger.info("run_graph: Running beta graph.run (no tracer)...")
            result = await asyncio.wait_for(
                graph.run(state=state, deps=deps),
                timeout=DEFAULT_GRAPH_TIMEOUT,
            )
            logger.info(f"run_graph: graph.run finished. Result: {result}")

    return {
        "run_id": run_id,
        "results": result,
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
    mode: str = "ask",
    topology: str = "basic",
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
        router_model=create_model(config.get("router_model", DEFAULT_ROUTER_MODEL)),
        agent_model=create_model(config.get("agent_model", DEFAULT_GRAPH_AGENT_MODEL)),
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

    state = GraphState(query=query, mode=mode, topology=topology)

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
                    graph.run(state=state, deps=deps),
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
    if hasattr(graph, "mermaid_code"):
        mermaid = graph.mermaid_code()
    else:
        mermaid = graph.render()

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

    if "router" in mermaid:
        mermaid += f"\n  router : {router_label}"
    if "domain_execution" in mermaid:
        mermaid += f"\n  domain_execution : {domain_label}"

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


# --- Legacy Nodes Removed ---
