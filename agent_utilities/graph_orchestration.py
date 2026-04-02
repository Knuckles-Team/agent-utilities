#!/usr/bin/python

from __future__ import annotations

import os
import re
import json
import logging
import asyncio
import importlib
import time
import functools


from typing import Any, List, Optional, TYPE_CHECKING, Union

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
    ExecutionStep,
    GraphPlan,
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

    # Dynamic Planning State
    plan: GraphPlan = field(default_factory=GraphPlan)
    """The fully determined execution plan for this run."""

    step_cursor: int = 0
    """Current index in the sequential plan steps."""

    results_registry: dict[str, Any] = field(default_factory=dict)
    """Aggregated results from each execution step."""

    pending_parallel_count: int = 0
    """Number of parallel tasks we are currently waiting for."""

    current_node_retries: int = 0
    """Number of local retries attempted on the current expert node."""

    global_research_loops: int = 0
    """Number of times the entire graph has looped back to research/re-planning."""

    needs_replan: bool = False
    """Flag to trigger a re-evaluation of the graph plan due to implementation failures."""

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


# Domain classification models used by the router step
class DomainChoice(BaseModel):
    """Structured output from the router LLM."""

    domain: str = Field(description="The domain tag to route to")
    confidence: float = Field(ge=0, le=1, description="Routing confidence 0-1")
    reasoning: str = Field(description="Brief reasoning for the classification")


class MultiDomainChoice(BaseModel):
    """Structured output for dynamic graph planning."""

    plan: GraphPlan = Field(description="The sequential/parallel execution plan")
    reasoning: str = Field(
        description="Brief reasoning for the plan architecture"
    )
    is_resumed: bool = Field(False, description="Whether this is a resumed operation")


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


def get_step_descriptions() -> str:
    """Returns a catalog of available expert nodes for the LLM planner."""
    from .a2a import discover_agents
    
    # Discovery of local agent packages
    discovered = discover_agents()
    
    steps = {
        "researcher": "Multi-vector discovery expert. Trigger this when information is missing or assumptions need validation. Can be spawned in parallel for simultaneous Web, Code, and Workspace research.",
        "architect": "System design expert. Analyzes requirements and defines high-level structures. Performs 'Gap Analysis' to identify missing context.",
        "planner": "Task orchestration expert. Bridges the gap between architecture and execution. Assesses missing knowledge and spawns researchers to validate assumptions.",
        "python_programmer": "Specialized Python engineer for implementation, refactoring, and standalone scripts.",
        "typescript_programmer": "Frontend and Node.js expert specializing in TypeScript and React ecosystems.",
        "javascript_programmer": "General-purpose JavaScript and web development specialist.",
        "rust_expert": "Systems programming and memory safety expert.",
        "golang_expert": "Cloud-native and high-performance backend expert.",
        "security_auditor": "Expert in threat modeling, vulnerability scanning, and secure coding practices.",
        "qa_expert": "Quality assurance lead. Designs test plans and implements automated test suites.",
        "debugger_expert": "Interpreting error logs and fixing complex bugs.",
        "ui_ux_designer": "Frontend design, CSS, and user interface expert.",
        "devops_engineer": "CI/CD, Docker, and infrastructure expert.",
        "cloud_architect": "Cloud services and architecture design expert.",
        "database_expert": "SQL/NoSQL design and query optimization expert.",
        "verifier": "Final quality gate. Validates that the implementation meets the original query requirements.",
        "mcp_server": "General-purpose tool hub for any task not covered by specialized nodes.",
    }
    
    # Merge discovered agents into the catalog
    for tag, meta in discovered.items():
        if tag not in steps:
            agent_type = meta.get("type", "local")
            if agent_type == "remote_a2a":
                steps[tag] = f"Remote A2A Specialist '{meta['name']}': {meta['description']} (Capabilities: {meta.get('capabilities', 'N/A')})"
            else:
                steps[tag] = f"Specialized Domain Agent '{meta['name']}': {meta['description']}. This agent has its own internal graph and specialized toolsets."

    return "\n".join([f"- {k}: {v}" for k, v in steps.items()])


# Mapping of Graph Nodes to Universal Skills and Skill Graphs
# This ensures each specialist has the highest-fidelity tools/docs for their domain.
NODE_SKILL_MAP = {
    "researcher": [
        "web-search", "web-fetch", "web-crawler", "agent-browser", "systems-manager",
        "browser-tools", "web-artifacts", "workspace-analyst"
    ],
    "python_programmer": [
        "agent-builder", "tdd-methodology", "mcp-builder", "developer-utilities",
        "jupyter-notebook", "python-docs", "pydantic-ai-docs", "fastapi-docs",
        "agent-package-builder"
    ],
    "typescript_programmer": [
        "react-development", "web-artifacts", "tdd-methodology", "canvas-design",
        "nodejs-docs", "react-docs", "nextjs-docs", "shadcn-docs"
    ],
    "javascript_programmer": [
        "web-artifacts", "canvas-design", "nodejs-docs", "react-docs"
    ],
    "rust_expert": ["rust-docs"],
    "golang_expert": ["go-docs"],
    "security_auditor": ["security-tools", "linux-docs"],
    "debugger_expert": ["developer-utilities", "agent-builder"],
    "qa_expert": ["qa-planning", "tdd-methodology", "testing-library-docs"],
    "ui_ux_designer": [
        "theme-factory", "canvas-design", "brand-guidelines", "algorithmic-art",
        "shadcn-docs", "tailwind-docs", "framer-docs"
    ],
    "devops_engineer": ["cloudflare-deploy", "docker-docs", "terraform-docs"],
    "cloud_architect": ["c4-architecture", "aws-docs", "azure-docs", "gcp-docs"],
    "database_expert": ["database-tools", "postgres-docs", "mongodb-docs", "redis-docs"],
    "architect": ["c4-architecture", "product-management", "product-strategy", "user-research"],
    "planner": ["project-planning", "product-management", "brainstorming", "internal-comms"],
    "verifier": ["qa-planning", "tdd-methodology"]
}


async def _get_domain_tools(node_id: str, deps: GraphDeps) -> list[Any]:
    """Helper to dynamically fetch tools for a specialized domain expert."""
    # Start with universal developer tools
    from .tools.developer_tools import developer_tools
    tools = list(developer_tools)
    
    # Add skills from the mapping
    skills_to_load = NODE_SKILL_MAP.get(node_id, [])
    logger.debug(f"Loading {len(skills_to_load)} specialized skills for {node_id}")
    
    return tools


async def researcher_step(
    ctx: StepContext[GraphState, GraphDeps, ExecutionStep | str],
) -> str:
    """Unified discovery agent with Web, Codebase, and Workspace capability."""
    logger.info("Researcher: Triangulating context...")
    unified_context = await fetch_unified_context()
    
    # If the dispatcher sent a specific question, use it as the prompt
    step_input = ctx.inputs
    research_query = ctx.state.query
    if isinstance(step_input, ExecutionStep) and step_input.input_data:
        if isinstance(step_input.input_data, dict):
            research_query = step_input.input_data.get("question", research_query)
        elif isinstance(step_input.input_data, str):
            research_query = step_input.input_data

    planner_prompt = load_specialized_prompts("planner")
    architect_prompt = load_specialized_prompts("architect")
    researcher_prompt = load_specialized_prompts("researcher")

    planner = Agent(
        model=ctx.deps.agent_model,
        system_prompt=(
            f"{planner_prompt}\n\n"
            f"Context: {unified_context}"
        ),
    )
    architect = Agent(
        model=ctx.deps.agent_model,
        system_prompt=(
            f"{architect_prompt}\n\n"
            f"Context: {unified_context}"
        ),
    )

    # researcher agent with ALL discovery tools
    researcher = Agent(
        model=ctx.deps.agent_model,
        system_prompt=(
            f"{researcher_prompt}\n\n"
            f"Workspace Context: {unified_context}"
        )
    )
    # Register all discovery toolsets
    researcher.tool(project_search)
    researcher.tool(list_files)
    researcher.tool(get_git_status)
    from .tools.workspace_tools import read_workspace_file
    researcher.tool(read_workspace_file)
    
    # Bind optional MCP research toolsets (web search, etc.)
    for toolset in ctx.deps.mcp_toolsets:
        researcher.toolsets.append(toolset)

    try:
        res = await researcher.run(research_query)
        ctx.state._update_usage(getattr(res, "usage", None))
        
        # Save to registry for other agents to consume
        node_uid = f"researcher_{ctx.state.step_cursor}" 
        ctx.state.results_registry[node_uid] = str(res.data)
        
        return "joiner"
    except Exception as e:
        logger.error(f"Researcher failed: {e}")
        return "Error"


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
) -> GraphPlan | str:
    """Determines the optimal execution plan for the query."""
    deps = ctx.deps
    import time

    emit_graph_event(
        deps.event_queue,
        "planning_started",
        query=ctx.state.query,
    )

    # Track re-planning loops to prevent infinite cycles
    ctx.state.global_research_loops += 1
    if ctx.state.global_research_loops > 3:
        logger.error("Router: Max planning loops exceeded. Aborting.")
        return "Error"

    # Reset cursor for the new plan
    ctx.state.step_cursor = 0

    failure_context = ""
    if ctx.state.error:
        failure_context = f"### PREVIOUS FAILURE CONTEXT\nThe last attempt failed with the following error:\n{ctx.state.error}\nUse this information to update your plan. You may need more research or a different approach."

    try:
        router_prompt = load_specialized_prompts("router")
        router_agent = Agent(
            model=deps.router_model,
            result_type=GraphPlan,
            system_prompt=(
                f"{router_prompt}\n\n"
                f"### FAILURE CONTEXT\n{failure_context}\n\n"
                f"### AVAILABLE STEPS\n{step_info}\n\n"
                f"### PROJECT CONTEXT\n{unified_context}"
            ),
        )
        result = await router_agent.run(ctx.state.query)
        ctx.state._update_usage(getattr(result, "usage", None))
        
        ctx.state.plan = result.data
        ctx.state.step_cursor = 0
        
        emit_graph_event(
            deps.event_queue,
            "planning_completed",
            plan=ctx.state.plan.model_dump(),
            reasoning=ctx.state.plan.metadata.get("reasoning", "Optimal dynamic plan"),
        )

        return "dispatcher"
    except Exception as e:
        logger.error(f"Router planning failed: {e}")
        ctx.state.error = f"Planning failed: {e}"
        return "Error"


async def dispatcher_step(
    ctx: StepContext[GraphState, GraphDeps, Any],
) -> str | list[ExecutionStep] | End[dict]:
    """Manages the execution flow using sequential or parallel steps."""
    if ctx.state.step_cursor >= len(ctx.state.plan.steps):
        return End({"status": "completed", "results": ctx.state.results_registry})

    # Sequential execution case (default for first step or non-parallel)
    current_step = ctx.state.plan.steps[ctx.state.step_cursor]
    
    # Check if this is the start of a parallel batch
    if not current_step.is_parallel:
        ctx.state.step_cursor += 1
        ctx.state.pending_parallel_count = 1
        # Simple string nodes for sequential
        return current_step.node_id

    # Gather all subsequent steps marked for parallel execution
    batch = []
    while (
        ctx.state.step_cursor < len(ctx.state.plan.steps) 
        and ctx.state.plan.steps[ctx.state.step_cursor].is_parallel
    ):
        batch.append(ctx.state.plan.steps[ctx.state.step_cursor])
        ctx.state.step_cursor += 1
    
    # Set the barrier count
    ctx.state.pending_parallel_count = len(batch)
    logger.info(f"Dispatcher: Dispatching parallel batch of {len(batch)} tasks...")

    return batch


async def expert_executor_step(
    ctx: StepContext[GraphState, GraphDeps, ExecutionStep],
) -> str:
    """A generic wrapper for parallel batch execution with local retries."""
    step = ctx.inputs
    node_id = step.node_id
    
    # Reset local retries for this new expert node
    ctx.state.current_node_retries = 0
    max_retries = 2

    while ctx.state.current_node_retries <= max_retries:
        try:
            logger.info(f"Expert Execution: Attempt {ctx.state.current_node_retries + 1}/{max_retries + 1} for node '{node_id}'")
            
            # Special Handling for Researcher
            if node_id == "researcher":
                await researcher_step(ctx)
            
            # Professional Expert Steps
            elif node_id == "python_programmer":
                await python_step(ctx)
            elif node_id == "typescript_programmer" or node_id == "javascript_programmer":
                await typescript_step(ctx)
            elif node_id == "rust_expert":
                await rust_step(ctx)
            elif node_id == "golang_expert":
                await golang_step(ctx)
            elif node_id == "security_auditor":
                await security_step(ctx)
            elif node_id == "qa_expert":
                await qa_step(ctx)
            elif node_id == "debugger_expert" or node_id == "debugger_step":
                await debugger_step(ctx)
            elif node_id == "ui_ux_designer" or node_id == "ui_ux_step":
                await ui_ux_step(ctx)
            elif node_id == "devops_engineer" or node_id == "devops_step":
                await devops_step(ctx)
            elif node_id == "cloud_architect" or node_id == "cloud_step":
                await cloud_step(ctx)
            elif node_id == "database_expert" or node_id == "database_step":
                await database_step(ctx)
            
            # Generic MCP Step
            elif node_id == "mcp_server":
                domain = ""
                input_data = step.input_data
                if isinstance(input_data, dict):
                    domain = input_data.get("domain", "")
                await _execute_domain_logic(ctx, domain)
            
            # Pipeline Steps
            elif node_id == "explorer":
                await explorer_step(ctx)
            elif node_id == "architect":
                await architect_step(ctx)
            elif node_id == "planner":
                await planner_step(ctx)
            elif node_id == "verifier":
                await verifier_step(ctx)

            # Dynamic Discovery Execution
            else:
                from .a2a import discover_agents, A2AClient
                discovered = discover_agents()
                if node_id in discovered:
                    meta = discovered[node_id]
                    # This logic is now mostly handled by the native steps below,
                    # but kept as fallback for legacy direct-calls if needed.
                    await _execute_agent_package_logic(ctx, node_id, meta)

            # Execution successful, clear error and break retry loop
            ctx.state.error = None
            break

            # Execution successful, clear error and break retry loop
            ctx.state.error = None
            break

        except Exception as e:
            logger.error(f"Execution failed for node '{node_id}' (Attempt {ctx.state.current_node_retries + 1}): {e}")
            ctx.state.error = f"Node {node_id} failed: {e}"
            ctx.state.current_node_retries += 1
            
            if ctx.state.current_node_retries > max_retries:
                logger.warning(f"Node '{node_id}' exhausted all retries. Escalating to re-planning.")
                ctx.state.needs_replan = True
                break
            
            # Short sleep before local retry
            await asyncio.sleep(1)

            # Return to appropriate joiner for synchronization
            if node_id in ["researcher", "architect", "planner"]:
                return "research_joiner"
            return "execution_joiner"


async def _execute_agent_package_logic(
    ctx: StepContext[GraphState, GraphDeps, ExecutionStep | Any],
    node_id: str,
    meta: dict,
) -> str:
    """Core logic to execute a specialized agent package (Local or A2A)."""
    deps = ctx.deps
    
    if meta.get("type") == "remote_a2a":
        # Remote A2A Execution
        from .a2a import A2AClient
        peer_url = meta["url"]
        logger.info(f"Expert Execution: Calling remote A2A agent '{node_id}' at {peer_url}")
        client = A2AClient(timeout=deps.approval_timeout or 300.0, ssl_verify=deps.ssl_verify)
        
        # Use the expert's specific question or the original query
        sub_query = ctx.state.query
        step_input = ctx.inputs
        if isinstance(step_input, ExecutionStep) and step_input.input_data:
            if isinstance(step_input.input_data, dict):
                sub_query = step_input.input_data.get("question", sub_query)
            elif isinstance(step_input.input_data, str):
                sub_query = step_input.input_data
                
        res_content = await client.execute_task(peer_url, sub_query)
        ctx.state.results_registry[f"{node_id}_{ctx.state.step_cursor}"] = str(res_content)
    else:
        # Local Agent Package Execution
        pkg_name = meta["package"]
        logger.info(f"Expert Execution: Dynamically loading sub-agent '{pkg_name}'")
        # Load the agent template
        module = importlib.import_module(f"{pkg_name}.agent_server")
        
        # Merge global toolsets with sub-agent context
        sub_graph_bundle = module.agent_template(
            provider=ctx.deps.provider,
            agent_model=ctx.deps.agent_model,
            base_url=ctx.deps.base_url,
            api_key=ctx.deps.api_key,
        )
        # Handle the result which is usually a (Graph, config) tuple
        if isinstance(sub_graph_bundle, tuple) and len(sub_graph_bundle) == 2:
            sub_graph, sub_config = sub_graph_bundle
            
            # Use run_graph helper to preserve SSE events and observability
            res = await run_graph(
                graph=sub_graph,
                config=sub_config,
                query=ctx.state.query,
                eq=deps.event_queue,
            )
            ctx.state.results_registry[f"{node_id}_{ctx.state.step_cursor}"] = str(res.get("results", ""))
        else:
            # Fallback for simple pydantic-ai Agent instances
            res = await sub_graph_bundle.run(ctx.state.query)
            ctx.state._update_usage(getattr(res, "usage", None))
            ctx.state.results_registry[f"{node_id}_{ctx.state.step_cursor}"] = str(res.data)
            
    return "execution_joiner"


async def agent_package_step(
    ctx: StepContext[GraphState, GraphDeps, ExecutionStep | Any],
    node_id: str,
) -> str:
    """Functional step for a specific agent package."""
    from .a2a import discover_agents
    discovered = discover_agents()
    if node_id not in discovered:
        logger.error(f"Agent package node '{node_id}' not found in discovery.")
        return "Error"
        
    meta = discovered[node_id]
    return await _execute_agent_package_logic(ctx, node_id, meta)


async def join_step(
    ctx: StepContext[GraphState, GraphDeps, Any],
) -> str | None:
    """Synchronizes parallel executions by decrementing the pending count."""
    async with lock:
        ctx.state.pending_parallel_count -= 1
        count = ctx.state.pending_parallel_count
        logger.debug(f"Join: Remaining parallel tasks = {count}")
        
        if count <= 0:
            logger.info("Join: All parallel tasks completed.")
            if ctx.state.needs_replan:
                logger.warning("Join: Re-planning required due to failures. Routing to router_step.")
                ctx.state.needs_replan = False # Reset for the next plan
                return "router_step"
            return "dispatcher"
    
    # Still waiting for others
    return None


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
    # Merged prompt for both validation and verification
    validator_prompt = load_specialized_prompts("validator")

    from pydantic_ai import Agent

    verifier = Agent(
        model=ctx.deps.agent_model,
        system_prompt=(
            f"{validator_prompt}\n\n"
            f"### ARCHITECTURAL INTENT\n{ctx.state.architectural_decisions}\n\n"
            f"### EXPLORATION FINDINGS\n{ctx.state.exploration_notes}"
        ),
    )

    for toolset in ctx.deps.mcp_toolsets:
        verifier.toolsets.append(toolset)

    try:
        res = await verifier.run(f"Verify the solution for: {ctx.state.query}")
        result_text = str(res.data) if hasattr(res, "data") else str(res)
        
        if "VERDICT: PASS" in result_text.upper() or "IS_VALID: TRUE" in result_text.upper():
            return End({"status": "success", "summary": result_text})

        ctx.state.verification_feedback = result_text
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
        validator_prompt = load_specialized_prompts("validator")
        validator_agent = Agent(
            model=deps.router_model,
            result_type=ValidationResult,
            system_prompt=(
                f"{validator_prompt}\n\n"
                f"### CONTEXT\n"
                f"Original Query: {ctx.state.query}\n"
                f"Agent Result: {result_text}\n"
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


async def python_programmer_step(ctx: StepContext[GraphState, GraphDeps, None]) -> str | End[Any]:
    """Specialized Python expert step."""
    return await _execute_specialized_step(ctx, "python_programmer")


async def golang_programmer_step(ctx: StepContext[GraphState, GraphDeps, None]) -> str | End[Any]:
    """Specialized Golang expert step."""
    return await _execute_specialized_step(ctx, "golang_programmer")


async def typescript_programmer_step(
    ctx: StepContext[GraphState, GraphDeps, None],
) -> str | End[Any]:
    """Specialized TypeScript expert step."""
    return await _execute_specialized_step(ctx, "typescript_programmer")


async def rust_programmer_step(ctx: StepContext[GraphState, GraphDeps, None]) -> str | End[Any]:
    """Specialized Rust expert step."""
    return await _execute_specialized_step(ctx, "rust_programmer")


async def security_auditor_step(
    ctx: StepContext[GraphState, GraphDeps, None],
) -> str | End[Any]:
    """Specialized Security expert step."""
    return await _execute_specialized_step(ctx, "security_auditor")


async def javascript_programmer_step(
    ctx: StepContext[GraphState, GraphDeps, None],
) -> str | End[Any]:
    """Specialized JavaScript expert step."""
    return await _execute_specialized_step(ctx, "javascript_programmer")


async def c_programmer_step(ctx: StepContext[GraphState, GraphDeps, None]) -> str | End[Any]:
    """Specialized C expert step."""
    return await _execute_specialized_step(ctx, "c_programmer")


async def cpp_programmer_step(ctx: StepContext[GraphState, GraphDeps, None]) -> str | End[Any]:
    """Specialized C++ expert step."""
    return await _execute_specialized_step(ctx, "cpp_programmer")


async def qa_expert_step(ctx: StepContext[GraphState, GraphDeps, None]) -> str | End[Any]:
    """Specialized QA expert step."""
    return await _execute_specialized_step(ctx, "qa_expert")


async def debugger_expert_step(
    ctx: StepContext[GraphState, GraphDeps, None],
) -> str | End[Any]:
    """Specialized Debugging expert step."""
    return await _execute_specialized_step(ctx, "debugger_expert")


async def ui_ux_designer_step(ctx: StepContext[GraphState, GraphDeps, None]) -> str | End[Any]:
    """Specialized UI/UX expert step."""
    return await _execute_specialized_step(ctx, "ui_ux_designer")


async def devops_engineer_step(ctx: StepContext[GraphState, GraphDeps, None]) -> str | End[Any]:
    """Specialized DevOps expert step."""
    return await _execute_specialized_step(ctx, "devops_engineer")


async def cloud_architect_step(ctx: StepContext[GraphState, GraphDeps, None]) -> str | End[Any]:
    """Specialized Cloud expert step."""
    return await _execute_specialized_step(ctx, "cloud_architect")


async def database_expert_step(
    ctx: StepContext[GraphState, GraphDeps, None],
) -> str | End[Any]:
    """Specialized Database expert step."""
    return await _execute_specialized_step(ctx, "database_expert")


async def _execute_specialized_step(
    ctx: StepContext[GraphState, GraphDeps, None], prompt_name: str
) -> str | End[Any]:
    """Shared logic for specialized steps using migrated prompts and skill injection."""
    prompt = load_specialized_prompts(prompt_name)
    
    # Dynamic Skill Distribution
    custom_tools = await _get_domain_tools(prompt_name, ctx.deps)
    
    memory_instruction = load_specialized_prompts("memory_instruction")
    
    agent = Agent(
        model=ctx.deps.agent_model,
        system_prompt=(
            f"{memory_instruction}\n\n"
            f"{prompt}\n\n"
            f"### CONTEXT\n{ctx.state.exploration_notes}"
        ),
        tools=custom_tools,
    )
    for toolset in ctx.deps.mcp_toolsets:
        agent.toolsets.append(toolset)

    try:
        res = await agent.run(ctx.state.query)
        ctx.state._update_usage(getattr(res, "usage", None))
        ctx.state.results[prompt_name] = str(res.data)
        
        # In Dynamic Plan mode, we return to the joiner/dispatcher
        if ctx.state.plan and ctx.state.plan.steps:
            return "joiner"

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


async def memory_selection_step(
    ctx: StepContext[GraphState, GraphDeps, Any],
) -> str | End[Any]:
    """Identifies relevant memories to load for the current query."""
    logger.info("Memory Selection: Identifying relevant context...")
    prompt_content = load_specialized_prompts("memory_selection")
    
    # Discovery of local memory files
    root = ctx.state.project_root or os.getcwd()
    memories = []
    for p in Path(root).rglob("*.md"):
        if ".gemini" in str(p) or "node_modules" in str(p):
            continue
        try:
            content = p.read_text(encoding="utf-8", errors="ignore")
            # Extract description from frontmatter if available
            description = "General project memory"
            if content.startswith("---"):
                match = re.search(r"description:\s*(.*)", content)
                if match:
                    description = match.group(1).strip()
            memories.append(f"- {p.name}: {description}")
        except Exception:
            pass
            
    from pydantic_ai import Agent
    
    # We use a simple list of memories as context
    selectors = Agent(
        model=ctx.deps.agent_model,
        system_prompt=prompt_content,
        result_type=dict # Simple selected filenames
    )
    
    try:
        res = await selectors.run(
            f"Query: {ctx.state.query}\n\nAvailable memories:\n" + "\n".join(memories[:20])
        )
        selected = res.data.get("selected_memories", [])
        logger.info(f"Memory Selection: Selected {len(selected)} relevant files: {selected}")
        
        # Load the selected content into the state for other agents to consume
        loaded_context = []
        for filename in selected:
            for p in Path(root).rglob(filename):
                 loaded_context.append(f"### {filename}\n{p.read_text(encoding='utf-8', errors='ignore')}")
                 break
        
        ctx.state.exploration_notes += "\n\n### SELECTED MEMORIES\n" + "\n\n".join(loaded_context)
        return ctx.inputs # Continue to original intended node
    except Exception as e:
        logger.error(f"Memory Selection failed: {e}")
        return ctx.inputs

async def planner_step(
    ctx: StepContext[GraphState, GraphDeps, MultiDomainChoice | None],
) -> TaskList | str | End[Any]:
    """
    Consolidated planning step. Handles both project-mode decomposition and 
    simple sequential task list creation.
    """
    logger.info("Planner: Analyzing request and creating execution path...")
    planner_prompt = load_specialized_prompts("planner")
    memory_instruction = load_specialized_prompts("memory_instruction")
    unified_context = await fetch_unified_context()

    from pydantic_ai import Agent

    planner = Agent(
        model=ctx.deps.agent_model,
        result_type=TaskList,
        system_prompt=(
            f"{memory_instruction}\n\n"
            f"{planner_prompt}\n\n"
            f"### CONTEXT\n{unified_context}\n\n"
            f"### FINDINGS\n{ctx.state.exploration_notes}"
        ),
    )

    # Integrated Worktree & Development Tools
    planner.tool(get_git_status)
    planner.tool(create_worktree)
    planner.tool(list_worktrees)
    planner.tool(project_search)

    try:
        # Determine goal based on input type or state
        goal = ctx.state.query
        if isinstance(ctx.inputs, MultiDomainChoice):
            goal = f"Goal: {goal}\nReasoning: {ctx.inputs.reasoning}"

        res = await planner.run(goal)
        ctx.state.task_list = res.data
        ctx.state.sync_to_disk()

        # In 'Plan' mode, we might just end here
        if ctx.state.mode == "plan":
            return End({"status": "planned", "plan": ctx.state.task_list.model_dump()})

        # Return the next node in the graph (usually ProjectExecutor or similar)
        return "ProjectExecutor"
    except Exception as e:
        logger.error(f"Planning failed: {e}")
        return "router_step" # Retry via re-planning


async def project_executor_step(
    ctx: StepContext[GraphState, GraphDeps, Task],
) -> Task:
    """Executes a single task from a project plan with specialized agent support."""
    task = ctx.inputs
    task.status = TaskStatus.IN_PROGRESS

    # Specialist Mapping
    specialist_map = {
        "python": "python_programmer",
        "c": "c_programmer",
        "cpp": "cpp_programmer",
        "golang": "golang_programmer",
        "javascript": "javascript_programmer",
        "typescript": "typescript_programmer",
        "rust": "rust_programmer",
        "security": "security_auditor",
        "qa": "qa_expert",
        "debugger": "debugger_expert",
        "ui_ux": "ui_ux_designer",
        "devops": "devops_engineer",
        "cloud": "cloud_architect",
        "database": "database_expert",
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
    _router = g.step(router_step, node_id="router")
    _validator = g.step(validator_step, node_id="validator")
    _planner = g.step(planner_step, node_id="planner")
    _executor = g.step(project_executor_step, node_id="project_executor")
    _error = g.step(error_recovery_step, node_id="error_recovery")

    # Dynamic Dispatcher Nodes
    _dispatcher = g.step(dispatcher_step, node_id="dispatcher")
    _expert_executor = g.step(expert_executor_step, node_id="expert_executor")
    
    # Dual Joiners for Phase Separation
    _research_joiner = g.step(join_step, node_id="research_joiner")
    _execution_joiner = g.step(join_step, node_id="execution_joiner")
    _joiner = g.step(join_step, node_id="joiner") # Legacy fallback

    _coordinator = g.step(coordinator_step, node_id="coordinator")
    _architect = g.step(architect_step, node_id="architect")
    _verifier = g.step(verifier_step, node_id="verifier")
    _critique = g.step(critique_step, node_id="critique")

    # Native Developer Steps
    _researcher = g.step(researcher_step, node_id="researcher")
    _python = g.step(python_programmer_step, node_id="python_programmer")
    _c = g.step(c_programmer_step, node_id="c_programmer")
    _cpp = g.step(cpp_programmer_step, node_id="cpp_programmer")
    _golang = g.step(golang_programmer_step, node_id="golang_programmer")
    _javascript = g.step(javascript_programmer_step, node_id="javascript_programmer")
    _typescript = g.step(typescript_programmer_step, node_id="typescript_programmer")
    _security = g.step(security_auditor_step, node_id="security_auditor")
    _qa = g.step(qa_expert_step, node_id="qa_expert")
    _debugger = g.step(debugger_expert_step, node_id="debugger_expert")
    _ui_ux = g.step(ui_ux_designer_step, node_id="ui_ux_designer")
    _devops = g.step(devops_engineer_step, node_id="devops_engineer")
    _cloud = g.step(cloud_architect_step, node_id="cloud_architect")
    _database = g.step(database_expert_step, node_id="database_expert")
    _rust = g.step(rust_programmer_step, node_id="rust_programmer")
    _memory_selection = g.step(memory_selection_step, node_id="memory_selection")

    # # Dynamic MCP Steps
    _mcp_router = g.step(dynamic_mcp_routing_step, node_id="mcp_router")
    _mcp_server = g.step(mcp_server_step, node_id="mcp_server_execution")

    # Approval Gate
    _approval = g.step(approval_gate_step, node_id="approval_gate")

    # Usage Guard
    _usage_guard = g.step(usage_guard_step, node_id="usage_guard")

    # --- Dynamic Agent Package Registration ---
    from .a2a import discover_agents
    discovered_agents_map = discover_agents()
    agent_nodes = {}
    
    for tag, meta in discovered_agents_map.items():
        # Create a unique step for each agent using functools.partial
        # node_id must be the tag so dispatcher can match it
        step_func = functools.partial(agent_package_step, node_id=tag)
        # Pydantic Graph uses the function name for default node IDs, 
        # so we must explicitly set node_id.
        agent_nodes[tag] = g.step(step_func, node_id=tag)
        logger.debug(f"Registered native graph step for agent: {tag}")

    # Define Graph Topology
    g.add(
        # Start -> UsageGuard -> Router -> Planner -> Dispatcher
        g.edge_from(g.start_node).to(_usage_guard),
        g.edge_from(_usage_guard).to(_router),
        g.edge_from(_router).to(_planner),
        g.edge_from(_planner).to(_memory_selection),
        g.edge_from(_memory_selection).to(_dispatcher),

        # Dispatcher: The Main Dynamic Branching Logic
        # Sequential Routes: routes by node_id (string) returned by dispatcher_step
        g.edge_from(_dispatcher).to(
            _python, _typescript, _rust, _golang, _security, 
            _qa, _debugger, _ui_ux, _devops, _cloud, 
            _database, _researcher, _architect, _planner, 
            _verifier, _mcp_router, _error,
            _c, _cpp, _javascript, _joiner,
            *agent_nodes.values()
        ),
        
        # Type 2: Parallel Batch Execution
        g.edge_from(_dispatcher).map().to(_expert_executor),

        # Expert Nodes: Return to Joiner for synchronization
        g.edge_from(_researcher).to(_research_joiner),
        g.edge_from(_architect).to(_research_joiner),
        g.edge_from(_planner).to(_research_joiner),

        g.edge_from(_python).to(_execution_joiner),
        g.edge_from(_golang).to(_execution_joiner),
        g.edge_from(_typescript).to(_execution_joiner),
        g.edge_from(_rust).to(_execution_joiner),
        g.edge_from(_c).to(_execution_joiner),
        g.edge_from(_cpp).to(_execution_joiner),
        g.edge_from(_javascript).to(_execution_joiner),
        g.edge_from(_qa).to(_execution_joiner),
        g.edge_from(_security).to(_execution_joiner),
        g.edge_from(_debugger).to(_execution_joiner),
        g.edge_from(_ui_ux).to(_execution_joiner),
        g.edge_from(_devops).to(_execution_joiner),
        g.edge_from(_cloud).to(_execution_joiner),
        g.edge_from(_database).to(_execution_joiner),
        
        # Add return edges for all discovered agent package nodes
        *(g.edge_from(node).to(_execution_joiner) for node in agent_nodes.values()),

        g.edge_from(_expert_executor).to(_execution_joiner),

        # Special Handling for MCP Parallel Flow
        g.edge_from(_mcp_router).map().to(_mcp_server),
        g.edge_from(_mcp_server).to(_execution_joiner),

        # Joiners: Return control to Dispatcher
        g.edge_from(_research_joiner).to(_dispatcher),
        g.edge_from(_execution_joiner).to(_dispatcher),
        g.edge_from(_joiner).to(_dispatcher),

        # Error handling and Finalization
        g.edge_from(_error).to(g.end_node),
        g.edge_from(_verifier).to(g.end_node),
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
