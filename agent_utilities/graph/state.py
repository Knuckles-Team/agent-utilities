#!/usr/bin/python
# coding: utf-8
"""Graph State Module.

This module defines the core data structures for managing the state and
dependencies of the agent graph orchestrator. It includes GraphDeps for
runtime configuration and GraphState for tracking queries, plans,
execution results, and usage statistics across the agent lifecycle.
"""

from __future__ import annotations

import asyncio
import os
import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from pydantic_ai import Agent

from ..config import (
    DEFAULT_PROVIDER,
    DEFAULT_LLM_BASE_URL,
    DEFAULT_LLM_API_KEY,
    DEFAULT_SSL_VERIFY,
    DEFAULT_ROUTER_MODEL,
    DEFAULT_GRAPH_AGENT_MODEL,
    DEFAULT_GRAPH_ROUTER_TIMEOUT,
    DEFAULT_GRAPH_VERIFIER_TIMEOUT,
    TOOL_GUARD_MODE,
)


from ..models import (
    TaskList,
    ProgressLog,
    SprintContract,
    UsageStatistics,
    GraphPlan,
    ParallelBatch,
)

logger = logging.getLogger(__name__)


@dataclass
class GraphDeps:
    """Runtime dependencies and configuration for graph execution.

    This container is passed to every node in the graph, providing
    access to shared resources like MCP toolsets, model configurations,
    and event queues.
    """

    tag_prompts: dict[str, str]
    tag_env_vars: dict[str, str]
    mcp_toolsets: list[Any]
    nodes: dict[str, Any] = field(default_factory=dict)
    """Registry of graph nodes (Steps) for explicit transitions.
    Typed as Any to avoid circular dependencies with Step."""
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
    router_timeout: float = DEFAULT_GRAPH_ROUTER_TIMEOUT
    verifier_timeout: float = DEFAULT_GRAPH_VERIFIER_TIMEOUT
    project_root: str = ""
    max_parallel_agents: int = 3
    auto_approve_plan: bool = False
    auto_approve_tasks: bool = False
    approval_timeout: float = 0.0
    tool_guard_mode: str = TOOL_GUARD_MODE
    message_history_cache: dict[str, Any] = field(default_factory=dict)
    server_health: dict[str, Any] = field(default_factory=dict)
    discovery_metadata: dict[str, list[str]] = field(default_factory=dict)
    """Map of server_id to list of tool names found during discovery phase."""


@dataclass
class GraphState:
    """The central state container for an orchestrator session.

    This class tracks the entire lifecycle of a request, from the initial
    query and high-level plan to parallel execution results and final
    verification feedback. It supports persistence and usage monitoring.
    """

    query: str
    """The original user query."""

    query_parts: list[dict[str, Any]] = field(default_factory=list)
    """Rich multi-modal parts of the user query (text, images, etc.)."""

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

    verification_attempts: int = 0
    """Number of times the verifier has attempted to validate the results."""

    deferred_events: list[dict[str, Any]] = field(default_factory=list)
    """Events received asynchronously during execution that need to be processed by the dispatcher."""

    pending_batch: Optional[ParallelBatch] = None
    """Temporary storage for a batch of tasks during transition from dispatcher to processor."""

    needs_replan: bool = False
    """Flag to trigger a re-evaluation of the graph plan due to implementation failures."""

    node_transitions: int = 0
    """Total number of node transitions during this execution. Used for infinite-loop protection."""

    MAX_NODE_TRANSITIONS: int = 50
    """Hard ceiling on the total number of node transitions before the graph force-terminates."""

    def _update_usage(self, result_usage: Any):
        """Update the accumulated session usage statistics.

        Processes usage metadata from an agent run and increments input,
        output, and total token counts. Also performs simple cost
        estimation.

        Args:
            result_usage: The usage object returned by a pydantic-ai Agent.

        """
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
        """Persist key state artifacts to the local workspace for inspection.

        Serializes task lists, progress logs, and usage statistics to
        JSON files within the project root.

        Args:
            artifact_prefix: Optional prefix for the generated filenames.

        """
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
        """Recover project state from existing JSON artifacts in the workspace.

        Attempts to reload task lists, logs, and usage data from the
        file system to resume a previous session.

        Returns:
            True if any artifacts were successfully loaded, False otherwise.

        """
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
