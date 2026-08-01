"""Governed upstream DynamicWorkflow integration for GraphOS.

CONCEPT:AU-ORCH.execution.dynamic-workflows

``pydantic-ai-harness`` owns exactly one thing here: evaluating the
model-authored orchestration script in its Monty sandbox.  Every catalog
function re-enters :meth:`agent_utilities.orchestration.manager.Orchestrator.execute_agent`;
the script never receives a connector tool, model client, graph handle, or
credential.  Tenant/session authority, agent and skill resolution, tool
contracts, model-class routing, budgets, cancellation, and RunTrace/ToolCall
provenance therefore remain on the one GraphOS execution plane.

The optional Harness dependency is loaded lazily.  Callers must explicitly
choose the ordinary stored-DAG runner as a fallback when it is unavailable;
execution failures never silently change orchestration engines.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, version
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from agent_utilities.capabilities.checkpointing import Checkpoint, CheckpointStore

from agent_utilities.core.event_loop import run_blocking_ordered
from agent_utilities.models.execution_manifest import ExecutionManifest, ExecutionResult
from agent_utilities.models.graph import (
    GraphExecutionEvidence,
    GraphPlan,
    GraphTaskEvidence,
    GraphTransitionEvidence,
)
from agent_utilities.models.sdd import Task
from agent_utilities.orchestration.run_identity import new_run_id

logger = logging.getLogger(__name__)


class DynamicWorkflowUnavailableError(RuntimeError):
    """The optional upstream DynamicWorkflow runtime cannot be used."""


class WorkflowResourceLimits(BaseModel):
    """Host and sandbox limits for one governed workflow run."""

    max_duration_secs: float = Field(default=300.0, gt=0, le=3600.0)
    max_concurrency: int = Field(default=8, ge=1, le=64)
    max_memory_bytes: int = Field(
        default=256 * 1024 * 1024,
        ge=16 * 1024 * 1024,
        le=2 * 1024 * 1024 * 1024,
    )
    max_tokens_per_agent: int | None = Field(default=None, ge=1)
    orchestrator_token_budget: int | None = Field(default=None, ge=1)


class DelegationStep(BaseModel):
    """One reviewed GraphOS catalog entry available to the sandbox script."""

    id: str
    description: str
    depends_on: list[str] = Field(default_factory=list)
    kind: Literal["auto", "agent", "skill"] = "auto"
    target_name: str | None = None
    allowed_tools: list[str] = Field(default_factory=list)
    required_tools: list[str] = Field(default_factory=list)
    tool_server: str | None = None
    model_class: Literal["economy", "standard"] = "standard"
    reasoning_effort: Literal["low", "medium", "high"] | None = None
    max_steps: int = Field(default=30, ge=1, le=300)
    model_menu: list[str] = Field(default_factory=list)
    model_id: str | None = None
    timeout_secs: float | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def validate_contract(self) -> DelegationStep:
        if self.model_id and self.model_menu and self.model_id not in self.model_menu:
            raise ValueError("model_id must be one of model_menu")
        if self.required_tools and self.allowed_tools:
            missing = set(self.required_tools) - set(self.allowed_tools)
            if missing:
                raise ValueError(f"required_tools must be allowed: {sorted(missing)}")
        if self.tool_server and self.kind != "skill":
            raise ValueError("tool_server requires kind='skill'")
        if self.target_name is not None and not self.target_name.strip():
            raise ValueError("target_name must not be blank")
        return self


class WorkflowScriptEvidence(BaseModel):
    """Privacy-safe evidence for one upstream ``run_workflow`` tool call."""

    tool_call_id: str
    sha256: str
    byte_count: int
    line_count: int


class ChildRunEvidence(BaseModel):
    """Trace linkage for one catalog function dispatched through GraphOS.

    ``outcome`` is one of ``ok`` | ``failed`` | ``timeout`` | ``cancelled`` |
    ``replayed`` -- ``replayed`` is the truthful marker for a resumed run that
    reused a prior halted attempt's persisted output instead of re-dispatching
    through GraphOS (CONCEPT:AU-ORCH.execution.dynamic-workflow-resume).
    """

    step_id: str = ""
    agent_name: str
    run_id: str
    trace_ref: str
    outcome: str


class GovernedDynamicWorkflowResult(BaseModel):
    """Structured result and trace evidence for an upstream workflow run.

    ``resumed``/``replayed_step_ids`` are the truthfulness contract for the
    resume path (CONCEPT:AU-ORCH.execution.dynamic-workflow-resume): a run
    that reused any prior catalog-call result from a halted attempt of the
    SAME ``workflow_run_id`` is never reported identical to a clean single-shot
    success — ``resumed`` is true and the reused step ids are named.
    """

    status: Literal["completed"] = "completed"
    output: Any
    workflow_run_id: str
    trace_ref: str
    backend: str = "pydantic-ai-harness.dynamic_workflow.DynamicWorkflow"
    upstream_version: str
    script_evidence: list[WorkflowScriptEvidence] = Field(default_factory=list)
    child_runs: list[ChildRunEvidence] = Field(default_factory=list)
    usage: dict[str, int] = Field(default_factory=dict)
    fallback_used: bool = False
    resumed: bool = False
    replayed_step_ids: list[str] = Field(default_factory=list)
    checkpoint_ids: list[str] = Field(default_factory=list)


@dataclass
class _WorkflowRuntime:
    semaphore: asyncio.Semaphore
    completed: set[str] = field(default_factory=set)
    child_runs: list[ChildRunEvidence] = field(default_factory=list)
    # Resume state (CONCEPT:AU-ORCH.execution.dynamic-workflow-resume). ``engine``/
    # ``workflow_run_id`` let a completed dispatch persist itself immediately;
    # ``resume_cache`` is pre-seeded from prior persisted entries for the SAME
    # ``workflow_run_id`` so a catalog call already completed before a halt
    # (budget exhaustion, cancellation, process restart) short-circuits to the
    # persisted output instead of re-dispatching through GraphOS -- the host
    # choke point that makes "restart produces no duplicate ToolCalls" hold
    # regardless of what script the model writes on retry.
    engine: Any = None
    workflow_run_id: str = ""
    resume_cache: dict[tuple[str, str], dict[str, Any]] = field(default_factory=dict)
    persisted_entries: list[dict[str, Any]] = field(default_factory=list)


def _load_upstream_dynamic_workflow() -> tuple[type[Any], type[Any], str]:
    """Load the pinned Harness API or fail with one actionable error."""

    try:
        from pydantic_ai_harness.dynamic_workflow import (
            DynamicWorkflow,
            WorkflowAgent,
        )
    except (ImportError, AttributeError) as exc:
        raise DynamicWorkflowUnavailableError(
            "pydantic-ai-harness DynamicWorkflow is unavailable; install "
            "agent-utilities[dynamic-workflow] (Harness >=0.14,<0.15), or "
            "explicitly use the stored-DAG fallback"
        ) from exc
    try:
        harness_version = version("pydantic-ai-harness")
    except PackageNotFoundError:
        harness_version = "unknown"
    return DynamicWorkflow, WorkflowAgent, harness_version


def _safe_harness_version() -> str:
    """Best-effort Harness version for a failure-path evidence record."""

    try:
        return version("pydantic-ai-harness")
    except PackageNotFoundError:
        return "unknown"


async def _await_with_cancellation(
    awaitable: Any,
    cancellation: asyncio.Event | None,
) -> Any:
    """Await one operation while making a shared cancellation event authoritative."""

    operation = asyncio.create_task(awaitable)
    cancelled: asyncio.Task[bool] | None = None
    try:
        if cancellation is None:
            return await operation
        if cancellation.is_set():
            raise asyncio.CancelledError("governed dynamic workflow cancelled")
        cancelled = asyncio.create_task(cancellation.wait())
        done, _ = await asyncio.wait(
            {operation, cancelled}, return_when=asyncio.FIRST_COMPLETED
        )
        if cancelled in done:
            raise asyncio.CancelledError("governed dynamic workflow cancelled")
        return await operation
    finally:
        if cancelled is not None and not cancelled.done():
            cancelled.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await cancelled
        if not operation.done():
            operation.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await operation


# ---------------------------------------------------------------------------
# Resume cache (CONCEPT:AU-ORCH.execution.dynamic-workflow-resume)
#
# Upstream pydantic-graph v2 removed ``pydantic_graph.persistence`` and a Monty
# ``run_workflow`` sandbox is stateless per attempt ("a fresh sandbox with no
# accumulated state"), so there is no runnable graph-resume token to restore.
# What IS real and restorable is the completed-catalog-call ledger this module
# already builds (``ChildRunEvidence``): a halted attempt's successfully
# dispatched sub-agent calls are real, side-effecting GraphOS runs whose
# outputs must never be silently redone. This cache persists that ledger,
# keyed by ``workflow_run_id`` (stable across a halt + restart because the
# caller reuses the SAME id), and seeds it back in on the next attempt so a
# catalog call already completed short-circuits to its persisted output
# instead of re-entering GraphOS -- the host choke point, independent of
# whatever script the model happens to write on retry.
#
# This is a SEPARATE node from the privacy-redacted ``RunTrace``/``ToolCall``
# audit trail: replay requires the literal task/output value, not a digest.
# ---------------------------------------------------------------------------

_RESUME_NODE_LABEL = "WorkflowResumeState"


def _resume_state_node_id(workflow_run_id: str) -> str:
    return f"workflow-resume:{workflow_run_id}"


def _load_resume_cache(engine: Any, workflow_run_id: str) -> list[dict[str, Any]]:
    """Best-effort load of prior completed dispatches for this workflow run."""

    if engine is None or not workflow_run_id:
        return []
    node_id = _resume_state_node_id(workflow_run_id)
    try:
        if getattr(engine, "backend_type", None) == "rust":
            if not engine.has_node(node_id):
                return []
            raw = engine[node_id].get("entries_json", "[]")
        elif hasattr(engine, "graph"):
            if node_id not in engine.graph:
                return []
            raw = engine.graph.nodes[node_id].get("entries_json", "[]")
        else:
            return []
        entries = json.loads(raw or "[]")
        return entries if isinstance(entries, list) else []
    except Exception as exc:  # noqa: BLE001 — a corrupt/absent cache must not block a run
        logger.warning("Could not load DynamicWorkflow resume cache: %s", exc)
        return []


def _save_resume_cache(
    engine: Any, workflow_run_id: str, entries: list[dict[str, Any]]
) -> None:
    """Best-effort persist of the full completed-dispatch ledger, overwrite semantics."""

    if engine is None or not workflow_run_id:
        return
    node_id = _resume_state_node_id(workflow_run_id)
    try:
        engine.add_node(
            node_id,
            _RESUME_NODE_LABEL,
            properties={
                "id": node_id,
                "workflow_run_id": workflow_run_id,
                "entries_json": json.dumps(
                    entries, sort_keys=True, separators=(",", ":"), default=str
                ),
                "entry_count": len(entries),
            },
        )
    except Exception as exc:  # noqa: BLE001 — resume is best-effort, never fails the dispatch
        logger.warning("Could not persist DynamicWorkflow resume cache: %s", exc)


def _catalog_topology(steps: list[DelegationStep]) -> str:
    """Deterministic structural rendering of the reviewed catalog DAG."""

    lines = [
        f"{step.id}<-{','.join(sorted(step.depends_on))}:{step.kind}:{step.target_name or ''}"
        for step in steps
    ]
    return "\n".join(sorted(lines))


def _content_digest(value: str) -> str:
    return f"sha256:{hashlib.sha256(value.encode('utf-8')).hexdigest()}"


class _CapturingCheckpointStore(CheckpointStore):
    """Wrap a ``CheckpointStore``, capturing every saved checkpoint id.

    The conductor agent is given a real ``CheckpointMiddleware`` (one message-
    history snapshot per ``run_workflow`` tool call) by default -- "restore and
    continue a run" for the DynamicWorkflow path is genuinely available via
    :func:`agent_utilities.capabilities.checkpointing.fork_from_checkpoint`.
    This wrapper makes the ids inspectable as execution evidence without
    depending on ``GraphCheckpointStore.list()`` (not yet a real Cypher query).
    """

    def __init__(self, inner: CheckpointStore) -> None:
        self._inner = inner
        self.ids: list[str] = []

    async def save(self, checkpoint: Checkpoint) -> None:
        await self._inner.save(checkpoint)
        self.ids.append(checkpoint.id)

    async def get(self, checkpoint_id: str) -> Checkpoint | None:
        return await self._inner.get(checkpoint_id)

    async def list(self, limit: int = 10) -> list[Checkpoint]:
        return await self._inner.list(limit=limit)


def _string_list(value: Any) -> list[str]:
    """Normalize reviewed string-list metadata without accepting scalar coercion."""

    if not isinstance(value, (list, tuple, set)):
        return []
    return [str(item) for item in value if str(item).strip()]


def _optional_string(value: Any) -> str | None:
    """Return a non-empty metadata string."""

    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def _delegation_kind(
    metadata: dict[str, Any], assigned_to: str | None
) -> Literal["auto", "agent", "skill"]:
    """Resolve the stored catalog entry kind from reviewed Task metadata."""

    explicit = str(metadata.get("delegation_kind") or "").strip().lower()
    if explicit == "auto":
        return "auto"
    if explicit == "agent":
        return "agent"
    if explicit == "skill":
        return "skill"
    if _optional_string(metadata.get("skill_name")):
        return "skill"
    if _optional_string(metadata.get("agent_name")) or _optional_string(assigned_to):
        return "agent"
    return "auto"


def _delegation_target(
    metadata: dict[str, Any],
    *,
    kind: str,
    assigned_to: str | None,
) -> str | None:
    """Resolve the governed agent/skill target while keeping the catalog id stable."""

    if kind == "skill":
        return _optional_string(metadata.get("skill_name"))
    if kind == "agent":
        return _optional_string(metadata.get("agent_name")) or _optional_string(
            assigned_to
        )
    return None


def _result_payload(raw: Any) -> tuple[Any, str]:
    """Return the child output and its truthful GraphOS terminal outcome."""

    try:
        payload = json.loads(raw) if isinstance(raw, str) else raw
    except (TypeError, ValueError):
        return raw, "ok"
    if not isinstance(payload, dict):
        return payload, "ok"
    summary = payload.get("run_summary")
    outcome = str(summary.get("outcome") or "ok") if isinstance(summary, dict) else "ok"
    return payload.get("output", raw), outcome


_GRAPHOS_AGENT_CLASS: type[Any] | None = None


def _graphos_agent_class() -> type[Any]:
    global _GRAPHOS_AGENT_CLASS
    if _GRAPHOS_AGENT_CLASS is not None:
        return _GRAPHOS_AGENT_CLASS

    from pydantic_ai import Agent
    from pydantic_ai.agent import WrapperAgent
    from pydantic_ai.run import AgentRunResult

    class GraphOSWorkflowAgent(WrapperAgent[Any, str]):
        """Harness-compatible agent facade over the canonical GraphOS dispatcher."""

        def __init__(
            self,
            *,
            orchestrator: Any,
            step: DelegationStep,
            workflow_run_id: str,
            runtime: _WorkflowRuntime,
            cancellation: asyncio.Event | None,
            limits: WorkflowResourceLimits,
            trace_carrier: dict[str, str],
        ) -> None:
            description = step.description
            if step.depends_on:
                description += (
                    " Host-enforced prerequisites: "
                    + ", ".join(sorted(step.depends_on))
                    + "."
                )
            metadata_agent = Agent(
                model=None,
                name=step.id,
                description=description,
                defer_model_check=True,
            )
            super().__init__(metadata_agent)
            self._orchestrator = orchestrator
            self._step = step
            self._workflow_run_id = workflow_run_id
            self._runtime = runtime
            self._cancellation = cancellation
            self._limits = limits
            self._trace_carrier = trace_carrier

        # Harness invokes only ``run(task)`` on catalog entries.  Repeating the
        # complete, provider-specific AbstractAgent overload set here would
        # expose arguments this GraphOS boundary intentionally refuses.
        async def run(  # type: ignore[override]
            self,
            user_prompt: Any = None,
            **_kwargs: Any,
        ) -> AgentRunResult[str]:
            if not isinstance(user_prompt, str):
                raise TypeError("DynamicWorkflow catalog tasks must be strings")
            missing = set(self._step.depends_on) - self._runtime.completed
            if missing:
                raise RuntimeError("GraphOS dependency gate refused this catalog call")
            if self._step.model_id:
                raise RuntimeError(
                    "exact model_id selection is not supported on the GraphOS "
                    "DynamicWorkflow boundary; use governed model_class routing"
                )

            from agent_utilities.observability.correlation import bind_carrier
            from agent_utilities.observability.trace_ontology import trace_id

            # Resume short-circuit: a catalog call already completed for this
            # EXACT (step, task) pair in a prior halted attempt of the SAME
            # ``workflow_run_id`` returns its persisted output directly --
            # GraphOS's ``execute_agent`` is never re-entered, so this never
            # produces a duplicate ``:ToolCall``. Reported outcome is the
            # honest "replayed" (never "ok"), so a resumed run is never
            # indistinguishable from a clean single-shot success.
            cached = self._runtime.resume_cache.get((self._step.id, user_prompt))
            if cached is not None:
                cached_run_id = str(cached.get("run_id") or new_run_id())
                self._runtime.completed.add(self._step.id)
                self._runtime.child_runs.append(
                    ChildRunEvidence(
                        step_id=self._step.id,
                        agent_name=self._step.target_name or self._step.id,
                        run_id=cached_run_id,
                        trace_ref=trace_id(cached_run_id),
                        outcome="replayed",
                    )
                )
                return AgentRunResult(output=str(cached.get("output", "")))

            child_run_id = new_run_id()

            async def dispatch() -> Any:
                target_name = self._step.target_name or self._step.id
                with bind_carrier(self._trace_carrier):
                    return await self._orchestrator.execute_agent(
                        agent_name=target_name,
                        skill_name=(
                            target_name if self._step.kind == "skill" else None
                        ),
                        tool_server=self._step.tool_server,
                        task=user_prompt,
                        max_steps=self._step.max_steps,
                        budget_tokens=self._limits.max_tokens_per_agent,
                        allowed_tools=self._step.allowed_tools or None,
                        required_tools=self._step.required_tools or None,
                        reasoning_effort=self._step.reasoning_effort,
                        model_class=self._step.model_class,
                        run_id=child_run_id,
                        session_id=self._workflow_run_id,
                        include_run_summary=True,
                    )

            outcome = "failed"
            try:
                async with self._runtime.semaphore:
                    child_timeout = min(
                        self._step.timeout_secs or self._limits.max_duration_secs,
                        self._limits.max_duration_secs,
                    )
                    raw = await asyncio.wait_for(
                        _await_with_cancellation(dispatch(), self._cancellation),
                        timeout=child_timeout,
                    )
                output, outcome = _result_payload(raw)
                if outcome != "ok":
                    raise RuntimeError(
                        "GraphOS catalog delegation did not produce a successful outcome"
                    )
                self._runtime.completed.add(self._step.id)
                entry = {
                    "step_id": self._step.id,
                    "task": user_prompt,
                    "output": str(output),
                    "outcome": outcome,
                    "run_id": child_run_id,
                }
                self._runtime.persisted_entries.append(entry)
                self._runtime.resume_cache[(self._step.id, user_prompt)] = entry
                _save_resume_cache(
                    self._runtime.engine,
                    self._runtime.workflow_run_id,
                    self._runtime.persisted_entries,
                )
                return AgentRunResult(output=str(output))
            except TimeoutError:
                outcome = "timeout"
                raise
            except asyncio.CancelledError:
                outcome = "cancelled"
                raise
            finally:
                self._runtime.child_runs.append(
                    ChildRunEvidence(
                        step_id=self._step.id,
                        agent_name=self._step.target_name or self._step.id,
                        run_id=child_run_id,
                        trace_ref=trace_id(child_run_id),
                        outcome=outcome,
                    )
                )

    _GRAPHOS_AGENT_CLASS = GraphOSWorkflowAgent
    return GraphOSWorkflowAgent


class GovernedDynamicWorkflow(BaseModel):
    """Run upstream DynamicWorkflow with GraphOS-only catalog functions."""

    name: str = "governed-dynamic-workflow"
    query: str = ""
    steps: list[DelegationStep]
    max_agent_calls: int = Field(default=50, ge=1, le=300)
    resource_limits: WorkflowResourceLimits = Field(
        default_factory=WorkflowResourceLimits
    )
    # Retained for static-manifest compatibility. Upstream execution derives
    # trace authority only from the verified ambient GraphSession/correlation.
    trace_context: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_budget_and_dag(self) -> GovernedDynamicWorkflow:
        if not self.steps:
            raise ValueError("dynamic workflow requires at least one catalog step")
        if len(self.steps) > self.max_agent_calls:
            raise ValueError("steps exceed max_agent_calls")
        ids = [step.id for step in self.steps]
        if len(ids) != len(set(ids)):
            raise ValueError("dynamic workflow step ids must be unique")
        unknown = {dep for step in self.steps for dep in step.depends_on} - set(ids)
        if unknown:
            raise ValueError(
                f"dynamic workflow has unknown dependencies: {sorted(unknown)}"
            )
        remaining = {step.id: set(step.depends_on) for step in self.steps}
        while remaining:
            ready = {step_id for step_id, deps in remaining.items() if not deps}
            if not ready:
                raise ValueError("dynamic workflow dependencies must form a DAG")
            remaining = {
                step_id: deps - ready
                for step_id, deps in remaining.items()
                if step_id not in ready
            }
        return self

    @classmethod
    def from_graph_plan(
        cls,
        plan: GraphPlan,
        *,
        name: str,
        query: str,
        max_agent_calls: int = 50,
        max_concurrency: int = 8,
        budget_tokens: int | None = None,
        max_steps: int = 30,
    ) -> GovernedDynamicWorkflow:
        """Materialize a reviewed stored DAG as an upstream agent catalog."""

        unsupported = [
            step.id
            for step in plan.steps
            if str(getattr(step, "kind", "task")).lower() in {"gate", "approval"}
        ]
        if unsupported:
            raise DynamicWorkflowUnavailableError(
                "upstream DynamicWorkflow cannot suspend or resume approval gates; "
                "use the stored-DAG runner for this workflow"
            )
        exact_model_steps = [
            step.id for step in plan.steps if getattr(step, "model_id", None)
        ]
        if exact_model_steps:
            raise DynamicWorkflowUnavailableError(
                "upstream GraphOS catalog calls support governed economy/standard "
                "model-class routing, not exact per-step model_id overrides; use "
                "the stored-DAG runner for this workflow"
            )
        if len(plan.steps) > max_agent_calls:
            raise ValueError("stored workflow steps exceed max_agent_calls")
        steps: list[DelegationStep] = []
        for step in plan.steps:
            metadata = step.metadata if isinstance(step.metadata, dict) else {}
            kind = _delegation_kind(metadata, step.assigned_to)
            steps.append(
                DelegationStep(
                    id=step.id,
                    description=str(
                        step.refined_subtask
                        or step.description
                        or f"Execute stored workflow step {step.id}"
                    ),
                    depends_on=list(step.depends_on),
                    kind=kind,
                    target_name=_delegation_target(
                        metadata,
                        kind=kind,
                        assigned_to=step.assigned_to,
                    ),
                    allowed_tools=_string_list(metadata.get("allowed_tools")),
                    required_tools=_string_list(metadata.get("required_tools")),
                    tool_server=_optional_string(metadata.get("tool_server")),
                    model_class=(
                        "economy"
                        if str(getattr(step, "model_tier", "")).lower()
                        in {"small", "cheap"}
                        else "standard"
                    ),
                    reasoning_effort=(
                        metadata.get("reasoning_effort")
                        if metadata.get("reasoning_effort") in {"low", "medium", "high"}
                        else None
                    ),
                    model_id=getattr(step, "model_id", None),
                    model_menu=list(getattr(step, "delegation_model_menu", ()) or ()),
                    timeout_secs=float(getattr(step, "timeout", 300.0) or 300.0),
                    max_steps=max_steps,
                )
            )
        return cls(
            name=name,
            query=query,
            steps=steps,
            max_agent_calls=max_agent_calls,
            resource_limits=WorkflowResourceLimits(
                max_concurrency=max_concurrency,
                max_tokens_per_agent=budget_tokens,
                orchestrator_token_budget=budget_tokens,
            ),
        )

    def to_graph_plan(self) -> GraphPlan:
        """Compile the declaration for the explicit stored-DAG fallback."""

        tasks = []
        for step in self.steps:
            timeout = min(
                step.timeout_secs or self.resource_limits.max_duration_secs,
                self.resource_limits.max_duration_secs,
            )
            tasks.append(
                Task(
                    id=step.id,
                    description=step.description,
                    refined_subtask=step.description,
                    depends_on=step.depends_on,
                    parallel=not step.depends_on,
                    timeout=timeout,
                    model_id=step.model_id,
                    delegation_model_menu=step.model_menu,
                )
            )
        return GraphPlan(
            steps=tasks,
            metadata={
                "source": "governed_dynamic_workflow_static_fallback",
                "max_agent_calls": self.max_agent_calls,
                "resource_limits": self.resource_limits.model_dump(),
                "trace_context": self.trace_context,
            },
        )

    def to_manifest(self) -> ExecutionManifest:
        """Materialize the explicit static-fallback input for ParallelEngine."""

        manifest = ExecutionManifest.from_graph_plan(
            self.to_graph_plan(), name=self.name, query=self.query
        ).model_copy(
            update={
                "source": "governed_dynamic_workflow_static_fallback",
                "max_concurrency": self.resource_limits.max_concurrency,
            },
            deep=True,
        )
        for spec, step in zip(manifest.agents, self.steps, strict=True):
            spec.tools = list(step.allowed_tools)
        return manifest

    async def execute_static(
        self,
        parallel_engine: Any,
        *,
        graph_deps: Any | None = None,
        cancellation: asyncio.Event | None = None,
    ) -> ExecutionResult:
        """Run the explicit non-Harness fallback through ParallelEngine."""

        return await _await_with_cancellation(
            parallel_engine.execute(self.to_manifest(), graph_deps=graph_deps),
            cancellation,
        )

    def _build_upstream_runtime(
        self,
        orchestrator: Any,
        *,
        workflow_run_id: str,
        cancellation: asyncio.Event | None,
        trace_carrier: dict[str, str],
    ) -> tuple[Any, _WorkflowRuntime, str]:
        DynamicWorkflow, WorkflowAgent, harness_version = (
            _load_upstream_dynamic_workflow()
        )
        if any(step.model_id for step in self.steps):
            raise DynamicWorkflowUnavailableError(
                "exact model_id selection is not supported on the GraphOS "
                "DynamicWorkflow boundary; use governed model_class routing "
                "or the stored-DAG runner"
            )
        runtime = _WorkflowRuntime(
            semaphore=asyncio.Semaphore(self.resource_limits.max_concurrency)
        )
        engine = getattr(orchestrator, "engine", None)
        runtime.engine = engine
        runtime.workflow_run_id = workflow_run_id
        prior_entries = _load_resume_cache(engine, workflow_run_id)
        runtime.persisted_entries = list(prior_entries)
        for prior in prior_entries:
            step_id = str(prior.get("step_id") or "")
            task = str(prior.get("task") or "")
            if step_id and prior.get("outcome") == "ok":
                runtime.resume_cache[(step_id, task)] = prior
                runtime.completed.add(step_id)
        agent_cls = _graphos_agent_class()
        catalog = [
            WorkflowAgent(
                agent=agent_cls(
                    orchestrator=orchestrator,
                    step=step,
                    workflow_run_id=workflow_run_id,
                    runtime=runtime,
                    cancellation=cancellation,
                    limits=self.resource_limits,
                    trace_carrier=trace_carrier,
                ),
                name=step.id,
                description=None,
            )
            for step in self.steps
        ]
        capability = DynamicWorkflow(
            id=f"graphos-dynamic-workflow:{self.name}",
            agents=catalog,
            max_agent_calls=self.max_agent_calls,
            max_retries=3,
            # GraphOS child runs own token accounting; forwarding Pydantic's
            # accumulator here would falsely claim those calls used this model.
            forward_usage=False,
            inherit_model=False,
            resource_limits={
                "max_duration_secs": self.resource_limits.max_duration_secs,
                "max_memory": self.resource_limits.max_memory_bytes,
            },
        )
        return capability, runtime, harness_version

    def _build_graph_execution_evidence(
        self,
        *,
        harness_version: str,
        runtime: _WorkflowRuntime | None,
        checkpoint_ids: list[str],
    ) -> dict[str, Any]:
        """Build the SAME ``GraphExecutionEvidence`` shape ``pydantic_graph`` runs
        write onto their ``RunTrace``, but honestly ``resume_supported=True`` --
        every node/transition below is backed by a real, restorable
        ``WorkflowResumeState`` cache entry (CONCEPT:AU-ORCH.execution.dynamic-workflow-resume),
        not an observational snapshot.
        """

        topology = _catalog_topology(self.steps)
        child_runs = list(runtime.child_runs) if runtime is not None else []
        version_payload = json.dumps(
            {
                "schema": "governed-dynamic-workflow-resume-v1",
                "harness_version": harness_version,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        transitions = [
            GraphTransitionEvidence(
                sequence=index + 1,
                scheduled_tasks=[
                    GraphTaskEvidence(node_id=child.agent_name, task_id=child.run_id)
                ],
            )
            for index, child in enumerate(child_runs)
        ]
        evidence = GraphExecutionEvidence(
            topology=topology,
            topology_digest=_content_digest(topology),
            version_digest=_content_digest(version_payload),
            runtime_version=harness_version,
            node_sequence=[child.agent_name for child in child_runs],
            transitions=transitions,
            checkpoint_ids=list(checkpoint_ids),
            resume_supported=True,
        )
        return evidence.model_dump(mode="json")

    def build_upstream_capability(
        self,
        orchestrator: Any,
        *,
        workflow_run_id: str | None = None,
        cancellation: asyncio.Event | None = None,
        trace_carrier: dict[str, str] | None = None,
    ) -> Any:
        """Return the real Harness capability for discovery and integration tests."""

        capability, _, _ = self._build_upstream_runtime(
            orchestrator,
            workflow_run_id=workflow_run_id or new_run_id(),
            cancellation=cancellation,
            trace_carrier=dict(trace_carrier or {}),
        )
        return capability

    async def execute_upstream(
        self,
        orchestrator: Any,
        *,
        orchestrator_model: Any,
        cancellation: asyncio.Event | None = None,
        workflow_run_id: str | None = None,
    ) -> GovernedDynamicWorkflowResult:
        """Run a real Harness DynamicWorkflow with GraphOS-governed sub-agents."""

        workflow_run_id = workflow_run_id or new_run_id()
        started = time.monotonic()
        active_runtime: _WorkflowRuntime | None = None
        active_checkpoint_store: _CapturingCheckpointStore | None = None

        from pydantic_ai.messages import ToolCallPart
        from pydantic_ai.usage import UsageLimits

        from agent_utilities.capabilities.checkpointing import (
            CheckpointMiddleware,
            GraphCheckpointStore,
            InMemoryCheckpointStore,
        )
        from agent_utilities.core.contextual_model import create_context_agent
        from agent_utilities.harness.tracing import trace
        from agent_utilities.observability.correlation import current_carrier
        from agent_utilities.observability.trace_ontology import trace_id

        @trace(
            name="governed_dynamic_workflow.upstream",
            tags=["orchestration", "dynamic-workflow", "pydantic-ai-harness"],
            metadata={"backend": "pydantic-ai-harness.dynamic_workflow"},
        )
        async def run_harness() -> tuple[Any, _WorkflowRuntime, str]:
            nonlocal active_runtime, active_checkpoint_store
            # Capture the trace only after the trace wrapper has established it.
            # Caller-provided ``trace_context`` is deliberately not authority.
            capability, runtime, harness_version = self._build_upstream_runtime(
                orchestrator,
                workflow_run_id=workflow_run_id,
                cancellation=cancellation,
                trace_carrier=current_carrier(),
            )
            active_runtime = runtime
            # Default-ON conductor checkpointing (CONCEPT:AU-ORCH.execution.dynamic-workflow-resume):
            # this is the ONE place `default_capabilities=False` previously left the
            # conductor with ZERO reliability capabilities. Checkpointing is safe to
            # default on here specifically -- unlike the workspace-wide
            # `default_runtime_capabilities(include_checkpoints=False)` default, which
            # governs every agent (many high-volume/short-lived, where a snapshot per
            # tool call would be wasteful) -- because a DynamicWorkflow conductor makes
            # exactly one bounded, low-frequency tool call (`run_workflow`) per attempt.
            inner_store = (
                GraphCheckpointStore(runtime.engine)
                if runtime.engine is not None
                else InMemoryCheckpointStore()
            )
            checkpoint_store = _CapturingCheckpointStore(inner_store)
            active_checkpoint_store = checkpoint_store
            parent = create_context_agent(
                model=orchestrator_model,
                name=f"dynamic-workflow:{self.name}",
                description="GraphOS governed dynamic workflow conductor",
                instructions=(
                    "Use the run_workflow capability to complete the task. "
                    "Only catalog functions may perform work. Respect their "
                    "host-enforced dependency contracts and return the final result."
                ),
                capabilities=[
                    capability,
                    CheckpointMiddleware(
                        store=checkpoint_store, frequency="every_tool"
                    ),
                ],
                default_capabilities=False,
            )
            from agent_utilities.orchestration.loop_guards import (
                DEFAULT_PER_REQUEST_INPUT_TOKENS_LIMIT,
            )

            limits = UsageLimits(
                request_limit=self.max_agent_calls + 5,
                total_tokens_limit=self.resource_limits.orchestrator_token_budget,
                # CONCEPT:AU-ORCH.execution.execution-budget-caps — a single
                # oversized catalog-function result must not blow the workflow
                # orchestrator's budget in one request.
                per_request_input_tokens_limit=DEFAULT_PER_REQUEST_INPUT_TOKENS_LIMIT,
            )
            result = await _await_with_cancellation(
                parent.run(
                    self.query,
                    run_id=workflow_run_id,
                    usage_limits=limits,
                    metadata={
                        "workflow_run_id": workflow_run_id,
                        "workflow_name": self.name,
                    },
                ),
                cancellation,
            )
            return result, runtime, harness_version

        try:
            run_result, runtime, harness_version = await asyncio.wait_for(
                run_harness(),
                timeout=self.resource_limits.max_duration_secs,
            )
        except BaseException as exc:
            # A halted attempt's evidence is still real: every completed
            # dispatch in ``active_runtime.child_runs`` is already persisted to
            # the resume cache (CONCEPT:AU-ORCH.execution.dynamic-workflow-resume),
            # so the NEXT call with the SAME ``workflow_run_id`` (a genuine
            # "budget halt then restart") will not re-dispatch them.
            failure_evidence = self._build_graph_execution_evidence(
                harness_version=_safe_harness_version(),
                runtime=active_runtime,
                checkpoint_ids=(
                    list(active_checkpoint_store.ids)
                    if active_checkpoint_store is not None
                    else []
                ),
            )
            # ``_persist_parent_trace`` performs synchronous native graph writes
            # (RunTrace/Session nodes, PARENT_RUN edges). Run it on a worker so
            # a foreground DynamicWorkflow completion/failure never blocks the
            # shared GraphOS event loop, and keep it ordered so the trace is
            # guaranteed to land before this exception (including
            # cancellation) is re-raised to the caller.
            await run_blocking_ordered(
                self._persist_parent_trace,
                orchestrator,
                workflow_run_id=workflow_run_id,
                status=(
                    "cancelled"
                    if isinstance(exc, asyncio.CancelledError)
                    else "timeout"
                    if isinstance(exc, TimeoutError)
                    else "failed"
                ),
                duration_ms=(time.monotonic() - started) * 1000,
                result_preview="",
                child_runs=(
                    list(active_runtime.child_runs)
                    if active_runtime is not None
                    else []
                ),
                error=type(exc).__name__,
                graph_execution_evidence=failure_evidence,
            )
            raise

        script_evidence: list[WorkflowScriptEvidence] = []
        script_artifacts: list[dict[str, Any]] = []
        for message in run_result.all_messages():
            for part in getattr(message, "parts", ()) or ():
                if (
                    not isinstance(part, ToolCallPart)
                    or part.tool_name != "run_workflow"
                ):
                    continue
                args = part.args_as_dict()
                code = str(args.get("code") or "")
                sha256 = hashlib.sha256(code.encode()).hexdigest()
                byte_count = len(code.encode())
                line_count = len(code.splitlines())
                script_evidence.append(
                    WorkflowScriptEvidence(
                        tool_call_id=part.tool_call_id,
                        sha256=sha256,
                        byte_count=byte_count,
                        line_count=line_count,
                    )
                )
                # The FULL generated script is stored as a trace artifact (see
                # ``_persist_parent_trace``'s ``script_artifacts``) -- the
                # ``script_evidence`` above stays the privacy-safe digest form
                # returned to every caller.
                script_artifacts.append(
                    {
                        "tool_call_id": part.tool_call_id,
                        "sha256": sha256,
                        "byte_count": byte_count,
                        "line_count": line_count,
                        "code": code,
                    }
                )

        usage = run_result.usage
        usage_data = {
            "requests": int(getattr(usage, "requests", 0) or 0),
            "input_tokens": int(getattr(usage, "input_tokens", 0) or 0),
            "output_tokens": int(getattr(usage, "output_tokens", 0) or 0),
        }
        checkpoint_ids = (
            list(active_checkpoint_store.ids)
            if active_checkpoint_store is not None
            else []
        )
        replayed_step_ids = [
            child.step_id or child.agent_name
            for child in runtime.child_runs
            if child.outcome == "replayed"
        ]
        evidence = self._build_graph_execution_evidence(
            harness_version=harness_version,
            runtime=runtime,
            checkpoint_ids=checkpoint_ids,
        )
        # Same worker isolation as the failure path above, so a successful
        # completion's lineage write cannot block the shared event loop either.
        await run_blocking_ordered(
            self._persist_parent_trace,
            orchestrator,
            workflow_run_id=workflow_run_id,
            status="completed",
            duration_ms=(time.monotonic() - started) * 1000,
            result_preview=str(run_result.output)[:500],
            child_runs=runtime.child_runs,
            error=None,
            graph_execution_evidence=evidence,
            script_artifacts=script_artifacts,
        )
        return GovernedDynamicWorkflowResult(
            output=run_result.output,
            workflow_run_id=workflow_run_id,
            trace_ref=trace_id(workflow_run_id),
            upstream_version=harness_version,
            script_evidence=script_evidence,
            child_runs=runtime.child_runs,
            usage=usage_data,
            resumed=bool(replayed_step_ids),
            replayed_step_ids=replayed_step_ids,
            checkpoint_ids=checkpoint_ids,
        )

    async def execute(
        self,
        orchestrator: Any,
        *,
        orchestrator_model: Any,
        cancellation: asyncio.Event | None = None,
        workflow_run_id: str | None = None,
    ) -> GovernedDynamicWorkflowResult:
        """Canonical alias for the upstream execution path."""

        return await self.execute_upstream(
            orchestrator,
            orchestrator_model=orchestrator_model,
            cancellation=cancellation,
            workflow_run_id=workflow_run_id,
        )

    def _persist_parent_trace(
        self,
        orchestrator: Any,
        *,
        workflow_run_id: str,
        status: str,
        duration_ms: float,
        result_preview: str,
        child_runs: list[ChildRunEvidence],
        error: str | None,
        graph_execution_evidence: dict[str, Any] | None = None,
        script_artifacts: list[dict[str, Any]] | None = None,
    ) -> None:
        """Persist one parent RunTrace, its resume evidence, and parent/child lineage.

        ``script_artifacts`` (CONCEPT:AU-ORCH.execution.dynamic-workflow-resume)
        stores the actual model-authored orchestration code the ``RunTrace``'s
        privacy-safe ``graph_topology``/evidence only summarizes -- as a
        dedicated, non-redacted ``WorkflowScriptArtifact`` node, the same way
        ``CheckpointNode`` stores full replayable message content rather than a
        digest.
        """

        engine = getattr(orchestrator, "engine", None)
        if engine is None:
            return
        try:
            from agent_utilities.observability.trace_ontology import trace_id
            from agent_utilities.orchestration.agent_runner import (
                _record_execution_trace,
            )

            _record_execution_trace(
                engine,
                workflow_run_id,
                f"dynamic-workflow:{self.name}",
                self.query,
                status=status,
                error=error,
                duration_ms=duration_ms,
                result_preview=result_preview,
                execution_mode="dynamic_workflow",
                tool_call_count=len(child_runs),
                graph_execution_evidence=graph_execution_evidence,
            )
            session_node = f"session:{workflow_run_id}"
            parent_trace = trace_id(workflow_run_id)
            engine.add_node(
                session_node,
                "Session",
                properties={
                    "id": session_node,
                    "session_id": workflow_run_id,
                },
            )
            engine.link_nodes(session_node, parent_trace, "HAS_RUN")
            for child in child_runs:
                engine.link_nodes(child.trace_ref, parent_trace, "PARENT_RUN")
            for artifact in script_artifacts or []:
                script_node = (
                    f"workflow-script:{workflow_run_id}:{artifact['tool_call_id']}"
                )
                engine.add_node(
                    script_node,
                    "WorkflowScriptArtifact",
                    properties={
                        "id": script_node,
                        "workflow_run_id": workflow_run_id,
                        "tool_call_id": artifact["tool_call_id"],
                        "sha256": artifact["sha256"],
                        "byte_count": artifact["byte_count"],
                        "line_count": artifact["line_count"],
                        "code": artifact["code"],
                    },
                )
                engine.link_nodes(script_node, parent_trace, "SCRIPT_OF")
        except Exception as exc:
            # Trace persistence must never convert an execution result to failure.
            logger.warning(
                "Could not persist DynamicWorkflow trace lineage: %s",
                exc,
                exc_info=True,
            )
