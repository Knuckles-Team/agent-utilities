"""Typed AU agent-application operations and injected authority ports.

These contracts intentionally stop before GraphOS hosting and transport. They
describe the AU use cases GraphOS may call and the typed ports a composition
root must provide; they do not create a registrar or generic dispatcher.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from agent_utilities.api.session import GraphSession

CapabilityKind = Literal["agent", "skill", "workflow"]
WorkItemStatus = Literal[
    "submitted",
    "ready",
    "leased",
    "running",
    "succeeded",
    "failed",
    "cancelled",
    "dead_letter",
]


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class AgentControlPlaneUnavailable(RuntimeError):
    """A required application port is absent or not supported by its authority."""


class AgentWorkItemNotCancelable(RuntimeError):
    """The current verified caller cannot cancel the requested WorkItem."""


class CapabilitySearchRequest(_StrictModel):
    """Typed request for an authorized capability search."""

    task: str = Field(min_length=1, max_length=10_000)
    agent_name: str | None = Field(default=None, max_length=512)
    limit: int = Field(default=24, ge=1, le=256)


class CapabilityCandidate(_StrictModel):
    """One authorized candidate returned by the EG-backed search port."""

    kind: CapabilityKind
    name: str = Field(min_length=1, max_length=512)
    component_id: str = Field(min_length=1, max_length=512)
    score: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    source: str = Field(min_length=1, max_length=128)


class CapabilityResolution(_StrictModel):
    """Selected capability and bounded alternatives for an agent task."""

    kind: CapabilityKind
    name: str = Field(min_length=1, max_length=512)
    component_id: str = ""
    score: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    source: Literal["caller", "eg_search", "default"]
    alternatives: tuple[CapabilityCandidate, ...] = Field(max_length=3)


@runtime_checkable
class CapabilitySearchPort(Protocol):
    """Session-authorized EG capability search; callers cannot supply identity."""

    async def search(
        self, request: CapabilitySearchRequest, *, session: GraphSession
    ) -> Sequence[CapabilityCandidate]: ...


class AgentExecutionRequest(_StrictModel):
    """The supported AU agent execution inputs, independent of transport."""

    agent_name: str = Field(min_length=1, max_length=512)
    task: str = Field(min_length=1, max_length=100_000)
    max_steps: int = Field(default=30, ge=1, le=256)
    return_mermaid: bool = False
    context: str | None = Field(default=None, max_length=100_000)
    budget_tokens: int | None = Field(default=None, ge=1, le=10_000_000)
    context_ref: str | None = Field(default=None, max_length=512)
    allowed_tools: tuple[str, ...] | None = Field(default=None, max_length=128)
    required_tools: tuple[str, ...] | None = Field(default=None, max_length=128)
    credential_ref: str | None = Field(default=None, max_length=512)
    session_ref: str | None = Field(default=None, max_length=512)
    open_channel: bool = False
    memento_source: str | None = Field(default=None, max_length=512)
    execution_profile: Literal["task", "chat"] | None = None
    reasoning_effort: str | None = Field(default=None, max_length=64)
    model_class: str = Field(default="standard", max_length=64)
    response_format: Literal["text", "json"] = "text"
    run_id: str | None = Field(default=None, max_length=512)
    include_run_summary: bool = False
    skill_name: str | None = Field(default=None, max_length=512)
    tool_server: str | None = Field(default=None, max_length=512)
    execution_mode: Literal["auto", "direct", "graph"] = "auto"
    grounding: Literal["required", "best_effort", "none"] = "required"


class AgentExecutionResult(_StrictModel):
    """Typed result returned by the AU-owned agent execution port."""

    run_id: str = Field(min_length=1, max_length=512)
    output: str
    execution_mode: Literal["direct", "graph"] | None = None


@runtime_checkable
class AgentExecutionPort(Protocol):
    """AU execution capability; no GraphOS or legacy KG engine is implied."""

    async def execute_agent(
        self, request: AgentExecutionRequest, *, session: GraphSession
    ) -> AgentExecutionResult: ...


class WorkItemSubmission(_StrictModel):
    """AU application intent to admit one task into the durable WorkItem store."""

    work_item_id: str = Field(min_length=1, max_length=512)
    idempotency_key: str = Field(min_length=1, max_length=512)
    kind: str = Field(min_length=1, max_length=128)
    description: str = Field(min_length=1, max_length=100_000)
    priority: int = Field(default=0, ge=-1024, le=1024)
    deadline_unix: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    max_attempts: int = Field(default=3, ge=1, le=4096)
    metadata: dict[str, JsonValue] = Field(default_factory=dict)


class WorkItemSnapshot(_StrictModel):
    """Tenant-authorized current WorkItem view with a version fence."""

    work_item_id: str = Field(min_length=1, max_length=512)
    kind: str = Field(min_length=1, max_length=128)
    status: WorkItemStatus
    payload_ref: str = Field(default="", max_length=512)
    description: str = Field(default="", max_length=100_000)
    metadata: dict[str, JsonValue] = Field(default_factory=dict)
    version: int = Field(ge=1)
    updated_at_ms: int = Field(ge=0)


class WorkItemSubmissionResult(_StrictModel):
    """Durable admission result; replay is distinct from first creation."""

    item: WorkItemSnapshot
    created: bool
    replayed: bool

    @model_validator(mode="after")
    def _validate_consistency(self) -> WorkItemSubmissionResult:
        if self.created == self.replayed:
            raise ValueError("WorkItem admission must be created xor replayed")
        return self


class WorkItemGetRequest(_StrictModel):
    work_item_id: str = Field(min_length=1, max_length=512)


class WorkItemListRequest(_StrictModel):
    cursor: str | None = Field(default=None, max_length=4096)
    limit: int = Field(default=50, ge=1, le=100)
    kind: str | None = Field(default=None, max_length=128)


class WorkItemPage(_StrictModel):
    items: tuple[WorkItemSnapshot, ...] = Field(max_length=100)
    next_cursor: str | None = Field(default=None, max_length=4096)


class WorkItemCancelRequest(_StrictModel):
    work_item_id: str = Field(min_length=1, max_length=512)
    reason: Literal["caller_cancelled", "dispatch_admission_failed"] = (
        "caller_cancelled"
    )


@runtime_checkable
class WorkItemStorePort(Protocol):
    """Explicit typed WorkItem authority, scoped by the verified session."""

    async def submit(
        self, request: WorkItemSubmission, *, session: GraphSession
    ) -> WorkItemSubmissionResult: ...

    async def get(
        self, request: WorkItemGetRequest, *, session: GraphSession
    ) -> WorkItemSnapshot | None: ...

    async def list(
        self, request: WorkItemListRequest, *, session: GraphSession
    ) -> WorkItemPage: ...

    async def cancel(
        self, request: WorkItemCancelRequest, *, session: GraphSession
    ) -> WorkItemSnapshot | None: ...


class SignedAgentDispatchRequest(_StrictModel):
    """Reference-only agent-turn enqueue input; no body or caller authority."""

    job_id: str = Field(min_length=1, max_length=512)
    work_item_id: str = Field(min_length=1, max_length=512)
    session_ref: str = Field(min_length=1, max_length=512)
    kind: Literal["orchestrator_task"] = "orchestrator_task"
    agent_name: str = Field(min_length=1, max_length=512)


class SignedAgentDispatchReceipt(_StrictModel):
    job_id: str = Field(min_length=1, max_length=512)
    accepted: bool


@runtime_checkable
class SignedAgentDispatchPort(Protocol):
    """AU queue operation that signs and verifies the dispatch envelope."""

    async def enqueue(
        self, request: SignedAgentDispatchRequest, *, session: GraphSession
    ) -> SignedAgentDispatchReceipt: ...


class AgentTaskDispatchRequest(_StrictModel):
    """Complete AU task-admission request; identity is always session-derived."""

    work_item_id: str = Field(min_length=1, max_length=512)
    idempotency_key: str = Field(min_length=1, max_length=512)
    job_id: str = Field(min_length=1, max_length=512)
    session_ref: str = Field(min_length=1, max_length=512)
    task: str = Field(min_length=1, max_length=100_000)
    agent_name: str | None = Field(default=None, max_length=512)
    metadata: dict[str, JsonValue] = Field(default_factory=dict)


class AgentTaskDispatchResult(_StrictModel):
    capability: CapabilityResolution
    admission: WorkItemSubmissionResult
    dispatch: SignedAgentDispatchReceipt | None = None


@dataclass(frozen=True, slots=True)
class AgentOperationDescriptor:
    """Static schema metadata for a direct AU operation, not a registrar."""

    name: str
    request_schema: dict[str, JsonValue]
    result_schema: dict[str, JsonValue]
    required_scope: Literal["kg:read", "kg:write"]
    action_scopes: tuple[tuple[str, Literal["kg:read", "kg:write"]], ...] = ()
    tool_name: str | None = None
