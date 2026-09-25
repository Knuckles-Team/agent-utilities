"""Typed AU agent-application operations and injected authority ports.

These contracts intentionally stop before GraphOS hosting and transport. They
describe the AU use cases GraphOS may call and the typed ports a composition
root must provide; they do not create a registrar or generic dispatcher.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Annotated, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from agent_utilities.api.session import GraphSession

CapabilityKind = Literal["agent", "skill", "workflow"]
#: The native EG task vocabulary. Free text is never sent to EG as a task;
#: a caller (or an AU classification run, EH-206) maps it to one of these.
TaskIri = Literal[
    "eg:task/research",
    "eg:task/implement",
    "eg:task/review",
    "eg:task/operate",
    "eg:task/communicate",
]
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


#: Bound on a signed execution tool allowlist (AU-2 / EH-044); mirrors
#: ``orchestration.agent_dispatch.MAX_DISPATCH_ALLOWED_TOOLS``.
MAX_ALLOWED_TOOLS = 64
AllowedTools = tuple[Annotated[str, Field(min_length=1, max_length=256)], ...]


class AgentControlPlaneUnavailable(RuntimeError):
    """A required application port is absent or not supported by its authority."""


class AgentWorkItemNotCancelable(RuntimeError):
    """The current verified caller cannot cancel the requested WorkItem."""


class CapabilitySearchRequest(_StrictModel):
    """Typed request for an authorized capability search."""

    task: str = Field(min_length=1, max_length=10_000)
    agent_name: str | None = Field(default=None, max_length=512)
    limit: int = Field(default=24, ge=1, le=256)
    #: Typed task term for EG's ontology search; ``task`` stays AU-local text.
    task_iri: TaskIri | None = None


class CapabilityCandidate(_StrictModel):
    """One authorized candidate returned by the EG-backed search port."""

    kind: CapabilityKind
    name: str = Field(min_length=1, max_length=512)
    component_id: str = Field(min_length=1, max_length=512)
    content_digest: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    score: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    source: str = Field(min_length=1, max_length=128)


#: Evidence classes a Decide-layer-facing claim can carry. Only ``"claim"``
#: exists here: AU never asserts ``"proof"`` for a classification it derived
#: itself (DECIDE-LAYER-DESIGN.md §7.1, DECISIONS.md 2026-09-17 afternoon).
EvidenceClass = Literal["claim"]


class TaskClassificationClaim(_StrictModel):
    """A free-text task's proposed mapping onto one of EG's five native task
    IRIs (EH-206) -- ALWAYS a labelled claim, never a proof.

    Deterministic and LLM-free (lexical keyword overlap against each IRI's
    own ontology labels, ``agent_utilities.api.task_classification``); never
    the raw task text, only a digest, so the claim can be logged/persisted
    without duplicating the (already screened/redacted) task string
    elsewhere. Matches EG's own claim-premise shape for a caller-supplied
    task mapping (DECIDE-LAYER-DESIGN.md §7.1) and its ``UnmappedTask
    {text_digest}`` abstention shape when no IRI is close enough (in which
    case no ``TaskClassificationClaim`` is produced at all -- see
    :func:`agent_utilities.api.task_classification.classify_task_text`).
    """

    task_iri: TaskIri
    confidence: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    method: Literal["lexical_keyword_overlap"]
    matched_keywords: tuple[str, ...] = Field(max_length=32)
    text_digest: str = Field(min_length=64, max_length=64)
    evidence_class: EvidenceClass = "claim"


class CapabilityResolution(_StrictModel):
    """Selected capability and bounded alternatives for an agent task."""

    kind: CapabilityKind
    name: str = Field(min_length=1, max_length=512)
    component_id: str = ""
    content_digest: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    score: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    source: Literal["caller", "eg_search", "default"]
    alternatives: tuple[CapabilityCandidate, ...] = Field(max_length=3)
    #: Set only when ``source == "eg_search"`` and the caller supplied free
    #: text with no ``task_iri``/``agent_name``: the deterministic claim
    #: (EH-206) that proposed the task IRI actually searched. ``None`` when
    #: the caller supplied a typed ``task_iri`` or ``agent_name`` directly
    #: (no classification was needed) -- never fabricated after the fact.
    task_claim: TaskClassificationClaim | None = None


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
    task_iri: TaskIri | None = None
    #: The native sub-agent allowance of the committed topology plan node this
    #: run executes (``SubagentAllowance`` fields); absent grants none.
    subagents: dict[str, JsonValue] | None = None


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


class WorkItemDelegationBinding(_StrictModel):
    """AU-selected, EG-pinned agent identity for a signed worker dispatch."""

    delegation_id: str = Field(min_length=1, max_length=512)
    run_id: str = Field(min_length=1, max_length=512)
    agent_id: str = Field(min_length=1, max_length=512)
    agent_name: str = Field(min_length=1, max_length=512)
    capability_digest: str = Field(pattern=r"^[0-9a-f]{64}$")


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
    delegation_binding: WorkItemDelegationBinding | None = None


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
    #: Signed into the dispatch carrier and enforced by the AU worker's
    #: toolset construction; ``None`` leaves the agent's own tool set.
    allowed_tools: AllowedTools | None = Field(
        default=None, max_length=MAX_ALLOWED_TOOLS
    )


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
    allowed_tools: AllowedTools | None = Field(
        default=None, max_length=MAX_ALLOWED_TOOLS
    )
    task_iri: TaskIri | None = None


class AgentTaskDispatchResult(_StrictModel):
    capability: CapabilityResolution
    admission: WorkItemSubmissionResult
    dispatch: SignedAgentDispatchReceipt | None = None


class RunOutputRequest(_StrictModel):
    run_id: str = Field(min_length=1, max_length=512)


RunStatus = Literal["running", "succeeded", "failed", "unknown"]


class RunOutput(_StrictModel):
    """Bounded, redacted final answer of one agent run (AU-5)."""

    run_id: str = Field(min_length=1, max_length=512)
    status: RunStatus
    output: str = Field(default="", max_length=100_000)
    truncated: bool = False


@runtime_checkable
class RunOutputPort(Protocol):
    """Session-scoped read of a run's final output; ``None`` when not visible."""

    async def get_run_output(
        self, request: RunOutputRequest, *, session: GraphSession
    ) -> RunOutput | None: ...


@dataclass(frozen=True, slots=True)
class AgentOperationDescriptor:
    """Static schema metadata for a direct AU operation, not a registrar."""

    name: str
    request_schema: dict[str, JsonValue]
    result_schema: dict[str, JsonValue]
    required_scope: Literal["kg:read", "kg:write"]
    action_scopes: tuple[tuple[str, Literal["kg:read", "kg:write"]], ...] = ()
    tool_name: str | None = None
