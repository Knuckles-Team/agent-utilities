"""The verified, application-level RLM operation retained by GraphOS.

GraphOS owns transport registration, routing, and its public action manifest. This
module owns the RLM behavior and requires the caller to bind it to an injected,
session-scoped epistemic-graph client before any operation can run.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Annotated, Any, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, field_validator

from agent_utilities.api.agent_control_contracts import (
    AgentControlPlaneUnavailable,
    AgentExecutionPort,
    AgentExecutionRequest,
    AgentExecutionResult,
    AgentOperationDescriptor,
    AgentTaskDispatchRequest,
    AgentTaskDispatchResult,
    CapabilityCandidate,
    CapabilityResolution,
    CapabilitySearchPort,
    CapabilitySearchRequest,
    SignedAgentDispatchPort,
    SignedAgentDispatchReceipt,
    SignedAgentDispatchRequest,
    WorkItemCancelRequest,
    WorkItemGetRequest,
    WorkItemListRequest,
    WorkItemPage,
    WorkItemSnapshot,
    WorkItemStorePort,
    WorkItemSubmission,
    WorkItemSubmissionResult,
)
from agent_utilities.api.session import GraphSession, resolve_session
from agent_utilities.rlm.benchmarks.base import BenchResult
from agent_utilities.rlm.telemetry import FailureClass
from agent_utilities.security.error_surface import public_error_payload

_RequiredScope: TypeAlias = Literal["kg:read", "kg:write"]
_OperationSpec: TypeAlias = tuple[
    str,
    Any,
    Any,
    _RequiredScope,
    tuple[tuple[str, _RequiredScope], ...],
]


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class GraphRlmBenchmarkOptions(_StrictModel):
    """Bounded long-context benchmark controls."""

    scales: list[int] = Field(
        default_factory=lambda: [50_000], min_length=1, max_length=4
    )
    cases_per_scale: int = Field(default=3, ge=1, le=10)

    @field_validator("scales")
    @classmethod
    def _validate_scales(cls, scales: list[int]) -> list[int]:
        if any(scale < 1 or scale > 10_000_000 for scale in scales):
            raise ValueError("benchmark scales must be between 1 and 10000000")
        return scales


class GraphRlmPromptExample(_StrictModel):
    query: str = Field(min_length=1, max_length=100_000)
    response: str = Field(max_length=100_000)


class GraphRlmEvolutionOptions(_StrictModel):
    """Bounded GEPA prompt-evolution controls."""

    objectives: list[str] = Field(
        default_factory=lambda: ["accuracy"], min_length=1, max_length=8
    )
    iterations: int = Field(default=1, ge=1, le=20)
    batch_size: int = Field(default=2, ge=1, le=16)
    dataset: list[GraphRlmPromptExample] | None = Field(default=None, max_length=500)

    @field_validator("objectives")
    @classmethod
    def _validate_objectives(cls, objectives: list[str]) -> list[str]:
        if any(not objective.strip() for objective in objectives):
            raise ValueError("objectives must be non-empty strings")
        if len(set(objectives)) != len(objectives):
            raise ValueError("objectives must be unique")
        return objectives

    @field_validator("dataset")
    @classmethod
    def _validate_dataset(
        cls, dataset: list[GraphRlmPromptExample] | None
    ) -> list[GraphRlmPromptExample] | None:
        if dataset is not None and not dataset:
            raise ValueError("dataset must be omitted or contain at least one example")
        return dataset


class GraphRlmRunRequest(_StrictModel):
    action: Literal["run"] = "run"
    task: str = Field(min_length=1, max_length=10_000)
    input_text: str = Field(default="", max_length=10_000_000)


class GraphRlmBenchmarkRequest(_StrictModel):
    action: Literal["benchmark"] = "benchmark"
    task: str = Field(default="s_niah", max_length=256)
    options: GraphRlmBenchmarkOptions = Field(default_factory=GraphRlmBenchmarkOptions)


class GraphRlmEvolvePromptRequest(_StrictModel):
    action: Literal["evolve_prompt"] = "evolve_prompt"
    task: str = Field(default="", max_length=10_000)
    options: GraphRlmEvolutionOptions = Field(default_factory=GraphRlmEvolutionOptions)


GraphRlmRequest: TypeAlias = Annotated[
    GraphRlmRunRequest | GraphRlmBenchmarkRequest | GraphRlmEvolvePromptRequest,
    Field(discriminator="action"),
]


class GraphRlmError(_StrictModel):
    """Sanitized, correlation-safe failure metadata; never includes exception text."""

    code: Literal[
        "operation_failed",
        "invalid_request",
        "permission_denied",
        "dependency_unavailable",
        "engine_degraded",
    ]
    correlation_id: str
    detail_ref: str | None = None
    retryable: bool = False
    error_class: str
    failing_layer: str


class GraphRlmRunResult(_StrictModel):
    action: Literal["run"]
    ok: bool
    task: str
    result: str | None = None
    usage: dict[str, int | float] = Field(default_factory=dict)
    max_depth: int = 0
    failure_class: FailureClass | None = None
    error: GraphRlmError | None = None


class GraphRlmBenchmarkResult(_StrictModel):
    action: Literal["benchmark"]
    ok: bool
    task: str
    results: list[BenchResult] = Field(default_factory=list)
    scoreboard: str = ""
    available_tasks: list[str] = Field(default_factory=list)
    error: GraphRlmError | None = None


class GraphRlmEvolutionResult(_StrictModel):
    action: Literal["evolve_prompt"]
    ok: bool
    winning_prompt: str | None = None
    scores: dict[str, float] = Field(default_factory=dict)
    generation: int | None = None
    reward_weights: dict[str, float] = Field(default_factory=dict)
    frontier_size: int = 0
    error: GraphRlmError | None = None


GraphRlmResult: TypeAlias = Annotated[
    GraphRlmRunResult | GraphRlmBenchmarkResult | GraphRlmEvolutionResult,
    Field(discriminator="action"),
]

_GRAPH_RLM_SCOPES: dict[type[BaseModel], Literal["kg:read", "kg:write"]] = {
    GraphRlmRunRequest: "kg:read",
    GraphRlmBenchmarkRequest: "kg:read",
    GraphRlmEvolvePromptRequest: "kg:write",
}


class AgentControlPlane:
    """AU application operations bound to verified graph identity and policy.

    Work-item, capability, execution, and signed-dispatch behavior is supplied
    through explicit ports. The control plane deliberately has no legacy graph
    engine adapter and never reconstructs those operations with raw queries.
    """

    def __init__(
        self,
        eg_client: Any,
        session: GraphSession,
        *,
        capability_search: CapabilitySearchPort | None = None,
        agent_executor: AgentExecutionPort | None = None,
        work_item_store: WorkItemStorePort | None = None,
        signed_dispatch: SignedAgentDispatchPort | None = None,
    ) -> None:
        if eg_client is None:
            raise ValueError("a session-routed epistemic-graph client is required")
        if not isinstance(session, GraphSession):
            raise TypeError("a verified GraphSession is required")
        use_verified_context = getattr(eg_client, "use_verified_context", None)
        if not callable(use_verified_context):
            raise TypeError("the epistemic-graph client must support verified context")

        self._eg_client = eg_client
        self._session = session
        self._capability_search = capability_search
        self._agent_executor = agent_executor
        self._work_item_store = work_item_store
        self._signed_dispatch = signed_dispatch
        # Validate eagerly so invalid or expired authority never creates a usable
        # control plane. It is checked again immediately before every call.
        session.engine_verified_context()

    @property
    def operation_descriptors(self) -> tuple[AgentOperationDescriptor, ...]:
        """Schemas for direct operations; these are metadata, not a dispatcher."""
        specs: tuple[_OperationSpec, ...] = (
            (
                "graph_rlm",
                GraphRlmRequest,
                GraphRlmResult,
                "kg:read",
                (("evolve_prompt", "kg:write"),),
            ),
            (
                "resolve_capability",
                CapabilitySearchRequest,
                CapabilityResolution,
                "kg:read",
                (),
            ),
            (
                "execute_agent",
                AgentExecutionRequest,
                AgentExecutionResult,
                "kg:write",
                (),
            ),
            (
                "submit_agent_task",
                AgentTaskDispatchRequest,
                AgentTaskDispatchResult,
                "kg:write",
                (),
            ),
            (
                "get_work_item",
                WorkItemGetRequest,
                WorkItemSnapshot | None,
                "kg:read",
                (),
            ),
            ("list_work_items", WorkItemListRequest, WorkItemPage, "kg:read", ()),
            (
                "cancel_work_item",
                WorkItemCancelRequest,
                WorkItemSnapshot | None,
                "kg:write",
                (),
            ),
        )
        return tuple(
            AgentOperationDescriptor(
                name=name,
                request_schema=TypeAdapter(request_type).json_schema(),
                result_schema=TypeAdapter(result_type).json_schema(),
                required_scope=scope,
                action_scopes=action_scopes,
            )
            for name, request_type, result_type, scope, action_scopes in specs
        )

    def _verified_session(
        self, required_scope: Literal["kg:read", "kg:write"] | None
    ) -> GraphSession:
        session = resolve_session(self._session, required_scope=required_scope)
        session.engine_verified_context()
        return session

    @contextmanager
    def _verified_client_context(self, session: GraphSession) -> Iterator[None]:
        claims = session.engine_verified_context()
        with self._eg_client.use_verified_context(claims):
            yield

    @staticmethod
    def _require_port(port: Any, name: str) -> Any:
        if port is None:
            raise AgentControlPlaneUnavailable(
                f"the {name} application port is not configured"
            )
        return port

    async def resolve_capability(
        self, request: CapabilitySearchRequest
    ) -> CapabilityResolution:
        """Return only a capability authorized by the injected EG search port."""
        if not isinstance(request, CapabilitySearchRequest):
            raise TypeError("request must be a validated CapabilitySearchRequest")
        session = self._verified_session("kg:read")
        port = self._require_port(self._capability_search, "capability-search")
        with self._verified_client_context(session):
            candidates = tuple(await port.search(request, session=session))
        if any(not isinstance(item, CapabilityCandidate) for item in candidates):
            raise AgentControlPlaneUnavailable(
                "the capability-search port returned an invalid candidate"
            )

        if request.agent_name is not None:
            candidates = tuple(
                item for item in candidates if item.name == request.agent_name
            )
            if not candidates:
                raise LookupError("the requested agent is not an authorized capability")

        ranked = sorted(
            candidates,
            key=lambda item: (-item.score, item.name.casefold(), item.component_id),
        )
        if not ranked:
            raise LookupError("no authorized capability matched the task")
        selected = ranked[0]
        return CapabilityResolution(
            kind=selected.kind,
            name=selected.name,
            component_id=selected.component_id,
            score=selected.score,
            source="caller" if request.agent_name is not None else "eg_search",
            alternatives=tuple(ranked[1:4]),
        )

    async def _prepare_agent_task(
        self, task: str, agent_name: str | None
    ) -> tuple[str, CapabilityResolution]:
        from agent_utilities.orchestration.task_guard import (
            screen_and_redact_agent_task,
        )

        sanitized_task = screen_and_redact_agent_task(task)
        capability = await self.resolve_capability(
            CapabilitySearchRequest(task=sanitized_task, agent_name=agent_name)
        )
        return sanitized_task, capability

    async def execute_agent(
        self, request: AgentExecutionRequest
    ) -> AgentExecutionResult:
        """Delegate execution only to a composed AU execution implementation."""
        if not isinstance(request, AgentExecutionRequest):
            raise TypeError("request must be a validated AgentExecutionRequest")
        session = self._verified_session("kg:write")
        port = self._require_port(self._agent_executor, "agent-execution")
        sanitized_task, capability = await self._prepare_agent_task(
            request.task, request.agent_name
        )
        authorized_request = request.model_copy(
            update={"agent_name": capability.name, "task": sanitized_task}
        )
        with self._verified_client_context(session):
            result = await port.execute_agent(authorized_request, session=session)
        if not isinstance(result, AgentExecutionResult):
            raise AgentControlPlaneUnavailable(
                "the agent-execution port returned an invalid result"
            )
        return result

    async def submit_agent_task(
        self, request: AgentTaskDispatchRequest
    ) -> AgentTaskDispatchResult:
        """Screen, admit, and signed-enqueue one typed agent task."""
        if not isinstance(request, AgentTaskDispatchRequest):
            raise TypeError("request must be a validated AgentTaskDispatchRequest")
        session = self._verified_session("kg:write")
        store = self._require_port(self._work_item_store, "work-item-store")
        dispatch = self._require_port(self._signed_dispatch, "signed-dispatch")
        self._reject_authority_metadata(request.metadata)
        sanitized_task, capability = await self._prepare_agent_task(
            request.task, request.agent_name
        )

        submission = WorkItemSubmission(
            work_item_id=request.work_item_id,
            idempotency_key=request.idempotency_key,
            kind="orchestrator_task",
            description=sanitized_task,
            metadata=dict(request.metadata),
        )
        with self._verified_client_context(session):
            admission = await store.submit(submission, session=session)
        if not isinstance(admission, WorkItemSubmissionResult):
            raise AgentControlPlaneUnavailable(
                "the work-item store returned an invalid admission result"
            )
        if admission.item.work_item_id != request.work_item_id:
            raise AgentControlPlaneUnavailable(
                "the work-item store returned an unrelated admission"
            )

        receipt: SignedAgentDispatchReceipt | None = None
        if admission.item.status in {"submitted", "ready"}:
            dispatch_request = SignedAgentDispatchRequest(
                job_id=request.job_id,
                work_item_id=admission.item.work_item_id,
                session_ref=request.session_ref,
                agent_name=capability.name,
            )
            with self._verified_client_context(session):
                receipt = await dispatch.enqueue(dispatch_request, session=session)
            if (
                not isinstance(receipt, SignedAgentDispatchReceipt)
                or receipt.job_id != request.job_id
                or not receipt.accepted
            ):
                raise AgentControlPlaneUnavailable(
                    "the signed-dispatch port did not accept the admitted work item"
                )
        return AgentTaskDispatchResult(
            capability=capability, admission=admission, dispatch=receipt
        )

    async def get_work_item(
        self, request: WorkItemGetRequest
    ) -> WorkItemSnapshot | None:
        if not isinstance(request, WorkItemGetRequest):
            raise TypeError("request must be a validated WorkItemGetRequest")
        session = self._verified_session("kg:read")
        store = self._require_port(self._work_item_store, "work-item-store")
        with self._verified_client_context(session):
            result = await store.get(request, session=session)
        if result is not None and not isinstance(result, WorkItemSnapshot):
            raise AgentControlPlaneUnavailable(
                "the work-item store returned an invalid snapshot"
            )
        return result

    async def list_work_items(self, request: WorkItemListRequest) -> WorkItemPage:
        if not isinstance(request, WorkItemListRequest):
            raise TypeError("request must be a validated WorkItemListRequest")
        session = self._verified_session("kg:read")
        store = self._require_port(self._work_item_store, "work-item-store")
        with self._verified_client_context(session):
            result = await store.list(request, session=session)
        if not isinstance(result, WorkItemPage):
            raise AgentControlPlaneUnavailable(
                "the work-item store returned an invalid page"
            )
        return result

    async def cancel_work_item(
        self, request: WorkItemCancelRequest
    ) -> WorkItemSnapshot | None:
        if not isinstance(request, WorkItemCancelRequest):
            raise TypeError("request must be a validated WorkItemCancelRequest")
        session = self._verified_session("kg:write")
        store = self._require_port(self._work_item_store, "work-item-store")
        with self._verified_client_context(session):
            result = await store.cancel(request, session=session)
        if result is not None and not isinstance(result, WorkItemSnapshot):
            raise AgentControlPlaneUnavailable(
                "the work-item store returned an invalid snapshot"
            )
        return result

    @classmethod
    def _reject_authority_metadata(cls, value: Any) -> None:
        authority_names = {
            "actor",
            "actor_id",
            "audience",
            "auth",
            "authorization",
            "graph",
            "owner",
            "owner_id",
            "policy",
            "policy_version",
            "principal",
            "scope",
            "scopes",
            "tenant",
            "tenant_id",
        }
        if isinstance(value, dict):
            for key, nested in value.items():
                normalized = str(key).casefold().replace("-", "_")
                if normalized in authority_names:
                    raise ValueError("metadata may not contain caller authority fields")
                cls._reject_authority_metadata(nested)
        elif isinstance(value, (list, tuple)):
            for nested in value:
                cls._reject_authority_metadata(nested)

    async def graph_rlm(self, request: GraphRlmRequest) -> GraphRlmResult:
        """Execute one typed RLM action under the injected verified authority."""
        if not isinstance(
            request,
            (GraphRlmRunRequest, GraphRlmBenchmarkRequest, GraphRlmEvolvePromptRequest),
        ):
            raise TypeError("request must be a validated GraphRlmRequest")

        try:
            resolved_session = self._verified_session(_GRAPH_RLM_SCOPES[type(request)])
            with self._verified_client_context(resolved_session):
                return await self._execute(request)
        except PermissionError as exc:
            return self._failed_result(request, exc, code="permission_denied")
        except Exception as exc:  # noqa: BLE001 - public API returns typed failures
            return self._failed_result(request, exc)

    async def _execute(self, request: GraphRlmRequest) -> GraphRlmResult:
        if isinstance(request, GraphRlmRunRequest):
            from agent_utilities.rlm.runner import run_rlm

            output = await run_rlm(request.task, input_text=request.input_text)
            if not output.get("ok"):
                failure_class = output.get("failure_class", "unknown")
                if failure_class not in {
                    "model_generated_bad_code",
                    "host_tool_timeout",
                    "sandbox_exec_timeout",
                    "sandbox_fatal",
                    "sandbox_escalated",
                    "evaluator_reject",
                    "unknown",
                }:
                    failure_class = "unknown"
                return GraphRlmRunResult(
                    action="run",
                    ok=False,
                    task=request.task,
                    failure_class=failure_class,
                    error=self._failure(
                        RuntimeError("RLM execution failed"),
                        code="operation_failed",
                    ),
                )
            return GraphRlmRunResult(
                action="run",
                ok=True,
                task=request.task,
                result=output.get("result"),
                usage=output.get("usage") or {},
                max_depth=int(output.get("max_depth") or 0),
            )

        if isinstance(request, GraphRlmBenchmarkRequest):
            from agent_utilities.rlm.benchmarks import (
                list_tasks,
                render_scoreboard,
                run_benchmark,
            )

            benchmark = request.task or "s_niah"
            available_tasks = list_tasks()
            if benchmark not in available_tasks:
                error = self._failure(
                    ValueError("Unknown benchmark task"), code="invalid_request"
                )
                return GraphRlmBenchmarkResult(
                    action="benchmark",
                    ok=False,
                    task=benchmark,
                    available_tasks=available_tasks,
                    error=error,
                )
            results = await run_benchmark(
                benchmark,
                scales=request.options.scales,
                cases_per_scale=request.options.cases_per_scale,
            )
            return GraphRlmBenchmarkResult(
                action="benchmark",
                ok=True,
                task=benchmark,
                results=results,
                scoreboard=render_scoreboard(results),
            )

        return await _evolve_prompt(request)

    def _failed_result(
        self,
        request: GraphRlmRequest,
        exc: BaseException,
        *,
        code: str = "operation_failed",
    ) -> GraphRlmResult:
        error = self._failure(exc, code=code)
        if isinstance(request, GraphRlmRunRequest):
            return GraphRlmRunResult(
                action="run", ok=False, task=request.task, error=error
            )
        if isinstance(request, GraphRlmBenchmarkRequest):
            return GraphRlmBenchmarkResult(
                action="benchmark", ok=False, task=request.task, error=error
            )
        return GraphRlmEvolutionResult(action="evolve_prompt", ok=False, error=error)

    @staticmethod
    def _failure(exc: BaseException, *, code: str) -> GraphRlmError:
        payload = public_error_payload(
            exc, code=code, context={"operation": "graph_rlm"}
        )
        return GraphRlmError(
            code=payload["error"]["code"],
            correlation_id=payload["error"]["correlation_id"],
            detail_ref=payload["error"].get("detail_ref"),
            retryable=payload["error"]["retryable"],
            error_class=payload["error_class"],
            failing_layer=payload["failing_layer"],
        )


def compose_agent_control_plane(
    eg_client: Any,
    session: GraphSession,
    *,
    capability_search: CapabilitySearchPort | None = None,
    agent_executor: AgentExecutionPort | None = None,
    work_item_store: WorkItemStorePort | None = None,
    signed_dispatch: SignedAgentDispatchPort | None = None,
) -> AgentControlPlane:
    """Compose AU application operations from verified inputs and explicit ports."""
    return AgentControlPlane(
        eg_client,
        session,
        capability_search=capability_search,
        agent_executor=agent_executor,
        work_item_store=work_item_store,
        signed_dispatch=signed_dispatch,
    )


async def _evolve_prompt(
    request: GraphRlmEvolvePromptRequest,
) -> GraphRlmEvolutionResult:
    """Run the retained GEPA/DW-GRPO prompt-evolution behavior."""
    from pydantic import BaseModel

    from agent_utilities.harness.program_optimization import graded_score
    from agent_utilities.rlm.gepa import GEPAInstance, GEPAOptimizer
    from agent_utilities.rlm.predict_rlm import InputField, OutputField

    class PromptEvolutionSignature(BaseModel):
        """Answer the given query."""

        query: str = InputField(default="", description="The task input.")
        response: str = OutputField(default="", description="The model's answer.")

    options = request.options
    rows = options.dataset or [
        GraphRlmPromptExample(query="What is the capital of France?", response="Paris"),
        GraphRlmPromptExample(query="What is 2 + 2?", response="4"),
    ]
    dataset = [
        GEPAInstance(
            id=f"inst_{index}",
            input_data={"query": row.query},
            reference_output=row.response,
        )
        for index, row in enumerate(rows)
    ]

    async def evaluator(
        instance: GEPAInstance, model_output: Any, _trace: str
    ) -> tuple[dict[str, float], str]:
        score = graded_score(
            str(instance.reference_output or ""),
            getattr(model_output, "response", "") or "",
        )
        return {"accuracy": score}, f"graded_score={score:.3f}"

    optimizer = GEPAOptimizer(
        signature_class=PromptEvolutionSignature,
        base_prompt=request.task or "Answer the user's query accurately and concisely.",
        evaluator_fn=evaluator,
        objectives=options.objectives,
    )
    best = await optimizer.optimize(
        dataset,
        iterations=options.iterations,
        batch_size=options.batch_size,
    )
    return GraphRlmEvolutionResult(
        action="evolve_prompt",
        ok=True,
        winning_prompt=best.prompt_text,
        scores=best.scores,
        generation=best.generation,
        reward_weights=optimizer.pool.reward_weights,
        frontier_size=len(optimizer.pool.get_frontier()),
    )
