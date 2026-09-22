"""The verified, application-level RLM operation retained by GraphOS.

GraphOS owns transport registration, routing, and its public action manifest. This
module owns the RLM behavior and requires the caller to bind it to an injected,
session-scoped epistemic-graph client before any operation can run.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, field_validator

from agent_utilities.knowledge_graph.core.session import GraphSession, resolve_session
from agent_utilities.rlm.benchmarks.base import BenchResult
from agent_utilities.rlm.telemetry import FailureClass
from agent_utilities.security.error_surface import public_error_payload


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


class AgentControlPlane:
    """Per-instance RLM behavior bound to verified graph identity and policy."""

    def __init__(self, eg_client: Any, session: GraphSession) -> None:
        if eg_client is None:
            raise ValueError("a session-routed epistemic-graph client is required")
        if not isinstance(session, GraphSession):
            raise TypeError("a verified GraphSession is required")
        use_verified_context = getattr(eg_client, "use_verified_context", None)
        if not callable(use_verified_context):
            raise TypeError("the epistemic-graph client must support verified context")

        self._eg_client = eg_client
        self._session = session
        # Validate eagerly so invalid or expired authority never creates a usable
        # control plane. It is checked again immediately before every call.
        session.engine_verified_context()

    async def graph_rlm(self, request: GraphRlmRequest) -> GraphRlmResult:
        """Execute one typed RLM action under the injected verified authority."""
        if not isinstance(
            request,
            (GraphRlmRunRequest, GraphRlmBenchmarkRequest, GraphRlmEvolvePromptRequest),
        ):
            raise TypeError("request must be a validated GraphRlmRequest")

        try:
            resolved_session = resolve_session(self._session)
            claims = resolved_session.engine_verified_context()
            with self._eg_client.use_verified_context(claims):
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
    eg_client: Any, session: GraphSession
) -> AgentControlPlane:
    """Compose the one retained AU application operation from verified inputs."""
    return AgentControlPlane(eg_client, session)


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
