"""In-run reservation of received aggregate limits across parallel tasks."""

from __future__ import annotations

import asyncio
import math
from contextlib import nullcontext
from typing import Any

from pydantic_ai import RunContext
from pydantic_ai.capabilities import AbstractCapability, WrapModelRequestHandler
from pydantic_ai.exceptions import UsageLimitExceeded
from pydantic_ai.messages import ModelResponse, ToolCallPart
from pydantic_ai.models import ModelRequestContext
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.usage import RunUsage, UsageLimits

from .state import RunBudget, RunPlan, RunPlanFailure


def initial_run_usage(plan: RunPlan | RunPlanFailure) -> RunUsage:
    """Snapshot only the public counters; never estimate unreported usage."""
    usage = plan.initial_usage
    if not isinstance(usage, RunUsage):
        raise ValueError("initial_usage must be RunUsage")
    counters = {
        key: getattr(usage, key)
        for key in ("requests", "tool_calls", "input_tokens", "output_tokens")
    }
    if any(type(value) is not int or value < 0 for value in counters.values()):
        raise ValueError("initial_usage counters must be nonnegative integers")
    if usage.total_tokens != counters["input_tokens"] + counters["output_tokens"]:
        raise ValueError("initial_usage total_tokens must match input and output")
    elapsed = plan.initial_elapsed_seconds
    if type(elapsed) not in (int, float) or not math.isfinite(elapsed) or elapsed < 0:
        raise ValueError("initial_elapsed_seconds must be finite and nonnegative")
    return RunUsage(**counters)


async def _request_drained(
    handler: WrapModelRequestHandler, request_context: ModelRequestContext
) -> ModelResponse:
    """Deliver cancellation once despite the framework's level cancel scopes."""

    async def call() -> ModelResponse:
        return await handler(request_context)

    request = asyncio.create_task(call())
    interrupted: BaseException | None = None
    response: ModelResponse | None = None
    try:
        response = await asyncio.shield(request)
    except BaseException as exc:
        interrupted = exc
    finally:
        if not request.done():
            request.cancel()
        drain = asyncio.gather(request, return_exceptions=True)
        while not drain.done():
            try:
                await asyncio.shield(drain)
            except asyncio.CancelledError as exc:
                interrupted = interrupted or exc
    if interrupted is not None:
        raise interrupted
    assert response is not None
    return response


class SharedBudget(AbstractCapability[Any]):
    """Reserve before dispatch, never reset the budget for a child or retry.

    Reservations have no await between their check and increment. They are
    atomic within the run's event loop. A failed/cancelled attempt still consumes
    its reservation. An unspecified tool limit uses a finite fallback of 50.
    Token accounting uses reported responses, as in the serial
    loop: unknown in-flight token usage cannot be predicted or pre-reserved.
    """

    def __init__(
        self,
        budget: RunBudget,
        *,
        max_concurrent_requests: int | None = None,
        initial_usage: RunUsage | None = None,
    ) -> None:
        self.usage = RunUsage()
        if initial_usage is not None:
            self.usage.incr(initial_usage)
        self._has_initial_usage = initial_usage is not None
        self.limits = UsageLimits(
            request_limit=budget.request_limit,
            tool_calls_limit=50
            if budget.tool_calls_limit is None
            else budget.tool_calls_limit,
            total_tokens_limit=budget.total_tokens_limit,
        )
        self.slots = (
            asyncio.Semaphore(max_concurrent_requests)
            if max_concurrent_requests is not None
            else None
        )

    def check_initial_consumption(self) -> None:
        """Refuse an exhausted host ledger before constructing any child."""
        self.limits.check_before_request(self.usage)
        token_limit = self.limits.total_tokens_limit
        if token_limit is not None and self.usage.total_tokens >= token_limit:
            raise UsageLimitExceeded(
                f"Initial usage exhausts the total_tokens_limit of {token_limit}"
            )
        # An exact tool cap still allows a model-only answer. Pydantic AI's
        # generic check treats zero as an unspecified limit, so enforce the
        # received bound explicitly for both prior and new host calls.
        limit = self.limits.tool_calls_limit
        if limit is not None and self.usage.tool_calls > limit:
            raise UsageLimitExceeded(
                f"Initial usage exceeds the tool_calls_limit of {limit}"
            )

    async def wrap_model_request(
        self,
        ctx: RunContext[Any],
        *,
        request_context: ModelRequestContext,
        handler: WrapModelRequestHandler,
    ) -> ModelResponse:
        return await self._wrap_model_request(ctx, request_context, handler, None)

    async def _wrap_model_request(
        self,
        ctx: RunContext[Any],
        request_context: ModelRequestContext,
        handler: WrapModelRequestHandler,
        local_usage: RunUsage | None,
    ) -> ModelResponse:
        # Only a model request holds a permit, never a parent waiting on its
        # child tools. Nested recursive runs cannot deadlock the request pool.
        async with self.slots if self.slots is not None else nullcontext():
            self._ensure_live()
            self.limits.check_before_request(self.usage)
            token_limit = self.limits.total_tokens_limit
            if (
                self._has_initial_usage
                and token_limit is not None
                and self.usage.total_tokens >= token_limit
            ):
                raise UsageLimitExceeded(
                    f"The next request would exceed the total_tokens_limit of {token_limit}"
                )
            self.usage.requests += 1
            if local_usage is not None:
                local_usage.requests += 1
            before_input = ctx.usage.input_tokens
            before_output = ctx.usage.output_tokens
            response: ModelResponse | None = None
            try:
                response = await _request_drained(handler, request_context)
            finally:
                # The framework records provider usage before response hooks,
                # including billed fallback attempts and hook-induced retries.
                # Local node usage avoids counting another child's work twice.
                input_delta = ctx.usage.input_tokens - before_input
                output_delta = ctx.usage.output_tokens - before_output
                # FunctionModel and a few host adapters return RequestUsage
                # without mutating RunContext. Preserve their reported usage
                # while keeping the context delta authoritative when present.
                if response is not None:
                    if input_delta == 0 and type(response.usage.input_tokens) is int:
                        input_delta = response.usage.input_tokens
                    if output_delta == 0 and type(response.usage.output_tokens) is int:
                        output_delta = response.usage.output_tokens
                self.usage.input_tokens += input_delta
                self.usage.output_tokens += output_delta
                if local_usage is not None:
                    local_usage.input_tokens += input_delta
                    local_usage.output_tokens += output_delta
            self.limits.check_tokens(self.usage)
            assert response is not None
            return response

    async def before_tool_execute(
        self,
        ctx: RunContext[Any],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: dict[str, Any],
    ) -> dict[str, Any]:
        self._reserve_tool_call(None)
        return args

    def reserve_tool_call(self) -> None:
        """Reserve one host-bound dependency call before it can have an effect."""
        self._reserve_tool_call(None)

    def _reserve_tool_call(self, local_usage: RunUsage | None) -> None:
        self._ensure_live()
        limit = self.limits.tool_calls_limit
        if limit is not None and self.usage.tool_calls >= limit:
            raise UsageLimitExceeded(
                f"The next tool call would exceed the tool_calls_limit of {limit}"
            )
        self.usage.tool_calls += 1
        if local_usage is not None:
            local_usage.tool_calls += 1

    def _ensure_live(self) -> None:
        """Ordinary run budgets remain valid for their owning run."""


class LiveSharedBudget(SharedBudget):
    """One host-held, in-process authority shared by concurrent runs.

    This object is neither serializable authority nor a durable spending
    ledger. The host must only pass it to runs it admitted under the same
    execution, and close it when that authority ends. It cannot survive a
    process restart or cross an event loop.
    """

    def __init__(self, budget: RunBudget, *, deadline: float) -> None:
        loop = asyncio.get_running_loop()
        if (
            not math.isfinite(deadline)
            or deadline > loop.time() + budget.timeout_seconds
        ):
            raise ValueError("shared budget deadline is invalid")
        # Providers report tokens only after a response. Serialize requests
        # under a finite token limit so overlapping calls cannot each observe
        # the same remaining allowance before either reports usage.
        slots = (
            1
            if budget.total_tokens_limit is not None
            else budget.max_concurrent_requests
        )
        super().__init__(budget, max_concurrent_requests=slots)
        # Enforce the token ceiling at equality even when this authority began
        # with no recorded usage.
        self._has_initial_usage = True
        self.budget = budget
        self.deadline = deadline
        self._loop = loop
        self._active = True

    def _ensure_live(self) -> None:
        if not self._active or asyncio.get_running_loop() is not self._loop:
            raise RuntimeError("shared execution budget is not live")
        if self._loop.time() >= self.deadline:
            raise TimeoutError("shared execution budget expired")

    def require(self, budget: RunBudget) -> None:
        self._ensure_live()
        if budget != self.budget:
            raise ValueError("shared execution budget does not match the received plan")
        self.check_initial_consumption()

    def run_view(self) -> SharedBudget:
        """Create a run-local accounting view over this same live authority."""
        self._ensure_live()
        return _SharedBudgetView(self)

    def close(self) -> None:
        """Prevent new effects; the host remains responsible for draining in-flight work."""
        self._active = False


class _SharedBudgetView(SharedBudget):
    """Per-run counters; all reservations still debit the same live owner."""

    def __init__(self, owner: LiveSharedBudget) -> None:
        self._owner = owner
        self.usage = RunUsage()
        self.limits = owner.limits
        self.slots = owner.slots

    async def wrap_model_request(
        self,
        ctx: RunContext[Any],
        *,
        request_context: ModelRequestContext,
        handler: WrapModelRequestHandler,
    ) -> ModelResponse:
        return await self._owner._wrap_model_request(
            ctx, request_context, handler, self.usage
        )

    async def before_tool_execute(
        self,
        ctx: RunContext[Any],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: dict[str, Any],
    ) -> dict[str, Any]:
        self._owner._reserve_tool_call(self.usage)
        return args

    def reserve_tool_call(self) -> None:
        self._owner._reserve_tool_call(self.usage)
