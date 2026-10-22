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
        # An exact tool cap still allows a model-only answer. Only prior usage
        # above the cap refuses the run; before_tool_execute rejects new tools.
        self.limits.check_before_tool_call(self.usage)

    async def wrap_model_request(
        self,
        ctx: RunContext[Any],
        *,
        request_context: ModelRequestContext,
        handler: WrapModelRequestHandler,
    ) -> ModelResponse:
        # Only a model request holds a permit, never a parent waiting on its
        # child tools. Nested recursive runs cannot deadlock the request pool.
        async with self.slots if self.slots is not None else nullcontext():
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
            before_input = ctx.usage.input_tokens
            before_output = ctx.usage.output_tokens
            try:
                response = await _request_drained(handler, request_context)
            finally:
                # The framework records provider usage before response hooks,
                # including billed fallback attempts and hook-induced retries.
                # Local node usage avoids counting another child's work twice.
                self.usage.input_tokens += ctx.usage.input_tokens - before_input
                self.usage.output_tokens += ctx.usage.output_tokens - before_output
            self.limits.check_tokens(self.usage)
            return response

    async def before_tool_execute(
        self,
        ctx: RunContext[Any],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: dict[str, Any],
    ) -> dict[str, Any]:
        limit = self.limits.tool_calls_limit
        if limit is not None and self.usage.tool_calls >= limit:
            raise UsageLimitExceeded(
                f"The next tool call would exceed the tool_calls_limit of {limit}"
            )
        self.usage.tool_calls += 1
        return args
