"""Enforce a received text-only constraint at the existing model boundary.

This is an in-run mechanism, not anonymous admission or a provider policy.
Only the ordinary OpenAI chat protocol is supported by this initial slice.
"""

from __future__ import annotations

import asyncio
import math
from contextlib import asynccontextmanager
from typing import Any

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    TextPart,
    UserPromptPart,
)
from pydantic_ai.models.wrapper import WrapperModel


def _require(condition: bool) -> None:
    if not condition:
        raise ValueError("received text-only execution is unavailable")


def validate_text_plan(plan: Any) -> None:
    """Reject widened plans before any host history, lease or publication hook."""
    from pydantic_ai.models.openai import OpenAIChatModel

    from .state import HarnessConfig, RunBudget, RunPlan, StopRule

    _require(type(plan) is RunPlan)
    _require(type(plan.model) is OpenAIChatModel)
    _require(type(plan.harness) is HarnessConfig and type(plan.budget) is RunBudget)
    _require(type(plan.stop) is StopRule)
    _require(plan.execution_constraint == "text-only")
    _require(type(plan.steps) is tuple and len(plan.steps) == 1)
    _require(type(plan.steps[0]) is str and 0 < len(plan.steps[0]) <= 65536)
    _require(type(plan.instructions) is str and len(plan.instructions) <= 65536)
    _require(plan.tool_names == () and plan.toolsets == ())
    _require(plan.input_media_types == () and plan.dependency_plan is None)
    _require(plan.client_transport is None and plan.harness.client_tools == ())
    _require(plan.harness.client_tool_limits is None)
    _require(plan.harness.plan_scaffold == "off")
    _require(plan.harness.response_format == "text")
    _require(type(plan.harness.retries) is int and plan.harness.retries == 0)
    _require(type(plan.stop.max_steps) is int and plan.stop.max_steps == 1)
    _require(plan.stop.stop_markers == ())
    for name in ("request_limit", "total_tokens_limit"):
        value = getattr(plan.budget, name)
        _require(type(value) is int and 0 < value <= 2**53 - 1)
    for name in ("tool_calls_limit", "max_attachments", "max_attachment_bytes"):
        value = getattr(plan.budget, name)
        _require(type(value) is int and value == 0)
    _require(
        type(plan.budget.max_concurrent_requests) is int
        and plan.budget.max_concurrent_requests == 1
    )
    seconds = plan.budget.timeout_seconds
    _require(
        type(seconds) in (int, float)
        and math.isfinite(seconds)
        and 0 < seconds <= 2**53 - 1
    )
    _settings(
        plan.harness.model_settings, plan.budget.total_tokens_limit, required=True
    )


def _settings(value: Any, ceiling: int, *, required: bool = False) -> None:
    _require(value is None or type(value) is dict)
    settings = value or {}
    _require(set(settings) <= {"max_tokens"})
    if required or "max_tokens" in settings:
        cap = settings.get("max_tokens")
        _require(type(cap) is int and 0 < cap <= ceiling)


class TextOnlyModel(WrapperModel):
    """Check the exact resolved provider and effective request before dispatch."""

    def __init__(self, model: Any, plan: Any, *, deadline: float | None) -> None:
        from pydantic_ai.models.openai import OpenAIChatModel

        validate_text_plan(plan)
        _require(type(model) is OpenAIChatModel)
        super().__init__(model)
        self._timeout = plan.budget.timeout_seconds
        self._ceiling = plan.budget.total_tokens_limit
        _require(type(deadline) in (int, float))
        assert deadline is not None
        _require(math.isfinite(deadline))
        self._deadline = deadline
        self._check_provider()

    def _check_provider(self) -> None:
        from httpx import Timeout
        from openai import AsyncOpenAI
        from pydantic_ai.models.openai import OpenAIChatModel
        from pydantic_ai.providers.openai import OpenAIProvider

        _require(type(self.wrapped) is OpenAIChatModel)
        provider = self.wrapped.provider
        if type(provider) is not OpenAIProvider:
            raise ValueError("received text-only provider is unavailable")
        _require(type(provider.client) is AsyncOpenAI)
        _require(
            type(provider.client.max_retries) is int
            and provider.client.max_retries == 0
        )
        timeout = provider.client.timeout
        values: tuple[int | float | None, ...]
        if type(timeout) in (int, float):
            assert isinstance(timeout, (int, float))
            values = (timeout,)
        else:
            _require(isinstance(timeout, Timeout))
            assert isinstance(timeout, Timeout)
            values = (timeout.connect, timeout.read, timeout.write, timeout.pool)
        for value in values:
            _require(type(value) in (int, float))
            assert isinstance(value, (int, float))
            _require(math.isfinite(value) and 0 < value <= self._timeout)
        _require(not provider.client.default_query)
        _settings(self.wrapped.settings, self._ceiling)

    def _live(self) -> None:
        task = asyncio.current_task()
        if task is not None and task.cancelling():
            raise asyncio.CancelledError
        if asyncio.get_running_loop().time() >= self._deadline:
            raise TimeoutError

    async def request(
        self, messages: Any, model_settings: Any, model_request_parameters: Any
    ) -> ModelResponse:
        self._live()
        self._check_provider()
        _settings(model_settings, self._ceiling, required=True)
        params = model_request_parameters
        _require(
            not params.function_tools
            and not params.native_tools
            and not params.output_tools
        )
        _require(params.output_mode == "text" and params.output_object is None)
        _require(
            params.allow_text_output is True and params.allow_image_output is False
        )
        for message in messages:
            _require(type(message) in (ModelRequest, ModelResponse))
            for part in message.parts:
                _require(type(part) in (SystemPromptPart, UserPromptPart, TextPart))
                _require(type(part.content) is str)
        try:
            return await self.wrapped.request(messages, model_settings, params)
        except Exception as exc:
            raise RuntimeError("text-only provider outcome is unavailable") from exc

    @asynccontextmanager
    async def request_stream(self, *args: Any, **kwargs: Any) -> Any:
        raise ValueError("received text-only streaming is unavailable")
        yield  # pragma: no cover


@asynccontextmanager
async def no_host_scope() -> Any:
    """A restricted run owns no inherited host lease or scaffold publisher."""
    yield None


class TextOnlyResponseFence(AbstractCapability[Any]):
    """Refuse unsupported replies after accounting, before tools or retry."""

    def __init__(self, model: TextOnlyModel) -> None:
        self._model = model

    async def after_model_request(
        self, ctx: Any, *, request_context: Any, response: ModelResponse
    ) -> ModelResponse:
        self._model._live()
        self._model._check_provider()
        _require(type(response) is ModelResponse)
        _require(
            all(
                type(part) is TextPart and type(part.content) is str
                for part in response.parts
            )
        )
        return response
