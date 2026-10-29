"""The recursive run loop.

The root agent never receives the whole input. It sees the input length and a
preview, reads slices through ``read_input``, and hands bounded sub-questions
to ``sub_query``, which runs a child agent one level deeper. The loop stops
recursing at ``max_depth`` and enforces the request and token budgets it is
given on every level. Model, depth and budgets come from :class:`RLMConfig`;
when the config names no model, the host run plan for agent ``"rlm"`` supplies
the model and budget (see :func:`agent_utilities.graph.resolve_plan`).
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pydantic_ai import Agent, RunContext
from pydantic_ai.usage import UsageLimits

from agent_utilities import host_ports
from agent_utilities.graph.budget import SharedBudget
from agent_utilities.graph.executor import resolve_plan
from agent_utilities.graph.live_events import (
    LineageBinding,
    LineageIdentityError,
    emit_lineage_event,
    lineage_relationship,
)
from agent_utilities.graph.state import RunBudget, RunRequest

from .telemetry import SandboxFatalError, classify_failure

ROOT_INSTRUCTIONS = (
    "Answer the task. The input is not in this prompt: read it with "
    "read_input(start, end) and delegate focused questions over a slice with "
    "sub_query(question, start, end)."
)


@dataclass(frozen=True)
class RLMConfig:
    """Limits and models of one recursive run, as received from the host."""

    model: Any = None
    sub_model: Any = None
    max_depth: int = 3
    preview_chars: int = 500
    max_slice_chars: int = 20_000
    request_limit: int = 50
    total_tokens_limit: int | None = None
    timeout_seconds: float = 600.0
    max_concurrent_requests: int = 4
    tool_calls_limit: int | None = 50


@dataclass
class _Env:
    text: str
    depth: int
    config: RLMConfig
    budget: SharedBudget
    lineage: LineageBinding | None = None
    child_identity_provider: (
        Callable[[LineageBinding, str], LineageBinding | None] | None
    ) = None
    event_sequence: int = 0


def _emit_lineage(
    env: _Env,
    kind: str,
    *,
    relationships: tuple[dict[str, Any], ...] = (),
    **fields: Any,
) -> dict[str, Any] | None:
    """Publish one event only when the host supplied an explicit identity."""

    if env.lineage is None:
        return None
    env.event_sequence += 1
    return emit_lineage_event(
        kind,
        env.lineage,
        env.event_sequence,
        relationships=relationships,
        **fields,
    )


def _model(value: Any) -> Any:
    return host_ports.resolve_model(value) if isinstance(value, str) else value


def _agent(
    config: RLMConfig, depth: int, output_type: Any, budget: SharedBudget
) -> Agent[_Env, Any]:
    model = _model(config.model if depth == 0 else (config.sub_model or config.model))
    agent: Agent[_Env, Any] = Agent(
        model,
        output_type=output_type,
        deps_type=_Env,
        instructions=ROOT_INSTRUCTIONS,
        capabilities=[budget],
    )

    @agent.tool
    def read_input(ctx: RunContext[_Env], start: int = 0, end: int = 0) -> str:
        """Return input[start:end] (capped at the slice limit)."""
        stop = end or start + ctx.deps.config.max_slice_chars
        stop = min(stop, start + ctx.deps.config.max_slice_chars)
        return ctx.deps.text[start:stop]

    @agent.tool
    async def sub_query(
        ctx: RunContext[_Env], question: str, start: int = 0, end: int = 0
    ) -> str:
        """Answer ``question`` over input[start:end] with a child run."""
        if ctx.deps.depth + 1 >= ctx.deps.config.max_depth:
            return "refused: maximum recursion depth reached"
        child_lineage: LineageBinding | None = None
        if ctx.deps.lineage is not None:
            if ctx.deps.child_identity_provider is None:
                raise LineageIdentityError(
                    "lineage-bound child requires an admitted identity provider"
                )
            child_lineage = ctx.deps.child_identity_provider(ctx.deps.lineage, question)
            if child_lineage is None:
                raise LineageIdentityError(
                    "lineage-bound child requires an explicit identity"
                )
            if not isinstance(child_lineage, LineageBinding):
                raise LineageIdentityError(
                    "child identity provider returned an invalid binding"
                )
            relationship = lineage_relationship(
                "spawn", ctx.deps.lineage, child_lineage
            )
            _emit_lineage(
                ctx.deps,
                "child_spawned",
                relationships=(relationship,),
                child_run_id=child_lineage.run_id,
                child_node=child_lineage.local_id,
            )
        child = _Env(
            text=ctx.deps.text[start : end or None],
            depth=ctx.deps.depth + 1,
            config=ctx.deps.config,
            budget=ctx.deps.budget,
            lineage=child_lineage,
            child_identity_provider=ctx.deps.child_identity_provider,
        )
        _emit_lineage(child, "child_started")
        out = await _run_drained(question, child, str)
        return str(out)

    return agent


def _prompt(task: str, env: _Env) -> str:
    preview = env.text[: env.config.preview_chars]
    return f"{task}\n\n[input: {len(env.text)} chars; preview]\n{preview}"


async def _run_level(task: str, env: _Env, output_type: Any) -> Any:
    limits = UsageLimits(
        request_limit=None,
        tool_calls_limit=env.budget.limits.tool_calls_limit,
    )
    result = await _agent(env.config, env.depth, output_type, env.budget).run(
        _prompt(task, env), deps=env, usage_limits=limits
    )
    return result.output


async def _run_drained(task: str, env: _Env, output_type: Any) -> Any:
    """Cancel the framework once, then shield all descendant cleanup."""
    child = asyncio.create_task(_run_level(task, env, output_type))
    interrupted: BaseException | None = None
    value: Any = None
    try:
        value = await asyncio.shield(child)
    except BaseException as exc:
        interrupted = exc
    finally:
        if not child.done():
            child.cancel()
        drain = asyncio.gather(child, return_exceptions=True)
        while not drain.done():
            try:
                await asyncio.shield(drain)
            except asyncio.CancelledError as exc:
                # Preserve the original cancellation while refusing to deliver
                # another cancellation into a descendant's async finally block.
                interrupted = interrupted or exc
    if interrupted is not None:
        raise interrupted
    return value


async def _config_from_host(task: str) -> RLMConfig:
    plan = await resolve_plan(RunRequest(task=task, agent_name="rlm"), {})
    return RLMConfig(
        model=plan.model,
        request_limit=50
        if plan.budget.request_limit is None
        else plan.budget.request_limit,
        total_tokens_limit=plan.budget.total_tokens_limit,
        timeout_seconds=plan.budget.timeout_seconds,
        max_concurrent_requests=plan.budget.max_concurrent_requests,
        tool_calls_limit=50
        if plan.budget.tool_calls_limit is None
        else plan.budget.tool_calls_limit,
    )


async def run_rlm(
    task: str,
    input_text: str = "",
    *,
    config: RLMConfig | None = None,
    output_type: Any = str,
    lineage: LineageBinding | None = None,
    child_identity_provider: Callable[[LineageBinding, str], LineageBinding | None]
    | None = None,
) -> dict[str, Any]:
    """Run one recursive task over ``input_text``.

    Returns ``{"ok": True, "result", "task", "usage", "max_depth"}`` or
    ``{"ok": False, "error", "failure_class", "task"}``. A
    :class:`SandboxFatalError` propagates so the caller fails fast.
    """
    try:
        if lineage is not None and not isinstance(lineage, LineageBinding):
            raise TypeError("lineage must be a LineageBinding when supplied")
        if host_ports.run_checkpoint_factory_bound():
            raise ValueError("recursive checkpoint restore is unavailable")
        history_request = (
            RunRequest(task=task, agent_name="rlm", run_id=lineage.run_id)
            if lineage is not None
            else None
        )
        if await host_ports.read_recursive_run_history(history_request) is not None:
            raise ValueError("recursive conversation history is unavailable")
        cfg = config if config is not None and config.model else None
        cfg = cfg or await _config_from_host(task)
        for name in ("max_concurrent_requests",):
            value = getattr(cfg, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in (
            "request_limit",
            "tool_calls_limit",
            "total_tokens_limit",
            "max_depth",
        ):
            value = getattr(cfg, name)
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError(f"{name} must be a nonnegative integer")
        budget = SharedBudget(
            RunBudget(
                request_limit=cfg.request_limit,
                tool_calls_limit=cfg.tool_calls_limit,
                total_tokens_limit=cfg.total_tokens_limit,
            ),
            max_concurrent_requests=cfg.max_concurrent_requests,
        )
        if child_identity_provider is not None and lineage is None:
            raise ValueError("child identity provider requires a root lineage")
        env = _Env(
            text=input_text,
            depth=0,
            config=cfg,
            budget=budget,
            lineage=lineage,
            child_identity_provider=child_identity_provider,
        )
        _emit_lineage(env, "run_started")
        async with asyncio.timeout(cfg.timeout_seconds):
            value = await _run_drained(task, env, output_type)
    except SandboxFatalError:
        raise
    except Exception as exc:  # noqa: BLE001 - the entry returns a typed failure
        return {
            "ok": False,
            "error": str(exc),
            "failure_class": classify_failure(exc),
            "task": task,
        }
    usage = {
        "requests": env.budget.usage.requests,
        "input_tokens": env.budget.usage.input_tokens,
        "output_tokens": env.budget.usage.output_tokens,
        "total": env.budget.usage.input_tokens + env.budget.usage.output_tokens,
    }
    return {
        "ok": True,
        "result": value,
        "task": task,
        "usage": usage,
        "max_depth": cfg.max_depth,
    }


__all__ = [
    "ROOT_INSTRUCTIONS",
    "RLMConfig",
    "run_rlm",
]
