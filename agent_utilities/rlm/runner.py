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
from dataclasses import dataclass
from typing import Any

from pydantic_ai import Agent, RunContext
from pydantic_ai.usage import RunUsage, UsageLimits

from agent_utilities import host_ports
from agent_utilities.graph.executor import resolve_plan
from agent_utilities.graph.state import RunRequest

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


@dataclass
class _Env:
    text: str
    depth: int
    config: RLMConfig
    usage: RunUsage


def _model(value: Any) -> Any:
    return host_ports.resolve_model(value) if isinstance(value, str) else value


def _agent(config: RLMConfig, depth: int, output_type: Any) -> Agent[_Env, Any]:
    model = _model(config.model if depth == 0 else (config.sub_model or config.model))
    agent: Agent[_Env, Any] = Agent(
        model, output_type=output_type, deps_type=_Env, instructions=ROOT_INSTRUCTIONS
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
        child = _Env(
            text=ctx.deps.text[start : end or None],
            depth=ctx.deps.depth + 1,
            config=ctx.deps.config,
            usage=ctx.deps.usage,
        )
        out = await _run_level(question, child, str)
        return str(out)

    return agent


def _prompt(task: str, env: _Env) -> str:
    preview = env.text[: env.config.preview_chars]
    return f"{task}\n\n[input: {len(env.text)} chars; preview]\n{preview}"


async def _run_level(task: str, env: _Env, output_type: Any) -> Any:
    limits = UsageLimits(
        request_limit=env.config.request_limit,
        total_tokens_limit=env.config.total_tokens_limit,
    )
    result = await _agent(env.config, env.depth, output_type).run(
        _prompt(task, env), deps=env, usage_limits=limits, usage=env.usage
    )
    return result.output


async def _config_from_host(task: str) -> RLMConfig:
    plan = await resolve_plan(RunRequest(task=task, agent_name="rlm"), {})
    return RLMConfig(
        model=plan.model,
        request_limit=plan.budget.request_limit or 50,
        total_tokens_limit=plan.budget.total_tokens_limit,
        timeout_seconds=plan.budget.timeout_seconds,
    )


async def run_rlm(
    task: str,
    input_text: str = "",
    *,
    config: RLMConfig | None = None,
    output_type: Any = str,
) -> dict[str, Any]:
    """Run one recursive task over ``input_text``.

    Returns ``{"ok": True, "result", "task", "usage", "max_depth"}`` or
    ``{"ok": False, "error", "failure_class", "task"}``. A
    :class:`SandboxFatalError` propagates so the caller fails fast.
    """
    try:
        cfg = config if config is not None and config.model else None
        cfg = cfg or await _config_from_host(task)
        env = _Env(text=input_text, depth=0, config=cfg, usage=RunUsage())
        async with asyncio.timeout(cfg.timeout_seconds):
            value = await _run_level(task, env, output_type)
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
        "requests": env.usage.requests,
        "input_tokens": env.usage.input_tokens,
        "output_tokens": env.usage.output_tokens,
        "total": env.usage.input_tokens + env.usage.output_tokens,
    }
    return {
        "ok": True,
        "result": value,
        "task": task,
        "usage": usage,
        "max_depth": cfg.max_depth,
    }


__all__ = ["ROOT_INSTRUCTIONS", "RLMConfig", "run_rlm"]
