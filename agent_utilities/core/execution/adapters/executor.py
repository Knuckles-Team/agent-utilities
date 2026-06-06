"""CONCEPT:ORCH-1.33 — Spawn an adapter backend and normalise its stream.

Bridges an :class:`AdapterDefinition` to a real subprocess: builds argv, delivers the prompt (argv or
stdin per ``prompt_delivery``), runs it, and normalises stdout via the stream-format handler into
canonical :class:`ExecEvent` objects. Pure stdlib ``asyncio.subprocess`` (no new deps).

This is the seam EPIC 2 (ORCH-1.35) extends for keep-stdin-open mid-turn tool-result injection.
"""

from __future__ import annotations

import asyncio
import logging
import shutil

from ..stream_handlers import collect_text, get_stream_handler
from .base import AdapterDefinition, ExecEvent, ExecEventType, PromptDelivery

logger = logging.getLogger(__name__)


class AdapterExecutionError(RuntimeError):
    """Raised when an adapter backend cannot be spawned or exits non-zero with no output."""


async def run_adapter(
    definition: AdapterDefinition,
    prompt: str,
    *,
    model: str | None = None,
    cwd: str | None = None,
    env: dict[str, str] | None = None,
    timeout: float = 120.0,
) -> list[ExecEvent]:
    """Spawn ``definition``'s backend for ``prompt`` and return canonical events.

    Raises :class:`AdapterExecutionError` if the binary is missing or the process fails to start.
    """
    path = shutil.which(definition.bin)
    if not path:
        raise AdapterExecutionError(
            f"adapter {definition.id!r}: {definition.bin!r} not on PATH"
        )

    eff_model = definition.resolve_model(model, env)
    deliver_via_args = definition.prompt_delivery is PromptDelivery.ARGS
    arg_prompt = prompt if deliver_via_args else ""
    argv = [path, *definition.build_args(eff_model, arg_prompt)]

    proc_env = {**definition.env, **(env or {})} or None
    stdin_target = None if deliver_via_args else asyncio.subprocess.PIPE
    try:
        proc = await asyncio.create_subprocess_exec(
            *argv,
            stdin=stdin_target,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd,
            env=({**__import__("os").environ, **proc_env} if proc_env else None),
        )
    except OSError as exc:  # pragma: no cover - exercised via missing-bin path
        raise AdapterExecutionError(
            f"adapter {definition.id!r}: spawn failed: {exc}"
        ) from exc

    stdin_payload: bytes | None = None
    if not deliver_via_args:
        if definition.prompt_delivery is PromptDelivery.STDIN_JSONL:
            import json

            stdin_payload = (
                json.dumps({"type": "user", "text": prompt}) + "\n"
            ).encode()
        else:  # STDIN_TEXT
            stdin_payload = prompt.encode()

    try:
        stdout_b, stderr_b = await asyncio.wait_for(
            proc.communicate(stdin_payload), timeout=timeout
        )
    except TimeoutError as exc:
        proc.kill()
        raise AdapterExecutionError(
            f"adapter {definition.id!r}: timed out after {timeout}s"
        ) from exc

    stdout = stdout_b.decode(errors="replace")
    handler = get_stream_handler(definition.stream_format)
    events = list(handler(stdout.splitlines(keepends=True)))
    if proc.returncode and not any(
        e.type is ExecEventType.TEXT_DELTA and e.text for e in events
    ):
        err = stderr_b.decode(errors="replace").strip()
        events.append(
            ExecEvent(ExecEventType.ERROR, text=err or f"exit {proc.returncode}")
        )
    return events


async def run_adapter_text(
    definition: AdapterDefinition, prompt: str, **kw: object
) -> str:
    """Convenience: run an adapter and fold the event stream into final text."""
    events = await run_adapter(definition, prompt, **kw)  # type: ignore[arg-type]
    return collect_text(events)
