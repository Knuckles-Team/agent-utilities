"""MCP Context Helper Utilities.

Standardized helpers for the FastMCP ``Context`` object. Every MCP server
in the agent-packages ecosystem should import these instead of hand-rolling
ctx interactions. All helpers are safe when *ctx* is ``None`` (backward
compatible with callers that do not inject a context).

CONCEPT:ECO-4.0 — MCP Standardized Interfaces
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


async def ctx_progress(ctx: Any, progress: int, total: int = 100) -> None:
    """Report progress to the MCP client if a context is available.

    Args:
        ctx: FastMCP ``Context`` (may be ``None``).
        progress: Current progress value.
        total: Total steps (default 100 for percentage-style reporting).

    """
    if ctx:
        await ctx.report_progress(progress=progress, total=total)


async def ctx_confirm_destructive(
    ctx: Any,
    action_description: str,
) -> bool:
    """Standard elicitation guard for destructive operations.

    When a ``Context`` is available this asks the human user to confirm
    before proceeding.  If no context is provided (e.g. headless / test
    invocation) the operation is allowed by default.

    Args:
        ctx: FastMCP ``Context`` (may be ``None``).
        action_description: Human-readable description of the action,
            e.g. ``"delete stack 'production'"``.

    Returns:
        ``True`` if the operation should proceed, ``False`` if cancelled.

    """
    if not ctx:
        return True  # No context → allow (headless / test mode)
    try:
        result = await ctx.elicit(
            f"⚠️ Are you sure you want to {action_description}?",
            response_type=bool,
        )
        return result.action == "accept" and bool(result.data)
    except Exception as exc:
        logger.warning("Elicitation failed (%s); allowing operation by default.", exc)
        return True


def ctx_log(
    ctx: Any,
    server_logger: logging.Logger,
    level: str,
    message: str,
) -> None:
    """Dual-log a message to *both* the server-side logger and the MCP client.

    This ensures that diagnostic output is visible in two places:
    • The server process logs (for operators / container logs).
    • The MCP client log console (for the AI agent / human user).

    Args:
        ctx: FastMCP ``Context`` (may be ``None``).
        server_logger: A standard Python :class:`logging.Logger`.
        level: Log level string — ``"debug"``, ``"info"``, ``"warning"``,
            or ``"error"``.
        message: The log message.

    """
    getattr(server_logger, level, server_logger.info)(message)
    if ctx:
        client_fn = getattr(ctx, level, None) or getattr(ctx, "info", None)
        if client_fn:
            try:
                import asyncio
                import inspect

                res = client_fn(message)
                if inspect.iscoroutine(res):
                    try:
                        loop = asyncio.get_running_loop()
                        loop.create_task(res)
                    except RuntimeError:
                        pass
            except Exception:
                pass  # nosec: B110 - Never let client-side logging break tool execution


async def ctx_set_state(
    ctx: Any,
    project: str,
    key: str,
    value: Any,
) -> None:
    """Store a value in the MCP session state with a standardized key.

    Keys are namespaced as ``"{project}_{key}"`` to prevent collisions
    across different MCP servers sharing the same session.

    Args:
        ctx: FastMCP ``Context`` (may be ``None``).
        project: Project namespace (e.g. ``"portainer"``, ``"gitlab"``).
        key: State key (e.g. ``"auth_token"``, ``"active_context"``).
        value: The value to store.

    """
    if ctx and hasattr(ctx, "session"):
        try:
            namespaced = f"{project}_{key}"
            await ctx.session.set_state(namespaced, value)
        except Exception as exc:
            logger.debug("ctx_set_state failed for %s_%s: %s", project, key, exc)


async def ctx_get_state(
    ctx: Any,
    project: str,
    key: str,
    default: Any = None,
) -> Any:
    """Retrieve a value from the MCP session state.

    Args:
        ctx: FastMCP ``Context`` (may be ``None``).
        project: Project namespace.
        key: State key.
        default: Fallback if the key is missing or ctx is unavailable.

    Returns:
        The stored value, or *default*.

    """
    if ctx and hasattr(ctx, "session"):
        try:
            namespaced = f"{project}_{key}"
            val = await ctx.session.get_state(namespaced)
            return val if val is not None else default
        except Exception:
            pass  # nosec: B110
    return default


async def ctx_sample(
    ctx: Any,
    prompt: str,
    system_prompt: str | None = None,
) -> str | None:
    """Ask the client LLM to generate a response (sampling).

    This is an optional capability — it only works when the connected MCP
    client supports the ``sampling`` feature.  Returns ``None`` silently
    when sampling is unavailable.

    Args:
        ctx: FastMCP ``Context`` (may be ``None``).
        prompt: The user-turn prompt to send to the LLM.
        system_prompt: Optional system prompt to guide the LLM.

    Returns:
        The LLM-generated text, or ``None`` if sampling is unavailable.

    """
    if not ctx:
        return None
    try:
        from mcp.types import SamplingMessage, TextContent

        messages = [
            SamplingMessage(
                role="user",
                content=TextContent(type="text", text=prompt),
            )
        ]
        result = await ctx.sample(
            messages=messages,
            system_prompt=system_prompt,
            max_tokens=2048,
        )
        if result and hasattr(result, "content"):
            return (
                result.content.text
                if hasattr(result.content, "text")
                else str(result.content)
            )
        return None
    except ImportError:
        logger.debug("mcp.types not available — sampling disabled.")
        return None
    except Exception as exc:
        logger.debug("ctx_sample failed: %s", exc)
        return None
