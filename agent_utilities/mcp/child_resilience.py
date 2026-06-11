"""Per-child resilience runtime for the MCP multiplexer.

CONCEPT:ECO-4.34 — Fleet-Scale MCP Multiplexer Hardening.

The multiplexer aggregates ~50 child MCP servers behind one endpoint. Before
this module, every child was a single shared ``ClientSession`` with no
concurrency control: one slow or wedged child head-of-line blocked every
caller, and a crashed child hard-failed all of its tools until the whole
multiplexer was restarted.

:class:`ChildRuntime` wraps each child with the per-server hardening layer:

* **Bounded concurrency** — an ``asyncio.Semaphore`` caps in-flight calls per
  child (``MCP_CHILD_MAX_CONCURRENCY``, per-server ``max_concurrency``
  override in ``mcp_config.json``). Excess calls queue for at most
  ``MCP_CHILD_QUEUE_TIMEOUT`` seconds, then fail with the typed
  :class:`MCPChildBusyError` instead of hanging.

Tenant note: all callers still share each child's credentials (the child
process owns ONE identity). Per-caller credential injection is a deployment
follow-up, not handled here.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

logger = logging.getLogger("mcp_multiplexer.child")


# ---------------------------------------------------------------------------
# Typed errors — callers (and the multiplexer's error envelope) can tell
# *why* a child call failed without parsing prose.
# ---------------------------------------------------------------------------


class MCPChildError(RuntimeError):
    """Base class for typed per-child multiplexer failures."""

    def __init__(self, server: str, message: str) -> None:
        self.server = server
        super().__init__(message)


class MCPChildBusyError(MCPChildError):
    """The child's concurrency slots stayed full past the queue timeout."""


def _cfg_value(cfg: dict[str, Any], key: str, fallback: Any) -> Any:
    """Per-server config override with a global-config fallback."""
    value = cfg.get(key)
    return fallback if value is None else value


class ChildRuntime:
    """Hardened call path for ONE child MCP server.

    Owns the child's live session(s) plus the per-server semaphore. The
    multiplexer routes every proxied tool call through :meth:`call_tool`.
    """

    def __init__(
        self,
        name: str,
        cfg: dict[str, Any] | None = None,
        *,
        max_concurrency: int | None = None,
        queue_timeout: float | None = None,
    ) -> None:
        from agent_utilities.core.config import config

        self.name = name
        self.cfg = dict(cfg or {})

        self.max_concurrency = int(
            max_concurrency
            if max_concurrency is not None
            else _cfg_value(
                self.cfg, "max_concurrency", config.mcp_child_max_concurrency
            )
        )
        self.queue_timeout = float(
            queue_timeout
            if queue_timeout is not None
            else _cfg_value(self.cfg, "queue_timeout", config.mcp_child_queue_timeout)
        )

        self._semaphore: asyncio.Semaphore | None = (
            asyncio.Semaphore(self.max_concurrency) if self.max_concurrency > 0 else None
        )
        self._sessions: list[Any] = []
        self._rr_index = 0
        self._in_flight = 0
        self._queued = 0

    # ------------------------------------------------------------------
    # Session binding
    # ------------------------------------------------------------------

    def adopt_sessions(self, sessions: list[Any]) -> None:
        """Bind already-connected client session(s) to this runtime."""
        self._sessions = list(sessions)

    @property
    def primary_session(self) -> Any | None:
        return self._sessions[0] if self._sessions else None

    @property
    def in_flight(self) -> int:
        return self._in_flight

    @property
    def queued(self) -> int:
        return self._queued

    def _pick_session(self) -> Any:
        if not self._sessions:
            raise MCPChildError(
                self.name, f"Child server '{self.name}' has no active session"
            )
        session = self._sessions[self._rr_index % len(self._sessions)]
        self._rr_index += 1
        return session

    # ------------------------------------------------------------------
    # Bounded-concurrency slot management
    # ------------------------------------------------------------------

    async def _acquire_slot(self) -> None:
        """Take a concurrency slot, queueing at most ``queue_timeout`` seconds."""
        if self._semaphore is None:
            return
        if self._semaphore.locked():
            self._queued += 1
            try:
                await asyncio.wait_for(
                    self._semaphore.acquire(), timeout=self.queue_timeout
                )
            except TimeoutError:
                raise MCPChildBusyError(
                    self.name,
                    f"Child server '{self.name}' is at its concurrency limit "
                    f"({self.max_concurrency} in-flight calls); call queued "
                    f"longer than {self.queue_timeout}s. Retry later or raise "
                    f"'max_concurrency' for this server in mcp_config.json.",
                ) from None
            finally:
                self._queued -= 1
        else:
            await self._semaphore.acquire()

    def _release_slot(self) -> None:
        if self._semaphore is not None:
            self._semaphore.release()

    # ------------------------------------------------------------------
    # Call path
    # ------------------------------------------------------------------

    async def call_tool(self, original_name: str, arguments: dict[str, Any]) -> Any:
        """Forward one tool call to the child under the per-server limits."""
        await self._acquire_slot()
        self._in_flight += 1
        try:
            session = self._pick_session()
            return await session.call_tool(original_name, arguments)
        finally:
            self._in_flight -= 1
            self._release_slot()

    # ------------------------------------------------------------------
    # Health surface
    # ------------------------------------------------------------------

    def status(self) -> dict[str, Any]:
        """Machine-readable per-child health snapshot."""
        return {
            "server": self.name,
            "sessions": len(self._sessions),
            "max_concurrency": self.max_concurrency,
            "in_flight": self._in_flight,
            "queued": self._queued,
        }


__all__ = [
    "ChildRuntime",
    "MCPChildBusyError",
    "MCPChildError",
]
