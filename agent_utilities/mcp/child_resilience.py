"""Per-child resilience runtime for the MCP multiplexer.

CONCEPT:AU-ECO.mcp.profile-differences-from-client — Fleet-Scale MCP Multiplexer Hardening.

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
* **Session pools** — remote children may hold N round-robin connections
  (``MCP_CHILD_POOL_SIZE`` / per-server ``pool_size``); stdio children are
  single-pipe and keep exactly one session.
* **Cancellation-safe dispatch** — the child-side call runs in its own
  shielded task; a caller timeout/cancel detaches cleanly without corrupting
  the shared session's request/response bookkeeping.
* **Restart-on-crash** — each connection generation is owned by a supervisor
  task; transport failures tear the generation down and reconnect with
  exponential backoff (cap + jitter). More than ``MCP_CHILD_MAX_RESTARTS``
  restarts inside ``MCP_CHILD_RESTART_WINDOW`` parks the child as ``failed``.
  Calls to a restarting child wait briefly for recovery, then fail with the
  typed :class:`MCPChildUnavailableError` naming the child and its state.
* **Circuit breaker** — consecutive transport failures/timeouts open a
  per-child breaker (``MCP_CHILD_BREAKER_THRESHOLD`` /
  ``MCP_CHILD_BREAKER_COOLDOWN``), short-circuiting calls with the typed
  :class:`MCPChildCircuitOpenError` until a half-open probe succeeds. The
  state machine is the shared OS-5.23 engine-client breaker
  (``knowledge_graph.core.engine_breaker.CircuitBreaker``), subclassed for
  child wording and the per-child state gauge.

Crash detection is call-path driven: a stdio process exit or HTTP transport
failure surfaces as a stream/connection error on the next forwarded call,
which triggers the restart cycle (no idle polling of ~50 children).

Metrics land on the OS-5.23 registry (``observability.gateway_metrics``) and
degrade to no-ops when ``prometheus_client`` (the optional ``metrics`` extra)
is absent — the multiplexer runs standalone, so this stays import-light:
``agent_utilities_mcp_child_calls_total{server,outcome}``,
``..._mcp_child_breaker_state{server}``, ``..._mcp_child_restarts_total{server}``,
``..._mcp_child_queue_depth{server}``.

Tenant note: all callers still share each child's credentials (the child
process owns ONE identity). Per-caller credential injection is a deployment
follow-up, not handled here.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import random
import time
from collections import deque
from collections.abc import Awaitable, Callable
from typing import Any

import anyio

McpError: Any = ()
try:  # MCP protocol error (e.g. a terminated streamable-http session)
    from mcp.shared.exceptions import McpError
except ImportError:  # pragma: no cover - mcp always present for the multiplexer
    pass

from agent_utilities.knowledge_graph.core.engine_breaker import CircuitBreaker
from agent_utilities.observability.gateway_metrics import (
    MCP_CHILD_BREAKER_STATE,
    MCP_CHILD_CALLS,
    MCP_CHILD_QUEUE_DEPTH,
    MCP_CHILD_RESTARTS,
)

logger = logging.getLogger("mcp_multiplexer.child")

# Transport-level failures that indicate a dead child (stdio process exit,
# closed pipe, HTTP connect/reset). Application-level tool errors are NOT in
# this set — a child that answers with an error is alive.
TRANSPORT_EXCEPTIONS: tuple[type[BaseException], ...] = (
    OSError,
    EOFError,
    anyio.BrokenResourceError,
    anyio.ClosedResourceError,
)


def is_session_dead(exc: BaseException) -> bool:
    """Whether ``exc`` means the child's streamable-http session is gone — e.g.
    the backend redeployed and no longer recognizes the session id.

    The MCP client raises ``McpError(code=32600, "Session terminated")`` for a
    server-terminated session; we also match session-not-found wording. Such an
    error is a *transport* failure (the connection must be rebuilt), not an
    application error — unlike a tool that simply answered with an error."""
    if not McpError or not isinstance(exc, McpError):
        return False
    err = getattr(exc, "error", None)
    code = getattr(err, "code", None)
    message = str(getattr(err, "message", "") or exc).lower()
    return (
        code == 32600
        or "session terminated" in message
        or ("session" in message and "not found" in message)
    )


def _exc_leaves(exc: BaseException) -> list[BaseException]:
    """Flatten a (possibly nested) ``BaseExceptionGroup`` to its leaf exceptions.

    A child that crashes mid-call surfaces through anyio as a
    ``BaseExceptionGroup`` whose own ``str()`` is empty/opaque, hiding the real
    transport error inside — so callers must inspect the leaves.
    """
    if isinstance(exc, BaseExceptionGroup):
        out: list[BaseException] = []
        for sub in exc.exceptions:
            out.extend(_exc_leaves(sub))
        return out
    return [exc]


def is_transient_child_death(exc: BaseException) -> bool:
    """Whether ``exc`` is a *retryable* child death — the child process crashed/
    exited mid-call, the pipe closed, or the session was terminated (redeploy).

    Unlike an application tool error (a live child answering with an error), this
    means the connection must be rebuilt and the call re-issued on a fresh
    generation. Covers ``is_session_dead`` plus any ``TRANSPORT_EXCEPTIONS`` leaf
    (including ones wrapped in a ``BaseExceptionGroup`` from the anyio task group)
    — exactly the mid-call-crash case that previously surfaced as an empty
    ``Error executing tool:`` instead of self-healing.
    """
    return any(
        is_session_dead(e) or isinstance(e, TRANSPORT_EXCEPTIONS)
        for e in _exc_leaves(exc)
    )


# Reconnect backoff defaults (overridable per-runtime for tests): exponential
# growth from BASE up to CAP, multiplied by uniform jitter so a fleet of
# children never thunders back in lockstep.
RESTART_BACKOFF_BASE = 0.5
RESTART_BACKOFF_CAP = 30.0
_JITTER_RANGE = (0.5, 1.5)

# How long a call will wait for a restarting child to come back before it
# fails fast (bounded further by the child's queue timeout).
_READY_WAIT_CEILING = 5.0

# After a child died mid-call, how long the single in-call retry waits for the
# respawned generation to come back (a cold child rebuilds its engine, which
# takes longer than the fast-fail ceiling). Bounded so a caller is never stuck.
_RECOVERY_WAIT = 45.0

#: ``connect`` contract: open all transports/sessions on the given stack and
#: return ``(sessions, tools)``. The stack is owned (entered AND exited) by
#: the supervisor task, which keeps anyio cancel scopes single-task.
ConnectFn = Callable[
    [contextlib.AsyncExitStack], Awaitable[tuple[list[Any], list[Any]]]
]


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


class MCPChildCallTimeoutError(MCPChildError):
    """The child accepted the call but did not answer within the call timeout.

    The abandoned call is detached: it keeps its concurrency slot until the
    child actually finishes (or the session dies), so a wedged child applies
    backpressure instead of corrupting the shared session."""


class MCPChildUnavailableError(MCPChildError):
    """The child is not serving calls (restarting after a crash, or failed)."""

    def __init__(self, server: str, state: str, message: str) -> None:
        self.state = state
        super().__init__(server, message)


class MCPChildCircuitOpenError(MCPChildError):
    """The child's circuit breaker is open — failing fast, not forwarding."""


class _BreakerOpenSignal(ConnectionError):
    """Internal: what the shared breaker raises before it is re-typed with
    the child's name as :class:`MCPChildCircuitOpenError`."""


class ChildCircuitBreaker(CircuitBreaker):
    """The OS-5.23 engine-client breaker state machine, per multiplexer child.

    Same closed/open/half-open semantics and thread-safety; only the wording
    and the exported gauge differ (per-child ``server`` label instead of the
    engine ``endpoint`` label)."""

    error_cls = _BreakerOpenSignal
    subject = "MCP child server"

    def _export_state(self) -> None:
        MCP_CHILD_BREAKER_STATE.labels(server=self.endpoint).set(self._state_value())


def _cfg_value(cfg: dict[str, Any], key: str, fallback: Any) -> Any:
    """Per-server config override with a global-config fallback."""
    value = cfg.get(key)
    return fallback if value is None else value


class ChildRuntime:
    """Hardened call path + lifecycle supervisor for ONE child MCP server.

    Owns the child's live session pool, the per-server semaphore, and (when
    constructed with a ``connect`` factory) the supervisor task that restarts
    crashed connections. The multiplexer routes every proxied tool call
    through :meth:`call_tool`.
    """

    def __init__(
        self,
        name: str,
        cfg: dict[str, Any] | None = None,
        *,
        connect: ConnectFn | None = None,
        max_concurrency: int | None = None,
        queue_timeout: float | None = None,
        restart_backoff_base: float = RESTART_BACKOFF_BASE,
        restart_backoff_cap: float = RESTART_BACKOFF_CAP,
        session_max_age: float | None = None,
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
        # Per-call ceiling: reuses the server entry's existing ``timeout`` key
        # (historically the connect/handshake budget) unless a dedicated
        # ``call_timeout`` is given. <=0 disables the ceiling.
        self.call_timeout = float(
            _cfg_value(self.cfg, "call_timeout", self.cfg.get("timeout", 300.0))
        )
        self.connect_timeout = float(self.cfg.get("timeout", 300.0))
        self.max_restarts = int(
            _cfg_value(self.cfg, "max_restarts", config.mcp_child_max_restarts)
        )
        self.restart_window = float(
            _cfg_value(self.cfg, "restart_window", config.mcp_child_restart_window)
        )
        self.restart_backoff_base = float(restart_backoff_base)
        self.restart_backoff_cap = float(restart_backoff_cap)

        # Per-child circuit breaker (shared OS-5.23 state machine; thread-safe
        # by construction, trivially so on the multiplexer's single loop).
        self.breaker = ChildCircuitBreaker(
            name,
            threshold=int(
                _cfg_value(
                    self.cfg, "breaker_threshold", config.mcp_child_breaker_threshold
                )
            ),
            cooldown=float(
                _cfg_value(
                    self.cfg, "breaker_cooldown", config.mcp_child_breaker_cooldown
                )
            ),
        )

        self._semaphore: asyncio.Semaphore | None = (
            asyncio.Semaphore(self.max_concurrency)
            if self.max_concurrency > 0
            else None
        )
        self._sessions: list[Any] = []
        self._rr_index = 0
        self._in_flight = 0
        self._queued = 0
        # Calls whose caller timed out (outcome already recorded as
        # ``timeout``); their completion must not double-count the call.
        self._abandoned: set[asyncio.Future] = set()

        # Lifecycle (restart-on-crash supervisor)
        self._connect = connect
        self.state = "starting"
        self.restart_count = 0
        self._restart_times: deque[float] = deque()
        self._ready = asyncio.Event()
        self._stop_generation: asyncio.Event | None = None
        self._supervisor: asyncio.Task | None = None
        self._closed = False
        # Recycle this generation before its bearer expires (None = never): a
        # service-authenticated child session is authed once at connect and its
        # result stream then stays open, so it must reconnect before the token
        # TTL elapses or in-flight calls wedge (CONCEPT:AU-OS.identity.so-jwt-protected-children).
        self.session_max_age = session_max_age
        self._generation_started_at = 0.0
        # Set when a generation is torn down for a PLANNED token recycle (not a
        # crash): the supervisor then reconnects immediately without counting it
        # toward the restart budget or applying crash backoff.
        self._recycle_requested = False

    # ------------------------------------------------------------------
    # Session binding
    # ------------------------------------------------------------------

    def adopt_sessions(self, sessions: list[Any]) -> None:
        """Bind already-connected client session(s) to this runtime.

        Used when the connection lifecycle is owned elsewhere (no supervisor:
        no auto-restart, matching the pre-hardening behaviour)."""
        self._sessions = list(sessions)
        self.state = "up"
        self._ready.set()

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
            raise self._unavailable(
                self.state,
                f"Child server '{self.name}' has no active session "
                f"(state={self.state})",
            )
        session = self._sessions[self._rr_index % len(self._sessions)]
        self._rr_index += 1
        return session

    # ------------------------------------------------------------------
    # Lifecycle — supervisor task owns each connection generation
    # ------------------------------------------------------------------

    async def start(self) -> list[Any]:
        """Start the supervisor and wait for the first generation's tools.

        Raises whatever the first connect attempt raised (incl.
        ``TimeoutError`` after ``connect_timeout``); a boot failure does NOT
        enter the restart cycle — the child is simply not loaded, exactly as
        before the hardening layer."""
        if self._connect is None:
            raise RuntimeError(
                f"ChildRuntime('{self.name}') has no connect factory; "
                "bind sessions via adopt_sessions() instead."
            )
        first: asyncio.Future = asyncio.get_running_loop().create_future()
        self._supervisor = asyncio.create_task(
            self._supervise(first), name=f"mcp-child-supervisor-{self.name}"
        )
        try:
            return await first
        except BaseException:
            await self.aclose()
            raise

    async def _supervise(self, first: asyncio.Future | None) -> None:
        """Run connection generations until closed or parked as failed.

        Each generation's transports/sessions live on an ``AsyncExitStack``
        that is entered AND exited inside this task, so anyio cancel scopes
        (stdio_client, streamablehttp_client) never cross task boundaries."""
        assert self._connect is not None
        backoff = self.restart_backoff_base
        while not self._closed:
            try:
                async with contextlib.AsyncExitStack() as stack:
                    sessions, tools = await asyncio.wait_for(
                        self._connect(stack), timeout=self.connect_timeout
                    )
                    self._sessions = list(sessions)
                    self._generation_started_at = time.monotonic()
                    self._set_state("up")
                    backoff = self.restart_backoff_base
                    # A fresh generation proves the child reachable again, so
                    # an open breaker closes instead of short-circuiting
                    # calls against the recovered child.
                    self.breaker.record_success()
                    self._stop_generation = asyncio.Event()
                    self._ready.set()
                    if first is not None:
                        first.set_result(tools)
                        first = None
                    else:
                        logger.info(
                            "Child server '%s' recovered after restart #%d",
                            self.name,
                            self.restart_count,
                        )
                    await self._stop_generation.wait()
            except asyncio.CancelledError:
                raise
            except BaseException as e:
                if first is not None:
                    # First connect failed: report to start() and stop — the
                    # multiplexer skips the child (no tools to serve anyway).
                    self._set_state("failed")
                    first.set_exception(e)
                    return
                logger.warning(
                    "Reconnect to child server '%s' failed: %s: %s",
                    self.name,
                    type(e).__name__,
                    e,
                )
            finally:
                self._ready.clear()
                self._sessions = []
            if self._closed:
                return

            # Planned token recycle: reconnect immediately, NOT a crash — it does
            # not count toward the restart budget and skips crash backoff.
            if self._recycle_requested:
                self._recycle_requested = False
                self._set_state("restarting")
                continue

            # Restart bookkeeping: sliding window of recent restarts.
            now = time.monotonic()
            self._restart_times.append(now)
            while self._restart_times and self._restart_times[0] < (
                now - self.restart_window
            ):
                self._restart_times.popleft()
            if self.max_restarts <= 0 or len(self._restart_times) > self.max_restarts:
                self._set_state("failed")
                logger.error(
                    "Child server '%s' exceeded %d restarts in %.0fs — "
                    "marking failed (calls now fail fast). Restart the "
                    "multiplexer or fix the child to recover.",
                    self.name,
                    self.max_restarts,
                    self.restart_window,
                )
                return

            self.restart_count += 1
            self._on_restart()
            self._set_state("restarting")
            delay = min(backoff, self.restart_backoff_cap) * random.uniform(  # nosec B311 - backoff jitter, not crypto
                *_JITTER_RANGE
            )
            backoff = min(backoff * 2, self.restart_backoff_cap)
            logger.warning(
                "Restarting child server '%s' in %.2fs (restart #%d)",
                self.name,
                delay,
                self.restart_count,
            )
            await asyncio.sleep(delay)

    def _set_state(self, state: str) -> None:
        if state != self.state:
            logger.info("Child server '%s': %s -> %s", self.name, self.state, state)
        self.state = state

    def _record(self, outcome: str) -> None:
        MCP_CHILD_CALLS.labels(server=self.name, outcome=outcome).inc()

    def _on_restart(self) -> None:
        """Restart side-effects: counted on the OS-5.23 metrics registry."""
        MCP_CHILD_RESTARTS.labels(server=self.name).inc()

    def request_restart(self, reason: str = "") -> None:
        """Tear down the current generation and reconnect (supervised only)."""
        if self._closed or self._supervisor is None or self.state != "up":
            return
        logger.warning(
            "Child server '%s' transport failure%s — recycling connection",
            self.name,
            f" ({reason})" if reason else "",
        )
        self._set_state("restarting")
        self._ready.clear()
        if self._stop_generation is not None:
            self._stop_generation.set()

    def request_recycle(self) -> None:
        """Tear down + reconnect for a PLANNED token refresh (supervised only).

        Unlike :meth:`request_restart`, this is not a crash: the supervisor
        reconnects immediately without counting it toward the restart budget or
        applying backoff, so a child can recycle every token window indefinitely
        without being parked as ``failed``."""
        if self._closed or self._supervisor is None or self.state != "up":
            return
        logger.info(
            "Child server '%s' recycling session before token expiry", self.name
        )
        self._recycle_requested = True
        self._set_state("restarting")
        self._ready.clear()
        if self._stop_generation is not None:
            self._stop_generation.set()

    def _unavailable(self, state: str, message: str) -> MCPChildUnavailableError:
        self._record("unavailable")
        return MCPChildUnavailableError(self.name, state, message)

    async def _await_ready(self) -> None:
        """Gate calls on child availability with a brief recovery wait."""
        if self.state == "failed":
            raise self._unavailable(
                "failed",
                f"Child server '{self.name}' is marked FAILED after "
                f"{self.restart_count} restarts; not accepting calls.",
            )
        if self._ready.is_set():
            return
        wait = min(self.queue_timeout, _READY_WAIT_CEILING)
        try:
            await asyncio.wait_for(self._ready.wait(), timeout=wait)
        except TimeoutError:
            raise self._unavailable(
                self.state,
                f"Child server '{self.name}' is {self.state} (restart "
                f"#{self.restart_count}); still unavailable after waiting "
                f"{wait}s. Retry shortly.",
            ) from None
        if self.state == "failed":
            raise self._unavailable(
                "failed",
                f"Child server '{self.name}' is marked FAILED after "
                f"{self.restart_count} restarts; not accepting calls.",
            )

    async def aclose(self) -> None:
        """Shut the runtime down: stop the generation and the supervisor."""
        self._closed = True
        self._ready.clear()
        if self._stop_generation is not None:
            self._stop_generation.set()
        if self._supervisor is not None:
            self._supervisor.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._supervisor
            self._supervisor = None
        self._sessions = []
        self._set_state("closed")

    # ------------------------------------------------------------------
    # Bounded-concurrency slot management
    # ------------------------------------------------------------------

    async def _acquire_slot(self) -> None:
        """Take a concurrency slot, queueing at most ``queue_timeout`` seconds."""
        if self._semaphore is None:
            return
        if self._semaphore.locked():
            self._queued += 1
            MCP_CHILD_QUEUE_DEPTH.labels(server=self.name).set(self._queued)
            try:
                await asyncio.wait_for(
                    self._semaphore.acquire(), timeout=self.queue_timeout
                )
            except TimeoutError:
                self._record("busy")
                raise MCPChildBusyError(
                    self.name,
                    f"Child server '{self.name}' is at its concurrency limit "
                    f"({self.max_concurrency} in-flight calls); call queued "
                    f"longer than {self.queue_timeout}s. Retry later or raise "
                    f"'max_concurrency' for this server in mcp_config.json.",
                ) from None
            finally:
                self._queued -= 1
                MCP_CHILD_QUEUE_DEPTH.labels(server=self.name).set(self._queued)
        else:
            await self._semaphore.acquire()

    def _release_slot(self) -> None:
        if self._semaphore is not None:
            self._semaphore.release()

    # ------------------------------------------------------------------
    # Call path
    # ------------------------------------------------------------------

    def _finish_call(self, task: asyncio.Task) -> None:
        """Slot bookkeeping when the underlying child call actually completes.

        Runs even when the awaiting caller timed out or was cancelled: the
        slot belongs to the *child-side* call, so it is only returned once the
        child has truly finished — a wedged child exerts backpressure rather
        than letting abandoned calls stack up invisibly."""
        self._in_flight -= 1
        self._release_slot()
        abandoned = task in self._abandoned
        self._abandoned.discard(task)
        if task.cancelled():
            return
        exc = task.exception()  # consume so abandoned failures don't warn
        if exc is None:
            self.breaker.record_success()
            if not abandoned:  # abandoned calls were already counted (timeout)
                self._record("ok")
            return
        if isinstance(exc, TRANSPORT_EXCEPTIONS) or is_session_dead(exc):
            # Dead pipe / closed stream / terminated session: the child is gone,
            # not erroring. Rebuild the connection (a redeployed backend drops
            # the session, which would otherwise wedge every later call).
            self.breaker.record_failure()
            if not abandoned:
                self._record("transport_error")
            reason = (
                "session_terminated" if is_session_dead(exc) else type(exc).__name__
            )
            self.request_restart(reason=reason)
        else:
            # Application-level tool error: the child answered, so the
            # breaker stays closed (mirrors the OS-5.23 engine guard).
            if not abandoned:
                self._record("error")
            logger.debug(
                "Child '%s' call finished with %s after the caller detached",
                self.name,
                type(exc).__name__,
            )

    async def _recycle_if_stale(self) -> None:
        """Reconnect before the current generation's bearer token expires.

        A no-op unless this child has a ``session_max_age`` (set only for
        service-authenticated remote children) and a live, supervised generation
        that has outlived it. Lazy (checked on the call path, not a background
        timer) so idle children cost nothing and only an actively-used child
        reconnects, exactly once per token window, just before a call."""
        if (
            self.session_max_age is None
            or self._supervisor is None
            or self.state != "up"
            or self._generation_started_at <= 0.0
        ):
            return
        if (time.monotonic() - self._generation_started_at) < self.session_max_age:
            return
        self.request_recycle()
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(
                self._ready.wait(), timeout=min(self.connect_timeout, _RECOVERY_WAIT)
            )

    async def call_tool(self, original_name: str, arguments: dict[str, Any]) -> Any:
        """Forward one tool call to the child under the per-server limits.

        Cancellation-safe: the child-side call runs in its own task and is
        shielded from the caller. A caller timeout/cancel detaches cleanly —
        the shared session keeps its request/response bookkeeping intact and
        the concurrency slot is released only when the child finishes.

        Self-healing: if the child's session was terminated (e.g. the backend
        redeployed), the failed attempt triggers a reconnect and the call is
        retried ONCE on the fresh generation — so a redeploy is invisible to
        the caller instead of stranding it on "Session terminated"."""
        # Recycle a service-authenticated session before its bearer expires, so
        # the call never lands on a session whose auth context has died (which
        # would wedge it until call_timeout instead of erroring).
        await self._recycle_if_stale()
        for attempt in range(2):
            try:
                return await self._call_once(original_name, arguments)
            except BaseException as exc:
                # Retry ONCE on a transient child death — a terminated session
                # (redeploy) OR the child process crashing mid-call (the
                # post-restart warm-up race that otherwise surfaced as an empty
                # "Error executing tool:"). _finish_call already asked the
                # supervisor to reconnect; make it deterministic and then wait
                # for the respawned generation (which rebuilds its engine, longer
                # than the fast-fail ceiling) before re-issuing on the fresh
                # session.
                if attempt == 0 and is_transient_child_death(exc):
                    self.request_restart(reason="transient_child_death")
                    with contextlib.suppress(TimeoutError):
                        await asyncio.wait_for(
                            self._ready.wait(),
                            timeout=min(self.connect_timeout, _RECOVERY_WAIT),
                        )
                    continue
                raise
        raise AssertionError("unreachable")  # pragma: no cover

    async def _call_once(self, original_name: str, arguments: dict[str, Any]) -> Any:
        """One forwarding attempt (the body the retry loop wraps)."""
        try:
            self.breaker.before_call()
        except _BreakerOpenSignal as signal:
            self._record("short_circuited")
            raise MCPChildCircuitOpenError(self.name, str(signal)) from None
        # before_call() claims the single half-open probe slot when the
        # breaker is recovering; if this call dies before reaching the child
        # (busy/unavailable), the slot must be returned — as a failed probe,
        # since the child was not demonstrably reachable. In the closed state
        # those same pre-call rejections never touch the breaker.
        probe_claimed = self.breaker.state == "half_open"
        try:
            await self._await_ready()
            await self._acquire_slot()
        except BaseException:
            if probe_claimed:
                self.breaker.record_failure()
            raise
        try:
            session = self._pick_session()
        except BaseException:
            self._release_slot()
            if probe_claimed:
                self.breaker.record_failure()
            raise
        self._in_flight += 1
        inner = asyncio.ensure_future(session.call_tool(original_name, arguments))
        inner.add_done_callback(self._finish_call)
        try:
            if self.call_timeout > 0:
                return await asyncio.wait_for(
                    asyncio.shield(inner), timeout=self.call_timeout
                )
            return await asyncio.shield(inner)
        except TimeoutError:
            # Consecutive timeouts count toward opening the circuit; if the
            # detached call eventually succeeds, its completion closes it.
            self.breaker.record_failure()
            self._record("timeout")
            self._abandoned.add(inner)
            raise MCPChildCallTimeoutError(
                self.name,
                f"Tool '{original_name}' on child server '{self.name}' did "
                f"not answer within {self.call_timeout}s; the call was "
                f"detached and its slot is held until the child finishes.",
            ) from None

    # ------------------------------------------------------------------
    # Health surface
    # ------------------------------------------------------------------

    def status(self) -> dict[str, Any]:
        """Machine-readable per-child health snapshot."""
        return {
            "server": self.name,
            "state": self.state,
            "restart_count": self.restart_count,
            "breaker": self.breaker.state,
            "sessions": len(self._sessions),
            "max_concurrency": self.max_concurrency,
            "in_flight": self._in_flight,
            "queued": self._queued,
        }


__all__ = [
    "ChildCircuitBreaker",
    "ChildRuntime",
    "MCPChildBusyError",
    "MCPChildCallTimeoutError",
    "MCPChildCircuitOpenError",
    "MCPChildError",
    "MCPChildUnavailableError",
    "TRANSPORT_EXCEPTIONS",
    "is_session_dead",
]
