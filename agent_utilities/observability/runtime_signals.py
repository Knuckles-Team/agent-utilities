#!/usr/bin/python
from __future__ import annotations

"""Runtime-reliability signal recording — the EMIT stage of the detect→gap→heal loop.

CONCEPT:AU-OS.observability.runtime-reliability-signal — the missing intake for a class
of RUNTIME failures the self-improvement stack never noticed. DSPy/Langfuse and the
gap-flywheel key off AGENT-RUN QUALITY (reward, failure_analyzer) and LLM spans, so
failures that never become a *run* were invisible: a messaging poller dying on a 409
(never a run), a retrieval blocking the event loop → SIGKILL (a k8s restart, not a run),
an O(N) retrieval perf regression (slow, not wrong → no reward penalty), engine
write-contention. The signals to *detect* these already exist as WARN logs + metrics
(``engine_breaker`` slow-call, ``router`` listener supervisor, ``contextual_model``
retrieval-degrade, ``agent_runner`` run_summary); this module is the cheap, hot-path-safe
intake that turns those otherwise-log-only events into structured, aggregatable evidence
the EXISTING gap flywheel can reason over (see
:mod:`agent_utilities.knowledge_graph.research.runtime_reliability` for the analyzer that
folds them into the canonical ``:Gap``).

Design (the ONE invariant): a signal write must NEVER affect the hot path.
:func:`record_runtime_signal` therefore only appends a privacy-safe dict to a bounded
in-process ring buffer under a short lock — O(1), no engine I/O, no thread spawn, and it
swallows every exception (a failing emit is a dropped signal, never a raised one, never a
stalled caller). ALL engine I/O — persisting the batch as ``:RuntimeSignal`` nodes and
reading them back over a window — happens later, off the hot path, from the consolidated
maintenance scheduler's background-priority tick. Persistence/read go through the SAME
``engine.add_node``/``engine.query_cypher`` surface :mod:`...research.gaps` uses (not
``native_ingest``), so the store is backend-agnostic, mockable, and durable/queryable.
"""

import contextlib
import itertools
import json
import logging
import re
import threading
import time
from collections import deque
from typing import Any

logger = logging.getLogger(__name__)

#: The graph label for a persisted runtime signal (aggregation fodder for the analyzer).
RUNTIME_SIGNAL_LABEL = "RuntimeSignal"

#: The four recognized signal kinds — each maps to one existing detection site.
KIND_ENGINE_LATENCY = "engine_latency"  # engine_breaker slow-call path
KIND_LISTENER_RESTART = "listener_restart"  # messaging router self-healing supervisor
KIND_RETRIEVAL_DEGRADED = (
    "retrieval_degraded"  # contextual_model bounded-compile degrade
)
KIND_DELEGATION_OVER_BUDGET = "delegation_over_budget"  # agent_runner over wall-clock
_KINDS = frozenset(
    {
        KIND_ENGINE_LATENCY,
        KIND_LISTENER_RESTART,
        KIND_RETRIEVAL_DEGRADED,
        KIND_DELEGATION_OVER_BUDGET,
    }
)

SEVERITY_INFO = "info"
SEVERITY_WARNING = "warning"
SEVERITY_CRITICAL = "critical"

#: Bounded in-process ring buffer. The maxlen is the sole memory bound on the hot-path
#: intake: a burst that outruns the background drain drops the OLDEST signal (a lost
#: sample, never unbounded growth, never a stalled emitter). Sized generously — a drain
#: runs every few minutes and real emits are exceptional events, not per-call.
_MAX_BUFFERED_SIGNALS = 512
_SIGNAL_BUFFER: deque[dict[str, Any]] = deque(maxlen=_MAX_BUFFERED_SIGNALS)
_BUFFER_LOCK = threading.Lock()

#: Monotonic per-occurrence sequence — makes each persisted node id unique even when two
#: same-``(kind, subject)`` signals land in the same millisecond, so a burst is counted as
#: N occurrences (not collapsed into one node). ``next`` on an ``itertools.count`` is atomic.
_SEQ = itertools.count()

#: Default aggregation/read window shared with the analyzer (kept here so the store and
#: its reader agree). A named constant, not an env knob (configuration discipline).
_DEFAULT_WINDOW_S = 900.0  # 15 minutes

#: Bounds on the privacy-safe payload so a signal can never carry prompt/message content
#: or an unbounded blob into the KG.
_MAX_SUBJECT_LEN = 120
_MAX_DETAIL_KEYS = 12
_MAX_DETAIL_STR_LEN = 80


def _now() -> float:
    return time.time()


def _iso(ts: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts))


def _slug(text: str, *, limit: int = 80) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", (text or "").lower()).strip("-")
    return (s[:limit] or "unknown").rstrip("-")


def _sanitize_detail(detail: Any) -> dict[str, Any]:
    """Keep ONLY privacy-safe scalars (numbers/bools + short tokens), bounded in count.

    ``detail`` is meant to be numbers (durations, thresholds, counts). Anything that
    isn't a number/bool/short-string is dropped, so no caller can smuggle prompt or
    message content into a signal.
    """
    out: dict[str, Any] = {}
    if not isinstance(detail, dict):
        return out
    for key, value in detail.items():
        if len(out) >= _MAX_DETAIL_KEYS:
            break
        k = str(key)[:40]
        if isinstance(value, bool | int | float):
            out[k] = value
        elif isinstance(value, str):
            out[k] = value[:_MAX_DETAIL_STR_LEN]
    return out


def _build_signal(
    kind: str, subject: str, detail: Any, severity: str
) -> dict[str, Any]:
    ts = _now()
    return {
        "kind": str(kind),
        "subject": str(subject or "")[:_MAX_SUBJECT_LEN],
        "severity": str(severity or SEVERITY_WARNING),
        "detail": _sanitize_detail(detail),
        "ts": ts,
        "at": _iso(ts),
    }


def record_runtime_signal(
    kind: str,
    subject: str,
    detail: dict[str, Any] | None = None,
    *,
    severity: str = SEVERITY_WARNING,
) -> None:
    """Fire-and-forget: record ONE runtime-reliability signal. NEVER raises, NEVER blocks.

    This is the ONLY function the four detection sites call, and it is the hot path's
    entire cost: build a small privacy-safe dict and append it to the bounded ring
    buffer under a short lock. There is no engine contact here by design — the engine
    could be exactly what is contended (an ``engine_latency`` signal is emitted BECAUSE
    the engine is slow); writing to it synchronously from the hot path would compound the
    very failure we are recording. Every exception is swallowed: a signal is best-effort
    telemetry, and losing one must never perturb the caller.

    ``kind`` should be one of the ``KIND_*`` constants; an unknown kind is still recorded
    (the analyzer routes unknown kinds to the flywheel). ``subject`` is the op/backend/run
    id the signal concerns; ``detail`` is a small dict of NUMBERS (durations, thresholds,
    counts).
    """
    try:
        signal = _build_signal(kind, subject, detail, severity)
        with _BUFFER_LOCK:
            _SIGNAL_BUFFER.append(signal)
    except Exception:  # noqa: BLE001 — a signal must never affect the caller's hot path
        with contextlib.suppress(Exception):
            logger.debug("record_runtime_signal dropped a signal (kind=%s)", kind)


def buffered_runtime_signals() -> list[dict[str, Any]]:
    """A snapshot copy of the buffered signals (non-destructive; for tests/inspection)."""
    with _BUFFER_LOCK:
        return list(_SIGNAL_BUFFER)


def drain_buffered_signals() -> list[dict[str, Any]]:
    """Atomically take-and-clear the buffered batch (the background drain's intake)."""
    with _BUFFER_LOCK:
        items = list(_SIGNAL_BUFFER)
        _SIGNAL_BUFFER.clear()
    return items


def _signal_node_id(signal: dict[str, Any]) -> str:
    """A UNIQUE per-occurrence ``:RuntimeSignal`` id: ``runtime:signal:<kind>:<subject>:<ts>-<seq>``.

    The trailing sequence guarantees two same-``(kind, subject)`` signals in the same
    millisecond do not collide on one node (which would undercount a burst pattern)."""
    ts_ms = int(float(signal.get("ts") or _now()) * 1000)
    return (
        f"runtime:signal:{_slug(str(signal.get('kind')), limit=40)}:"
        f"{_slug(str(signal.get('subject')), limit=80)}:{ts_ms}-{next(_SEQ)}"
    )


def persist_runtime_signals(engine: Any, signals: list[dict[str, Any]]) -> int:
    """Best-effort write of a batch of buffered signals as ``:RuntimeSignal`` nodes.

    Runs OFF the hot path (from the maintenance tick), through the same
    ``engine.add_node`` surface :func:`...research.gaps.submit_gap` uses — so it is
    backend-agnostic and mockable, and it never touches the (possibly contended) engine
    on a caller's thread. Every node write is individually exception-isolated: one bad
    write never drops the rest, and a wholly unavailable engine simply persists nothing.
    Returns the count actually written.
    """
    if engine is None or not signals:
        return 0
    written = 0
    for signal in signals:
        try:
            sid = _signal_node_id(signal)
            engine.add_node(
                sid,
                RUNTIME_SIGNAL_LABEL,
                properties={
                    "id": sid,
                    "node_type": "runtime_signal",
                    "kind": str(signal.get("kind")),
                    "subject": str(signal.get("subject") or ""),
                    "severity": str(signal.get("severity") or SEVERITY_WARNING),
                    "detail": json.dumps(signal.get("detail") or {}, default=str),
                    "ts": float(signal.get("ts") or _now()),
                    "timestamp": str(signal.get("at") or _iso(_now())),
                },
            )
            written += 1
        except Exception as e:  # noqa: BLE001 — persist is best-effort, per-signal isolated
            logger.debug("persist_runtime_signals: node write failed: %s", e)
    return written


def read_recent_runtime_signals(
    engine: Any, *, window_s: float = _DEFAULT_WINDOW_S, limit: int = 5000
) -> list[dict[str, Any]]:
    """Read persisted ``:RuntimeSignal`` events observed within ``window_s`` (flat dicts).

    Backend-agnostic read via ``engine.query_cypher`` (the same label-scan idiom
    :func:`...research.gaps.open_gaps` uses). Best-effort: ``[]`` on any read failure or
    with no reachable engine. The window keeps aggregation bounded regardless of how many
    ``:RuntimeSignal`` nodes have accumulated.
    """
    if engine is None:
        return []
    try:
        rows = engine.query_cypher(
            f"MATCH (n:{RUNTIME_SIGNAL_LABEL}) RETURN n LIMIT {int(limit)}"
        )
    except Exception as e:  # noqa: BLE001 — read is best-effort
        logger.debug("read_recent_runtime_signals query failed: %s", e)
        return []
    cutoff = _now() - float(window_s)
    out: list[dict[str, Any]] = []
    for row in rows or []:
        props = row.get("n") if isinstance(row, dict) else None
        if not isinstance(props, dict):
            continue
        _ts_raw = props.get("ts")
        try:
            ts = float(_ts_raw) if _ts_raw is not None else None
        except (TypeError, ValueError):
            ts = None
        if ts is not None and ts < cutoff:
            continue
        detail = props.get("detail")
        if isinstance(detail, str):
            try:
                detail = json.loads(detail)
            except (TypeError, ValueError):
                detail = {}
        out.append(
            {
                "kind": props.get("kind"),
                "subject": props.get("subject"),
                "severity": props.get("severity"),
                "detail": detail if isinstance(detail, dict) else {},
                "ts": ts,
                "at": props.get("timestamp"),
            }
        )
    return out


def prune_old_runtime_signals(engine: Any, *, retention_s: float) -> None:
    """Best-effort delete of ``:RuntimeSignal`` nodes older than ``retention_s``.

    A bound on KG accumulation where the backend supports DELETE via ``query_cypher``;
    fully guarded, so a backend that exposes a read-only cypher surface simply keeps the
    nodes (they are then aged out of analysis by the window read, and swept by general KG
    hygiene). Never raises.
    """
    if engine is None:
        return
    cutoff = _now() - float(retention_s)
    with contextlib.suppress(Exception):
        engine.query_cypher(
            f"MATCH (n:{RUNTIME_SIGNAL_LABEL}) WHERE n.ts < $cutoff DETACH DELETE n",
            {"cutoff": cutoff},
        )


__all__ = [
    "RUNTIME_SIGNAL_LABEL",
    "KIND_ENGINE_LATENCY",
    "KIND_LISTENER_RESTART",
    "KIND_RETRIEVAL_DEGRADED",
    "KIND_DELEGATION_OVER_BUDGET",
    "SEVERITY_INFO",
    "SEVERITY_WARNING",
    "SEVERITY_CRITICAL",
    "record_runtime_signal",
    "buffered_runtime_signals",
    "drain_buffered_signals",
    "persist_runtime_signals",
    "read_recent_runtime_signals",
    "prune_old_runtime_signals",
]
