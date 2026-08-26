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
stalled caller). ALL engine I/O — persisting the batch as ``:RuntimeSignal`` nodes,
reading them back over a window, and retiring the expired ones — happens later, off the
hot path, from the consolidated maintenance scheduler's background-priority tick.

RPC BUDGET (the second invariant): the engine carries ~1s of fixed overhead per call and
this is a high-volume writer, so NOTHING here may scale RPCs with signal count.
:func:`persist_runtime_signals` commits a whole drain in ONE governed typed batch;
:func:`prune_old_runtime_signals` retires expired rows a bounded page at a time (two
statements per page). Reads and deletes are backend-agnostic and mockable, but the
DELETE deliberately does NOT go through ``engine.query_cypher``: that is the read
chokepoint, and the engine refuses a mutation declared as a read — see
:func:`_statement_executor`.
"""

import contextlib
import itertools
import json
import logging
import re
import threading
import time
from collections import deque
from dataclasses import dataclass
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

#: Retention sweep shape. The sweep is SELECT-a-bounded-page + DELETE-those-ids, the
#: same two-statement form :meth:`...core.maintainer.KnowledgeMaintainer.prune_expired_traces`
#: already proved against the native backend (a single-statement
#: ``MATCH ... WITH n LIMIT $n DETACH DELETE n`` is OUTSIDE the engine's Cypher WRITE
#: subset — a write statement has no read-pipeline stage, so ``WITH`` between the MATCH
#: and the write clause is rejected at the wire boundary). Two RPCs per page, NOT one
#: per node.
_RETENTION_BATCH_SIZE = 1_000
_RETENTION_MAX_BATCHES = 1_000

#: ARMING SWITCH for the destructive half of retention. ``False`` (the shipped default)
#: makes :func:`prune_old_runtime_signals` a DRY RUN: it counts what it would delete and
#: logs that count loudly, but removes nothing. Flip to ``True`` — a one-line, reviewable
#: change — to arm the real sweep.
#:
#: Why it ships disarmed: retention has been structurally broken since it was written
#: (see :func:`prune_old_runtime_signals`), so the first armed sweep on a live graph is a
#: bulk deletion of a large accumulated backlog, not the incremental trim the 2-hour TTL
#: implies. Merging to ``agent-utilities`` ``main`` IS a production deployment, so that
#: deletion is the owner's decision to make explicitly, not a side effect of a bug fix.
RETENTION_DELETE_ENABLED = False

#: Bounds on the privacy-safe payload so a signal can never carry prompt/message content
#: or an unbounded blob into the KG.
_MAX_SUBJECT_LEN = 120
_MAX_DETAIL_KEYS = 12
_MAX_DETAIL_STR_LEN = 80


@dataclass(frozen=True, slots=True)
class RetentionReport:
    """What one retention sweep did — a TYPED result, not a bag of string keys.

    ``deleted``   nodes actually removed this sweep (always ``0`` on a dry run).
    ``expired``   how many nodes were older than the cutoff. Only populated on a dry
                  run (that is the whole question a dry run answers); ``None`` when the
                  backend returned no countable row, never a fabricated ``0``.
    ``dry_run``   whether the destructive half was disarmed (see
                  :data:`RETENTION_DELETE_ENABLED`).
    ``truncated`` more expired rows remained than this sweep's page budget — the next
                  tick continues. Distinguishes "finished" from "ran out of budget", so
                  ``deleted`` is never misread as "nothing left".
    """

    deleted: int = 0
    expired: int | None = None
    dry_run: bool = True
    truncated: bool = False


class RuntimeSignalRetentionError(RuntimeError):
    """The ``:RuntimeSignal`` retention sweep could not run.

    Retention is the ONLY bound on this population's growth, so its failure is a
    DEFECT, never a best-effort miss to swallow. This type exists so the caller can
    report the failure (loudly, with the cause attached) instead of the
    ``contextlib.suppress(Exception)`` that hid it for the population's whole lifetime.
    """


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


def _signal_properties(signal: dict[str, Any], sid: str) -> dict[str, Any]:
    """The persisted ``:RuntimeSignal`` property bag for one buffered signal."""
    return {
        "id": sid,
        "node_type": "runtime_signal",
        "kind": str(signal.get("kind")),
        "subject": str(signal.get("subject") or ""),
        "severity": str(signal.get("severity") or SEVERITY_WARNING),
        "detail": json.dumps(signal.get("detail") or {}, default=str),
        "ts": float(signal.get("ts") or _now()),
        "timestamp": str(signal.get("at") or _iso(_now())),
    }


def persist_runtime_signals(engine: Any, signals: list[dict[str, Any]]) -> int:
    """Write a drained batch of buffered signals as ``:RuntimeSignal`` nodes in ONE RPC.

    Runs OFF the hot path (from the maintenance tick). The batch goes through
    ``engine.batch_typed_mutations`` — the governed typed-batch seam that keeps the
    public ``add_node`` contract (verified write authority, label normalization,
    ownership/classification stamping, bitemporal fields) but collapses the whole drain
    into a SINGLE native ``BatchUpdate`` transaction.

    That single RPC is the point. The per-signal ``engine.add_node`` loop this replaces
    issued ONE RPC PER NODE, and each ``add_node`` is itself a one-operation
    ``BatchUpdate`` round-trip (``EpistemicGraphBackend.add_node``). At the engine's ~1s
    fixed per-call overhead and a 512-deep drain buffer, one tick could cost ~512s of
    engine time on a tick scheduled every 180s — a background task capable of
    monopolizing the writer it is supposed to be observing. The batched form is O(1) RPCs
    per tick regardless of how many signals drained.

    Returns the count written; ``0`` when there is nothing to write, no engine, or the
    batch failed (logged, never raised — a lost telemetry batch must not break the tick).
    """
    if engine is None or not signals:
        return 0

    mutations: list[dict[str, Any]] = []
    for signal in signals:
        try:
            sid = _signal_node_id(signal)
            mutations.append(
                {
                    "kind": "node",
                    "id": sid,
                    "node_type": RUNTIME_SIGNAL_LABEL,
                    "properties": _signal_properties(signal, sid),
                }
            )
        except Exception as e:  # noqa: BLE001 — one malformed signal never drops the batch
            logger.debug("persist_runtime_signals: skipped a malformed signal: %s", e)
    if not mutations:
        return 0

    batch = getattr(engine, "batch_typed_mutations", None)
    if callable(batch):
        try:
            if batch(mutations):
                return len(mutations)
        except Exception as e:  # noqa: BLE001 — an all-or-nothing batch failure wrote nothing
            logger.error(
                "persist_runtime_signals: typed batch of %d signal(s) failed: %s",
                len(mutations),
                e,
            )
            return 0
        # A falsy return means "this backend has no native typed-batch capability"
        # (never a partial write), so the per-node fallback below is safe.

    written = 0
    for mutation in mutations:
        try:
            engine.add_node(
                mutation["id"],
                RUNTIME_SIGNAL_LABEL,
                properties=mutation["properties"],
            )
            written += 1
        except Exception as e:  # noqa: BLE001 — fallback is best-effort, per-signal isolated
            logger.debug("persist_runtime_signals: node write failed: %s", e)
    return written


def read_recent_runtime_signals(
    engine: Any, *, window_s: float = _DEFAULT_WINDOW_S, limit: int = 5000
) -> list[dict[str, Any]]:
    """Read persisted ``:RuntimeSignal`` events observed within ``window_s`` (flat dicts).

    Backend-agnostic read via ``engine.query_cypher``. Best-effort: ``[]`` on any read
    failure or with no reachable engine.

    The ``ts`` cutoff is pushed INTO the query, not applied only in Python afterwards.
    Without it this was an unordered whole-label scan capped by ``LIMIT`` — and once the
    population outgrew that cap, the engine could return ``limit`` rows that were ALL
    older than the window, every one of which Python then discarded. The pass saw an
    empty window, concluded there was nothing to analyze, and returned before it reached
    retention — so the larger the population grew, the less likely retention was to run
    at all. The predicate makes the read bounded and correct no matter how many
    ``:RuntimeSignal`` nodes exist. The Python-side ``ts`` filter below is kept as a
    safety net for a backend that ignores the predicate.
    """
    if engine is None:
        return []
    cutoff = _now() - float(window_s)
    try:
        rows = engine.query_cypher(
            f"MATCH (n:{RUNTIME_SIGNAL_LABEL}) WHERE n.ts >= $cutoff "
            f"RETURN n LIMIT {int(limit)}",
            {"cutoff": cutoff},
        )
    except Exception as e:  # noqa: BLE001 — read is best-effort
        logger.debug("read_recent_runtime_signals query failed: %s", e)
        return []
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


def _statement_executor(engine: Any) -> Any:
    """The engine surface that can run a ``:RuntimeSignal`` retention statement.

    ``engine.backend.execute`` FIRST, and that ordering is the entire bug fix.
    ``EpistemicGraphBackend.execute`` classifies the statement and dispatches a write to
    the engine's ``cypher_write`` wire mode; ``engine.query_cypher`` is a READ chokepoint
    that always calls ``execute_read`` (``kg:read`` scope, wire ``mode="read"``). The
    engine re-parses every statement and rejects a mode mismatch before execution
    (``src/server/handlers/query.rs::validate_cypher_mode`` vs
    ``eg_query::classify_cypher``, which classifies ``DETACH DELETE`` as ``Write``), so a
    deletion sent through ``query_cypher`` cannot ever have deleted anything.

    ``query_cypher`` remains as the fallback so a duck-typed engine/test double that only
    implements that one method still works; both take ``(query, params)`` positionally.
    """
    execute = getattr(getattr(engine, "backend", None), "execute", None)
    if callable(execute):
        return execute
    execute = getattr(engine, "query_cypher", None)
    if callable(execute):
        return execute
    return None


def _as_int(rows: Any, keys: tuple[str, ...]) -> int | None:
    """First integer found under any of ``keys`` across a backend's result rows."""
    for row in (rows if isinstance(rows, list) else [rows]) or []:
        if not isinstance(row, dict):
            continue
        for key in keys:
            if key in row:
                try:
                    return int(row[key])
                except (TypeError, ValueError):
                    return None
    return None


def count_expired_runtime_signals(engine: Any, *, retention_s: float) -> int | None:
    """How many ``:RuntimeSignal`` nodes are older than ``retention_s`` — ONE read.

    The dry-run counterpart of :func:`prune_old_runtime_signals`: it answers "what would
    the sweep delete?" without deleting anything. ``None`` when no engine is reachable or
    the backend returned no countable row (never a fabricated ``0``).
    """
    if engine is None:
        return None
    execute = _statement_executor(engine)
    if execute is None:
        return None
    rows = execute(
        f"MATCH (n:{RUNTIME_SIGNAL_LABEL}) WHERE n.ts < $cutoff "
        f"RETURN count(n) AS expired",
        {"cutoff": _now() - float(retention_s)},
    )
    return _as_int(rows, ("expired", "count(n)", "count"))


def prune_old_runtime_signals(
    engine: Any,
    *,
    retention_s: float,
    delete: bool | None = None,
) -> RetentionReport:
    """Delete ``:RuntimeSignal`` nodes older than ``retention_s``. Raises on failure.

    WHY THIS EXISTS IN THIS SHAPE — the previous implementation was a single unbatched
    ``MATCH (n:RuntimeSignal) WHERE n.ts < $cutoff DETACH DELETE n`` issued through
    ``engine.query_cypher`` and wrapped in ``contextlib.suppress(Exception)``. Both
    halves were wrong, and together they were invisible:

    * ``query_cypher`` is the READ chokepoint (``kg:read`` scope, wire ``mode="read"``),
      and the engine rejects a declared-mode/parsed-statement mismatch before executing
      anything. The deletion was refused at the wire boundary on EVERY tick since the
      code was written — it never removed a single node.
    * ``suppress(Exception)`` then discarded that refusal, so a nominal 2-hour TTL could
      accumulate an unbounded population while reporting nothing at all. The TTL was
      never wrong; it was never applied.

    So: the statement now goes through :func:`_statement_executor` (write-capable), and
    NOTHING is suppressed — a failure is logged at ERROR and raised as
    :class:`RuntimeSignalRetentionError` for the caller to report.

    Shape: SELECT one bounded page of ids (a READ, where ``LIMIT`` is unrestricted), then
    DELETE exactly those ids (a WRITE) — two RPCs per page, never one per node, and never
    one unbounded delete. This is the same form ``prune_expired_traces`` already proved
    against the native backend, for the same reason: ``MATCH ... WITH n LIMIT $n DETACH
    DELETE n`` is outside the engine's Cypher WRITE subset.

    ``delete`` defaults to :data:`RETENTION_DELETE_ENABLED`. When it is ``False`` this is
    a DRY RUN: it counts what it would remove, logs that count, and deletes nothing.
    """
    if delete is None:
        delete = RETENTION_DELETE_ENABLED
    if engine is None:
        return RetentionReport(dry_run=not delete)

    execute = _statement_executor(engine)
    if execute is None:
        raise RuntimeSignalRetentionError(
            "runtime-signal retention cannot run: the engine exposes neither "
            "backend.execute nor query_cypher"
        )

    cutoff = _now() - float(retention_s)
    select_query = (
        f"MATCH (n:{RUNTIME_SIGNAL_LABEL}) WHERE n.ts < $cutoff "
        f"RETURN n.id AS id LIMIT $batch_size"
    )
    delete_query = (
        f"MATCH (n:{RUNTIME_SIGNAL_LABEL}) WHERE n.id IN $ids DETACH DELETE n"
    )

    deleted = 0
    truncated = False
    try:
        if not delete:
            expired = count_expired_runtime_signals(engine, retention_s=retention_s)
            logger.warning(
                "[runtime-reliability] retention DRY RUN: %s :%s node(s) are older "
                "than %ds and would be deleted. Nothing was removed — set "
                "runtime_signals.RETENTION_DELETE_ENABLED = True to arm the sweep.",
                expired,
                RUNTIME_SIGNAL_LABEL,
                int(retention_s),
            )
            return RetentionReport(expired=expired, dry_run=True)

        for _page in range(_RETENTION_MAX_BATCHES):
            rows = execute(
                select_query,
                {"cutoff": cutoff, "batch_size": _RETENTION_BATCH_SIZE},
            )
            ids = [
                row["id"]
                for row in (rows if isinstance(rows, list) else [rows]) or []
                if isinstance(row, dict) and row.get("id") is not None
            ]
            if not ids:
                break
            execute(delete_query, {"ids": ids})
            deleted += len(ids)
            if len(ids) < _RETENTION_BATCH_SIZE:
                break
        else:
            # Fell out of the loop without a short/empty page: more expired rows
            # remain than this tick's budget. The next tick continues; say so
            # rather than let the caller read `deleted` as "nothing left".
            truncated = True
    except Exception as e:
        logger.error(
            "[runtime-reliability] retention sweep FAILED after deleting %d :%s "
            "node(s) (retention_s=%s): %s — this population has no other bound on "
            "its growth, so this is a defect, not a best-effort miss",
            deleted,
            RUNTIME_SIGNAL_LABEL,
            retention_s,
            e,
        )
        raise RuntimeSignalRetentionError(
            f"runtime-signal retention sweep failed: {e}"
        ) from e

    if deleted or truncated:
        logger.info(
            "[runtime-reliability] retention deleted %d :%s node(s) older than %ds%s",
            deleted,
            RUNTIME_SIGNAL_LABEL,
            int(retention_s),
            " (budget reached; more remain)" if truncated else "",
        )
    return RetentionReport(deleted=deleted, dry_run=False, truncated=truncated)


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
    "count_expired_runtime_signals",
    "RuntimeSignalRetentionError",
    "RETENTION_DELETE_ENABLED",
    "RetentionReport",
]
