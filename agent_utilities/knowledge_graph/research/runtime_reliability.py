#!/usr/bin/python
from __future__ import annotations

"""Runtime-reliability gap analyzer + primitive reconciler — the detect→gap→heal tail.

CONCEPT:AU-AHE.harness.runtime-reliability-loop — extends the canonical gap flywheel
(:mod:`agent_utilities.knowledge_graph.research.gaps`) with a fifth discovery track,
``SOURCE_RUNTIME``, so a class of RUNTIME reliability failures the self-improvement stack
never noticed becomes first-class flywheel input. The reward/failure-analyzer flywheel
keys off AGENT-RUN QUALITY and LLM spans; the four signals wired this session
(``engine_latency``/``listener_restart``/``retrieval_degraded``/``delegation_over_budget``,
see :mod:`agent_utilities.observability.runtime_signals`) are runtime events that often
never became a *run* at all. This module is the analyzer that reads those persisted
``:RuntimeSignal`` events, aggregates them over a window, and — for a pattern crossing a
threshold — OPENS the SAME canonical ``:Gap`` every other track uses (:func:`submit_gap`),
deduped against already-open gaps.

The division of labor mirrors the task's detect→gap→heal shape and is deliberately
conservative (recommendation-only or already-safe; NO speculative auto-mutation of prod):

* :func:`runtime_reliability_analyzer` runs one full pass — drain+persist the buffered
  signals, read the window, aggregate by ``(kind, subject)``, and for each crossing
  pattern open a flywheel ``:Gap`` for the UNRECOGNIZED classes (``delegation_over_budget``
  and any future/unknown kind — genuine "investigate" work the SDD flywheel should pick
  up), then hand the RECOGNIZED classes to the reconciler.
* :func:`runtime_reconciler` handles the recognized classes with a SAFE known
  disposition: ``listener_restart`` is ALREADY auto-healed by the messaging supervisor →
  it records a *resolved* heal (a closed-loop annotation, not open work); ``engine_latency``
  / ``retrieval_degraded`` → it opens a RECOMMENDATION ``:Gap`` (config/perf — "consider
  batching/caching", "review the retrieval budget") rather than mutate anything.

Everything is best-effort and engine-guarded; the pass never raises. It runs under the
consolidated maintenance scheduler's background priority (bucket 3), native/default-on,
alongside the other analysis ticks (``anomaly_consumer``/``tms_revalidation``/…).
"""

import logging
from typing import Any

from agent_utilities.knowledge_graph.research.gaps import (
    SOURCE_RUNTIME,
    canonical_gap_id,
    get_gap,
    mark_gap_resolved,
    open_gaps,
    submit_gap,
)
from agent_utilities.observability import runtime_signals

logger = logging.getLogger(__name__)

#: Aggregation window (shared default with the signal store) and per-tick KG retention.
_WINDOW_S = runtime_signals._DEFAULT_WINDOW_S  # 15 minutes
_RETENTION_S = _WINDOW_S * 8  # keep a few windows of history, then best-effort prune

#: How many times a ``(kind, subject)`` pattern must recur within the window before it is
#: worth a gap. Per-kind, auto-sized to each signal's base rate (a slow engine call is
#: noisier than a listener restart), named constants — not an env-flag family.
_MIN_COUNT: dict[str, int] = {
    runtime_signals.KIND_ENGINE_LATENCY: 5,
    runtime_signals.KIND_LISTENER_RESTART: 3,
    runtime_signals.KIND_RETRIEVAL_DEGRADED: 3,
    runtime_signals.KIND_DELEGATION_OVER_BUDGET: 3,
}
_DEFAULT_MIN_COUNT = 5

#: Disposition per recognized class (the reconciler's policy). Everything NOT listed here
#: is an unrecognized/flywheel class the analyzer opens a plain investigate-gap for.
_HEAL = (
    "heal"  # already auto-healed elsewhere → record a resolved, closed-loop annotation
)
_RECOMMEND = (
    "recommend"  # a safe config/perf recommendation → open a recommendation gap
)
_RECONCILE_DISPOSITION: dict[str, str] = {
    runtime_signals.KIND_LISTENER_RESTART: _HEAL,
    runtime_signals.KIND_ENGINE_LATENCY: _RECOMMEND,
    runtime_signals.KIND_RETRIEVAL_DEGRADED: _RECOMMEND,
}

#: Per-kind severity (0..1) for the opened gap → maps to the shared priority bucket.
_SEVERITY: dict[str, float] = {
    runtime_signals.KIND_DELEGATION_OVER_BUDGET: 0.6,
    runtime_signals.KIND_ENGINE_LATENCY: 0.5,
    runtime_signals.KIND_RETRIEVAL_DEGRADED: 0.5,
    runtime_signals.KIND_LISTENER_RESTART: 0.3,
}
_DEFAULT_SEVERITY = 0.5

#: A recommendation phrase per kind — the "consider X" half of the gap topic.
_RECOMMENDATION_HINT: dict[str, str] = {
    runtime_signals.KIND_ENGINE_LATENCY: (
        "consider batching/caching this op or reducing calls on the retrieval path"
    ),
    runtime_signals.KIND_RETRIEVAL_DEGRADED: (
        "review the retrieval time budget / index — compilation is timing out or erroring"
    ),
}


def _signature(kind: str, subject: str) -> str:
    """Stable dedupe key for a ``(kind, subject)`` pattern → one gap id across ticks."""
    return f"{kind}:{subject}"


def _aggregate(
    signals: list[dict[str, Any]],
) -> dict[tuple[str, str], dict[str, Any]]:
    """Group recent signals by ``(kind, subject)`` with a count + a compact evidence roll-up."""
    groups: dict[tuple[str, str], dict[str, Any]] = {}
    for sig in signals:
        kind = str(sig.get("kind") or "")
        subject = str(sig.get("subject") or "")
        if not kind:
            continue
        key = (kind, subject)
        agg = groups.get(key)
        if agg is None:
            agg = groups[key] = {
                "kind": kind,
                "subject": subject,
                "count": 0,
                "severity": "warning",
                "last_detail": {},
                "last_at": "",
            }
        agg["count"] += 1
        if sig.get("detail"):
            agg["last_detail"] = sig["detail"]
        if sig.get("at"):
            agg["last_at"] = sig["at"]
        if str(sig.get("severity")) == runtime_signals.SEVERITY_CRITICAL:
            agg["severity"] = "critical"
    return groups


def _crosses_threshold(agg: dict[str, Any]) -> bool:
    threshold = _MIN_COUNT.get(str(agg.get("kind")), _DEFAULT_MIN_COUNT)
    return int(agg.get("count", 0)) >= threshold


def _evidence_refs(agg: dict[str, Any]) -> list[str]:
    detail = agg.get("last_detail") or {}
    detail_str = ",".join(f"{k}={v}" for k, v in list(detail.items())[:8])
    return [
        f"runtime_signal:{agg['kind']}:{agg['subject']}",
        f"count={agg['count']} window_s={int(_WINDOW_S)}",
        f"last={agg.get('last_at', '')} {detail_str}".strip(),
    ]


# ── Code references (the golden egg) ────────────────────────────────────────
#
# The ecosystem's source (agent-utilities / epistemic-graph / universal-skills) is
# ingested into the KG via tree-sitter/AST + code embeddings as ``:Code`` nodes carrying
# ``file_path``/``line``/``name``. So a runtime gap does not just say "something is slow" —
# it can point at the EXACT ingested code the fix concerns, with line numbers, and link to
# it. That is what lets the gap flow through the standardized evolution path (Gap → spec →
# implement via the agent graph) with real, traversable targets instead of prose.
#
# Known detection / fix sites per kind (real agent-utilities symbols). Line numbers are
# resolved from the KG (never hard-coded, so a suggested-change reference cannot drift); the
# file+symbol is the drift-free fallback when that symbol's code isn't ingested yet.
_FIX_SITES: dict[str, tuple[tuple[str, str], ...]] = {
    runtime_signals.KIND_ENGINE_LATENCY: (
        ("agent_utilities/knowledge_graph/core/engine_breaker.py", "_observe_latency"),
    ),
    runtime_signals.KIND_LISTENER_RESTART: (
        ("agent_utilities/messaging/router.py", "_supervise_backend"),
    ),
    runtime_signals.KIND_RETRIEVAL_DEGRADED: (
        (
            "agent_utilities/core/contextual_model.py",
            "_compiled_evidence_and_bundle_bounded",
        ),
    ),
    runtime_signals.KIND_DELEGATION_OVER_BUDGET: (
        ("agent_utilities/orchestration/agent_runner.py", "_execute_single_server"),
    ),
}


def _resolve_code_anchors(engine: Any, kind: str, subject: str) -> list[dict[str, Any]]:
    """Best-effort resolve the fix-site symbol(s) + the subject to REAL ingested ``:Code``
    nodes (``{id, symbol, file, line}``) so the gap carries precise, traversable code
    references. ``[]`` when the code isn't ingested / unresolvable / the retriever is absent.
    """
    try:
        from agent_utilities.knowledge_graph.retrieval.code_context import (
            resolve_anchors,
        )
    except Exception:  # noqa: BLE001 — retriever optional; degrade to static refs
        return []
    queries = [sym for _f, sym in _FIX_SITES.get(kind, ())]
    if subject:
        queries.append(subject)
    anchors: list[dict[str, Any]] = []
    seen: set[str] = set()
    for q in queries:
        try:
            rows = resolve_anchors(engine, query=q, limit=2)
        except Exception as e:  # noqa: BLE001 — one bad resolve never drops the rest
            logger.debug("runtime-reliability: code anchor resolve failed: %s", e)
            rows = []
        for a in rows or []:
            nid = str(a.get("id") or "")
            if a.get("file") and nid and nid not in seen:
                seen.add(nid)
                anchors.append(a)
    return anchors


def _code_evidence(kind: str, anchors: list[dict[str, Any]]) -> list[str]:
    """Code references as evidence strings: the drift-free fix-site file+symbol always, plus
    a ``file:line`` for every anchor the KG resolved from ingested code."""
    refs = [
        f"code:{file_path} ({symbol}) — suggested fix site"
        for file_path, symbol in _FIX_SITES.get(kind, ())
    ]
    for a in anchors:
        line = a.get("line")
        loc = f"{a.get('file')}:{line}" if line else str(a.get("file"))
        refs.append(f"code:{loc} ({a.get('symbol')})")
    return refs


def _submit_runtime_gap(
    engine: Any,
    *,
    kind: str,
    subject: str,
    statement: str,
    severity: float,
    agg: dict[str, Any],
    open_ids: set[str],
    lease: bool = True,
    resolve: bool = False,
) -> dict[str, Any] | None:
    """Open ONE canonical ``:Gap`` for a runtime pattern, WITH code references — the single
    gap-creation seam every disposition (flywheel / recommendation / heal) shares.

    Dedupes against already-tracked gaps, attaches the signal evidence + resolved code
    references, and links each resolved ``:Code`` anchor to the gap with the EXISTING
    ``(:Code)-[:EVIDENCES]->(:Gap)`` provenance convention (the same edge the failure
    analyzer / anomaly consumer use) so the anchor is traversable from the gap by the SAME
    machinery the SDD/implementer path already walks. ``resolve=True`` records a
    born-resolved heal; ``lease=False`` skips the WorkItem for non-schedulable heals.
    """
    signature = _signature(kind, subject)
    gap_id = canonical_gap_id(SOURCE_RUNTIME, signature)
    if gap_id in open_ids or get_gap(engine, gap_id) is not None:
        return None  # already tracked (open, specified, or resolved) — do not clobber
    anchors = _resolve_code_anchors(engine, kind, subject)
    evidence = _evidence_refs(agg) + _code_evidence(kind, anchors)
    gap = submit_gap(
        engine,
        source=SOURCE_RUNTIME,
        signature=signature,
        statement=statement,
        domain="runtime_reliability",
        severity=severity,
        evidence_refs=evidence,
        lease=lease,
    )
    if not gap:
        return None
    open_ids.add(gap_id)
    for a in anchors:
        nid = a.get("id")
        if not nid:
            continue
        try:  # (:Code)-[:EVIDENCES]->(:Gap) — traversable suggested-change anchor
            engine.add_edge(nid, gap["id"], "EVIDENCES")
        except Exception as e:  # noqa: BLE001 — provenance edge is best-effort
            logger.debug("runtime gap EVIDENCES edge failed: %s", e)
    if resolve:
        mark_gap_resolved(engine, gap["id"])
    return gap


def _open_flywheel_gap(
    engine: Any, agg: dict[str, Any], open_ids: set[str]
) -> str | None:
    """Open a plain investigate ``:Gap`` for an UNRECOGNIZED runtime pattern, deduped."""
    statement = (
        f"runtime: {agg['kind']} on '{agg['subject']}' recurred {agg['count']}× within "
        f"{int(_WINDOW_S // 60)}min — a reliability failure that never surfaced as a run; "
        f"investigate the root cause"
    )
    gap = _submit_runtime_gap(
        engine,
        kind=str(agg["kind"]),
        subject=str(agg["subject"]),
        statement=statement,
        severity=_SEVERITY.get(str(agg["kind"]), _DEFAULT_SEVERITY),
        agg=agg,
        open_ids=open_ids,
    )
    if gap:
        logger.info(
            "[runtime-reliability] opened flywheel gap %s (%s)", gap["id"], statement
        )
        return gap["id"]
    return None


def runtime_reliability_analyzer(engine: Any) -> dict[str, Any]:
    """One detect→gap pass over recent ``:RuntimeSignal`` events (background-priority tick).

    Drains the hot-path buffer and persists it (off the hot path), reads the window,
    aggregates by ``(kind, subject)``, and for every pattern crossing its threshold opens
    a flywheel ``:Gap`` for the UNRECOGNIZED classes and delegates the RECOGNIZED classes
    to :func:`runtime_reconciler`. Dedupes against already-open gaps (:func:`open_gaps`).
    Best-effort throughout: returns a summary and never raises.
    """
    report: dict[str, Any] = {
        "scanned": 0,
        "persisted": 0,
        "patterns": 0,
        "gaps_opened": 0,
        "recommendations": 0,
        "heals": 0,
    }
    if engine is None:
        return report

    # 1) Drain the hot-path buffer and persist it off the hot path (best-effort), so the
    #    window read below spans this and prior ticks — a pattern building across ticks is
    #    caught. Persist failure is non-fatal: we still analyze the drained batch.
    try:
        drained = runtime_signals.drain_buffered_signals()
        report["persisted"] = runtime_signals.persist_runtime_signals(engine, drained)
    except Exception as e:  # noqa: BLE001 — persist must never break the analysis pass
        logger.debug("runtime-reliability: drain/persist failed: %s", e)
        drained = []

    # 2) Read the window (falls back to just this tick's drained batch if the read is
    #    unavailable, so a single-tick burst is still analyzable without a working read).
    recent = runtime_signals.read_recent_runtime_signals(engine, window_s=_WINDOW_S)
    if not recent:
        recent = drained
    report["scanned"] = len(recent)
    if not recent:
        return report

    # 3) Aggregate + dedupe against currently-open gaps.
    aggregates = _aggregate(recent)
    try:
        open_ids = {str(g.get("id")) for g in open_gaps(engine) if g.get("id")}
    except Exception as e:  # noqa: BLE001 — a failed dedupe read just yields an empty open_ids set, so a pattern that already has an open gap might get re-filed (gap_id is deterministic per signature, so it MERGEs rather than duplicating) — never a lost signal
        logger.debug("runtime-reliability: open_gaps read failed: %s", e)
        open_ids = set()

    recognized: list[dict[str, Any]] = []
    for agg in aggregates.values():
        if not _crosses_threshold(agg):
            continue
        report["patterns"] += 1
        if str(agg["kind"]) in _RECONCILE_DISPOSITION:
            recognized.append(agg)
            continue
        if _open_flywheel_gap(engine, agg, open_ids) is not None:
            report["gaps_opened"] += 1

    # 4) Known-class reconciliation (recommendation-only or already-safe).
    healed = runtime_reconciler(engine, recognized, open_ids=open_ids)
    report["recommendations"] = healed.get("recommendations", 0)
    report["heals"] = healed.get("heals", 0)

    # 5) Best-effort bound on KG accumulation.
    runtime_signals.prune_old_runtime_signals(engine, retention_s=_RETENTION_S)
    return report


def runtime_reconciler(
    engine: Any,
    aggregates: list[dict[str, Any]] | None = None,
    *,
    open_ids: set[str] | None = None,
) -> dict[str, Any]:
    """Apply the SAFE known-class disposition to recognized runtime patterns.

    ``listener_restart`` — already auto-healed by the messaging router's self-healing
    supervisor — is recorded as a *resolved* heal (a closed-loop annotation via the
    existing gap lifecycle: born ``open`` then immediately ``resolved``, ``lease=False`` so
    it is not scheduled as work), deduped so a recurring restart is annotated once, not
    every tick. ``engine_latency`` / ``retrieval_degraded`` open a RECOMMENDATION ``:Gap``
    (config/perf) that stays open for a human/the flywheel — NOTHING here mutates prod.

    Can be called standalone (it reads+aggregates the window itself when ``aggregates`` is
    ``None``) or from the analyzer with pre-computed, threshold-crossing aggregates.
    Best-effort; returns ``{recommendations, heals}`` and never raises.
    """
    result = {"recommendations": 0, "heals": 0}
    if engine is None:
        return result
    if aggregates is None:
        recent = runtime_signals.read_recent_runtime_signals(engine, window_s=_WINDOW_S)
        aggregates = [a for a in _aggregate(recent).values() if _crosses_threshold(a)]
    if open_ids is None:
        try:
            open_ids = {str(g.get("id")) for g in open_gaps(engine) if g.get("id")}
        except Exception:  # noqa: BLE001
            open_ids = set()

    for agg in aggregates:
        kind = str(agg.get("kind"))
        disposition = _RECONCILE_DISPOSITION.get(kind)
        if disposition is None:
            continue
        subject = str(agg.get("subject"))

        if disposition == _HEAL:
            # Already auto-healed by the supervisor — record it ONCE as a closed loop.
            statement = (
                f"runtime: listener '{subject}' restarted {agg['count']}× within "
                f"{int(_WINDOW_S // 60)}min — auto-healed by the messaging self-healing "
                f"supervisor (no action needed; recorded for visibility)"
            )
            gap = _submit_runtime_gap(
                engine,
                kind=kind,
                subject=subject,
                statement=statement,
                severity=_SEVERITY.get(kind, _DEFAULT_SEVERITY),
                agg=agg,
                open_ids=open_ids,
                lease=False,  # a resolved heal is not schedulable work
                resolve=True,  # born open then immediately resolved
            )
            if gap:
                result["heals"] += 1
                logger.info(
                    "[runtime-reliability] recorded resolved heal %s (supervisor "
                    "auto-healed %s×)",
                    gap["id"],
                    agg["count"],
                )
            continue

        # _RECOMMEND — open a config/perf recommendation gap (stays open), deduped.
        hint = _RECOMMENDATION_HINT.get(kind, "review this runtime path")
        statement = (
            f"runtime: {kind} on '{subject}' recurred {agg['count']}× within "
            f"{int(_WINDOW_S // 60)}min — {hint}"
        )
        gap = _submit_runtime_gap(
            engine,
            kind=kind,
            subject=subject,
            statement=statement,
            severity=_SEVERITY.get(kind, _DEFAULT_SEVERITY),
            agg=agg,
            open_ids=open_ids,
        )
        if gap:
            result["recommendations"] += 1
            logger.info("[runtime-reliability] opened recommendation gap %s", gap["id"])
    return result


__all__ = [
    "runtime_reliability_analyzer",
    "runtime_reconciler",
]
