#!/usr/bin/python
from __future__ import annotations

"""Closed-loop agent mining over tool-call provenance (workstream C6).

CONCEPT:AU-KG.evolution.insight-engine-closed-loop — closed-loop agent mining

Mines the KG's agent-run provenance for REPEATED FAILURE tool-call sequences —
"whenever the agent hits this same tool-call shape, the run tends to fail" —
and feeds each mined pattern through the EXISTING C4 CandidateInsight→Claim→
Validation→Action-gate pipeline (:mod:`.candidate_insight`,
:mod:`.promotion_governance`, :mod:`~agent_utilities.orchestration.action_policy`)
so it is reviewed with the SAME rigor as an association rule or an anomaly,
rather than a fourth bespoke mining path.

The source is the same normalized ontology the live runtime writes:
``RunTrace -[:USED_TOOL]-> ToolCall`` and
``RunTrace -[:PRODUCED_OUTCOME]-> OutcomeEvaluation``.  A numeric
``event_sequence`` cursor makes each pass incremental and restartable; this
module never falls back to timestamp or lexical cursor ordering.

Mining itself is delegated to the engine's ``graph_mine`` "sequence" action
(frequent ORDERED subsequences) through the SAME ``_invoke`` helper every
other mining pass in this codebase uses
(:func:`agent_utilities.mcp.tools.engine_surface_tools._invoke`) — so it
degrades exactly like ``loop_controller._mine_association_rules`` et al. on a
no-mining engine build (empty result, never raises), rather than
reimplementing frequency counting in Python.
"""

import logging
from typing import Any

from agent_utilities.observability.trace_ontology import (
    TRACE_PRODUCED_OUTCOME_EDGE,
    TRACE_USED_TOOL_EDGE,
    TraceCursor,
)

logger = logging.getLogger(__name__)

__all__ = ["gather_failure_tool_sequences", "mine_trace_patterns"]

#: A run below this reward counts as a FAILURE for this mining pass — mirrors
#: ``engine_ahe.run_self_improvement_cycle``'s own "o.reward < 0.5" threshold
#: for the identical RunTrace/OutcomeEvaluation schema (kept in sync
#: deliberately, not re-derived).
FAILURE_REWARD_THRESHOLD = 0.5

#: Cap on failure traces scanned per cycle — keeps the query bounded like
#: every other mining pass's row LIMIT.
_TRACE_SCAN_LIMIT = 200

#: Minimum fraction of failure traces a tool-call subsequence must appear in
#: to be reported as a pattern (passed straight to ``graph_mine``'s own
#: ``min_support`` — the SAME units as the mining surface itself).
_MIN_SUPPORT = 0.3


def gather_failure_tool_sequences(
    engine: Any,
    *,
    reward_threshold: float = FAILURE_REWARD_THRESHOLD,
    limit: int = _TRACE_SCAN_LIMIT,
    after_sequence: int = 0,
) -> tuple[list[str], list[list[str]]]:
    """Query failed traces' ordered tool-call sequences after a durable cursor.

    Returns ``(trace_ids, sequences)`` — parallel lists, one entry per
    failed trace that used at least one tool. Empty (never raises) when the
    engine is unavailable or the query fails, matching every other mining
    pass's query-failure tolerance in this codebase.
    """
    ids, sequences, _ = _gather_failure_tool_sequences(
        engine,
        reward_threshold=reward_threshold,
        limit=limit,
        after_sequence=after_sequence,
    )
    return ids, sequences


def _gather_failure_tool_sequences(
    engine: Any,
    *,
    reward_threshold: float,
    limit: int,
    after_sequence: int,
) -> tuple[list[str], list[list[str]], TraceCursor]:
    if engine is None:
        return [], [], TraceCursor(after_sequence)
    try:
        rows = (
            engine.query_cypher(
                f"MATCH (r:RunTrace)-[:{TRACE_PRODUCED_OUTCOME_EDGE}]->(o:OutcomeEvaluation) "
                "WHERE o.reward < $threshold AND r.event_sequence > $after_sequence "
                f"MATCH (r)-[:{TRACE_USED_TOOL_EDGE}]->(t:ToolCall) "
                "RETURN r.id AS trace_id, r.event_sequence AS event_sequence, "
                "t.tool_name AS tool_name, t.sequence AS tool_sequence "
                "ORDER BY event_sequence, trace_id, tool_sequence "
                f"LIMIT {int(limit)}",
                {
                    "threshold": reward_threshold,
                    "after_sequence": int(after_sequence),
                },
            )
            or []
        )
    except Exception as e:  # noqa: BLE001 — a query failure degrades, never raises
        logger.debug("trace_pattern_miner: failure-sequence query failed: %s", e)
        return [], [], TraceCursor(after_sequence)

    # LIMIT counts tool-call rows, not traces. If the page is full, its last
    # trace may be partial; defer that whole trace so advancing the cursor can
    # never skip its remaining calls on the next pass.
    if len(rows) >= int(limit) and rows:
        partial_trace = rows[-1].get("trace_id") if isinstance(rows[-1], dict) else None
        if partial_trace:
            rows = [
                row
                for row in rows
                if not isinstance(row, dict) or row.get("trace_id") != partial_trace
            ]

    ordered_ids: list[str] = []
    by_trace: dict[str, list[str]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        trace_id = row.get("trace_id")
        tool_name = row.get("tool_name")
        if not trace_id or not tool_name:
            continue
        if trace_id not in by_trace:
            by_trace[trace_id] = []
            ordered_ids.append(trace_id)
        by_trace[trace_id].append(str(tool_name))

    sequences = [by_trace[tid] for tid in ordered_ids if len(by_trace[tid]) >= 2]
    trace_ids = [tid for tid in ordered_ids if len(by_trace[tid]) >= 2]
    cursor = TraceCursor.from_rows(rows)
    return trace_ids, sequences, max(cursor, TraceCursor(after_sequence))


def mine_trace_patterns(
    engine: Any,
    *,
    reward_threshold: float = FAILURE_REWARD_THRESHOLD,
    min_support: float = _MIN_SUPPORT,
    limit: int = _TRACE_SCAN_LIMIT,
    after_sequence: int = 0,
) -> dict[str, Any]:
    """Mine repeated FAILURE tool-call sequences via the engine's ``graph_mine`` surface.

    Returns a compact, JSON-able summary shaped like
    ``loop_controller._run_mine_discovery``'s per-substep results —
    ``{"patterns": {...}, "failure_traces": n, "errors": [...]}`` — where
    ``patterns`` is the raw ``graph_mine action="sequence"`` result
    (``{"patterns": [{"items", "support", "count"}], ...}``), ready for
    :func:`~.candidate_insight.candidates_from_sequential_patterns`. Never
    raises: an unreachable engine, a query failure, or a no-mining engine
    build all degrade to an empty ``patterns`` result.
    """
    import json as _json

    errors: list[str] = []
    trace_ids, sequences, cursor = _gather_failure_tool_sequences(
        engine,
        reward_threshold=reward_threshold,
        limit=limit,
        after_sequence=after_sequence,
    )
    if not sequences:
        return {
            "patterns": {"patterns": []},
            "failure_traces": len(trace_ids),
            "sequences_mined": 0,
            "next_event_sequence": cursor.event_sequence,
            "errors": errors,
        }

    try:
        from agent_utilities.mcp.tools.engine_surface_tools import _invoke

        raw = _invoke(
            surface="mining",
            action="sequence",
            graph="",
            candidates=(("mining", "sequence"),),
            params={
                "sequences": sequences,
                "min_support": min_support,
                "writeback": True,
            },
        )
        payload = _json.loads(raw)
    except Exception as e:  # noqa: BLE001 — never let mining break the cycle
        errors.append(f"trace_pattern_miner:invoke: {e}")
        return {
            "patterns": {"patterns": []},
            "failure_traces": len(trace_ids),
            "sequences_mined": len(sequences),
            "next_event_sequence": cursor.event_sequence,
            "errors": errors,
        }

    if not (isinstance(payload, dict) and "error" not in payload):
        errors.append(f"trace_pattern_miner: {payload.get('error') or payload}")
        return {
            "patterns": {"patterns": []},
            "failure_traces": len(trace_ids),
            "sequences_mined": len(sequences),
            "next_event_sequence": cursor.event_sequence,
            "errors": errors,
        }

    result = payload.get("result") or {}
    return {
        "patterns": result,
        "failure_traces": len(trace_ids),
        "sequences_mined": len(sequences),
        "next_event_sequence": cursor.event_sequence,
        "errors": errors,
    }
