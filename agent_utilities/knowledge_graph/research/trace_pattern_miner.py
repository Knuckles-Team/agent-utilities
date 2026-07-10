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

Provenance schema note: the durable Agent-OS objects this workstream's task
description names (``AgentTaskNode``/"AgentOutcomeNode") do not yet carry a
tool-call-sequence edge in this schema — the REAL, already-wired provenance
chain for "what tools did a run use, and did it succeed" is the
self-improvement harness's ``Episode -[:PRODUCED_OUTCOME]-> OutcomeEvaluation``
+ ``Episode -[:USED_TOOL]-> ToolCall`` schema (see
``knowledge_graph/orchestration/engine_ahe.py::propose_new_skill_from_experience``,
which mines the SUCCESS side of this same chain in-process). This module mines
the FAILURE side of that identical, real schema — the defensible mapping of
"AgentTask/AgentOutcome" onto this codebase's actual provenance nodes, mirroring
how ``loop_controller._run_mine_discovery`` documents its own schema
simplifications rather than inventing an unwired edge.

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

logger = logging.getLogger(__name__)

__all__ = ["gather_failure_tool_sequences", "mine_trace_patterns"]

#: A run below this reward counts as a FAILURE for this mining pass — mirrors
#: ``engine_ahe.run_self_improvement_cycle``'s own "o.reward < 0.5" threshold
#: for the identical Episode/OutcomeEvaluation schema (kept in sync
#: deliberately, not re-derived).
FAILURE_REWARD_THRESHOLD = 0.5

#: Cap on failure episodes scanned per cycle — keeps the query bounded like
#: every other mining pass's row LIMIT.
_TRACE_SCAN_LIMIT = 200

#: Minimum fraction of failure episodes a tool-call subsequence must appear in
#: to be reported as a pattern (passed straight to ``graph_mine``'s own
#: ``min_support`` — the SAME units as the mining surface itself).
_MIN_SUPPORT = 0.3


def gather_failure_tool_sequences(
    engine: Any,
    *,
    reward_threshold: float = FAILURE_REWARD_THRESHOLD,
    limit: int = _TRACE_SCAN_LIMIT,
) -> tuple[list[str], list[list[str]]]:
    """Query FAILED episodes' ordered tool-call sequences.

    Returns ``(episode_ids, sequences)`` — parallel lists, one entry per
    failed episode that used at least one tool. Empty (never raises) when the
    engine is unavailable or the query fails, matching every other mining
    pass's query-failure tolerance in this codebase.
    """
    if engine is None:
        return [], []
    try:
        rows = (
            engine.query_cypher(
                "MATCH (e:Episode)-[:PRODUCED_OUTCOME]->(o:OutcomeEvaluation) "
                "WHERE o.reward < $threshold "
                "MATCH (e)-[:USED_TOOL]->(t:ToolCall) "
                "RETURN e.id AS episode_id, t.tool_name AS tool_name, "
                "t.timestamp AS ts ORDER BY episode_id, ts "
                f"LIMIT {int(limit)}",
                {"threshold": reward_threshold},
            )
            or []
        )
    except Exception as e:  # noqa: BLE001 — a query failure degrades, never raises
        logger.debug("trace_pattern_miner: failure-sequence query failed: %s", e)
        return [], []

    ordered_ids: list[str] = []
    by_episode: dict[str, list[str]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        ep_id = row.get("episode_id")
        tool_name = row.get("tool_name")
        if not ep_id or not tool_name:
            continue
        if ep_id not in by_episode:
            by_episode[ep_id] = []
            ordered_ids.append(ep_id)
        by_episode[ep_id].append(str(tool_name))

    sequences = [by_episode[eid] for eid in ordered_ids if len(by_episode[eid]) >= 2]
    episode_ids = [eid for eid in ordered_ids if len(by_episode[eid]) >= 2]
    return episode_ids, sequences


def mine_trace_patterns(
    engine: Any,
    *,
    reward_threshold: float = FAILURE_REWARD_THRESHOLD,
    min_support: float = _MIN_SUPPORT,
    limit: int = _TRACE_SCAN_LIMIT,
) -> dict[str, Any]:
    """Mine repeated FAILURE tool-call sequences via the engine's ``graph_mine`` surface.

    Returns a compact, JSON-able summary shaped like
    ``loop_controller._run_mine_discovery``'s per-substep results —
    ``{"patterns": {...}, "failure_episodes": n, "errors": [...]}`` — where
    ``patterns`` is the raw ``graph_mine action="sequence"`` result
    (``{"patterns": [{"items", "support", "count"}], ...}``), ready for
    :func:`~.candidate_insight.candidates_from_sequential_patterns`. Never
    raises: an unreachable engine, a query failure, or a no-mining engine
    build all degrade to an empty ``patterns`` result.
    """
    import json as _json

    errors: list[str] = []
    episode_ids, sequences = gather_failure_tool_sequences(
        engine, reward_threshold=reward_threshold, limit=limit
    )
    if not sequences:
        return {
            "patterns": {"patterns": []},
            "failure_episodes": len(episode_ids),
            "sequences_mined": 0,
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
            "failure_episodes": len(episode_ids),
            "sequences_mined": len(sequences),
            "errors": errors,
        }

    if not (isinstance(payload, dict) and "error" not in payload):
        errors.append(f"trace_pattern_miner: {payload.get('error') or payload}")
        return {
            "patterns": {"patterns": []},
            "failure_episodes": len(episode_ids),
            "sequences_mined": len(sequences),
            "errors": errors,
        }

    result = payload.get("result") or {}
    return {
        "patterns": result,
        "failure_episodes": len(episode_ids),
        "sequences_mined": len(sequences),
        "errors": errors,
    }
