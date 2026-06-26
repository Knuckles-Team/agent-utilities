#!/usr/bin/python
from __future__ import annotations

"""Moat trace analytics (CONCEPT:KG-2.257) — the observability queries Opik CANNOT do.

Because traces, online-scores, assertion verdicts, generations, and prompt versions are
all FIRST-CLASS KG nodes (not opaque ClickHouse rows), the engine can graph-reason over
them. Three queries surface the differentiator:

* ``trace_rootcause`` — every FAILED assertion / low online-score with its trace's agent,
  grouped by agent — "what is failing and where does it come from".
* ``prompt_regression`` — mean online-score per prompt version (which prompt version
  scores worse), via GenerationNode.prompt_version_id → trace → scores.
* ``failure_cluster`` — failing traces clustered by the assertion that failed (systemic
  breaks ≥ N agents share), the pile-attack signal for triage.

All read via ``backend.execute(<cypher>)`` (the same path the eval corpus uses), matching
on the node ``type`` property. Degrade to empty results when no backend/query is available.
"""

import logging
from collections import Counter, defaultdict
from typing import Any, cast

logger = logging.getLogger(__name__)


def _rows(backend: Any, cypher: str) -> list[dict[str, Any]]:
    if backend is None or not hasattr(backend, "execute"):
        return []
    try:
        return backend.execute(cypher) or []
    except Exception as exc:  # pragma: no cover - dialect tolerant
        logger.debug("trace-analytics query failed: %s", exc)
        return []


def _trace_agents(backend: Any) -> dict[str, dict[str, Any]]:
    rows = _rows(
        backend,
        "MATCH (t) WHERE t.type = 'trace' "
        "RETURN t.id AS id, t.agent AS agent, t.status AS status, t.name AS name",
    )
    return {str(r.get("id")): r for r in rows if r.get("id")}


def trace_rootcause(
    backend: Any, capability: str = "", top_k: int = 20
) -> dict[str, Any]:
    """FAILED assertions + low online-scores joined to their trace's agent (KG-2.257)."""
    traces = _trace_agents(backend)
    fails = _rows(
        backend,
        "MATCH (a) WHERE a.type = 'assertion_result' AND a.status = 'failed' "
        "RETURN a.trace_id AS trace_id, a.assertion AS assertion, a.reasoning AS reasoning",
    )
    lows = _rows(
        backend,
        "MATCH (s) WHERE s.type = 'online_score' AND s.score < 0.5 "
        "RETURN s.trace_id AS trace_id, s.dimension AS dimension, s.score AS score",
    )
    findings: list[dict[str, Any]] = []
    for f in fails:
        t = traces.get(str(f.get("trace_id")), {})
        findings.append(
            {
                "trace_id": f.get("trace_id"),
                "agent": t.get("agent") or t.get("name") or "?",
                "kind": "assertion_failed",
                "detail": f.get("assertion"),
                "reasoning": f.get("reasoning"),
            }
        )
    for s in lows:
        t = traces.get(str(s.get("trace_id")), {})
        findings.append(
            {
                "trace_id": s.get("trace_id"),
                "agent": t.get("agent") or t.get("name") or "?",
                "kind": "low_score",
                "detail": f"{s.get('dimension')}={s.get('score')}",
            }
        )
    if capability:
        findings = [
            f for f in findings if capability.lower() in str(f["agent"]).lower()
        ]
    by_agent = Counter(f["agent"] for f in findings)
    return {
        "findings": findings[:top_k],
        "by_agent": dict(by_agent.most_common()),
        "total": len(findings),
    }


def prompt_regression(backend: Any, top_k: int = 20) -> dict[str, Any]:
    """Mean online-score per prompt version (which version regressed) (KG-2.257)."""
    gens = _rows(
        backend,
        "MATCH (g) WHERE g.type = 'generation' AND g.prompt_version_id <> '' "
        "RETURN g.prompt_version_id AS pv, g.trace_id AS trace_id",
    )
    scores = _rows(
        backend,
        "MATCH (s) WHERE s.type = 'online_score' "
        "RETURN s.trace_id AS trace_id, s.score AS score",
    )
    score_by_trace: dict[str, list[float]] = defaultdict(list)
    for s in scores:
        tid, sc = s.get("trace_id"), s.get("score")
        if tid is None or sc is None:
            continue
        try:
            score_by_trace[str(tid)].append(float(sc))
        except (TypeError, ValueError):
            continue
    per_version: dict[str, list[float]] = defaultdict(list)
    for g in gens:
        per_version[str(g.get("pv"))].extend(score_by_trace.get(str(g.get("trace_id")), []))
    summary = {
        pv: {"mean_score": round(sum(v) / len(v), 4), "n": len(v)}
        for pv, v in per_version.items()
        if v
    }
    ranked = sorted(summary.items(), key=lambda kv: kv[1]["mean_score"])
    return {
        "by_version": dict(ranked[:top_k]),
        "worst": ranked[0][0] if ranked else None,
    }


def failure_cluster(
    backend: Any, min_agents: int = 1, top_k: int = 20
) -> dict[str, Any]:
    """Failing traces clustered by the assertion that failed — systemic breaks (KG-2.257)."""
    traces = _trace_agents(backend)
    fails = _rows(
        backend,
        "MATCH (a) WHERE a.type = 'assertion_result' AND a.status = 'failed' "
        "RETURN a.trace_id AS trace_id, a.assertion AS assertion",
    )
    clusters: dict[str, set[str]] = defaultdict(set)
    counts: Counter[str] = Counter()
    for f in fails:
        key = f.get("assertion") or "?"
        counts[key] += 1
        t = traces.get(str(f.get("trace_id")), {})
        clusters[key].add(str(t.get("agent") or t.get("name") or f.get("trace_id")))
    out = [
        {"assertion": k, "failures": counts[k], "agents": sorted(clusters[k])}
        for k in counts
        if len(clusters[k]) >= min_agents
    ]
    out.sort(key=lambda c: cast(int, c["failures"]), reverse=True)
    return {"clusters": out[:top_k], "total_failing": sum(counts.values())}


__all__ = ["trace_rootcause", "prompt_regression", "failure_cluster"]
