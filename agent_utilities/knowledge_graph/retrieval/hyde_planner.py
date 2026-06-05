"""CONCEPT:KG-2.12 — Memory-First Retrieval: HyDE planning & fidelity helpers.

Pure, LLM-free helpers backing ``HybridRetriever.plan_and_retrieve``. Assimilated from Quarq
Agent's retrieval stack (agent-oss/agent.py:1817-2052, 2435, 3211) and made graph-native:

- :class:`HydePlan` — the structured retrieval strategy (multi-vector queries + keywords + mode).
- :func:`threshold_for_mode` — Quarq's dual thresholds (standard 0.38 / deep 0.28).
- :func:`parse_hyde_plan` — defensive JSON parse with single-query fallback (never raises).
- :func:`merge_retrievals` — id-dedup + score-sort merge across multi-query results.
- :func:`build_evidence_ledger` — quantitative-fidelity ACCEPT/REJECT ledger over retrieved nodes.

Keeping these side-effect-free lets the whole memory-first policy be unit-tested without a backend
or an LLM, while the orchestration method stays on the retriever (3-hop Wire-First ceiling).
"""

from __future__ import annotations

import json
import re
from typing import Any, Literal

from pydantic import BaseModel, Field

SearchMode = Literal["standard", "deep"]

# Quarq's dual retrieval thresholds (agent-oss/agent.py:2014). Deep mode casts a wider net
# for aggregations / temporal spans; standard mode is strict point-fact precision.
HYDE_THRESHOLDS: dict[str, float] = {"standard": 0.38, "deep": 0.28}

# Quarq's duplicate-detection ceiling (agent-oss/agent.py:1178,1252) — surfaced for reuse.
DUPLICATE_SIMILARITY: float = 0.95


def threshold_for_mode(mode: str) -> float:
    """Return the relevance threshold for a search mode (defaults to standard 0.38)."""
    return HYDE_THRESHOLDS.get(mode, HYDE_THRESHOLDS["standard"])


# CONCEPT:KG-2.15 — social-closer / triviality gate (memory-os scripts/context_enhancer.py:586).
# Trivial turns skip the whole HyDE plan + retrieval, saving the planner LLM call and embedding work.
SOCIAL_CLOSERS: frozenset[str] = frozenset(
    {
        "hi",
        "hello",
        "hey",
        "yo",
        "sup",
        "ok",
        "okay",
        "k",
        "kk",
        "yes",
        "no",
        "yep",
        "nope",
        "thanks",
        "thank you",
        "thx",
        "ty",
        "great",
        "cool",
        "nice",
        "bye",
        "goodbye",
        "cya",
        "later",
        "np",
        "sure",
        "got it",
        "gotcha",
        "done",
    }
)


def is_trivial_query(text: str) -> bool:
    """True if a message is a social closer / too trivial to warrant retrieval (CONCEPT:KG-2.15).

    Trivial = an exact social-closer phrase, or a sub-6-char ASCII-only message, or emoji/symbol
    only. Such turns carry no retrievable intent, so the HyDE planner + vector search are skipped.
    """
    t = (text or "").strip().lower()
    if not t:
        return True
    if t in SOCIAL_CLOSERS:
        return True
    # Strip trailing punctuation, then re-check social closers (e.g. "ok!", "yep.").
    core = t.rstrip("!.?,;: ")
    if core in SOCIAL_CLOSERS:
        return True
    # Emoji / punctuation-only message (no alphabetic content, short).
    if len(t) <= 4 and not any(c.isalpha() for c in t):
        return True
    return False


class HydePlan(BaseModel):
    """Structured retrieval strategy emitted by the planner role (ORCH-1.27).

    Mirrors Quarq's HyDE plan: several complementary vector formulations (baseline / entity /
    action / literal-unit), exact keywords for direct matching, and a recall-vs-precision mode.
    """

    vector_queries: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    search_mode: SearchMode = "standard"

    def effective_queries(self, original: str) -> list[str]:
        """Non-empty, de-duplicated query list, always including the original as a fallback."""
        seen: dict[str, None] = {}
        for q in [*self.vector_queries, original]:
            q = (q or "").strip()
            if q and q not in seen:
                seen[q] = None
        return list(seen)


def parse_hyde_plan(
    raw: str, *, original_query: str, mode_hint: str | None = None
) -> HydePlan:
    """Parse a planner LLM response into a :class:`HydePlan`, with a safe fallback.

    Accepts a JSON object (optionally embedded in prose / fenced block). On any parse failure
    returns a single-query plan over ``original_query`` at ``mode_hint`` (or standard), so the
    caller always has a usable plan — exactly Quarq's text-fallback posture (agent.py:1988-2006).
    """
    fallback_mode: SearchMode = "deep" if mode_hint == "deep" else "standard"
    try:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            raise ValueError("no JSON object found")
        data = json.loads(match.group())
        # Tolerate Quarq's comma-string keywords as well as a list.
        kw = data.get("keywords", [])
        if isinstance(kw, str):
            kw = [k.strip() for k in kw.split(",") if k.strip()]
        mode = data.get("search_mode", fallback_mode)
        if mode not in HYDE_THRESHOLDS:
            mode = fallback_mode
        plan = HydePlan(
            vector_queries=[
                str(q) for q in data.get("vector_queries", []) if str(q).strip()
            ],
            keywords=[str(k) for k in kw],
            search_mode=mode,
        )
        if not plan.vector_queries:
            plan.vector_queries = [original_query]
        return plan
    except (ValueError, TypeError, json.JSONDecodeError):
        return HydePlan(vector_queries=[original_query], search_mode=fallback_mode)


def merge_retrievals(
    results_lists: list[list[dict[str, Any]]], context_window: int
) -> list[dict[str, Any]]:
    """Merge multi-query retrieval results: dedup by node id, keep max ``_score``, sort desc.

    Mirrors Quarq's ID-dedup + recency/score ordering (agent.py:2035-2052) but preserves the
    graph-native ``_score`` (which already includes backlink boost + positional encodings).
    """
    best: dict[str, dict[str, Any]] = {}
    for rl in results_lists:
        for node in rl:
            nid = node.get("id")
            if nid is None:
                continue
            prev = best.get(nid)
            if prev is None or node.get("_score", 0.0) > prev.get("_score", 0.0):
                best[nid] = node
    merged = sorted(best.values(), key=lambda n: n.get("_score", 0.0), reverse=True)
    return merged[:context_window]


def build_evidence_ledger(query: str, nodes: list[dict[str, Any]]) -> dict[str, Any]:
    """Build a quantitative-fidelity ACCEPT/REJECT evidence ledger over retrieved nodes.

    CONCEPT:KG-2.12 — assimilates Quarq's evidence table (agent.py:2435, 2478-2518). Each node
    becomes a ledger row scored against the query; rows at/above the standard threshold are
    ACCEPTed, the rest REJECTed (near-miss / sibling category). Numeric tokens in accepted
    content are surfaced so a generator can attribute actor/action/event/exactness and aggregate
    a *complete ledger* rather than answering from the single most salient row.

    Pure and LLM-free: the structured ledger is what a generation prompt consumes.
    """
    accept_floor = HYDE_THRESHOLDS["standard"]
    num_re = re.compile(r"(?<![\w$])\$?\d[\d,]*(?:\.\d+)?")
    rows: list[dict[str, Any]] = []
    for rank, node in enumerate(nodes):
        score = float(node.get("_score", 0.0))
        content = str(node.get("content") or node.get("name") or "")
        decision = "ACCEPT" if score >= accept_floor else "REJECT"
        rows.append(
            {
                "rank": rank,
                "id": node.get("id"),
                "score": round(score, 4),
                "event_time": node.get("event_time"),
                "decision": decision,
                "reason": "above-threshold" if decision == "ACCEPT" else "near-miss",
                "numbers": num_re.findall(content),
                "content": content[:280],
            }
        )
    accepted = [r for r in rows if r["decision"] == "ACCEPT"]
    return {
        "query": query,
        "rows": rows,
        "accepted_ids": [r["id"] for r in accepted],
        "accept_count": len(accepted),
        "reject_count": len(rows) - len(accepted),
        # A complete-ledger hint: every numeric token across ACCEPT rows, for aggregation.
        "accepted_numbers": [n for r in accepted for n in r["numbers"]],
    }
