#!/usr/bin/python
from __future__ import annotations

"""Apply learned/asserted governance rules at retrieval time (CONCEPT:EG-KG.storage.nonblocking-checkpoint).

This is the missing link that makes corrections-turned-rules *change behaviour*.
Synthesized preferences/principles and human-asserted voice/source rules are
stored as graph nodes; here they are loaded and used to **filter or re-rank**
designations so the brain stops repeating a corrected mistake.

A rule is a plain dict::

    {"kind": "forbid"|"prefer"|"demote", "target": "<id-or-substring>",
     "weight": 0.2, "reason": "...", "capability": "<optional cap>"}

``forbid`` drops matching designations; ``prefer``/``demote`` nudge their score.
The function is pure and side-effect free; loading is best-effort and tolerant.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)

_RULE_NODE_TYPES = ("voice_rule", "source_rule", "governance_rule", "preference")


def _matches(designation: Any, rule: dict[str, Any]) -> bool:
    target = str(rule.get("target", "")).strip()
    cap = str(rule.get("capability", "")).strip()
    did = str(getattr(designation, "id", ""))
    if target and target in did:
        return True
    if cap and cap in {str(c) for c in getattr(designation, "capabilities", set())}:
        return True
    return False


def apply_governance_rules(
    designations: list[Any], rules: list[dict[str, Any]] | None
) -> list[Any]:
    """Filter/re-rank ``designations`` against ``rules`` (returns a new list)."""
    if not rules or not designations:
        return designations
    kept: list[Any] = []
    for d in designations:
        forbidden = False
        delta = 0.0
        for rule in rules:
            if not _matches(d, rule):
                continue
            kind = str(rule.get("kind", "")).lower()
            weight = float(rule.get("weight", 0.2))
            if kind == "forbid":
                forbidden = True
                break
            if kind == "prefer":
                delta += weight
            elif kind == "demote":
                delta -= weight
        if forbidden:
            continue
        if delta:
            try:
                d.score = float(getattr(d, "score", 0.0)) + delta
            except Exception:  # pragma: no cover - score is always numeric
                pass
        kept.append(d)
    kept.sort(key=lambda x: getattr(x, "score", 0.0), reverse=True)
    return kept


def load_active_rules(store: Any) -> list[dict[str, Any]]:
    """Best-effort load of active governance-rule nodes from the graph.

    Returns ``[]`` on any failure (rules are an enhancement, never a hard
    dependency of retrieval).
    """
    if store is None or not hasattr(store, "execute"):
        return []
    rules: list[dict[str, Any]] = []
    try:
        types = ", ".join(f"'{t}'" for t in _RULE_NODE_TYPES)
        rows = store.execute(
            f"MATCH (r) WHERE r.type IN [{types}] AND r.active = true "
            "RETURN r.kind AS kind, r.target AS target, r.weight AS weight, "
            "r.capability AS capability, r.reason AS reason"
        )
        for row in rows or []:
            if isinstance(row, dict) and row.get("kind") and row.get("target"):
                rules.append(row)
    except Exception as exc:  # pragma: no cover - dialect/availability tolerant
        logger.debug("load_active_rules failed (non-fatal): %s", exc)
    return rules
