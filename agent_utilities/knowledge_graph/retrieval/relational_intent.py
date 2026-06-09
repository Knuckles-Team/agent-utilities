#!/usr/bin/python
from __future__ import annotations

"""Deterministic relational-intent retrieval.

CONCEPT:KG-2.34 — Relational-Intent Retrieval

Mirrors gbrain's ``relational-recall.ts``: parse a natural-language relational
question — "which papers *support* transformers", "what *contradicts* X" — with
**regex only (zero LLM)**, resolve the seed entity, and walk the typed-edge graph to
return the related nodes directly. For non-relational queries the parser returns
``None`` so the arm is a strict no-op and never regresses ordinary retrieval.

The verb→edge-type vocabulary is supplied by the active Schema Pack
(``SchemaPack.relational_verbs``), so the same machinery serves a VC brain
(``invested_in``/``founded``) or a research brain (``supports``/``contradicts``)
without code changes. Edge types are validated against ``RegistryEdgeType`` before
they touch a query, so a pack typo can never inject Cypher.
"""


import logging
import re
from dataclasses import dataclass
from typing import Any, Literal

from agent_utilities.models.knowledge_graph import RegistryEdgeType

logger = logging.getLogger(__name__)

_VALID_EDGE_VALUES = {e.value for e in RegistryEdgeType}

# A query is only considered relational if it opens with one of these interrogative
# leads (or contains "that <verb>"); this keeps the arm a strict no-op for ordinary
# semantic queries like "summarize the dataset".
_INTERROGATIVE_LEAD = re.compile(
    r"^\s*(which|what|who|whose|find|show|list|papers?|works?|nodes?|entities|things?)\b",
    re.IGNORECASE,
)
# Inverse-direction marker: "... is <verb> by <seed>" / "<verb>ed by <seed>".
_INVERSE_BY = re.compile(r"\bby\s+(?P<seed>.+?)\s*\??$", re.IGNORECASE)


@dataclass
class RelationalQuery:
    """A parsed relational query (CONCEPT:KG-2.34)."""

    verb_edge: str
    direction: Literal["out", "in"]
    seed_text: str


def _strip_trailing(seed: str) -> str:
    return seed.strip().rstrip("?").strip()


def _find_verb(
    tokens: list[str], relational_verbs: dict[str, str]
) -> tuple[int, int, str] | None:
    """Find the longest registered verb phrase in ``tokens``.

    Returns ``(start_index, length_in_tokens, edge_type)`` for the longest verb
    phrase that appears as a contiguous token run, or ``None``. Longer phrases win
    so multi-word verbs ("invested in") beat single tokens ("invested").
    """
    lowered = [t.lower().strip(".,:;") for t in tokens]
    best: tuple[int, int, str] | None = None
    for verb, edge in relational_verbs.items():
        vparts = verb.lower().split()
        if not vparts:
            continue
        n = len(vparts)
        for i in range(len(lowered) - n + 1):
            if lowered[i : i + n] == vparts:
                if best is None or n > best[1]:
                    best = (i, n, edge)
    return best


def parse_relational_intent(
    query: str, relational_verbs: dict[str, str]
) -> RelationalQuery | None:
    """Parse a relational query, or return ``None`` for non-relational queries.

    Strategy (deterministic, zero-LLM): require an interrogative lead, locate the
    longest registered verb phrase, and take everything after it as the seed. An
    ``is <verb> by <seed>`` / ``<verb> by <seed>`` shape flips the traversal
    direction to ``in`` (the inverse edge).

    Args:
        query: The raw user query string.
        relational_verbs: Pack vocabulary mapping NL verb phrases to edge-type values.

    Returns:
        A ``RelationalQuery`` when the query is relational and its verb maps to a
        *valid* edge type; otherwise ``None`` (no-op guarantee).
    """
    if not relational_verbs or not query.strip():
        return None
    if not _INTERROGATIVE_LEAD.match(query):
        return None

    tokens = query.split()
    found = _find_verb(tokens, relational_verbs)
    if found is None:
        return None
    start, length, edge = found
    if edge not in _VALID_EDGE_VALUES:
        return None

    after = " ".join(tokens[start + length :]).strip()
    # Inverse direction: "is cited by X" / "cited by X" -> walk the edge backwards.
    inv = _INVERSE_BY.match(after) if after.lower().startswith("by ") else None
    if inv:
        seed = _strip_trailing(inv.group("seed"))
        direction: Literal["out", "in"] = "in"
    else:
        seed = _strip_trailing(after)
        direction = "out"
    if not seed:
        return None
    return RelationalQuery(edge, direction, seed)


def traverse(engine: Any, rq: RelationalQuery, top_k: int = 10) -> list[dict[str, Any]]:
    """Resolve the seed and walk the typed edge, returning related node dicts.

    Precision-first: returns ``[]`` unless the seed resolves *and* at least one
    typed edge is found (matching gbrain's relational-recall behaviour). Degrades
    to ``[]`` on any backend error so it can never corrupt the main retrieval arm.
    """
    if engine is None or rq.verb_edge not in _VALID_EDGE_VALUES:
        return []
    try:
        seeds = engine._search_keyword(rq.seed_text, top_k=3)
    except Exception as e:  # pragma: no cover - defensive
        logger.debug("relational seed resolution failed: %s", e)
        return []
    if not seeds:
        return []
    seed_id = seeds[0].get("id")
    if not seed_id or not getattr(engine, "backend", None):
        return []

    # Validated, fixed-vocabulary edge type — safe to inline as a relationship label.
    edge = rq.verb_edge
    if rq.direction == "out":
        cypher = f"MATCH (s {{id: $id}})-[:`{edge}`]->(t) RETURN t.id AS id, t AS data LIMIT {int(top_k)}"
    else:
        cypher = f"MATCH (s)-[:`{edge}`]->(t {{id: $id}}) RETURN s.id AS id, s AS data LIMIT {int(top_k)}"

    try:
        rows = engine.backend.execute(cypher, {"id": seed_id})
    except Exception as e:
        logger.debug("relational traversal failed (%s): %s", edge, e)
        return []

    out: list[dict[str, Any]] = []
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        data = row.get("data")
        node = dict(data) if isinstance(data, dict) else {}
        node["id"] = row.get("id") or node.get("id")
        if not node.get("id"):
            continue
        node["_score"] = node.get("_score", 1.0)
        node["_relational_hit"] = edge
        node["evidence"] = "relational"
        out.append(node)
    return out


__all__ = ["RelationalQuery", "parse_relational_intent", "traverse"]
