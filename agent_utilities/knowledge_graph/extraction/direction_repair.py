"""Relation-direction repair via ``rdfs:domain``/``rdfs:range`` (CONCEPT:AU-KG.enrichment.direction-repair).

Closes sift-kg's schema-driven direction repair (``postprocessor.py:346
fix_relation_directions``) using our OWL domain/range constraints — which the
ontology already declares and the engine already reasons over
(``reasoning.rs infer_domain_range``, the ``OwlReason`` op). This is a Wire-First
win: pure Python over the :class:`~.extraction_schema.ExtractionSchema` + the
grounding types, **no new engine op**.

Behaviour per extracted edge, given the predicate's declared ``domain → range``:

* **forward-ok** — the grounded subject/object types already satisfy the
  constraint → unchanged.
* **reversed** — they satisfy the constraint only when flipped → swap
  subject/object (and the grounding types with them) so arrows point the
  ontology-declared way.
* **symmetric** (``owl:SymmetricProperty``) — never flipped.
* **violation** — neither orientation satisfies domain/range → the edge is
  **flagged** (a ``domain_range_violation`` tag + damped confidence) and left for
  the existing SHACL contradiction shape / engine contradiction machinery
  (KG-2.251/2.252) to record, rather than silently dropped.

Conservative by construction: repair fires only when the predicate is typed in
the schema AND both endpoints grounded to a known type. Untyped predicates or
endpoints pass through unchanged.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .extraction_schema import ExtractionSchema

logger = logging.getLogger(__name__)

# Confidence multiplier applied to an edge whose direction violates domain/range
# but which we keep (down-weight so corroboration/SHACL can act on it).
_VIOLATION_CONFIDENCE_FACTOR = 0.5
_VIOLATION_TAG = "domain_range_violation"


def repair_direction(
    fact: Any,
    grounding: dict[str, Any],
    rel_by_pred: dict[str, Any],
) -> tuple[Any, dict[str, Any], str]:
    """Repair one edge's direction in place; return ``(fact, grounding, status)``.

    ``rel_by_pred`` maps a snake_case predicate to its
    :class:`~.extraction_schema.Relation` (build once via
    ``schema.relations_by_predicate()``). ``status`` is one of ``ok``,
    ``swapped``, ``violation``, ``symmetric``, ``untyped``.
    """
    rel = rel_by_pred.get(getattr(fact, "predicate", ""))
    if rel is None:
        return fact, grounding, "untyped"
    if getattr(rel, "symmetric", False):
        return fact, grounding, "symmetric"

    s_type = str(grounding.get("subject_type") or "").lower()
    o_type = str(grounding.get("object_type") or "").lower()
    if not s_type or not o_type:
        return fact, grounding, "untyped"

    domain = {d.lower() for d in rel.domain}
    rng = {r.lower() for r in rel.range}
    if not domain or not rng:
        # predicate declares no class-level domain AND range (e.g. range is a
        # literal xsd type) — nothing to orient against.
        return fact, grounding, "untyped"

    forward_ok = s_type in domain and o_type in rng
    if forward_ok:
        return fact, grounding, "ok"

    reverse_ok = o_type in domain and s_type in rng
    if reverse_ok:
        # Flip the edge so subject matches the declared domain, and swap the
        # grounding types with it so downstream annotation stays correct.
        fact.subject, fact.object = fact.object, fact.subject
        grounding = {
            **grounding,
            "subject_type": grounding.get("object_type"),
            "object_type": grounding.get("subject_type"),
        }
        return fact, grounding, "swapped"

    # Neither orientation satisfies the constraint → flag, don't drop.
    try:
        fact.confidence = int(
            max(
                0,
                min(
                    100,
                    round(
                        getattr(fact, "confidence", 0) * _VIOLATION_CONFIDENCE_FACTOR
                    ),
                ),
            )
        )
        tags = list(getattr(fact, "tags", []) or [])
        if _VIOLATION_TAG not in tags:
            tags.append(_VIOLATION_TAG)
            fact.tags = tags
    except Exception:  # noqa: BLE001 — flagging never breaks ingest
        pass
    return fact, grounding, "violation"


def repair_directions(
    grounded: list[tuple[Any, dict[str, Any]]],
    schema: ExtractionSchema | None,
) -> tuple[list[tuple[Any, dict[str, Any]]], dict[str, int]]:
    """Batch-repair a list of ``(fact, grounding)`` against ``schema``.

    Returns the repaired ``(fact, grounding)`` list (same length, possibly
    swapped) plus a ``{status: count}`` tally for observability. With no schema
    (or an empty one) the input is returned unchanged.
    """
    tally: dict[str, int] = {}
    if schema is None or schema.is_empty:
        return grounded, tally
    rel_by_pred = schema.relations_by_predicate()
    out: list[tuple[Any, dict[str, Any]]] = []
    for fact, grounding in grounded:
        fact, grounding, status = repair_direction(fact, grounding, rel_by_pred)
        tally[status] = tally.get(status, 0) + 1
        out.append((fact, grounding))
    if tally.get("swapped") or tally.get("violation"):
        logger.debug("direction_repair: %s", tally)
    return out, tally


__all__ = ["repair_direction", "repair_directions"]
