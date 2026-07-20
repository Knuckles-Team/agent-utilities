from __future__ import annotations

"""Deterministic fact-extraction metrics for native program optimization.

The engine-owned optimizer consumes document training references. These helpers
provide the label-free deduplication and canonical-name signals used to evaluate
candidate extraction programs without introducing a Python optimizer stack.
"""

import re
from collections.abc import Callable, Sequence
from typing import Any

_WS = re.compile(r"\s+")


def _norm_entity(name: str) -> str:
    """Return the canonical comparison form of an entity surface string."""
    return _WS.sub(" ", str(name).strip().lower())


def canonical_consistency(facts: Sequence[Any]) -> float:
    """Return the fraction of entity mentions with one canonical surface form."""
    forms: dict[str, set[str]] = {}
    for fact in facts:
        for key in ("subject", "object"):
            raw = fact.get(key) if isinstance(fact, dict) else getattr(fact, key, "")
            if raw:
                forms.setdefault(_norm_entity(raw), set()).add(str(raw))
    if not forms:
        return 1.0
    fragmented = sum(1 for values in forms.values() if len(values) > 1)
    return 1.0 - fragmented / len(forms)


def extraction_quality(
    facts: Sequence[Any],
    *,
    embed_fn: Callable[[str], list[float]] | None = None,
    dedup_threshold: float = 0.90,
    dedup_weight: float = 0.6,
) -> dict[str, float]:
    """Return deduplication and canonical-consistency quality signals."""
    from agent_utilities.knowledge_graph.extraction.fact_extractor import (
        ExtractedFact,
        FactDeduper,
    )

    fact_objects: list[ExtractedFact] = []
    for fact in facts:
        if isinstance(fact, ExtractedFact):
            fact_objects.append(fact)
        elif isinstance(fact, dict):
            subject = str(fact.get("subject") or "")
            object_value = str(fact.get("object") or "")
            if not subject and not object_value:
                continue
            fact_objects.append(
                ExtractedFact(
                    subject=subject,
                    predicate=str(fact.get("predicate") or ""),
                    object=object_value,
                    title=str(fact.get("title") or ""),
                    description=str(fact.get("description") or ""),
                )
            )
    count = len(fact_objects)
    if count == 0:
        return {
            "score": 0.0,
            "non_duplicate_rate": 0.0,
            "canonical_consistency": 0.0,
            "n_facts": 0,
        }

    deduper = FactDeduper(embed_fn=embed_fn, threshold=dedup_threshold)
    duplicates = sum(1 for fact in fact_objects if deduper.check(fact)[0])
    non_duplicate_rate = 1.0 - duplicates / count
    consistency = canonical_consistency(fact_objects)
    score = dedup_weight * non_duplicate_rate + (1.0 - dedup_weight) * consistency
    return {
        "score": score,
        "non_duplicate_rate": non_duplicate_rate,
        "canonical_consistency": consistency,
        "n_facts": float(count),
    }


__all__ = ["canonical_consistency", "extraction_quality"]
