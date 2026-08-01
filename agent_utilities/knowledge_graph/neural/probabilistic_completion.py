from __future__ import annotations

"""Exact-first, probabilistic-separate result shaping (CONCEPT:AU-KG.mining.governed-neural-layer, KG-6.6).

The repo survey found NO existing ``allow_probabilistic``/speculative-result
gating pattern anywhere in the query layer — every MCP query/search tool
(``graph_query``/``graph_ask``/``graph_search``) returns one unified
``EvidenceBundle`` with a single ``confidence`` field, mixing whatever it
resolved into one list. Wiring a probabilistic channel INTO that shared,
heavily-used MCP surface is explicitly out of this lane's scope (surgical:
that module belongs to the query/retrieval surface, not the connector+neural
lane this task owns — see the deferred register).

What this module provides instead: a small, ADDITIVE composition helper a
caller can use downstream of an already-executed exact query to attach
governed neural material, with the exact/probabilistic split enforced in the
return shape itself — never merged into one list, and the probabilistic key is
straight-up ABSENT (not just empty) unless the caller opts in. Wiring this into
``graph_query``'s MCP surface is a natural next step, deliberately deferred
(D-73-n) rather than risking a heavily-tested, high-traffic tool signature.
"""

from typing import Any

from .models import (
    EntityResolutionProposal,
    RelationLinkPrediction,
    TenantScopedEmbedding,
)

__all__ = ["ExactWithOptionalProbabilistic", "compose_result"]

Neural = EntityResolutionProposal | RelationLinkPrediction | TenantScopedEmbedding


class ExactWithOptionalProbabilistic(dict):
    """A plain dict shaped ``{"exact": [...]}`` or ``{"exact": [...], "probabilistic": [...]}``.

    Subclassing ``dict`` (rather than a pydantic model) keeps this a drop-in
    return shape for existing callers that already expect dict-like query
    results, while still making the "probabilistic key only exists when opted
    in" contract enforceable by :func:`compose_result` construction.
    """


def compose_result(
    exact_rows: list[dict[str, Any]],
    *,
    allow_probabilistic: bool = False,
    probabilistic: list[Neural] | None = None,
) -> ExactWithOptionalProbabilistic:
    """Return exact results, with governed neural material attached SEPARATELY.

    Args:
        exact_rows: Already-resolved Cypher/SPARQL/OWL rows — untouched,
            passed through verbatim, always present.
        allow_probabilistic: The gate. ``False`` (default) — the returned dict
            has NO ``"probabilistic"`` key at all, so a caller that forgets to
            check for it cannot accidentally treat neural output as fact by
            iterating a key that doesn't exist. ``True`` — attaches
            ``probabilistic`` as its own list, each item still carrying its own
            ``decision_status`` (``EntityResolutionProposal``/
            ``RelationLinkPrediction``) so even an opted-in caller can see it
            is unreviewed.
        probabilistic: Governed proposal/prediction objects (never raw
            floats/dicts) — e.g. from
            :func:`.candidate_generation.generate_entity_resolution_proposals`.
            Ignored (never attached) when ``allow_probabilistic`` is ``False``,
            even if supplied — the flag is authoritative, not the argument's
            mere presence.
    """
    result = ExactWithOptionalProbabilistic(exact=list(exact_rows))
    if allow_probabilistic and probabilistic:
        result["probabilistic"] = list(probabilistic)
    return result
