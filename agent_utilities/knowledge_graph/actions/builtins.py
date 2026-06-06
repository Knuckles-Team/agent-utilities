#!/usr/bin/python
from __future__ import annotations

"""Ontology Action System — built-in actions (CONCEPT:KG-2.25).

Real, live-path verbs registered into the default registry at import time. Each
wraps an existing safe capability so the action layer is *invoked*, not a shell:

  - ``kg.search`` — a permission-filtered, audited Knowledge Graph read
    (wraps the :class:`KnowledgeGraph` facade's guarded Cypher read).
  - ``finance.forensic_screen`` — the engine-grounded forensic-accounting screen
    (wraps :class:`ForensicScreener`); never fabricates numbers offline.

Handlers are defensive: they degrade gracefully when the backend/engine is
offline (returning an empty/UNAVAILABLE result) so the action still completes
and is audited.
"""

import logging
from typing import Any

from .models import ActionEffect, ActionParameter, OntologyAction
from .registry import ActionRegistry

logger = logging.getLogger(__name__)


# ── kg.search ──────────────────────────────────────────────────────────────

KG_SEARCH = OntologyAction(
    name="kg.search",
    verb="search",
    description=(
        "Run a permission-filtered, tenant-scoped, audited read over the "
        "Knowledge Graph (Cypher). Read-only — no ontology mutation."
    ),
    parameters=[
        ActionParameter(
            name="cypher",
            type="string",
            required=True,
            description="A read-only Cypher query.",
        ),
        ActionParameter(
            name="params",
            type="object",
            required=False,
            description="Optional bound parameters for the query.",
        ),
    ],
    acts_on=["concept", "document", "fact", "node"],
    required_capability="kg_read",
    produces_effect=ActionEffect.READ,
    idempotent=True,
)


def _handle_kg_search(params: dict[str, Any]) -> list[dict[str, Any]]:
    """Execute the guarded KG read. Degrades to ``[]`` when no backend exists."""
    try:
        from agent_utilities.knowledge_graph.facade import KnowledgeGraph

        kg = KnowledgeGraph()
        return kg.query(params["cypher"], params.get("params") or {})
    except Exception as exc:  # noqa: BLE001 — offline/no-backend degrades cleanly
        logger.debug("kg.search degraded (no backend): %s", exc)
        return []


# ── finance.forensic_screen ─────────────────────────────────────────────────

FORENSIC_SCREEN = OntologyAction(
    name="finance.forensic_screen",
    verb="screen",
    description=(
        "Engine-grounded forensic-accounting screen for an equity (Beneish "
        "M-score / Altman Z / Piotroski F / Sloan accruals). Calls the "
        "epistemic-graph engine; never fabricates figures when offline."
    ),
    parameters=[
        ActionParameter(
            name="ticker", type="string", required=True, description="Equity symbol."
        ),
        ActionParameter(
            name="this_year",
            type="object",
            required=True,
            description="Standardized line items for the most recent fiscal year.",
        ),
        ActionParameter(
            name="prior_year",
            type="object",
            required=True,
            description="Standardized line items for the prior fiscal year.",
        ),
    ],
    acts_on=["financial_instrument", "company"],
    required_capability="finance_screen",
    produces_effect=ActionEffect.EXTERNAL,
    idempotent=True,
)


def _handle_forensic_screen(params: dict[str, Any]) -> dict[str, Any]:
    """Run the forensic screen. Returns an UNAVAILABLE verdict when offline."""
    from agent_utilities.domains.finance.forensic_screener import ForensicScreener

    verdict = ForensicScreener().screen(
        ticker=params["ticker"],
        this_year=params.get("this_year") or {},
        prior_year=params.get("prior_year") or {},
    )
    return {
        "ticker": verdict.ticker,
        "available": verdict.available,
        "verdict": verdict.verdict,
        "citation": verdict.citation(),
        "is_red_flag": verdict.is_red_flag,
    }


def register_builtins(registry: ActionRegistry) -> None:
    """Register all built-in ontology actions into ``registry``.

    Idempotent across registries; raises if a duplicate name already exists in
    the given registry (per :meth:`ActionRegistry.register` contract).
    """
    registry.register(KG_SEARCH, _handle_kg_search)
    registry.register(FORENSIC_SCREEN, _handle_forensic_screen)
