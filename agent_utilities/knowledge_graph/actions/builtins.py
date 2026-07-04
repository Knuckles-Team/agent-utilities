#!/usr/bin/python
from __future__ import annotations

"""Ontology Action System — built-in actions (CONCEPT:AU-KG.ontology.ontology-action-system).

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

from .models import (
    ActionEffect,
    ActionEffectSpec,
    ActionParameter,
    EffectKind,
    OntologyAction,
    SubmissionCriterion,
)
from .models import CriterionOp as _Op
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


# ── kg.annotate_concept ─────────────────────────────────────────────────────
# A REAL mutating built-in (CONCEPT:AU-KG.ontology.batch-actions-executor): it sets a review annotation on a
# concept object purely via a typed side-effect applied through the C1 Edit
# Ledger — no bespoke handler write path. Because the mutation is journaled as an
# ``object_edit``, the action is fully audited and revertible (``executor.undo``).

ANNOTATE_CONCEPT = OntologyAction(
    name="kg.annotate_concept",
    verb="annotate",
    description=(
        "Attach a reviewer annotation to a concept object. The change is applied "
        "as a typed MODIFY_OBJECT side-effect journaled in the Edit Ledger, so it "
        "is durably audited and revertible (undo)."
    ),
    parameters=[
        ActionParameter(
            name="concept_id",
            type="string",
            required=True,
            description="Id of the concept object to annotate.",
        ),
        ActionParameter(
            name="note",
            type="string",
            required=True,
            description="The annotation text to set on the concept.",
        ),
        ActionParameter(
            name="reviewer",
            type="string",
            required=False,
            description="Optional reviewer attribution.",
        ),
    ],
    acts_on=["concept"],
    required_capability="kg_write",
    produces_effect=ActionEffect.MUTATION,
    idempotent=False,
    submission_criteria=[
        SubmissionCriterion(
            field="params.note",
            op=_Op.NON_EMPTY,
            message="annotation note must be non-empty",
        ),
    ],
    side_effects=[
        ActionEffectSpec(
            kind=EffectKind.MODIFY_OBJECT,
            target="$concept_id",
            params={"review_note": "$note", "reviewer": "$reviewer"},
        ),
    ],
)


def _handle_annotate_concept(params: dict[str, Any]) -> dict[str, Any]:
    """Handler body — the durable mutation is the declarative side-effect.

    The handler returns a small result summary; the actual graph change is the
    action's typed MODIFY_OBJECT side-effect, applied by the executor through the
    Edit Ledger (so this action is revertible). Real logic, no shell.
    """
    return {
        "annotated": params["concept_id"],
        "note_len": len(str(params.get("note", ""))),
    }


def register_builtins(registry: ActionRegistry) -> None:
    """Register all built-in ontology actions into ``registry``.

    Idempotent across registries; raises if a duplicate name already exists in
    the given registry (per :meth:`ActionRegistry.register` contract).
    """
    registry.register(KG_SEARCH, _handle_kg_search)
    registry.register(FORENSIC_SCREEN, _handle_forensic_screen)
    registry.register(ANNOTATE_CONCEPT, _handle_annotate_concept)
