#!/usr/bin/python
from __future__ import annotations

"""Ontology Action System — submission criteria + typed side-effects (CONCEPT:KG-2.42).

Provenance (Palantir AIP doc: *action-types/overview*): an Action Type validates
its parameters against **submission criteria** and, on submission, applies a list
of **typed edits** (create/modify/delete object, add/remove link) atomically. We
realise both here, wiring the edit leg through the C1 Edit Ledger
(``ontology/edits``) so every applied side-effect is a durable, revertible
``object_edit`` and the whole action is auditable + undoable.

This module is pure orchestration over existing fabric — it neither reinvents
permissions/audit (the executor owns those) nor the edit journal (C1 owns that).
"""

import logging
from typing import Any

from agent_utilities.knowledge_graph.ontology.edits import Edit, EditLedger

from .models import (
    ActionEffectSpec,
    CriterionOp,
    EffectKind,
    OntologyAction,
    SubmissionCriterion,
)

logger = logging.getLogger(__name__)

__all__ = [
    "resolve_template",
    "resolve_params",
    "evaluate_submission_criteria",
    "apply_side_effect",
    "apply_side_effects",
]


def resolve_template(value: Any, params: dict[str, Any]) -> Any:
    """Resolve ``$param`` / ``${param}`` references in a string against ``params``.

    A bare ``"$name"`` resolves to the *typed* value of ``params['name']`` (so a
    dict/number passes through intact); embedded ``"...${name}..."`` interpolates
    a stringified value. Non-string values pass through unchanged.
    """
    if not isinstance(value, str):
        return value
    if value.startswith("$") and "${" not in value and " " not in value:
        key = value[1:]
        return params.get(key, value)
    out = value
    for key, val in params.items():
        out = out.replace("${" + key + "}", str(val))
    return out


def resolve_params(raw: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
    """Resolve every value in a spec param map against invocation ``params``."""
    return {k: resolve_template(v, params) for k, v in raw.items()}


def _scope_value(field: str, params: dict[str, Any], actor_id: str, actor_caps: list[str]) -> Any:
    """Resolve a criterion ``field`` against the params/actor scope."""
    if field == "actor.id":
        return actor_id
    if field in ("actor.capabilities", "actor.caps"):
        return list(actor_caps)
    if field.startswith("params."):
        return params.get(field[len("params.") :])
    if field.startswith("actor."):
        return None
    return params.get(field)


def _check(op: CriterionOp, actual: Any, expected: Any) -> bool:
    """Evaluate one comparison; returns True when the rule HOLDS."""
    if op == CriterionOp.REQUIRED:
        return bool(actual)
    if op == CriterionOp.NON_EMPTY:
        return actual is not None and actual != "" and actual != [] and actual != {}
    if op == CriterionOp.EQUALS:
        return actual == expected
    if op == CriterionOp.NOT_EQUALS:
        return actual != expected
    if op == CriterionOp.IN:
        try:
            return actual in expected
        except TypeError:
            return False
    if op == CriterionOp.NOT_IN:
        try:
            return actual not in expected
        except TypeError:
            return True
    # Ordered comparisons require both sides comparable.
    try:
        if op == CriterionOp.GT:
            return actual > expected
        if op == CriterionOp.GTE:
            return actual >= expected
        if op == CriterionOp.LT:
            return actual < expected
        if op == CriterionOp.LTE:
            return actual <= expected
    except TypeError:
        return False
    return False


def evaluate_submission_criteria(
    action: OntologyAction,
    params: dict[str, Any],
    actor_id: str,
    actor_caps: list[str],
) -> list[str]:
    """Return the messages of any submission criteria that FAIL (empty == pass).

    CONCEPT:KG-2.42 — Palantir submission-criteria gate. An action with no
    criteria always passes (preserving KG-2.25 semantics).
    """
    failures: list[str] = []
    for crit in action.submission_criteria:
        actual = _scope_value(crit.field, params, actor_id, actor_caps)
        expected = resolve_template(crit.value, params)
        if not _check(crit.op, actual, expected):
            failures.append(crit.describe())
    return failures


def apply_side_effect(
    ledger: EditLedger,
    spec: ActionEffectSpec,
    params: dict[str, Any],
    *,
    actor: str = "system",
    invocation_ref: str = "",
    provenance: str = "",
) -> Edit:
    """Apply one typed side-effect through the C1 :class:`EditLedger`.

    Each effect kind maps onto the matching ledger constructor, recording a
    durable ``object_edit`` (with full before/after snapshot) so the action is
    revertible via the C1 revert path. Templated ``target``/``params`` are
    resolved against the invocation ``params`` first.
    """
    target = resolve_template(spec.target, params)
    eff_params = resolve_params(spec.params, params)

    if spec.kind == EffectKind.CREATE_OBJECT:
        return ledger.create_object(
            target,
            eff_params,
            actor=actor,
            provenance=provenance,
            invocation_ref=invocation_ref,
        )
    if spec.kind == EffectKind.MODIFY_OBJECT:
        return ledger.set_property(
            target,
            eff_params,
            actor=actor,
            provenance=provenance,
            invocation_ref=invocation_ref,
        )
    if spec.kind == EffectKind.DELETE_OBJECT:
        return ledger.delete_object(
            target,
            actor=actor,
            provenance=provenance,
            invocation_ref=invocation_ref,
        )
    link_target = eff_params.get("link_target", "")
    link_label = eff_params.get("link_label", "related")
    if spec.kind == EffectKind.ADD_LINK:
        return ledger.add_link(
            target,
            link_target,
            link_label,
            actor=actor,
            provenance=provenance,
            invocation_ref=invocation_ref,
        )
    # REMOVE_LINK
    return ledger.remove_link(
        target,
        link_target,
        link_label,
        actor=actor,
        provenance=provenance,
        invocation_ref=invocation_ref,
    )


def apply_side_effects(
    ledger: EditLedger,
    action: OntologyAction,
    params: dict[str, Any],
    *,
    actor: str = "system",
    invocation_ref: str = "",
) -> list[Edit]:
    """Apply all of an action's declared side-effects in order, journaling each.

    Returns the durable :class:`Edit` records (in applied order) so the executor
    can hang them off the invocation for undo.
    """
    edits: list[Edit] = []
    for spec in action.side_effects:
        edit = apply_side_effect(
            ledger,
            spec,
            params,
            actor=actor,
            invocation_ref=invocation_ref,
            provenance=f"action:{action.name}",
        )
        edits.append(edit)
    return edits
