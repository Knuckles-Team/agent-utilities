#!/usr/bin/python
from __future__ import annotations

"""Ontology Action System — data models (CONCEPT:AU-KG.ontology.ontology-action-system).

Pydantic models for the governed *verb* layer over the ontology: the
parameterized, permission-gated, audited :class:`OntologyAction` definition, its
:class:`ActionParameter` schema, and the :class:`ActionInvocation` audit record.

These are pure data — no engine, no I/O — so they import cleanly with no
optional backend running.
"""

import time
import uuid
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class ActionEffect(StrEnum):
    """Side-effect class of an :class:`OntologyAction`.

    Drives audit/approval policy: ``READ`` actions never mutate ontology state,
    ``MUTATION`` writes the graph, ``EXTERNAL`` calls an out-of-band system.
    """

    READ = "read"
    MUTATION = "mutation"
    EXTERNAL = "external"


class EffectKind(StrEnum):
    """The class of typed side-effect an action declares. CONCEPT:AU-KG.ontology.batch-actions-executor.

    Provenance (Palantir AIP doc: *action-types/overview*): an Action Type may
    apply MULTIPLE typed edits in one logical submission — create/modify/delete
    an object, or add/remove a link. Each is captured as an
    :class:`ActionEffectSpec` and, when executed, recorded as a durable C1 Edit
    (``ontology/edits``) so every action is fully audited and revertible.
    """

    CREATE_OBJECT = "create_object"
    MODIFY_OBJECT = "modify_object"
    DELETE_OBJECT = "delete_object"
    ADD_LINK = "add_link"
    REMOVE_LINK = "remove_link"


class ActionEffectSpec(BaseModel):
    """One declared, typed side-effect of an :class:`OntologyAction`. CONCEPT:AU-KG.ontology.batch-actions-executor.

    Provenance (Palantir AIP doc: *action-types/overview* — "Modifying the
    Ontology"): an action's effects are a typed list of object/link edits applied
    atomically on submission. This spec is the *declaration*; the executor binds
    it to live params at run time and applies it through the C1
    :class:`~agent_utilities.knowledge_graph.ontology.edits.EditLedger` so the
    mutation is journaled as a revertible ``object_edit``.

    Templating: ``target`` and any string value in ``params`` may reference an
    invocation parameter with ``"$paramName"`` (or embedded ``"${paramName}"``),
    resolved against the validated invocation params before the edit is applied.

    Attributes:
        kind: Which typed edit this effect performs.
        target: The object id (CREATE/MODIFY/DELETE) or the link *source* id
            (ADD/REMOVE link). Supports ``$param`` substitution.
        params: For object edits, the property map to set; for link edits, the
            keys ``link_target`` (required) and ``link_label`` (default
            ``"related"``). String values support ``$param`` substitution.
    """

    kind: EffectKind
    target: str = ""
    params: dict[str, Any] = Field(default_factory=dict)


class ActionStatus(StrEnum):
    """Outcome of an :class:`ActionInvocation`."""

    ALLOWED = "allowed"
    DENIED = "denied"
    SUCCESS = "success"
    ERROR = "error"


class ActionParameter(BaseModel):
    """A typed input parameter to an :class:`OntologyAction`."""

    name: str
    type: str = "string"
    required: bool = True
    description: str = ""


class CriterionOp(StrEnum):
    """Comparison operator for a :class:`SubmissionCriterion`. CONCEPT:AU-KG.ontology.batch-actions-executor."""

    REQUIRED = "required"  # field must be present + truthy
    EQUALS = "equals"
    NOT_EQUALS = "not_equals"
    IN = "in"
    NOT_IN = "not_in"
    GT = "gt"
    GTE = "gte"
    LT = "lt"
    LTE = "lte"
    NON_EMPTY = "non_empty"


class SubmissionCriterion(BaseModel):
    """A submission rule predicate gating whether an action may run. CONCEPT:AU-KG.ontology.batch-actions-executor.

    Provenance (Palantir AIP doc: *action-types/overview* — "Submission
    criteria"): before an action's edits are applied, a set of validation rules
    over the parameters (and the invoking actor) must hold; a failing rule blocks
    submission with a message. This is a declarative, data-only predicate so a
    definition stays pure (no engine needed to author one).

    The ``field`` is resolved against a scope object: ``"params.<key>"`` reads an
    invocation parameter, ``"actor.id"`` / ``"actor.capabilities"`` read the
    actor, and a bare name is treated as ``params.<name>``.
    """

    field: str
    op: CriterionOp = CriterionOp.REQUIRED
    value: Any = None
    message: str = ""

    def describe(self) -> str:
        """Human-readable rendering used in deny messages."""
        if self.message:
            return self.message
        if self.op in (CriterionOp.REQUIRED, CriterionOp.NON_EMPTY):
            return f"{self.field} {self.op}"
        return f"{self.field} {self.op} {self.value!r}"


class FunctionRef(BaseModel):
    """Reference to a registered ontology Function backing an action. CONCEPT:AU-KG.ontology.batch-actions-executor.

    Provenance (Palantir AIP doc: *action-types/overview* — "Function-backed
    actions"): an action may delegate its core logic to a typed, versioned
    Function. We reference it by ``name`` (+ optional pinned ``version``); the
    executor resolves it against the Wave-1 functions runtime
    (``ontology.functions.FunctionRuntime``) via a soft import so authoring a
    definition never hard-depends on the runtime being importable.
    """

    name: str
    version: str = ""


class NotificationSpec(BaseModel):
    """A notification to fire after an action's edits are applied. CONCEPT:AU-KG.ontology.batch-actions-executor.

    Provenance (Palantir AIP doc: *action-types/overview* — "Notifications"):
    actions can notify recipients on submission. ``template`` supports the same
    ``$param`` substitution as effects so the message can embed invocation data.
    """

    channel: str = "default"
    recipient: str = ""
    template: str = ""


class WebhookSpec(BaseModel):
    """An outbound webhook to POST after an action's edits are applied. CONCEPT:AU-KG.ontology.batch-actions-executor.

    Provenance (Palantir AIP doc: *action-types/overview* — "Webhooks"): an
    action can call an external system on submission. Dispatched for real via
    httpx when available, else recorded as a durable outbound record (never a
    silent no-op).
    """

    url: str
    method: str = "POST"
    headers: dict[str, str] = Field(default_factory=dict)
    payload: dict[str, Any] = Field(default_factory=dict)


class OntologyAction(BaseModel):
    """A first-class, governed verb over ontology objects. CONCEPT:AU-KG.ontology.ontology-action-system.

    An action is a *definition* (Palantir-AIP-style ``Action Type``): it names a
    verb, declares its typed parameters, the ontology object types it
    ``acts_on``, the capability it ``requires`` to run, and the class of effect
    it produces. The :class:`~agent_utilities.knowledge_graph.actions.executor.ActionExecutor`
    binds a definition to a handler and runs it under authorization + audit.

    Attributes:
        id: Stable identifier (defaults to ``action:<name>``).
        name: Unique action name (the registry key).
        verb: The imperative verb (e.g. ``"screen"``, ``"search"``).
        description: Human/LLM-facing description.
        parameters: Typed parameter schema.
        acts_on: Ontology object/node types this action operates over.
        required_capability: The ServiceCapability an actor must hold to invoke.
        produces_effect: read | mutation | external.
        idempotent: Whether re-running with the same params is side-effect-free.
        risk_tier: Optional explicit HITL risk tier ("low".."critical") that
            overrides the tier derived from ``produces_effect``/``idempotent``
            when the :class:`EscalationGate` decides whether human approval is
            required (CONCEPT:AU-OS.observability.empty-derive-from-effect). Empty → derive from effect.
        value_tier: Default value/business-impact tier for this action
            ("low".."critical") used by the escalation matrix when the caller
            does not supply a per-invocation value hint (CONCEPT:AU-OS.observability.empty-derive-from-effect).
    """

    id: str = ""
    name: str
    verb: str
    description: str = ""
    parameters: list[ActionParameter] = Field(default_factory=list)
    acts_on: list[str] = Field(default_factory=list)
    required_capability: str
    produces_effect: ActionEffect = ActionEffect.READ
    idempotent: bool = True
    risk_tier: str = ""
    value_tier: str = "low"
    # ── Action-Type extension (CONCEPT:AU-KG.ontology.batch-actions-executor) — all optional, defaults
    # preserve existing KG-2.25 semantics (no side-effects, no criteria). ──
    side_effects: list[ActionEffectSpec] = Field(default_factory=list)
    submission_criteria: list[SubmissionCriterion] = Field(default_factory=list)
    function_ref: FunctionRef | None = None
    notifications: list[NotificationSpec] = Field(default_factory=list)
    webhooks: list[WebhookSpec] = Field(default_factory=list)
    batch: bool = False

    def model_post_init(self, __context: Any) -> None:
        if not self.id:
            self.id = f"action:{self.name}"

    def validate_params(self, params: dict[str, Any]) -> list[str]:
        """Validate ``params`` against this action's parameter schema.

        Returns a list of human-readable error strings — empty when the params
        satisfy the schema (all required present, no unknown keys).
        """
        errors: list[str] = []
        known = {p.name for p in self.parameters}
        for p in self.parameters:
            if p.required and p.name not in params:
                errors.append(f"missing required parameter '{p.name}'")
        for key in params:
            if key not in known:
                errors.append(f"unknown parameter '{key}'")
        return errors


class ActionInvocation(BaseModel):
    """An audited record of one :class:`OntologyAction` invocation. CONCEPT:AU-KG.ontology.ontology-action-system.

    Persisted as a KG ``action_invocation`` node (lazily, when a backend is
    reachable) with ``INVOKED_BY`` → actor and ``ACTS_ON`` → target edges.
    """

    id: str = ""
    action_name: str
    actor_id: str = "system"
    params: dict[str, Any] = Field(default_factory=dict)
    target_id: str = ""
    status: ActionStatus = ActionStatus.ALLOWED
    result_summary: str = ""
    error: str = ""
    audit_ref: str = ""
    persisted: bool = False
    # ── Action-Type extension (CONCEPT:AU-KG.ontology.batch-actions-executor) ──
    # Ids of the C1 EditLedger edits this invocation produced (drives undo).
    edit_ids: list[str] = Field(default_factory=list)
    # Recorded outbound dispatches (notifications + webhooks), each a dict
    # describing the attempt + outcome so dispatch is never a silent no-op.
    dispatches: list[dict[str, Any]] = Field(default_factory=list)
    # Per-target sub-invocation records when this is a batch invocation.
    batch_results: list[dict[str, Any]] = Field(default_factory=list)
    timestamp: float = Field(default_factory=time.time)

    def model_post_init(self, __context: Any) -> None:
        if not self.id:
            self.id = f"invocation:{self.action_name}:{uuid.uuid4().hex[:12]}"

    @property
    def ok(self) -> bool:
        """True when the action ran successfully."""
        return self.status == ActionStatus.SUCCESS
