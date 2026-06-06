#!/usr/bin/python
from __future__ import annotations

"""Ontology Action System — data models (CONCEPT:KG-2.25).

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


class OntologyAction(BaseModel):
    """A first-class, governed verb over ontology objects. CONCEPT:KG-2.25.

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
            required (CONCEPT:OS-5.12). Empty → derive from effect.
        value_tier: Default value/business-impact tier for this action
            ("low".."critical") used by the escalation matrix when the caller
            does not supply a per-invocation value hint (CONCEPT:OS-5.12).
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
    """An audited record of one :class:`OntologyAction` invocation. CONCEPT:KG-2.25.

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
    timestamp: float = Field(default_factory=time.time)

    def model_post_init(self, __context: Any) -> None:
        if not self.id:
            self.id = f"invocation:{self.action_name}:{uuid.uuid4().hex[:12]}"

    @property
    def ok(self) -> bool:
        """True when the action ran successfully."""
        return self.status == ActionStatus.SUCCESS
