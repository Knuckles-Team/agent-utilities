#!/usr/bin/python
from __future__ import annotations

"""CONCEPT:AU-KG.ontology.ontology-action-system — Ontology Action System.

First-class, governed *verbs* for the ontology — the layer that closes the one
true gap versus Palantir AIP, whose ontology unifies data + logic + ACTIONS +
security. The ontology already models nouns (typed nodes) and capability
properties (``providesCapability`` / ``requiresCapability`` / ``swappableWith``);
this subpackage adds the missing action type: a parameterized, permission-gated,
audited verb that operates on ontology objects.

Building blocks:
  - :class:`OntologyAction` — an action *definition* (verb, parameters,
    ``acts_on`` object types, ``required_capability``, ``produces_effect``).
  - :class:`ActionRegistry` — binds definitions to handlers; rejects duplicates;
    lookup by name or by ontology object type.
  - :class:`ActionInvocation` — the audited per-call record (status, result
    summary, audit ref), persisted as a KG node.
  - :class:`ActionExecutor` — the governed entry point: authorize (via the
    existing :class:`PermissionsKernel`) → validate params → run handler → audit
    (via :class:`AuditLogger`) → persist (lazy/optional KG backend).

Governance integration (reuses the existing fabric — nothing reinvented):
  - **Permissions**: ``required_capability`` is authorized against the actor's
    role/capabilities through ``security.permissions_kernel.PermissionsKernel``.
  - **Audit**: every invocation emits an ``observability.audit_logger.AuditLogger``
    entry (actor / action / resource / outcome).
  - **Persistence**: invocations become KG ``action_invocation`` nodes with
    ``INVOKED_BY`` → actor and ``ACTS_ON`` → target edges (engine probed lazily;
    skipped cleanly offline).
  - **OWL / SHACL**: ``OntologyAction`` / ``ActionInvocation`` are registered for
    OWL promotion (``owl_bridge.PROMOTABLE_NODE_TYPES``) and an
    ``OntologyActionShape`` in ``shapes/governance.shapes.ttl`` quarantines
    invalid action defs. The ``ontology_action.ttl`` module defines the
    ``mayBeInvokedBy`` property chain ``( :requiresCapability :providedBy )`` so
    an Agent that ``providesCapability`` an action's required capability is
    *reasoned* to be eligible to invoke it — the OWL-substrate dividend.

A module-level :data:`DEFAULT_REGISTRY` is populated at import with real
built-in actions (``kg.search``, ``finance.forensic_screen``) and a
:data:`DEFAULT_EXECUTOR` is bound to it — a live path, not an empty shell.
"""

from .builtins import register_builtins
from .dispatch import (
    Notifier,
    RecordingNotifier,
    get_default_notifier,
    send_notification,
    send_webhook,
    set_default_notifier,
)
from .effects import (
    apply_side_effects,
    evaluate_submission_criteria,
)
from .executor import ActionExecutor, reset_persistence_cache
from .fleet_writeback import register_fleet_writeback
from .models import (
    ActionEffect,
    ActionEffectSpec,
    ActionInvocation,
    ActionParameter,
    ActionStatus,
    CriterionOp,
    EffectKind,
    FunctionRef,
    NotificationSpec,
    OntologyAction,
    SubmissionCriterion,
    WebhookSpec,
)
from .registry import ActionHandler, ActionRegistry

# CONCEPT:AU-KG.ontology.ontology-action-system — the default registry is populated at import, not an empty
# shell. A value stored but never invoked is a bug per AGENTS.md, so the
# DEFAULT_EXECUTOR below binds these registered actions to a live governed path.
DEFAULT_REGISTRY = ActionRegistry()
register_builtins(DEFAULT_REGISTRY)
register_fleet_writeback(DEFAULT_REGISTRY)

DEFAULT_EXECUTOR = ActionExecutor(DEFAULT_REGISTRY)

__all__ = [
    "ActionEffect",
    "ActionEffectSpec",
    "ActionHandler",
    "ActionInvocation",
    "ActionParameter",
    "ActionStatus",
    "ActionExecutor",
    "ActionRegistry",
    "CriterionOp",
    "EffectKind",
    "FunctionRef",
    "NotificationSpec",
    "OntologyAction",
    "SubmissionCriterion",
    "WebhookSpec",
    "Notifier",
    "RecordingNotifier",
    "set_default_notifier",
    "get_default_notifier",
    "send_notification",
    "send_webhook",
    "apply_side_effects",
    "evaluate_submission_criteria",
    "DEFAULT_REGISTRY",
    "DEFAULT_EXECUTOR",
    "register_builtins",
    "reset_persistence_cache",
]
