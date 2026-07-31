"""Persistence eligibility — derived automatically from the caller's own authority.

CONCEPT:AU-OS.governance.checkpoint-persistence-eligibility — a KV cache is **derived
from user content**. Pinning one in RAM is a performance decision; writing one to the
engine's durable blob store is **data-at-rest**, and keeping it past the session that
produced it is a **retention** decision. Those are governance questions, not caching
questions, so the tiering layer (:mod:`agent_utilities.kvcache.tiering`) may never
promote a checkpoint to disk without first consulting a gate declared here.

CONCEPT:AU-OS.governance.authority-derived-persistence-eligibility — **how** that gate
decides. There is no operator table, no per-request grant flag, and no env switch.
Eligibility is a *derived* property of two things the platform already carries:

1. **The authority the context was assembled under** — the verified
   :class:`~agent_utilities.knowledge_graph.core.session.GraphSession` (actor, tenant,
   scopes, policy version), narrowed by any active
   :class:`~agent_utilities.security.delegation.SpawnDelegation` ceiling.
2. **The labels of every source that contributed to the context** — each contributing
   source's data classification, residency regions, retention limit, and mandatory
   markings.

The rule, in one sentence:

    A checkpoint may be written to disk **only into the tenancy of the session that
    produced it**, and only where the caller's *effective* authority dominates the
    **most restrictive** composition of **every** contributing source's labels.

Inheritance vs delegation — both, and they run in opposite directions
--------------------------------------------------------------------
* **Labels inherit downward and restrictively.** A checkpoint assembled from N sources
  inherits the **intersection** of what those sources permit, never the union:
  classification is the *maximum* (most secret) of the contributors, residency is the
  *set intersection* of permitted regions, retention is the *minimum* of the limits,
  and mandatory markings are the *union* (a marking on any contributor marks the whole
  checkpoint). Adding a source can only ever make a checkpoint less persistable.
* **Authority delegates downward and monotonically non-increasing.** A spawned agent's
  effective authority is its own authority **intersected with its delegator's ceiling**
  (:attr:`~agent_utilities.security.delegation.SpawnDelegation.ceiling`, the ultimate
  principal's base capabilities). A delegate can never hold persistence authority its
  delegator lacked.

  This intersection is applied **unconditionally**, independent of the
  ``ENABLE_DELEGATED_IDENTITY`` rollout posture. That flag's ``warn`` mode deliberately
  lets a spawn keep tools it would lose under enforcement — an acceptable
  observe-before-enforce trade for *tool scope*, and an unacceptable one for
  *data-at-rest*. See :func:`_narrow_by_delegation`.

The agent-authorized case, enforced rather than asserted
--------------------------------------------------------
The operator wants an agent to be able to decide a checkpoint is worth persisting. That
is legitimate exactly when the agent is **acting under an authority that already permits
it** — so ``trigger`` (which of the three checkpoint paths fired) is recorded as
provenance and is **not read by the authority computation at all**. The model's
worthiness judgement is evidence about *whether it is worth persisting*; the delegation
chain is authority about *whether it may be*. An agent spawned by a principal who could
persist this material persists it automatically; an agent spawned by one who could not is
refused, with the principal and the dropped capabilities named.

Absence denies — the invariant this module is built around
-----------------------------------------------------------
This program has already found **five safety gates that failed open when the KG was
degraded**, because a tolerant reader returning ``[]``/``0`` read as a permissive
verdict. Every "absent" here is therefore an explicit denial, not an empty permissive
set:

* **No verified session** → :meth:`CallerAuthority.unverified` → deny.
* **No contributing sources declared** (``sources=()``) → deny. An unprovenanced
  checkpoint is never persistable; emptiness is never "nothing restricts it".
* **A source missing a label on any axis** → deny, naming ``<source_id>:<axis>``.
  A :class:`SourceLabelResolver` that cannot read a source MUST return a fully
  *unlabelled* :class:`ContributingSource` for it — never omit it from the result.
* **An empty residency intersection** → deny (no region satisfies every contributor).
* **A delegation whose ceiling cannot be resolved** → deny.

The one derivation from a *present* label: a source explicitly classified
:attr:`~agent_utilities.models.company_brain.DataClassification.PUBLIC` is read as
residency-unrestricted with no retention limit. That is not a default applied to a
missing label — it is the meaning of the label that is there, and every decision names
the axes it derived that way.

Two invariants the rest of the layer depends on
-----------------------------------------------
1. **RAM never implies disk consent.** A RAM-tier checkpoint is taken without ever
   consulting this module; promoting that same checkpoint to disk consults it in full,
   re-deriving the authority *at promotion time* (so a since-expired credential or a
   since-revoked delegation refuses).
2. **Deny is the default answer, and it is inspectable.** Every decision — permit or
   deny — carries the deciding gate's name, a human-readable reason, and a full
   :class:`PersistenceDerivation` (the per-source contributions, the composed label, the
   effective authority, and every check's verdict), so an operator can always ask *why*
   a checkpoint was (or was not) persisted.
"""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:  # pragma: no cover - typing only
    from agent_utilities.knowledge_graph.core.session import GraphSession
    from agent_utilities.security.brain_context import ActorContext
    from agent_utilities.security.delegation import SpawnDelegation

logger = logging.getLogger(__name__)

__all__ = [
    "ANY_REGION",
    "DURABLE_WRITE_SCOPE",
    "UNLIMITED_RETENTION_DAYS",
    "AlwaysDenyEligibility",
    "AuthorityDerivedEligibility",
    "CallerAuthority",
    "ComposedLabel",
    "ContributingSource",
    "EligibilityCheck",
    "EligibilityDecision",
    "GraphSourceLabelResolver",
    "PersistenceDerivation",
    "PersistenceEligibilityGate",
    "PersistenceRequest",
    "SourceContribution",
    "SourceLabelResolver",
    "compose_source_labels",
    "derive_caller_authority",
    "get_persistence_eligibility_gate",
    "get_source_label_resolver",
    "set_persistence_eligibility_gate",
    "set_source_label_resolver",
]

#: A source's declaration that it imposes **no** residency restriction. It is a
#: *present* label with the value "anywhere" — never the encoding for "unknown"
#: (which is an empty ``residency_regions`` and denies).
ANY_REGION = "*"

#: A source's declaration that it imposes **no** retention limit. Again a present
#: label; "unknown" is ``retention_days=None`` and denies.
UNLIMITED_RETENTION_DAYS = -1

#: The scope a caller must hold for a durable KV write. Writing a checkpoint blob plus
#: its ``:KVCheckpoint`` node is a graph write, so it demands the graph-write scope the
#: caller would have needed to put the same material in the graph by any other route —
#: not a bespoke ``kv:*`` scope nobody is provisioned with (which would just be the
#: operator table again, wearing a scope's clothes).
DURABLE_WRITE_SCOPE = "kg:write"

#: Ordering of the regulatory classification ladder. Mirrors
#: ``knowledge_graph.ontology.permissioning._CLASS_ORDER`` and ``core.secured_reads`` so
#: all three layers agree on "most restrictive". Kept as plain strings here so this
#: module stays importable without the ontology stack on the import path.
_CLASS_ORDER: dict[str, int] = {
    "public": 0,
    "internal": 1,
    "confidential": 2,
    "restricted": 3,
}
_CLASS_BY_LEVEL: dict[int, str] = {v: k for k, v in _CLASS_ORDER.items()}

#: Capabilities that mean "unrestricted" — mirrors
#: ``security.delegation._UNRESTRICTED_CAPABILITIES``. A principal holding one of these
#: is not narrowed by the ceiling intersection.
_UNRESTRICTED_CAPABILITIES = frozenset(
    {"*", "kg:*", "kg:admin", "admin", "superuser", "root"}
)

#: Which of the three trigger paths took this checkpoint. **Provenance only.** It is
#: recorded on the decision and the audit record; it is deliberately NOT an input to the
#: authority computation (see the module docstring's "agent-authorized case").
Initiator = Literal["user", "agent", "system"]


# ---------------------------------------------------------------------------
# The label lattice — what the sources permit
# ---------------------------------------------------------------------------


class ContributingSource(BaseModel):
    """One source that contributed material to the checkpointed context.

    Every axis defaults to **unlabelled**, and an unlabelled axis denies (naming this
    source). A resolver that cannot read a source returns it in exactly this default
    state rather than dropping it — see :class:`SourceLabelResolver`.
    """

    source_id: str = Field(
        min_length=1,
        description="Stable identifier of the contributing source (the graph node id, "
        "connector document id, or source ref carried on the citation).",
    )
    classification: str = Field(
        default="",
        description="Regulatory classification: 'public' | 'internal' | 'confidential' "
        "| 'restricted'. Empty means UNLABELLED, which denies — it never means "
        "'unclassified'.",
    )
    residency_regions: frozenset[str] = Field(
        default_factory=frozenset,
        description="Regions this source's data may come to rest in. A set containing "
        "'*' is the explicit 'no restriction' declaration. Empty means UNLABELLED, "
        "which denies (except where derived from a PUBLIC classification).",
    )
    retention_days: int | None = Field(
        default=None,
        description="Maximum days this source's material may be retained at rest. "
        "``-1`` declares 'no limit'; ``0`` declares 'must not be retained'. ``None`` "
        "means UNLABELLED, which denies (except where derived from a PUBLIC "
        "classification).",
    )
    markings: frozenset[str] = Field(
        default_factory=frozenset,
        description="Mandatory-control compartment names (the KG-2.46 ``Marking`` "
        "model). The caller must hold ``marking:<name>`` for every one of them.",
    )
    label_source: str = Field(
        default="",
        description="Where these labels were read from, for the audit trail (e.g. "
        "'node-acl', 'unresolvable').",
    )

    model_config = ConfigDict(extra="forbid", frozen=True)


class SourceContribution(BaseModel):
    """How one source's labels landed in the composed label — the explain row."""

    source_id: str
    classification: str = ""
    residency_regions: tuple[str, ...] = ()
    retention_days: int | None = None
    markings: tuple[str, ...] = ()
    label_source: str = ""
    derived_axes: tuple[str, ...] = Field(
        default=(),
        description="Axes filled in from a PUBLIC classification rather than declared "
        "outright. Named so a permit granted on a derivation is visibly a derivation.",
    )
    missing_axes: tuple[str, ...] = Field(
        default=(),
        description="Axes this source declared nothing for. Non-empty means this "
        "source alone is sufficient to deny.",
    )

    model_config = ConfigDict(extra="forbid", frozen=True)


class ComposedLabel(BaseModel):
    """The most-restrictive composition of every contributing source's labels.

    The intersection, never the union: adding a source can only narrow this.
    """

    classification: str = Field(
        default="",
        description="The MAXIMUM (most secret) classification across contributors.",
    )
    residency_regions: frozenset[str] = Field(
        default_factory=frozenset,
        description="The INTERSECTION of permitted regions. A set containing '*' means "
        "every contributor declared no restriction. Empty with contributors present "
        "means no region satisfies all of them.",
    )
    retention_days: int = Field(
        default=0,
        description="The MINIMUM retention limit across contributors (``-1`` = every "
        "contributor declared no limit).",
    )
    markings: frozenset[str] = Field(
        default_factory=frozenset,
        description="The UNION of mandatory markings — a marking on any contributor "
        "marks the whole checkpoint.",
    )
    contributions: tuple[SourceContribution, ...] = ()
    unlabelled: tuple[str, ...] = Field(
        default=(),
        description="``<source_id>:<axis>`` for every label a contributor did not "
        "declare. Non-empty ALWAYS denies.",
    )
    no_sources: bool = Field(
        default=False,
        description="True when no contributing source was declared at all. Denies — "
        "an empty source set is never read as 'nothing restricts this'.",
    )

    model_config = ConfigDict(extra="forbid", frozen=True)

    @property
    def classification_level(self) -> int:
        """Numeric level, defaulting to the MOST restrictive for an unknown label."""
        return _CLASS_ORDER.get(self.classification, _CLASS_ORDER["restricted"])


def compose_source_labels(
    sources: tuple[ContributingSource, ...] | list[ContributingSource],
) -> ComposedLabel:
    """Compose ``sources`` into the single most-restrictive label (the intersection).

    * classification → ``max`` (most secret wins)
    * residency      → set intersection (``ANY_REGION`` is the identity element)
    * retention      → ``min`` (``UNLIMITED_RETENTION_DAYS`` is the identity element)
    * markings       → union (mandatory controls accumulate; they never cancel)

    A source that declares nothing on an axis contributes ``<id>:<axis>`` to
    :attr:`ComposedLabel.unlabelled` instead of silently contributing the identity
    element — the whole point of the "absence denies" invariant.
    """
    items = tuple(sources)
    if not items:
        return ComposedLabel(no_sources=True)

    level = 0
    regions: frozenset[str] | None = None
    retention = UNLIMITED_RETENTION_DAYS
    markings: set[str] = set()
    contributions: list[SourceContribution] = []
    unlabelled: list[str] = []

    for source in items:
        missing: list[str] = []
        derived: list[str] = []

        cls = (source.classification or "").strip().lower()
        if cls not in _CLASS_ORDER:
            missing.append("classification")
        else:
            level = max(level, _CLASS_ORDER[cls])

        # A PUBLIC classification IS a residency/retention declaration ("this material
        # is public, so it may rest anywhere for any length of time"). That is a
        # derivation from a label that is present, never a default for one that is not.
        is_public = cls == "public"

        source_regions = frozenset(source.residency_regions)
        if source_regions:
            if ANY_REGION not in source_regions:
                regions = (
                    source_regions if regions is None else (regions & source_regions)
                )
        elif is_public:
            derived.append("residency")
        else:
            missing.append("residency")

        if source.retention_days is not None:
            if source.retention_days != UNLIMITED_RETENTION_DAYS:
                retention = (
                    source.retention_days
                    if retention == UNLIMITED_RETENTION_DAYS
                    else min(retention, source.retention_days)
                )
        elif is_public:
            derived.append("retention")
        else:
            missing.append("retention")

        markings |= set(source.markings)
        unlabelled.extend(f"{source.source_id}:{axis}" for axis in missing)
        contributions.append(
            SourceContribution(
                source_id=source.source_id,
                classification=cls,
                residency_regions=tuple(sorted(source_regions)),
                retention_days=source.retention_days,
                markings=tuple(sorted(source.markings)),
                label_source=source.label_source,
                derived_axes=tuple(derived),
                missing_axes=tuple(missing),
            )
        )

    return ComposedLabel(
        classification=_CLASS_BY_LEVEL[level],
        residency_regions=frozenset({ANY_REGION}) if regions is None else regions,
        retention_days=retention,
        markings=frozenset(markings),
        contributions=tuple(contributions),
        unlabelled=tuple(unlabelled),
    )


# ---------------------------------------------------------------------------
# The authority side — what the caller may do
# ---------------------------------------------------------------------------


class CallerAuthority(BaseModel):
    """The caller's **effective** authority, after any delegation ceiling is applied.

    Built by :func:`derive_caller_authority` from the verified ambient
    :class:`~agent_utilities.knowledge_graph.core.session.GraphSession` and the ambient
    :class:`~agent_utilities.security.delegation.SpawnDelegation`. It is never
    constructed from a caller payload: an MCP/REST argument claiming an actor, a tenant,
    or a clearance is ignored outright.
    """

    verified: bool = Field(
        default=False,
        description="True only when this authority was derived from a verified "
        "GraphSession. False ALWAYS denies.",
    )
    unverified_reason: str = Field(
        default="",
        description="Why the authority could not be verified (empty when it was).",
    )
    actor_id: str = ""
    actor_type: str = ""
    tenant: str = ""
    scopes: frozenset[str] = Field(default_factory=frozenset)
    clearance: int = Field(
        default=0,
        description="Numeric classification clearance from "
        "``permissioning.actor_clearance`` over the CEILING-NARROWED actor.",
    )
    markings_held: frozenset[str] = Field(
        default_factory=frozenset,
        description="Marking names the ceiling-narrowed actor clears.",
    )
    unrestricted: bool = Field(
        default=False,
        description="True when the actor holds a graph-administration capability, "
        "which clears mandatory markings (matching the permissioning read gate).",
    )
    policy_version: str = ""
    delegated: bool = False
    delegation_principal: str = ""
    delegation_chain: tuple[str, ...] = ()
    ceiling_applied: bool = Field(
        default=False,
        description="True when a delegation ceiling actually narrowed this authority.",
    )
    narrowed_away: tuple[str, ...] = Field(
        default=(),
        description="Capabilities the caller holds that its delegator's ceiling does "
        "NOT — dropped from the effective authority. Named so a refusal can point at "
        "the exact capability the delegator lacked.",
    )

    model_config = ConfigDict(extra="forbid", frozen=True)

    @classmethod
    def unverified(cls, reason: str) -> CallerAuthority:
        """The fail-closed authority: verified=False, and a reason a human can read."""
        return cls(verified=False, unverified_reason=reason)

    @property
    def clearance_label(self) -> str:
        return _CLASS_BY_LEVEL.get(self.clearance, "public")


def _narrow_by_delegation(
    actor: ActorContext,
    delegation: SpawnDelegation | None,
) -> tuple[ActorContext, bool, tuple[str, ...], str]:
    """Intersect ``actor``'s capabilities with its delegator's ceiling.

    Returns ``(effective_actor, ceiling_applied, narrowed_away, unresolved_reason)``.

    Deliberately **not** :func:`~agent_utilities.security.delegation.enforce_ceiling`,
    which is correct for its own job and wrong for this one on two counts:

    * It is a no-op in the shipped ``ENABLE_DELEGATED_IDENTITY=warn`` posture. A soak
      posture that lets a spawn keep tools it would lose under enforcement is a
      reasonable trade for tool scope and an unacceptable one for data-at-rest, so the
      intersection here applies in every posture.
    * It treats an empty/unresolvable ceiling as "do not narrow". For a data-at-rest
      decision an unresolvable ceiling is an **absent label**, and absence denies.
    """
    from dataclasses import replace

    if delegation is None:
        return actor, False, (), ""

    ceiling = frozenset(delegation.ceiling)
    if not ceiling:
        return (
            actor,
            True,
            (),
            (
                "a delegated agent is persisting, but its delegator's capability "
                f"ceiling could not be resolved (principal={delegation.principal!r}); "
                "an unresolvable ceiling is an absent label and denies"
            ),
        )
    if ceiling & _UNRESTRICTED_CAPABILITIES:
        # The ultimate principal is unrestricted, so the ceiling cannot narrow anything.
        return actor, False, (), ""

    roles = tuple(str(r) for r in (getattr(actor, "roles", ()) or ()))
    kept = tuple(r for r in roles if r in ceiling)
    dropped = tuple(r for r in roles if r not in ceiling)
    return replace(actor, roles=kept), True, dropped, ""


def derive_caller_authority(
    *,
    session: GraphSession | None = None,
    delegation: SpawnDelegation | None = None,
) -> CallerAuthority:
    """Derive the effective persistence authority from the caller's verified identity.

    Never raises and never synthesizes authority: every failure path returns
    :meth:`CallerAuthority.unverified`, which denies. ``session``/``delegation`` default
    to the ambient ones so the normal call is argument-free.
    """
    resolved_session = session
    if resolved_session is None:
        try:
            from agent_utilities.knowledge_graph.core.session import current_session

            resolved_session = current_session()
        except Exception as exc:  # noqa: BLE001 — every failure path is a denial
            return CallerAuthority.unverified(
                "no verified GraphSession is bound to this call "
                f"({type(exc).__name__}: {exc})"
            )
        if resolved_session is None:
            # ``current_session`` returns None rather than raising when authority has
            # been explicitly suspended. Checked here so the refusal says WHY instead of
            # surfacing an AttributeError from the next line — the difference between an
            # operator reading "no verified GraphSession" and reading "'NoneType' object
            # has no attribute ...".
            return CallerAuthority.unverified(
                "no verified GraphSession is bound to this call (graph authority is "
                "suspended or was never established)"
            )

    resolved_delegation = delegation
    if resolved_delegation is None:
        try:
            from agent_utilities.security.delegation import current_delegation

            resolved_delegation = current_delegation()
        except Exception as exc:  # noqa: BLE001
            return CallerAuthority.unverified(
                "the active spawn delegation could not be resolved "
                f"({type(exc).__name__}: {exc})"
            )

    try:
        resolved_session.ensure_authority_current()
    except Exception as exc:  # noqa: BLE001 — an expired credential denies
        return CallerAuthority.unverified(
            f"the caller's verified authority is no longer current ({exc})"
        )

    actor = resolved_session.actor
    effective, ceiling_applied, narrowed_away, unresolved = _narrow_by_delegation(
        actor, resolved_delegation
    )
    if unresolved:
        return CallerAuthority.unverified(unresolved)

    try:
        from agent_utilities.knowledge_graph.ontology.permissioning import (
            actor_clearance,
        )

        clearance = actor_clearance(effective)
    except Exception as exc:  # noqa: BLE001 — an unreadable clearance denies
        return CallerAuthority.unverified(
            "the caller's classification clearance could not be resolved "
            f"({type(exc).__name__}: {exc})"
        )

    roles = frozenset(str(r) for r in (getattr(effective, "roles", ()) or ()))
    actor_type = getattr(effective, "actor_type", "")
    return CallerAuthority(
        verified=True,
        actor_id=str(getattr(effective, "actor_id", "") or ""),
        actor_type=str(getattr(actor_type, "value", actor_type) or ""),
        tenant=str(resolved_session.tenant or ""),
        scopes=frozenset(str(s) for s in (resolved_session.scopes or ())),
        clearance=clearance,
        markings_held=frozenset(
            r.split(":", 1)[1] for r in roles if r.startswith("marking:")
        ),
        unrestricted=bool(roles & _UNRESTRICTED_CAPABILITIES),
        policy_version=str(resolved_session.policy_version or ""),
        delegated=resolved_delegation is not None,
        delegation_principal=(
            resolved_delegation.principal if resolved_delegation is not None else ""
        ),
        delegation_chain=(
            tuple(resolved_delegation.chain) if resolved_delegation is not None else ()
        ),
        ceiling_applied=ceiling_applied,
        narrowed_away=narrowed_away,
    )


# ---------------------------------------------------------------------------
# The request / decision contract
# ---------------------------------------------------------------------------


class PersistenceRequest(BaseModel):
    """One durable-persistence request, presented to the gate for a verdict.

    Deliberately carries only what a gate could plausibly reason over. It does NOT
    carry the KV bytes: a gate decides on authority and provenance, never by inspecting
    cache contents (which are opaque model state, not inspectable text).

    Note what is **absent** compared with a hand-authorized design: there is no
    ``operator_grant`` flag and no caller-declared classification/residency. Both were
    removable because a caller cannot be trusted to describe its own authority — over
    the MCP surface, ``operator_grant=true`` was a claim any agent could simply make.
    """

    tenant: str = Field(
        min_length=1,
        description="Owning tenant of the checkpoint. Must equal the verified session "
        "tenant; a persistence request into another tenancy can never be authorized.",
    )
    run_id: str = Field(default="", description="Run that produced the checkpoint.")
    point: str = Field(default="", description="Label for the checkpointed moment.")
    size_bytes: int = Field(
        default=0, ge=0, description="Size of the payload that would be written."
    )
    trigger: Initiator = Field(
        default="system",
        description="Which trigger path took this checkpoint — 'user', 'agent', or "
        "'system'. PROVENANCE ONLY: the authority computation never reads it.",
    )
    worthiness_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="The scorer layer's aggregate worthiness for this checkpoint — "
        "evidence about whether it is worth persisting, never authority to do so.",
    )
    authority: CallerAuthority = Field(
        default_factory=lambda: CallerAuthority.unverified(
            "no caller authority was attached to this persistence request"
        ),
        description="The caller's effective authority (see derive_caller_authority). "
        "Defaults to unverified, which denies.",
    )
    sources: tuple[ContributingSource, ...] = Field(
        default=(),
        description="Every source that contributed material to the checkpointed "
        "context. EMPTY DENIES — an unprovenanced checkpoint is never persistable.",
    )
    target_region: str = Field(
        default="",
        description="The region the durable store physically writes into. Consulted "
        "ONLY when some contributing source actually restricts residency; an unknown "
        "region then denies.",
    )

    model_config = ConfigDict(extra="forbid", frozen=True)


class EligibilityCheck(BaseModel):
    """One step of the derivation, with its verdict — the audit unit of ``explain``."""

    name: str
    passed: bool
    detail: str = ""

    model_config = ConfigDict(extra="forbid", frozen=True)


class PersistenceDerivation(BaseModel):
    """The full derivation behind a verdict, so a human can ask *why*."""

    authority: CallerAuthority
    label: ComposedLabel
    checks: tuple[EligibilityCheck, ...] = ()
    target_region: str = ""
    rule: str = Field(
        default=(
            "persist iff the caller's effective authority (session INTERSECT delegation "
            "ceiling) dominates the most-restrictive composition of every contributing "
            "source's labels, within the session's own tenancy"
        ),
        description="The derivation rule in one line, restated on every decision.",
    )

    model_config = ConfigDict(extra="forbid", frozen=True)


class EligibilityDecision(BaseModel):
    """A gate's verdict — always inspectable, permit or deny.

    :attr:`derivation` is the load-bearing field: it carries the composed label, the
    per-source contributions, the effective authority (including the delegation chain),
    and every check's verdict. :attr:`unresolved` names policy questions the deciding
    gate could not answer; the authority-derived gate leaves it empty because it answers
    all three from labels and denies when it cannot read one.
    """

    permitted: bool
    gate: str = Field(description="Name of the gate that produced this decision.")
    reason: str = Field(description="Human-readable justification for the verdict.")
    unresolved: tuple[str, ...] = Field(
        default=(),
        description="Policy questions this gate could not answer. Empty means the "
        "gate's policy covered everything it needed.",
    )
    policy_ref: str = Field(
        default="",
        description="Optional pointer to the governing policy document/version, for "
        "a deployment whose gate is backed by a real published policy.",
    )
    derivation: PersistenceDerivation | None = Field(
        default=None,
        description="How the verdict was derived. Present on every decision the "
        "authority-derived gate produces.",
    )

    model_config = ConfigDict(extra="forbid", frozen=True)


@runtime_checkable
class PersistenceEligibilityGate(Protocol):
    """The pluggable eligibility contract.

    One method, no side effects. A gate that needs to consult an external policy service
    should cache aggressively — this is consulted on the persistence path, not the read
    path.
    """

    #: Stable identifier recorded on every decision this gate produces.
    name: str

    def evaluate(self, request: PersistenceRequest) -> EligibilityDecision:
        """Return the verdict for ``request``. MUST NOT raise for a well-formed
        request; a gate that cannot decide returns ``permitted=False`` with the
        reason, so an internal failure can never read as consent."""
        ...  # pragma: no cover - protocol


# ---------------------------------------------------------------------------
# The default gate
# ---------------------------------------------------------------------------


class AuthorityDerivedEligibility:
    """The DEFAULT gate: eligibility derived from the caller's authority and the
    contributing sources' labels.

    CONCEPT:AU-OS.governance.authority-derived-persistence-eligibility. Runs eight
    ordered checks and stops at the first failure, so a refusal always names the single
    thing that caused it — the missing label, the offending source, or the delegator
    whose ceiling was too low.
    """

    name = "authority-derived"

    def evaluate(self, request: PersistenceRequest) -> EligibilityDecision:
        authority = request.authority
        label = compose_source_labels(request.sources)
        checks: list[EligibilityCheck] = []

        def deny(name: str, detail: str) -> EligibilityDecision:
            checks.append(EligibilityCheck(name=name, passed=False, detail=detail))
            return EligibilityDecision(
                permitted=False,
                gate=self.name,
                reason=f"disk persistence denied ({name}): {detail}",
                derivation=PersistenceDerivation(
                    authority=authority,
                    label=label,
                    checks=tuple(checks),
                    target_region=request.target_region,
                ),
            )

        def ok(name: str, detail: str) -> None:
            checks.append(EligibilityCheck(name=name, passed=True, detail=detail))

        # 1. Verified authority. No session, an expired credential, an unresolvable
        #    delegation ceiling, or an unreadable clearance all land here.
        if not authority.verified:
            return deny(
                "verified_authority",
                authority.unverified_reason or "the caller has no verified authority",
            )
        ok(
            "verified_authority",
            f"actor={authority.actor_id!r} tenant={authority.tenant!r} "
            f"clearance={authority.clearance_label}"
            + (
                f" (delegated: {' -> '.join(authority.delegation_chain)})"
                if authority.delegated
                else ""
            ),
        )

        # 2. Tenancy. A checkpoint may only be persisted into the tenancy of the
        #    session that produced it.
        if request.tenant != authority.tenant:
            return deny(
                "tenancy",
                f"checkpoint tenant {request.tenant!r} is not the verified session "
                f"tenant {authority.tenant!r}; a checkpoint may only be persisted into "
                "the tenancy that produced it",
            )
        ok("tenancy", f"checkpoint and session both in tenant {request.tenant!r}")

        # 3. Scope. Durable persistence is a graph write.
        if DURABLE_WRITE_SCOPE not in authority.scopes:
            detail = (
                f"the session does not hold {DURABLE_WRITE_SCOPE!r} (scopes="
                f"{sorted(authority.scopes) or 'none recorded'})"
            )
            if authority.ceiling_applied and authority.narrowed_away:
                detail += (
                    "; note the delegator's ceiling removed "
                    f"{sorted(authority.narrowed_away)}"
                )
            return deny("durable_write_scope", detail)
        ok("durable_write_scope", f"session holds {DURABLE_WRITE_SCOPE!r}")

        # 4. Provenance. No declared sources means no basis to reason at all.
        if label.no_sources:
            return deny(
                "source_provenance",
                "no contributing source was declared for this checkpoint; an "
                "unprovenanced context cannot be shown to permit data-at-rest (an "
                "empty source set is never read as 'nothing restricts it')",
            )
        ok(
            "source_provenance",
            f"{len(label.contributions)} contributing source(s): "
            + ", ".join(c.source_id for c in label.contributions[:8]),
        )

        # 5. Every label present. Absence denies, and names itself.
        if label.unlabelled:
            return deny(
                "labels_present",
                "these contributing source labels are undeclared, and an absent label "
                f"denies: {', '.join(label.unlabelled)}",
            )
        derived = [
            f"{c.source_id}:{axis}"
            for c in label.contributions
            for axis in c.derived_axes
        ]
        ok(
            "labels_present",
            "every contributing source declares classification, residency and retention"
            + (
                f" (derived from a PUBLIC classification: {', '.join(derived)})"
                if derived
                else ""
            ),
        )

        # 6. Clearance dominates the composed (most restrictive) classification.
        if authority.clearance < label.classification_level:
            offenders = [
                c.source_id
                for c in label.contributions
                if _CLASS_ORDER.get(c.classification, 0) == label.classification_level
            ]
            return deny(
                "classification",
                f"the composed classification is {label.classification!r} (raised by "
                f"{', '.join(offenders)}) but the caller clears only "
                f"{authority.clearance_label!r}"
                + (
                    "; the delegator's ceiling narrowed this authority (principal="
                    f"{authority.delegation_principal!r}, dropped "
                    f"{sorted(authority.narrowed_away)})"
                    if authority.ceiling_applied
                    else ""
                ),
            )
        ok(
            "classification",
            f"caller clears {authority.clearance_label!r} >= composed "
            f"{label.classification!r}",
        )

        # 7. Mandatory markings — the union across contributors. Graph administration
        #    clears markings, matching the permissioning read gate.
        if label.markings and not authority.unrestricted:
            missing = label.markings - authority.markings_held
            if missing:
                carriers = {
                    m: [c.source_id for c in label.contributions if m in c.markings]
                    for m in sorted(missing)
                }
                return deny(
                    "mandatory_markings",
                    "the caller does not hold every mandatory marking carried by the "
                    "contributing sources: "
                    + "; ".join(
                        f"{m} (from {', '.join(srcs)})" for m, srcs in carriers.items()
                    ),
                )
        ok(
            "mandatory_markings",
            f"caller clears {sorted(label.markings) or 'no'} mandatory marking(s)",
        )

        # 8. Residency. Only consulted when a contributor actually restricts it.
        if ANY_REGION not in label.residency_regions:
            if not label.residency_regions:
                return deny(
                    "residency",
                    "no region satisfies every contributing source — their permitted "
                    "regions intersect to the empty set, so this context may not come "
                    "to rest anywhere",
                )
            if not request.target_region:
                return deny(
                    "residency",
                    "contributing sources restrict residency to "
                    f"{sorted(label.residency_regions)} but the region the durable "
                    "store writes into is unknown; an unknown target region denies",
                )
            if request.target_region not in label.residency_regions:
                offenders = [
                    c.source_id
                    for c in label.contributions
                    if c.residency_regions
                    and ANY_REGION not in c.residency_regions
                    and request.target_region not in c.residency_regions
                ]
                return deny(
                    "residency",
                    f"the durable store writes into {request.target_region!r}, which "
                    f"{', '.join(offenders) or 'a contributing source'} does not permit "
                    f"(permitted: {sorted(label.residency_regions)})",
                )
        ok(
            "residency",
            f"target region {request.target_region or '(unconstrained)'} is permitted "
            "by every contributing source",
        )

        # 9. Retention. A contributor that permits no durable retention vetoes.
        if (
            label.retention_days != UNLIMITED_RETENTION_DAYS
            and label.retention_days <= 0
        ):
            offenders = [
                c.source_id
                for c in label.contributions
                if c.retention_days is not None and c.retention_days == 0
            ]
            return deny(
                "retention",
                f"{', '.join(offenders) or 'a contributing source'} permits no durable "
                "retention of its material, so this checkpoint may not be written at "
                "rest",
            )
        ok(
            "retention",
            "composed retention limit is "
            + (
                "unlimited"
                if label.retention_days == UNLIMITED_RETENTION_DAYS
                else f"{label.retention_days} day(s)"
            ),
        )

        return EligibilityDecision(
            permitted=True,
            gate=self.name,
            reason=(
                f"disk persistence permitted: {authority.actor_id!r} "
                + (
                    f"(delegated by {authority.delegation_principal!r}) "
                    if authority.delegated
                    else ""
                )
                + f"clears {label.classification!r} in tenant {request.tenant!r}, and "
                f"every one of {len(label.contributions)} contributing source(s) "
                "permits data-at-rest here"
            ),
            derivation=PersistenceDerivation(
                authority=authority,
                label=label,
                checks=tuple(checks),
                target_region=request.target_region,
            ),
        )


class AlwaysDenyEligibility:
    """A locked-down gate: no KV checkpoint is ever written to disk.

    Registered by a deployment that has decided KV cache state must never become
    data-at-rest at all — regardless of how well-labelled the sources are or how
    privileged the caller is.
    """

    name = "always-deny"

    def evaluate(self, request: PersistenceRequest) -> EligibilityDecision:
        return EligibilityDecision(
            permitted=False,
            gate=self.name,
            reason=(
                "disk persistence denied: this deployment prohibits durable KV-cache "
                "checkpoints outright."
            ),
            unresolved=(),
        )


# ---------------------------------------------------------------------------
# Source label resolution
# ---------------------------------------------------------------------------


@runtime_checkable
class SourceLabelResolver(Protocol):
    """Resolves source references to their governance labels.

    **The contract that keeps this fail-closed:** the returned tuple MUST contain one
    :class:`ContributingSource` for **every** ref passed in, in order. A ref whose
    labels cannot be read is returned *unlabelled* (so it denies and names itself) — it
    is never dropped, and the resolver never returns a shorter tuple. That is exactly
    the shape of the five gates that previously failed open when the KG was degraded: a
    tolerant reader returning fewer rows read as a permissive verdict.
    """

    name: str

    def resolve(
        self, source_refs: tuple[str, ...], *, tenant: str
    ) -> tuple[ContributingSource, ...]:
        ...  # pragma: no cover - protocol


class GraphSourceLabelResolver:
    """Reads each source's labels from its governance record in the graph.

    Labels live on the node's :class:`~agent_utilities.models.company_brain.NodeACL`
    (``classification`` / ``data_residency_regions`` / ``retention_days``) plus its
    mandatory markings — the platform's existing per-node governance record, extended
    rather than duplicated. A ref with no ACL, or one whose read fails, comes back
    unlabelled and therefore denies.
    """

    name = "node-acl"

    def resolve(
        self, source_refs: tuple[str, ...], *, tenant: str
    ) -> tuple[ContributingSource, ...]:
        resolved = tuple(self._resolve_one(ref, tenant=tenant) for ref in source_refs)
        if len(resolved) != len(source_refs):  # pragma: no cover - defensive invariant
            raise AssertionError(
                "a source label resolver must return one label per ref; returning "
                "fewer would silently drop a restrictive contributor"
            )
        return resolved

    def _resolve_one(self, ref: str, *, tenant: str) -> ContributingSource:
        unlabelled = ContributingSource(source_id=ref, label_source="unresolvable")
        try:
            from agent_utilities.knowledge_graph.core.company_brain_runtime import (
                get_company_brain,
            )
            from agent_utilities.knowledge_graph.ontology.permissioning import (
                markings_for,
            )

            acl = get_company_brain().permissions.get_acl(ref)
            if acl is None:
                logger.info(
                    "[CONCEPT:AU-OS.governance."
                    "authority-derived-persistence-eligibility] source %r has no "
                    "governance ACL — returning it unlabelled so it denies rather than "
                    "silently permitting",
                    ref,
                )
                return unlabelled
            markings = markings_for(ref, tenant=tenant)
        except Exception as exc:  # noqa: BLE001 — an unreadable label is an absent one
            logger.warning(
                "[CONCEPT:AU-OS.governance.authority-derived-persistence-eligibility] "
                "could not read governance labels for source %r (%s: %s) — treating it "
                "as unlabelled, which DENIES",
                ref,
                type(exc).__name__,
                exc,
            )
            return unlabelled

        regions = frozenset(
            str(r).strip() for r in (getattr(acl, "data_residency_regions", ()) or ())
        )
        return ContributingSource(
            source_id=ref,
            classification=str(getattr(acl, "classification", "") or ""),
            residency_regions=regions,
            retention_days=getattr(acl, "retention_days", None),
            markings=frozenset(str(m) for m in markings),
            label_source=self.name,
        )


_RESOLVER_LOCK = threading.Lock()
_RESOLVER: SourceLabelResolver = GraphSourceLabelResolver()


def get_source_label_resolver() -> SourceLabelResolver:
    """The registered source-label resolver (graph node ACLs by default)."""
    with _RESOLVER_LOCK:
        return _RESOLVER


def set_source_label_resolver(resolver: SourceLabelResolver) -> SourceLabelResolver:
    """Install ``resolver`` as the process-wide source-label authority; returns the
    previous one.

    The seam a deployment uses to read labels from its own catalog (a data-governance
    tool, a classification service) instead of node ACLs. It cannot widen anything by
    itself: whatever it returns still has to survive the composition and the authority
    comparison.
    """
    if not isinstance(resolver, SourceLabelResolver):
        raise TypeError(
            "source label resolver must implement SourceLabelResolver (name + "
            f"resolve); got {type(resolver).__name__}"
        )
    global _RESOLVER
    with _RESOLVER_LOCK:
        previous = _RESOLVER
        _RESOLVER = resolver
    return previous


# ---------------------------------------------------------------------------
# The registered gate
# ---------------------------------------------------------------------------

_GATE_LOCK = threading.Lock()
_GATE: PersistenceEligibilityGate = AuthorityDerivedEligibility()


def get_persistence_eligibility_gate() -> PersistenceEligibilityGate:
    """The registered gate (:class:`AuthorityDerivedEligibility` by default)."""
    with _GATE_LOCK:
        return _GATE


def set_persistence_eligibility_gate(
    gate: PersistenceEligibilityGate,
) -> PersistenceEligibilityGate:
    """Install ``gate`` as the process-wide eligibility authority; returns the previous.

    This is the extension point for a deployment whose governing policy is *narrower*
    than the derived rule (:class:`AlwaysDenyEligibility` being the extreme). It is
    deliberately a **code** seam rather than a configuration flag: an eligibility policy
    is a decision with an owner, not a tunable, and *widening* what may be written to
    disk should require someone to write and review the rule that widens it.
    """
    if not isinstance(gate, PersistenceEligibilityGate):
        raise TypeError(
            "persistence eligibility gate must implement PersistenceEligibilityGate "
            f"(name + evaluate); got {type(gate).__name__}"
        )
    global _GATE
    with _GATE_LOCK:
        previous = _GATE
        _GATE = gate
    logger.info(
        "[CONCEPT:AU-OS.governance.checkpoint-persistence-eligibility] persistence "
        "eligibility gate set to %r (was %r)",
        getattr(gate, "name", type(gate).__name__),
        getattr(previous, "name", type(previous).__name__),
    )
    return previous
