"""Persistence eligibility — the required gate every durable KV checkpoint passes.

CONCEPT:AU-OS.governance.checkpoint-persistence-eligibility — a KV cache is **derived
from user content**. Pinning one in RAM is a performance decision; writing one to the
engine's durable blob store is **data-at-rest**, and keeping it past the session that
produced it is a **retention** decision. Those are governance questions, not caching
questions, so the tiering layer (:mod:`agent_utilities.kvcache.tiering`) may never
promote a checkpoint to disk without first consulting a gate declared here.

Why this module exists as an abstraction and not a policy
---------------------------------------------------------
The privacy/residency half of the request this layer implements is **unresolved** —
there is no source of truth in this platform for a payload's residency region, its
data classification, or a retention period (deferred item ``D-5.1-3``). Inventing one
here would be worse than having none: it would look authoritative while being fiction.

So this module ships the **contract** and a deliberately conservative default:

* :class:`PersistenceEligibilityGate` — the pluggable interface. One method,
  one decision, no side effects.
* :class:`OperatorGrantEligibility` — the DEFAULT. It denies every disk persistence
  except one that carries an explicit per-request operator grant, and it reports every
  policy question it could **not** answer in
  :attr:`EligibilityDecision.unresolved` rather than guessing. A gate that abstains
  loudly is safer than one that approves quietly.
* :func:`set_persistence_eligibility_gate` — the extension point the real policy
  plugs into once an operator defines it. Registering a gate is the ONLY way to widen
  what may be persisted; there is no env flag, no allowlist, and no bypass argument.

Two invariants the rest of the layer depends on
-----------------------------------------------
1. **RAM never implies disk consent.** A RAM-tier checkpoint is taken without ever
   consulting this module; promoting that same checkpoint to disk consults it in full.
   Consent is not inherited across tiers.
2. **Deny is the default answer, and it is inspectable.** Every decision — permit or
   deny — carries the deciding gate's name, a human-readable reason, and the set of
   unanswered policy questions, so an operator can always ask *why* a checkpoint was
   (or was not) persisted.
"""

from __future__ import annotations

import logging
import threading
from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)

__all__ = [
    "AlwaysDenyEligibility",
    "EligibilityDecision",
    "OperatorGrantEligibility",
    "PersistenceEligibilityGate",
    "PersistenceRequest",
    "get_persistence_eligibility_gate",
    "set_persistence_eligibility_gate",
]

#: The policy questions this platform currently has **no source of truth for**
#: (deferred ``D-5.1-3``). The default gate reports every one of them as unresolved on
#: every decision, so the gap is visible at runtime and in the audit record rather than
#: only in a report. A real gate answers the ones its policy covers and drops them from
#: this set.
UNRESOLVED_POLICY_QUESTIONS: tuple[str, ...] = (
    "data_residency_region",
    "data_classification",
    "retention_period",
)

#: Who asked for the persistence. The default gate treats these very differently: an
#: operator can consent for themselves, an agent's judgement is evidence rather than
#: authority, and the autonomous scorer path has no standing to consent at all.
Initiator = Literal["user", "agent", "system"]


class PersistenceRequest(BaseModel):
    """One durable-persistence request, presented to the gate for a verdict.

    Deliberately carries only what a gate could plausibly reason over. It does NOT
    carry the KV bytes: a gate decides on provenance and declared intent, never by
    inspecting cache contents (which are opaque model state, not inspectable text).
    """

    tenant: str = Field(
        min_length=1,
        description="Owning tenant of the checkpoint. Mandatory — a persistence "
        "request with no tenant can never be authorized.",
    )
    run_id: str = Field(default="", description="Run that produced the checkpoint.")
    point: str = Field(default="", description="Label for the checkpointed moment.")
    size_bytes: int = Field(
        default=0, ge=0, description="Size of the payload that would be written."
    )
    initiator: Initiator = Field(
        default="system",
        description="Who requested persistence: 'user' (a human operator), 'agent' "
        "(the model decided), or 'system' (the autonomous scorer path).",
    )
    operator_grant: bool = Field(
        default=False,
        description="True iff a human operator explicitly authorized THIS persistence. "
        "Never inferred, never defaulted true, and never satisfied by a RAM-tier "
        "checkpoint having already been taken.",
    )
    worthiness_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="The scorer layer's aggregate worthiness for this checkpoint — "
        "evidence a gate MAY weigh, never authority on its own.",
    )
    declared_classification: str = Field(
        default="",
        description="Caller-declared data classification, when the deployment has a "
        "source for one. Empty means 'unknown', which a conservative gate must not "
        "read as 'unclassified'.",
    )
    declared_residency_region: str = Field(
        default="",
        description="Caller-declared residency region, when the deployment has a "
        "source for one. Empty means 'unknown'.",
    )

    model_config = ConfigDict(extra="forbid", frozen=True)


class EligibilityDecision(BaseModel):
    """A gate's verdict — always inspectable, permit or deny.

    :attr:`unresolved` is the load-bearing field: it names the policy questions the
    deciding gate could not answer. A permit with a non-empty ``unresolved`` is a
    permit granted *despite* missing policy, which is exactly the thing an operator
    auditing "why was this persisted?" needs to see.
    """

    permitted: bool
    gate: str = Field(description="Name of the gate that produced this decision.")
    reason: str = Field(description="Human-readable justification for the verdict.")
    unresolved: tuple[str, ...] = Field(
        default=(),
        description="Policy questions this gate could not answer (see "
        "UNRESOLVED_POLICY_QUESTIONS). Empty means the gate's policy covered "
        "everything it needed.",
    )
    policy_ref: str = Field(
        default="",
        description="Optional pointer to the governing policy document/version, for "
        "a deployment whose gate is backed by a real published policy.",
    )

    model_config = ConfigDict(extra="forbid", frozen=True)


@runtime_checkable
class PersistenceEligibilityGate(Protocol):
    """The pluggable eligibility contract.

    One method, no side effects, no I/O expectation. A gate that needs to consult an
    external policy service should cache aggressively — this is consulted on the
    persistence path, not the read path.
    """

    #: Stable identifier recorded on every decision this gate produces.
    name: str

    def evaluate(self, request: PersistenceRequest) -> EligibilityDecision:
        """Return the verdict for ``request``. MUST NOT raise for a well-formed
        request; a gate that cannot decide returns ``permitted=False`` with the
        reason, so an internal failure can never read as consent."""
        ...  # pragma: no cover - protocol


class OperatorGrantEligibility:
    """The DEFAULT gate: deny disk persistence unless a human explicitly permitted it.

    CONCEPT:AU-OS.governance.checkpoint-persistence-eligibility. This gate makes **no
    residency or classification judgement** — it cannot, because this platform has no
    source for either (``D-5.1-3``). It therefore refuses to *infer* consent and
    honours exactly one thing it can verify: a request that states a human operator
    authorized this specific persistence (``initiator="user"`` **and**
    ``operator_grant=True``).

    Consequences, all intentional:

    * An **agent-initiated** persistence is denied. The model's judgement that a
      checkpoint deserves to outlive the session is recorded as evidence
      (:attr:`PersistenceRequest.worthiness_score`) and surfaced to the operator, but
      it is not authority. A deployment that wants agent-authorized persistence
      registers a gate that says so.
    * A **system-autonomous** persistence is denied. The scorer path may checkpoint to
      RAM on its own; it may never write to disk on its own.
    * Every decision reports :data:`UNRESOLVED_POLICY_QUESTIONS`, so even a permitted
      persistence is visibly a permit granted without residency policy.
    """

    name = "operator-grant-default"

    def evaluate(self, request: PersistenceRequest) -> EligibilityDecision:
        if request.initiator != "user":
            return EligibilityDecision(
                permitted=False,
                gate=self.name,
                reason=(
                    f"disk persistence denied: initiator={request.initiator!r} cannot "
                    "authorize data-at-rest. Only an explicit operator grant is "
                    "honoured by the default gate; register a deployment gate to widen "
                    "this."
                ),
                unresolved=UNRESOLVED_POLICY_QUESTIONS,
            )
        if not request.operator_grant:
            return EligibilityDecision(
                permitted=False,
                gate=self.name,
                reason=(
                    "disk persistence denied: no explicit operator grant on the "
                    "request. A RAM-tier checkpoint does not imply consent to write "
                    "the same context to disk."
                ),
                unresolved=UNRESOLVED_POLICY_QUESTIONS,
            )
        return EligibilityDecision(
            permitted=True,
            gate=self.name,
            reason=(
                "disk persistence permitted by explicit operator grant. NOTE: this "
                "deployment has no residency/classification/retention policy source, "
                "so the grant is the only control applied."
            ),
            unresolved=UNRESOLVED_POLICY_QUESTIONS,
        )


class AlwaysDenyEligibility:
    """A locked-down gate: no KV checkpoint is ever written to disk.

    Registered by a deployment that has decided KV cache state must never become
    data-at-rest at all — the strictest reading, and the correct one until a residency
    policy exists for environments handling regulated content.
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


_GATE_LOCK = threading.Lock()
_GATE: PersistenceEligibilityGate = OperatorGrantEligibility()


def get_persistence_eligibility_gate() -> PersistenceEligibilityGate:
    """The currently registered gate (:class:`OperatorGrantEligibility` by default)."""
    with _GATE_LOCK:
        return _GATE


def set_persistence_eligibility_gate(
    gate: PersistenceEligibilityGate,
) -> PersistenceEligibilityGate:
    """Install ``gate`` as the process-wide eligibility authority; returns the previous.

    This is the extension point for the real policy. It is deliberately a **code**
    seam rather than a configuration flag: an eligibility policy is a decision with an
    owner, not a tunable, and *widening* what may be written to disk should require
    someone to write and review the rule that widens it.
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
