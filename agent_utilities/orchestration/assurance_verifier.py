#!/usr/bin/python
from __future__ import annotations

"""Deterministic pre-execution assurance verifier for ActionPolicy.

CONCEPT:AU-OS.governance.assurance-state-machine-verifier — Fast, deterministic
state-machine verifier that validates an :class:`ActionRequest` routing
payload against declared invariants BEFORE the action is allowed to execute.
This is the meritorious sliver of the "PureReason" formal-verification
feedback: sub-millisecond, pure-Python, no network/LLM calls — a structural
check of the payload the calling agent hands to :mod:`action_policy`, not a
live re-derivation of world state (that would cost a network round trip and
blow the latency budget).

Four invariants, checked in order (first failure short-circuits):

1. **Role / allowed-set (RLS)** — CONCEPT:AU-OS.governance.assurance-state-machine-verifier.
   ``request.source`` is the actor's declared role. :data:`ROLE_ALLOWED_KINDS`
   is a conservative allowlist per KNOWN role (built from every real call site
   in this repo — see the module docstring for the survey); an unrecognized
   role is unrestricted here (the existing tier gate still applies), so wiring
   this in by default cannot regress any current caller. A *registered* role
   invoking a kind outside its declared set is denied — this is what makes the
   check meaningful (e.g. ``source="reconciler"`` cannot ``secret.delete``).
2. **Argument shape/types (schema)** — CONCEPT:AU-OS.governance.precondition-invariants.
   :data:`INVARIANTS` declares ``required_params: {name: type}`` per action
   kind; a missing or mistyped argument is denied.
3. **Preconditions (state machine)** — CONCEPT:AU-OS.governance.precondition-invariants.
   Some kinds are lifecycle transitions (``run.select`` HELD→materialized,
   ``merge_promotion`` PROPOSED→active, ...). :data:`INVARIANTS` declares
   ``requires_state`` — the set of current states the payload may legally
   claim in ``params["state"]``. Checked against the payload's own declared
   state (never a live KG read — that is what keeps this sub-ms); a claimed
   state outside the legal set is an illegal transition. **Opt-in per call**:
   most real callers of these kinds (``auto_merge.py``, ``spec_proposals.py``,
   ``retained_output.py``) predate this gate and never claim a ``state`` at
   all — that absence is NOT itself a violation (else every existing caller
   would be denied by default). Declaring a *wrong* state IS a violation; this
   still catches the case the invariant exists for — a caller/agent that
   asserts an illegal transition — without retrofitting every production
   call site in the same change.
4. **Reference existence (anti-hallucination)** — CONCEPT:AU-OS.governance.fail-closed-claim-check.
   :data:`INVARIANTS` may declare ``ref_params`` (payload keys that must name
   a real tool/target). The check is a cheap set-membership test against a
   caller-supplied ``known_tools``/``known_targets`` registry — best-effort:
   with no registry supplied the reference is unverifiable and is skipped (no
   false denies), but when a registry IS supplied, an unresolvable reference
   is denied (**fail-closed: unknown ⇒ deny**).

An unregistered action ``kind`` (not in :data:`INVARIANTS`) has no declared
schema/precondition/reference invariant, so it passes checks 2-4 trivially —
"unknown ⇒ deny" is scoped to check 4's registered-but-unresolvable case and
check 1's registered-but-out-of-role case, NOT to every unmodeled kind (that
would deny the majority of already-shipped auto-tier actions and is not the
intent of this hardening pass).
"""

import time
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "KindInvariant",
    "INVARIANTS",
    "ROLE_ALLOWED_KINDS",
    "VerifyResult",
    "verify_action",
]


@dataclass(frozen=True)
class KindInvariant:
    """Declared invariants for one action ``kind``. All fields optional (no restriction)."""

    kind: str = "*"
    required_params: dict[str, type] = field(default_factory=dict)
    requires_state: frozenset[str] | None = None
    ref_params: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# (a) Role → allowed action-kind set. Built from a full-repo survey of every
# ``ActionRequest(..., source=...)`` call site (fleet_reconciler.py,
# remediation_playbooks.py, deploy_watch.py, self_deploy.py, secret_tools.py,
# spec_proposals.py, change_publisher.py, messaging/bus.py) — every currently
# real ``source`` value's actual kinds are included, so enabling this check by
# default never denies existing traffic. "manual" is the dataclass default
# (unset ``source``) and stays unrestricted — a human operator / ungoverned
# caller is scoped by the tier gate, not this RBAC layer.
# ---------------------------------------------------------------------------
ROLE_ALLOWED_KINDS: dict[str, frozenset[str]] = {
    "manual": frozenset({"*"}),
    "reconciler": frozenset({"stop_service", "restart_service", "scale_service"}),
    "playbook": frozenset({"restart_service", "investigate_resource_pressure"}),
    "autoscaler": frozenset({"scale_service"}),
    "deploy_watch": frozenset({"rollback_service"}),
    "self_deploy": frozenset({"restart_service"}),
}


def _role_allows(role: str, kind: str) -> bool:
    allowed = ROLE_ALLOWED_KINDS.get(role)
    if allowed is None:
        return True  # unrecognized role — not restricted by RBAC, tier gate still applies
    if "*" in allowed:
        return True
    import fnmatch

    return any(fnmatch.fnmatchcase(kind, pat) for pat in allowed)


# ---------------------------------------------------------------------------
# (b)+(c)+(d) Per-kind schema / state-machine / reference invariants. Mirrors
# the lifecycle kinds already documented in action_policy.py's DEFAULT_POLICY
# comments (run.select/run.discard = held→materialized/discarded,
# merge_promotion = proposed→active, spec_promotion = proposed→develop) plus
# the argument shape a well-formed payload for each mutating kind must carry.
# ---------------------------------------------------------------------------
INVARIANTS: dict[str, KindInvariant] = {
    "scale_service": KindInvariant(
        kind="scale_service", required_params={"replicas": int}
    ),
    "deploy_service": KindInvariant(
        kind="deploy_service", required_params={"image": str}
    ),
    "secret.set": KindInvariant(
        kind="secret.set", required_params={"path": str, "value": str}
    ),
    "secret.delete": KindInvariant(kind="secret.delete", required_params={"path": str}),
    # NOTE: the id for these lifecycle kinds is already carried in
    # ``request.target`` (the dataclass already requires a target) — no
    # separate required_params id is declared, so no schema check is added
    # beyond what ActionRequest itself already enforces structurally.
    "run.select": KindInvariant(kind="run.select", requires_state=frozenset({"held"})),
    "run.discard": KindInvariant(
        kind="run.discard", requires_state=frozenset({"held"})
    ),
    "merge_promotion": KindInvariant(
        kind="merge_promotion", requires_state=frozenset({"proposed"})
    ),
    "spec_promotion": KindInvariant(
        kind="spec_promotion", requires_state=frozenset({"proposed"})
    ),
    "workspace.computer_use": KindInvariant(
        kind="workspace.computer_use", ref_params=("tool",)
    ),
}


@dataclass
class VerifyResult:
    """The verifier's structured verdict."""

    ok: bool
    reason: str = ""
    invariant: str = ""  # which check failed: role | schema | precondition | reference
    latency_ms: float = 0.0


def verify_action(
    request: Any,
    *,
    known_tools: frozenset[str] | set[str] | None = None,
    known_targets: frozenset[str] | set[str] | None = None,
) -> VerifyResult:
    """Verify ``request`` (an :class:`~agent_utilities.orchestration.action_policy.ActionRequest`)
    against the declared invariants. Pure, synchronous, no I/O — sub-ms by construction.
    """
    t0 = time.perf_counter()
    invariant = INVARIANTS.get(request.kind, KindInvariant(kind=request.kind))
    role = request.source or "manual"

    # (a) role / allowed-set.
    if not _role_allows(role, request.kind):
        return VerifyResult(
            ok=False,
            reason=f"actor role {role!r} is not permitted to invoke {request.kind!r}",
            invariant="role",
            latency_ms=(time.perf_counter() - t0) * 1000,
        )

    # (b) argument shape/types.
    params = request.params if isinstance(getattr(request, "params", None), dict) else {}
    for name, typ in invariant.required_params.items():
        if name not in params:
            return VerifyResult(
                ok=False,
                reason=f"{request.kind!r} requires argument {name!r} (missing)",
                invariant="schema",
                latency_ms=(time.perf_counter() - t0) * 1000,
            )
        value = params[name]
        # bool is a subclass of int — reject bool where int/float declared.
        type_ok = isinstance(value, typ) and not (
            typ in (int, float) and isinstance(value, bool)
        )
        if not type_ok:
            return VerifyResult(
                ok=False,
                reason=(
                    f"{request.kind!r} argument {name!r} expected {typ.__name__}, "
                    f"got {type(value).__name__}"
                ),
                invariant="schema",
                latency_ms=(time.perf_counter() - t0) * 1000,
            )

    # (c) preconditions — a small declared state machine, checked against the
    # payload's OWN claimed current state (no live KG read: that would cost a
    # network round trip and defeat the sub-ms budget). Opt-in: a payload that
    # doesn't claim a state at all is not itself a violation (most production
    # callers of these kinds predate this gate); a payload that DOES claim one
    # must claim a legal one.
    if invariant.requires_state is not None and "state" in params:
        claimed_state = params.get("state")
        if claimed_state not in invariant.requires_state:
            return VerifyResult(
                ok=False,
                reason=(
                    f"{request.kind!r} requires current state in "
                    f"{sorted(invariant.requires_state)}, payload claims {claimed_state!r}"
                ),
                invariant="precondition",
                latency_ms=(time.perf_counter() - t0) * 1000,
            )

    # (d) reference existence — cheap, best-effort anti-hallucination check.
    # Unresolvable (no registry supplied) ⇒ skip (never a false deny); a
    # registry supplied AND the reference not found in it ⇒ deny.
    for ref_name in invariant.ref_params:
        ref_value = params.get(ref_name)
        if ref_value is None:
            continue
        registry = known_tools if ref_name in ("tool", "tool_name") else known_targets
        if registry is not None and ref_value not in registry:
            return VerifyResult(
                ok=False,
                reason=f"{ref_name}={ref_value!r} does not exist (unresolvable reference)",
                invariant="reference",
                latency_ms=(time.perf_counter() - t0) * 1000,
            )

    return VerifyResult(ok=True, latency_ms=(time.perf_counter() - t0) * 1000)
