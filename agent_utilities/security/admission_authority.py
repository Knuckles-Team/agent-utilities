"""The one admission credential: a verified principal signing as itself.

CONCEPT:AU-OS.identity.single-admission-authority — Single Admission Authority

Why this module replaced four copies of the same thing
------------------------------------------------------
``tenant_admission_cli``, ``system_rbac_admission`` and ``tier2_admission_cli``
each carried a byte-identical ``resolve_provisioner_authority`` that read one
shared secret (``engine-admission/provisioner``) out of the ``__secrets__``
graph, plus its own copy of the same three-field authority dataclass under a
different name and its own error type. That shape was not merely duplicated —
it could never work, for two independent reasons:

1. **It is circular.** The provisioner secret lives in the ``__secrets__``
   graph, so reading it needs a ``__secrets__``-pinned client view. Admission
   runs inside a request whose ambient ``GraphSession`` is bound to a
   different graph, and ``graph_compute._send_routed`` refuses the mismatch
   ("A graph-scoped view cannot retarget the verified GraphSession"). Fetching
   the credential that grants graph access itself required graph access, so
   every WebUI sign-in failed at ``resolve_provisioner_authority``.

2. **The engine rejects the concept.** ``verify_register_identity_signature``
   (``epistemic-graph/src/server/auth.rs``) verifies the signature against the
   signer registry and then requires ``signer == context.principal()``, or it
   answers ``SIGNER_TRUST_DENIED``. A provisioner identity *distinct from the
   calling principal* is therefore not a credential the engine has ever
   accepted — the separate provisioner could not have worked even with the
   secret in hand.

So there is exactly one admission credential, and it is not a secret to fetch:
it is the caller's own verified identity. ``signer_id`` is the principal,
because the engine accepts nothing else.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

__all__ = [
    "AdmissionAuthority",
    "AdmissionAuthorityError",
    "resolve_admission_authority",
]

#: Engine-side registry of trusted signer ids to HMAC keys. The engine reads
#: the same variable (``epistemic-graph`` AGENTS.md); a principal absent from
#: it cannot sign an admission operation on any engine.
SIGNER_REGISTRY_ENV = "EPISTEMIC_GRAPH_SIGNER_KEYS_JSON"


class AdmissionAuthorityError(PermissionError):
    """No usable admission credential for the current verified principal.

    Fail-closed: never substituted with a synthesized or downgraded authority.
    """


@dataclass(frozen=True)
class AdmissionAuthority:
    """One principal's authority to sign an engine admission operation.

    ``agent_id`` and ``signer_id`` are always the same verified principal —
    the engine enforces it — but both are carried explicitly so the value sent
    on the wire is the value that was checked here, not one re-derived later.
    """

    agent_id: str
    signer_id: str
    signer_key: str = field(repr=False)

    def __post_init__(self) -> None:
        if not self.agent_id.strip():
            raise ValueError("agent_id must be a non-empty opaque identifier")
        if not self.signer_id.strip():
            raise ValueError("signer_id must be a non-empty opaque identifier")
        if not self.signer_key:
            raise ValueError("signer_key must be non-empty")
        if self.agent_id != self.signer_id:
            raise ValueError(
                "signer_id must equal agent_id — the engine's "
                "verify_register_identity_signature refuses any other pairing "
                "with SIGNER_TRUST_DENIED"
            )

    def __repr__(self) -> str:
        return (
            f"AdmissionAuthority(agent_id={self.agent_id!r}, "
            f"signer_id={self.signer_id!r}, signer_key=<redacted>)"
        )


def _signer_key_for(principal: str) -> str | None:
    """Return this process's own signer key for ``principal``, if it holds one.

    Two sources, in the order a deployment actually provides them, and neither
    is a graph read:

    * ``EPISTEMIC_GRAPH_SIGNER_KEYS_JSON`` — the operator-provisioned registry,
      the documented mechanism for a process talking to a *remote* engine.
    * the running process engine's own bootstrap identity — what a component
      that self-hosts its engine as a child (graph-os) generates and injects
      into that child.
    """

    raw = str(os.environ.get(SIGNER_REGISTRY_ENV, "") or "").strip()
    if raw:
        try:
            registry = json.loads(raw)
        except (TypeError, ValueError):
            raise AdmissionAuthorityError(
                f"{SIGNER_REGISTRY_ENV} is set but is not valid JSON"
            ) from None
        if not isinstance(registry, dict):
            raise AdmissionAuthorityError(
                f"{SIGNER_REGISTRY_ENV} must decode to a JSON object"
            )
        # The engine's registry accepts TWO shapes for a signer entry
        # (`SignerKeySpec`, epistemic-graph/src/server/auth.rs):
        #   legacy: {"<id>": "<key>"}
        #   scoped: {"<id>": {"key": "...", "allowed_roles": [...], ...}}
        # A scoped entry is what grants the signer any authority at all -- a
        # legacy one maps to `allowed_roles: []` and can register nothing -- so
        # reading only the legacy shape means the moment a deployment is
        # correctly scoped, the key becomes invisible here.
        entry = registry.get(principal)
        if isinstance(entry, str) and entry:
            return entry
        if isinstance(entry, dict):
            key = entry.get("key")
            if isinstance(key, str) and key:
                return key

    from agent_utilities.knowledge_graph.core.graph_compute import GraphComputeEngine

    engine = GraphComputeEngine.get_active()
    prepared = getattr(engine, "_local_bootstrap_identity", None)
    if prepared:
        actor_id, signer_key, _context = prepared
        if actor_id == principal and signer_key:
            return str(signer_key)
    return None


def resolve_admission_authority() -> AdmissionAuthority:
    """Return the current verified principal's authority to sign admission.

    Raises:
        AdmissionAuthorityError: if no verified actor is bound, or this process
            holds no signer key for that principal. Both are refusals, never a
            fallback — an admission that cannot be signed must not proceed.
    """

    from agent_utilities.security.brain_context import current_actor

    try:
        actor = current_actor()
    except Exception as exc:
        raise AdmissionAuthorityError(
            "admission requires a bound verified actor; none is in scope"
        ) from exc

    principal = str(getattr(actor, "actor_id", "") or "").strip()
    if not principal:
        raise AdmissionAuthorityError("the bound actor carries no principal identifier")

    signer_key = _signer_key_for(principal)
    if not signer_key:
        raise AdmissionAuthorityError(
            f"this process holds no signer key for verified principal "
            f"{principal!r}. The engine requires the signer to BE the calling "
            f"principal, so admission cannot be delegated to another "
            f"credential. Provision {principal!r} into {SIGNER_REGISTRY_ENV} "
            f"for both this process and the engine, or route the admission "
            f"through a component that already holds one. Full procedure: "
            f"agent_utilities/skills/workflows/agent-os-genesis/references/"
            f"engine-identity-admission.md"
        )

    return AdmissionAuthority(
        agent_id=principal, signer_id=principal, signer_key=signer_key
    )
