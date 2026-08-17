#!/usr/bin/python
from __future__ import annotations

"""GOC-19 — WorkItem submission command-log preparation (AU gateway side).

CONCEPT:AU-ORCH.dispatch.workitem-command-log

**Scope and honesty about what this module is NOT.** There is no engine-native
``SubmitWorkItem`` mutation yet: confirmed against the current ``epistemic-graph``
tree before writing this (2026-08-16), no ``protocol::Method::SubmitWorkItem``
variant exists anywhere in ``crates/eg-types/src/protocol.rs``, and
:func:`agent_utilities.orchestration.work_item.submit_work_item`/
:func:`~agent_utilities.orchestration.work_item.submit_work_item_atomic` still
perform a read-before-create sequence (a separate idempotency probe, a separate
tenant quota read, a per-dependency read loop, then node creation) against the
engine, exactly as documented there. ``work_item.py`` is a concurrent multi-lane
collision surface (GOC-17/19/35) — this module deliberately makes **no edit** to
it and calls **none** of its write paths, so it carries zero risk of colliding
with that work.

What this module DOES provide, usable today and ready to wire into a future
native gateway call the moment one exists:

1. :func:`prepare_submit_work_item_command` — canonicalizes a WorkItem
   submission request into the frozen wire shape the lane doc's design section
   specifies (``SubmitWorkItemRequest``) plus a deterministic
   ``command_digest``: replaying the SAME logical request (same fields, any
   dict key order, any call) always yields the SAME digest, and any change to
   a field that matters to the command changes it — the client-side half of
   GOC-03's ``CommitDescriptorV1.mutation_digest``/``idempotency_key`` replay
   contract (``crates/eg-types/src/commit_descriptor.rs``).
2. :func:`require_verified_carrier` — a thin, fail-closed wrapper around
   GOC-15's :func:`agent_utilities.security.request_identity.validate_carrier_claims`
   that this gateway calls before it will canonicalize ANY command, so a
   caller cannot construct a command digest under an unverified or
   malformed-shaped identity.

Neither function performs any engine I/O. Wiring an actual atomic
``SubmitWorkItem`` gateway call is real follow-on engine work (a new
``protocol::Method`` variant, an ``eg-capabilities`` policy entry,
``mutation_batch.rs``/``redb_store.rs`` apply logic, and the AU producer
migration in ``work_item.py``/``engine_tasks.py`` the lane doc's W04
describes) — not something this module can honestly claim to deliver by
itself.
"""

import hashlib
import json
from collections.abc import Sequence
from typing import Any

from agent_utilities.security.request_identity import validate_carrier_claims

__all__ = [
    "CommandCarrierRejected",
    "SubmitWorkItemCommand",
    "canonical_command_digest",
    "prepare_submit_work_item_command",
    "require_verified_carrier",
]

#: Bounds mirrored from the lane doc's "Bounds" section (payload/dependency
#: limits for a single SubmitWorkItem command) -- enforced HERE, at
#: preparation time, so an oversize request is refused before it is ever
#: canonicalized/digested, not discovered later at an engine boundary this
#: module does not call.
MAX_PAYLOAD_REF_BYTES = 1024 * 1024
MAX_DEPENDENCIES = 1024


class CommandCarrierRejected(ValueError):
    """Raised when the supplied carrier claims are not verified-carrier-shaped."""


def require_verified_carrier(claims: dict[str, Any]) -> dict[str, Any]:
    """Fail closed unless ``claims`` is a well-formed GOC-15 verified carrier.

    Delegates the shape check to GOC-15's
    :func:`~agent_utilities.security.request_identity.validate_carrier_claims`
    (the one live verified-identity carrier contract this codebase ships) and
    re-raises as :class:`CommandCarrierRejected` so a WorkItem command-log
    caller gets a domain-specific exception rather than a bare ``ValueError``.
    This is a SHAPE check, not cryptographic verification — see that
    function's own docstring; it does not replace the engine's own
    MAC/audience/tenant/policy verification.
    """

    try:
        validate_carrier_claims(claims)
    except ValueError as exc:
        raise CommandCarrierRejected(str(exc)) from exc
    return claims


def canonical_command_digest(value: Any) -> str:
    """Bare lowercase-hex sha256 digest of ``value``'s canonical JSON encoding.

    Bare hex (no ``sha256:`` prefix) to match GOC-03's
    ``CommitDescriptorV1.mutation_digest``/``idempotency_key`` wire contract
    (``crates/eg-types/src/commit_descriptor.rs``'s ``validate_hex_digest``:
    exactly 64 lowercase hex characters). Sorted keys + compact separators
    make the digest independent of dict insertion/key order — the mechanical
    property a replay depends on.
    """

    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class SubmitWorkItemCommand(dict[str, Any]):
    """A canonicalized, digested ``SubmitWorkItemRequest`` (lane doc "Schema and API").

    A plain ``dict`` subclass (not a dataclass/pydantic model) to match this
    codebase's W2.5 control-plane convention of shipping wire requests as
    plain, directly-serializable data (see ``work_item.py``'s own
    ``WORK_ITEM_STATES`` docstring for the same rationale) — there is no
    engine client yet for this shape to bind against, so a heavier model
    would only add a translation layer with nothing on the other end.
    """


def prepare_submit_work_item_command(
    *,
    carrier_claims: dict[str, Any],
    kind: str,
    idempotency_key: str,
    tenant: str | None = None,
    work_item_id: str | None = None,
    priority: int = 2,
    depends_on: Sequence[str] = (),
    input_ref: str = "",
    policy_digest: str,
    catalog_digest: str,
    model_digest: str,
    max_attempts: int = 3,
    deadline_unix: float | None = None,
    metadata: dict[str, Any] | None = None,
) -> SubmitWorkItemCommand:
    """Canonicalize one ``SubmitWorkItemRequest`` and compute its command digest.

    Raises:
        CommandCarrierRejected: ``carrier_claims`` is not a well-formed GOC-15
            verified carrier.
        ValueError: ``kind``/``idempotency_key``/``policy_digest``/
            ``catalog_digest``/``model_digest`` is empty, ``depends_on``
            exceeds :data:`MAX_DEPENDENCIES`, or ``input_ref`` exceeds
            :data:`MAX_PAYLOAD_REF_BYTES`.

    The returned command's ``tenant`` is always the CALLER-VERIFIED tenant
    from ``carrier_claims["tenant"]`` — an explicit ``tenant=`` argument must
    match it exactly (GOC-15's isolation key is ``(tenant, actor_id)``; this
    function refuses to let a caller author a command for a tenant its own
    verified carrier does not claim). Two calls with identical field values
    (regardless of ``depends_on`` ordering or dict key order in ``metadata``)
    always produce the same ``command_digest`` — the replay-safety property a
    future gateway's idempotency check depends on.
    """

    require_verified_carrier(carrier_claims)
    verified_tenant = str(carrier_claims["tenant"]).strip()
    if tenant is not None and str(tenant).strip() != verified_tenant:
        raise ValueError(
            f"requested tenant {tenant!r} does not match the verified carrier's "
            f"tenant {verified_tenant!r}"
        )

    if not str(kind).strip():
        raise ValueError("kind must not be empty")
    if not str(idempotency_key).strip():
        raise ValueError("idempotency_key must not be empty")
    for field_name, field_value in (
        ("policy_digest", policy_digest),
        ("catalog_digest", catalog_digest),
        ("model_digest", model_digest),
    ):
        if not str(field_value).strip():
            raise ValueError(f"{field_name} must not be empty")

    deduped_deps = list(dict.fromkeys(str(dep) for dep in depends_on if dep))
    if len(deduped_deps) > MAX_DEPENDENCIES:
        raise ValueError(
            f"depends_on has {len(deduped_deps)} entries, exceeding the "
            f"{MAX_DEPENDENCIES} bound"
        )
    if len(str(input_ref).encode("utf-8")) > MAX_PAYLOAD_REF_BYTES:
        raise ValueError(f"input_ref exceeds the {MAX_PAYLOAD_REF_BYTES}-byte bound")

    context = {
        "principal": carrier_claims["principal"],
        "tenant": verified_tenant,
        "audience": carrier_claims["audience"],
        "agent_id": carrier_claims["agent_id"],
        "policy_version": carrier_claims["policy_version"],
    }
    request: dict[str, Any] = {
        "schema_version": 1,
        "context": context,
        "work_item_id": work_item_id,
        "idempotency_key": str(idempotency_key),
        "kind": str(kind),
        "priority": int(priority),
        # Sorted: dependency ORDER is not part of the command's logical
        # identity (the engine indexes them as a set), so two callers listing
        # the same dependencies in a different order must digest identically.
        "depends_on": sorted(deduped_deps),
        "input_ref": str(input_ref),
        "policy_digest": str(policy_digest),
        "catalog_digest": str(catalog_digest),
        "model_digest": str(model_digest),
        "max_attempts": int(max_attempts),
        "deadline_unix": deadline_unix,
        "metadata": dict(metadata or {}),
    }
    command_digest = canonical_command_digest(request)
    return SubmitWorkItemCommand({**request, "command_digest": command_digest})
