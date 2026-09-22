"""Governed authority for GraphOS semantic connector-pack provisioning.

This module is an AU application adapter, not a second connector-pack
protocol.  The generated ``McpCatalogSnapshotBinding`` and
``AgentLibraryMutationContext`` remain owned by epistemic-graph and are direct
runtime dependencies of this port.  AU contributes the verified session and
the existing :class:`ActionPolicy` decision; GraphOS contributes the two
authoritative EG-bound values that AU cannot derive safely (the current
catalog binding and the EG owner principal).

The returned callable matches agent-connector-sdk's
``PackImportAuthorityResolver`` contract::

    async def resolver(connector: str) -> tuple[binding, context]

No binding or context is issued before the ambient session, required scope,
and an effect-authorizing policy receipt have all been checked.  A missing
generated contract is an installation error; a stale session, provider
failure, malformed binding, or non-opaque principal is a hard error. None is
converted into an empty/default authority.
"""

from __future__ import annotations

import hashlib
import inspect
import re
import secrets
import time
from collections.abc import Awaitable, Callable
from typing import Any, Protocol, TypeVar

from epistemic_graph.generated.connector_pack import (
    AgentLibraryMutationContext,
    McpCatalogSnapshotBinding,
)

from agent_utilities.knowledge_graph.core.session import GraphSession, resolve_session
from agent_utilities.orchestration.action_policy import (
    ActionRequest,
    get_action_policy,
)

__all__ = [
    "PackImportAuthorityResolver",
    "ProvisioningAuthorityError",
    "pack_import_authority",
]


class ProvisioningAuthorityError(RuntimeError):
    """The verified provisioning authority could not be issued."""


class _Decision(Protocol):
    """Minimum existing ActionPolicy decision surface used by this adapter."""

    receipt: Any | None

    @property
    def allowed(self) -> bool: ...


class _Policy(Protocol):
    def decide(self, request: ActionRequest) -> _Decision: ...


T = TypeVar("T")
ProvisioningProvider = Callable[[], T | Awaitable[T]]
PackImportAuthorityResolver = Callable[
    [str], Awaitable[tuple[McpCatalogSnapshotBinding, AgentLibraryMutationContext]]
]

_REQUIRED_SCOPE = "agent:pack-control"
_ACTION_KIND = "connector_pack_import"
_OPAQUE_PRINCIPAL = re.compile(r"^principal:sha256:[0-9a-f]{64}$")


async def _resolve_provider(provider: ProvisioningProvider[T], *, label: str) -> T:
    try:
        value = provider()
        if inspect.isawaitable(value):
            return await value
    except Exception as exc:  # noqa: BLE001 - authority providers fail closed
        raise ProvisioningAuthorityError(f"{label} authority is unavailable") from exc
    return value


def _opaque_principal(actor_id: str) -> str:
    """Match EG's verified-request persistence principal derivation."""

    normalized = actor_id.strip()
    if not normalized:
        raise ProvisioningAuthorityError("verified actor principal is missing")
    if _OPAQUE_PRINCIPAL.fullmatch(normalized):
        return normalized
    return f"principal:sha256:{hashlib.sha256(normalized.encode('utf-8')).hexdigest()}"


def _connector_name(connector: str) -> str:
    if not isinstance(connector, str):
        raise ProvisioningAuthorityError("connector identity must be text")
    normalized = connector.strip()
    if (
        not normalized
        or len(normalized) > 256
        or any(ord(character) < 0x20 for character in normalized)
    ):
        raise ProvisioningAuthorityError("connector identity is invalid")
    return normalized


def _require_receipt(policy: _Policy, request: ActionRequest) -> tuple[Any, str]:
    """Require an exact effect-authorizing receipt from AU's policy gate."""

    try:
        decision = policy.decide(request)
        receipt = decision.receipt
        allowed = bool(decision.allowed)
    except Exception as exc:  # noqa: BLE001 - policy errors deny authority
        raise ProvisioningAuthorityError("provisioning policy is unavailable") from exc
    if not allowed or receipt is None or not receipt.authorizes_effect:
        raise ProvisioningAuthorityError(
            "connector-pack provisioning is not authorized by policy"
        )
    if receipt.request_digest != request.digest():
        raise ProvisioningAuthorityError(
            "provisioning policy receipt is not request-bound"
        )
    receipt_id = str(receipt.receipt_id).strip()
    if not receipt_id:
        raise ProvisioningAuthorityError("provisioning policy receipt has no identity")
    return receipt, receipt_id


def pack_import_authority(
    engine: Any,
    session: GraphSession,
    *,
    catalog_binding: ProvisioningProvider[McpCatalogSnapshotBinding],
    serving_principal: ProvisioningProvider[str],
) -> PackImportAuthorityResolver:
    """Return a policy-gated resolver for connector-pack imports.

    ``catalog_binding`` must read the current generated EG binding from the
    GraphOS/MCP catalog authority.  ``serving_principal`` must read the current
    opaque EG owner principal from the authenticated EG authority.  Neither
    value is accepted from a request payload or synthesized by AU.  Both
    providers are called only after the verified session and AU policy gate
    authorize the exact connector target.  The gate is resolved from AU's
    existing ``get_action_policy(engine)`` authority and cannot be supplied by
    the caller.

    The resolver revalidates the ambient session for every invocation because
    a long-lived SDK runner may outlive the session under which it was built.
    """

    if not callable(catalog_binding) or not callable(serving_principal):
        raise TypeError(
            "catalog_binding and serving_principal must be authoritative providers"
        )
    bound = resolve_session(session, required_scope=_REQUIRED_SCOPE)
    policy_gate: _Policy = get_action_policy(engine)

    async def resolve(
        connector: str,
    ) -> tuple[McpCatalogSnapshotBinding, AgentLibraryMutationContext]:
        connector_id = _connector_name(connector)
        verified = resolve_session(bound, required_scope=_REQUIRED_SCOPE)
        request = ActionRequest(
            kind=_ACTION_KIND,
            target=connector_id,
            params={"tenant_id": verified.tenant, "scope": _REQUIRED_SCOPE},
            source="graph-os",
            reason="authorize semantic connector-pack provisioning",
            actor_id=str(verified.actor.actor_id),
        )
        receipt, receipt_id = _require_receipt(policy_gate, request)

        binding = await _resolve_provider(catalog_binding, label="catalog binding")
        owner_principal = await _resolve_provider(
            serving_principal, label="serving principal"
        )
        if not isinstance(binding, McpCatalogSnapshotBinding):
            raise ProvisioningAuthorityError(
                "catalog binding is not the generated epistemic-graph contract"
            )
        if not isinstance(owner_principal, str) or not _OPAQUE_PRINCIPAL.fullmatch(
            owner_principal.strip()
        ):
            raise ProvisioningAuthorityError(
                "serving principal is not an opaque epistemic-graph principal"
            )

        caller_principal = _opaque_principal(str(verified.actor.actor_id))
        policy_revision = str(verified.policy_version).strip()
        if not policy_revision:
            raise ProvisioningAuthorityError(
                "verified session has no policy revision for provisioning"
            )
        try:
            context = AgentLibraryMutationContext(
                request_id=secrets.randbits(63),
                principal=owner_principal.strip(),
                caller_principal=caller_principal,
                attempt_nonce=secrets.token_hex(32),
                tenant_id=verified.tenant,
                actor_scope=caller_principal,
                purpose_id="connector-pack:import",
                policy_revision=policy_revision,
                # Bind the AU policy receipt to this request. EG's documented
                # admitted-context step replaces this input with its current
                # owner policy digest before any durable commit.
                policy_digest=f"sha256:{receipt.request_digest}",
                policy_decision_id=receipt_id,
                idempotency_key=f"connector-pack:{connector_id}:pending",
                expected_revision=None,
                trace_id=verified.trace_context,
                created_at_ms=int(time.time() * 1000),
            )
        except Exception as exc:  # noqa: BLE001 - generated validation denies
            raise ProvisioningAuthorityError(
                "generated epistemic-graph mutation context is invalid"
            ) from exc
        return binding, context

    return resolve
