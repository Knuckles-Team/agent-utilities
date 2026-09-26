#!/usr/bin/python
from __future__ import annotations

import contextlib

from .admission_authority import AdmissionAuthority

"""Tenant-graph RBAC admission through EG's verified identity authority.

EG owns the durable identity and RBAC records. ``CreateGraph`` provisions a
``tenant:<slug>`` role and the graph grants; this module is the deployment
adapter that enrolls ordinary principals into that role. An admin-gated
``GetIdentity`` read supplies the complete current identity before every
replacing ``RegisterIdentity`` write. A manifest's ``existing_roles`` field is
only an optional consistency assertion; it cannot override EG state. Failed,
malformed, or unauthorized reads stop before any write.

For a genuinely absent identity, the manifest supplies the initial Agent role
and teams. A self-admission is still a no-write path because an ordinary
principal cannot use EG's admin-gated identity read or register operation. It
does not provision a missing role; the privileged deployment admission pass
must run first. Production admission remains an explicit apply operation.
"""

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)

__all__ = [
    "EngineIdentityClient",
    "FixtureEngineIdentityClient",
    "LiveEngineIdentityClient",
    "TenantAccessOutcome",
    "TenantAccessResult",
    "AdmissionAuthority",
    "TenantAdmissionError",
    "TenantPrincipal",
    "provision_tenant_access",
    "resolve_engine_identity_client",
    "tenant_role_name",
]


def tenant_role_name(tenant_slug: str) -> str:
    """The durable RBAC role name the engine's own
    ``IsolationLayer::provision_tenant_graph_access`` provisions for
    ``tenant_slug`` (``crates/eg-core/src/isolation.rs``) — ``tenant:<slug>``.
    Kept as one named function (never re-derived inline) so this module and
    the engine's naming convention can never independently drift."""

    slug = tenant_slug.strip()
    if not slug:
        raise ValueError("tenant_slug must be a non-empty opaque identifier")
    return f"tenant:{slug}"


@dataclass(frozen=True, slots=True)
class TenantPrincipal:
    """One target principal from the deployment's Tier-1 roster.

    ``agent_id`` must match the verified request principal on later requests.
    ``role`` and ``teams`` initialize only an identity EG confirms absent.
    ``existing_roles`` is an optional consistency assertion against EG's
    ``GetIdentity`` response; it never supplies the role set for an existing
    principal's replacing ``RegisterIdentity`` write.
    """

    agent_id: str
    role: str = "Agent"
    teams: tuple[str, ...] = ()
    existing_roles: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if not self.agent_id.strip():
            raise ValueError("agent_id must be a non-empty opaque identifier")
        if self.role == "System":
            raise ValueError(
                "TenantPrincipal.role must never be 'System' — System bypasses "
                "RBAC entirely and is out of scope for tenant content access "
                "(see engine_rbac_admission for Tier-2 admin admission instead)"
            )


class TenantAdmissionError(RuntimeError):
    """A tenant-access admission RPC failed. Never swallowed — a failed
    admission must fail the provisioning pass, not silently leave a principal
    under-admitted (the same fail-closed contract
    ``engine_rbac_admission.EngineAdmissionError`` states)."""


@dataclass(frozen=True, slots=True)
class TenantAccessOutcome:
    """What happened for one principal admitted into one tenant's role."""

    agent_id: str
    tenant_slug: str
    role: str
    already_held: bool
    detail: str


@dataclass(frozen=True, slots=True)
class TenantAccessResult:
    """The complete result of one :func:`provision_tenant_access` run."""

    tenant_slug: str
    role: str
    outcomes: tuple[TenantAccessOutcome, ...]

    @property
    def all_admitted(self) -> bool:
        """Always ``True`` for a returned result, by construction —
        :func:`provision_tenant_access` raises :class:`TenantAdmissionError`
        immediately on the first failure rather than ever returning a partial
        result with a not-admitted outcome mixed in (fail closed). Kept as an
        explicit, checkable property (mirroring
        :attr:`~agent_utilities.security.engine_rbac_admission.AdmissionResult.all_admitted`)
        so a caller's assertion reads as an intent, not an implementation
        detail of how failure is signaled."""
        return len(self.outcomes) > 0


@runtime_checkable
class EngineIdentityClient(Protocol):
    """The minimal engine surface tenant-access admission needs — small and
    Protocol-typed so :class:`FixtureEngineIdentityClient` (every test) and
    :class:`LiveEngineIdentityClient` (the real engine) are interchangeable,
    mirroring :class:`~agent_utilities.security.engine_rbac_admission.EngineAdmissionClient`.
    """

    def get_identity(self, agent_id: str) -> Mapping[str, Any] | None: ...

    def register_identity(
        self,
        *,
        agent_id: str,
        role: Any,
        teams: list[str],
        roles: list[str],
        signer_id: str,
        signer_key: str,
    ) -> str: ...


#: The engine keeps its identity/RBAC store in this graph; ``register_identity``
#: targets it explicitly, so an admission call must be scoped there too.
IDENTITY_GRAPH = "__commons__"


@contextlib.contextmanager
def _identity_store_scope() -> Any:
    """Bind the caller's session to the identity store for one admission call.

    Two separate contracts have to hold at once, and they pull in opposite
    directions:

    * The engine (and the client's own pre-send check) require the detached
      RegisterIdentity signature to be produced under the SAME principal it will
      be verified against. The routed transport only applies the session context
      around the SEND (``graph_compute._invoke_at``), so at signing time it is
      still the zero-authority context the socket was opened with.
    * ``ConsensusClient.register_identity`` explicitly targets
      ``graph="__commons__"`` -- the identity store -- while an ordinary session
      is tenant-scoped (e.g. ``tenant__homelab____commons__``). ``_send_routed``
      then refuses with "An explicit graph cannot retarget the verified
      GraphSession".

    So the session is re-scoped to ``__commons__`` (``GraphSession.with_graph``,
    which preserves the verified actor and therefore the principal) and that
    re-scoped context is bound onto the native transport for the whole call.
    """

    from ..knowledge_graph.core.graph_compute import GraphComputeEngine
    from ..knowledge_graph.core.session import (
        current_session,
        resolve_session,
        use_session,
    )

    session = current_session()
    if session is None:
        raise TenantAdmissionError(
            "engine identity registration requires a bound verified GraphSession"
        )
    scoped = resolve_session(session).with_graph(IDENTITY_GRAPH)

    view = GraphComputeEngine.get_or_create().client
    native = getattr(getattr(view, "_client", view), "_base", None)
    with use_session(scoped):
        if native is None or not hasattr(native, "use_verified_context"):
            # No seam to bind: proceed and let the call fail loudly rather than
            # silently signing under the wrong context.
            yield
            return
        with native.use_verified_context(scoped.engine_verified_context()):
            yield


def _record_fixture_identity_registration(
    calls: list[tuple[str, tuple[Any, ...]]],
    identities: dict[str, dict[str, Any]],
    *,
    agent_id: str,
    role: Any,
    teams: list[str],
    roles: list[str],
) -> str:
    """Shared in-memory bookkeeping for an admission fixture's
    ``register_identity`` (CX-DUP-ENFORCE): record the call, then store the
    identity dict shape both :class:`FixtureEngineIdentityClient` here and
    :class:`~agent_utilities.security.system_rbac_admission.FixtureSystemAdmissionClient`
    need. Each domain keeps its OWN class, docstring, and ``role`` type
    annotation (``str`` here, ``IdentityRole`` there) — the WIRE-FIRST
    per-domain fixture separation those classes document is intentional and
    unchanged; only this literal bookkeeping body was duplicated, not the
    fixtures themselves.
    """
    calls.append(("register_identity", (agent_id, role, tuple(teams), tuple(roles))))
    identities[agent_id] = {"role": role, "teams": list(teams), "roles": list(roles)}
    return "registered"


class FixtureEngineIdentityClient:
    """In-memory :class:`EngineIdentityClient` double — every test in
    ``tests/unit/security/test_tenant_rbac_admission.py`` drives this, never a
    socket. WIRE-FIRST (D-OB-9) NOTE: only ever constructed by that test module
    and its own fixtures, mirroring the exact
    ``FixtureEngineAdmissionClient`` precedent already accepted in
    ``scripts/wire_first_baseline.json``.
    """

    def __init__(self) -> None:
        #: agent_id -> {"role": str, "teams": list[str], "roles": list[str]}
        self.identities: dict[str, dict[str, Any]] = {}
        self.calls: list[tuple[str, tuple[Any, ...]]] = []

    def get_identity(self, agent_id: str) -> Mapping[str, Any] | None:
        """Mirror EG's admin-gated complete identity read-back."""
        self.calls.append(("get_identity", (agent_id,)))
        identity = self.identities.get(agent_id)
        if identity is None:
            return None
        return {"agent_id": agent_id, **identity}

    def register_identity(
        self,
        *,
        agent_id: str,
        role: Any,
        teams: list[str],
        roles: list[str],
        signer_id: str,
        signer_key: str,
    ) -> str:
        return _record_fixture_identity_registration(
            self.calls,
            self.identities,
            agent_id=agent_id,
            role=role,
            teams=teams,
            roles=roles,
        )


class LiveEngineIdentityClient:
    """Mutating adapter over the real engine's ``ConsensusClient``, via the SAME
    process-authority path
    :class:`~agent_utilities.security.engine_rbac_admission.LiveEngineAdmissionClient`
    uses. Constructing this class does nothing by itself; every method call is a
    real RPC. Only ever instantiated by deployment tooling running the tenant
    admission pass explicitly — never implicitly, never as a default, and
    NEVER against a live cluster from this repository's own tests (see the
    module docstring's PREPARE-ONLY rule)."""

    def __init__(self, *, config: Any = None) -> None:
        self._config = config

    def _client(self) -> Any:
        from ..knowledge_graph.core.graph_compute import GraphComputeEngine

        return GraphComputeEngine.get_or_create().client

    def get_identity(self, agent_id: str) -> Mapping[str, Any] | None:
        """Read EG's full identity under its own ``security:admin`` gate."""
        try:
            with _identity_store_scope():
                reader = getattr(self._client().consensus, "get_identity", None)
                if not callable(reader):
                    raise TenantAdmissionError(
                        "engine client lacks required consensus.get_identity capability"
                    )
                identity = reader(agent_id)
        except TenantAdmissionError:
            raise
        except Exception as exc:
            raise TenantAdmissionError(
                f"engine get_identity({agent_id!r}) failed"
            ) from exc
        if identity is not None and not isinstance(identity, Mapping):
            raise TenantAdmissionError(
                f"engine get_identity({agent_id!r}) returned a non-mapping identity"
            )
        return identity

    def register_identity(
        self,
        *,
        agent_id: str,
        role: Any,
        teams: list[str],
        roles: list[str],
        signer_id: str,
        signer_key: str,
    ) -> str:
        client = self._client()
        try:
            # RegisterIdentity carries a DETACHED signature that the generated
            # client computes BEFORE it sends anything, and the signature is
            # bound to whatever `_effective_verified_context()` returns at that
            # moment. The routed transport only applies the caller's session
            # context around the SEND (`graph_compute._invoke_at`), so at signing
            # time the context is still the zero-authority one the socket was
            # opened with -- and the client's own
            # `signer_id != context["principal"]` check then fails with
            # "identity signer must match the verified principal".
            #
            # Bind the session context around the WHOLE call so the signature is
            # computed under the same principal it will be verified against.
            with _identity_store_scope():
                return str(
                    client.consensus.register_identity(
                        agent_id,
                        role,
                        teams,
                        roles,
                        signer_id=signer_id,
                        signer_key=signer_key,
                    )
                )
        except Exception as exc:
            raise TenantAdmissionError(
                f"engine register_identity({agent_id!r}, roles={roles!r}) failed"
            ) from exc


def resolve_engine_identity_client(config: Any = None) -> EngineIdentityClient:
    """Return a :class:`LiveEngineIdentityClient` bound to ``config``.
    Construction never connects by itself — mirrors
    :func:`~agent_utilities.security.engine_rbac_admission.resolve_engine_admission_client`.
    """

    return LiveEngineIdentityClient(config=config)


def _confirmed_identity_shape(
    client: EngineIdentityClient, principal: TenantPrincipal
) -> tuple[Any, tuple[str, ...], tuple[str, ...]]:
    """Use EG's complete identity as the authority for a replacing upsert."""
    try:
        identity = client.get_identity(principal.agent_id)
    except Exception as exc:
        raise TenantAdmissionError(
            f"cannot admit {principal.agent_id!r}: GetIdentity read failed"
        ) from exc
    if identity is None:
        if principal.existing_roles not in (None, ()):
            raise TenantAdmissionError(
                f"cannot admit {principal.agent_id!r}: manifest roles disagree "
                "with EG's absent identity"
            )
        return principal.role, principal.teams, ()
    if (
        not isinstance(identity, Mapping)
        or set(identity) != {"agent_id", "role", "teams", "roles"}
        or identity["agent_id"] != principal.agent_id
    ):
        raise TenantAdmissionError(
            f"cannot admit {principal.agent_id!r}: GetIdentity returned an "
            "incomplete or mismatched identity"
        )
    from .system_rbac_admission import (
        SystemAdmissionError,
        _copy_identity_role,
        _identity_string_list,
    )

    try:
        role = _copy_identity_role(identity["role"])
        teams = _identity_string_list(principal.agent_id, identity, "teams")
        roles = _identity_string_list(principal.agent_id, identity, "roles")
    except (ValueError, SystemAdmissionError) as exc:
        raise TenantAdmissionError(
            f"cannot admit {principal.agent_id!r}: GetIdentity returned an "
            "invalid identity"
        ) from exc
    if role == "System":
        raise TenantAdmissionError(
            f"cannot admit {principal.agent_id!r}: System identity is outside "
            "tenant role admission"
        )
    if principal.existing_roles is not None and set(principal.existing_roles) != set(
        roles
    ):
        raise TenantAdmissionError(
            f"cannot admit {principal.agent_id!r}: manifest roles disagree "
            "with EG's identity"
        )
    return role, teams, roles


def provision_tenant_access(
    client: EngineIdentityClient,
    tenant_slug: str,
    principals: list[TenantPrincipal],
    *,
    admin_authority: AdmissionAuthority,
) -> TenantAccessResult:
    """Idempotently enroll every principal in ``principals`` into
    ``tenant_slug``'s durable RBAC role (``tenant:<slug>``), so each one can
    Read/Write every graph the engine's own
    ``IsolationLayer::provision_tenant_graph_access`` already covers for that
    tenant (``Pattern("tenant__<slug>__*")`` — see the module docstring).

    Does **NOT** create the tenant's role or grants itself — those come from
    the engine-side ``CreateGraph`` auto-provisioning (or already exist if any
    graph for this tenant was ever created; provisioning access for a tenant
    that has no graph yet is a legitimate no-op ordering — the grant activates
    the moment the tenant's first graph is created, whichever happens first).

    A self-admission is a no-write path. Every other target first gets an
    admin-gated EG ``GetIdentity`` read. An existing identity's exact role,
    teams, and role set are preserved; an absent identity uses the manifest's
    initial role and teams. A stale manifest assertion, malformed read, or
    denied read fails before ``RegisterIdentity``.

    A failure on any one principal raises :class:`TenantAdmissionError`
    immediately — fail closed, never leave a partial admission unreported.
    Idempotent: re-running this for the same tenant/principals never
    duplicates a role-membership entry.
    """

    if not principals:
        raise ValueError("principals must be non-empty")

    role = tenant_role_name(tenant_slug)
    outcomes: list[TenantAccessOutcome] = []
    for principal in principals:
        if principal.agent_id == admin_authority.signer_id:
            # Self-admission: this principal is admitting ITSELF (the engine
            # requires signer == the admitted principal's own key, so this
            # call could only have been signed at all because the principal
            # already exists and is the one calling). Skip unconditionally,
            # regardless of what existing_roles says -- see the module
            # docstring, "Self-admission is a no-op": there is nothing to
            # enrol, and a write here could only ever DESTROY a role
            # (RegisterIdentity replaces, never merges from an unknown prior
            # state), never usefully grant one.
            outcomes.append(
                TenantAccessOutcome(
                    agent_id=principal.agent_id,
                    tenant_slug=tenant_slug,
                    role=role,
                    already_held=True,
                    detail=(
                        f"{principal.agent_id!r} is the admitting principal "
                        "itself (self-admission) — skipped, no "
                        "register_identity call made"
                    ),
                )
            )
            continue
        identity_role, teams, existing_roles = _confirmed_identity_shape(
            client, principal
        )
        if role in existing_roles:
            outcomes.append(
                TenantAccessOutcome(
                    agent_id=principal.agent_id,
                    tenant_slug=tenant_slug,
                    role=role,
                    already_held=True,
                    detail=f"{principal.agent_id!r} already carries {role!r}",
                )
            )
            continue
        merged_roles = sorted({*existing_roles, role})
        client.register_identity(
            agent_id=principal.agent_id,
            role=identity_role,
            teams=list(teams),
            roles=merged_roles,
            signer_id=admin_authority.signer_id,
            signer_key=admin_authority.signer_key,
        )
        outcomes.append(
            TenantAccessOutcome(
                agent_id=principal.agent_id,
                tenant_slug=tenant_slug,
                role=role,
                already_held=False,
                detail=(
                    f"granted {role!r} to {principal.agent_id!r} "
                    f"(roles now {merged_roles!r})"
                ),
            )
        )

    return TenantAccessResult(
        tenant_slug=tenant_slug, role=role, outcomes=tuple(outcomes)
    )
