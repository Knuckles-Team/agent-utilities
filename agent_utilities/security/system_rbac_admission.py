#!/usr/bin/python
from __future__ import annotations

from .admission_authority import AdmissionAuthority

"""Engine-side admission for au's own SYSTEM principal(s) — the fix for
BUG-295 (P0: the scheduler has never fired; ~175 consecutive
``CypherEngineError(PermissionError)`` failures, 0 successes since pod boot).

CONCEPT:AU-OS.identity.system-principal-admission — closes the gap
:mod:`agent_utilities.security.tenant_rbac_admission` (ordinary tenant
content access) and :mod:`agent_utilities.security.engine_rbac_admission`
(the 5 Tier-2 ``admin:*`` actions) leave open: the engine's own daemon
threads — the unified scheduler chief among them — run under au's own
process-minted identity (``_mint_process_session`` /
``kg_server.py:3549-3582``, captured into every background thread by
``knowledge_graph/core/engine_tasks.py``'s
``_capture_verified_background_session`` / ``_run_with_background_authority``,
``engine_tasks.py:38-98``), and **nothing in this repository has ever
registered that identity, or granted it anything, on the engine's own
independent RBAC store**. ``RbacPolicy::evaluate`` is default-deny on an
empty ``identity.roles`` no matter how many grants exist elsewhere, and
``check_access`` returns false before RBAC is even consulted if the
``agent_id`` is not registered at all — so every scheduler tick has failed,
identically, since the day this process type was introduced.

Two confirmed root causes (NE-009 / NE-020) — corrects an earlier, wrong
diagnosis
------------------------------------------------------------------------
An earlier session concluded the fix was a missing *tenant* RBAC grant and
applied ``Pattern("tenant__homelab__*")``. That diagnosis was wrong: the
graph was already readable before the grant (55,801 nodes / 25,508 edges),
and the scheduler failed identically after it, same error, same
``query_ref``, same cadence. Recorded here as a correction, not repeated.

1. **Wrong resource selector.** ``:Schedule`` lives on the isolated control
   graph ``__control__`` (``CONTROL_GRAPH_NAME``,
   ``knowledge_graph/core/shard_topology.py:58``), reached via
   ``engine.control_backend`` (``knowledge_graph/core/engine.py:171,
   225-269`` — ``self.backend.for_graph(CONTROL_GRAPH_NAME)``), from
   ``core/schedule_engine.py``'s ``_control_backend`` (:154-165) whose
   ``_load_all``/``_upsert`` (:319-341, :292-317) issue the Cypher.
   ``IsolationLayer::provision_tenant_graph_access`` only ever fires for
   graph names matching ``tenant__<slug>__{__commons__|default}`` — a
   ``Pattern("tenant__homelab__*")`` grant can **never** match
   ``__control__``, regardless of how many roles the caller holds. The
   correct grant is a plain ``ResourceSelector::Graph("__control__")`` — no
   label-scoped selector is needed (``:Schedule`` is the only label the
   scheduler ever touches on this graph, and a graph-level grant already
   covers it).

2. **No role assignment.** The scheduler's principal — au's own process
   identity, minted once at boot from ``KG_AUTH_TOKEN_REF``/
   ``KG_IDENTITY_OAUTH2``'s JWT ``sub`` — has never been the subject of a
   ``RegisterIdentity``/``RbacAdmin`` call. It is not a registered identity
   at all from the engine's point of view, so ``check_access`` denies it
   before RBAC policy is even consulted.

**Does the scheduler need Write, not just Read, on ``__control__``?** Yes —
traced from source, not assumed. ``core/schedule_engine.py``'s
``run_scheduler_tick`` calls ``_upsert(engine, spec)`` (:292-317, which
calls ``backend.add_node(...)`` against ``_control_backend(engine)``) on
**every** due schedule: in the coalesce-skip branch (a prior tick is still
in flight — advance-and-persist run state, :599) and in the fired branch
(a job was just enqueued — advance-and-persist run state, :655).
``seed_schedules`` (called once per process from inside the tick itself,
:558-562, and again at every re-seed) upserts every ``deploy/schedules.yml``
entry the same way. So the scheduler mutates the control graph on
essentially every tick that finds due work, not merely on rare
administrative action — the grant below carries both Read (to run
``MATCH (s:Schedule) …``) and Write (to advance the node's own run state)
on ``Graph("__control__")``. Nothing else, and nothing scoped wider.

Two designs rejected, on purpose
---------------------------------
* **Not** ``AgentRole::System``. ``check_access`` gives ``System`` an
  unconditional bypass of all RBAC — handing a 60-second background poller
  unrestricted read/write over every graph in the cluster, to read one
  label on one graph, is precisely the kind of blast-radius mistake this
  fix exists to avoid. :class:`SystemPrincipal` refuses ``role="System"``
  the same way
  :class:`~agent_utilities.security.tenant_rbac_admission.TenantPrincipal`
  does, and :func:`provision_system_principal_access` refuses a target RBAC
  role literally named ``"System"``.
* **Not** an engine-side "auto-assign a default role on identity
  registration" change. That would contradict the engine's explicit, tested
  invariant that an empty policy is a valid fail-closed bootstrap image, and
  would silently widen every future registered identity, including end
  users admitted through
  :mod:`~agent_utilities.security.tenant_rbac_admission`. Out of scope for
  this repo regardless (epistemic-graph needs no change for BUG-295 — see
  the program brief).

Design mirrors three already-reviewed precedents exactly
-----------------------------------------------------------
* :mod:`~agent_utilities.security.tenant_rbac_admission` — the
  ``TenantPrincipal``/Protocol/Fixture/Live client split, and the
  "``RegisterIdentity`` is a full-identity upsert, so the caller always
  supplies the FULL desired shape" convention. This module additionally
  uses the engine's ``GetIdentity`` read-back before its boot-time merge.
* :mod:`~agent_utilities.security.engine_rbac_admission` — the
  ``add_role``/``add_grant`` pair for minting a **narrow, named** role
  (never bare ``System``) and granting it directly, rather than only
  enrolling into a role the engine happens to create as a side effect.
* ``agent-webui``'s ``agent_webui.graph_admission.ensure_tenant_admission``
  (BUG-286) — the runtime shape for a **process-local, cache-after,
  backoff-on-failure** admission call:  a positive outcome is cached
  forever for this process's lifetime; a negative one is cached for
  :data:`_FAILURE_BACKOFF_SECONDS` so a still-broken precondition (e.g. the
  missing signer credential — see NE-021 below) is not re-attempted on
  every single call; concurrent callers for the same key collapse onto one
  attempt via a per-key lock.

Mirror-image defect (fixed alongside ``tenant_rbac_admission``'s own incident)
-------------------------------------------------------------------------------
``ensure_system_principal_access``'s only caller (``kg_server.py``'s
daemon-role boot path, below) constructed ``SystemPrincipal(agent_id=agent_id)``
with no ``existing_roles`` at all. Combined with the empty-tuple default that
field used to carry, that silently told :func:`provision_system_principal_access`
"this principal holds nothing else" — which is exactly the ``ensure_tenant_admission``
defect (see ``tenant_rbac_admission``'s module docstring, "Incident"), aimed at
the SAME principal in the opposite direction: if ``ensure_tenant_admission`` had
already granted ``tenant:homelab``, this boot path would silently overwrite it
down to ``roles=['control:system']`` alone. Two independently-triggered,
auto-at-boot admission passes for one principal, each blind to the other's
grant, each an upsert that replaces rather than merges — they clobber each
other regardless of which runs first or second. Fixed the same way: the
``existing_roles`` default is ``None`` (unknown), never an implicit empty
tuple, and :func:`provision_system_principal_access` refuses to write when it
is ``None`` rather than guess. The boot path resolves that unknown state
through the engine's admin-gated ``GetIdentity`` read-back and verifies its
complete identity shape before it calls the replacing ``RegisterIdentity``
upsert. A confirmed absent identity is the only read result treated as an
empty prior role set; a failed or malformed read fails closed before any
role, grant, or identity write. Existing identities retain their full
``role`` (including the Manager mapping shape), ``teams``, and ``roles``.
Crucially, that read is gated by ``security:admin``: an under-admitted
ordinary Agent cannot use it to bootstrap itself. A successful read therefore
proves the caller is already System or holds an explicit Admin grant; an
``ACCESS_DENIED`` result remains a hard pre-write failure.

Boot verification and operator-authorized remediation
------------------------------------------------------
Unlike a WebUI end-user principal (minted dynamically, one per signed-in
human, no static roster), au's own scheduler identity is static and known
at deploy time. ``kg_server.py`` calls
:func:`ensure_system_principal_access` at daemon boot, but this is not an
authority-escalation mechanism:

* An identity already registered as ``System`` is recognized as authorized
  and returns a no-write success; replacing it with a narrower role would be
  destructive and unnecessary.
* An ordinary Agent can be merged only when the calling context already has
  engine Admin capability, which is exactly what the successful admin-gated
  identity read proves. Its complete identity shape is then preserved while
  the narrow ``control:system`` role is appended.
* An under-admitted Agent, an empty RBAC store, a failed read, or malformed
  identity data fails closed before ``add_role``, ``add_grant``, or
  ``RegisterIdentity``. Boot logs the actionable denial and continues
  degraded; it never turns the outage into privilege escalation.

The normal recovery path is therefore an already-admin deployment/operator
context invoking this same composition before rollout, either through
:mod:`agent_utilities.security.system_admission_cli`, mirroring
``tenant_admission_cli.py``/``tier2_admission_cli.py`` exactly (manifest
JSON, dry-run unless ``--apply``), or by calling this function for a target
principal while that operator context is bound. Both paths call the SAME
:func:`provision_system_principal_access` composition, so they always
produce identical provisioning for the same principal.

Credential resolution is
:func:`~agent_utilities.security.admission_authority.resolve_admission_authority`:
the caller's own verified principal, signing as itself. An already-admin
operator may target another ordinary identity only when its signer-registry
allowance permits the complete role set being written; there is no unbound
credential that can sign independently of the verified caller. The engine's
``verify_register_identity_signature`` requires ``signer ==
context.principal()`` and answers ``SIGNER_TRUST_DENIED`` to anything else —
the operator is always the authenticated signer, never an impersonated target.
A process that holds no signer key for its own principal raises
:class:`~agent_utilities.security.admission_authority.AdmissionAuthorityError`
naming that principal and the registry to provision it into (never a key
value — this module never mints, prints, logs, or persists one, matching
every sibling admission module's doctrine, AGENTS.md "Secrets & credential
retrieval"). The ``kg_server.py`` call site catches this, logs it once per
backoff window, and continues serving degraded, with an actionable diagnosis
instead of a bare ``CypherEngineError`` repeating forever.

**Prod operations are PREPARE-ONLY here too**, same as
``tenant_rbac_admission.py``/``engine_rbac_admission.py``/
``graph_ownership_apply.py``. No test in this repository ever calls
:func:`provision_system_principal_access` with a
:class:`LiveSystemAdmissionClient` against a real engine.
"""

import logging
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from ..knowledge_graph.core.shard_topology import CONTROL_GRAPH_NAME
from .tenant_rbac_admission import (
    _identity_store_scope,
    _record_fixture_identity_registration,
)

logger = logging.getLogger(__name__)

__all__ = [
    "CONTROL_ROLE_NAME",
    "FixtureSystemAdmissionClient",
    "LiveSystemAdmissionClient",
    "SystemAccessOutcome",
    "SystemAccessResult",
    "AdmissionAuthority",
    "SystemAdmissionClient",
    "SystemAdmissionError",
    "SystemPrincipal",
    "ensure_system_principal_access",
    "provision_system_principal_access",
    "resolve_system_admission_client",
]

#: The one graph this role ever grants access to — au's own isolated
#: control plane (``:Schedule``/WorkItem authority). Imported at module top,
#: never re-derived, so this module and the engine's own naming convention
#: can never independently drift (the same discipline
#: ``tenant_rbac_admission.tenant_role_name`` documents for its own
#: constant).

#: A narrow, purpose-named RBAC role — never ``System`` (see module
#: docstring, "Two designs rejected"). Carries exactly Read + Write on
#: ``Graph(CONTROL_GRAPH_NAME)``, nothing else.
CONTROL_ROLE_NAME = "control:system"


#: How long a NEGATIVE outcome (missing provisioner credential, or the
#: engine/admission RPC unreachable) is remembered before the next call
#: retries. Bounds the cost of a still-broken precondition without
#: requiring a process restart once it is fixed — mirrors
#: ``agent_webui.graph_admission._FAILURE_BACKOFF_SECONDS`` exactly.
_FAILURE_BACKOFF_SECONDS = 30.0


IdentityRole = str | dict[str, Any]


def _exact_mapping(value: object, key: str, message: str) -> Mapping[str, Any]:
    """Return a one-key mapping or reject an incomplete wire shape."""

    if not isinstance(value, Mapping) or set(value) != {key}:
        raise ValueError(message)
    return value


def _copy_manager_subordinates(value: object) -> list[str]:
    """Validate and detach a Manager role's subordinate identifiers."""

    subordinates = value
    if not isinstance(subordinates, list):
        raise ValueError("Manager.subordinates must be an explicit string list")
    detached: list[str] = []
    seen: set[str] = set()
    for subordinate in subordinates:
        if not isinstance(subordinate, str) or not subordinate.strip():
            raise ValueError("Manager.subordinates entries must be non-empty strings")
        if subordinate in seen:
            raise ValueError("Manager.subordinates contains a duplicate entry")
        seen.add(subordinate)
        detached.append(subordinate)
    return detached


def _copy_identity_role(role: object) -> IdentityRole:
    """Validate and detach the engine's complete ``AgentRole`` wire shape."""

    if isinstance(role, str):
        if role not in {"Agent", "System"}:
            raise ValueError("role must be System, Agent, or a Manager value")
        return role
    manager_role = _exact_mapping(
        role,
        "Manager",
        "role must be System, Agent, or a Manager value",
    )
    manager = _exact_mapping(
        manager_role["Manager"],
        "subordinates",
        "Manager role must contain only subordinates",
    )
    return {
        "Manager": {"subordinates": _copy_manager_subordinates(manager["subordinates"])}
    }


@dataclass(frozen=True, slots=True)
class SystemPrincipal:
    """One au system principal to admit into the control-graph role.
    ``agent_id`` MUST equal the value that appears as the principal's
    ``VerifiedRequestContext.agent_id`` on its future requests (its
    ``KG_AUTH_TOKEN_REF``/``KG_IDENTITY_OAUTH2`` JWT ``sub``) — the same
    binding requirement
    :class:`~agent_utilities.security.tenant_rbac_admission.TenantPrincipal`
    and
    :class:`~agent_utilities.security.engine_rbac_admission.ServiceAdmissionEntry`
    document. ``role``/``teams``/``existing_roles`` are this principal's
    current full identity shape, sent in full on every
    ``RegisterIdentity`` upsert. The boot path obtains that shape from the
    engine's ``GetIdentity`` read-back before it attempts a replacing upsert.

    ``existing_roles`` carries one of two meanings, and they are NOT
    interchangeable — mirrors
    :attr:`~agent_utilities.security.tenant_rbac_admission.TenantPrincipal.existing_roles`
    exactly, for the same reason (see this module's docstring, "Mirror-image
    defect"): an explicit ``()`` means the caller has CONFIRMED this
    principal currently holds no other roles; the default, ``None``, means
    the caller does not know. :func:`provision_system_principal_access`
    merges the former and refuses the latter."""

    agent_id: str
    role: IdentityRole = "Agent"
    teams: tuple[str, ...] = ()
    existing_roles: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if not self.agent_id.strip():
            raise ValueError("agent_id must be a non-empty opaque identifier")
        role = _copy_identity_role(self.role)
        if role == "System":
            raise ValueError(
                "SystemPrincipal.role must never be 'System' — System bypasses "
                "RBAC entirely; this module grants a narrow named role instead "
                "(see the module docstring, 'Two designs rejected')"
            )
        object.__setattr__(self, "role", role)


class SystemAdmissionError(RuntimeError):
    """A system-principal admission RPC, or credential resolution, failed.
    Never swallowed — a failed admission must remain visibly attributable
    (fail closed), not silently leave the scheduler under-admitted with no
    diagnosis, which is exactly BUG-295's own failure mode."""


@dataclass(frozen=True, slots=True)
class SystemAccessOutcome:
    """What happened for one principal admitted into the control-graph
    role."""

    agent_id: str
    role: str
    already_held: bool
    detail: str


@dataclass(frozen=True, slots=True)
class SystemAccessResult:
    """The complete result of one :func:`provision_system_principal_access`
    run."""

    role: str
    outcomes: tuple[SystemAccessOutcome, ...]

    @property
    def all_admitted(self) -> bool:
        """Always ``True`` for a returned result, by construction —
        :func:`provision_system_principal_access` raises
        :class:`SystemAdmissionError` immediately on the first failure
        rather than ever returning a partial result (fail closed), mirroring
        :attr:`~agent_utilities.security.tenant_rbac_admission.TenantAccessResult.all_admitted`.
        """
        return len(self.outcomes) > 0


@runtime_checkable
class SystemAdmissionClient(Protocol):
    """The minimal engine surface system-principal admission needs —
    ``get_identity``/``register_identity`` (from
    :class:`~agent_utilities.security.tenant_rbac_admission.EngineIdentityClient`)
    plus ``add_role``/``add_grant`` (from
    :class:`~agent_utilities.security.engine_rbac_admission.EngineAdmissionClient`),
    because this module must both mint a role/grant (like Tier-2 admission's
    ``admin_grant`` mode) AND enroll a specific principal into it (like
    tenant admission) — the two precedents this module composes."""

    def register_identity(
        self,
        *,
        agent_id: str,
        role: IdentityRole,
        teams: list[str],
        roles: list[str],
        signer_id: str,
        signer_key: str,
    ) -> str: ...

    def get_identity(self, agent_id: str) -> Mapping[str, Any] | None: ...

    def add_role(self, role: str) -> str: ...

    def add_grant(
        self, role: str, resource: dict[str, str] | str, action: str, effect: str
    ) -> str: ...


class FixtureSystemAdmissionClient:
    """In-memory :class:`SystemAdmissionClient` double reimplementing enough
    of ``IsolationLayer::check_access``'s role-grant evaluation
    (``crates/eg-core/src/isolation.rs``) to let a test PROVE reachability
    (:meth:`_has_access`) end-to-end, not merely record calls — the same
    "fixture proves the real semantics" discipline
    :class:`~agent_utilities.security.engine_rbac_admission.FixtureEngineAdmissionClient`
    documents.

    WIRE-FIRST (D-OB-9) NOTE: only ever constructed by
    ``tests/unit/security/test_system_rbac_admission.py`` and this module's
    own dry-run preview path — never a socket.
    """

    def __init__(self, *, identity_read_authorized: bool = True) -> None:
        #: agent_id -> {"role": IdentityRole, "teams": list[str], "roles": list[str]}
        self.identities: dict[str, dict[str, Any]] = {}
        self.roles: set[str] = set()
        #: (role, resource_repr, action, effect)
        self.grants: set[tuple[str, str, str, str]] = set()
        self.calls: list[tuple[str, tuple[Any, ...]]] = []
        self.identity_read_authorized = identity_read_authorized

    def register_identity(
        self,
        *,
        agent_id: str,
        role: IdentityRole,
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

    def get_identity(self, agent_id: str) -> Mapping[str, Any] | None:
        """Return the same complete shape as the engine's read-only RPC."""

        self.calls.append(("get_identity", (agent_id,)))
        if not self.identity_read_authorized:
            raise SystemAdmissionError(
                "ACCESS_DENIED: verified principal lacks admin capability "
                "required for 'security:admin'"
            )
        identity = self.identities.get(agent_id)
        if identity is None:
            return None
        return {
            "agent_id": agent_id,
            "role": identity["role"],
            "teams": list(identity["teams"]),
            "roles": list(identity["roles"]),
        }

    def add_role(self, role: str) -> str:
        self.calls.append(("add_role", (role,)))
        self.roles.add(role)
        return "role_added"

    def add_grant(
        self, role: str, resource: dict[str, str] | str, action: str, effect: str
    ) -> str:
        self.calls.append(("add_grant", (role, resource, action, effect)))
        self.grants.add((role, repr(resource), action, effect))
        return "grant_added"

    def _has_access(self, agent_id: str, action: str) -> bool:
        """Reimplements the (non-``System``) role-grant branch of
        ``IsolationLayer::check_access`` exactly: an agent qualifies for
        ``action`` on ``Graph(CONTROL_GRAPH_NAME)`` only if it is a
        registered identity holding a role for which an ``Allow`` grant of
        that exact action, on that exact resource, was added. Deliberately
        excludes the ``System``-bypass branch — this fixture exists to
        prove the OPPOSITE: that a narrow role is sufficient, never that
        ``System`` is required."""

        identity = self.identities.get(agent_id)
        if not identity:
            return False
        agent_roles = set(identity.get("roles", []))
        target = repr({"Graph": CONTROL_GRAPH_NAME})
        for role, resource_repr, act, effect in self.grants:
            if (
                role in agent_roles
                and act == action
                and effect == "Allow"
                and resource_repr == target
            ):
                return True
        return False


class LiveSystemAdmissionClient:
    """Identity read/write adapter over the real engine's ``ConsensusClient``/
    ``RbacClient``, via the SAME process-authority path
    :class:`~agent_utilities.security.tenant_rbac_admission.LiveEngineIdentityClient`
    /
    :class:`~agent_utilities.security.engine_rbac_admission.LiveEngineAdmissionClient`
    use. Constructing this class does nothing by itself; every method call
    is a real RPC. Only ever instantiated by
    :func:`ensure_system_principal_access` / the deployment CLI — never
    implicitly, and NEVER against a live cluster from this repository's own
    tests (see the module docstring's PREPARE-ONLY rule)."""

    def __init__(self, *, config: Any = None) -> None:
        self._config = config

    def _client(self) -> Any:
        from ..knowledge_graph.core.graph_compute import GraphComputeEngine

        return GraphComputeEngine.get_or_create().client

    def get_identity(self, agent_id: str) -> Mapping[str, Any] | None:
        """Delegate the read-only identity lookup to EG, failing closed."""

        try:
            with _identity_store_scope():
                reader = getattr(self._client().consensus, "get_identity", None)
                if not callable(reader):
                    raise SystemAdmissionError(
                        "engine client lacks required consensus.get_identity capability"
                    )
                identity = reader(agent_id)
        except SystemAdmissionError:
            raise
        except Exception as exc:
            raise SystemAdmissionError(
                f"engine get_identity({agent_id!r}) failed"
            ) from exc
        if identity is None:
            return None
        if not isinstance(identity, Mapping):
            raise SystemAdmissionError(
                f"engine get_identity({agent_id!r}) returned a non-mapping identity"
            )
        return dict(identity)

    def register_identity(
        self,
        *,
        agent_id: str,
        role: IdentityRole,
        teams: list[str],
        roles: list[str],
        signer_id: str,
        signer_key: str,
    ) -> str:
        try:
            with _identity_store_scope():
                return str(
                    self._client().consensus.register_identity(
                        agent_id,
                        role,
                        teams,
                        roles,
                        signer_id=signer_id,
                        signer_key=signer_key,
                    )
                )
        except Exception as exc:
            raise SystemAdmissionError(
                f"engine register_identity({agent_id!r}, roles={roles!r}) failed"
            ) from exc

    def add_role(self, role: str) -> str:
        try:
            return str(self._client().rbac.add_role(role))
        except Exception as exc:
            raise SystemAdmissionError(f"engine add_role({role!r}) failed") from exc

    def add_grant(
        self, role: str, resource: dict[str, str] | str, action: str, effect: str
    ) -> str:
        try:
            return str(self._client().rbac.add_grant(role, resource, action, effect))
        except Exception as exc:
            raise SystemAdmissionError(
                f"engine add_grant({role!r}, {resource!r}, {action!r}, {effect!r}) failed"
            ) from exc


def resolve_system_admission_client(config: Any = None) -> SystemAdmissionClient:
    """Return a :class:`LiveSystemAdmissionClient` bound to ``config``.
    Construction never connects by itself — mirrors
    :func:`~agent_utilities.security.tenant_rbac_admission.resolve_engine_identity_client`.
    """

    return LiveSystemAdmissionClient(config=config)


_IDENTITY_FIELDS = frozenset({"agent_id", "role", "teams", "roles"})


def _identity_string_list(
    agent_id: str, identity: Mapping[str, Any], field: str
) -> tuple[str, ...]:
    values = identity[field]
    if not isinstance(values, list):
        raise SystemAdmissionError(
            f"cannot admit {agent_id!r}: GetIdentity returned invalid {field}"
        )
    detached: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not isinstance(value, str) or not value.strip() or value in seen:
            raise SystemAdmissionError(
                f"cannot admit {agent_id!r}: GetIdentity returned invalid {field}"
            )
        seen.add(value)
        detached.append(value)
    return tuple(detached)


def _principal_from_identity(
    agent_id: str, identity: Mapping[str, Any] | None
) -> SystemPrincipal | None:
    """Build a complete replacing-upsert input from one confirmed EG read."""

    if identity is None:
        return SystemPrincipal(agent_id=agent_id, existing_roles=())
    if set(identity) != _IDENTITY_FIELDS:
        raise SystemAdmissionError(
            f"cannot admit {agent_id!r}: GetIdentity returned an incomplete "
            "identity; expected agent_id, role, teams, and roles"
        )
    if identity["agent_id"] != agent_id:
        raise SystemAdmissionError(
            f"cannot admit {agent_id!r}: GetIdentity returned a mismatched agent_id"
        )
    try:
        role = _copy_identity_role(identity["role"])
        teams = _identity_string_list(agent_id, identity, "teams")
        existing_roles = _identity_string_list(agent_id, identity, "roles")
        if role == "System":
            return None
        return SystemPrincipal(
            agent_id=agent_id,
            role=role,
            teams=teams,
            existing_roles=existing_roles,
        )
    except ValueError as exc:
        raise SystemAdmissionError(
            f"cannot admit {agent_id!r}: GetIdentity returned an invalid identity role"
        ) from exc


def _principal_for_admission(
    client: SystemAdmissionClient,
    agent_id: str,
) -> SystemPrincipal | None:
    """Read the complete identity under EG's admin authorization contract."""

    return _principal_from_identity(agent_id, client.get_identity(agent_id))


def _complete_principal_roles(principal: SystemPrincipal, role: str) -> tuple[str, ...]:
    """Return a confirmed complete role set before any policy mutation."""

    if principal.existing_roles is None:
        raise SystemAdmissionError(
            f"cannot admit {principal.agent_id!r} into {role!r}: "
            "existing_roles is unknown (None). RegisterIdentity REPLACES "
            "a principal's whole role set, so direct provisioning requires "
            "the complete current roles or an explicit empty tuple ()."
        )
    return principal.existing_roles


def _admit_principal(
    client: SystemAdmissionClient,
    principal: SystemPrincipal,
    existing_roles: tuple[str, ...],
    admin_authority: AdmissionAuthority,
    role: str,
) -> SystemAccessOutcome:
    """Return a no-op or perform one complete replacing identity upsert."""

    if role in existing_roles:
        return SystemAccessOutcome(
            agent_id=principal.agent_id,
            role=role,
            already_held=True,
            detail=f"{principal.agent_id!r} already carries {role!r}",
        )
    merged_roles = sorted({*existing_roles, role})
    client.register_identity(
        agent_id=principal.agent_id,
        role=principal.role,
        teams=list(principal.teams),
        roles=merged_roles,
        signer_id=admin_authority.signer_id,
        signer_key=admin_authority.signer_key,
    )
    return SystemAccessOutcome(
        agent_id=principal.agent_id,
        role=role,
        already_held=False,
        detail=(
            f"granted {role!r} (Read+Write on "
            f"Graph({CONTROL_GRAPH_NAME!r})) to {principal.agent_id!r} "
            f"(roles now {merged_roles!r})"
        ),
    )


def provision_system_principal_access(
    client: SystemAdmissionClient,
    principals: list[SystemPrincipal],
    *,
    admin_authority: AdmissionAuthority,
    role: str = CONTROL_ROLE_NAME,
) -> SystemAccessResult:
    """Idempotently mint ``role`` (Read + Write ``Allow`` grants on
    ``Graph(CONTROL_GRAPH_NAME)``) and enroll every principal in
    ``principals`` into it.

    Two steps, both upserts (safe to run on every boot / every call):

    1. ``client.add_role(role)`` then ``client.add_grant(role,
       {"Graph": CONTROL_GRAPH_NAME}, "Read", "Allow")`` and the same for
       ``"Write"`` — see the module docstring for the source trace proving
       the scheduler tick genuinely mutates the control graph, so Write is
       not granted speculatively.
    2. For each principal: raises immediately if ``principal.existing_roles``
       is ``None`` (unknown — see the module docstring, "Mirror-image
       defect"; never guesses an empty set). Otherwise: a no-op
       (``already_held=True``) when ``role in principal.existing_roles``;
       else re-registers the identity with its EXACT existing
       ``role``/``teams`` plus ``role`` appended to ``existing_roles`` —
       never dropping a role/team the caller did not ask to change
       (``RegisterIdentity`` replaces the whole identity, so this always
       sends the FULL desired shape. The boot wrapper obtains that shape from
       ``GetIdentity`` before calling this direct provisioning composition).

    Refuses ``role="System"`` outright with a :class:`ValueError` (see
    module docstring, "Two designs rejected") — a caller programming error,
    not an RPC failure, so it is never conflated with
    :class:`SystemAdmissionError`.

    A failure on any RPC, or on an unknown ``existing_roles``, raises
    immediately — fail closed, never leave a partial admission unreported.
    """

    if role == "System":
        raise ValueError(
            "provision_system_principal_access: role must never be 'System' "
            "— System bypasses RBAC entirely (see module docstring, "
            "'Two designs rejected')"
        )
    if not principals:
        raise ValueError("principals must be non-empty")

    known_principals = [
        (principal, _complete_principal_roles(principal, role))
        for principal in principals
    ]

    client.add_role(role)
    client.add_grant(role, {"Graph": CONTROL_GRAPH_NAME}, "Read", "Allow")
    client.add_grant(role, {"Graph": CONTROL_GRAPH_NAME}, "Write", "Allow")

    outcomes = [
        _admit_principal(client, principal, existing_roles, admin_authority, role)
        for principal, existing_roles in known_principals
    ]

    return SystemAccessResult(role=role, outcomes=tuple(outcomes))


# ── Process-local cache + backoff (mirrors agent_webui.graph_admission) ─────
_ADMITTED: dict[tuple[str, str], float] = {}
_FAILURES: dict[tuple[str, str], tuple[float, SystemAdmissionError]] = {}
_STATE_LOCK = threading.Lock()
_KEY_LOCKS: dict[tuple[str, str], threading.Lock] = {}


def _lock_for(key: tuple[str, str]) -> threading.Lock:
    with _STATE_LOCK:
        lock = _KEY_LOCKS.get(key)
        if lock is None:
            lock = threading.Lock()
            _KEY_LOCKS[key] = lock
        return lock


def _reset_admission_cache_for_tests() -> None:
    """Test-only reset of the process-local admission cache. The engine
    exposes no equivalent reset RPC — this exists purely so
    ``tests/unit/security/test_system_rbac_admission.py`` cases do not leak
    cache state into one another (module-level state, by design — see
    :func:`ensure_system_principal_access`'s own docstring for why it is
    process-lifetime, not per-call)."""

    with _STATE_LOCK:
        _ADMITTED.clear()
        _FAILURES.clear()
        _KEY_LOCKS.clear()


def _validated_admission_key(agent_id: str, role: str) -> tuple[str, tuple[str, str]]:
    """Validate public request fields and return the process-cache key."""

    normalized_agent_id = str(agent_id or "").strip()
    if not normalized_agent_id:
        raise ValueError("agent_id must be a non-empty opaque identifier")
    if role == "System":
        raise ValueError(
            "ensure_system_principal_access: role must never be 'System' — "
            "System bypasses RBAC entirely"
        )
    return normalized_agent_id, (role, normalized_agent_id)


def _attempt_system_principal_access(
    client: SystemAdmissionClient,
    agent_id: str,
    role: str,
) -> SystemAccessOutcome:
    """Perform the admin-gated read and any required narrow admission."""

    principal = _principal_for_admission(client, agent_id)
    if principal is None:
        return SystemAccessOutcome(
            agent_id=agent_id,
            role=role,
            already_held=True,
            detail=(
                f"{agent_id!r} already has engine System authority; "
                "no narrower role or identity write was made"
            ),
        )

    from .admission_authority import resolve_admission_authority

    authority = resolve_admission_authority()
    result = provision_system_principal_access(
        client,
        [principal],
        admin_authority=authority,
        role=role,
    )
    return result.outcomes[0]


def ensure_system_principal_access(
    agent_id: str,
    *,
    role: str = CONTROL_ROLE_NAME,
    client: SystemAdmissionClient | None = None,
) -> SystemAccessOutcome:
    """Verify or, under existing Admin authority, admit au's own process
    principal ``agent_id`` into the control-graph role idempotently (see the
    module docstring, "Boot verification and operator-authorized remediation").

    * **Positive outcome** — cached in-process, forever (this process's
      lifetime). A returning call for the same ``(role, agent_id)`` is a
      dict lookup, never a round trip.
    * **Negative outcome** (missing provisioner credential — NE-021 today
      — an engine RPC/read-back failure, or a malformed identity shape)
      — cached for :data:`_FAILURE_BACKOFF_SECONDS`, so a still-broken
      precondition is not retried on every call, while the next call after
      the backoff window retries automatically — an operator fixing NE-021
      is picked up without a process restart.
    * Concurrent callers for the same key collapse onto one attempt via a
      per-key lock (double-checked against the cache once the lock is
      held).

    The function always reads ``GetIdentity`` first; there is no partial
    ``existing_roles`` override because roles without the current role/teams
    shape would make the replacing upsert destructive. A confirmed absent
    identity becomes the sole empty-role case. A System identity returns an
    authorized no-op. An ordinary identity proceeds only after the
    ``security:admin``-gated read succeeds, preserving its complete
    ``role``/``teams``/``roles`` shape. Any denial, failed read, or malformed
    response stops before a write.

    Raises :class:`SystemAdmissionError` on a negative outcome — never
    silently proceeds and never returns a value that looks like success.
    The caller (``kg_server.py``'s daemon bootstrap path) is responsible
    for catching this, logging it, and continuing to serve degraded — this
    function itself never catches its own failure, so it never pretends
    admission succeeded (see module docstring, NE-021).
    """

    agent_id, key = _validated_admission_key(agent_id, role)
    with _STATE_LOCK:
        if key in _ADMITTED:
            return SystemAccessOutcome(
                agent_id=agent_id,
                role=role,
                already_held=True,
                detail=f"{agent_id!r} already admitted into {role!r} (cached)",
            )

    lock = _lock_for(key)
    with lock:
        with _STATE_LOCK:
            if key in _ADMITTED:
                return SystemAccessOutcome(
                    agent_id=agent_id,
                    role=role,
                    already_held=True,
                    detail=f"{agent_id!r} already admitted into {role!r} (cached)",
                )
            failure = _FAILURES.get(key)
        if failure is not None:
            attempted_at, cached_exc = failure
            if time.monotonic() - attempted_at < _FAILURE_BACKOFF_SECONDS:
                raise cached_exc

        try:
            live_client = (
                client if client is not None else resolve_system_admission_client()
            )
            outcome = _attempt_system_principal_access(live_client, agent_id, role)
        except SystemAdmissionError as exc:
            with _STATE_LOCK:
                _FAILURES[key] = (time.monotonic(), exc)
            raise
        except Exception as exc:  # noqa: BLE001 - normalize to our own error type
            wrapped = SystemAdmissionError(
                f"system-principal admission failed for {agent_id!r}: {exc}"
            )
            with _STATE_LOCK:
                _FAILURES[key] = (time.monotonic(), wrapped)
            raise wrapped from exc

        with _STATE_LOCK:
            _ADMITTED[key] = time.monotonic()
            _FAILURES.pop(key, None)

        from .persistence_privacy import persistence_reference

        logger.info(
            "system-principal access verified: %s role=%s already-held=%s",
            persistence_reference("agent", agent_id, namespace="rbac-admission"),
            role,
            outcome.already_held,
        )
        return outcome
