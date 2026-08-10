#!/usr/bin/python
from __future__ import annotations

"""Engine-side Tier-2 RBAC admission (GOC-62 D3(a), closes the missing half of
BUG-030/BUG-038).

CONCEPT:AU-OS.identity.engine-rbac-admission — See
``plans/graph-os-completion-program/decisions/GOC-62-keycloak-auth-standard.md``
§2.1 for the full design. Short version:

The engine (``epistemic-graph``) gates 5 of its 86 ``authz_action`` strings — spanning
20 wire ``Method`` variants (``admin:cluster-read`` incl. ``PlacementRoute``,
``admin:cluster``, ``admin:backup``, ``admin:sqlite-file``, ``security:admin``) —
behind a **second, independent** check: ``IsolationLayer::has_admin_capability``
(``crates/eg-core/src/isolation.rs``), which consults ONLY a durable, engine-local
``agents``/RBAC store. No Keycloak scope, however broad, ever satisfies it — that
store is written only by an explicit ``RegisterIdentity``/``RbacAdmin`` engine RPC,
and nothing in Keycloak-side provisioning (``agent-webui/scripts/provision_identity.py``)
calls it. A fresh engine store therefore has NO admin identity at all, and every
Tier-2 action fails ``ACCESS_DENIED`` even for a service Keycloak granted the
matching scope to (BUG-030's live-cluster gap; BUG-038's "fresh-store deploy would
fail" proof).

This module is the missing bridge: an idempotent, source-controlled admission pass
run by deployment tooling immediately after Keycloak-side (Tier-1) provisioning.
It performs the SAME two-step admission a human operator did by hand on the live
cluster (per DEC-015 §4's verdict on commit ``ee35179``), but reproducibly, from
source, for every fresh store:

1. **Bootstrap** (once, only on a genuinely pristine store): the provisioner's own
   identity self-registers as the first ``System`` identity via
   ``ConsensusClient.bootstrap_system_identity`` — the engine's own one-time,
   ``security:bootstrap``-scoped gate (``crates/eg-core/src/isolation.rs``
   ``try_bootstrap_system_identity``). Not a Keycloak grant; this is the one
   admission step in the whole GOC-62 standard that intentionally is not
   IdP-derived, because the very first admin cannot be granted by an RBAC store
   that has no admin in it yet.
2. **Admission** (every deploy, idempotent): using an already-admitted identity's
   signer credentials, register or grant every service identity the Tier-1
   Keycloak manifest says needs a Tier-2 action — ``RegisterIdentity`` (System
   role) or ``RbacAdmin.AddGrant`` (a narrower ``Admin`` grant on
   ``Graph("__admin__")``). Both underlying engine ops are upserts (register/add-or
   -replace), so running this pass twice, or against a store where some identities
   are already admitted, is safe.

Mirrors the existing ``graph_ownership_apply.py`` Protocol/Fixture/Live client split
(the established pattern in this codebase for a mutating admin action gated behind
an explicit apply step, with tests that never touch a socket) — reused here rather
than reinvented.

**Never mints, prints, logs, or persists a signer key.** Every credential this
module touches is passed in by the caller (resolved from OpenBao/the configured
secret provider — this module's own job starts and ends at the engine RPC
boundary), consistent with this repo's "Secrets & credential retrieval" doctrine
(AGENTS.md): code never persists, echoes, or infers a secret value.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)

__all__ = [
    "AdmissionAuthority",
    "AdmissionOutcome",
    "AdmissionResult",
    "EngineAdmissionClient",
    "EngineAdmissionError",
    "FixtureEngineAdmissionClient",
    "LiveEngineAdmissionClient",
    "ServiceAdmissionEntry",
    "TIER2_ACTION_PREFIXES",
    "provision_tier2_admission",
    "resolve_engine_admission_client",
]

#: The exact 5 Tier-2 ``authz_action`` strings DEC-015 §2.3 / GOC-62 §2.1
#: enumerate — mirrors ``access.rs::is_admin_authz_action`` (``action ==
#: "security:admin" || action.starts_with("admin:")``) so a manifest entry can be
#: validated against the SAME prefix rule the engine itself enforces, not a second
#: hand-maintained copy.
TIER2_ACTION_PREFIXES: tuple[str, ...] = (
    "admin:cluster-read",
    "admin:cluster",
    "admin:backup",
    "admin:sqlite-file",
    "security:admin",
)


def is_tier2_action(action: str) -> bool:
    """Mirror ``eg-capabilities``'s ``is_admin_authz_action`` predicate exactly
    (``action == "security:admin" || action.starts_with("admin:")``) so this
    module's notion of "Tier-2" never drifts from the engine's own gate."""

    return action == "security:admin" or action.startswith("admin:")


@dataclass(frozen=True, slots=True)
class AdmissionAuthority:
    """An already-verified engine identity's signing credentials, resolved by the
    CALLER (from OpenBao / the configured secret provider) — this module never
    resolves, mints, or persists one itself.

    ``signer_key`` is deliberately excluded from ``repr``/``str`` so it can never
    end up in a log line, traceback, or report by accident.
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

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"AdmissionAuthority(agent_id={self.agent_id!r}, signer_id={self.signer_id!r}, signer_key=<redacted>)"


@dataclass(frozen=True, slots=True)
class ServiceAdmissionEntry:
    """One row of the Tier-2 admission manifest — the AU-side mirror of one
    ``SERVICE_ROLES`` entry (Keycloak/Tier-1) whose granted scopes include a
    Tier-2 action. ``agent_id`` MUST equal the value that will appear as the
    service's ``VerifiedRequestContext.agent_id``/``principal`` on its future
    requests (its Keycloak client id / subject) — GOC-62 §2.1 step 2's binding
    requirement — so the Tier-1 grant and this Tier-2 admission resolve to the
    SAME identity at request time.

    ``grant_mode``:
      - ``"system_role"`` — register as engine ``AgentRole::System`` (unconditional
        admin capability — the coarse, "needs everything" case, e.g. a cluster
        placement/topology service).
      - ``"admin_grant"`` — a narrower RBAC ``Admin`` grant on ``Graph("__admin__")``
        via a named ``role`` (does not require full ``System`` identity).
    """

    agent_id: str
    tier2_actions: tuple[str, ...]
    grant_mode: str = "system_role"
    role: str | None = None

    def __post_init__(self) -> None:
        if not self.agent_id.strip():
            raise ValueError("agent_id must be a non-empty opaque identifier")
        if not self.tier2_actions:
            raise ValueError(
                f"{self.agent_id!r}: tier2_actions must be non-empty — an entry "
                "with no Tier-2 action does not belong in this manifest"
            )
        for action in self.tier2_actions:
            if not is_tier2_action(action):
                raise ValueError(
                    f"{self.agent_id!r}: {action!r} is not a Tier-2 authz_action "
                    f"(must match one of {TIER2_ACTION_PREFIXES})"
                )
        if self.grant_mode not in ("system_role", "admin_grant"):
            raise ValueError(
                f"{self.agent_id!r}: grant_mode must be 'system_role' or 'admin_grant', got {self.grant_mode!r}"
            )
        if self.grant_mode == "admin_grant" and not self.role:
            raise ValueError(
                f"{self.agent_id!r}: grant_mode='admin_grant' requires an explicit role name"
            )


class EngineAdmissionError(RuntimeError):
    """A Tier-2 admission RPC failed. Never swallowed — restated from this
    standard's §6: a failed admission must fail the provisioning pass, not
    silently leave a service under-admitted (the exact BUG-033 shape, applied to
    provisioning instead of a graph write)."""


@dataclass(frozen=True, slots=True)
class AdmissionOutcome:
    """What happened for one :class:`ServiceAdmissionEntry`."""

    agent_id: str
    bootstrapped: bool
    admitted: bool
    detail: str


@dataclass(frozen=True, slots=True)
class AdmissionResult:
    """The complete result of one :func:`provision_tier2_admission` run."""

    bootstrap_attempted: bool
    bootstrap_succeeded: bool
    bootstrap_already_consumed: bool
    outcomes: tuple[AdmissionOutcome, ...]

    @property
    def all_admitted(self) -> bool:
        return all(o.admitted for o in self.outcomes)


@runtime_checkable
class EngineAdmissionClient(Protocol):
    """The minimal engine surface Tier-2 admission needs — small and
    Protocol-typed so :class:`FixtureEngineAdmissionClient` (every test) and
    :class:`LiveEngineAdmissionClient` (the real engine) are interchangeable,
    mirroring :class:`~agent_utilities.knowledge_graph.maintenance.graph_ownership_apply.RbacAdminClient`.
    """

    def bootstrap_system_identity(
        self, *, agent_id: str, signer_id: str, signer_key: str
    ) -> str: ...

    def register_identity(
        self,
        *,
        agent_id: str,
        role: str,
        signer_id: str,
        signer_key: str,
    ) -> str: ...

    def add_role(self, role: str) -> str: ...

    def add_grant(
        self, role: str, resource: dict[str, str] | str, action: str, effect: str
    ) -> str: ...


class FixtureEngineAdmissionClient:
    """In-memory double implementing :class:`EngineAdmissionClient` with the SAME
    admission semantics ``IsolationLayer`` enforces (verified by direct reading of
    ``crates/eg-core/src/isolation.rs``, cited in each method below) — not just a
    call recorder. Lets a test prove Tier-2 reachability (``has_admin_capability``)
    end-to-end without a live engine.
    """

    def __init__(self) -> None:
        #: agent_id -> role ("System" or a plain string role name)
        self.agents: dict[str, str] = {}
        #: (role, resource_repr, action, effect)
        self.grants: set[tuple[str, str, str, str]] = set()
        self.roles: set[str] = set()
        #: mirrors ``IdentityBootstrapState`` (isolation.rs:305-346) — pristine
        #: until the first successful bootstrap.
        self._bootstrap_consumed = False
        self.calls: list[tuple[str, tuple[Any, ...]]] = []

    def identity_bootstrap_pending(self) -> bool:
        """Test-only introspection (the real engine exposes no equivalent RPC —
        GOC-62 §D3(a)'s admission pass never queries this; it always ATTEMPTS
        bootstrap and reads the failure reason, exactly like production must)."""

        return not self._bootstrap_consumed and not self.agents

    def bootstrap_system_identity(
        self, *, agent_id: str, signer_id: str, signer_key: str
    ) -> str:
        self.calls.append(("bootstrap_system_identity", (agent_id, signer_id)))
        # isolation.rs try_bootstrap_system_identity: only while pending.
        if self._bootstrap_consumed or self.agents:
            raise EngineAdmissionError(
                "ACCESS_DENIED: identity bootstrap is not pending"
            )
        if signer_id != agent_id:
            raise EngineAdmissionError(
                "bootstrap requires matching explicit identities and only "
                "security:bootstrap authority"
            )
        self.agents[agent_id] = "System"
        self._bootstrap_consumed = True
        return "registered"

    def register_identity(
        self,
        *,
        agent_id: str,
        role: str,
        signer_id: str,
        signer_key: str,
    ) -> str:
        self.calls.append(("register_identity", (agent_id, role, signer_id)))
        # is_admin_authz_action("security:admin") gates RegisterIdentity itself —
        # the caller (signer_id) must already hold admin capability, exactly like
        # require_admin_capability (access.rs:921-948).
        if not self.has_admin_capability(signer_id):
            raise EngineAdmissionError(
                "ACCESS_DENIED: verified principal lacks admin capability "
                "required for 'security:admin'"
            )
        self.agents[agent_id] = role
        return "registered"

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

    def has_admin_capability(self, agent_id: str) -> bool:
        """Reimplements ``IsolationLayer::has_admin_capability`` (isolation.rs
        ~611-633) exactly: ``System`` role always qualifies; otherwise an RBAC
        ``Admin`` grant evaluated against ``ResourceContext::graph("__admin__")``.
        This is the assertion a conformance/provisioning test uses to PROVE a
        Tier-2 action became reachable — not just that a call was made."""

        if self.agents.get(agent_id) == "System":
            return True
        for role, resource_repr, action, effect in self.grants:
            if (
                self.agents.get(agent_id) == role
                and action == "Admin"
                and effect == "Allow"
                and resource_repr == repr({"Graph": "__admin__"})
            ):
                return True
        return False


class LiveEngineAdmissionClient:
    """Mutating adapter over the real engine's ``ConsensusClient``/``RbacClient``,
    via the SAME process-authority path
    :class:`~agent_utilities.knowledge_graph.maintenance.graph_ownership_apply.LiveRbacAdminClient`
    uses. Constructing this class does nothing by itself; every method call is a
    real RPC. Only ever instantiated by deployment tooling running the admission
    pass explicitly — never implicitly, never as a default, and NEVER against a
    live cluster from this pass (see the GOC-62 record's hard rules)."""

    def __init__(self, *, config: Any = None) -> None:
        self._config = config

    def _client(self) -> Any:
        from ..knowledge_graph.core.graph_compute import GraphComputeEngine

        return GraphComputeEngine.get_or_create().client

    def bootstrap_system_identity(
        self, *, agent_id: str, signer_id: str, signer_key: str
    ) -> str:
        try:
            return str(
                self._client().consensus.bootstrap_system_identity(
                    agent_id=agent_id, signer_id=signer_id, signer_key=signer_key
                )
            )
        except Exception as exc:
            raise EngineAdmissionError(
                f"engine bootstrap_system_identity({agent_id!r}) failed"
            ) from exc

    def register_identity(
        self,
        *,
        agent_id: str,
        role: str,
        signer_id: str,
        signer_key: str,
    ) -> str:
        try:
            return str(
                self._client().consensus.register_identity(
                    agent_id,
                    role,
                    [],
                    [],
                    signer_id=signer_id,
                    signer_key=signer_key,
                )
            )
        except Exception as exc:
            raise EngineAdmissionError(
                f"engine register_identity({agent_id!r}, {role!r}) failed"
            ) from exc

    def add_role(self, role: str) -> str:
        try:
            return str(self._client().rbac.add_role(role))
        except Exception as exc:
            raise EngineAdmissionError(f"engine add_role({role!r}) failed") from exc

    def add_grant(
        self, role: str, resource: dict[str, str] | str, action: str, effect: str
    ) -> str:
        try:
            return str(self._client().rbac.add_grant(role, resource, action, effect))
        except Exception as exc:
            raise EngineAdmissionError(
                f"engine add_grant({role!r}, {resource!r}, {action!r}, {effect!r}) failed"
            ) from exc


def resolve_engine_admission_client(config: Any = None) -> EngineAdmissionClient:
    """Return a :class:`LiveEngineAdmissionClient` bound to ``config``. Construction
    never connects by itself — mirrors
    :func:`~agent_utilities.knowledge_graph.maintenance.graph_ownership_apply.resolve_rbac_admin_client`.
    """

    return LiveEngineAdmissionClient(config=config)


def provision_tier2_admission(
    client: EngineAdmissionClient,
    manifest: list[ServiceAdmissionEntry],
    *,
    admin_authority: AdmissionAuthority,
    bootstrap_authority: AdmissionAuthority | None = None,
) -> AdmissionResult:
    """Run the two-step Tier-2 admission pass (GOC-62 §2.1) against ``client``.

    Idempotent and safe to run on every deploy:

    1. If ``bootstrap_authority`` is given, ATTEMPT bootstrap (never pre-check —
       the real engine exposes no "is bootstrap pending" RPC; production code must
       attempt-and-interpret-the-failure exactly like this does). Success means
       this was a genuinely fresh store and ``bootstrap_authority`` is now the
       admin identity. A failure whose message names bootstrap as already
       consumed is the EXPECTED, idempotent outcome on every subsequent run — not
       an error — and the pass proceeds using ``admin_authority``. Any OTHER
       failure (e.g. a malformed bootstrap credential) is not swallowed.
    2. For every ``manifest`` entry, admit it using ``admin_authority``'s signer
       credentials — ``register_identity`` (``system_role``) or
       ``add_role``+``add_grant`` (``admin_grant``). A failure here raises
       :class:`EngineAdmissionError` and the pass stops — restated from this
       standard's §6: a failed admission must fail the deploy step, never leave a
       service silently under-admitted.

    Returns a full :class:`AdmissionResult` naming exactly what happened per
    entry, so a deployment log records this as a reproducible, auditable step —
    never a silent hand operation.
    """

    bootstrap_attempted = bootstrap_authority is not None
    bootstrap_succeeded = False
    bootstrap_already_consumed = False

    if bootstrap_authority is not None:
        try:
            client.bootstrap_system_identity(
                agent_id=bootstrap_authority.agent_id,
                signer_id=bootstrap_authority.signer_id,
                signer_key=bootstrap_authority.signer_key,
            )
            bootstrap_succeeded = True
            logger.info(
                "engine RBAC admission: bootstrapped first System identity %r "
                "(fresh store)",
                bootstrap_authority.agent_id,
            )
        except EngineAdmissionError as exc:
            if "not pending" in str(exc):
                bootstrap_already_consumed = True
                logger.debug(
                    "engine RBAC admission: bootstrap already consumed "
                    "(expected on a non-fresh store) — proceeding with the "
                    "configured admin authority"
                )
            else:
                raise

    outcomes: list[AdmissionOutcome] = []
    for entry in manifest:
        if entry.grant_mode == "system_role":
            client.register_identity(
                agent_id=entry.agent_id,
                role="System",
                signer_id=admin_authority.signer_id,
                signer_key=admin_authority.signer_key,
            )
            outcomes.append(
                AdmissionOutcome(
                    agent_id=entry.agent_id,
                    bootstrapped=False,
                    admitted=True,
                    detail=f"registered as System (tier2_actions={entry.tier2_actions})",
                )
            )
        else:
            assert entry.role is not None  # enforced in __post_init__
            client.add_role(entry.role)
            client.add_grant(entry.role, {"Graph": "__admin__"}, "Admin", "Allow")
            outcomes.append(
                AdmissionOutcome(
                    agent_id=entry.agent_id,
                    bootstrapped=False,
                    admitted=True,
                    detail=(
                        f"granted role {entry.role!r} Admin on __admin__ "
                        f"(tier2_actions={entry.tier2_actions})"
                    ),
                )
            )

    return AdmissionResult(
        bootstrap_attempted=bootstrap_attempted,
        bootstrap_succeeded=bootstrap_succeeded,
        bootstrap_already_consumed=bootstrap_already_consumed,
        outcomes=tuple(outcomes),
    )
