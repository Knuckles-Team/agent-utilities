#!/usr/bin/python
from __future__ import annotations

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
  supplies the FULL desired shape, never a read-then-merge" convention (the
  engine exposes no ``GetIdentity``/``ListIdentities`` RPC).
* :mod:`~agent_utilities.security.engine_rbac_admission` — the
  ``add_role``/``add_grant`` pair for minting a **narrow, named** role
  (never bare ``System``) and granting it directly, rather than only
  enrolling into a role the engine happens to create as a side effect.
* ``agent-webui``'s ``agent_webui.graph_admission.ensure_tenant_admission``
  (BUG-286) — the runtime shape for a **process-local, cache-after,
  backoff-on-failure** admission call:  a positive outcome is cached
  forever for this process's lifetime; a negative one is cached for
  :data:`_FAILURE_BACKOFF_SECONDS` so a still-broken precondition (e.g. the
  missing provisioner credential — see NE-021 below) is not re-attempted on
  every single call, while still self-healing without a restart once an
  operator fixes it; concurrent callers for the same key collapse onto one
  attempt via a per-key lock.

Auto-admission at boot, not operator-gated — the explicit choice, and why
-----------------------------------------------------------------------------
Unlike a WebUI end-user principal (minted dynamically, one per signed-in
human, no static roster), au's own scheduler identity is static and known
at deploy time — which is an argument FOR operator-gating this the same way
Tier-2 admission is (``tier2_admission_cli.py``, a deliberate manual/CI
deploy step, never automatic).

This module chooses **auto-admission at process boot** instead
(:func:`ensure_system_principal_access`, called once from
``kg_server.py``'s daemon-role bootstrap path — see that module's call
site), for three reasons specific to this defect:

1. **This *is* the outage.** BUG-295 is a P0 precisely because nothing ever
   performs this admission. An operator-gated-only design would mean fixing
   NE-021 (seeding the missing provisioner credential) is not sufficient by
   itself to end the outage — an operator would *also* have to remember to
   run the CLI and restart every graph-os daemon pod. Auto-admission
   collapses that to: seed the credential, and the next daemon boot
   (already a routine rollout event) self-heals.
2. **The blast radius is already bounded to the narrow, reviewed role**
   this module grants — never ``System``, never a wider selector than
   ``Graph("__control__")``. Auto-provisioning a *narrow, purpose-named*
   role is a materially different risk than auto-provisioning admin
   capability (which is why Tier-2 stays operator-gated) or than the
   engine auto-assigning roles on its own (rejected above).
3. **It degrades honestly, so "auto" never means "silently pretend it
   worked."** See NE-021 below — a missing credential or an unreachable
   engine is a caught, logged, actionable, non-secret-leaking condition,
   never a crash and never a false success. A subsequent scheduler tick
   failing remains visibly attributable to the same root cause.

The operator-gated path stays reachable for the cases auto-admission does
not cover: pre-provisioning before a rollout, an environment that
deliberately wants a manual step, or re-running by hand after fixing
NE-021 without waiting for the next boot — ship
:mod:`agent_utilities.security.system_admission_cli`, mirroring
``tenant_admission_cli.py``/``tier2_admission_cli.py`` exactly (manifest
JSON, dry-run unless ``--apply``). Both paths call the SAME
:func:`provision_system_principal_access` composition, so they always
produce identical provisioning for the same principal.

NE-021 — the provisioner credential does not exist in either secrets
backend on the target deployment (verified metadata-only; this
deployment's ``SecretsClient`` backend is ``"engine"``, not OpenBao).
:func:`ensure_system_principal_access` and
:func:`resolve_provisioner_authority` therefore cannot be activated live
until an operator supplies ``engine-admission/provisioner``. This is
expected and handled, not a bug in this module: a missing credential
raises :class:`SystemAdmissionError` naming exactly the missing secret key
and the CLI command to seed it (never the secret value itself — this
module never mints, prints, logs, or persists one, matching every sibling
admission module's doctrine, AGENTS.md "Secrets & credential retrieval").
The ``kg_server.py`` call site catches this, logs it once per backoff
window, and continues serving degraded — exactly today's (broken)
scheduler behavior, but now with an actionable diagnosis instead of a bare
``CypherEngineError`` repeating forever with no explanation.

**Prod operations are PREPARE-ONLY here too**, same as
``tenant_rbac_admission.py``/``engine_rbac_admission.py``/
``graph_ownership_apply.py``. No test in this repository ever calls
:func:`provision_system_principal_access` with a
:class:`LiveSystemAdmissionClient` against a real engine.
"""

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from ..knowledge_graph.core.shard_topology import CONTROL_GRAPH_NAME

logger = logging.getLogger(__name__)

__all__ = [
    "CONTROL_ROLE_NAME",
    "DEFAULT_PROVISIONER_SECRET_KEY",
    "FixtureSystemAdmissionClient",
    "LiveSystemAdmissionClient",
    "SystemAccessOutcome",
    "SystemAccessResult",
    "SystemAdmissionAuthority",
    "SystemAdmissionClient",
    "SystemAdmissionError",
    "SystemPrincipal",
    "ensure_system_principal_access",
    "provision_system_principal_access",
    "resolve_provisioner_authority",
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


#: Reuses the SAME provisioner credential ``tier2_admission_cli.py`` /
#: ``tenant_admission_cli.py`` read — one already-admitted provisioner
#: identity is the signer for every engine-admin RPC this repo's
#: deployment tooling issues, never a separate credential per bridge.
DEFAULT_PROVISIONER_SECRET_KEY = "engine-admission/provisioner"

#: How long a NEGATIVE outcome (missing provisioner credential, or the
#: engine/admission RPC unreachable) is remembered before the next call
#: retries. Bounds the cost of a still-broken precondition without
#: requiring a process restart once it is fixed — mirrors
#: ``agent_webui.graph_admission._FAILURE_BACKOFF_SECONDS`` exactly.
_FAILURE_BACKOFF_SECONDS = 30.0


@dataclass(frozen=True, slots=True)
class SystemAdmissionAuthority:
    """An already-verified engine identity's signing credentials, resolved by
    the CALLER (from the configured secrets backend) — mirrors
    :class:`~agent_utilities.security.engine_rbac_admission.AdmissionAuthority`
    /
    :class:`~agent_utilities.security.tenant_rbac_admission.TenantAdmissionAuthority`
    exactly; this module never resolves, mints, or persists a credential
    itself.
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
        return (
            f"SystemAdmissionAuthority(agent_id={self.agent_id!r}, "
            f"signer_id={self.signer_id!r}, signer_key=<redacted>)"
        )


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
    ``RegisterIdentity`` upsert (the engine exposes no identity read-back
    RPC — see the module docstring)."""

    agent_id: str
    role: str = "Agent"
    teams: tuple[str, ...] = ()
    existing_roles: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.agent_id.strip():
            raise ValueError("agent_id must be a non-empty opaque identifier")
        if self.role == "System":
            raise ValueError(
                "SystemPrincipal.role must never be 'System' — System bypasses "
                "RBAC entirely; this module grants a narrow named role instead "
                "(see the module docstring, 'Two designs rejected')"
            )


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
    ``register_identity`` (from
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
        role: str,
        teams: list[str],
        roles: list[str],
        signer_id: str,
        signer_key: str,
    ) -> str: ...

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

    def __init__(self) -> None:
        #: agent_id -> {"role": str, "teams": list[str], "roles": list[str]}
        self.identities: dict[str, dict[str, Any]] = {}
        self.roles: set[str] = set()
        #: (role, resource_repr, action, effect)
        self.grants: set[tuple[str, str, str, str]] = set()
        self.calls: list[tuple[str, tuple[Any, ...]]] = []

    def register_identity(
        self,
        *,
        agent_id: str,
        role: str,
        teams: list[str],
        roles: list[str],
        signer_id: str,
        signer_key: str,
    ) -> str:
        self.calls.append(
            ("register_identity", (agent_id, role, tuple(teams), tuple(roles)))
        )
        self.identities[agent_id] = {
            "role": role,
            "teams": list(teams),
            "roles": list(roles),
        }
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
    """Mutating adapter over the real engine's ``ConsensusClient``/
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

    def register_identity(
        self,
        *,
        agent_id: str,
        role: str,
        teams: list[str],
        roles: list[str],
        signer_id: str,
        signer_key: str,
    ) -> str:
        try:
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


def provision_system_principal_access(
    client: SystemAdmissionClient,
    principals: list[SystemPrincipal],
    *,
    admin_authority: SystemAdmissionAuthority,
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
    2. For each principal: a no-op (``already_held=True``) when ``role in
       principal.existing_roles``; otherwise re-registers the identity with
       its EXACT existing ``role``/``teams`` plus ``role`` appended to
       ``existing_roles`` — never dropping a role/team the caller did not
       ask to change (``RegisterIdentity`` replaces the whole identity, so
       this always sends the FULL desired shape — see the module docstring
       for why no read-back is attempted).

    Refuses ``role="System"`` outright with a :class:`ValueError` (see
    module docstring, "Two designs rejected") — a caller programming error,
    not an RPC failure, so it is never conflated with
    :class:`SystemAdmissionError`.

    A failure on any RPC raises immediately — fail closed, never leave a
    partial admission unreported.
    """

    if role == "System":
        raise ValueError(
            "provision_system_principal_access: role must never be 'System' "
            "— System bypasses RBAC entirely (see module docstring, "
            "'Two designs rejected')"
        )
    if not principals:
        raise ValueError("principals must be non-empty")

    client.add_role(role)
    client.add_grant(role, {"Graph": CONTROL_GRAPH_NAME}, "Read", "Allow")
    client.add_grant(role, {"Graph": CONTROL_GRAPH_NAME}, "Write", "Allow")

    outcomes: list[SystemAccessOutcome] = []
    for principal in principals:
        if role in principal.existing_roles:
            outcomes.append(
                SystemAccessOutcome(
                    agent_id=principal.agent_id,
                    role=role,
                    already_held=True,
                    detail=f"{principal.agent_id!r} already carries {role!r}",
                )
            )
            continue
        merged_roles = sorted({*principal.existing_roles, role})
        client.register_identity(
            agent_id=principal.agent_id,
            role=principal.role,
            teams=list(principal.teams),
            roles=merged_roles,
            signer_id=admin_authority.signer_id,
            signer_key=admin_authority.signer_key,
        )
        outcomes.append(
            SystemAccessOutcome(
                agent_id=principal.agent_id,
                role=role,
                already_held=False,
                detail=(
                    f"granted {role!r} (Read+Write on "
                    f"Graph({CONTROL_GRAPH_NAME!r})) to {principal.agent_id!r} "
                    f"(roles now {merged_roles!r})"
                ),
            )
        )

    return SystemAccessResult(role=role, outcomes=tuple(outcomes))


def resolve_provisioner_authority(
    *, secrets_client: Any = None, key: str = DEFAULT_PROVISIONER_SECRET_KEY
) -> SystemAdmissionAuthority:
    """Resolve the provisioner's signer credentials from the configured
    secrets backend. Never returns a placeholder — a missing or malformed
    secret is a :class:`SystemAdmissionError` naming exactly the missing key
    (never the value — this module never mints, prints, logs, or persists a
    secret), matching
    :func:`~agent_utilities.security.tenant_admission_cli.resolve_provisioner_authority`
    /
    :func:`~agent_utilities.security.tier2_admission_cli.resolve_provisioner_authority`.

    This is the exact NE-021 condition: on the target deployment this
    raises every time, because the ``engine-admission/provisioner`` secret
    exists in neither configured secrets backend yet."""

    if secrets_client is None:
        from .secrets_client import create_secrets_client

        secrets_client = create_secrets_client()

    raw = secrets_client.get(key)
    if not raw:
        raise SystemAdmissionError(
            f"no provisioner credential at secret key {key!r} — an operator "
            "must seed it once via `python -m agent_utilities.security.cli "
            f"set {key} --value-ref <vault://...>` before system-principal "
            "admission can be applied against a real engine (NE-021); until "
            "then the scheduler's control-graph access remains unprovisioned "
            "and every scheduler tick keeps failing with the same "
            "CypherEngineError this admission pass exists to fix. For the full "
            "provisioning procedure (dedicated signer, the matching engine-side "
            "EPISTEMIC_GRAPH_SIGNER_KEYS_JSON entry, and how to verify it "
            "actually worked) see "
            "agent_utilities/skills/workflows/agent-os-genesis/references/"
            "engine-identity-admission.md"
        )

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SystemAdmissionError(
            f"secret key {key!r} is not valid JSON — expected "
            '{"agent_id": ..., "signer_id": ..., "signer_key": ...}'
        ) from exc
    if not isinstance(payload, dict):
        raise SystemAdmissionError(
            f"secret key {key!r} must decode to a JSON object with "
            "agent_id/signer_id/signer_key"
        )
    try:
        return SystemAdmissionAuthority(
            agent_id=str(payload["agent_id"]),
            signer_id=str(payload["signer_id"]),
            signer_key=str(payload["signer_key"]),
        )
    except (KeyError, ValueError) as exc:
        raise SystemAdmissionError(
            f"secret key {key!r} is missing or has an invalid "
            "agent_id/signer_id/signer_key"
        ) from exc


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


def ensure_system_principal_access(
    agent_id: str,
    *,
    role: str = CONTROL_ROLE_NAME,
    client: SystemAdmissionClient | None = None,
    secrets_client: Any = None,
    secret_key: str = DEFAULT_PROVISIONER_SECRET_KEY,
) -> SystemAccessOutcome:
    """Ensure au's own process principal ``agent_id`` is admitted into the
    control-graph role, idempotently — the boot-time auto-admission
    entrypoint (see module docstring, "Auto-admission at boot").

    * **Positive outcome** — cached in-process, forever (this process's
      lifetime). A returning call for the same ``(role, agent_id)`` is a
      dict lookup, never a round trip.
    * **Negative outcome** (missing provisioner credential — NE-021 today
      — or an engine RPC failure) — cached for
      :data:`_FAILURE_BACKOFF_SECONDS`, so a still-broken precondition is
      not retried on every call, while the next call after the backoff
      window retries automatically — an operator fixing NE-021 is picked
      up without a process restart.
    * Concurrent callers for the same key collapse onto one attempt via a
      per-key lock (double-checked against the cache once the lock is
      held).

    Raises :class:`SystemAdmissionError` on a negative outcome — never
    silently proceeds and never returns a value that looks like success.
    The caller (``kg_server.py``'s daemon bootstrap path) is responsible
    for catching this, logging it, and continuing to serve degraded — this
    function itself never catches its own failure, so it never pretends
    admission succeeded (see module docstring, NE-021).
    """

    agent_id = str(agent_id or "").strip()
    if not agent_id:
        raise ValueError("agent_id must be a non-empty opaque identifier")

    key = (role, agent_id)
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
            authority = resolve_provisioner_authority(
                secrets_client=secrets_client, key=secret_key
            )
            live_client = (
                client if client is not None else resolve_system_admission_client()
            )
            result = provision_system_principal_access(
                live_client,
                [SystemPrincipal(agent_id=agent_id)],
                admin_authority=authority,
                role=role,
            )
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

        outcome = result.outcomes[0]
        with _STATE_LOCK:
            _ADMITTED[key] = time.monotonic()
            _FAILURES.pop(key, None)

        from .persistence_privacy import persistence_reference

        logger.info(
            "system-principal admission: %s admitted into %s",
            persistence_reference("agent", agent_id, namespace="rbac-admission"),
            role,
        )
        return outcome
