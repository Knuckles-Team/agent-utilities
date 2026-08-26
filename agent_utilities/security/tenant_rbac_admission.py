#!/usr/bin/python
from __future__ import annotations

import contextlib

from .admission_authority import AdmissionAuthority

"""Engine-side tenant-graph Read/Write RBAC admission — the ordinary-access sibling
of :mod:`agent_utilities.security.engine_rbac_admission` (which covers only the 5
Tier-2 ``admin:*``/``security:admin`` actions).

CONCEPT:AU-OS.identity.tenant-rbac-admission — root-cause fix for the P0 "agent-webui
``/graph`` shows 0 nodes/0 edges" incident. Confirmed live: the engine logged 855
``CypherEngineError`` read failures plus an unredacted write failure —
``RuntimeError: ACCESS_DENIED: verified principal lacks Write access to graph
'tenant__homelab____commons__'`` — from ``agent_utilities/orchestration/agent_runner.py``.
The AU-side check (``GraphSession`` carrying ``kg:read``/``kg:write``) passed; the
rejection was the engine's OWN, independent ``IsolationLayer``/RBAC store
(``epistemic-graph`` ``crates/eg-core/src/isolation.rs::check_access``). Per
``plans/au-eg-program/HANDOFF-2026-07-22.md`` §7-8, ``tenant__homelab____commons__``
was created explicitly, by hand, as the bootstrapped ``System`` identity
(``homelab-system``) — ``System`` unconditionally bypasses RBAC
(``isolation.rs`` ~724-726), so that one-off operator action never needed, and
never created, an actual Read/Write grant for anyone else. Every ordinary
(non-``System``) principal — including every real agent-webui end-user — has
therefore always been denied on this graph, because no grant for it has ever
existed for anyone but ``System``.

**The structural half of this fix lives in the Rust engine**, not here:
``epistemic-graph``'s ``Method::CreateGraph`` handler
(``src/server/dispatch.rs``) now calls
``IsolationLayer::provision_tenant_graph_access`` as part of the SAME durable
graph-creation call, which idempotently provisions a ``tenant:<slug>`` role +
Read/Write ``Allow`` grants on ``Pattern("tenant__<slug>__*")`` (covering every
graph that tenant owns, not just the one just created), and enrolls the
CREATOR in that role if the creator is already a registered durable identity.
That closes "creating a tenant graph never provisioned access to it" for every
FUTURE tenant graph, going forward, for its creator.

**What that engine-side fix does NOT do:** enroll any OTHER principal that
shares the same tenant but did not happen to create the graph — e.g. N
different agent-webui end-users, all mapped to the SAME tenant claim
(``homelab``), each with their own distinct engine ``agent_id``. That is an
identity-provisioning step, not a graph-creation step, and belongs on THIS
side (the same side ``engine_rbac_admission.provision_tier2_admission``
already handles for Tier-2 admin identities) — this module is that missing
piece for ordinary tenant content access.

**Design deliberately mirrors ``provision_tier2_admission`` exactly, including
its one binding constraint:** the engine exposes no "read one identity back"
RPC (only ``RbacAdmin{op:"List"}``, which returns the ROLE/GRANT policy, never
the identity registry — confirmed by reading ``crates/eg-types/src/protocol.rs``:
there is no ``GetIdentity``/``ListIdentities`` wire method). So, exactly like
``ServiceAdmissionEntry``, the CALLER supplies each principal's full current
identity shape (``role``, ``teams``, ``existing_roles`` — normally read from the
caller's OWN provisioning manifest, e.g. Tier-1 Keycloak's resolved service/user
record) rather than this module reading it back first; ``RegisterIdentity`` is
an upsert (replaces the whole identity), so re-registering the SAME shape plus
the tenant role is what makes this idempotent, not a get-then-merge round trip.

**Incident: a successful admission destroyed the capability admission itself
depends on.** ``TenantPrincipal.existing_roles`` was defined from the start,
and :func:`provision_tenant_access` always merged it correctly — the defect
was that ``existing_roles`` also had an *empty-tuple* default, which is
indistinguishable, at the call site, from "confirmed this principal holds
nothing else." A real caller (``agent-webui``'s ``ensure_tenant_admission``)
called this module for a principal that ALSO held ``control:system`` (granted
separately by :mod:`agent_utilities.security.system_rbac_admission`) without
populating ``existing_roles`` — so it silently registered ``roles=
['tenant:homelab']`` only, dropping ``control:system`` (which itself carries
``security:admin``) from underneath a principal that needed it to keep
functioning. The next admission pass then failed
``ACCESS_DENIED: ... lacks admin capability required for 'security:admin'``
end to end. Since this module has no read-back RPC (above) and therefore no
way to independently confirm which case it is, the empty-tuple default is
now gone: ``existing_roles`` defaults to ``None`` — an explicit "unknown,"
distinct from a caller-confirmed empty tuple — and :func:`provision_tenant_access`
raises :class:`TenantAdmissionError` immediately for any principal whose
``existing_roles`` is ``None``, rather than writing a role set that might be
silently short one. This can never widen privilege (it never invents a role),
only ever refuses to write when the truth is unknown — see AGENTS.md
"Fail closed."

**Self-admission is a no-op — the fix that makes the above safe to actually
ship.** Failing loud on unknown ``existing_roles`` fixes the destructive
write, but ``agent-webui``'s live ``ensure_tenant_admission`` calls this
module exactly the way the incident above describes — ``TenantPrincipal
(agent_id=agent_id)``, no ``existing_roles`` — for every authenticated
request, admitting the CALLER's own principal into its own tenant. Left as
just "fail loud," that turns a self-destructing success into an immediate,
permanent 503 on every request: strictly worse. Two ways a caller could avoid
that were considered:

1. *Have the caller pass an explicit* ``existing_roles=()``. Sound for a
   genuinely fresh identity (a first-time browser user has nothing to lose),
   but it cannot be the general answer for THIS caller: the very fact that
   ``resolve_admission_authority()`` produced a signer for this ``agent_id``
   already proves the engine has previously validated a signature for it —
   asserting ``()`` would be asserting a specific, unverified fact
   (``existing_roles`` is a claim about the ENGINE's RBAC identity store,
   which this module still cannot read), not a safe default. It also requires
   a change in ``agent-webui`` — a different repository, out of scope here.
2. **Recognize self-admission structurally and skip it — the fix taken.**
   When ``principal.agent_id == admin_authority.signer_id``, the principal is
   admitting itself, signing as itself. The engine's own rule
   (``signer == context.principal()``, see :mod:`admission_authority`) means
   this call could only be SIGNED at all by a principal that already exists.
   There is therefore nothing this call could legitimately add: either the
   principal already carries the tenant role (no-op) or it doesn't yet, but
   granting it here — with no verified knowledge of what else it holds — risks
   exactly the destructive overwrite this whole fix exists to stop. The only
   sound action is neither "write the merge" nor "fail loud" but **skip**:
   report success, make no RPC call, change nothing. This is deliberately
   narrower than "trust any caller admitting itself" — it fires only on an
   EXACT match against ``admin_authority.signer_id`` (never a heuristic on
   role name or agent-id shape), so an actually-unprovisioned principal being
   admitted BY a different, already-provisioned caller (the ``provisioner:
   deploy`` / N-distinct-end-users shape the rest of this module supports)
   still gets the normal merge-or-fail-loud treatment above; it is
   distinguished, not bypassed. This does not contradict "never widen
   privilege": a skip grants nothing.

**Never mints, prints, logs, or persists a signer key or a secret value** —
mirrors ``engine_rbac_admission.py``'s own doctrine exactly (this repo's
"Secrets & credential retrieval" standard, AGENTS.md).

**Prod operations are PREPARE-ONLY here too, same as
``graph_ownership_apply.py``.** Nothing in this repository's tests ever calls
:func:`provision_tenant_access` with a :class:`LiveEngineIdentityClient`
against a real engine; that is an explicit, operator-gated deployment step
(HG-4 — a security-boundary mutation), never something a test or an
autonomous pass performs against a live cluster.
"""

import logging
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
    """One principal to admit into a tenant's RBAC role — the AU-side mirror of
    one Tier-1 (Keycloak) resolved identity record. ``agent_id`` MUST equal the
    value that will appear as the principal's ``VerifiedRequestContext.agent_id``
    on its future requests (its OIDC ``sub`` / service-account subject), exactly
    like ``ServiceAdmissionEntry.agent_id``'s binding requirement. ``role``/
    ``teams``/``existing_roles`` are this principal's CURRENT full identity
    shape as the caller's own provisioning source of truth knows it — never
    guessed by this module (see the module docstring: the engine exposes no
    identity read-back RPC).

    ``existing_roles`` carries one of two meanings, and they are NOT
    interchangeable: an explicit ``()`` means the caller has CONFIRMED this
    principal currently holds no other roles; the default, ``None``, means
    the caller does not know. :func:`provision_tenant_access` merges the
    former and refuses the latter (see the module docstring, "Incident") —
    never treat "I didn't check" the same as "I checked and it's empty"."""

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

    For each principal: a self-admission (``principal.agent_id ==
    admin_authority.signer_id`` — the principal admitting itself into its own
    tenant, signing as itself) is always a no-op, skipped before
    ``existing_roles`` is even inspected — see the module docstring,
    "Self-admission is a no-op". This is exactly ``agent-webui``'s
    ``ensure_tenant_admission`` shape, and it is safe precisely because the
    caller there could NOT establish ``existing_roles`` for itself. For every
    OTHER principal: raises immediately if ``principal.existing_roles`` is
    ``None`` (unknown — see the module docstring, "Incident"; never guesses
    an empty set). Otherwise: a no-op (``already_held=True``) when
    ``tenant_role in principal.existing_roles``; else re-registers the
    identity with its EXACT existing ``role``/``teams`` plus the tenant role
    appended to ``existing_roles`` — never dropping a role/team the caller
    didn't ask to change (``RegisterIdentity`` replaces the whole identity, so
    this module always sends the FULL desired shape, sourced from the caller's
    ``TenantPrincipal`` — see the module docstring for why no read-back is
    attempted).

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
        if principal.existing_roles is None:
            raise TenantAdmissionError(
                f"cannot admit {principal.agent_id!r} into {role!r}: "
                "existing_roles is unknown (None). RegisterIdentity REPLACES "
                "a principal's whole role set and this module has no "
                "identity read-back RPC, so writing an unknown role set risks "
                "silently dropping roles the principal already holds — see "
                "the module docstring, 'Incident'. The caller MUST supply "
                "TenantPrincipal.existing_roles as the principal's full, "
                "currently-known role set, or an explicit empty tuple () to "
                "affirmatively confirm it holds none."
            )
        if role in principal.existing_roles:
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
