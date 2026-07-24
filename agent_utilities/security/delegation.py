"""CONCEPT:AU-OS.identity.per-agent-on-behalf-delegation — Per-agent on-behalf-of identity.

Connects the three previously-disconnected delegation primitives into ONE pipeline for a
spawned agent instance, so a delegated/spawned agent stops running as one fixed service
account and instead carries its real caller chain end to end:

1. **Exchange** — RFC 8693 token exchange of the caller's token
   (:func:`agent_utilities.mcp.delegated_auth.exchange_token_for_agent`) → a delegated token
   whose logical actor chain appends ``agent:<name>``.
2. **Envelope chain** — the engine envelope's *validated* ``delegation`` chain
   (``[principal, …, agent:<name>:<run_id>]``, principal-first / agent-last / len≥2), emitted
   by :meth:`GraphSession.engine_verified_context` when a delegation is active.
3. **Run scope** — an HMAC run-scoped token (:mod:`agent_utilities.security.run_token`) minted
   per spawn instance, its endpoint/operation allowlist derived from the task, TTL ≤ run budget.

plus the **ceiling intersection** (a spawn can never exceed its ultimate principal's
:func:`~agent_utilities.security.identity.base_capabilities`), **provenance** (the chain on the
``:RunTrace``), and **bounded-time revocation** (lease-renewal revalidation of the delegated
credential's expiry — revoking the caller kills the spawn at its next lease renewal).

Rollout is three-state: ``ENABLE_DELEGATED_IDENTITY = off | warn | on`` (shipped default
``warn``):

* ``off``   — legacy identity everywhere; no delegation object is built.
* ``warn``  — the full pipeline is COMPUTED and every decision that WOULD change is logged
  (denied tools, the chain, the run-token, a would-be-failed lease renewal), but the wire
  envelope stays legacy and nothing is enforced. This is the observe-before-enforce soak posture.
* ``on``    — enforced: the envelope carries the real chain, the ceiling narrows the spawn's
  tools, and an expired delegated credential fails the next lease renewal.

Every identity decision is a first-class log line (a denied-would-be in ``warn`` mode included).
"""

from __future__ import annotations

import contextvars
import logging
import re
import time
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import StrEnum

from agent_utilities.core._env import setting

logger = logging.getLogger(__name__)


class DelegationMode(StrEnum):
    """Rollout posture for per-agent delegated identity (``ENABLE_DELEGATED_IDENTITY``)."""

    OFF = "off"
    WARN = "warn"
    ON = "on"


def delegation_mode() -> DelegationMode:
    """Resolve the live rollout posture from ``ENABLE_DELEGATED_IDENTITY`` (default ``warn``).

    Read live (via :func:`config.setting`) so a deployment can flip ``warn`` → ``on`` after a
    clean soak without a code change. An unrecognized value degrades to ``warn`` and is logged
    once-shaped (never silently treated as ``on``/``off``).
    """
    raw = str(setting("ENABLE_DELEGATED_IDENTITY", "warn") or "warn").strip().lower()
    try:
        return DelegationMode(raw)
    except ValueError:
        logger.warning(
            "ENABLE_DELEGATED_IDENTITY=%r is not one of off|warn|on; defaulting to warn",
            raw,
        )
        return DelegationMode.WARN


def delegation_enabled() -> bool:
    """True when the delegated-identity pipeline runs at all (``warn`` or ``on``)."""
    return delegation_mode() is not DelegationMode.OFF


# ---------------------------------------------------------------------------
# Agent-instance identity
# ---------------------------------------------------------------------------

_LABEL_SANITIZE = re.compile(r"[^A-Za-z0-9._-]+")

# Capabilities that mean "unrestricted" — a principal holding any of these is not further
# narrowed by the tool-scope ceiling (an admin/service principal keeps its full toolset).
_UNRESTRICTED_CAPABILITIES = frozenset(
    {"*", "kg:*", "kg:admin", "admin", "superuser", "root"}
)


def agent_instance_label(agent_name: str, run_id: str) -> str:
    """Return the per-run agent-instance identity ``agent:<name>:<run_id>``.

    This is the *instance* identity (decision 2: ``agent_id`` is per-run, not just the agent
    name), so two concurrent spawns of the same agent are distinct principals in the chain and
    in provenance. Both components are sanitized to the envelope-safe ``[A-Za-z0-9._-]`` charset.
    """
    name = (
        _LABEL_SANITIZE.sub("-", str(agent_name or "agent").strip()).strip("-")
        or "agent"
    )
    rid = _LABEL_SANITIZE.sub("-", str(run_id or "").strip()).strip("-")
    return f"agent:{name}:{rid}" if rid else f"agent:{name}"


# ---------------------------------------------------------------------------
# The delegation object + its ambient propagation
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SpawnDelegation:
    """The resolved on-behalf-of identity for one spawned agent instance.

    ``chain`` is the authoritative delegation array the engine envelope validates: it runs
    ``[principal, …, agent_instance_id]`` (principal-first, agent-last, length ≥ 2). ``ceiling``
    is the ultimate principal's :func:`base_capabilities` — the maximum the spawn may hold.
    ``oidc_token`` is the RFC 8693 exchanged token that rides the envelope's ``oidc_token``
    field once the wire supports it (decision 5). ``expires_at`` is the delegated credential's
    absolute UNIX expiry — the bound for revocation at the next lease renewal.
    """

    mode: DelegationMode
    principal: str
    chain: tuple[str, ...]
    agent_instance_id: str
    ceiling: tuple[str, ...] = ()
    run_token: str = ""
    oidc_token: str | None = None
    expires_at: float | None = None

    @property
    def enforced(self) -> bool:
        """True only in ``on`` mode — the posture that changes wire/enforcement behavior."""
        return self.mode is DelegationMode.ON


_current: contextvars.ContextVar[SpawnDelegation | None] = contextvars.ContextVar(
    "agent_utilities_spawn_delegation", default=None
)


def current_delegation() -> SpawnDelegation | None:
    """Return the delegation active for the current spawn, or ``None`` (legacy path)."""
    return _current.get()


def enter_delegation(
    delegation: SpawnDelegation | None,
) -> contextvars.Token[SpawnDelegation | None] | None:
    """Bind ``delegation`` as ambient; return a token for :func:`reset_delegation` (or ``None``)."""
    return _current.set(delegation) if delegation is not None else None


def reset_delegation(token: contextvars.Token[SpawnDelegation | None] | None) -> None:
    """Restore the prior delegation (paired with :func:`enter_delegation`)."""
    if token is not None:
        _current.reset(token)


@contextmanager
def use_delegation(
    delegation: SpawnDelegation | None,
) -> Iterator[SpawnDelegation | None]:
    """Scope a block to ``delegation`` (restores the previous value on exit)."""
    token = enter_delegation(delegation)
    try:
        yield delegation
    finally:
        reset_delegation(token)


def _dedup(items: Iterable[str]) -> tuple[str, ...]:
    """Order-preserving de-duplication of non-empty strings."""
    return tuple(dict.fromkeys(s for i in items if (s := str(i).strip())))


def build_spawn_delegation(
    *,
    agent_name: str,
    run_id: str,
    principal: str,
    ceiling: Iterable[str] = (),
    run_token: str = "",
    parent_chain: Iterable[str] = (),
    oidc_token: str | None = None,
    expires_at: float | None = None,
    mode: DelegationMode | None = None,
) -> SpawnDelegation:
    """Construct the delegation for a spawn, appending ``agent:<name>:<run_id>`` to the chain.

    ``parent_chain`` carries a nesting spawn's own chain so a spawn-of-a-spawn extends the same
    principal→…→agent array rather than starting fresh. The result always satisfies the engine's
    wire contract: principal-first, agent-last, length ≥ 2, entries unique and non-empty.
    """
    resolved_mode = mode or delegation_mode()
    principal = str(principal or "").strip()
    if not principal:
        raise ValueError("delegation requires a non-empty ultimate principal")

    label = agent_instance_label(agent_name, run_id)

    base = list(_dedup(parent_chain)) or [principal]
    if base[0] != principal:
        # The parent chain must start at the ultimate principal; re-anchor defensively.
        base = [principal] + [c for c in base if c != principal]
    if label in base:
        # Vanishingly unlikely (run-id collision); keep the agent-instance id unique so the
        # chain never carries a duplicate (the engine rejects duplicate delegation entries).
        label = f"{label}:{len(base)}"

    chain = tuple(base) + (label,)
    if len(chain) < 2 or chain[0] != principal or chain[-1] != label:
        raise ValueError(
            "delegation chain must run principal-first / agent-last (len>=2); "
            f"got {chain!r}"
        )

    return SpawnDelegation(
        mode=resolved_mode,
        principal=principal,
        chain=chain,
        agent_instance_id=label,
        ceiling=_dedup(ceiling),
        run_token=str(run_token or ""),
        oidc_token=oidc_token,
        expires_at=expires_at,
    )


# ---------------------------------------------------------------------------
# Ceiling intersection (decision 4)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CeilingDecision:
    """The outcome of intersecting a spawn's requested tools with its principal ceiling."""

    effective: tuple[str, ...]
    denied: tuple[str, ...] = ()
    enforced: bool = False


def _ctx_suffix(context: str) -> str:
    return f" [{context}]" if context else ""


def enforce_ceiling(
    requested: Iterable[str],
    ceiling: Iterable[str],
    *,
    mode: DelegationMode,
    context: str = "",
) -> CeilingDecision:
    """Intersect a spawn's ``requested`` tools with its principal's capability ``ceiling``.

    A spawn can never exceed its ultimate principal (decision 4). A principal holding an
    unrestricted capability (``*``/``kg:admin``/…), or with no resolvable ceiling, is not
    narrowed. Otherwise the effective allow-list is ``requested ∩ ceiling``:

    * ``on``   — the intersection is returned (tools exceeding the ceiling are DENIED, logged).
    * ``warn`` — ``requested`` is returned unchanged (legacy identity still governs), but every
      tool that WOULD be denied is logged (the observe-before-enforce signal).
    * ``off``  — no narrowing.

    Every decision is logged: a denied (``on``) or would-be-denied (``warn``) tool is a
    first-class identity log line, never silently dropped.
    """
    req = _dedup(requested)
    ceil = frozenset(_dedup(ceiling))
    if mode is DelegationMode.OFF or not ceil or (ceil & _UNRESTRICTED_CAPABILITIES):
        return CeilingDecision(effective=req, denied=(), enforced=False)

    denied = tuple(t for t in req if t not in ceil)
    if not denied:
        return CeilingDecision(effective=req, denied=(), enforced=False)

    if mode is DelegationMode.ON:
        effective = tuple(t for t in req if t in ceil)
        logger.warning(
            "[delegation] ceiling ENFORCED%s: denied %d tool(s) exceeding principal "
            "capabilities: denied=%s ceiling=%s",
            _ctx_suffix(context),
            len(denied),
            list(denied),
            sorted(ceil)[:12],
        )
        return CeilingDecision(effective=effective, denied=denied, enforced=True)

    logger.warning(
        "[delegation] warn: ceiling WOULD DENY %d tool(s)%s (legacy identity still in "
        "effect): would_deny=%s ceiling=%s",
        len(denied),
        _ctx_suffix(context),
        list(denied),
        sorted(ceil)[:12],
    )
    return CeilingDecision(effective=req, denied=denied, enforced=False)


# ---------------------------------------------------------------------------
# Run-scoped token minting (decision 3)
# ---------------------------------------------------------------------------

# Default run-token lifetime when the caller supplies no tighter budget. A spawn is expected
# to finish well inside this; it is the outer bound, not the target.
DEFAULT_RUN_TOKEN_TTL_S = 3600.0


def mint_spawn_run_token(
    run_id: str,
    *,
    principal: str = "",
    tenant: str = "",
    allowed_tools: Iterable[str] | None = None,
    ttl_seconds: float = DEFAULT_RUN_TOKEN_TTL_S,
    operations: Iterable[str] = ("read", "write"),
    now: float | None = None,
) -> tuple[str, float]:
    """Mint the per-spawn run-scoped token and return ``(token, expires_at)``.

    The endpoint allowlist is derived from the task's resolved tool scope (``allowed_tools``);
    when the spawn is unrestricted the token is endpoint-wildcarded (``*``) but still time- and
    operation-bounded. The TTL is capped at :data:`DEFAULT_RUN_TOKEN_TTL_S` so a caller cannot
    mint an over-long run credential. Fails closed (via :mod:`run_token`) when delegation is on
    and no signing secret is configured.
    """
    from agent_utilities.security.run_token import mint_token

    endpoints = tuple(_dedup(allowed_tools)) if allowed_tools else ("*",)
    ttl = max(1.0, min(float(ttl_seconds), DEFAULT_RUN_TOKEN_TTL_S))
    issued = time.time() if now is None else now
    token = mint_token(
        run_id,
        endpoints=endpoints,
        operations=tuple(_dedup(operations)) or ("read",),
        ttl_seconds=ttl,
        actor_id=str(principal or ""),
        tenant_id=str(tenant or ""),
        now=issued,
    )
    return token, issued + ttl


# ---------------------------------------------------------------------------
# Revocation / bounded-time expiry (decision 6)
# ---------------------------------------------------------------------------


def is_delegation_live(
    delegation: SpawnDelegation | None = None, *, now: float | None = None
) -> bool:
    """False when a delegated credential/run-token has expired — the revocation bound.

    Called at WorkItem lease renewal: revoking the caller (or the credential simply expiring)
    makes the next renewal fail, so the spawn's lease lapses and it is reaped — bounded-time
    revocation without a separate kill channel. A path with no delegation is always live.
    """
    delegation = delegation if delegation is not None else current_delegation()
    if delegation is None:
        return True
    moment = time.time() if now is None else now
    if delegation.expires_at is not None and moment >= delegation.expires_at:
        return False
    if delegation.run_token:
        try:
            from agent_utilities.security.run_token import decode_token

            if moment >= decode_token(delegation.run_token).expires_at:
                return False
        except Exception as exc:  # noqa: BLE001 — a bad/expired token is not-live, fail closed
            logger.warning(
                "[delegation] run-token revalidation failed (treating as expired): %s",
                exc,
            )
            return False
    return True


# ---------------------------------------------------------------------------
# Ceiling derivation from the ambient caller identity
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PrincipalIdentity:
    """The ultimate caller a spawn acts on behalf of, plus its capability ceiling."""

    principal: str = ""
    tenant: str = ""
    ceiling: tuple[str, ...] = field(default_factory=tuple)


def resolve_principal_identity() -> PrincipalIdentity:
    """Resolve the ambient caller principal + :func:`base_capabilities` ceiling (best-effort).

    Prefers a validated OIDC token's claims (normalized across Okta/Keycloak/… then reduced to
    the base capability set); falls back to the bound :class:`ActorContext` whose ``roles`` are
    already the effective capability set. Returns an empty identity when nothing is bound (a
    tiny/local process) — the caller then treats the ceiling as unresolved (no narrowing).
    """
    # 1) Validated delegated OIDC claims (the richest ceiling source).
    try:
        from agent_utilities.mcp.delegated_auth import get_user_claims

        claims = get_user_claims()
    except Exception:  # noqa: BLE001 — delegation middleware may be absent
        claims = None
    if claims:
        try:
            from agent_utilities.security.identity import (
                base_capabilities,
                normalize_identity,
            )

            ident = normalize_identity(claims)
            return PrincipalIdentity(
                principal=ident.subject,
                tenant=ident.tenant,
                ceiling=tuple(base_capabilities(ident)),
            )
        except Exception as exc:  # noqa: BLE001 — malformed claims fall through to the actor
            logger.debug("delegation: claim-derived ceiling skipped: %s", exc)

    # 2) The bound ActorContext (roles are already the effective capability set).
    try:
        from agent_utilities.security.brain_context import current_actor

        actor = current_actor()
        if actor.actor_id:
            return PrincipalIdentity(
                principal=str(actor.actor_id),
                tenant=str(actor.tenant_id or ""),
                ceiling=tuple(_dedup(actor.roles)),
            )
    except Exception as exc:  # noqa: BLE001 — no bound actor (tiny/local): unresolved ceiling
        logger.debug("delegation: actor-derived ceiling unavailable: %s", exc)
    return PrincipalIdentity()


__all__ = [
    "CeilingDecision",
    "DelegationMode",
    "PrincipalIdentity",
    "SpawnDelegation",
    "agent_instance_label",
    "build_spawn_delegation",
    "current_delegation",
    "delegation_enabled",
    "delegation_mode",
    "enforce_ceiling",
    "enter_delegation",
    "is_delegation_live",
    "mint_spawn_run_token",
    "reset_delegation",
    "resolve_principal_identity",
    "use_delegation",
]
