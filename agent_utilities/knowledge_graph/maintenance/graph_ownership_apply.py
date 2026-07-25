#!/usr/bin/python
from __future__ import annotations

"""Grant-APPLICATION half of the W2.8 ownership pass (HG-4) — the only module in
this pair allowed to mutate RBAC state, and only when explicitly told to.

CONCEPT:AU-KG.audit.graph-ownership-disposition — see
:mod:`agent_utilities.knowledge_graph.maintenance.graph_ownership` for the
disposition report this module consumes; that module is strictly read-only.

Program decision this module implements: **grants are auto-apply-UNAMBIGUOUS /
hold-ambiguous** (:func:`plan_grants` defaults to UNAMBIGUOUS-only), and **prod
operations are PREPARE-ONLY** — this module is exercised end-to-end against fixtures
only; nothing in this repository's tests, scripts, or this task ever calls
:func:`apply_plan` with ``dry_run=False`` against a live engine. The dry-run/apply
split plus a required, non-default ``--apply-unambiguous`` CLI flag
(``scripts/apply_graph_ownership_grants.py``) are the two independent gates a human
operator must clear before anything here reaches a real store; HG-4 is the human
gate that decides to run it live at all.

Every applied grant's exact inverse (a ``remove_grant`` call) is returned by
:func:`rollback_for` / included in :func:`apply_plan`'s result — undo is a first-class
output, not an afterthought. Role *removal* is deliberately NOT part of the automated
rollback: a role (e.g. ``owner:system-code-ingestion``) is intentionally coarse and
reused across many graphs' grants in one apply run, so removing it on rollback could
strip access a DIFFERENT still-applied grant depends on; a human confirms that
before ever removing a role.
"""

import logging
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from .graph_ownership import OwnershipReport

logger = logging.getLogger(__name__)

__all__ = [
    "AppliedGrant",
    "FixtureRbacAdminClient",
    "GrantApplicationError",
    "GrantPlanEntry",
    "LiveRbacAdminClient",
    "RbacAdminClient",
    "RollbackEntry",
    "apply_plan",
    "plan_grants",
    "resolve_rbac_admin_client",
    "rollback_for",
    "rollback_to_json",
]


@dataclass(frozen=True)
class GrantPlanEntry:
    """One RBAC grant :func:`apply_plan` would issue — mirrors
    ``RbacClient.add_grant(role, resource, action, effect)`` exactly."""

    graph: str
    owner: str | None
    role: str
    resource: dict[str, str]
    action: str
    effect: str


def plan_grants(
    report: OwnershipReport, *, include_ambiguous: bool = False
) -> list[GrantPlanEntry]:
    """The concrete grant plan for ``report`` — what ``--apply-unambiguous`` would
    apply.

    Default (``include_ambiguous=False``, what the CLI always uses): only
    UNAMBIGUOUS, not-already-covered rows — the program's auto-apply-UNAMBIGUOUS /
    hold-ambiguous decision. ``include_ambiguous=True`` exists only for a human who
    has manually reviewed one specific AMBIGUOUS row and wants to force it into a
    plan they build and apply themselves; the apply CLI never exposes it as a flag
    (a blanket "also apply ambiguous ones" switch would defeat the entire point of
    the disposition split), so reaching it requires calling this function directly.
    """
    plan: list[GrantPlanEntry] = []
    for d in report.dispositions:
        if d.status != "UNAMBIGUOUS" and not include_ambiguous:
            continue
        if d.already_covered:
            continue
        for rg in d.recommended_grants:
            plan.append(
                GrantPlanEntry(
                    graph=d.graph,
                    owner=d.owner,
                    role=rg.role,
                    resource=dict(rg.resource),
                    action=rg.action,
                    effect=rg.effect,
                )
            )
    return plan


@dataclass(frozen=True)
class RollbackEntry:
    """The exact inverse of one applied grant: ``rbac.remove_grant(role, resource,
    action, effect)`` — a plan entry and its rollback share the same 4-tuple by
    construction, so "the inverse" is never a guess."""

    role: str
    resource: dict[str, str]
    action: str
    effect: str


def rollback_for(plan: list[GrantPlanEntry]) -> list[RollbackEntry]:
    """The rollback list for a PLAN (dry-run preview — what undo would look like if
    every entry were applied). :func:`apply_plan` returns the rollback for what was
    ACTUALLY applied, which may be a strict prefix of this on a failure partway
    through."""
    return [
        RollbackEntry(
            role=e.role, resource=e.resource, action=e.action, effect=e.effect
        )
        for e in plan
    ]


def rollback_to_json(rollback: list[RollbackEntry]) -> list[dict[str, Any]]:
    """Serializable rollback op list (``scripts/apply_graph_ownership_grants.py
    --rollback-out`` writes exactly this)."""
    return [
        {
            "op": "remove_grant",
            "role": r.role,
            "resource": r.resource,
            "action": r.action,
            "effect": r.effect,
        }
        for r in rollback
    ]


class GrantApplicationError(RuntimeError):
    """A mutating RBAC call failed against a live client. Never raised by the
    dry-run path (which never calls the client).

    Carries whatever :attr:`applied`/:attr:`rollback` :func:`apply_plan` had
    already built BEFORE the failing call — a partial apply must never lose the
    undo information for the grants that DID succeed (that would defeat the whole
    point of computing a rollback plan). A caller catching this can always recover
    the true rollback for what actually landed via ``exc.rollback``, never just the
    full originally-requested plan.
    """

    def __init__(
        self,
        message: str,
        *,
        applied: list[AppliedGrant] | None = None,
        rollback: list[RollbackEntry] | None = None,
    ) -> None:
        super().__init__(message)
        self.applied: list[AppliedGrant] = list(applied or [])
        self.rollback: list[RollbackEntry] = list(rollback or [])


@runtime_checkable
class RbacAdminClient(Protocol):
    """The minimal mutating surface :func:`apply_plan` needs — small and
    Protocol-typed so :class:`FixtureRbacAdminClient` (every test, and the sample
    dry-run demo) and :class:`LiveRbacAdminClient` (the real engine) are
    interchangeable."""

    def existing_roles(self) -> set[str]:
        ...

    def add_role(self, role: str) -> str:
        ...

    def add_grant(
        self, role: str, resource: dict[str, str] | str, action: str, effect: str
    ) -> str:
        ...

    def remove_grant(
        self, role: str, resource: dict[str, str] | str, action: str, effect: str
    ) -> dict[str, Any]:
        ...


class FixtureRbacAdminClient:
    """In-memory :class:`RbacAdminClient` — records every call it receives
    (``self.calls``) so a test can assert exactly what an apply run would have done,
    without ever touching a socket."""

    def __init__(
        self,
        *,
        roles: set[str] | None = None,
        grants: set[tuple[str, str, str, str]] | None = None,
    ) -> None:
        self.roles: set[str] = set(roles or ())
        # grants keyed as (role, resource_repr, action, effect) — resource is stored
        # as its repr() since dicts aren't hashable; good enough for an in-memory fixture.
        self.grants: set[tuple[str, str, str, str]] = set(grants or ())
        self.calls: list[tuple[str, tuple[Any, ...]]] = []

    def existing_roles(self) -> set[str]:
        return set(self.roles)

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

    def remove_grant(
        self, role: str, resource: dict[str, str] | str, action: str, effect: str
    ) -> dict[str, Any]:
        self.calls.append(("remove_grant", (role, resource, action, effect)))
        key = (role, repr(resource), action, effect)
        removed = key in self.grants
        self.grants.discard(key)
        return {"removed": removed}


class LiveRbacAdminClient:
    """Mutating adapter over the real engine's ``RbacClient``, via the SAME
    process-authority path as
    :class:`~agent_utilities.knowledge_graph.maintenance.graph_ownership.LiveEngineCatalogClient`.
    Constructing this class does nothing by itself; every method call is a real
    mutating (``add_role``/``add_grant``) or read (``existing_roles``) RPC and can
    raise :class:`GrantApplicationError`. Only ever instantiated by
    ``scripts/apply_graph_ownership_grants.py`` when the operator passes
    ``--apply-unambiguous`` explicitly — never implicitly, never as a default.
    """

    def __init__(self, *, config: Any = None) -> None:
        self._config = config

    def _client(self) -> Any:
        from ..core.graph_compute import GraphComputeEngine

        return GraphComputeEngine.get_or_create().client

    def existing_roles(self) -> set[str]:
        try:
            policy = self._client().rbac.list()
        except Exception as exc:
            raise GrantApplicationError("engine rbac.list() failed") from exc
        roles = policy.get("roles") if isinstance(policy, dict) else None
        names: set[str] = set()
        for entry in roles or []:
            if not isinstance(entry, dict):
                continue
            name = entry.get("name")
            if isinstance(name, str):
                names.add(name)
        return names

    def add_role(self, role: str) -> str:
        try:
            return str(self._client().rbac.add_role(role))
        except Exception as exc:
            raise GrantApplicationError(f"engine add_role({role!r}) failed") from exc

    def add_grant(
        self, role: str, resource: dict[str, str] | str, action: str, effect: str
    ) -> str:
        try:
            return str(self._client().rbac.add_grant(role, resource, action, effect))
        except Exception as exc:
            raise GrantApplicationError(
                f"engine add_grant({role!r}, {resource!r}, {action!r}, {effect!r}) failed"
            ) from exc

    def remove_grant(
        self, role: str, resource: dict[str, str] | str, action: str, effect: str
    ) -> dict[str, Any]:
        try:
            result = self._client().rbac.remove_grant(role, resource, action, effect)
        except Exception as exc:
            raise GrantApplicationError(
                f"engine remove_grant({role!r}, {resource!r}, {action!r}, {effect!r}) failed"
            ) from exc
        return dict(result) if isinstance(result, dict) else {"removed": bool(result)}


def resolve_rbac_admin_client(config: Any = None) -> RbacAdminClient:
    """Return a :class:`LiveRbacAdminClient` bound to ``config``. Construction never
    connects by itself; the FIRST real call (e.g. :meth:`existing_roles`) is where a
    live-apply run would discover an unreachable engine and raise
    :class:`GrantApplicationError` — callers must not attempt this against an
    unreachable engine (there is no dry-run-style silent fallback for the apply
    path: an apply run either has a working live client or it does not run)."""
    return LiveRbacAdminClient(config=config)


@dataclass(frozen=True)
class AppliedGrant:
    """One plan entry's outcome."""

    entry: GrantPlanEntry
    role_created: bool
    grant_result: str


def apply_plan(
    plan: list[GrantPlanEntry],
    client: RbacAdminClient,
    *,
    dry_run: bool = True,
) -> tuple[list[AppliedGrant], list[RollbackEntry]]:
    """Apply (or, by default, simulate) ``plan`` against ``client``.

    ``dry_run=True`` (the default, and the CLI's default) never calls a single
    mutating method on ``client`` — it only builds the would-apply preview and its
    rollback list, matching :func:`rollback_for` on the same plan exactly.

    ``dry_run=False`` actually mutates: for each entry, ``add_role`` only if the
    role is not already in ``existing_roles()`` (idempotent — never re-adds a role
    already present, mirroring the prior emergency grant script's convention), then
    ``add_grant``. Applies in plan order and **stops at the first failure**: the
    failing call's exception is wrapped in :class:`GrantApplicationError` — never
    swallowed, cause preserved via ``raise ... from`` — and re-raised carrying the
    exact ``applied``/``rollback`` built so far on its ``.applied``/``.rollback``
    attributes, so a caller catching it can still recover and undo precisely the
    partial mutation (never the full originally-requested plan, and never nothing
    at all — a bare exception with no state would silently strand an un-rollbackable
    partial apply, which this explicitly avoids).
    """
    if dry_run:
        preview = [
            AppliedGrant(
                entry=e, role_created=False, grant_result="DRY-RUN (not applied)"
            )
            for e in plan
        ]
        return preview, rollback_for(plan)

    applied: list[AppliedGrant] = []
    rollback: list[RollbackEntry] = []
    seen_roles = client.existing_roles()
    for entry in plan:
        role_created = False
        try:
            if entry.role not in seen_roles:
                client.add_role(entry.role)
                seen_roles.add(entry.role)
                role_created = True
            result = client.add_grant(
                entry.role, entry.resource, entry.action, entry.effect
            )
        except Exception as exc:
            logger.error(
                "grant apply failed on graph %s after %d prior grant(s) succeeded "
                "(%s: %s)",
                entry.graph,
                len(applied),
                type(exc).__name__,
                exc,
            )
            raise GrantApplicationError(
                f"apply failed on graph {entry.graph!r} after {len(applied)} prior "
                f"grant(s) succeeded: {exc}",
                applied=applied,
                rollback=rollback,
            ) from exc
        applied.append(
            AppliedGrant(entry=entry, role_created=role_created, grant_result=result)
        )
        rollback.append(
            RollbackEntry(
                role=entry.role,
                resource=entry.resource,
                action=entry.action,
                effect=entry.effect,
            )
        )
    return applied, rollback
