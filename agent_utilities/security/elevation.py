"""Just-in-time RBAC elevation: the one client every surface shares (AU-SEC-R006).

epistemic-graph owns the authority (AU-SEC-R006): an elevation is a time-boxed
``rbac.elevation`` lease the graph-access chokepoint consults, approved by a
second identity, hard-expiring, never extended and audited on every use. This
module is the typed client over EG's ``RbacElevation`` method that the served
surfaces (GraphOS MCP/REST, the browser console, A2A and chat agents) share, so
their refusal rules cannot drift apart.

What the surfaces must never weaken, and where that is enforced here:

* **Identity comes from the verified session, never from a body.** No model in
  this module has an actor, approver or requester field; EG stamps the actor
  from the verified request context before consensus.
* **Only an operator console approves.** Agent tools, chat and A2A can request,
  list and revoke; an approval from any other :class:`ElevationSurface` is
  refused before anything is sent.
* **The approver acts directly and holds the exact approval scope.** A
  delegated session (an agent acting on someone's behalf) is refused, and so is
  a session without ``rbac:approve-elevation`` itself (``*``/``kg:admin`` do not
  stand in). EG re-checks both.
* **No self-approval, exact request, no replay.** The approval names the
  ``request_digest`` of the view the approver decided on; a stale view, an
  unknown id or the approver's own request is refused here and again by EG's
  two-person and digest rules.
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Callable, Mapping
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

__all__ = [
    "APPROVAL_SCOPE",
    "ElevationApproval",
    "ElevationRefused",
    "ElevationRequest",
    "ElevationRevocation",
    "ElevationScope",
    "ElevationService",
    "ElevationSurface",
    "ElevationView",
    "MAX_SPAN_MS",
]

#: The exact scope an approver's own session must carry (EG re-derives it).
APPROVAL_SCOPE = "rbac:approve-elevation"
#: EG's hard cap on one elevation window (the ControlLease cap).
MAX_SPAN_MS = 24 * 60 * 60 * 1000
_RESERVED_GRAPHS = frozenset({"__admin__"})


class ElevationSurface(StrEnum):
    """Where an elevation operation came from."""

    OPERATOR_CONSOLE = "operator_console"
    AGENT_TOOL = "agent_tool"
    A2A = "a2a"


#: The only surface an approval is accepted from: a human operator's
#: authenticated browser session.
APPROVING_SURFACES = frozenset({ElevationSurface.OPERATOR_CONSOLE})


class ElevationRefused(PermissionError):
    """A surface-side refusal, with a stable machine code."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code


class _Body(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ElevationScope(_Body):
    """One (graph, action) pair; the graph is a literal graph name."""

    graph: str = Field(min_length=1, max_length=512)
    action: Literal["read", "write"]

    @field_validator("graph")
    @classmethod
    def _literal_graph(cls, graph: str) -> str:
        if "*" in graph or "?" in graph or graph in _RESERVED_GRAPHS:
            raise ValueError("an elevation names one literal, non-reserved graph")
        return graph


class ElevationRequest(_Body):
    """Ask for ``scopes`` for ``span_ms`` once a second identity approves."""

    scopes: list[ElevationScope] = Field(min_length=1, max_length=16)
    span_ms: int = Field(gt=0, le=MAX_SPAN_MS)
    justification: str = Field(min_length=1, max_length=2048)
    elevation_id: str | None = Field(
        default=None, min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_.:-]+$"
    )


class ElevationApproval(_Body):
    """Approve the request whose digest the approver saw."""

    elevation_id: str = Field(min_length=1, max_length=128)
    request_digest: str = Field(min_length=1, max_length=256)


class ElevationRevocation(_Body):
    """End a requested or active elevation now."""

    elevation_id: str = Field(min_length=1, max_length=128)


class ElevationView(BaseModel):
    """One elevation as EG reports it, plus the time left for a countdown."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    elevation_id: str
    grantee: str
    status: str
    scopes: list[ElevationScope]
    span_ms: int
    justification: str
    request_digest: str
    requested_at_ms: int
    revision: int
    approved_at_ms: int | None = None
    hard_expires_at_ms: int | None = None
    ended_at_ms: int | None = None
    remaining_ms: int = 0
    own: bool = False

    @classmethod
    def from_lease(
        cls, lease: Mapping[str, Any], now_ms: int, caller: str = ""
    ) -> ElevationView:
        """``own`` marks the caller's own request, which it can never approve."""
        expires = lease.get("hard_expires_at_ms")
        active = lease.get("status") == "active" and isinstance(expires, int)
        remaining = max(0, expires - now_ms) if active and expires else 0
        own = bool(caller) and lease.get("grantee") == caller
        return cls.model_validate({**lease, "remaining_ms": remaining, "own": own})


def _clock_ms() -> int:
    return time.time_ns() // 1_000_000


def _decoded(payload: Any) -> Any:
    if isinstance(payload, bytes | str):
        return json.loads(payload)
    return payload


def _approver_refusal(
    surface: ElevationSurface, claims: Mapping[str, Any]
) -> ElevationRefused | None:
    """Why this caller may not approve at all, before any lease is read."""
    if surface not in APPROVING_SURFACES:
        return ElevationRefused(
            "ELEVATION_APPROVAL_SURFACE",
            "elevations are approved only from an operator console session",
        )
    if claims.get("delegation"):
        return ElevationRefused(
            "ELEVATION_APPROVER_DELEGATED", "an approver must act directly"
        )
    if APPROVAL_SCOPE not in (claims.get("scopes") or ()):
        return ElevationRefused(
            "ELEVATION_APPROVER_SCOPE",
            f"the session does not hold the exact {APPROVAL_SCOPE} scope",
        )
    return None


def _lease_refusal(
    view: ElevationView | None, approval: ElevationApproval, agent_id: str
) -> ElevationRefused | None:
    """Why this approval does not match the lease the approver can see."""
    if view is None:
        return ElevationRefused("ELEVATION_NOT_FOUND", "no such elevation")
    if view.grantee == agent_id:
        return ElevationRefused(
            "ELEVATION_SELF_APPROVAL", "an elevation is approved by someone else"
        )
    if view.request_digest != approval.request_digest:
        return ElevationRefused(
            "ELEVATION_STALE_VIEW",
            "the request changed since it was shown; review it again",
        )
    return None


class ElevationService:
    """Typed request/approve/revoke/list over EG ``RbacElevation``.

    ``client`` is a tenant-routed EG client already bound to the caller's
    verified context (``use_verified_context``); the service never binds or
    widens identity itself. ``caller`` (that context's ``agent_id``) only
    labels the caller's own requests in views; it grants nothing.
    """

    def __init__(
        self,
        client: Any,
        *,
        caller: str = "",
        clock_ms: Callable[[], int] = _clock_ms,
    ) -> None:
        self._client = client
        self._caller = caller
        self._clock_ms = clock_ms

    async def _send(self, op: dict[str, Any]) -> Any:
        from epistemic_graph.generated.security import send_rbac_elevation

        result = await send_rbac_elevation(self._client, {"op": op})
        return _decoded(result.payload)

    def _view(self, lease: Mapping[str, Any]) -> ElevationView:
        return ElevationView.from_lease(lease, self._clock_ms(), self._caller)

    async def request(self, ask: ElevationRequest) -> ElevationView:
        """File a request; it grants nothing until someone else approves it."""
        body = ask.model_dump(mode="json")
        body["elevation_id"] = ask.elevation_id or f"elevation-{uuid.uuid4().hex}"
        return self._view(await self._send({"op": "request", "request": body}))

    async def list_elevations(self) -> list[ElevationView]:
        """The elevations the caller is a party to (an approver sees all)."""
        leases = await self._send({"op": "list"})
        return [self._view(lease) for lease in leases or ()]

    async def revoke(self, revocation: ElevationRevocation) -> ElevationView:
        """End one elevation now; revocation only ever narrows access."""
        body = revocation.model_dump(mode="json")
        return self._view(await self._send({"op": "revoke", "request": body}))

    async def approve(
        self,
        approval: ElevationApproval,
        *,
        surface: ElevationSurface,
        claims: Mapping[str, Any],
    ) -> ElevationView:
        """Approve one exact request as the verified, direct approver in ``claims``."""
        refused = _approver_refusal(surface, claims)
        if refused is not None:
            raise refused
        views = {view.elevation_id: view for view in await self.list_elevations()}
        refused = _lease_refusal(
            views.get(approval.elevation_id), approval, str(claims.get("agent_id"))
        )
        if refused is not None:
            raise refused
        body = approval.model_dump(mode="json")
        return self._view(await self._send({"op": "approve", "request": body}))
