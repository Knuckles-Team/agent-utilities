"""An in-memory EG ``RbacElevation`` ledger for surface tests (EH-405).

It reproduces the EG rules a surface relies on -- the actor is the caller's
verified identity (never a body field), two-person approval, the approval
names the exact request digest, one-shot lifecycle -- so a surface test
proves what reaches the engine and what the engine answers.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class FakeElevationEngine:
    """A tenant EG client whose only method is ``RbacElevation``."""

    caller: str = "alice"
    now_ms: int = 1_000_000
    leases: dict[str, dict[str, Any]] = field(default_factory=dict)
    sent: list[dict[str, Any]] = field(default_factory=list)

    async def _send(
        self,
        method: str,
        params: dict[str, Any] | None,
        graph: str | None,
        *,
        idempotency_key: str | None = None,
    ) -> Any:
        assert method == "RbacElevation" and params is not None
        self.sent.append(params)
        assert "actor" not in params, "a surface never sends an actor"
        op = params["op"]
        return getattr(self, "_" + op["op"])(op.get("request"))

    def ops(self) -> list[str]:
        return [params["op"]["op"] for params in self.sent]

    def _request(self, body: dict[str, Any]) -> dict[str, Any]:
        digest = hashlib.sha256(
            json.dumps([body, self.caller], sort_keys=True).encode()
        ).hexdigest()
        lease = {
            "elevation_id": body["elevation_id"],
            "kind": "rbac.elevation",
            "grantee": self.caller,
            "requester_parties": [self.caller],
            "approver_parties": [],
            "scopes": body["scopes"],
            "span_ms": body["span_ms"],
            "justification": body["justification"],
            "status": "requested",
            "requested_at_ms": self.now_ms,
            "request_digest": digest,
            "revision": 1,
        }
        self.leases[body["elevation_id"]] = lease
        return lease

    def _list(self, _body: None) -> list[dict[str, Any]]:
        return list(self.leases.values())

    def _approve(self, body: dict[str, Any]) -> dict[str, Any]:
        lease = self.leases[body["elevation_id"]]
        if self.caller in lease["requester_parties"]:
            raise PermissionError("ELEVATION_SELF_APPROVAL")
        if (
            lease["status"] != "requested"
            or body["request_digest"] != lease["request_digest"]
        ):
            raise PermissionError("ELEVATION_CONFLICT")
        lease.update(
            status="active",
            approver_parties=[self.caller],
            approved_at_ms=self.now_ms,
            hard_expires_at_ms=self.now_ms + lease["span_ms"],
            revision=lease["revision"] + 1,
        )
        return lease

    def _revoke(self, body: dict[str, Any]) -> dict[str, Any]:
        lease = self.leases[body["elevation_id"]]
        lease.update(status="revoked", ended_at_ms=self.now_ms)
        lease["revision"] += 1
        return lease
