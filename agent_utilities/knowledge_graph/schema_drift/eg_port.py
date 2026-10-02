"""The EG surface schema repair drives, bound to ONE request graph.

AU sends EG typed record contracts only, never RDF: EG renders and validates
the SHACL (AU-SEC-R005). Repair touches two EG authorities that must agree on
the graph: the approval lease (``IssueControlLease``/``GetControlLease``/
``TransitionControlLease``) and the schema sources (``GraphSchema``). EG reads
an ``AttachApproved`` request's lease from that request's own graph, so every
call here names the same graph explicitly instead of trusting a client
default.

AU never constructs a candidate ``GraphSchema`` or pack, validates instances
against it, or runs a shadow ingest itself: :meth:`validate_repair` and
:meth:`attach_approved` are both typed ``GraphSchema`` requests EG alone
renders, validates and (once approved) activates.

:class:`SchemaRepairPort` is the port; :class:`EngineSchemaRepairPort` is the
adapter over EG's generated senders on the engine client's own loop.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

#: How long one repair call may block the sync that drives it.
DEFAULT_TIMEOUT_S = 30.0


class SchemaRepairPort(Protocol):
    """Approval leases and schema attachment on one request graph."""

    @property
    def graph(self) -> str: ...

    def issue_approval(self, request: Mapping[str, Any]) -> Mapping[str, Any]: ...

    def get_lease(self, tenant: str, lease_id: str) -> Mapping[str, Any] | None: ...

    def close_approval(self, tenant: str, lease_id: str, revision: int) -> None: ...

    def validate_repair(self, source_id: str, contract: Mapping[str, Any]) -> Any: ...

    def attach_approved(
        self, source_id: str, contract: Mapping[str, Any], approval_lease_id: str
    ) -> Any: ...

    def for_graph(self, graph: str) -> SchemaRepairPort: ...


def _payload(result: Any) -> Any:
    return getattr(result, "payload", result)


def _as_mapping(value: Any) -> Mapping[str, Any] | None:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return value if isinstance(value, Mapping) else None


@dataclass(frozen=True, slots=True)
class EngineSchemaRepairPort:
    """EG's generated senders, driven on the engine client's loop."""

    client: Any
    loop: asyncio.AbstractEventLoop
    graph: str
    timeout_s: float = DEFAULT_TIMEOUT_S

    @classmethod
    def of(cls, graph_compute: Any) -> EngineSchemaRepairPort:
        """Bind to the graph a ``GraphCompute`` view serves."""
        return cls(
            graph_compute._engine_async_client(),
            graph_compute._engine_loop(),
            str(graph_compute.graph_name),
        )

    def _coordination(self, sender_name: str, params: dict[str, Any], key: str) -> Any:
        from epistemic_graph.generated import coordination

        sender = getattr(coordination, sender_name)
        call = sender(self.client, params, self.graph, idempotency_key=key)
        future = asyncio.run_coroutine_threadsafe(call, self.loop)
        return _payload(future.result(timeout=self.timeout_s))

    def issue_approval(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
        key = str(request["idempotency_key"])
        params = {"request": dict(request)}
        answer = self._coordination("send_issue_control_lease", params, key)
        return _as_mapping(answer) or {}

    def get_lease(self, tenant: str, lease_id: str) -> Mapping[str, Any] | None:
        params = {"tenant": tenant, "lease_id": lease_id}
        answer = self._coordination("send_get_control_lease", params, f"get:{lease_id}")
        return _as_mapping(answer)

    def close_approval(self, tenant: str, lease_id: str, revision: int) -> None:
        key = f"schema-repair-close:{lease_id}"
        request = {
            "tenant": tenant,
            "lease_id": lease_id,
            "expected_revision": revision,
            "to": "expired",
            "idempotency_key": key,
        }
        params = {"request": request}
        self._coordination("send_transition_control_lease", params, key)

    def _schema(self, op: dict[str, Any], key: str) -> Any:
        from epistemic_graph.generated.reasoning import send_graph_schema

        call = send_graph_schema(
            self.client, {"op": op}, self.graph, idempotency_key=key
        )
        future = asyncio.run_coroutine_threadsafe(call, self.loop)
        return _as_mapping(_payload(future.result(timeout=self.timeout_s)))

    def validate_repair(self, source_id: str, contract: Mapping[str, Any]) -> Any:
        op = {
            "op": "validate_repair",
            "source_id": source_id,
            "contract": dict(contract),
        }
        return self._schema(op, f"schema-validate:{self.graph}:{source_id}")

    def attach_approved(
        self, source_id: str, contract: Mapping[str, Any], approval_lease_id: str
    ) -> Any:
        op = {
            "op": "attach_approved",
            "source_id": source_id,
            "contract": dict(contract),
            "approval_lease_id": approval_lease_id,
        }
        return self._schema(op, f"schema-approved:{approval_lease_id}")

    def for_graph(self, graph: str) -> EngineSchemaRepairPort:
        return EngineSchemaRepairPort(self.client, self.loop, graph, self.timeout_s)


__all__ = ["DEFAULT_TIMEOUT_S", "EngineSchemaRepairPort", "SchemaRepairPort"]
