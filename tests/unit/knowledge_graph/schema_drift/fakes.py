"""In-memory stand-ins for the KG node store and the EG repair surface."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

_MATCH = re.compile(r"MATCH \(n:(\w+) \{([^}]*)\}\)")


def _conditions(text: str, params: Mapping[str, Any]) -> dict[str, Any]:
    wanted: dict[str, Any] = {}
    for part in filter(None, (p.strip() for p in text.split(","))):
        key, _, value = part.partition(":")
        value = value.strip()
        wanted[key.strip()] = (
            params[value[1:]] if value.startswith("$") else value.strip("'")
        )
    return wanted


def _returned(query: str) -> list[tuple[str, str]]:
    columns = query.split("RETURN", 1)[1]
    pairs = []
    for column in columns.split(","):
        expr, _, alias = column.strip().partition(" AS ")
        pairs.append((expr.strip().removeprefix("n."), alias.strip()))
    return pairs


class FakeBackend:
    """Answers the single-label ``MATCH (n:L {k: v}) RETURN n.x AS y`` reads."""

    def __init__(self, nodes: dict[str, dict[str, Any]]) -> None:
        self.nodes = nodes
        self.fail_reads = False

    def execute(self, query: str, params: Mapping[str, Any] | None = None) -> list:
        if self.fail_reads:
            raise RuntimeError("store offline")
        match = _MATCH.search(query)
        if match is None:
            return []
        label, wanted = match.group(1), _conditions(match.group(2), params or {})
        rows = []
        for node_id, props in sorted(self.nodes.items()):
            view = {"id": node_id, **props}
            if props.get("node_type") != label:
                continue
            if all(view.get(k) == v for k, v in wanted.items()):
                rows.append({alias: view.get(expr) for expr, alias in _returned(query)})
        return rows


class FakeEngine:
    """``add_node`` upserts (merges) like the real engine; reads via the backend."""

    def __init__(self) -> None:
        self.nodes: dict[str, dict[str, Any]] = {}
        self.backend = FakeBackend(self.nodes)
        self.fail_writes = False

    def add_node(self, node_id: str, node_type: str, properties: dict | None = None):
        if self.fail_writes:
            raise RuntimeError("store read-only")
        merged = dict(self.nodes.get(node_id) or {})
        merged.update(properties or {})
        merged["node_type"] = node_type
        self.nodes[node_id] = merged
        return {"id": node_id}

    def labelled(self, label: str) -> dict[str, dict[str, Any]]:
        return {k: v for k, v in self.nodes.items() if v.get("node_type") == label}


@dataclass
class FakePort:
    """The EG repair surface: leases + schema attach, recorded per graph."""

    graph: str = "live"
    leases: dict[str, dict[str, Any]] = field(default_factory=dict)
    attached: list[tuple[str, str, str, str]] = field(default_factory=list)
    refuse_approved: str = ""

    def issue_approval(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
        lease_id = str(request["lease_id"])
        if lease_id in self.leases:
            return {"outcome": "collision"}
        self.leases[lease_id] = {
            "lease_id": lease_id,
            "kind": request["kind"],
            "grant": dict(request["grant"]),
            "status": "active",
            "revision": 1,
            "tenant": request["tenant"],
        }
        return {"outcome": "issued"}

    def get_lease(self, tenant: str, lease_id: str) -> Mapping[str, Any] | None:
        lease = self.leases.get(lease_id)
        return lease if lease and lease["tenant"] == tenant else None

    def decide(self, lease_id: str, status: str) -> None:
        self.leases[lease_id]["status"] = status
        self.leases[lease_id]["revision"] += 1

    def close_approval(self, tenant: str, lease_id: str, revision: int) -> None:
        lease = self.leases[lease_id]
        assert lease["revision"] == revision
        self.decide(lease_id, "expired")

    def attach_shadow(self, source_id: str, shapes_ttl: str) -> Any:
        self.attached.append((self.graph, "attach", source_id, shapes_ttl))
        return {"changed": True}

    def attach_approved(
        self, source_id: str, shapes_ttl: str, approval_lease_id: str
    ) -> Any:
        if self.refuse_approved:
            raise RuntimeError(self.refuse_approved)
        self.attached.append((self.graph, approval_lease_id, source_id, shapes_ttl))
        return {"changed": True}

    def for_graph(self, graph: str) -> FakePort:
        shadow = FakePort(graph, self.leases, self.attached, self.refuse_approved)
        return shadow
