"""An in-memory graph engine that answers the Agent Library's own reads."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


def _listed(node: dict[str, Any], params: dict[str, Any]) -> bool:
    """The library list predicate: any A2A agent, or a library-written entry."""
    rtype = node.get("resource_type")
    if rtype == params["a2a"]:
        return True
    owned = node.get("provider_ref") == params["ref"]
    return owned and rtype in {params["skill"], params["graph"]}


@dataclass
class FakeBackend:
    engine: FakeLibraryEngine

    def execute(self, query: str, params: dict[str, Any] | None = None) -> list:
        return self.engine.query_cypher(query, params)


@dataclass
class FakeLibraryEngine:
    """Nodes by id plus typed edges; reads match the library's query shapes."""

    nodes: dict[str, dict[str, Any]] = field(default_factory=dict)
    labels: dict[str, str] = field(default_factory=dict)
    edges: list[tuple[str, str, str]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.backend = FakeBackend(self)

    def add_node(self, node_id: str, label: str, props: dict[str, Any]) -> None:
        self.nodes[node_id] = {"id": node_id, **props}
        self.labels[node_id] = label

    def link_nodes(self, source: str, target: str, rel: str) -> None:
        self.edges.append((source, target, rel))

    def add_tool(self, tool_id: str, name: str, server: str) -> None:
        self.add_node(tool_id, "Tool", {"name": name, "mcp_server": server})

    def add_prompt(self, prompt_id: str, blueprint: dict[str, Any]) -> None:
        self.add_node(prompt_id, "Prompt", {"json_blueprint": json.dumps(blueprint)})

    def _with(self, label: str) -> list[dict[str, Any]]:
        return [n for i, n in self.nodes.items() if self.labels[i] == label]

    def query_cypher(self, query: str, params: dict[str, Any] | None = None) -> list:
        params = params or {}
        if query.startswith("MATCH (r:CallableResource) WHERE"):
            return [
                {"r": n} for n in self._with("CallableResource") if _listed(n, params)
            ]
        if query.startswith("MATCH (r:CallableResource {id: $id})"):
            node = self.nodes.get(params["id"])
            return [{"r": node}] if node else []
        if "-[:USES_TOOL]->" in query:
            return [
                {"id": t, "name": self.nodes.get(t, {}).get("name")}
                for s, t, rel in self.edges
                if s == params["id"] and rel == "USES_TOOL"
            ]
        if query.startswith("MATCH (t:Tool) WHERE t.mcp_server = $s"):
            return [
                {"id": n["id"]}
                for n in self._with("Tool")
                if n.get("mcp_server") == params["s"]
            ]
        if query.startswith("MATCH (p:Prompt)"):
            return [{"p": n} for n in self._with("Prompt")]
        raise AssertionError(f"unexpected query: {query}")
