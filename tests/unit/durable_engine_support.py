"""Shared recording engine double for the fail-closed ActionPolicy tests."""

from __future__ import annotations


class DurableEngine:
    """Minimal durable authority: the policy fails closed without receipt storage."""

    def __init__(self) -> None:
        self.nodes: dict[str, dict] = {}

    def add_node(self, node_id, node_type, properties=None, **_kw):
        self.nodes[node_id] = {"type": node_type, **(properties or {})}

    def query_cypher(self, *_args, **_kwargs):
        return []
