"""Shared in-memory graph-engine double for the assimilation test suites.

The assimilation plan-synthesis and pilot suites each drove a tiny graph stand-in
(a node dict plus out/in edge indexes behind ``engine.graph``) with the
``add_node``/``link_nodes`` engine surface. :class:`AssimilationEngine` is that one
double. Synthesis folds each feature into ONE canonical Gap through EG's typed Gap
surface, so ``with_market=True`` also attaches a
:class:`~tests.unit.work_market_fakes.FakeWorkMarket` as ``engine.market`` (exposed as
``engine.client``); run Gap-writing code inside
:func:`tests.unit.fleet_autonomy_fakes.verified_fleet_session`.
"""

from __future__ import annotations

from typing import Any

from tests.unit.work_market_fakes import attach_market

__all__ = [
    "AssimilationEngine",
    "AssimilationGraph",
    "market_engine",
    "two_open_features",
]


class AssimilationGraph:
    """A node dict with out/in edge indexes, read through a networkx-like surface."""

    def __init__(self, nodes: dict[str, dict[str, Any]]) -> None:
        self._n = dict(nodes)
        self._out: dict[str, list[tuple[str, str, dict[str, Any]]]] = {}
        self._in: dict[str, list[tuple[str, str, dict[str, Any]]]] = {}

    def nodes(self, data: bool = False) -> list[Any]:
        return list(self._n.items()) if data else list(self._n)

    def add_node(self, nid: str, attrs: dict[str, Any]) -> None:
        self._n[nid] = attrs

    def add_edge(self, src: str, dst: str, props: dict[str, Any]) -> None:
        self._out.setdefault(src, []).append((src, dst, props))
        self._in.setdefault(dst, []).append((src, dst, props))

    def out_edges(self, nid: str, data: bool = False) -> list[Any]:
        edges = self._out.get(nid, [])
        return edges if data else [(s, t) for s, t, _ in edges]

    def in_edges(self, nid: str, data: bool = False) -> list[Any]:
        edges = self._in.get(nid, [])
        return edges if data else [(s, t) for s, t, _ in edges]


class AssimilationEngine:
    """The ``add_node``/``link_nodes`` engine surface over an :class:`AssimilationGraph`."""

    def __init__(
        self,
        nodes: dict[str, dict[str, Any]] | None = None,
        *,
        with_market: bool = False,
    ) -> None:
        self.graph = AssimilationGraph(nodes or {})
        self.backend = None
        if with_market:
            # The canonical Gap is EG's typed upsert (the harness-evolution
            # work-market requirement), not a graph node.
            self.market = attach_market(self)

    def add_node(self, nid, node_type, properties=None, ephemeral=False) -> None:
        self.graph.add_node(nid, {**(properties or {}), "type": node_type})

    def link_nodes(self, src, dst, rel_type, properties=None, ephemeral=False) -> None:
        self.graph.add_edge(src, dst, properties or {})


def market_engine(nodes: dict[str, dict[str, Any]]) -> AssimilationEngine:
    """An :class:`AssimilationEngine` wired to a fresh EG work-market double."""
    return AssimilationEngine(nodes, with_market=True)


def two_open_features() -> dict[str, dict[str, Any]]:
    """Two open capability features (one KG, one ORCH), each citing one source."""
    return {
        "f1": {
            "type": "capability",
            "name": "exec-rag planner",
            "concept_ids": ["AU-KG.retrieval.memory-first-retrieval"],
            "research_sources": ["arxiv:pyrag"],
            "status": "open",
        },
        "f2": {
            "type": "capability",
            "name": "social swarm",
            "concept_ids": ["AU-ORCH.dispatch.kg-governed-agent-swarm"],
            "research_sources": ["arxiv:mass"],
            "status": "open",
        },
    }
