#!/usr/bin/python
from __future__ import annotations

"""Multi-Agent Social System (MASS) — swarm as a governed social graph.

CONCEPT:AU-ORCH.dispatch.kg-governed-agent-swarm — KG-Governed Agent Swarm

Distilled from "Social Theory Should Be a Structural Prior for Agentic AI"
(`.specify/specs/research-evolution-20260606/` plan b2-01). The paper models a
multi-agent system as a triple ``S = (f, g, G)`` — an information-exchange function
``f`` (state → message), an influence-dynamics function ``g`` (neighbor messages →
state update), over a networked interaction structure ``G`` — and names four
structural priors a flat worker pool violates:

* **strategic heterogeneity** — agents occupy distinct *archetype* roles; evaluate
  on the archetype distribution, not the aggregate mean (avoid median-collapse);
* **network-constrained dependence** — observation is local: an agent sees only its
  graph neighborhood ``N(i)``;
* **co-evolution** — agent activity reshapes ``G`` (``G(t+1)=h(G(t),{m})``);
* **distributional instability** — no stationary output distribution; track drift.

This module gives the swarm those priors as a lightweight, deterministic state
model plus a **swarm-health** report implementing the paper's P1–P4 hypothesis
tests (degree-partition heterogeneity, topology variance, neighbor co-evolution
slope, Wasserstein-1 drift — the last reusing
:func:`agent_utilities.graph.population_drift.wasserstein1`). Pure Python, no model;
wired into the live parallel engine as a per-run health snapshot.

Concept: social-system
"""

from dataclasses import dataclass, field

from .population_drift import wasserstein1


@dataclass
class MassAgent:
    """One agent in the social system: archetype role + latent state + neighbors."""

    agent_id: str
    archetype: str = "worker"
    latent_state: float = 0.0
    neighbors: set[str] = field(default_factory=set)


class MultiAgentSocialSystem:
    """Swarm modeled as ``S=(f,g,G)`` with archetypes over an interaction graph."""

    def __init__(self) -> None:
        self._agents: dict[str, MassAgent] = {}

    # -- construction --------------------------------------------------------
    def add_agent(
        self,
        agent_id: str,
        *,
        archetype: str = "worker",
        latent_state: float = 0.0,
        neighbors: set[str] | None = None,
    ) -> MassAgent:
        agent = MassAgent(
            agent_id=agent_id,
            archetype=archetype or "worker",
            latent_state=latent_state,
            neighbors=set(neighbors or ()),
        )
        self._agents[agent_id] = agent
        return agent

    def add_edge(self, a: str, b: str) -> None:
        """Add an undirected interaction edge between two known agents."""
        if a in self._agents and b in self._agents and a != b:
            self._agents[a].neighbors.add(b)
            self._agents[b].neighbors.add(a)

    def neighbors(self, agent_id: str) -> set[str]:
        agent = self._agents.get(agent_id)
        return set(agent.neighbors) if agent else set()

    def degree(self, agent_id: str) -> int:
        return len(self.neighbors(agent_id))

    # -- F3: network-constrained local observability -------------------------
    def observable_messages(
        self, agent_id: str, messages: dict[str, str]
    ) -> dict[str, str]:
        """Restrict ``messages`` to the agent's graph neighborhood ``N(i)``.

        Topology-bound visibility: an agent observes only messages from its
        neighbors (plus itself), never a global broadcast.
        """
        allowed = self.neighbors(agent_id) | {agent_id}
        return {k: v for k, v in messages.items() if k in allowed}

    # -- F2: strategic heterogeneity ----------------------------------------
    def archetype_distribution(self) -> dict[str, int]:
        dist: dict[str, int] = {}
        for a in self._agents.values():
            dist[a.archetype] = dist.get(a.archetype, 0) + 1
        return dist

    def heterogeneity(self) -> float:
        """Normalized Shannon entropy of the archetype distribution ∈ [0, 1].

        1.0 = maximally diverse archetypes; 0.0 = a single (collapsed) archetype.
        """
        import math

        counts = list(self.archetype_distribution().values())
        n = sum(counts)
        if n == 0 or len(counts) <= 1:
            return 0.0
        entropy = -sum((c / n) * math.log(c / n) for c in counts if c)
        return entropy / math.log(len(counts))

    # -- F4: co-evolution h-loop --------------------------------------------
    def co_evolve(self, interactions: list[tuple[str, str]]) -> dict[str, float]:
        """Reshape ``G`` from interaction traces; return updated degree centrality.

        Each observed ``(i, j)`` interaction forms/reinforces an edge — the
        ``G(t+1)=h(G(t),{m})`` update — so the topology tracks who actually
        influences whom, feeding centrality back into routing priority (OS-5.8).
        """
        for a, b in interactions:
            self.add_edge(a, b)
        return self.degree_centrality()

    def degree_centrality(self) -> dict[str, float]:
        n = len(self._agents)
        if n <= 1:
            return {aid: 0.0 for aid in self._agents}
        return {aid: self.degree(aid) / (n - 1) for aid in self._agents}

    # -- F5/F6: distributional instability + P1–P4 swarm-health --------------
    def swarm_health(self, prev_states: list[float] | None = None) -> dict:
        """Compute the P1–P4 swarm-health snapshot (b2-01 F6).

        * **P1** heterogeneity-by-degree: mean latent-state gap between high- and
          low-degree agents (structural, not aggregate).
        * **P2** topology variance: variance of latent states across the population.
        * **P3** co-evolution slope: OLS slope of latent state vs degree (does
          connectivity drive state?).
        * **P4** drift: Wasserstein-1 distance to the previous run's distribution.
        """
        states = [a.latent_state for a in self._agents.values()]
        degrees = [self.degree(a.agent_id) for a in self._agents.values()]
        drift = (
            wasserstein1(prev_states, states)
            if prev_states is not None and states
            else 0.0
        )
        return {
            "agents": len(self._agents),
            "archetypes": self.archetype_distribution(),
            "heterogeneity": round(self.heterogeneity(), 6),
            "degree_centrality": self.degree_centrality(),
            "heterogeneity_by_degree": round(_degree_partition_gap(states, degrees), 6),
            "topology_variance": round(_variance(states), 6),
            "coevolution_slope": round(_ols_slope(degrees, states), 6),
            "w1_drift": round(drift, 6),
        }


def _variance(xs: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    mean = sum(xs) / n
    return sum((x - mean) ** 2 for x in xs) / n


def _degree_partition_gap(states: list[float], degrees: list[int]) -> float:
    """Mean-state gap between above-median-degree and below-median-degree agents."""
    if len(states) < 2:
        return 0.0
    ordered = sorted(degrees)
    median = ordered[len(ordered) // 2]
    high = [s for s, d in zip(states, degrees, strict=False) if d > median]
    low = [s for s, d in zip(states, degrees, strict=False) if d <= median]
    if not high or not low:
        return 0.0
    return abs(sum(high) / len(high) - sum(low) / len(low))


def _ols_slope(xs: list[int], ys: list[float]) -> float:
    """Ordinary least-squares slope of ``ys`` on ``xs`` (0 if degenerate)."""
    n = len(xs)
    if n < 2:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    var_x = sum((x - mx) ** 2 for x in xs)
    if var_x == 0:
        return 0.0
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=False))
    return cov / var_x


__all__ = ["MassAgent", "MultiAgentSocialSystem"]
