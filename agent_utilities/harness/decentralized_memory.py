#!/usr/bin/python
from __future__ import annotations

"""Decentralized-but-collaborative agent memory.

CONCEPT:KG-2.82 — Decentralized Memory

Operationalizes DecentMem: "Self-Evolving Multi-Agent Systems via Decentralized
Memory" (arXiv:2605.22721). The prior behaviour we are surpassing is a single
*centralized* shared memory bank: every agent reads and writes one pool, which
mixes provenance, leaks one agent's mistakes into another's recall, and destroys
the signal of *who* actually solved each piece of a task.

DecentMem instead gives **each agent its own private memory**, split into two
pools:

* an **exploitation pool** — past trajectories the agent has actually run and can
  safely reuse, and
* an **exploration pool** — fresh, LLM-generated candidate strategies that have
  not yet proven themselves.

A candidate is *promoted* from the exploration pool into the exploitation pool
once it proves useful, so each agent's reusable memory grows from evidence rather
than from speculation. Decentralizing the memory would normally throw away the
coordination signal of a multi-agent system, so DecentMem additionally keeps a
**collaboration-aware trace** recording which agent solved which piece of which
task (and in what role) — the coordination signal survives even though the
working memory is private per agent.

This module is pure composition over
:class:`~agent_utilities.harness.evolving_memory.EvolvingMemoryStore`: each
``(agent_id, pool)`` gets its own private store instance (created lazily), so all
CRUD, dedup, reconciliation, and relevance ranking are reused rather than
reimplemented. Records carry their owning ``agent_id`` and ``pool`` in metadata.

Concept: decentralized-memory
"""

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from .evolving_memory import EvolvingMemoryStore, MemoryBank, MemoryRecord
from .explore_exploit_router import ExploreExploitRouter

__all__ = [
    "MemoryPool",
    "Contribution",
    "DecentralizedMemory",
]


class MemoryPool(StrEnum):
    """The two private pools each agent keeps under DecentMem.

    ``EXPLOIT`` holds proven, reusable past trajectories; ``EXPLORE`` holds fresh
    candidate strategies awaiting validation. Promotion moves a record from
    ``EXPLORE`` to ``EXPLOIT``.
    """

    EXPLOIT = "exploit"
    EXPLORE = "explore"


@dataclass
class Contribution:
    """One entry in the collaboration-aware trace.

    Records that ``agent_id`` played ``role`` in solving piece ``task_id``, with a
    short ``summary`` of what it contributed. This is the coordination signal that
    decentralizing the working memory would otherwise discard.
    """

    agent_id: str
    task_id: str
    role: str
    summary: str


class DecentralizedMemory:
    """Per-agent private memory with a shared collaboration trace (KG-2.82).

    Holds one :class:`EvolvingMemoryStore` per ``(agent_id, pool)`` pair, created
    on first use. Every public read is scoped to a single agent, so one agent can
    never recall another agent's records (privacy is structural, not a filter).
    The only cross-agent surface is the append-only collaboration trace, which
    deliberately preserves who-solved-what across the decentralized fleet.
    """

    def __init__(self, engine: Any = None, embedder: Any = None) -> None:
        """Create an empty decentralized memory.

        ``engine`` and ``embedder`` are forwarded unchanged to every per-agent
        :class:`EvolvingMemoryStore` so durable mirroring and semantic relevance
        behave identically to a centralized store when configured.
        """
        self._engine = engine
        self._embedder = embedder
        self._stores: dict[tuple[str, MemoryPool], EvolvingMemoryStore] = {}
        self._trace: list[Contribution] = []
        # CONCEPT:AHE-3.33 — one online exploit/explore bandit router per agent,
        # so each agent learns its OWN exploit/explore balance from reward feedback
        # (DecentMem: the balance is not a hard-coded schedule).
        self._routers: dict[str, ExploreExploitRouter] = {}

    # -- store routing ------------------------------------------------------

    def _store(self, agent_id: str, pool: MemoryPool) -> EvolvingMemoryStore:
        """Return (creating if needed) the private store for ``(agent_id, pool)``."""
        key = (agent_id, MemoryPool(pool))
        store = self._stores.get(key)
        if store is None:
            store = EvolvingMemoryStore(engine=self._engine, embedder=self._embedder)
            self._stores[key] = store
        return store

    def _tag(
        self,
        agent_id: str,
        pool: MemoryPool,
        metadata: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Stamp the owning agent + pool into a record's metadata."""
        tagged = dict(metadata or {})
        tagged["agent_id"] = agent_id
        tagged["pool"] = MemoryPool(pool).value
        return tagged

    # -- exploit/explore routing (CONCEPT:AHE-3.33) -------------------------

    def _router(self, agent_id: str) -> ExploreExploitRouter:
        """Return (creating if needed) ``agent_id``'s exploit/explore bandit."""
        router = self._routers.get(agent_id)
        if router is None:
            router = ExploreExploitRouter(
                arms=(MemoryPool.EXPLOIT.value, MemoryPool.EXPLORE.value)
            )
            self._routers[agent_id] = router
        return router

    def choose_pool(self, agent_id: str) -> MemoryPool:
        """Bandit-select whether ``agent_id`` should exploit or explore next."""
        return MemoryPool(self._router(agent_id).select())

    def reward(self, agent_id: str, pool: MemoryPool | str, reward: float) -> None:
        """Feed a stage-wise reward back into ``agent_id``'s bandit (KG-2.82).

        The router uses this to converge each agent's own exploit/explore balance
        toward what actually pays off — the DecentMem online-routing signal.
        """
        self._router(agent_id).update(MemoryPool(pool).value, reward)

    def router_stats(self, agent_id: str) -> dict[str, Any]:
        """Expose ``agent_id``'s bandit statistics (counts/means/regret)."""
        return self._router(agent_id).stats()

    # -- write --------------------------------------------------------------

    def record_trajectory(
        self,
        agent_id: str,
        content: str,
        *,
        bank: MemoryBank | str = "skill",
        importance: float = 0.6,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryRecord:
        """Store a proven past trajectory in ``agent_id``'s exploitation pool.

        Trajectories the agent has actually executed are reusable evidence, so
        they default to the ``SKILL`` bank and a higher importance than untested
        candidates.
        """
        store = self._store(agent_id, MemoryPool.EXPLOIT)
        return store.add(
            bank,
            content,
            importance=importance,
            metadata=self._tag(agent_id, MemoryPool.EXPLOIT, metadata),
        )

    def propose_candidate(
        self,
        agent_id: str,
        content: str,
        *,
        bank: MemoryBank | str = "insight",
        importance: float = 0.4,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryRecord:
        """Store a fresh, unproven candidate in ``agent_id``'s exploration pool.

        Candidates (typically LLM-generated strategies) default to the ``INSIGHT``
        bank and a lower importance until promoted by :meth:`promote`.
        """
        store = self._store(agent_id, MemoryPool.EXPLORE)
        return store.add(
            bank,
            content,
            importance=importance,
            metadata=self._tag(agent_id, MemoryPool.EXPLORE, metadata),
        )

    # -- read (privacy-scoped) ---------------------------------------------

    def recall(
        self,
        agent_id: str,
        query: str,
        *,
        pool: MemoryPool | None = None,
        top_k: int = 5,
    ) -> list[MemoryRecord]:
        """Recall ``agent_id``'s own most relevant records, ranked by relevance.

        Searches only this agent's private pools — both when ``pool`` is ``None``,
        or a single pool when specified. A record owned by any other agent can
        never be returned. Results across pools are merged and re-ranked by the
        underlying :meth:`EvolvingMemoryStore.resolve` relevance score, and the
        top ``top_k`` records are returned.
        """
        if pool is not None:
            pools = [MemoryPool(pool)]
        else:
            # CONCEPT:AHE-3.33 — let the agent's bandit pick which pool to favour;
            # both pools are still searched (recall stays complete), but the
            # router-preferred pool is consulted first so its records win ties.
            preferred = self.choose_pool(agent_id)
            other = (
                MemoryPool.EXPLORE
                if preferred is MemoryPool.EXPLOIT
                else MemoryPool.EXPLOIT
            )
            pools = [preferred, other]
        scored: list[tuple[MemoryRecord, float]] = []
        for p in pools:
            key = (agent_id, p)
            store = self._stores.get(key)
            if store is None:
                continue
            scored.extend(store.resolve(query, top_k=top_k))
        scored.sort(key=lambda t: t[1], reverse=True)
        return [rec for rec, _score in scored[:top_k]]

    # -- promotion ----------------------------------------------------------

    def promote(self, agent_id: str, record_id: str) -> MemoryRecord | None:
        """Move a proven candidate from EXPLORE to EXPLOIT for ``agent_id``.

        Looks up ``record_id`` in the agent's exploration pool, re-adds an
        equivalent record (same content/bank/importance, with
        ``metadata['promoted_from']`` set to the original id) to the agent's
        exploitation pool, then soft-retires the original explore record. Returns
        the new exploitation record, or ``None`` if no such active candidate
        exists for this agent.
        """
        explore = self._stores.get((agent_id, MemoryPool.EXPLORE))
        if explore is None:
            return None
        original = explore.get(record_id)
        if original is None or original.status != "active":
            return None

        metadata = self._tag(agent_id, MemoryPool.EXPLOIT, dict(original.metadata))
        metadata["promoted_from"] = record_id
        exploit = self._store(agent_id, MemoryPool.EXPLOIT)
        promoted = exploit.add(
            original.bank,
            original.content,
            importance=original.importance,
            metadata=metadata,
        )
        explore.remove(record_id)
        return promoted

    # -- collaboration trace ------------------------------------------------

    def record_contribution(
        self,
        agent_id: str,
        task_id: str,
        role: str,
        summary: str,
    ) -> Contribution:
        """Append a who-solved-what entry to the shared collaboration trace."""
        contribution = Contribution(
            agent_id=agent_id,
            task_id=task_id,
            role=role,
            summary=summary,
        )
        self._trace.append(contribution)
        return contribution

    def collaboration_trace(self, task_id: str | None = None) -> list[Contribution]:
        """Return the collaboration trace, optionally filtered to one task.

        Order preserves insertion order (the sequence in which agents contributed).
        """
        if task_id is None:
            return list(self._trace)
        return [c for c in self._trace if c.task_id == task_id]

    # -- introspection ------------------------------------------------------

    def agents(self) -> list[str]:
        """Return every agent_id that has at least one private store."""
        seen: dict[str, None] = {}
        for agent_id, _pool in self._stores:
            seen.setdefault(agent_id, None)
        return list(seen)

    def pool_size(self, agent_id: str, pool: MemoryPool) -> int:
        """Number of active records in ``agent_id``'s ``pool`` (0 if untouched)."""
        store = self._stores.get((agent_id, MemoryPool(pool)))
        return store.size if store is not None else 0

    def total_size(self) -> int:
        """Total active records across every agent and pool."""
        return sum(store.size for store in self._stores.values())
