#!/usr/bin/python
from __future__ import annotations

"""Prioritized replay buffer for sample-efficient self-evolution (CONCEPT:AHE-3.0).

Distilled from MEMO (`.specify/specs/research-evolution-20260606/` plan b4-03,
F4 "prioritized replay of decisive states"): a capacity-bounded buffer of past
evolution states that surfaces *rare / decisive* states preferentially, so the
exploration loop revisits the cases that carry the most information instead of the
common path. Inverse-frequency priority — a state whose ``key`` (e.g. its base
variant / regime) has been visited least is the most likely to be replayed —
combined with a configurable ``alpha`` sharpening exponent and a seed-faithful
sampler so a replay is reproducible.

Pure Python, deterministic given a seed; no model, no GPU. Wired into the live
``AgenticEvolutionEngine.run_evolution_cycle`` (each cycle pushes its outcome) and
read back via ``AgenticEvolutionEngine.sample_replay``.

Concept: replay-buffer
"""

import random
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ReplayItem:
    """One stored state and its replay bookkeeping."""

    payload: Any
    key: str
    visits: int = 0
    seq: int = 0  # insertion order (FIFO tiebreak for eviction)


@dataclass
class PrioritizedReplayBuffer:
    """Capacity-bounded, inverse-frequency prioritized replay buffer.

    Priority of an item is ``(1 / (1 + visits[key])) ** alpha`` — the less a key
    has been seen/replayed, the higher its priority (``alpha`` sharpens the
    preference). Sampling is weighted, without replacement, and **increments the
    sampled keys' visit counts** so repeated draws naturally spread coverage.
    """

    capacity: int = 128
    alpha: float = 0.6
    _items: list[ReplayItem] = field(default_factory=list)
    _key_visits: dict[str, int] = field(default_factory=dict)
    _seq: int = 0

    def add(self, payload: Any, *, key: str | None = None) -> ReplayItem:
        """Append a state (``key`` groups states for inverse-frequency priority)."""
        k = key if key is not None else str(self._seq)
        item = ReplayItem(
            payload=payload, key=k, visits=self._key_visits.get(k, 0), seq=self._seq
        )
        self._seq += 1
        self._items.append(item)
        self._evict_if_needed()
        return item

    def priority(self, item: ReplayItem) -> float:
        """Inverse-frequency priority for ``item`` (higher = replay sooner)."""
        visits = self._key_visits.get(item.key, 0)
        return (1.0 / (1.0 + visits)) ** self.alpha

    def sample(self, n: int = 1, *, seed: int | None = None) -> list[Any]:
        """Weighted, without-replacement sample of ``n`` payloads by priority.

        Sampled keys have their visit count incremented (so the next sample favors
        the now-rarer states). Deterministic for a fixed ``seed``.
        """
        if not self._items or n <= 0:
            return []
        rng = random.Random(seed)
        pool = list(self._items)
        chosen: list[Any] = []
        for _ in range(min(n, len(pool))):
            weights = [self.priority(it) for it in pool]
            total = sum(weights)
            if total <= 0:
                pick_idx = rng.randrange(len(pool))
            else:
                r = rng.random() * total
                upto = 0.0
                pick_idx = len(pool) - 1
                for i, w in enumerate(weights):
                    upto += w
                    if upto >= r:
                        pick_idx = i
                        break
            item = pool.pop(pick_idx)
            self._key_visits[item.key] = self._key_visits.get(item.key, 0) + 1
            item.visits = self._key_visits[item.key]
            chosen.append(item.payload)
        return chosen

    def _evict_if_needed(self) -> None:
        """Drop the lowest-priority items (FIFO tiebreak) over capacity."""
        if len(self._items) <= self.capacity:
            return
        # Keep the highest-priority `capacity` items (rarest/decisive survive);
        # on a priority tie keep the newer item (evict oldest, FIFO).
        self._items.sort(key=lambda it: (self.priority(it), it.seq), reverse=True)
        self._items = self._items[: self.capacity]

    def __len__(self) -> int:
        return len(self._items)

    def stats(self) -> dict[str, Any]:
        return {
            "size": len(self._items),
            "distinct_keys": len(self._key_visits),
            "capacity": self.capacity,
        }


__all__ = ["PrioritizedReplayBuffer", "ReplayItem"]
