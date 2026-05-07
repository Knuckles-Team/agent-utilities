#!/usr/bin/python
"""Multi-Timescale Memory Dynamics (CONCEPT:KG-2.1 Enhancement).

Derived from: Continual Knowledge Updating (arXiv:2605.05097v1, Score 11.2)

Three memory tiers with exponential decay, consolidation, and pruning:
- WORKING: 5min half-life, promotes at 3+ accesses
- EPISODIC: 4hr half-life, promotes at 5+ accesses
- SEMANTIC: 30-day half-life, permanent
"""

from __future__ import annotations

import hashlib
import logging
import math
from collections import defaultdict
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class MemoryTimescale(str, Enum):
    WORKING = "working"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"


DEFAULT_HALF_LIVES = {
    MemoryTimescale.WORKING: 300.0,
    MemoryTimescale.EPISODIC: 14400.0,
    MemoryTimescale.SEMANTIC: 2592000.0,
}
CONSOLIDATION_THRESHOLDS = {
    MemoryTimescale.WORKING: 3.0,
    MemoryTimescale.EPISODIC: 5.0,
    MemoryTimescale.SEMANTIC: float("inf"),
}


class MemoryEntry(BaseModel):
    """A memory entry with timescale-aware decay (CONCEPT:KG-2.1)."""

    memory_id: str
    content: str
    timescale: MemoryTimescale = MemoryTimescale.WORKING
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    last_accessed: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    access_count: int = 1
    activation: float = 1.0
    relevance_score: float = 0.5
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    source_session: str = ""

    def compute_current_activation(self, half_lives: dict | None = None) -> float:
        lives = half_lives or DEFAULT_HALF_LIVES
        half_life = lives.get(self.timescale, 3600.0)
        last = datetime.fromisoformat(self.last_accessed)
        elapsed = (datetime.now(UTC) - last).total_seconds()
        return self.activation * math.pow(2, -elapsed / half_life)

    def access(self) -> None:
        self.access_count += 1
        self.activation = min(self.activation + 0.5, 10.0)
        self.last_accessed = datetime.now(UTC).isoformat()


class TimescaleMemoryStore:
    """Multi-tier memory with consolidation (CONCEPT:KG-2.1)."""

    def __init__(self, half_lives: dict | None = None, decay_floor: float = 0.01):
        self.half_lives = half_lives or dict(DEFAULT_HALF_LIVES)
        self.decay_floor = decay_floor
        self._memories: dict[str, MemoryEntry] = {}

    def store(
        self,
        content: str,
        *,
        timescale: MemoryTimescale = MemoryTimescale.WORKING,
        tags: list[str] | None = None,
        relevance_score: float = 0.5,
        session_id: str = "",
        metadata: dict | None = None,
    ) -> MemoryEntry:
        memory_id = f"mem:{hashlib.sha256(content.encode()).hexdigest()[:12]}"
        if memory_id in self._memories:
            self._memories[memory_id].access()
            return self._memories[memory_id]
        entry = MemoryEntry(
            memory_id=memory_id,
            content=content,
            timescale=timescale,
            tags=list(tags or []),
            relevance_score=relevance_score,
            source_session=session_id,
            metadata=dict(metadata or {}),
        )
        self._memories[memory_id] = entry
        return entry

    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 5,
        timescale: MemoryTimescale | None = None,
        min_activation: float = 0.0,
    ) -> list[MemoryEntry]:
        query_words = set(query.lower().split())
        candidates = []
        for entry in self._memories.values():
            if timescale and entry.timescale != timescale:
                continue
            activation = entry.compute_current_activation(self.half_lives)
            if activation < min_activation:
                continue
            content_words = set(entry.content.lower().split())
            tag_words = set(t.lower() for t in entry.tags)
            overlap = len(query_words & (content_words | tag_words))
            score = (
                (overlap / max(len(query_words), 1))
                * activation
                * entry.relevance_score
            )
            if score > 0:
                candidates.append((score, entry))
        candidates.sort(key=lambda x: x[0], reverse=True)
        results = []
        for _, entry in candidates[:top_k]:
            entry.access()
            results.append(entry)
        return results

    def consolidate(self) -> list[tuple[str, MemoryTimescale, MemoryTimescale]]:
        promotions = []
        for entry in list(self._memories.values()):
            threshold = CONSOLIDATION_THRESHOLDS.get(entry.timescale, float("inf"))
            if entry.access_count >= threshold:
                old = entry.timescale
                new = self._next_timescale(old)
                if new and new != old:
                    entry.timescale = new
                    entry.activation = 1.0
                    entry.access_count = 0
                    promotions.append((entry.memory_id, old, new))
        return promotions

    def prune(self) -> int:
        to_prune = [
            mid
            for mid, e in self._memories.items()
            if e.compute_current_activation(self.half_lives) < self.decay_floor
        ]
        for mid in to_prune:
            del self._memories[mid]
        return len(to_prune)

    def get_stats(self) -> dict[str, Any]:
        by_tier: dict[str, int] = defaultdict(int)
        for e in self._memories.values():
            by_tier[e.timescale.value] += 1
        return {
            "total_memories": len(self._memories),
            "by_timescale": dict(by_tier),
            "total_accesses": sum(e.access_count for e in self._memories.values()),
        }

    @staticmethod
    def _next_timescale(current: MemoryTimescale) -> MemoryTimescale | None:
        return {
            MemoryTimescale.WORKING: MemoryTimescale.EPISODIC,
            MemoryTimescale.EPISODIC: MemoryTimescale.SEMANTIC,
        }.get(current)
