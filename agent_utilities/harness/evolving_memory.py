#!/usr/bin/python
from __future__ import annotations

"""Graph-native CRUD evolving-memory store.

CONCEPT:KG-2.1 — Unified Memory Manager

One self-curating CRUD memory substrate shared across orchestration, evolution,
and search — the "build-once" convergence of three research plans
(`.specify/specs/research-evolution-20260606/`) that each asked for a different
append/merge memory:

* **b4-03 MEMO** — a persistent CRUD *insight* bank with Add / Edit(merge-
  generalize) / Remove(on-conflict) reconciliation.
* **b8-06 Web2BigTable** — self-evolving *skill* banks (orchestrator/worker).
* **b5-02 BioMedArena** — a typed 4-bank global workspace (error/skill/tool/guide).

Records are typed by :class:`MemoryBank`, deduplicated by content signature, and
reconciled by merge (never hard-deleted — retirement is a soft status change so
provenance survives). The store is authoritative in-memory for the process and
mirrors to the durable GraphBackend best-effort (the repo's prefer-graph-native
policy), exactly like :class:`~agent_utilities.harness.eval_corpus.EvalCorpus`.
``resolve`` ranks records by lexical relevance by default; an embedding backend
(e.g. the HNSW ``CapabilityIndex``) can be supplied for semantic resolution.

Concept: evolving-memory
"""

import hashlib
import re
import time
import uuid
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

_WORD = re.compile(r"[a-z0-9]+")
_NODE_TYPE = "EvolvingMemoryRecord"


def _tokens(text: str) -> set[str]:
    return set(_WORD.findall((text or "").lower()))


def _signature(content: str) -> str:
    return hashlib.sha256(" ".join(_tokens(content)).encode()).hexdigest()[:16]


class MemoryBank(StrEnum):
    """Typed memory banks (b5-02 typed workspace + b4-03 insight + b8-06 skill)."""

    ERROR = "error"
    SKILL = "skill"
    TOOL = "tool"
    GUIDE = "guide"
    INSIGHT = "insight"


class MemoryRecord(BaseModel):
    """One CRUD memory record."""

    id: str
    bank: MemoryBank
    content: str
    signature: str
    status: str = "active"  # active | merged | retired
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    usage_count: int = 0
    merged_into: str | None = None
    created_at: str = Field(
        default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    )
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvolvingMemoryStore:
    """Persist, reconcile, and resolve typed memory records (CONCEPT:KG-2.1)."""

    def __init__(self, engine: Any = None, embedder: Any = None) -> None:
        self.engine = engine
        self.embedder = embedder
        self._records: dict[str, MemoryRecord] = {}

    # -- CRUD ---------------------------------------------------------------

    def add(
        self,
        bank: MemoryBank | str,
        content: str,
        *,
        signature: str | None = None,
        importance: float = 0.5,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryRecord:
        """Add a record; dedup by ``(bank, signature)``.

        On a duplicate active record the existing one is *reinforced*
        (usage_count++ and importance bumped toward the new value) and returned —
        the first half of b4-03's merge-generalize reconciliation.
        """
        bank = MemoryBank(bank)
        sig = signature or _signature(content)

        existing = self._find_active(bank, sig)
        if existing is not None:
            existing.usage_count += 1
            existing.importance = min(1.0, max(existing.importance, importance) + 0.02)
            if metadata:
                existing.metadata.update(metadata)
            self._persist(existing)
            return existing

        record = MemoryRecord(
            id=f"mem:{bank.value}:{uuid.uuid4().hex[:10]}",
            bank=bank,
            content=content,
            signature=sig,
            importance=max(0.0, min(1.0, importance)),
            metadata=dict(metadata or {}),
        )
        self._records[record.id] = record
        self._persist(record)
        return record

    def edit(
        self,
        record_id: str,
        *,
        content: str | None = None,
        importance: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryRecord | None:
        """Edit a record in place (re-signs when content changes)."""
        rec = self._records.get(record_id)
        if rec is None or rec.status != "active":
            return None
        if content is not None:
            rec.content = content
            rec.signature = _signature(content)
        if importance is not None:
            rec.importance = max(0.0, min(1.0, importance))
        if metadata:
            rec.metadata.update(metadata)
        self._persist(rec)
        return rec

    def merge(
        self, loser_id: str, survivor_id: str, *, generalize: bool = False
    ) -> bool:
        """Merge ``loser`` into ``survivor`` (soft-retire + MERGED_INTO link).

        The survivor absorbs the loser's usage_count + metadata and takes the
        max importance — b4-03's Edit-on-conflict generalization. The loser is
        never deleted (status='merged'), so provenance/as-of queries survive.

        With ``generalize=True`` the survivor also records the loser's distinct
        content under ``metadata['generalized_from']`` — the merge-*generalize*
        half: a canonical insight absorbs its near-duplicate variants while
        keeping their provenance, so the bank converges on general rules instead
        of accreting paraphrases.
        """
        loser = self._records.get(loser_id)
        survivor = self._records.get(survivor_id)
        if loser is None or survivor is None or loser_id == survivor_id:
            return False
        survivor.usage_count += loser.usage_count
        survivor.importance = min(1.0, max(survivor.importance, loser.importance))
        for k, v in loser.metadata.items():
            survivor.metadata.setdefault(k, v)
        if generalize and loser.content and loser.content != survivor.content:
            seen = survivor.metadata.setdefault("generalized_from", [])
            if loser.content not in seen:
                seen.append(loser.content)
        loser.status = "merged"
        loser.merged_into = survivor_id
        self._persist(survivor)
        self._persist(loser)
        self._link(loser_id, survivor_id, "MERGED_INTO")
        return True

    def remove(self, record_id: str) -> bool:
        """Soft-retire a record (status='retired'); never hard-deleted."""
        rec = self._records.get(record_id)
        if rec is None or rec.status == "retired":
            return False
        rec.status = "retired"
        self._persist(rec)
        return True

    # -- Read ---------------------------------------------------------------

    def get(self, record_id: str) -> MemoryRecord | None:
        return self._records.get(record_id)

    def query(
        self,
        bank: MemoryBank | str | None = None,
        *,
        status: str = "active",
        limit: int = 100,
    ) -> list[MemoryRecord]:
        """Return records, optionally filtered by bank and status."""
        bank_v = MemoryBank(bank).value if bank is not None else None
        out = [
            r
            for r in self._records.values()
            if (status is None or r.status == status)
            and (bank_v is None or r.bank.value == bank_v)
        ]
        out.sort(key=lambda r: (r.importance, r.usage_count), reverse=True)
        return out[:limit]

    def resolve(
        self,
        query_text: str,
        *,
        bank: MemoryBank | str | None = None,
        top_k: int = 5,
    ) -> list[tuple[MemoryRecord, float]]:
        """Rank active records by relevance to ``query_text``.

        Uses the supplied ``embedder`` when available (``embedder.score(query,
        text) -> float``), else a lexical containment+Jaccard fallback so the
        store always resolves with no model.
        """
        candidates = self.query(bank, status="active", limit=10_000)
        if not candidates:
            return []
        scored: list[tuple[MemoryRecord, float]] = []
        for rec in candidates:
            scored.append((rec, round(self._similarity(query_text, rec.content), 6)))
        scored.sort(key=lambda t: t[1], reverse=True)
        return scored[:top_k]

    def _similarity(self, a: str, b: str) -> float:
        """Relevance/similarity of two texts in [0, 1].

        Uses ``embedder.score`` when available, else lexical containment+Jaccard
        (the no-model fallback shared by ``resolve`` and ``reconcile_similar``).
        """
        if self.embedder is not None:
            return float(self.embedder.score(a, b))
        qa, qb = _tokens(a), _tokens(b)
        if not qa or not qb:
            return 0.0
        inter = len(qa & qb)
        containment = inter / len(qa)
        jaccard = inter / len(qa | qb)
        return 0.7 * containment + 0.3 * jaccard

    def reconcile(self, bank: MemoryBank | str | None = None) -> int:
        """Merge active records that share a signature within a bank.

        Returns the number of records merged away (the highest-importance record
        per signature survives).
        """
        groups: dict[tuple[str, str], list[MemoryRecord]] = {}
        for rec in self.query(bank, status="active", limit=10_000):
            groups.setdefault((rec.bank.value, rec.signature), []).append(rec)
        merged = 0
        for recs in groups.values():
            if len(recs) < 2:
                continue
            recs.sort(key=lambda r: (r.importance, r.usage_count), reverse=True)
            survivor = recs[0]
            for loser in recs[1:]:
                if self.merge(loser.id, survivor.id):
                    merged += 1
        return merged

    def reconcile_similar(
        self,
        bank: MemoryBank | str | None = None,
        *,
        threshold: float = 0.85,
    ) -> int:
        """Merge-generalize *near-duplicate* active records within a bank.

        Unlike :meth:`reconcile` (exact signature only), this collapses records
        whose pairwise similarity (:meth:`_similarity`) is ≥ ``threshold`` into the
        highest-importance survivor with ``generalize=True`` — b4-03's full
        merge-generalize: paraphrased insights converge on one canonical rule that
        keeps its variants' provenance. Returns the number merged away.
        """
        active = self.query(bank, status="active", limit=10_000)
        merged = 0
        consumed: set[str] = set()
        for i, survivor in enumerate(active):
            if survivor.id in consumed:
                continue
            for loser in active[i + 1 :]:
                if loser.id in consumed or loser.bank is not survivor.bank:
                    continue
                if self._similarity(survivor.content, loser.content) >= threshold:
                    if self.merge(loser.id, survivor.id, generalize=True):
                        consumed.add(loser.id)
                        merged += 1
        return merged

    @property
    def size(self) -> int:
        return sum(1 for r in self._records.values() if r.status == "active")

    # -- Durable mirror (best-effort) --------------------------------------

    def _find_active(self, bank: MemoryBank, signature: str) -> MemoryRecord | None:
        for r in self._records.values():
            if r.status == "active" and r.bank == bank and r.signature == signature:
                return r
        return None

    def _persist(self, rec: MemoryRecord) -> None:
        if self.engine is None or not hasattr(self.engine, "add_node"):
            return
        try:
            self.engine.add_node(
                rec.id,
                _NODE_TYPE,
                properties={
                    "bank": rec.bank.value,
                    "content": rec.content,
                    "signature": rec.signature,
                    "status": rec.status,
                    "importance_score": rec.importance,
                    "usage_count": rec.usage_count,
                    "merged_into": rec.merged_into or "",
                    "created_at": rec.created_at,
                },
            )
        except Exception:  # pragma: no cover - durability is best-effort
            pass

    def _link(self, src: str, dst: str, rel: str) -> None:
        if self.engine is None or not hasattr(self.engine, "link_nodes"):
            return
        try:
            self.engine.link_nodes(src, dst, rel)
        except Exception:  # pragma: no cover - best-effort
            pass
