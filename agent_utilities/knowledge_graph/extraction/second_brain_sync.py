#!/usr/bin/python
from __future__ import annotations

"""Second-brain sync — thin one-call composition over a personal note corpus.

CONCEPT:AU-KG.enrichment.second-brain-note-sync

Closes the gap the surpass-6mo audit flagged as the sharpest DONE/MISSING split
in the ecosystem (``reports/surpass-6mo/04-five-intersections.md`` §5): the
engine-side primitives a personal "second brain" needs were already real and
independently tested — atomic-triple fact extraction (:mod:`.fact_extractor`),
entity/claim extraction (:mod:`~agent_utilities.knowledge_graph.kb.
entity_claim_extractor`), and explicit contradiction/friction detection
(:mod:`~agent_utilities.knowledge_graph.adaptation.contradiction_detector`) —
but nothing sequenced them over a note corpus in one call. This module adds
exactly that sequencing and nothing else: every extraction/inference step below
delegates to an existing, independently-tested primitive.

Pipeline, per note under ``source`` changed since ``since``:

1. **fact_extractor** (:func:`~.fact_extractor.extract_facts` /
   :func:`~.fact_extractor.persist_facts`) — atomic
   ``(subject)-[predicate]->(object)`` facts carrying ``evidence_span`` (a
   verbatim quote from the note) and ``confidence``.
2. **EntityClaimExtractor** (:class:`~agent_utilities.knowledge_graph.kb.
   entity_claim_extractor.EntityClaimExtractor`) — typed entities/claims
   persisted as ``ClaimNode``/``EntityNode``, each new claim then PROPOSED
   (never silently accepted) into the governed :class:`~agent_utilities.
   knowledge_graph.research.claim_flywheel.ClaimFlywheel` lifecycle — the SAME
   state machine the ``graph_claims`` MCP tool drives, so a reviewer can
   ``validate``/``accept``/``retract`` it from there.
3. **ContradictionDetector**, scanning each new claim against topically
   similar EXISTING graph content (the same ``search_hybrid`` candidate
   retrieval the ``graph_analyze action="contradictions"`` tool already uses).
   A finding is persisted as a ``:BeliefRevisionProposal`` — the exact node
   shape :meth:`~agent_utilities.knowledge_graph.research.loop_controller.
   LoopController._run_belief_revision` already persists for its own periodic
   belief-revision pass — so any existing reader of that node type picks up a
   second-brain contradiction too, with zero new UI. Propose-only: this NEVER
   mutates or resolves anything; a human/agent reviewer decides.

Idempotent by construction: each note's id is content-addressed
(``doc:second_brain:<sha256(text)[:16]>``), so a note already synced under a
given corpus is a no-op the next time this runs — re-syncing an unchanged
corpus mints zero new facts, claims, or proposals. ``since`` (a Unix mtime) is
a cheap filesystem pre-filter on TOP of that content-hash guarantee, not a
substitute for it.

Live sources (Nextcloud, Paperless-ngx, ...) are reached through their own
existing connector presets/skills (``nextcloud-files``/``nextcloud-ingest``,
``paperless-documents``/``paperless-ngx-kg-ingestion`` — CONCEPT:AU-KG.ingest.mcp-tool-connector), which
land the notes as local files (or already-ingested Document nodes); this
module then points at that materialized note directory. It never calls a
connector API directly — one reuse path, no new connector code.

Deliberately out of scope (see the module's own tests + the SKILL.md for the
full flow): chunking a single very-large note (fact_extractor's own prompt
already treats a whole document as one unit — personal notes are typically
small enough that this is a non-issue) and full-text embedding/search
(complementary — a caller wanting that can ALSO run ``graph_ingest
action="ingest"`` over the same directory; it is independently idempotent).
"""

import hashlib
import logging
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from agent_utilities.knowledge_graph.adaptation.belief_revision import (
    explain_revision,
    recompute_confidence,
)
from agent_utilities.knowledge_graph.adaptation.contradiction_detector import (
    Claim,
    ContradictionDetector,
    FrictionFinding,
)
from agent_utilities.knowledge_graph.extraction.fact_extractor import (
    ExtractedFact,
    FactDeduper,
    StreamFn,
    extract_facts,
    persist_facts,
)
from agent_utilities.knowledge_graph.extraction.job_manager import EngineStoreAdapter
from agent_utilities.knowledge_graph.kb.entity_claim_extractor import (
    EntityClaimExtractor,
    claim_node_id,
)
from agent_utilities.knowledge_graph.research.claim_flywheel import ClaimFlywheel
from agent_utilities.models.knowledge_graph import BeliefNode

logger = logging.getLogger(__name__)

__all__ = ["SecondBrainSyncResult", "iter_note_files", "sync_second_brain"]

_NOTE_SUFFIXES = frozenset({".md", ".markdown", ".txt"})
# Prior confidence for an existing neighbour the search surface didn't attach
# its own confidence to (e.g. a plain search_hybrid row) — a neutral midpoint,
# never a fabricated strong signal either way.
_NEUTRAL_CONFIDENCE = 0.5
_TOP_K_NEIGHBOURS = 5


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or "corpus"


def iter_note_files(source: str, *, since: float | None = None) -> list[Path]:
    """Resolve ``source`` (a file or a directory) to a sorted note-file list.

    A directory is walked recursively for markdown/text notes
    (``.md``/``.markdown``/``.txt``); an explicit file path is always returned
    regardless of suffix. ``since`` (a Unix mtime) drops files unchanged since
    the last sync — a cheap pre-filter; :func:`sync_second_brain`'s
    content-hash check is the correctness backstop even when this is omitted.
    Returns ``[]`` for a path that is neither a file nor a directory (never
    raises).
    """
    path = Path(source)
    if path.is_file():
        candidates = [path]
    elif path.is_dir():
        candidates = sorted(
            p
            for p in path.rglob("*")
            if p.is_file() and p.suffix.lower() in _NOTE_SUFFIXES
        )
    else:
        return []
    if since is not None:
        candidates = [p for p in candidates if p.stat().st_mtime >= since]
    return candidates


class SecondBrainSyncResult(BaseModel):
    """Summary of one :func:`sync_second_brain` run — the MCP action's payload."""

    corpus_id: str
    notes_seen: int = 0
    notes_synced: int = 0
    notes_skipped_unchanged: int = 0
    facts: int = 0
    claims: int = 0
    claims_proposed: list[str] = Field(default_factory=list)
    contradictions: list[dict[str, Any]] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


def _node_exists(engine: Any, node_id: str) -> bool:
    """Best-effort existence check driving the per-note idempotency skip."""
    try:
        rows = engine.query_cypher(
            "MATCH (n) WHERE n.id = $id RETURN n.id AS id LIMIT 1", {"id": node_id}
        )
    except Exception as e:  # noqa: BLE001 — a failed check just means "not found yet"
        logger.debug("second_brain_sync: existence check failed for %s: %s", node_id, e)
        return False
    return bool(rows)


def _existing_claims(
    engine: Any, query: str, *, exclude_id: str, top_k: int = _TOP_K_NEIGHBOURS
) -> list[Claim]:
    """Topically-similar EXISTING graph content to scan one new claim against.

    Mirrors the ``graph_analyze action="contradictions"`` MCP action's own
    neighbour retrieval (``mcp/tools/analysis_tools.py``) so both surfaces see
    the same candidate pool — the search itself is delegated to the engine's
    ``search_hybrid``, never re-implemented here.
    """
    try:
        neighbours = engine.search_hybrid(query, top_k=top_k) or []
    except Exception as e:  # noqa: BLE001 — a retrieval failure just yields no neighbours
        logger.debug("second_brain_sync: search_hybrid failed: %s", e)
        return []
    existing: list[Claim] = []
    for i, n in enumerate(neighbours):
        if not isinstance(n, dict):
            continue
        cid = str(n.get("id") or (n.get("node", {}) or {}).get("id") or i)
        if cid == exclude_id:
            continue
        text = str(
            n.get("description")
            or n.get("name")
            or (n.get("node", {}) or {}).get("description")
            or ""
        )
        if text:
            existing.append(Claim(id=cid, text=text))
    return existing


def _propose_belief_revision(
    engine: Any,
    *,
    finding: FrictionFinding,
    new_text: str,
    new_confidence: float,
    existing_text: str,
    note_id: str,
    corpus_id: str,
    now: str,
) -> str | None:
    """Persist ONE ``:BeliefRevisionProposal`` for a detected friction finding.

    Reuses the SAME confidence-propagation primitive
    (:func:`~..adaptation.belief_revision.recompute_confidence` /
    :func:`~..adaptation.belief_revision.explain_revision`) and the SAME node
    shape :meth:`~..research.loop_controller.LoopController.
    _run_belief_revision` already persists for its own periodic pass, so any
    existing reader of ``type="BeliefRevisionProposal"`` picks up a
    second-brain contradiction too. Propose-only: never mutates or resolves
    anything. Returns the persisted node id, or ``None`` on a best-effort
    persistence failure.
    """
    new_belief = BeliefNode(
        id=finding.new_id,
        name=new_text[:80],
        statement=new_text,
        confidence=max(0.0, min(1.0, new_confidence)),
        last_reviewed=now,
    )
    existing_belief = BeliefNode(
        id=finding.conflict_id,
        name=existing_text[:80],
        statement=existing_text,
        confidence=_NEUTRAL_CONFIDENCE,
        last_reviewed=now,
    )
    revised_confidence = recompute_confidence(new_belief, [], [existing_belief])
    trace = explain_revision(new_belief, [], [existing_belief], revised_confidence)

    proposal_id = f"BeliefRevisionProposal:{finding.new_id}:{now}"
    payload = {
        "status": "proposal",
        "belief_id": finding.new_id,
        "old_confidence": round(new_belief.confidence, 6),
        "new_confidence": round(revised_confidence, 6),
        "delta": round(revised_confidence - new_belief.confidence, 6),
        "new_contradicted_by_node_ids": [finding.conflict_id],
        "last_reviewed": now,
        "reasoning_trace": trace,
        "reason": finding.reason,
        "severity": finding.severity,
        "similarity": finding.similarity,
        "source_note_id": note_id,
        "corpus_id": corpus_id,
    }
    try:
        engine.add_node(proposal_id, "BeliefRevisionProposal", properties=payload)
    except Exception as e:  # noqa: BLE001 — the proposal is best-effort, like the loop's own pass
        logger.debug("second_brain_sync: could not persist %s: %s", proposal_id, e)
        return None
    return proposal_id


async def sync_second_brain(
    engine: Any,
    source: str,
    *,
    since: float | None = None,
    corpus_name: str = "",
    fact_stream_fn: StreamFn | None = None,
    fact_deduper: FactDeduper | None = None,
) -> SecondBrainSyncResult:
    """One-call second-brain sync — see the module docstring for the pipeline.

    Args:
        engine: The KG engine (``add_node``/``add_edge``/``query_cypher``/
            ``search_hybrid``) — the same object every other
            ``graph_ingest``/``graph_analyze`` action already receives.
        source: A local notes directory or a single note file — a markdown
            folder, an Obsidian vault, or a synced Nextcloud/Paperless export
            (see the module docstring for how those connectors land here).
        since: Optional Unix mtime cursor — only notes modified at/after this
            time are considered (a pre-filter; idempotency itself comes from
            each note's own content hash, so omitting this is always safe).
        corpus_name: Human-readable corpus name; defaults to ``source``.
        fact_stream_fn: Optional override forwarded to
            :func:`~.fact_extractor.extract_facts` (the SAME injection point
            that function already exposes for testability). ``None`` (the
            default, used by the ``graph_ingest`` MCP action) resolves the
            real configured chat model exactly as ``fact_extract`` does.
        fact_deduper: Optional :class:`~.fact_extractor.FactDeduper` shared
            across every note in this sync, so facts dedup across the WHOLE
            corpus, not just within one note (extending fact_extractor's own
            documented cross-file dedup ambition to the corpus as a whole).
            ``None`` (the default) lets each note build its own fresh deduper,
            matching the standalone ``fact_extract`` action's behavior.

    Returns:
        A :class:`SecondBrainSyncResult` summary. Never raises — every
        per-note step is independently best-effort so one bad note never
        blocks the rest of the corpus (mirrors the ``_mine_*``/``_run_belief_
        revision`` sub-step tolerance convention in ``loop_controller.py``).
    """
    corpus_id = f"corpus:{_slug(corpus_name or source)}"
    result = SecondBrainSyncResult(corpus_id=corpus_id)

    try:
        engine.add_node(
            corpus_id,
            "PersonalCorpus",
            properties={"name": corpus_name or source, "source": source},
        )
    except Exception as e:  # noqa: BLE001 — the grouping node is best-effort
        result.errors.append(f"corpus_node: {e}")

    notes = iter_note_files(source, since=since)
    result.notes_seen = len(notes)
    if not notes:
        return result

    extractor = EntityClaimExtractor(engine)
    flywheel = ClaimFlywheel(engine)
    detector = ContradictionDetector()

    for note_path in notes:
        try:
            text = note_path.read_text(encoding="utf-8", errors="ignore")
        except OSError as e:
            result.errors.append(f"read {note_path}: {e}")
            continue
        if not text.strip():
            continue

        content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
        note_id = f"doc:second_brain:{content_hash}"

        if _node_exists(engine, note_id):
            result.notes_skipped_unchanged += 1
            continue

        try:
            engine.add_node(
                note_id,
                "Document",
                properties={
                    "name": note_path.name,
                    "path": str(note_path),
                    "content_hash": content_hash,
                    "corpus_id": corpus_id,
                    "synced_at": _now_iso(),
                },
            )
            engine.add_edge(note_id, corpus_id, "PART_OF")
        except Exception as e:  # noqa: BLE001 — provenance linking is best-effort
            result.errors.append(f"note_node {note_path}: {e}")

        # 1. facts — evidence_span + confidence, persisted as graph edges.
        try:
            facts: list[ExtractedFact] = []
            async for ev in extract_facts(
                text,
                rounds=1,
                source_file=str(note_path),
                stream_fn=fact_stream_fn,
                deduper=fact_deduper,
            ):
                if ev["type"] == "fact":
                    facts.append(ExtractedFact(**ev["fact"]))
            persist_facts(EngineStoreAdapter(engine), facts)
            result.facts += sum(1 for f in facts if not f.is_duplicate)
        except Exception as e:  # noqa: BLE001 — one note's fact pass never blocks the sync
            result.errors.append(f"fact_extract {note_path}: {e}")

        # 2. claims/entities — persisted, then PROPOSED into the governed
        #    ClaimFlywheel lifecycle (never silently accepted).
        try:
            extraction = extractor.extract_and_persist(
                content=text,
                source_id=note_id,
                article_id=note_id,
                domain="second_brain",
            )
        except Exception as e:  # noqa: BLE001 — one note's claim pass never blocks the sync
            result.errors.append(f"extract_claims {note_path}: {e}")
            result.notes_synced += 1
            continue

        now = _now_iso()
        for claim in extraction.claims:
            claim_id = claim_node_id(note_id, claim.claim_text)
            result.claims += 1
            try:
                flywheel.propose(
                    claim_id, reason=f"second-brain-sync: {note_path.name}"
                )
                result.claims_proposed.append(claim_id)
            except Exception as e:  # noqa: BLE001 — governance overlay is best-effort
                result.errors.append(f"propose {claim_id}: {e}")

            # 3. contradictions — propose-only :BeliefRevisionProposal.
            try:
                existing = _existing_claims(
                    engine, claim.claim_text, exclude_id=claim_id
                )
                new_claim = Claim(id=claim_id, text=claim.claim_text)
                for finding in detector.check(new_claim, existing):
                    conflict_text = next(
                        (c.text for c in existing if c.id == finding.conflict_id), ""
                    )
                    proposal_id = _propose_belief_revision(
                        engine,
                        finding=finding,
                        new_text=claim.claim_text,
                        new_confidence=claim.confidence,
                        existing_text=conflict_text,
                        note_id=note_id,
                        corpus_id=corpus_id,
                        now=now,
                    )
                    if proposal_id:
                        result.contradictions.append(
                            {
                                "proposal_id": proposal_id,
                                "new_id": finding.new_id,
                                "conflict_id": finding.conflict_id,
                                "severity": finding.severity,
                                "similarity": finding.similarity,
                            }
                        )
            except Exception as e:  # noqa: BLE001 — the friction scan is best-effort
                result.errors.append(f"contradictions {claim_id}: {e}")

        result.notes_synced += 1

    return result
