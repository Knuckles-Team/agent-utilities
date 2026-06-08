#!/usr/bin/python
"""Tests for the graph-native CRUD evolving-memory store + live evolution wiring.

CONCEPT:KG-2.1
"""

from unittest.mock import MagicMock

import pytest

from agent_utilities.harness.evolving_memory import (
    EvolvingMemoryStore,
    MemoryBank,
    MemoryRecord,
)

pytestmark = pytest.mark.concept("KG-2.1")


# --- add / dedup -----------------------------------------------------------


def test_add_returns_record_in_bank():
    store = EvolvingMemoryStore()
    rec = store.add(MemoryBank.INSIGHT, "Prefer cached results for repeat queries.")
    assert isinstance(rec, MemoryRecord)
    assert rec.bank == MemoryBank.INSIGHT
    assert rec.status == "active"
    assert store.size == 1


def test_add_dedup_reinforces_existing():
    store = EvolvingMemoryStore()
    a = store.add(MemoryBank.SKILL, "Parse JSON safely")
    b = store.add(MemoryBank.SKILL, "parse   json   safely")  # same signature
    assert a.id == b.id  # deduped
    assert b.usage_count == 1  # reinforced
    assert store.size == 1


def test_add_accepts_string_bank():
    store = EvolvingMemoryStore()
    rec = store.add("error", "Timeout on upstream call")
    assert rec.bank == MemoryBank.ERROR


# --- edit / merge / remove -------------------------------------------------


def test_edit_resigns_on_content_change():
    store = EvolvingMemoryStore()
    rec = store.add(MemoryBank.GUIDE, "old content")
    old_sig = rec.signature
    edited = store.edit(rec.id, content="completely new guidance text")
    assert edited is not None
    assert edited.signature != old_sig


def test_merge_soft_retires_loser_and_absorbs():
    store = EvolvingMemoryStore()
    s = store.add(MemoryBank.INSIGHT, "survivor insight", importance=0.6)
    losr = store.add(MemoryBank.INSIGHT, "loser insight", importance=0.9)
    losr.usage_count = 4
    assert store.merge(losr.id, s.id) is True
    assert store.get(losr.id).status == "merged"
    assert store.get(losr.id).merged_into == s.id
    assert store.get(s.id).usage_count == 4
    assert store.get(s.id).importance == pytest.approx(0.9)
    assert store.size == 1  # only survivor active


def test_remove_soft_retires():
    store = EvolvingMemoryStore()
    rec = store.add(MemoryBank.TOOL, "deprecated tool")
    assert store.remove(rec.id) is True
    assert store.get(rec.id).status == "retired"
    assert store.size == 0


# --- query / resolve / reconcile -------------------------------------------


def test_query_filters_bank_and_status():
    store = EvolvingMemoryStore()
    store.add(MemoryBank.SKILL, "skill one")
    store.add(MemoryBank.ERROR, "error one")
    assert len(store.query(MemoryBank.SKILL)) == 1
    assert len(store.query()) == 2


def test_resolve_ranks_by_relevance():
    store = EvolvingMemoryStore()
    store.add(MemoryBank.INSIGHT, "vector index tuning improves recall")
    store.add(MemoryBank.INSIGHT, "a note about cooking pasta")
    ranked = store.resolve("vector index recall", top_k=2)
    assert ranked[0][0].content.startswith("vector index")
    assert ranked[0][1] >= ranked[1][1]


def test_resolve_uses_injected_embedder():
    embedder = MagicMock()
    embedder.score.side_effect = lambda q, t: 1.0 if "good" in t else 0.0
    store = EvolvingMemoryStore(embedder=embedder)
    store.add(MemoryBank.GUIDE, "bad option")
    store.add(MemoryBank.GUIDE, "good option")
    ranked = store.resolve("anything", top_k=1)
    assert ranked[0][0].content == "good option"


def test_reconcile_merges_same_signature():
    store = EvolvingMemoryStore()
    # Two records with the same signature but added via separate banks won't dedup
    # on add (different bank); within one bank, force two actives by bypassing add dedup.
    r1 = store.add(MemoryBank.SKILL, "merge me", importance=0.4)
    # craft a second active with identical signature directly
    r2 = MemoryRecord(
        id="mem:skill:dup",
        bank=MemoryBank.SKILL,
        content="merge me",
        signature=r1.signature,
        importance=0.2,
    )
    store._records[r2.id] = r2
    merged = store.reconcile(MemoryBank.SKILL)
    assert merged == 1
    assert store.size == 1


# --- merge-generalize (b4-03 full reconciliation) --------------------------


def test_merge_generalize_records_provenance():
    store = EvolvingMemoryStore()
    survivor = store.add(
        MemoryBank.INSIGHT, "always validate inputs early", importance=0.8
    )
    loser = store.add(
        MemoryBank.INSIGHT, "validate the inputs up front", importance=0.5
    )
    assert store.merge(loser.id, survivor.id, generalize=True)
    s = store.get(survivor.id)
    assert "generalized_from" in s.metadata
    assert "validate the inputs up front" in s.metadata["generalized_from"]
    assert store.get(loser.id).status == "merged"


def test_reconcile_similar_collapses_near_duplicates():
    store = EvolvingMemoryStore()
    store.add(
        MemoryBank.INSIGHT, "prefer graph native storage for manifests", importance=0.9
    )
    store.add(
        MemoryBank.INSIGHT, "prefer graph native storage of manifests", importance=0.6
    )
    store.add(MemoryBank.INSIGHT, "unrelated note about caching tokens", importance=0.7)
    merged = store.reconcile_similar(MemoryBank.INSIGHT, threshold=0.6)
    assert merged == 1  # the two paraphrases collapse; the unrelated note stays
    actives = store.query(MemoryBank.INSIGHT)
    assert len(actives) == 2
    survivor = next(r for r in actives if "generalized_from" in r.metadata)
    assert survivor.importance >= 0.9  # kept the strongest


# --- durable mirror (live persistence) -------------------------------------


def test_add_mirrors_to_engine():
    engine = MagicMock()
    store = EvolvingMemoryStore(engine=engine)
    rec = store.add(MemoryBank.INSIGHT, "persist me")
    # best-effort graph persistence happened on a live engine
    call = next(c for c in engine.add_node.call_args_list if c.args[0] == rec.id)
    assert call.args[1] == "EvolvingMemoryRecord"
    assert call.kwargs["properties"]["bank"] == "insight"


# --- live path: AgenticEvolutionEngine writes an insight per cycle ----------


def test_evolution_cycle_records_insight():
    from agent_utilities.harness.agentic_evolution_engine import AgenticEvolutionEngine

    eng = AgenticEvolutionEngine(engine=MagicMock())
    eng._lazy_init()
    # Stub the variant pool so the cycle produces winners + health deterministically.
    vp = MagicMock()
    vp.tournament_select.return_value = ["v1", "v2"]
    vp.prune_losers.return_value = 1
    vp.population_health.return_value = {"spread": 0.3, "collapsed": False}
    eng._variant_pool = vp

    report = eng.run_evolution_cycle("base_1", top_k=2)
    assert "insight_id" in report
    # the insight is retrievable from the live store
    rec = eng._memory_store.get(report["insight_id"])
    assert rec is not None and rec.bank == MemoryBank.INSIGHT
    assert rec.metadata["base_id"] == "base_1"


def test_evolution_cycle_pushes_replay_state():
    from agent_utilities.harness.agentic_evolution_engine import AgenticEvolutionEngine

    eng = AgenticEvolutionEngine(engine=MagicMock())
    eng._lazy_init()
    vp = MagicMock()
    vp.tournament_select.return_value = ["v1"]
    vp.prune_losers.return_value = 0
    vp.population_health.return_value = {"spread": 0.1, "collapsed": True}
    eng._variant_pool = vp

    r1 = eng.run_evolution_cycle("base_rare", top_k=1)
    eng.run_evolution_cycle("base_common", top_k=1)
    eng.run_evolution_cycle("base_common", top_k=1)
    assert r1.get("replay_buffer_size", 0) >= 1
    # decisive cycles are replayable; sampling is seed-faithful.
    sample = eng.sample_replay(2, seed=0)
    assert sample and all("base_id" in s for s in sample)
