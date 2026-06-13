"""Live-path tests for the extraction job manager (CONCEPT:KG-2.65).

Drives the manager end-to-end (submit → GPU-slot scheduler → runner → persist)
against a fake engine, with the LLM call monkeypatched so no GPU is required.
"""

from __future__ import annotations

import asyncio

import pytest

from agent_utilities.knowledge_graph.extraction import job_manager as jm
from agent_utilities.knowledge_graph.extraction.fact_extractor import ExtractedFact
from agent_utilities.knowledge_graph.extraction.job_manager import (
    EngineStoreAdapter,
    ExtractionJobManager,
    GraphCheckpointStore,
)
from agent_utilities.knowledge_graph.ingestion.gpu_slot_scheduler import JobState


class _FakeEngine:
    def __init__(self) -> None:
        self.nodes: dict[str, dict] = {}
        self.edges: list[tuple] = []

    def add_node(self, node_id: str, node_type: str, properties=None) -> None:
        self.nodes[node_id] = {"type": node_type, **(properties or {})}

    def add_edge(self, source: str, target: str, rel_type: str = "", **props) -> None:
        self.edges.append((source, target, rel_type, props))

    def query(self, cypher: str, params=None):
        return [
            {"n": n} for n in self.nodes.values() if n.get("type") == "extraction_job"
        ]


async def _wait_until(predicate, timeout: float = 2.0) -> None:
    async def _poll() -> None:
        while not predicate():
            await asyncio.sleep(0.01)

    await asyncio.wait_for(_poll(), timeout)


@pytest.fixture
def _canned_facts(monkeypatch):
    """Monkeypatch extract_facts to emit two canned fact events (no LLM)."""

    async def _fake_extract(text, **kwargs):
        source = kwargs.get("source_file", "")
        yield {"type": "round_start", "round": 1, "seed": 1}
        for subj, obj in (("Jina AI", "v5"), ("Qwen", "MoE")):
            yield {
                "type": "fact",
                "round": 1,
                "fact": ExtractedFact(
                    subject=subj, predicate="rel", object=obj, confidence=90, source_file=source
                ).model_dump(),
                "is_duplicate": False,
                "max_similarity": 0.0,
            }
        yield {"type": "done", "total_facts": 2, "duplicate_facts": 0, "unique_facts": 2}

    monkeypatch.setattr(jm, "extract_facts", _fake_extract)


@pytest.mark.asyncio
async def test_submit_runs_and_persists_facts(_canned_facts) -> None:
    engine = _FakeEngine()
    mgr = ExtractionJobManager(engine)
    jid = await mgr.submit(text="some document", dedup=False)
    await _wait_until(
        lambda: (mgr.status(jid) or {}).get("state") == str(JobState.DONE)
    )
    status = mgr.status(jid)
    assert status["total_facts"] == 2
    # facts persisted as edges on the engine
    assert len(engine.edges) == 2
    rels = {e[2] for e in engine.edges}
    assert rels == {"rel"}
    # JSONL export reflects the kept facts
    assert "Jina AI" in mgr.jsonl(jid)
    await mgr._scheduler.stop()


@pytest.mark.asyncio
async def test_corpus_checkpoints_per_file(_canned_facts) -> None:
    engine = _FakeEngine()
    mgr = ExtractionJobManager(engine)
    files = [{"name": "a.md", "text": "doc a"}, {"name": "b.md", "text": "doc b"}]
    jid = await mgr.submit(files=files, dedup=False)
    await _wait_until(
        lambda: (mgr.status(jid) or {}).get("state") == str(JobState.DONE)
    )
    job = mgr._scheduler.get(jid)
    assert set(job.checkpoint.get("done_files", [])) == {"a.md", "b.md"}
    # 2 files × 2 facts each
    assert mgr.status(jid)["total_facts"] == 4
    await mgr._scheduler.stop()


def test_graph_checkpoint_store_roundtrip() -> None:
    engine = _FakeEngine()
    store = GraphCheckpointStore(engine)
    from agent_utilities.knowledge_graph.ingestion.gpu_slot_scheduler import Job

    store.save(Job(job_id="x", state=JobState.RUNNING, checkpoint={"done_files": ["a"]}))
    loaded = store.load_all()
    assert len(loaded) == 1
    assert loaded[0].job_id == "x"
    assert loaded[0].checkpoint == {"done_files": ["a"]}


def test_engine_store_adapter_maps_calls() -> None:
    engine = _FakeEngine()
    adapter = EngineStoreAdapter(engine)
    adapter.add_node("n1", label="N1")
    adapter.add_edge("n1", "n2", rel_type="links", confidence=0.5)
    assert engine.nodes["n1"]["label"] == "N1"
    assert engine.edges[0] == ("n1", "n2", "links", {"confidence": 0.5})
