"""Live-path tests for the extraction job manager (CONCEPT:AU-KG.compute.code-intelligence-tools).

Drives the manager end-to-end (submit → GPU-slot scheduler → runner → persist)
against a fake engine, with the LLM call monkeypatched so no GPU is required.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from agent_utilities.knowledge_graph.extraction import job_manager as jm
from agent_utilities.knowledge_graph.extraction.fact_extractor import ExtractedFact
from agent_utilities.knowledge_graph.extraction.job_manager import (
    EngineStoreAdapter,
    ExtractionJobManager,
    GraphCheckpointStore,
)
from agent_utilities.knowledge_graph.ingestion.gpu_slot_scheduler import JobState
from agent_utilities.security.persistence_privacy import persistence_reference


@pytest.fixture(autouse=True)
def _force_local_process_authority(monkeypatch):
    """Pin the deterministic, no-network local-process identity path.

    ``ExtractionJobManager.submit()``/``_run_job`` resolve identity per call
    via ``security.request_identity.system_write_session`` (BUG-060). Most
    tests in this file submit with no ambient ``GraphSession`` bound, so that
    call falls through to ``local_process_authority_enabled`` -- which is
    itself environment-dependent (the session-scoped ``_session_engine``
    autouse fixture in ``conftest.py`` exports ``GRAPH_SERVICE_ENDPOINTS`` for
    the whole run whenever a real engine binary is resolvable, even though
    these tests use ``_FakeEngine`` and never touch it). Pin the branch so
    these tests behave identically with or without a real engine binary
    present -- same technique as
    ``tests/unit/knowledge_graph/test_bug033_actor_binding.py``.
    """
    import agent_utilities.security.request_identity as ri

    monkeypatch.setattr(ri, "local_process_authority_enabled", lambda _config: True)
    monkeypatch.setattr(ri, "_system_write_session", None)
    yield
    monkeypatch.setattr(ri, "_system_write_session", None)


def _make_session(actor_id: str, tenant: str = "tenant-a"):
    """Build a real, verified ``GraphSession`` for ``actor_id`` (test helper)."""
    from agent_utilities.knowledge_graph.core.session import GraphSession
    from agent_utilities.models.company_brain import ActorType
    from agent_utilities.security.brain_context import ActorContext

    actor = ActorContext(
        actor_id=actor_id,
        actor_type=ActorType.HUMAN,
        roles=(),
        tenant_id=tenant,
        authenticated=True,
    )
    return GraphSession(
        actor=actor,
        tenant=tenant,
        scopes=frozenset({"kg:write"}),
        graph=f"{tenant}-graph",
        policy_version="v1",
        audience="test-audience",
    )


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


class _IdentityCapturingEngine(_FakeEngine):
    """A ``_FakeEngine`` that also records which ``GraphSession`` actor was
    AMBIENT at the moment each write happened -- exactly what the real
    ``IntelligenceGraphEngine.add_node``/``add_edge`` resolve via
    ``resolve_session(None, ...)`` -> ``GraphSession.from_ambient()`` before
    stamping ownership. This is the seam BUG-060's fix operates on (the
    ``session.py`` ``GraphSession`` contextvar), so it is the correct
    observation point for proving per-call attribution -- not
    ``brain_context.current_actor()``, which only the REAL engine binds
    internally (this fake never does)."""

    def __init__(self) -> None:
        super().__init__()
        self.edge_actors: list[str] = []

    def add_edge(self, source: str, target: str, rel_type: str = "", **props) -> None:
        from agent_utilities.knowledge_graph.core.session import current_session

        session = current_session()
        self.edge_actors.append(session.actor.actor_id if session else "NO-SESSION")
        super().add_edge(source, target, rel_type, **props)


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
                    subject=subj,
                    predicate="rel",
                    object=obj,
                    confidence=90,
                    source_file=source,
                ).model_dump(),
                "is_duplicate": False,
                "max_similarity": 0.0,
            }
        yield {
            "type": "done",
            "total_facts": 2,
            "duplicate_facts": 0,
            "unique_facts": 2,
        }

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
async def test_bug060_second_caller_is_not_attributed_to_the_first(
    _canned_facts,
) -> None:
    """BUG-060 known-bad input: two DIFFERENT authenticated identities submit
    jobs in sequence through the SAME manager -- i.e. the SAME persistent
    scheduler worker task/singleton, exactly as happens in the real gateway
    process. The manager's worker task is created lazily on the FIRST
    ``submit()``/``ensure_started()`` call, so job A's submission (under
    alice's ambient session) is what actually creates it.

    Pre-fix, ``asyncio.create_task`` snapshots alice's ambient
    ``GraphSession`` into that persistent task's own context at creation
    time, and every later job -- including job B, submitted afterward under
    bob's DIFFERENT ambient session -- silently runs (and writes its facts)
    under alice's identity, because the writer resolves identity from
    whatever is ambient *inside the persistent task*, not from who actually
    submitted that job. A test using only one identity cannot see this: it
    would pass whether or not the singleton-capture bug is present, because
    there is no second, different caller to be misattributed onto. Revert
    the ``use_session(...)`` binding in ``ExtractionJobManager._run_job`` and
    this test fails: job B's edges are observed under ``writer:alice``
    instead of ``writer:bob``.
    """
    from agent_utilities.knowledge_graph.core.session import use_session

    engine = _IdentityCapturingEngine()
    mgr = ExtractionJobManager(engine)

    session_alice = _make_session("writer:alice")
    session_bob = _make_session("writer:bob")

    # Job A: submitted while alice is ambient. This is also the call that
    # lazily creates the scheduler's persistent worker task, so -- pre-fix --
    # alice's identity is what gets baked into that task's own context.
    with use_session(session_alice):
        jid_a = await mgr.submit(text="doc from alice", dedup=False)
    await _wait_until(
        lambda: (mgr.status(jid_a) or {}).get("state") == str(JobState.DONE)
    )

    # Job B: submitted later, through the SAME already-running manager, while
    # a DIFFERENT identity (bob) is ambient. The persistent worker task is
    # not recreated -- it just picks up job B on its next loop iteration.
    with use_session(session_bob):
        jid_b = await mgr.submit(text="doc from bob", dedup=False)
    await _wait_until(
        lambda: (mgr.status(jid_b) or {}).get("state") == str(JobState.DONE)
    )

    await mgr._scheduler.stop()

    # Two facts per job -> two distinct (subject, predicate, object) edges.
    assert len(engine.edge_actors) == 4
    job_a_actors, job_b_actors = engine.edge_actors[:2], engine.edge_actors[2:]
    assert job_a_actors == ["writer:alice", "writer:alice"], (
        "job A's own facts must be attributed to job A's own submitter"
    )
    assert job_b_actors == ["writer:bob", "writer:bob"], (
        "job B's facts were attributed to the SECOND caller (bob), not "
        "silently reused from the FIRST caller (alice) who happened to be "
        "ambient when the persistent worker task was created -- the exact "
        "cross-user misattribution BUG-060 describes"
    )


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
    assert set(job.checkpoint.get("done_files", [])) == {
        persistence_reference("source", name, namespace="extraction-corpus")
        for name in ("a.md", "b.md")
    }
    # 2 files × 2 facts each
    assert mgr.status(jid)["total_facts"] == 4
    await mgr._scheduler.stop()


@pytest.mark.asyncio
async def test_stream_yields_facts_then_job_done(_canned_facts) -> None:
    engine = _FakeEngine()
    mgr = ExtractionJobManager(engine)
    jid = await mgr.submit(text="doc", dedup=False)
    # consume the live SSE stream to completion
    seen = [ev async for ev in mgr.stream(jid)]
    types = [e["type"] for e in seen]
    assert types[-1] == "job_done"
    assert types.count("fact") == 2
    assert "round_start" in types
    await mgr._scheduler.stop()


@pytest.mark.asyncio
async def test_jobs_are_owner_scoped_and_checkpoints_are_metadata_only(
    _canned_facts,
) -> None:
    engine = _FakeEngine()
    mgr = ExtractionJobManager(engine)
    owner = persistence_reference("owner", "tenant-a", namespace="test")
    other = persistence_reference("owner", "tenant-b", namespace="test")
    private_text = "person@example.test reads /private/location with top-secret"

    jid = await mgr.submit(text=private_text, dedup=False, owner_ref=owner)
    assert mgr.status(jid, owner_ref=owner) is not None
    assert mgr.status(jid, owner_ref=other) is None
    with pytest.raises(KeyError):
        mgr.jsonl(jid, owner_ref=other)

    persisted = json.dumps(engine.nodes, sort_keys=True)
    for forbidden in (
        private_text,
        "person@example.test",
        "/private/location",
        "top-secret",
    ):
        assert forbidden not in persisted
    assert owner in persisted
    await mgr._scheduler.stop()


@pytest.mark.asyncio
async def test_caller_owner_must_already_be_an_opaque_reference() -> None:
    mgr = ExtractionJobManager(_FakeEngine())
    with pytest.raises(ValueError, match="opaque persistence reference"):
        await mgr.submit(text="document", owner_ref="person@example.test")
    await mgr._scheduler.stop()


def test_graph_checkpoint_store_roundtrip() -> None:
    engine = _FakeEngine()
    store = GraphCheckpointStore(engine)
    from agent_utilities.knowledge_graph.ingestion.gpu_slot_scheduler import Job

    store.save(
        Job(job_id="x", state=JobState.RUNNING, checkpoint={"done_files": ["a"]})
    )
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
