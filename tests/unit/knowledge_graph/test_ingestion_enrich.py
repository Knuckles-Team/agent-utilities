"""The unified always-on enrichment seam (KG-2.8 concepts + KG-2.64 facts).

``IngestionEngine._enrich_text`` is the single text→intelligence seam every
content type funnels through, per AGENTS.md "Native by default". These tests
prove the *fact* layer (previously wired into no ingest path) now persists as
graph edges by default, and that the single opt-out skips the LLM layer.
"""

from __future__ import annotations

from typing import Any

import pytest

from agent_utilities.knowledge_graph.extraction import fact_extractor
from agent_utilities.knowledge_graph.extraction.fact_extractor import ExtractedFact
from agent_utilities.knowledge_graph.ingestion.engine import IngestionEngine


class _RecordingBackend:
    """Minimal GraphBackend double recording node/edge writes."""

    def __init__(self) -> None:
        self.nodes: list[tuple[str, dict[str, Any]]] = []
        self.edges: list[tuple[str, str, dict[str, Any]]] = []

    def add_node(self, node_id: str, **props: Any) -> None:
        self.nodes.append((node_id, props))

    def add_edge(
        self, source: str, target: str, rel_type: str = "", **props: Any
    ) -> None:
        self.edges.append((source, target, {"rel_type": rel_type, **props}))


def _engine() -> tuple[IngestionEngine, _RecordingBackend]:
    backend = _RecordingBackend()
    eng = IngestionEngine(kg_engine=object(), backend=backend)
    return eng, backend


def _fact_event(**over: Any) -> dict[str, Any]:
    fact = ExtractedFact(
        subject="ACME Corp",
        predicate="acquired",
        object="Beta Inc",
        title="ACME acquired Beta",
        description="ACME Corp acquired Beta Inc in 2024.",
        evidence_span="ACME Corp acquired Beta Inc",
        confidence=90,
        tags=["m&a", "corporate"],
        source_file="doc:1",
    )
    return {
        "type": "fact",
        "is_duplicate": False,
        "fact": {**fact.model_dump(), **over},
    }


@pytest.mark.asyncio
async def test_enrich_text_persists_facts_as_edges(monkeypatch):
    eng, backend = _engine()

    async def _fake_extract(text: str, **_kw: Any):  # noqa: ANN401
        yield {"type": "round_start", "round": 1}
        yield _fact_event()
        yield {"type": "done"}

    monkeypatch.setattr(fact_extractor, "extract_facts", _fake_extract)

    out = await eng._enrich_text("doc:1", "ACME Corp acquired Beta Inc.", "document")

    assert out["facts"] == 1
    # subject + object land as canonical Entity nodes; one typed edge interlinks.
    assert {n for n, _ in backend.nodes} == {"acme corp", "beta inc"}
    assert len(backend.edges) == 1
    src, tgt, props = backend.edges[0]
    assert (src, tgt, props["rel_type"]) == ("acme corp", "beta inc", "acquired")
    assert props["confidence"] == pytest.approx(0.9)
    # list tags are flattened so any backend persists them.
    assert props["tags"] == "m&a,corporate"


@pytest.mark.asyncio
async def test_enrich_facts_opt_out_skips_llm(monkeypatch):
    eng, backend = _engine()

    async def _boom(*_a: Any, **_kw: Any):  # noqa: ANN401
        raise AssertionError("fact extraction must not run when enrich_facts=False")
        yield  # pragma: no cover — makes this an async generator

    monkeypatch.setattr(fact_extractor, "extract_facts", _boom)

    out = await eng._enrich_text("d", "text", "document", enrich_facts=False)

    assert out["facts"] == 0
    assert backend.edges == []


@pytest.mark.asyncio
async def test_enrich_text_never_raises_on_extractor_failure(monkeypatch):
    eng, backend = _engine()

    async def _fail(*_a: Any, **_kw: Any):  # noqa: ANN401
        raise RuntimeError("vLLM down")
        yield  # pragma: no cover

    monkeypatch.setattr(fact_extractor, "extract_facts", _fail)

    # Best-effort contract: enrichment degrades to zero, ingestion is unharmed.
    out = await eng._enrich_text("d", "text", "document")
    assert out["facts"] == 0


@pytest.mark.asyncio
async def test_ingest_drains_enrichable_payloads_centrally(tmp_path, monkeypatch):
    """The central seam in ``ingest()`` enriches every adaptor's text payload —
    proving enrichment is global (a prompt now enriches without the prompt
    adaptor calling ``_enrich_text`` itself)."""
    from agent_utilities.knowledge_graph.ingestion.engine import (
        ContentType,
        IngestionManifest,
    )

    eng, _backend = _engine()
    calls: list[tuple[str, str]] = []

    async def _recorder(source_id, text, source_type, title="", **_kw):  # noqa: ANN001
        calls.append((source_id, source_type))
        return {"concepts": 2, "facts": 3}

    monkeypatch.setattr(eng, "_enrich_text", _recorder)
    # Avoid the durable delta-skip / hashing machinery in this unit test.
    monkeypatch.setattr(eng, "_content_identity", lambda _m: None)

    prompt = tmp_path / "p.md"
    prompt.write_text("You are a helpful assistant for ACME Corp.", encoding="utf-8")
    manifest = IngestionManifest(
        content_type=ContentType.PROMPT, source_uri=str(prompt)
    )

    result = await eng.ingest(manifest)

    assert result.status == "success"
    assert calls and calls[0][1] == "prompt"  # the prompt payload was enriched
    assert result.details["enrichment"] == {"concepts": 2, "facts": 3}


@pytest.mark.asyncio
async def test_ingest_enrich_opt_out(tmp_path, monkeypatch):
    from agent_utilities.knowledge_graph.ingestion.engine import (
        ContentType,
        IngestionManifest,
    )

    eng, _backend = _engine()

    async def _boom(*_a, **_kw):  # noqa: ANN002, ANN003
        raise AssertionError("enrich must not run when enrich=False")

    monkeypatch.setattr(eng, "_enrich_text", _boom)
    monkeypatch.setattr(eng, "_content_identity", lambda _m: None)

    prompt = tmp_path / "p.md"
    prompt.write_text("hello", encoding="utf-8")
    manifest = IngestionManifest(
        content_type=ContentType.PROMPT,
        source_uri=str(prompt),
        metadata={"enrich": False},
    )

    result = await eng.ingest(manifest)
    assert result.status == "success"
    assert "enrichment" not in result.details


@pytest.mark.asyncio
async def test_social_adaptor_surfaces_post_text(monkeypatch):
    """The SOCIAL adaptor bubbles the X post's verbatim text into the seam."""
    from agent_utilities.knowledge_graph.ingestion import engine as eng_mod
    from agent_utilities.knowledge_graph.ingestion.engine import (
        ContentType,
        IngestionManifest,
    )

    eng, _backend = _engine()
    seen: list[dict] = []

    async def _recorder(source_id, text, source_type, title="", **_kw):  # noqa: ANN001
        seen.append({"id": source_id, "text": text, "type": source_type})
        return {"concepts": 1, "facts": 1}

    monkeypatch.setattr(eng, "_enrich_text", _recorder)
    monkeypatch.setattr(eng, "_content_identity", lambda _m: None)

    class _FakeBridge:
        def __init__(self, **_kw):
            pass

        async def ingest_browse_result(self, **_kw):
            return {
                "action": "ingest",
                "node_id": "social:x:42",
                "content_text": "ACME Corp shipped a new product today.",
                "title": "X post by @acme",
            }

    monkeypatch.setattr(eng_mod, "XIngestionBridge", _FakeBridge, raising=False)
    # The adaptor imports XIngestionBridge lazily from kb.x_ingestion; patch there.
    import agent_utilities.knowledge_graph.kb.x_ingestion as x_mod

    monkeypatch.setattr(x_mod, "XIngestionBridge", _FakeBridge, raising=False)

    manifest = IngestionManifest(content_type=ContentType.SOCIAL, source_uri="{}")
    result = await eng.ingest(manifest)

    assert result.status == "success"
    assert seen and seen[0]["type"] == "social"
    assert "ACME Corp" in seen[0]["text"]
    assert seen[0]["id"] == "social:x:42"


@pytest.mark.asyncio
async def test_policy_adaptor_surfaces_constitution_text(monkeypatch):
    """The POLICY adaptor bubbles each ingest method's `enrichable` stats up."""
    from agent_utilities.knowledge_graph.ingestion.engine import (
        ContentType,
        IngestionManifest,
    )

    eng, _backend = _engine()
    seen: list[str] = []

    async def _recorder(source_id, text, source_type, title="", **_kw):  # noqa: ANN001
        seen.append(source_type)
        return {"concepts": 0, "facts": 0}

    monkeypatch.setattr(eng, "_enrich_text", _recorder)
    monkeypatch.setattr(eng, "_content_identity", lambda _m: None)

    class _FakeKG:
        def ingest_all_policies(self, **_kw):
            return {
                "total_policies": 3,
                "enrichable": [
                    {
                        "source_id": "project:acme",
                        "text": "The system MUST validate all inputs.",
                        "source_type": "policy",
                        "title": "acme",
                    }
                ],
            }

    eng.kg = _FakeKG()
    manifest = IngestionManifest(content_type=ContentType.POLICY, source_uri="/ws")
    result = await eng.ingest(manifest)

    assert result.status == "success"
    assert seen == ["policy"]


def test_enrichment_windows_bounds_large_text(monkeypatch):
    """A whole book must not become one infeasible mega-prompt: large text is
    chunked into bounded windows, capped by KG_ENRICH_MAX_CHUNKS."""
    eng, _backend = _engine()

    # Small text → exactly one window (unchanged behavior).
    assert eng._enrichment_windows("short text") == ["short text"]
    assert eng._enrichment_windows("") == []

    # Large text → many bounded windows, capped at the limit.
    monkeypatch.setenv("KG_ENRICH_CHUNK_THRESHOLD", "500")
    monkeypatch.setenv("KG_ENRICH_MAX_CHUNKS", "5")
    big = ("Reinforcement learning maximizes expected return. " * 50 + "\n\n") * 40
    windows = eng._enrichment_windows(big)
    assert 1 < len(windows) <= 5  # chunked AND capped
    assert all(w.strip() for w in windows)
