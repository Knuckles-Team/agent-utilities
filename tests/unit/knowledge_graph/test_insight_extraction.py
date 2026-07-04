"""Insight/Fact/Framework/Playbook extraction tests (CONCEPT:EG-KG.storage.nonblocking-checkpoint).

A fake llm_fn returns canned JSON so extraction is deterministic and offline.
Verifies typed nodes + DERIVED_FROM edges, and that the pipeline persists them.
"""

from __future__ import annotations

import json

from agent_utilities.knowledge_graph.enrichment.extractors.document import (
    extract_intelligence,
)
from agent_utilities.knowledge_graph.enrichment.models import (
    Fact,
    Framework,
    Insight,
    Playbook,
)
from agent_utilities.knowledge_graph.enrichment.pipeline import EnrichmentPipeline

_INTEL_JSON = json.dumps(
    {
        "insights": [
            {"title": "Buyers fear lock-in", "reasoning": "raised in 3 of 5 calls"}
        ],
        "facts": [{"statement": "Refund window is 30 days"}],
        "frameworks": [
            {
                "name": "Objection Handling",
                "summary": "LAER",
                "steps": ["Listen", "Answer"],
            }
        ],
        "playbooks": [
            {
                "name": "Lock-in Rebuttal",
                "steps": ["Acknowledge", "Show exit path"],
                "preconditions": ["prospect raised lock-in"],
                "expected_outcome": "objection neutralized",
            }
        ],
    }
)


def _fake_llm(prompt: str) -> str:
    return _INTEL_JSON


def test_extract_intelligence_typed_nodes_and_edges():
    nodes, edges = extract_intelligence(
        "call transcript text", "doc:call1", _fake_llm, source_type="transcript"
    )
    by_type = {type(n).__name__: n for n in nodes}
    assert isinstance(by_type["Insight"], Insight)
    assert isinstance(by_type["Fact"], Fact)
    assert isinstance(by_type["Framework"], Framework)
    assert isinstance(by_type["Playbook"], Playbook)
    assert by_type["Playbook"].steps == ["Acknowledge", "Show exit path"]
    assert by_type["Insight"].source_ids == ["doc:call1"]
    # every node has a DERIVED_FROM edge back to the source
    assert all(e.rel_type == "DERIVED_FROM" and e.source == "doc:call1" for e in edges)
    assert len(edges) == len(nodes) == 4


def test_extract_intelligence_empty_text_is_safe():
    nodes, edges = extract_intelligence("", "doc:x", _fake_llm)
    assert nodes == [] and edges == []


def test_extract_intelligence_tolerates_bad_json():
    nodes, edges = extract_intelligence("text", "doc:x", lambda p: "not json")
    assert nodes == [] and edges == []


from tests.kg_recording_backend import RecordingGraphBackend as FakeBackend


def test_pipeline_persists_intelligence(tmp_path, monkeypatch):
    (tmp_path / "call.md").write_text("# Discovery call\nLots of objections.\n")
    backend = FakeBackend()

    # parse_fn is unused for documents; pass a dummy.
    pipe = EnrichmentPipeline(
        backend, parse_fn=lambda fp, src: {"nodes": []}, llm_fn=_fake_llm
    )
    _concepts, _edges, summary = pipe.enrich_documents([tmp_path / "call.md"])

    assert summary.intelligence_nodes == 4
    types = {p.get("type") for p in backend.nodes.values()}
    assert {"Insight", "Fact", "Framework", "Playbook"} <= types
    # playbook steps were JSON-serialized as a scalar property
    pb = next(p for p in backend.nodes.values() if p.get("type") == "Playbook")
    assert json.loads(pb["steps"]) == ["Acknowledge", "Show exit path"]
    assert any(rel == "DERIVED_FROM" for _, _, rel in backend.edges)
