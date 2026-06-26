"""prod-trace → dataset → prompt-version closed loop (CONCEPT:AHE-3.68)."""

from __future__ import annotations

from agent_utilities.harness.eval_corpus import EvalCorpus
from agent_utilities.models.knowledge_graph import TraceNode
from agent_utilities.prompting.structured import StructuredPrompt


class _FakeKG:
    def __init__(self):
        self.nodes = {}

    def add_node(self, node_id, **props):
        self.nodes[node_id] = props


def test_add_from_trace_promotes_to_dataset_and_case():
    kg = _FakeKG()
    corpus = EvalCorpus(backend=kg)
    trace = TraceNode(id="t1", name="run", input="2+2?", output="4")
    cid = corpus.add_from_trace(trace, assertion="answer is 4", tags=["math"])
    # an eval case is in memory with provenance
    case = next(c for c in corpus._mem if c["id"] == cid)
    assert case["metadata"]["source_trace_id"] == "t1"
    assert case["assertion"] == "answer is 4"
    # and a DatasetItemNode(source=trace) was persisted
    di = [p for p in kg.nodes.values() if p.get("type") == "dataset_item"]
    assert di and di[0]["source"] == "trace" and di[0]["source_trace_id"] == "t1"


def test_prompt_version_is_content_addressed_and_stable():
    p1 = StructuredPrompt(task="agent_x", input="You are helpful.")
    p2 = StructuredPrompt(task="agent_x", input="You are helpful.")
    p3 = StructuredPrompt(task="agent_x", input="You are VERY helpful.")
    assert p1.version_hash() == p2.version_hash()  # same content → same version
    assert p1.version_hash() != p3.version_hash()  # changed content → new version


def test_prompt_version_persists_node():
    kg = _FakeKG()
    p = StructuredPrompt(task="agent_x", input="You are helpful.")
    node = p.version("agent_x", backend=kg, parent_hash="deadbeef")
    assert node.prompt_id == "agent_x" and node.parent_hash == "deadbeef"
    pv = [n for n in kg.nodes.values() if n.get("type") == "prompt_version"]
    assert pv and pv[0]["version_hash"] == node.version_hash
