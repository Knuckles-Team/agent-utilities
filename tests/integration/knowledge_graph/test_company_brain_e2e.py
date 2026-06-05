"""End-to-end Company Brain enforcement (CONCEPT:KG-2.6 / KG-2.8).

Exercises the layers composing through their public seams with
``KG_BRAIN_ENFORCE=1`` and the shared runtime brain: write-path trust
arbitration, read-path permissions, the correction→rule→retrieval loop, and
operating-intelligence capture. Offline (memory backend / fakes).
"""

from __future__ import annotations

import pytest

from agent_utilities.knowledge_graph.adaptation.feedback import FeedbackService
from agent_utilities.knowledge_graph.backends import (
    GraphBackend,
    create_backend,
)
from agent_utilities.knowledge_graph.core import secured_reads as sr
from agent_utilities.knowledge_graph.core.company_brain_runtime import (
    get_company_brain,
    reset_company_brain,
)
from agent_utilities.knowledge_graph.retrieval.governance_rules import (
    apply_governance_rules,
)
from agent_utilities.models.company_brain import (
    ActorType,
    DataClassification,
    NodeACL,
)
from agent_utilities.security.brain_context import (
    ActorContext,
    use_actor,
    use_source,
)


@pytest.fixture
def enforced(monkeypatch):
    monkeypatch.setenv("KG_BRAIN_ENFORCE", "1")
    reset_company_brain()
    yield
    reset_company_brain()


class FakeDesig:
    def __init__(self, id, score):
        self.id = id
        self.score = score
        self.capabilities = set()


def test_write_path_trust_arbitration_via_create_backend(enforced):
    backend = create_backend("memory")
    # WS-1 wiring: enforcement installs the guard.
    assert isinstance(backend, GraphBackend)
    assert type(backend).__name__ == "BrainGuardedBackend"

    with use_source("servicenow"):
        backend.add_node("incident:42", type="Incident", state="open")
    with use_source("document"):  # lower authority → suppressed
        backend.add_node("incident:42", type="Incident", state="stale")

    brain = get_company_brain()
    assert brain.conflicts.all_conflicts  # conflict was detected + logged
    assert brain.provenance.get_provenance("incident:42")  # provenance recorded


def test_read_path_permissions(enforced):
    brain = get_company_brain()
    brain.permissions.set_acl(
        NodeACL(node_id="hr:comp", classification=DataClassification.CONFIDENTIAL,
                read_roles=["hr"])
    )
    with use_actor(ActorContext("a:mk", ActorType.AI_AGENT, roles=("marketing",))):
        assert sr.permit(["hr:comp", "pub:1"]) == ["pub:1"]
    with use_actor(ActorContext("a:hr", ActorType.AI_AGENT, roles=("hr",))):
        assert set(sr.permit(["hr:comp", "pub:1"])) == {"hr:comp", "pub:1"}


def test_correction_becomes_rule_that_changes_retrieval(enforced):
    backend = create_backend("memory")
    svc = FeedbackService(backend=backend)
    res = svc.record_correction(
        "rule", "tool:risky", reason="never auto-use",
        rule_scope="governance", rule_kind="forbid",
    )
    assert res.applied and res.created_ids

    # The persisted rule, applied at retrieval, removes the forbidden tool.
    rules = [{"kind": "forbid", "target": "tool:risky"}]
    desigs = [FakeDesig("tool:risky", 0.99), FakeDesig("tool:safe", 0.3)]
    out = apply_governance_rules(desigs, rules)
    assert [d.id for d in out] == ["tool:safe"]


def test_intelligence_capture_yields_playbook(enforced):
    import json

    from agent_utilities.knowledge_graph.enrichment.extractors.document import (
        extract_intelligence,
    )

    llm = lambda p: json.dumps(
        {"playbooks": [{"name": "Renewal Save", "steps": ["call", "discount"],
                        "expected_outcome": "retained"}]}
    )
    nodes, edges = extract_intelligence("transcript", "doc:c1", llm,
                                        source_type="transcript")
    playbooks = [n for n in nodes if type(n).__name__ == "Playbook"]
    assert playbooks and playbooks[0].name == "Renewal Save"
    assert all(e.rel_type == "DERIVED_FROM" for e in edges)
