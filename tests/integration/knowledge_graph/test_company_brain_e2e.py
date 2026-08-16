"""End-to-end Company Brain enforcement (CONCEPT:AU-KG.research.research-pipeline-runner / KG-2.8).

Exercises the layers composing through their public seams with
the mandatory shared runtime brain: write-path trust
arbitration, read-path permissions, the correction→rule→retrieval loop, and
operating-intelligence capture. Offline (memory backend / fakes).
"""

from __future__ import annotations

import pytest

from agent_utilities.knowledge_graph.adaptation.feedback import FeedbackService
from agent_utilities.knowledge_graph.backends import (
    GraphBackend,
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
    reset_company_brain()
    yield
    reset_company_brain()


def _create_operational_backend() -> GraphBackend:
    """Build the same backend ``create_backend()`` would, bound to THIS test's
    isolated graph instead of a tenant graph the test never provisioned.

    ``create_backend()``'s bare ``EpistemicGraphBackend()`` resolves its own
    routing graph via ``resolve_routing_graph(None)`` from the ambient actor's
    ``tenant_id`` *before* ``GraphComputeEngine`` is ever asked for one, then
    hands that already-resolved name to ``GraphComputeEngine.get_or_create()``.
    Under the suite-wide ``isolate_graph_compute_engine`` fixture two things
    go wrong under a tenant-bearing ambient actor (the fixture's default):

    1. The resolved name is a tenant graph this test never provisioned --
       bypassing the fixture's redirect, which only catches a literal
       ``graph_name in (None, "__commons__", "__secrets__")``.
    2. Even clearing the tenant so resolution lands on the "__commons__"
       sentinel doesn't fully fix it: ``get_or_create()`` compares its
       *pre-redirect* ``graph_name`` argument ("__commons__") against the
       *post-redirect* ``root.graph_name`` (this test's real isolated name),
       finds them unequal, and builds a graph-scoped view PINNED to the
       literal string "__commons__" -- which then fails every call with
       "A graph-scoped view cannot retarget the verified GraphSession"
       because the ambient GraphSession is scoped to the real isolated graph.

    Constructing ``GraphComputeEngine`` directly (as the isolate fixture's own
    engines do) sidesteps both: the redirect applies to this exact call, and
    there is no second, mismatched ``get_or_create()`` comparison afterward.
    """
    from agent_utilities.knowledge_graph.backends.epistemic_graph_backend import (
        EpistemicGraphBackend,
    )
    from agent_utilities.knowledge_graph.core.graph_compute import GraphComputeEngine

    compute = GraphComputeEngine(backend_type="rust")
    backend: GraphBackend = object.__new__(EpistemicGraphBackend)
    backend._graph = compute
    backend.graph_name = compute.graph_name
    backend.create_schema()

    from agent_utilities.knowledge_graph.core.company_brain_runtime import (
        brain_enforcement_enabled,
        get_company_brain,
    )

    if brain_enforcement_enabled():
        from agent_utilities.knowledge_graph.backends.brain_guarded_backend import (
            BrainGuardedBackend,
        )

        backend = BrainGuardedBackend(backend, get_company_brain())
    return backend


class FakeDesig:
    def __init__(self, id, score):
        self.id = id
        self.score = score
        self.capabilities = set()


def test_write_path_trust_arbitration_via_create_backend(enforced):
    # create_backend()'s explicit backend_type strings (e.g. "memory") are
    # reserved for focused backend tests and connection-registry adapters and
    # do NOT alter GraphOS authority -- only the omitted-type "operational
    # authority" path installs the Company Brain write-path guard.
    backend = _create_operational_backend()
    # WS-1 wiring: enforcement installs the guard.
    assert isinstance(backend, GraphBackend)
    assert type(backend).__name__ == "BrainGuardedBackend"

    with use_source("servicenow"):
        backend.add_node("incident:42", node_type="Incident", state="open")
    with use_source("document"):  # lower authority → suppressed
        backend.add_node("incident:42", node_type="Incident", state="stale")

    brain = get_company_brain()
    assert brain.conflicts.all_conflicts  # conflict was detected + logged
    assert brain.provenance.get_provenance("incident:42")  # provenance recorded


def test_read_path_permissions(enforced):
    brain = get_company_brain()
    brain.permissions.set_acl(
        NodeACL(
            node_id="hr:comp",
            classification=DataClassification.CONFIDENTIAL,
            read_roles=["hr"],
        )
    )
    # secured_reads.permit() now default-denies nodes with no ACL at all
    # ("Nodes without an ACL are denied" -- a fail-closed tightening this test
    # predates, when an absent ACL implicitly meant "public"). Give "pub:1" an
    # explicit PUBLIC-classification ACL instead of relying on the old
    # implicit-allow default.
    brain.permissions.set_acl(
        NodeACL(node_id="pub:1", classification=DataClassification.PUBLIC)
    )
    # secured_reads.permit() requires a verified tenant authority (actor_id +
    # tenant_id + authenticated=True) before it even reaches role-based ACL
    # filtering -- an unauthenticated caller-supplied identity is rejected
    # fail-closed regardless of roles. Construct properly authenticated test
    # actors (the same pattern used elsewhere, e.g.
    # tests/unit/knowledge_graph/test_engine_sharding.py) instead of the bare,
    # unauthenticated ActorContext this test predates.
    with use_actor(
        ActorContext(
            "a:mk",
            ActorType.AI_AGENT,
            roles=("marketing",),
            tenant_id="t:company-brain-e2e",
            authenticated=True,
        )
    ):
        assert sr.permit(["hr:comp", "pub:1"]) == ["pub:1"]
    with use_actor(
        ActorContext(
            "a:hr",
            ActorType.AI_AGENT,
            roles=("hr",),
            tenant_id="t:company-brain-e2e",
            authenticated=True,
        )
    ):
        assert set(sr.permit(["hr:comp", "pub:1"])) == {"hr:comp", "pub:1"}


def test_correction_becomes_rule_that_changes_retrieval(enforced):
    backend = _create_operational_backend()
    svc = FeedbackService(backend=backend)
    res = svc.record_correction(
        "rule",
        "tool:risky",
        reason="never auto-use",
        rule_scope="governance",
        rule_kind="forbid",
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

    def llm(p):
        return json.dumps(
            {
                "playbooks": [
                    {
                        "name": "Renewal Save",
                        "steps": ["call", "discount"],
                        "expected_outcome": "retained",
                    }
                ]
            }
        )

    nodes, edges = extract_intelligence(
        "transcript", "doc:c1", llm, source_type="transcript"
    )
    playbooks = [n for n in nodes if type(n).__name__ == "Playbook"]
    assert playbooks and playbooks[0].name == "Renewal Save"
    assert all(e.rel_type == "DERIVED_FROM" for e in edges)
