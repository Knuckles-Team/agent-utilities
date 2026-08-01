"""Governed runtime capability authoring (CONCEPT:AU-AHE.harness.canonical-gap-lifecycle × harness
runtime authoring, track 6 of the pydantic-ai-native-adoption program).

Proves an agent-authored capability enters the SAME Gap->SDD->promote review any
other change to this codebase would, rather than the harness's own
``CapabilityStore.load_active()`` default of activating anything that merely passed
static validation.
"""

from __future__ import annotations

import re

import pytest

from agent_utilities.capabilities.governed_capability_authoring import (
    load_governed_active_capabilities,
    submit_authored_capability_for_review,
)
from agent_utilities.knowledge_graph.research import gaps
from agent_utilities.knowledge_graph.research.gaps import get_gap
from agent_utilities.knowledge_graph.research.spec_proposals import (
    get_spec,
    review_spec,
)
from tests.unit.fleet_autonomy_fakes import FakeEngine

pytestmark = pytest.mark.concept("AU-AHE.harness.canonical-gap-lifecycle")

_VALID_CAPABILITY_SOURCE = '''
from pydantic_ai.capabilities import AbstractCapability


class Loud(AbstractCapability):
    def get_instructions(self):
        return "Be extra enthusiastic."
'''

_INVALID_CAPABILITY_SOURCE = "this is not even python syntax :::"


class LifecycleEngine(FakeEngine):
    """Same in-memory double as ``tests/test_wave6_gap_lifecycle.py`` — the
    backend-agnostic Cypher shapes ``get_gap``/``get_spec``/``review_spec`` use."""

    def add_edge(self, src, dst, rel_type, properties=None):
        self.edges.append((src, dst, rel_type))

    def query_cypher(self, query, params=None):
        params = params or {}
        if "WHERE n.id = $id" in query:
            node = self.nodes.get(params.get("id"))
            return [{"n": dict(node)}] if node else []
        m = re.search(r"MATCH \(n:(\w+)\) RETURN n", query)
        if m:
            lbl = m.group(1)
            return [{"n": dict(v)} for v in self.nodes.values() if v.get("type") == lbl]
        if "[:RESOLVES]->" in query:
            lid = params.get("id")
            return [
                {"id": dst}
                for (s, dst, r) in self.edges
                if s == lid
                and r == "RESOLVES"
                and self.nodes.get(dst, {}).get("type") == gaps.GAP_LABEL
            ]
        return super().query_cypher(query, params)


def test_authoring_a_valid_capability_holds_it_and_opens_a_reviewable_gap(tmp_path):
    eng = LifecycleEngine()
    result = submit_authored_capability_for_review(
        eng,
        directory=tmp_path,
        name="loud",
        code=_VALID_CAPABILITY_SOURCE,
        reason="the user asked for more enthusiasm",
    )

    assert result["written"] is True
    assert result["valid"] is True
    assert result["status"] == "pending_review"
    assert result["gap_id"] == gaps.canonical_gap_id(
        "agent_authored_capability", "loud"
    )
    assert result["spec_id"]

    # The Gap and SpecProposal are real, queryable KG nodes...
    gap = get_gap(eng, result["gap_id"])
    assert gap["status"] == gaps.STATUS_SPECIFIED  # a spec is now in flight
    spec = get_spec(eng, result["spec_id"])
    assert spec["status"] == "pending_review"

    # ...and — the load-bearing assertion — the capability is NOT active on disk,
    # despite passing the harness's own static validation. This is the loophole
    # this module closes: the harness's own store.write() would otherwise leave it
    # 'active' for the next naive store.load_active() call.
    from pydantic_ai_harness.capability_creation._store import CapabilityStore

    store = CapabilityStore(directory=tmp_path)
    (record,) = store.list_all()
    assert record.status == "disabled"


def test_invalid_capability_source_never_opens_a_gap(tmp_path):
    eng = LifecycleEngine()
    result = submit_authored_capability_for_review(
        eng, directory=tmp_path, name="broken", code=_INVALID_CAPABILITY_SOURCE
    )

    assert result["written"] is True
    assert result["valid"] is False
    assert result["status"] == "validation_failed"
    assert "gap_id" not in result
    # Nothing entered the review lifecycle for code that fails even static validation.
    assert not [n for n in eng.nodes.values() if n.get("type") == gaps.GAP_LABEL]


def test_load_governed_active_capabilities_ignores_an_unreviewed_authored_capability(
    tmp_path,
):
    eng = LifecycleEngine()
    submit_authored_capability_for_review(
        eng, directory=tmp_path, name="loud", code=_VALID_CAPABILITY_SOURCE
    )

    # No review happened yet -> the governed loader returns NOTHING, even though
    # the harness's own (ungoverned) store.load_active() would already see it as
    # written+validated were it not for this module holding it back.
    active = load_governed_active_capabilities(eng, tmp_path)
    assert active == []


def test_load_governed_active_capabilities_activates_on_spec_approval(tmp_path):
    eng = LifecycleEngine()
    result = submit_authored_capability_for_review(
        eng, directory=tmp_path, name="loud", code=_VALID_CAPABILITY_SOURCE
    )

    # Still held pre-approval.
    assert load_governed_active_capabilities(eng, tmp_path) == []

    # The SAME review gate a human-authored spec goes through.
    review_spec(eng, result["spec_id"], "approve")

    # NOW the governed loader activates it — constructed via the harness's own
    # loader, so it is a real AbstractCapability instance, not a stub.
    active = load_governed_active_capabilities(eng, tmp_path)
    assert len(active) == 1
    assert type(active[0]).__name__ == "Loud"

    from pydantic_ai_harness.capability_creation._store import CapabilityStore

    store = CapabilityStore(directory=tmp_path)
    (record,) = store.list_all()
    assert record.status == "active"


def test_load_governed_active_capabilities_holds_a_rejected_one_forever(tmp_path):
    eng = LifecycleEngine()
    result = submit_authored_capability_for_review(
        eng, directory=tmp_path, name="loud", code=_VALID_CAPABILITY_SOURCE
    )

    review_spec(eng, result["spec_id"], "reject")

    assert load_governed_active_capabilities(eng, tmp_path) == []
    from pydantic_ai_harness.capability_creation._store import CapabilityStore

    store = CapabilityStore(directory=tmp_path)
    (record,) = store.list_all()
    assert record.status == "disabled"


def test_invalid_capability_name_fails_before_writing_anything(tmp_path):
    eng = LifecycleEngine()
    result = submit_authored_capability_for_review(
        eng, directory=tmp_path, name="Not-A-Valid-Name", code=_VALID_CAPABILITY_SOURCE
    )
    assert result["written"] is False
    assert "error" in result
    assert not list(tmp_path.iterdir())
