"""Write-path trust/provenance guard tests (CONCEPT:KG-2.6).

Exercises BrainGuardedBackend directly with a fake inner backend so no daemon or
real store is needed. Verifies provenance attachment, source-authority
arbitration, trust decay flipping the winner, and that the default (unwrapped)
path is unchanged.
"""

from __future__ import annotations

import pytest

from agent_utilities.knowledge_graph.backends.brain_guarded_backend import (
    BrainGuardedBackend,
)
from agent_utilities.knowledge_graph.core.company_brain_runtime import (
    get_company_brain,
    reset_company_brain,
)
from agent_utilities.models.company_brain import ActorType
from agent_utilities.security.brain_context import (
    ActorContext,
    use_actor,
    use_source,
)


class FakeBackend:
    def __init__(self):
        self.nodes: dict[str, dict] = {}
        self.edges: list[tuple] = []

    def add_node(self, node_id, **props):
        self.nodes[node_id] = props

    def add_edge(self, s, t, **props):
        self.edges.append((s, t, props))

    # a non-write method to prove delegation works
    def execute(self, q, params=None):
        return [{"ok": True}]


class FakeMergeBackend:
    """Readable backend (replace-semantics) → guard uses field-level survivorship.

    Replace-on-add_node proves Option B robustness: the guard writes the full
    reconciled record, so even a non-merging backend keeps every field.
    """

    def __init__(self):
        self.nodes: dict[str, dict] = {}

    def add_node(self, node_id, **props):
        self.nodes[node_id] = dict(props)  # replace, not merge

    def add_edge(self, s, t, **props):
        pass

    def get_node_properties(self, node_id):
        n = self.nodes.get(node_id)
        return dict(n) if n is not None else None


@pytest.fixture(autouse=True)
def _fresh_brain():
    reset_company_brain()
    yield
    reset_company_brain()


def _guard():
    return BrainGuardedBackend(FakeBackend(), get_company_brain())


def test_provenance_attached_on_write():
    g = _guard()
    with (
        use_actor(ActorContext("agent:x", ActorType.AI_AGENT)),
        use_source("servicenow"),
    ):
        g.add_node("incident:1", type="Incident", number="INC1")
    props = g.inner.nodes["incident:1"]
    assert props["type"] == "Incident"
    assert props["_source_system"] == "servicenow"
    assert props["_actor_id"] == "agent:x"
    assert "_ts" in props and "_confidence" in props
    # provenance ledger recorded the write
    assert get_company_brain().provenance.get_provenance("incident:1")


def test_delegates_unknown_methods():
    g = _guard()
    assert g.execute("MATCH (n) RETURN n") == [{"ok": True}]


def test_source_authority_wins_suppresses_lower():
    g = _guard()
    # High-authority live system writes first.
    with use_source("servicenow"):
        g.add_node("incident:9", type="Incident", state="open", _src="sn")
    # Lower-authority document tries to overwrite the same node -> suppressed.
    with use_source("document"):
        g.add_node("incident:9", type="Incident", state="STALE", _src="doc")
    assert g.inner.nodes["incident:9"]["_source_system"] == "servicenow"
    assert g.inner.nodes["incident:9"]["state"] == "open"
    # A conflict was recorded.
    assert get_company_brain().conflicts.all_conflicts


def test_higher_authority_overwrites_lower():
    g = _guard()
    with use_source("document"):
        g.add_node("svc:1", type="Service", name="from-doc")
    with use_source("servicenow"):
        g.add_node("svc:1", type="Service", name="from-sn")
    # Higher authority (servicenow) wins.
    assert g.inner.nodes["svc:1"]["name"] == "from-sn"
    assert g.inner.nodes["svc:1"]["_source_system"] == "servicenow"


def test_same_source_rewrite_is_idempotent_update():
    g = _guard()
    with use_source("servicenow"):
        g.add_node("n:1", v="a")
        g.add_node("n:1", v="b")  # same source -> latest wins (no suppression)
    assert g.inner.nodes["n:1"]["v"] == "b"


def test_trust_decay_flips_winner(monkeypatch):
    """A stale high-authority source loses to a fresh lower-authority one."""
    g = _guard()
    brain = get_company_brain()
    # First write from servicenow, then make it look very old by decaying hard.
    with use_source("servicenow"):
        g.add_node("d:1", v="sn")

    # Force a large age so servicenow's decayed authority drops below 'document'.
    real_eff = brain.conflicts.effective_authority

    def fake_eff(source_system, age_days=0.0):
        # servicenow only counts as stale when compared (age_days > 0)
        if source_system == "servicenow" and age_days > 0:
            return 0.10
        return real_eff(source_system, 0.0)

    monkeypatch.setattr(brain.conflicts, "effective_authority", fake_eff)
    with use_source("document"):
        g.add_node("d:1", v="doc-fresh")
    # document (0.55 fresh) now beats decayed servicenow (0.10) -> overwrite wins
    assert g.inner.nodes["d:1"]["v"] == "doc-fresh"
    assert g.inner.nodes["d:1"]["_source_system"] == "document"


# ── Field-level survivorship (Option B) ───────────────────────────────────────
def _merge_guard():
    return BrainGuardedBackend(FakeMergeBackend(), get_company_brain())


def test_field_level_survivorship_merges_by_attribute():
    g = _merge_guard()
    # ServiceNow owns status + priority (authority 0.90).
    with use_source("servicenow"):
        g.add_node("incident:7", type="Incident", status="open", priority="P1")
    # A human adds business_impact (authority 0.98).
    with use_source("human_review"):
        g.add_node("incident:7", business_impact="revenue-critical")
    # A low-authority document tries to clobber status AND adds a new note.
    with use_source("document"):
        g.add_node("incident:7", status="stale", note="draft")

    node = g.inner.nodes["incident:7"]
    # High-authority field retained; low-authority overwrite rejected...
    assert node["status"] == "open"
    assert node["priority"] == "P1"
    assert node["business_impact"] == "revenue-critical"
    # ...but the low-authority source still contributed a brand-new attribute.
    assert node["note"] == "draft"


def test_field_level_per_attribute_provenance():
    g = _merge_guard()
    with use_source("servicenow"):
        g.add_node("incident:8", status="open")
    with use_source("human_review"):
        g.add_node("incident:8", business_impact="high")
    with use_source("document"):
        g.add_node("incident:8", note="n")

    prov = get_company_brain().provenance

    def owner_src(field):
        rec = prov.field_owner("incident:8", field)
        assert rec is not None
        return rec.source_system

    assert owner_src("status") == "servicenow"
    assert owner_src("business_impact") == "human_review"
    assert owner_src("note") == "document"


def test_field_level_contested_overwrite_logs_conflict():
    g = _merge_guard()
    with use_source("servicenow"):
        g.add_node("incident:9", status="open")
    with use_source("document"):
        g.add_node("incident:9", status="stale")  # loses, but is a real contest
    conflicts = get_company_brain().conflicts.all_conflicts
    assert any(c.field_name == "status" for c in conflicts)


def test_field_level_first_write_populates_ledger():
    g = _merge_guard()
    with use_source("document"):  # low authority writes FIRST
        g.add_node("svc:2", owner_field="doc")
    # servicenow (higher) can still overwrite a doc-owned field
    with use_source("servicenow"):
        g.add_node("svc:2", owner_field="sn")
    assert g.inner.nodes["svc:2"]["owner_field"] == "sn"


def test_field_provenance_persisted_on_node():
    g = _merge_guard()
    with use_source("servicenow"):
        g.add_node("incident:10", status="open")
    import json as _json

    fp = _json.loads(g.inner.nodes["incident:10"]["_field_prov"])
    assert fp["status"]["src"] == "servicenow"
    assert "ts" in fp["status"]


def test_field_level_survives_restart():
    """A fresh guard over the same backend recovers prior ownership from the node."""
    backend = FakeMergeBackend()
    brain = get_company_brain()
    # Process 1: ServiceNow establishes status.
    g1 = BrainGuardedBackend(backend, brain)
    with use_source("servicenow"):
        g1.add_node("incident:11", status="open", priority="P1")

    # "Restart": brand-new guard with an EMPTY in-memory ledger, same backend.
    g2 = BrainGuardedBackend(backend, brain)
    assert g2._field_owner == {}  # no in-memory state carried over
    with use_source("document"):  # low authority tries to clobber after restart
        g2.add_node("incident:11", status="stale", note="late")

    node = backend.nodes["incident:11"]
    # Durable _field_prov let the fresh guard keep ServiceNow's status...
    assert node["status"] == "open"
    assert node["priority"] == "P1"
    # ...while still accepting a genuinely new attribute.
    assert node["note"] == "late"
