"""Tests for the GOC-61 admin ownership-claim capability
(CONCEPT: GOC-61-admin-claim, ``decisions/GOC-61-unowned-node-disposition.md``).

Covers:
- enumeration never depends on a Cypher NULL predicate (engine-side label index only)
- never reassigns an already-owned node, including a TOCTOU race between
  enumeration and the per-node write
- real counts (never an unconditional success string), and idempotency
- kg:admin is required, fail-closed
- owner_id is always required and explicit, never defaults to the caller
- private-class conversational types ARE claimable (2026-08-09 revised ruling),
  to an explicit owner
- 'Concept' cannot be swept by node_type (dual origin) but can be claimed by
  explicit node_ids
- selection is never blind (node_types/node_ids required)
- an audit record is written for both preview and apply
"""

from __future__ import annotations

import pytest

from agent_utilities.knowledge_graph.core import ownership_claim as oc
from agent_utilities.models.company_brain import ActorType
from agent_utilities.security.brain_context import ActorContext


def _admin(actor_id="root", tenant="acme"):
    return ActorContext(
        actor_id=actor_id,
        actor_type=ActorType.HUMAN,
        roles=("kg:admin",),
        tenant_id=tenant,
        authenticated=True,
    )


def _user(actor_id="alice", tenant="acme"):
    return ActorContext(
        actor_id=actor_id,
        actor_type=ActorType.HUMAN,
        roles=(),
        tenant_id=tenant,
        authenticated=True,
    )


class _FakeBackend:
    """In-memory backend: nodes_by_label + get_node_properties + execute(SET)."""

    def __init__(self, nodes: dict[str, dict]):
        # node_id -> properties dict (mutated in place by "execute" SET calls)
        self.nodes = nodes
        self.executed: list[tuple[str, dict]] = []

    def nodes_by_label(self, label: str, limit: int = 0):
        rows = [
            (nid, dict(props))
            for nid, props in self.nodes.items()
            if props.get("node_type") == label
        ]
        return rows if limit == 0 else rows[:limit]

    def get_node_properties(self, node_id: str):
        if node_id not in self.nodes:
            return None
        return dict(self.nodes[node_id])

    def execute(self, cypher: str, params: dict):
        self.executed.append((cypher, dict(params)))
        # Only the claim SET statement is exercised by this module.
        node_id = params.get("id")
        if node_id in self.nodes:
            self.nodes[node_id][oc.OWNER_KEY] = params.get("owner")
            self.nodes[node_id][oc.SCOPE_KEY] = params.get("scope")
        return []


class _FakeEngine:
    """Minimal IntelligenceGraphEngine-shaped stand-in: .backend + add_node/add_edge."""

    def __init__(self, nodes: dict[str, dict]):
        self.backend = _FakeBackend(nodes)
        self.audit_nodes: dict[str, dict] = {}
        self.audit_edges: list[tuple[str, str, str]] = []

    def add_node(self, node_id, node_type, properties=None, **_kw):
        self.audit_nodes[node_id] = {"node_type": node_type, **(properties or {})}
        return {"id": node_id}

    def add_edge(self, source, target, rel_type="", **_kw):
        self.audit_edges.append((source, target, rel_type))


def _engine(nodes: dict[str, dict]) -> _FakeEngine:
    return _FakeEngine(nodes)


# --- admin gating -----------------------------------------------------------


def test_preview_requires_kg_admin():
    eng = _engine({"w:1": {"node_type": "WorkItem"}})
    with pytest.raises(oc.OwnershipClaimError):
        oc.preview_claim(
            eng, owner_id="svc-system", node_types=["WorkItem"], actor=_user()
        )


def test_apply_requires_kg_admin():
    eng = _engine({"w:1": {"node_type": "WorkItem"}})
    with pytest.raises(oc.OwnershipClaimError):
        oc.apply_claim(
            eng, owner_id="svc-system", node_types=["WorkItem"], actor=_user()
        )


# --- selection must never be blind ------------------------------------------


def test_apply_refuses_with_no_selection():
    eng = _engine({})
    with pytest.raises(oc.OwnershipClaimError):
        oc.apply_claim(eng, owner_id="svc-system", actor=_admin())


def test_preview_refuses_with_no_selection():
    eng = _engine({})
    with pytest.raises(oc.OwnershipClaimError):
        oc.preview_claim(eng, owner_id="svc-system", actor=_admin())


# --- owner_id is always required and explicit -------------------------------


def test_apply_refuses_empty_owner_id():
    eng = _engine({"w:1": {"node_type": "WorkItem"}})
    with pytest.raises(oc.OwnershipClaimError):
        oc.apply_claim(eng, owner_id="", node_types=["WorkItem"], actor=_admin("root"))


def test_apply_never_defaults_owner_to_calling_actor():
    """KNOWN-BAD-INPUT proof: omitting owner_id must refuse, not silently
    stamp the calling admin's own id (the exact 'silent self-assignment /
    takeover shape' this module is required to prevent)."""
    eng = _engine({"w:1": {"node_type": "WorkItem"}})
    with pytest.raises(oc.OwnershipClaimError):
        oc.apply_claim(
            eng, owner_id=None, node_types=["WorkItem"], actor=_admin("root")
        )
    # Nothing was mutated by the refused call.
    assert oc.OWNER_KEY not in eng.backend.nodes["w:1"]


# --- never reassigns an already-owned node (the most important proof) ------


def test_apply_never_reassigns_an_already_owned_node():
    """KNOWN-BAD-INPUT proof: a node that already has an owner must be
    skipped, never overwritten, even though it matches the node_type filter."""
    eng = _engine(
        {
            "w:1": {"node_type": "WorkItem"},  # unowned -> claimable
            "w:2": {  # already owned -> must be refused/skipped
                "node_type": "WorkItem",
                oc.OWNER_KEY: "someone-else",
                oc.SCOPE_KEY: "private",
            },
        }
    )
    result = oc.apply_claim(
        eng, owner_id="svc-system", node_types=["WorkItem"], actor=_admin()
    )
    assert result.claimed_node_ids == ["w:1"]
    assert result.claimed_total == 1
    assert result.skipped_already_owned == 1
    # The pre-existing owner is untouched.
    assert eng.backend.nodes["w:2"][oc.OWNER_KEY] == "someone-else"


def test_apply_skips_node_that_becomes_owned_between_enumeration_and_write():
    """KNOWN-BAD-INPUT proof of the TOCTOU defense: enumeration sees the node
    as unowned, but by the time the per-node re-check runs (immediately before
    the write) it has been claimed by someone else. The write must be skipped,
    not raced/overwritten."""
    eng = _engine({"w:1": {"node_type": "WorkItem"}})
    real_lookup = eng.backend.get_node_properties
    calls = {"n": 0}

    def racy_lookup(node_id):
        # Enumeration (nodes_by_label) already ran and saw "w:1" as unowned --
        # this is the FIRST call to get_node_properties, which only happens
        # for the pre-write re-check. Simulate a concurrent claim landing in
        # the window between enumeration and this re-check.
        calls["n"] += 1
        eng.backend.nodes[node_id][oc.OWNER_KEY] = "concurrent-claimant"
        eng.backend.nodes[node_id][oc.SCOPE_KEY] = "private"
        return real_lookup(node_id)

    eng.backend.get_node_properties = racy_lookup

    result = oc.apply_claim(
        eng, owner_id="svc-system", node_types=["WorkItem"], actor=_admin()
    )
    assert result.claimed_total == 0
    assert result.skipped_already_owned == 1
    assert eng.backend.nodes["w:1"][oc.OWNER_KEY] == "concurrent-claimant"


def test_apply_explicit_node_ids_skips_already_owned():
    eng = _engine(
        {
            "m:1": {
                "node_type": "Message",
                oc.OWNER_KEY: "bob",
                oc.SCOPE_KEY: "private",
            }
        }
    )
    result = oc.apply_claim(eng, owner_id="alice", node_ids=["m:1"], actor=_admin())
    assert result.claimed_total == 0
    assert result.skipped_already_owned == 1
    assert eng.backend.nodes["m:1"][oc.OWNER_KEY] == "bob"


# --- real counts, never an unconditional success string ---------------------


def test_apply_reports_zero_when_nothing_matches():
    eng = _engine({})
    result = oc.apply_claim(
        eng, owner_id="svc-system", node_types=["WorkItem"], actor=_admin()
    )
    assert result.claimed_total == 0
    assert result.claimed_node_ids == []


# --- idempotency --------------------------------------------------------


def test_apply_is_idempotent():
    eng = _engine({"w:1": {"node_type": "WorkItem"}, "w:2": {"node_type": "WorkItem"}})
    first = oc.apply_claim(
        eng, owner_id="svc-system", node_types=["WorkItem"], actor=_admin()
    )
    assert first.claimed_total == 2
    second = oc.apply_claim(
        eng, owner_id="svc-system", node_types=["WorkItem"], actor=_admin()
    )
    assert second.claimed_total == 0
    assert second.skipped_already_owned == 2
    assert eng.backend.nodes["w:1"][oc.OWNER_KEY] == "svc-system"
    assert eng.backend.nodes["w:2"][oc.OWNER_KEY] == "svc-system"


# --- private-class conversational content IS claimable (revised ruling) ----


def test_apply_claims_private_conversational_type_to_explicit_human_owner():
    eng = _engine(
        {
            "t:1": {"node_type": "Thread"},
            "msg:1": {"node_type": "Message"},
        }
    )
    result = oc.apply_claim(
        eng,
        owner_id="the-real-human-user",
        node_types=["Thread", "Message"],
        actor=_admin(),
    )
    assert result.claimed_total == 2
    assert set(result.claimed_node_ids) == {"t:1", "msg:1"}
    assert eng.backend.nodes["t:1"][oc.OWNER_KEY] == "the-real-human-user"
    assert eng.backend.nodes["msg:1"][oc.OWNER_KEY] == "the-real-human-user"
    assert result.claimed_classification_by_type["Thread"] == "private_conversational"


def test_preview_reports_classification_and_owner_to_be():
    eng = _engine({"msg:1": {"node_type": "Message"}})
    preview = oc.preview_claim(
        eng, owner_id="the-real-human-user", node_types=["Message"], actor=_admin()
    )
    assert preview.owner_id == "the-real-human-user"
    assert preview.would_claim_by_type == {"Message": 1}
    assert (
        preview.would_claim_classification_by_type["Message"]
        == "private_conversational"
    )
    # Preview never mutates.
    assert oc.OWNER_KEY not in eng.backend.nodes["msg:1"]


# --- Concept dual-origin exclusion ------------------------------------------


def test_apply_refuses_concept_sweep_by_node_type():
    eng = _engine({"c:1": {"node_type": "Concept"}})
    with pytest.raises(oc.OwnershipClaimError):
        oc.apply_claim(
            eng, owner_id="svc-system", node_types=["Concept"], actor=_admin()
        )
    assert oc.OWNER_KEY not in eng.backend.nodes["c:1"]


def test_apply_allows_concept_via_explicit_node_ids():
    eng = _engine({"c:1": {"node_type": "Concept"}})
    result = oc.apply_claim(
        eng, owner_id="the-real-human-user", node_ids=["c:1"], actor=_admin()
    )
    assert result.claimed_total == 1
    assert eng.backend.nodes["c:1"][oc.OWNER_KEY] == "the-real-human-user"


def test_preview_reports_concept_type_sweep_as_excluded_not_silently_dropped():
    eng = _engine({"c:1": {"node_type": "Concept"}, "w:1": {"node_type": "WorkItem"}})
    preview = oc.preview_claim(
        eng, owner_id="svc-system", node_types=["Concept", "WorkItem"], actor=_admin()
    )
    assert preview.excluded_dual_origin_types == ["Concept"]
    assert preview.would_claim_by_type == {"WorkItem": 1}


# --- enumeration never depends on a Cypher NULL predicate -------------------


def test_enumeration_uses_label_index_never_execute_with_is_null():
    eng = _engine({"w:1": {"node_type": "WorkItem"}})

    def _forbidden_execute(cypher, params):
        assert "IS NULL" not in cypher.upper()
        return eng.backend.__class__.execute(eng.backend, cypher, params)

    eng.backend.execute = _forbidden_execute
    result = oc.apply_claim(
        eng, owner_id="svc-system", node_types=["WorkItem"], actor=_admin()
    )
    assert result.claimed_total == 1


def test_no_label_index_accessor_fails_loudly_not_silently():
    class _NoLabelIndexEngine:
        backend = object()  # no nodes_by_label / get_nodes_by_label at all

        def add_node(self, *a, **kw):
            return None

    with pytest.raises(oc.OwnershipClaimError):
        oc.apply_claim(
            _NoLabelIndexEngine(),
            owner_id="svc-system",
            node_types=["WorkItem"],
            actor=_admin(),
        )


# --- audit trail -------------------------------------------------------


def test_apply_writes_an_audit_record_with_real_counts():
    eng = _engine({"w:1": {"node_type": "WorkItem"}})
    oc.apply_claim(
        eng, owner_id="svc-system", node_types=["WorkItem"], actor=_admin("root")
    )
    assert len(eng.audit_nodes) == 1
    audit = next(iter(eng.audit_nodes.values()))
    assert audit["node_type"] == oc.AUDIT_NODE_LABEL
    assert audit["actor_id"] == "root"
    assert audit["owner_id"] == "svc-system"
    assert audit["claimed_total"] == 1
    assert audit["dry_run"] is False
    assert len(eng.audit_edges) == 1


def test_preview_writes_an_audit_record_marked_dry_run():
    eng = _engine({"w:1": {"node_type": "WorkItem"}})
    oc.preview_claim(
        eng, owner_id="svc-system", node_types=["WorkItem"], actor=_admin("root")
    )
    assert len(eng.audit_nodes) == 1
    audit = next(iter(eng.audit_nodes.values()))
    assert audit["dry_run"] is True
    assert audit["claimed_total"] == 0
    # Preview never mutates or creates claim edges.
    assert eng.audit_edges == []


def test_invalid_shared_scope_is_rejected():
    eng = _engine({"w:1": {"node_type": "WorkItem"}})
    with pytest.raises(oc.OwnershipClaimError):
        oc.apply_claim(
            eng,
            owner_id="svc-system",
            node_types=["WorkItem"],
            shared_scope="not-a-real-scope",
            actor=_admin(),
        )
