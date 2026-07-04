"""Capability write-back sink tests (CONCEPT:EG-KG.storage.nonblocking-checkpoint).

Fake Archi/LeanIX clients assert only provisional/derived capabilities are pushed,
existing names are skipped (idempotent), and one failing client never aborts the
batch. Now on the unified WritebackResult (pushes → ``created``, dedup → ``skipped``).
"""

from __future__ import annotations

from agent_utilities.knowledge_graph.enrichment.models import GraphNode
from agent_utilities.knowledge_graph.enrichment.writeback.sinks.capability import (
    make_writeback_fn,
    push_capabilities,
)


class FakeArchi:
    def __init__(self):
        self.added = []

    def add_element(self, type, name="", documentation="", properties=None):
        self.added.append((type, name))
        return f"id-{name}"


class FakeLeanIX:
    def __init__(self):
        self.created = []

    def postbusinesscapability(self, payload):
        self.created.append(payload["name"])


def _derived(cid, name):
    return GraphNode(
        id=cid,
        type="BusinessCapability",
        props={"name": name, "derived_from": "code", "provisional": True},
    )


def test_pushes_provisional_to_both_targets():
    archi, leanix = FakeArchi(), FakeLeanIX()
    result = push_capabilities(
        [_derived("capability:derived:billing", "Billing")],
        archi_client=archi,
        leanix_client=leanix,
    )
    assert result.created == 2  # one archi + one leanix
    assert archi.added == [("Capability", "Billing")]
    assert leanix.created == ["Billing"]


def test_skips_non_provisional_capabilities():
    archi = FakeArchi()
    upstream = GraphNode(
        id="capability:LEANIX", type="BusinessCapability", props={"name": "HR"}
    )
    result = push_capabilities([upstream], archi_client=archi)
    assert result.created == 0
    assert archi.added == []


def test_skips_existing_names_idempotent():
    archi = FakeArchi()
    result = push_capabilities(
        [_derived("capability:derived:billing", "Billing")],
        archi_client=archi,
        existing_names=["billing"],
    )
    assert result.skipped == 1
    assert result.created == 0


def test_dry_run_proposes_without_pushing():
    archi = FakeArchi()
    result = push_capabilities(
        [_derived("capability:derived:billing", "Billing")],
        archi_client=archi,
        dry_run=True,
    )
    assert result.created == 0
    assert archi.added == []
    assert result.proposals and result.proposals[0]["op"] == "create_capability"


def test_one_failing_client_does_not_abort():
    class Boom:
        def add_element(self, **_):
            raise RuntimeError("down")

    leanix = FakeLeanIX()
    result = push_capabilities(
        [_derived("capability:derived:x", "X")],
        archi_client=Boom(),
        leanix_client=leanix,
    )
    assert result.errors == 1
    assert result.created == 1  # leanix still succeeded


def test_make_writeback_fn_returns_callable():
    archi = FakeArchi()
    fn = make_writeback_fn(archi_client=archi)
    result = fn([_derived("capability:derived:y", "Y")])
    assert result.created == 1


def test_no_clients_is_noop():
    result = push_capabilities([_derived("capability:derived:z", "Z")])
    assert result.created == 0 and result.errors == 0 and not result.proposals
