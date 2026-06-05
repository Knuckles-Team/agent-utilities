"""Capability write-back tests (CONCEPT:KG-2.8).

Fake Archi/LeanIX clients assert that only provisional/derived capabilities are
pushed, that existing names are skipped (idempotent), and that one failing client
never aborts the batch.
"""

from __future__ import annotations

from agent_utilities.knowledge_graph.enrichment.capability_writeback import (
    make_writeback_fn,
    push_capabilities,
)
from agent_utilities.knowledge_graph.enrichment.models import GraphNode


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
    nodes = [_derived("capability:derived:billing", "Billing")]
    result = push_capabilities(nodes, archi_client=archi, leanix_client=leanix)

    assert result.archi_pushed == 1
    assert result.leanix_pushed == 1
    assert archi.added == [("Capability", "Billing")]
    assert leanix.created == ["Billing"]


def test_skips_non_provisional_capabilities():
    archi = FakeArchi()
    # A mirrored upstream capability (no provisional/derived flag) is not pushed.
    upstream = GraphNode(
        id="capability:LEANIX", type="BusinessCapability", props={"name": "HR"}
    )
    result = push_capabilities([upstream], archi_client=archi)
    assert result.archi_pushed == 0
    assert archi.added == []


def test_skips_existing_names_idempotent():
    archi = FakeArchi()
    nodes = [_derived("capability:derived:billing", "Billing")]
    result = push_capabilities(
        nodes, archi_client=archi, existing_names=["billing"]
    )
    assert result.skipped_existing == 1
    assert result.archi_pushed == 0


def test_one_failing_client_does_not_abort():
    class Boom:
        def add_element(self, **_):
            raise RuntimeError("down")

    leanix = FakeLeanIX()
    nodes = [_derived("capability:derived:x", "X")]
    result = push_capabilities(nodes, archi_client=Boom(), leanix_client=leanix)
    assert result.errors == 1
    assert result.leanix_pushed == 1  # leanix still succeeded


def test_make_writeback_fn_returns_callable():
    archi = FakeArchi()
    fn = make_writeback_fn(archi_client=archi)
    result = fn([_derived("capability:derived:y", "Y")])
    assert result.archi_pushed == 1


def test_no_clients_is_noop():
    result = push_capabilities([_derived("capability:derived:z", "Z")])
    assert result.as_dict() == {
        "archi_pushed": 0,
        "leanix_pushed": 0,
        "skipped_existing": 0,
        "errors": 0,
    }
