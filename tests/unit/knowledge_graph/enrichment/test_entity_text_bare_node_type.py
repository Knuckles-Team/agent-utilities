"""EH-328: a node carrying only a ``node_type`` embeds its type once.

``derive_entity_text``'s fallback seeded its parts with the resolved type and
then appended the same value again from the ``type``/``node_type`` key, so a
bare ``Order`` node embedded as ``"Order — Order"``. 82eff59f0 added both keys
to ``_ENTITY_TEXT_SKIP_KEYS``; this pins that behaviour.
"""

from __future__ import annotations

import pytest

from agent_utilities.knowledge_graph.enrichment.semantic import derive_entity_text


@pytest.mark.parametrize(
    "props",
    [
        {"type": "Order"},
        {"node_type": "Order"},
        {"type": "Order", "node_type": "Order"},
    ],
)
def test_bare_node_type_embeds_its_type_once(props) -> None:
    assert derive_entity_text(props) == "Order"


def test_fallback_keeps_other_leaves_after_the_type() -> None:
    assert (
        derive_entity_text({"node_type": "Order", "region": "EMEA"}) == "Order — EMEA"
    )


def test_named_entity_is_not_prefixed_with_its_type() -> None:
    assert derive_entity_text({"node_type": "Order", "name": "PO-7"}) == "PO-7"


@pytest.mark.spec("AU-RETIRE-R004")
def test_node_with_only_node_type_embeds_single_label_not_doubled() -> None:
    """AU-RETIRE-R004 acceptance: a node carrying only node_type (no other
    descriptive fields) composes a single label such as "Order", not a
    doubled compound string such as "Order and Order" / "Order — Order"."""
    text = derive_entity_text({"node_type": "Order"})
    assert text == "Order"
    assert text.count("Order") == 1
