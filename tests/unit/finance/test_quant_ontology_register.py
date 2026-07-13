"""``QuantOntology.register`` malformed-``properties`` dead-code hardening
(found alongside the ``add_node(id=...)`` kwarg-drift bug in
kg-exhaustive-batchD.md — see ``test_quant_add_node_kwarg_fix.py``).

Every ``engine.add_node(...)`` call in ``QuantOntology.register`` used to pass
``properties=[...]`` — a bare list of field-name strings — where
``IntelligenceGraphEngine.add_node`` requires ``properties: dict[str, Any] |
None`` (it unconditionally does ``props["type"] = node_type`` on whatever is
passed, which raises ``TypeError: list indices must be integers or slices,
not str`` against a list). Nothing in the repo currently calls
``QuantOntology.register`` (confirmed dead code), so this never reproduced
live, but it is fixed defensively — nested under a ``fields`` key — so it
doesn't crash the moment something wires it up.

Uses the same strict, real-signature fake engine as
``test_quant_add_node_kwarg_fix.py`` so a regression back to a bare list
raises exactly like the real engine would.
"""

from __future__ import annotations

from agent_utilities.domains.finance.quant_ontology import QuantOntology


class _StrictRealSignatureEngine:
    """Mirrors ``IntelligenceGraphEngine.add_node``/``add_edge`` exactly —
    ``properties`` must be a dict (or None); passing a list raises, exactly
    like the real engine's ``props["type"] = node_type`` would.
    """

    def __init__(self) -> None:
        self.nodes: list[dict] = []
        self.edges: list[tuple[str, str, str, str]] = []

    def add_node(
        self, node_id, node_type, properties=None, ephemeral=False, *, session=None
    ):
        props = properties if properties is not None else {}
        if not isinstance(props, dict):
            raise TypeError(
                "list indices must be integers or slices, not str"
            )  # mirrors the real engine's props["type"] = node_type crash
        props = dict(props)
        props["type"] = node_type
        self.nodes.append(
            {"node_id": node_id, "node_type": node_type, "properties": props}
        )

    def add_edge(self, source, target, rel_type="", *, relation_type="", **kwargs):
        self.edges.append((source, target, rel_type, relation_type))


def test_register_never_passes_a_bare_list_as_properties():
    engine = _StrictRealSignatureEngine()

    QuantOntology.register(engine)  # must not raise

    assert len(engine.nodes) == 10
    for node in engine.nodes:
        assert isinstance(node["properties"], dict)
        assert "fields" in node["properties"]
        assert isinstance(node["properties"]["fields"], list)
        assert node["properties"]["type"] == "OntologySchema"


def test_register_declares_the_expected_schema_ids():
    engine = _StrictRealSignatureEngine()

    QuantOntology.register(engine)

    ids = {n["node_id"] for n in engine.nodes}
    assert ids == {
        "Schema:TradingHypothesis",
        "Schema:BacktestResult",
        "Schema:TradingStrategy",
        "Schema:MarketRegime",
        "Schema:AlphaFactor",
        "Schema:TradingSignal",
        "Schema:Order",
        "Schema:Portfolio",
        "Schema:PredictionMarket",
        "Schema:EnsembleForecast",
    }


def test_register_declares_the_formal_relations():
    engine = _StrictRealSignatureEngine()

    QuantOntology.register(engine)

    assert len(engine.edges) == 10
    assert (
        "Schema:TradingHypothesis",
        "Schema:BacktestResult",
        "ALLOWED_RELATION",
        "TESTED_IN",
    ) in engine.edges
