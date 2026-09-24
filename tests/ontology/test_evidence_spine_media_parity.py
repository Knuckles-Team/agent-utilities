"""GOC-05 (universal artifact/evidence ontology) — ontology parity for the
node/edge vocabulary ``MediaStore`` (``knowledge_graph/memory/media_store.py``)
actually writes to the graph.

``MediaStore`` writes typed ``:AssetOccurrence``/``:Blob``/``:Evidence``/
``:Rendition``/``:SourceObject`` nodes (mirroring epistemic-graph's
``eg_modality::{Occurrence,Rendition}`` — CONCEPT:AU-KG.identity.evidence-spine-convergence /
CONCEPT:AU-KG.identity.asset-occurrence) linked by ``hasOccurrence``/``hasRendition``/
``hasBlob``/``extractedFrom``/``derivedFrom``/``SUPPORTS`` edges, but none of those
classes/properties had a declared ``owl:Class``/``owl:ObjectProperty`` in the
canonical ontology — a real class/property could be written to the graph with
no ontology entry describing it, the exact drift the former local ontology gate's
CONNECTED checks exist to catch for *files*, not per-symbol coverage. This test
is the per-symbol drift guard: it proves every node/edge vocabulary token
``MediaStore`` emits resolves to a declared ontology term, and that the check
is not vacuously true (a fabricated, never-written token is correctly absent).
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

KG_NS = "http://knuckles.team/kg#"


def _media_store_source() -> str:
    spec = importlib.util.find_spec(
        "agent_utilities.knowledge_graph.memory.media_store"
    )
    assert spec is not None and spec.origin is not None
    return Path(spec.origin).read_text(encoding="utf-8")


def _declared(terms) -> set[str]:
    return {
        str(term.iri).removeprefix(KG_NS)
        for term in terms
        if str(term.iri).startswith(KG_NS)
    }


# Node types MediaStore actually writes via ``client.nodes.add(id, {"node_type": ...})``.
_EXPECTED_NODE_TYPES = frozenset(
    {"AssetOccurrence", "Blob", "Evidence", "Rendition", "SourceObject"}
)

# Edge ``relationship`` values MediaStore actually writes via ``client.edges.add``,
# mapped to the ontology's camelCase ObjectProperty local name — matching the
# documented convention already used for e.g. ``:fragmentOf`` / ``FRAGMENT_OF``
# and ``:hasArtifact`` / ``HAS_ARTIFACT``.
_EXPECTED_EDGE_PROPERTIES = {
    "hasOccurrence": "hasOccurrence",
    "hasRendition": "hasRendition",
    "hasBlob": "hasBlob",
    "extractedFrom": "extractedFrom",
    "derivedFrom": "derivedFrom",
    "SUPPORTS": "supports",
}


def test_media_store_still_emits_the_vocabulary_this_test_pins() -> None:
    """Guard the guard: fail loudly (not vacuously) if MediaStore stops emitting
    one of these tokens, so this parity test cannot silently stop checking
    anything (CONCEPT:AU-OS.governance.fail-closed-degraded-read: a check with
    nothing to check must not report success)."""
    source = _media_store_source()
    for node_type in sorted(_EXPECTED_NODE_TYPES):
        assert f'"node_type": "{node_type}"' in source, (
            f"MediaStore no longer writes node_type={node_type!r} — "
            "update this pin (and consider removing the now-dead ontology class)"
        )
    for relationship in sorted(_EXPECTED_EDGE_PROPERTIES):
        assert re.search(rf'"relationship":\s*"{relationship}"', source), (
            f"MediaStore no longer writes relationship={relationship!r} — "
            "update this pin (and consider removing the now-dead ontology property)"
        )


# The canonical vocabulary is EG's composed GraphSchema, read through
# ``OntologyInspect`` (EH-471): Agent Utilities parses no ontology itself.


def test_every_media_store_node_type_has_a_declared_owl_class(engine_graph) -> None:
    declared = _declared(engine_graph.ontology_inspect().classes)
    missing = _EXPECTED_NODE_TYPES - declared
    assert not missing, (
        f"node_type(s) written by MediaStore with no owl:Class in the EG core: {missing}"
    )


def test_every_media_store_edge_relationship_has_a_declared_owl_property(
    engine_graph,
) -> None:
    declared = _declared(engine_graph.ontology_inspect().object_properties)
    missing = {
        relationship: prop_name
        for relationship, prop_name in _EXPECTED_EDGE_PROPERTIES.items()
        if prop_name not in declared
    }
    assert not missing, (
        f"edge relationship(s) written by MediaStore with no owl:ObjectProperty "
        f"in the EG core: {missing}"
    )


def test_a_fabricated_never_written_node_type_is_honestly_absent(engine_graph) -> None:
    """Known-bad proof: the check above is not vacuously true — a class this
    repo never declared and MediaStore never writes is correctly reported
    missing, not silently treated as present."""
    declared = _declared(engine_graph.ontology_inspect().classes)
    assert "TotallyFabricatedNodeTypeGOC05" not in declared
