"""A committed graph slice as typed triples for EG SHACL validation (EH-472/473).

Tests that prove Agent Utilities' emitted LPG vocabulary conforms to an EG core
shape describe the slice as plain typed triples and let Epistemic Graph validate
them (``shacl_validate_committed``); no test parses or validates RDF locally.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from agent_utilities.knowledge_graph.core.typed_triples import (
    TypedTriple,
    iri,
    kg,
    literal,
    triple,
    typed,
)


def _node(node_id: str) -> str:
    return kg(f"node/{node_id}")


def _value(value: Any) -> dict[str, Any] | None:
    if isinstance(value, bool) or not isinstance(value, str | int | float):
        return None
    if isinstance(value, str) and not value:
        return None
    return literal(value)


def slice_triples(
    entities: Iterable[dict[str, Any]],
    links: Iterable[dict[str, Any]],
    *,
    declared_types: Iterable[tuple[str, str]] = (),
) -> list[TypedTriple]:
    """Entities (typed nodes + scalar properties), links, and any extra
    ``(node_id, class)`` declarations for out-of-slice link targets."""
    triples: list[TypedTriple] = []
    for entity in entities:
        subject = _node(entity["id"])
        triples.append(typed(subject, kg(entity["node_type"])))
        for key, value in entity.items():
            obj = None if key in {"id", "node_type"} else _value(value)
            if obj is not None:
                triples.append(triple(subject, kg(key), obj))
    triples.extend(typed(_node(node_id), kg(cls)) for node_id, cls in declared_types)
    triples.extend(
        triple(
            _node(link["source"]), kg(link["relationship"]), iri(_node(link["target"]))
        )
        for link in links
    )
    return triples
