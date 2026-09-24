"""ST-1: the ``swarm-topology`` schema source and its one publisher.

The vocabulary is data (invariant T4): two Turtle documents shipped with AU,
attached to EG as ONE keyed ``GraphSchema`` source by an admin
(``security:admin``). EG reasons it in-process; AU never writes topology
vocabulary as generic graph nodes or Cypher.
"""

from __future__ import annotations

from importlib import resources
from typing import Any, Protocol

#: The GraphSchema key the vocabulary is attached under.
SOURCE_ID = "swarm-topology"
#: The vocabulary namespace.
SWARM_NS = "http://knuckles.team/kg/swarm#"

_PACKAGE = "agent_utilities.knowledge_graph.ontology"


def _document(name: str) -> str:
    return (
        resources.files(_PACKAGE)
        .joinpath("swarm_topology", name)
        .read_text(encoding="utf-8")
    )


def ontology_ttl() -> str:
    """The TBox: topology classes, slot roles, stop rules and admissibility."""
    return _document("topology.ttl")


def shapes_ttl() -> str:
    """The SHACL shapes over a template's RDF projection."""
    return _document("shapes.ttl")


class SchemaAttacher(Protocol):
    """The admin GraphSchema attach surface (``GraphCompute.graph_schema_attach``)."""

    def graph_schema_attach(
        self,
        source_id: str,
        *,
        ontology_ttl: str | None = None,
        shapes_ttl: str | None = None,
        if_composed_digest: str | None = None,
    ) -> Any: ...


def attach_swarm_topology(
    compute: SchemaAttacher, *, if_composed_digest: str | None = None
) -> Any:
    """Attach (or replace) the vocabulary under :data:`SOURCE_ID`.

    ``if_composed_digest`` fences the attach on the composed schema the caller
    last saw, so two admins cannot silently overwrite each other.
    """
    return compute.graph_schema_attach(
        SOURCE_ID,
        ontology_ttl=ontology_ttl(),
        shapes_ttl=shapes_ttl(),
        if_composed_digest=if_composed_digest,
    )


__all__ = [
    "SOURCE_ID",
    "SWARM_NS",
    "SchemaAttacher",
    "attach_swarm_topology",
    "ontology_ttl",
    "shapes_ttl",
]
