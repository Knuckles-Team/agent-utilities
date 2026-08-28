"""Typed request value objects for the ``graph_ontology`` action dispatcher.

Leaf module — imports NOTHING from ``agent_utilities`` (CX wD10-R-ARITY / BUG-CX-004 §9b,
same rationale as ``agent_utilities.mcp.bus_types``: ``agent_utilities.mcp.tools`` is the
highest-leverage node in au's import SCC because its ``__init__.py`` eagerly imports all 35
tool-registration modules, so anything defined *inside* that package pulls the whole cluster
in for any external caller. These dataclasses are pure data, so a caller that only needs to
build/read one doesn't have to pay that cost.

``graph_ontology`` itself keeps its full 25-parameter signature — that is the published MCP
tool schema and FastMCP derives it directly from the signature, so it is the wire contract and
must not change shape. These two dataclasses replace the *internal* over-cap dispatch layer
(``_graph_ontology_catalog`` was 13 params, ``_graph_ontology_proposal`` was 14) — the six
per-action leaf handlers underneath them (``_graph_ontology_load``, `_get`, `_update`, ...)
were already <=7 params and are untouched.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class OntologyCatalogRequest:
    """Fields ``_graph_ontology_catalog``'s six actions (load/list/get/update/delete/validate)
    draw from — each action reads only the 3-6 fields relevant to it."""

    source: str = ""
    source_type: str = "auto"
    iri: str = ""
    version: str = ""
    serialize: bool = False
    active_only: bool = False
    deprecated_only: bool = False
    drop_inferences: bool = False
    category: str = ""
    tags_json: str = ""
    search: str = ""
    tag: str = ""


@dataclass(frozen=True, slots=True)
class OntologyProposalRequest:
    """Fields ``_graph_ontology_proposal``'s six actions (propose/list_proposals/get_proposal/
    review_proposal/promote_proposal/rollback_proposal) draw from."""

    proposal_id: str = ""
    source: str = ""
    iri: str = ""
    source_type: str = "auto"
    evidence_refs_json: str = ""
    proposer: str = ""
    reason: str = ""
    approve: bool = True
    reviewer: str = ""
    notes: str = ""
    status: str = ""
