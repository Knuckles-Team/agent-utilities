"""Cross-source reports: the ontology selects sources; reads stay live.

A question such as "How does the supply chain affect our services?" runs in
four steps:

1. The ontology names the classes the question mentions
   (:meth:`TripleOntology.concepts`).
2. The ontology links those classes through object properties
   (:meth:`TripleOntology.path`).
3. The catalog selects, per class on that path, the approved
   :class:`VirtualMapping` that serves it. That choice is the source
   selection, and each choice cites its ontology and mapping facts.
4. The report joins live rows along the path. Each hop is a bind join: the
   keys of the known side go to the source of the new side. No entity row
   is copied into the graph.

A row budget bounds the join. Exceeding it marks the report incomplete; it
never truncates silently.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from agent_utilities.knowledge_graph.virtual_graph.contracts import (
    MetadataContract,
    SourceConnection,
    VirtualMapping,
    check_mapping,
)
from agent_utilities.knowledge_graph.virtual_graph.ontology import (
    ConceptMatch,
    Relation,
    TripleOntology,
)

Row = Mapping[str, Any]


class SourceAdapter(Protocol):
    """Reads one mapped entity live; ``field``/``values`` is a key filter."""

    async def fetch(
        self,
        mapping: VirtualMapping,
        field: str | None = None,
        values: Sequence[Any] | None = None,
    ) -> list[Row]: ...


@dataclass
class VirtualCatalog:
    """The registered sources, their metadata, approved mappings and adapters."""

    connections: dict[str, SourceConnection] = field(default_factory=dict)
    contracts: dict[str, MetadataContract] = field(default_factory=dict)
    adapters: dict[str, SourceAdapter] = field(default_factory=dict)
    mappings: list[VirtualMapping] = field(default_factory=list)

    def register(
        self,
        connection: SourceConnection,
        contract: MetadataContract,
        adapter: SourceAdapter,
    ) -> None:
        if contract.source_id != connection.source_id:
            raise ValueError("contract and connection name different sources")
        self.connections[connection.source_id] = connection
        self.contracts[connection.source_id] = contract
        self.adapters[connection.source_id] = adapter

    def add_mapping(self, mapping: VirtualMapping) -> None:
        contract = self.contracts.get(mapping.source_id)
        if contract is None:
            raise ValueError(f"unregistered source {mapping.source_id!r}")
        check_mapping(mapping, contract)
        self.mappings.append(mapping)

    def mapping_for(
        self, class_iri: str, ontology: TripleOntology
    ) -> VirtualMapping | None:
        """The approved mapping of ``class_iri`` or of one of its subclasses."""
        serving = [
            m
            for m in self.mappings
            if m.approved and class_iri in ontology.ancestors(m.class_iri)
        ]
        serving.sort(key=lambda m: (m.class_iri != class_iri, m.mapping_id))
        return serving[0] if serving else None


@dataclass(frozen=True, slots=True)
class SelectedSource:
    """One class on the path and the mapping chosen to serve it."""

    class_iri: str
    mapping: VirtualMapping
    contract_digest: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "class_iri": self.class_iri,
            "source_id": self.mapping.source_id,
            "mapping_id": self.mapping.mapping_id,
            "mapping_version": self.mapping.version,
            "contract_digest": self.contract_digest,
            "justification": {
                "mapsClass": self.mapping.class_iri,
                "approved": self.mapping.approved,
            },
        }


@dataclass(frozen=True, slots=True)
class SourceSelection:
    """What the ontology concluded the question needs, and what serves it."""

    concepts: tuple[ConceptMatch, ...]
    path: tuple[Relation, ...]
    sources: tuple[SelectedSource, ...]
    uncovered: tuple[str, ...]
    reason: str = "selected"

    @property
    def complete(self) -> bool:
        return not self.uncovered and self.reason == "selected"


def _path_classes(
    concepts: Sequence[ConceptMatch], path: Sequence[Relation]
) -> list[str]:
    ordered = [c.class_iri for c in concepts[:1]]
    for rel in path:
        for iri in (rel.subject, rel.object):
            if iri not in ordered:
                ordered.append(iri)
    return ordered


def _linking_path(
    concepts: Sequence[ConceptMatch], ontology: TripleOntology
) -> tuple[Relation, ...] | None:
    edges: list[Relation] = []
    start = concepts[0].class_iri
    for other in concepts[1:]:
        hop = ontology.path(start, other.class_iri)
        if hop is None:
            return None
        edges += [rel for rel in hop if rel not in edges]
    return tuple(edges)


def select_sources(
    question: str, ontology: TripleOntology, catalog: VirtualCatalog
) -> SourceSelection:
    """Select the sources for ``question`` from ontology and mapping facts."""
    concepts = tuple(ontology.concepts(question))
    if len(concepts) < 2:
        return SourceSelection(concepts, (), (), (), "fewer_than_two_concepts")
    path = _linking_path(concepts, ontology)
    if path is None:
        return SourceSelection(concepts, (), (), (), "no_ontology_path")
    sources: list[SelectedSource] = []
    uncovered: list[str] = []
    for iri in _path_classes(concepts, path):
        mapping = catalog.mapping_for(iri, ontology)
        if mapping is None:
            uncovered.append(iri)
            continue
        digest = catalog.contracts[mapping.source_id].digest
        sources.append(SelectedSource(iri, mapping, digest))
    return SourceSelection(concepts, path, tuple(sources), tuple(uncovered))


@dataclass
class CrossSourceReport:
    """The joined live rows and the provenance of every element."""

    question: str
    selection: SourceSelection
    rows: list[dict[str, Row]] = field(default_factory=list)
    reads: list[dict[str, Any]] = field(default_factory=list)
    complete: bool = True
    reason: str = "answered"

    def to_dict(self) -> dict[str, Any]:
        sel = self.selection
        return {
            "kind": "cross_source_report",
            "question": self.question,
            "complete": self.complete,
            "reason": self.reason,
            "concepts": [
                {"phrase": c.phrase, "class_iri": c.class_iri, "fact": list(c.fact)}
                for c in sel.concepts
            ],
            "path": [
                {
                    "subject": r.subject,
                    "predicate": r.predicate,
                    "object": r.object,
                    "facts": [list(f) for f in r.facts],
                }
                for r in sel.path
            ],
            "sources": [s.to_dict() for s in sel.sources],
            "uncovered_classes": list(sel.uncovered),
            "rows": [{k: dict(v) for k, v in row.items()} for row in self.rows],
            "reads": self.reads,
            "materialized_rows": 0,
        }


@dataclass(frozen=True, slots=True)
class _Hop:
    """One join step: from the known class to the new class over ``rel``."""

    known: str
    new: str
    known_field: str
    new_field: str


def _hop(rel: Relation, known: str, by_class: Mapping[str, VirtualMapping]) -> _Hop:
    forward = rel.subject == known
    new = rel.object if forward else rel.subject
    subj, obj = by_class[rel.subject], by_class[rel.object]
    fk = subj.field_for(rel.predicate)
    if fk is None:
        raise LookupError(f"mapping {subj.mapping_id} has no field for {rel.predicate}")
    if forward:
        return _Hop(known, new, fk, obj.key_field)
    return _Hop(known, new, obj.key_field, fk)


async def _read(
    catalog: VirtualCatalog,
    report: CrossSourceReport,
    mapping: VirtualMapping,
    field_name: str | None = None,
    values: Sequence[Any] | None = None,
) -> list[Row]:
    rows = await catalog.adapters[mapping.source_id].fetch(mapping, field_name, values)
    report.reads.append(
        {
            "source_id": mapping.source_id,
            "mapping_id": mapping.mapping_id,
            "filter_field": field_name,
            "keys": len(values or ()),
            "rows": len(rows),
            "mode": "virtual",
        }
    )
    return rows


def _join(
    chains: list[dict[str, Row]], hop: _Hop, fetched: Sequence[Row]
) -> list[dict[str, Row]]:
    index: dict[Any, list[Row]] = {}
    for row in fetched:
        index.setdefault(row.get(hop.new_field), []).append(row)
    joined: list[dict[str, Row]] = []
    for chain in chains:
        for match in index.get(chain[hop.known].get(hop.known_field), ()):
            joined.append({**chain, hop.new: match})
    return joined


def _next_hop(
    pending: list[Relation], bound: Mapping[str, Row], by_class: Mapping[str, Any]
) -> _Hop:
    """Take the first pending relation that touches an already-bound class."""
    rel = next(r for r in pending if r.subject in bound or r.object in bound)
    pending.remove(rel)
    return _hop(rel, rel.subject if rel.subject in bound else rel.object, by_class)


async def _extend(
    report: CrossSourceReport,
    catalog: VirtualCatalog,
    chains: list[dict[str, Row]],
    hop: _Hop,
    mapping: VirtualMapping,
) -> list[dict[str, Row]]:
    """Bind-join ``chains`` to the new class: send the known keys to its source."""
    keys = {c[hop.known].get(hop.known_field) for c in chains} - {None}
    fetched = await _read(
        catalog, report, mapping, hop.new_field, sorted(keys, key=str)
    )
    return _join(chains, hop, fetched)


async def _walk(
    report: CrossSourceReport, catalog: VirtualCatalog, row_budget: int
) -> None:
    sel = report.selection
    by_class = {s.class_iri: s.mapping for s in sel.sources}
    first = sel.concepts[0].class_iri
    chains = [{first: row} for row in await _read(catalog, report, by_class[first])]
    pending = list(sel.path)
    while pending and chains:
        hop = _next_hop(pending, chains[0], by_class)
        chains = await _extend(report, catalog, chains, hop, by_class[hop.new])
        if len(chains) > row_budget:
            report.complete, report.reason = False, "row_budget_exceeded"
            return
    report.rows = chains


async def cross_source_report(
    question: str,
    ontology: TripleOntology,
    catalog: VirtualCatalog,
    *,
    row_budget: int = 1000,
) -> CrossSourceReport:
    """Answer ``question`` by federating live reads over virtual mappings."""
    selection = select_sources(question, ontology, catalog)
    report = CrossSourceReport(question, selection)
    if not selection.complete:
        report.complete = False
        report.reason = "uncovered_classes" if selection.uncovered else selection.reason
        return report
    await _walk(report, catalog, row_budget)
    return report


OntologyProvider = Callable[[], Awaitable[TripleOntology]]
_INSTALLED: list[tuple[VirtualCatalog, OntologyProvider] | None] = [None]


def install_cross_source(
    catalog: VirtualCatalog | None, ontology: OntologyProvider | None = None
) -> None:
    """Install the process catalog and its ontology provider (graph-os)."""
    if catalog is None or ontology is None:
        _INSTALLED[0] = None
        return
    _INSTALLED[0] = (catalog, ontology)


async def answer_cross_source(question: str) -> CrossSourceReport | None:
    """The installed catalog's report, or ``None`` when the question is not
    cross-source (no catalog, fewer than two concepts, or no ontology path)."""
    installed = _INSTALLED[0]
    if installed is None:
        return None
    catalog, ontology = installed
    report = await cross_source_report(question, await ontology(), catalog)
    not_ours = {"fewer_than_two_concepts", "no_ontology_path"}
    return None if report.reason in not_ours else report


__all__ = [
    "answer_cross_source",
    "install_cross_source",
    "CrossSourceReport",
    "SelectedSource",
    "SourceAdapter",
    "SourceSelection",
    "VirtualCatalog",
    "cross_source_report",
    "select_sources",
]
