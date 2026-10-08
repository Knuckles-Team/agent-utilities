"""The ontology facts source selection reads: labels, ancestry, relations.

EG owns the reasoning. :func:`tbox_from_sparql` asks EG one bounded SPARQL
question per fact family; ``rdfs:subClassOf*`` closure runs in EG. The
answer is a :class:`TripleOntology`, a read-only view over those facts.
Tests and offline fixtures build the same view from literal triples.

Every fact keeps its triple, so a selection can cite the exact ontology
facts that justified it.
"""

from __future__ import annotations

import re
from collections import defaultdict, deque
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass

RDFS = "http://www.w3.org/2000/01/rdf-schema#"
SKOS = "http://www.w3.org/2004/02/skos/core#"
LABEL = RDFS + "label"
ALT_LABEL = SKOS + "altLabel"
SUBCLASS = RDFS + "subClassOf"
DOMAIN = RDFS + "domain"
RANGE = RDFS + "range"

Triple = tuple[str, str, str]
_WORD = re.compile(r"[a-z0-9]+")
MAX_HOPS = 4


@dataclass(frozen=True, slots=True)
class ConceptMatch:
    """A question phrase that names an ontology class, and the label fact."""

    phrase: str
    class_iri: str
    fact: Triple


@dataclass(frozen=True, slots=True)
class Relation:
    """An object property from ``subject`` class to ``object`` class."""

    subject: str
    predicate: str
    object: str

    @property
    def facts(self) -> tuple[Triple, Triple]:
        return (
            (self.predicate, DOMAIN, self.subject),
            (self.predicate, RANGE, self.object),
        )


def _words(text: str) -> tuple[str, ...]:
    return tuple(_WORD.findall(text.lower()))


def _contains(haystack: tuple[str, ...], needle: tuple[str, ...]) -> bool:
    width = len(needle)
    return any(
        haystack[i : i + width] == needle for i in range(len(haystack) - width + 1)
    )


class TripleOntology:
    """A read-only TBox view: labels, asserted ancestry and relations."""

    def __init__(self, triples: Iterable[Triple]) -> None:
        self._labels: list[Triple] = []
        self._parents: dict[str, set[str]] = defaultdict(set)
        domains: dict[str, str] = {}
        ranges: dict[str, str] = {}
        for s, p, o in triples:
            if p in (LABEL, ALT_LABEL):
                self._labels.append((s, p, o))
            elif p == SUBCLASS:
                self._parents[s].add(o)
            elif p == DOMAIN:
                domains[s] = o
            elif p == RANGE:
                ranges[s] = o
        self.relations = tuple(
            Relation(domains[p], p, ranges[p]) for p in sorted(domains) if p in ranges
        )

    def concepts(self, text: str) -> list[ConceptMatch]:
        """Classes whose label or alternative label occurs in ``text``.

        A longer label wins over a label it contains, so "supply chain"
        does not also yield a class labelled "chain".
        """
        words = _words(text)
        hits = [
            (len(_words(label)), ConceptMatch(label, iri, (iri, pred, label)))
            for iri, pred, label in self._labels
            if _words(label) and _contains(words, _words(label))
        ]
        hits.sort(key=lambda hit: (-hit[0], hit[1].class_iri))
        chosen: dict[str, ConceptMatch] = {}
        for _, match in hits:
            covered = any(
                _contains(_words(c.phrase), _words(match.phrase))
                for c in chosen.values()
            )
            if match.class_iri not in chosen and not covered:
                chosen[match.class_iri] = match
        return list(chosen.values())

    def ancestors(self, class_iri: str) -> frozenset[str]:
        """``class_iri`` and every asserted superclass, transitively."""
        seen = {class_iri}
        queue = deque([class_iri])
        while queue:
            for parent in self._parents.get(queue.popleft(), ()):
                if parent not in seen:
                    seen.add(parent)
                    queue.append(parent)
        return frozenset(seen)

    def path(self, start: str, goal: str) -> tuple[Relation, ...] | None:
        """The shortest relation path linking two classes, either direction."""
        prev: dict[str, tuple[str, Relation] | None] = {start: None}
        queue = deque([start])
        while queue:
            node = queue.popleft()
            if node == goal:
                return self._unwind(prev, goal)
            for other, rel in self._neighbours(node):
                if other not in prev and self._depth(prev, node) < MAX_HOPS:
                    prev[other] = (node, rel)
                    queue.append(other)
        return None

    def _neighbours(self, node: str) -> list[tuple[str, Relation]]:
        lineage = self.ancestors(node)
        out = [(r.object, r) for r in self.relations if r.subject in lineage]
        out += [(r.subject, r) for r in self.relations if r.object in lineage]
        return out

    @staticmethod
    def _depth(prev: Mapping[str, tuple[str, Relation] | None], node: str) -> int:
        depth = 0
        while (step := prev[node]) is not None:
            node, depth = step[0], depth + 1
        return depth

    @staticmethod
    def _unwind(
        prev: Mapping[str, tuple[str, Relation] | None], goal: str
    ) -> tuple[Relation, ...]:
        edges: list[Relation] = []
        node = goal
        while (step := prev[node]) is not None:
            node, rel = step
            edges.append(rel)
        return tuple(reversed(edges))


Sparql = Callable[[str], Awaitable[Sequence[Mapping[str, object]]]]

#: The bounded TBox questions; subclass closure is entailed by EG.
TBOX_QUERIES: tuple[tuple[str, str], ...] = (
    (
        "labels",
        f"SELECT ?s ?p ?o WHERE {{ VALUES ?p {{ <{LABEL}> <{ALT_LABEL}> }} ?s ?p ?o }}",
    ),
    ("ancestry", f"SELECT ?s ?o WHERE {{ ?s <{SUBCLASS}>+ ?o }}"),
    ("relations", f"SELECT ?p ?d ?r WHERE {{ ?p <{DOMAIN}> ?d ; <{RANGE}> ?r }}"),
)


def _rows_to_triples(kind: str, rows: Sequence[Mapping[str, object]]) -> list[Triple]:
    out: list[Triple] = []
    for row in rows:
        if kind == "labels":
            out.append((str(row["s"]), str(row["p"]), str(row["o"])))
        elif kind == "ancestry":
            out.append((str(row["s"]), SUBCLASS, str(row["o"])))
        else:
            out.append((str(row["p"]), DOMAIN, str(row["d"])))
            out.append((str(row["p"]), RANGE, str(row["r"])))
    return out


async def tbox_from_sparql(sparql: Sparql) -> TripleOntology:
    """Build the TBox view from EG's answers to :data:`TBOX_QUERIES`."""
    triples: list[Triple] = []
    for kind, query in TBOX_QUERIES:
        triples += _rows_to_triples(kind, await sparql(query))
    return TripleOntology(triples)


__all__ = [
    "ConceptMatch",
    "Relation",
    "TBOX_QUERIES",
    "TripleOntology",
    "tbox_from_sparql",
]
