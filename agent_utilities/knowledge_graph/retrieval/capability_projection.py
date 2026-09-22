"""Request-scoped capability subsumption projected by epistemic-graph.

The immutable capability TBox is owned and composed by epistemic-graph.  AU only
consumes the classified result returned by ``OwlReason``; it never parses or
caches a bundled Turtle document.  ``schema_digests`` bind every projection to
the exact composed GraphSchema sources that produced it.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

_KG_NAMESPACE = "http://knuckles.team/kg#"


class CapabilityProjectionUnavailable(RuntimeError):
    """The authoritative engine capability classification is unavailable."""


def _local_name(value: Any) -> str | None:
    iri = str(value).strip()
    if iri.startswith("<") and iri.endswith(">"):
        iri = iri[1:-1]
    if not iri.startswith(_KG_NAMESPACE):
        return None
    local = iri.removeprefix(_KG_NAMESPACE)
    return local or None


@dataclass(frozen=True)
class CapabilitySubsumptionProjection:
    """One engine-derived, GraphSchema-digest-bound subsumption projection."""

    relations: frozenset[tuple[str, str]]
    direct_relations: frozenset[tuple[str, str]]
    schema_digests: tuple[str, ...]
    _parents: dict[str, frozenset[str]] = field(init=False, repr=False)
    _children: dict[str, frozenset[str]] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.schema_digests:
            raise CapabilityProjectionUnavailable(
                "OwlReason did not bind capability classification to GraphSchema"
            )

        parents: dict[str, set[str]] = {}
        children: dict[str, set[str]] = {}
        for child, parent in self.direct_relations:
            if child == parent:
                continue
            parents.setdefault(child, set()).add(parent)
            children.setdefault(parent, set()).add(child)
        object.__setattr__(
            self, "_parents", {key: frozenset(value) for key, value in parents.items()}
        )
        object.__setattr__(
            self,
            "_children",
            {key: frozenset(value) for key, value in children.items()},
        )

    def parents_of(self, name: str) -> frozenset[str]:
        return self._parents.get(name, frozenset())

    def children_of(self, name: str) -> frozenset[str]:
        return self._children.get(name, frozenset())

    def ancestors(self, name: str) -> frozenset[str]:
        return frozenset(parent for child, parent in self.relations if child == name)

    def descendants(self, name: str) -> frozenset[str]:
        return frozenset(child for child, parent in self.relations if parent == name)

    def is_subtype_of(self, candidate: str, required: str) -> bool:
        return candidate == required or (candidate, required) in self.relations

    def subsumption_path(self, candidate: str, required: str) -> list[str] | None:
        if candidate == required:
            return [candidate]
        if (candidate, required) not in self.relations:
            return None
        queue: deque[list[str]] = deque([[candidate]])
        visited = {candidate}
        while queue:
            path = queue.popleft()
            for parent in sorted(self.parents_of(path[-1])):
                if parent in visited:
                    continue
                candidate_path = [*path, parent]
                if parent == required:
                    return candidate_path
                visited.add(parent)
                queue.append(candidate_path)
        return [candidate, required]


def _reason_capabilities(engine: Any) -> dict[str, Any]:
    graph = getattr(engine, "graph", None)
    reason = getattr(graph, "owl_reason", None)
    if not callable(reason):
        raise CapabilityProjectionUnavailable(
            "epistemic-graph OwlReason is required for capability routing"
        )
    try:
        result = reason(
            ontology=None,
            target_class=None,
            class_base=_KG_NAMESPACE,
        )
    except Exception as exc:
        raise CapabilityProjectionUnavailable(
            "epistemic-graph could not classify the composed GraphSchema"
        ) from exc
    if not isinstance(result, dict) or result.get("consistent") is not True:
        raise CapabilityProjectionUnavailable(
            "composed GraphSchema capability classification is inconsistent"
        )

    return result


def _schema_digests(result: dict[str, Any]) -> tuple[str, ...]:
    digests = result.get("schema_digests")
    if (
        not isinstance(digests, list)
        or not digests
        or not all(isinstance(digest, str) and digest for digest in digests)
    ):
        raise CapabilityProjectionUnavailable(
            "OwlReason omitted composed GraphSchema source digests"
        )

    return tuple(sorted(set(digests)))


def _normalized_relations(result: dict[str, Any], field: str) -> set[tuple[str, str]]:
    raw_relations = result.get(field)
    if not isinstance(raw_relations, list):
        raise CapabilityProjectionUnavailable(
            f"OwlReason returned no {field} named-class relation set"
        )
    normalized: set[tuple[str, str]] = set()
    for relation in raw_relations:
        if not isinstance(relation, (list, tuple)) or len(relation) != 2:
            continue
        child = _local_name(relation[0])
        parent = _local_name(relation[1])
        if child is not None and parent is not None:
            normalized.add((child, parent))
    return normalized


def load_capability_projection(engine: Any) -> CapabilitySubsumptionProjection:
    """Classify the composed GraphSchema once and return a request-scoped view.

    There is deliberately no local-file or process-singleton fallback. Missing
    schema lineage or an inconsistent classification fails closed.
    """
    result = _reason_capabilities(engine)
    digests = _schema_digests(result)

    relations = _normalized_relations(result, "subclasses")
    direct_relations = _normalized_relations(result, "direct_subclasses")
    if not relations:
        raise CapabilityProjectionUnavailable(
            "OwlReason returned no named-class subsumption closure"
        )
    if not direct_relations:
        raise CapabilityProjectionUnavailable(
            "OwlReason returned no direct named-class subsumption edges"
        )
    if not direct_relations.issubset(relations):
        raise CapabilityProjectionUnavailable(
            "OwlReason direct subsumption edges disagree with its closure"
        )
    return CapabilitySubsumptionProjection(
        relations=frozenset(relations),
        direct_relations=frozenset(direct_relations),
        schema_digests=digests,
    )


__all__ = [
    "CapabilityProjectionUnavailable",
    "CapabilitySubsumptionProjection",
    "load_capability_projection",
]
