"""Ontology-guided extraction schema (CONCEPT:AU-KG.retrieval.mmr-diversification).

Reads the OWL **TBox** (``owl:Class`` + ``owl:ObjectProperty`` with
``rdfs:domain``/``rdfs:range`` + labels) of the graph's committed GraphSchema
sources from Epistemic Graph (``OntologyInspect``, EH-471) into a compact,
prompt-ready :class:`ExtractionSchema`, so the LLM fact extractor
(:mod:`agent_utilities.knowledge_graph.extraction.fact_extractor`) extracts
**ontology-typed** entities and **direction-constrained** relations instead of
free snake_case predicates.

The schema *is* the ontology. sift-kg injects a flat YAML schema into its prompt;
we inject our formal OWL classes + ``rdfs:domain/range``, then keep the post-hoc
grounding (:mod:`.ontology_grounding`) and the engine's OWL reasoning downstream —
generation-time guidance *and* reasoning, which a flat schema cannot give.

Design notes:

* **Epistemic Graph is the only RDF parser.** Agent Utilities never loads an
  ontology document; it names EG core sources (``core:<module>@1``) and reads the
  typed vocabulary view back. An unavailable engine degrades to ``None``
  (free-vocab extraction), so ingestion never breaks on a schema read.
* The TBox namespace is ``http://knuckles.team/kg#`` — distinct from the engine
  LPG-projection ``au:`` namespace, which is for *instance* data (KG-2.240/2.242).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

logger = logging.getLogger(__name__)

_TBOX_NS = "http://knuckles.team/kg#"

# Cap the injected schema so a small model's prompt never bloats (top-N by
# relevance). One correct value, not a knob.
_MAX_ENTITY_TYPES = 40
_MAX_RELATIONS = 40

# Content types that are NOT prose entity/relation domains — they have their own
# extraction path (codebase → AST) or carry no graphable entities. These skip
# ontology-guided fact extraction (return None → unchanged free-vocab behaviour).
_SKIP_TYPES: frozenset[str] = frozenset(
    {"codebase", "config", "event", "mcp_server", "skill", "prompt", "sparql"}
)

# The foundational core source — applies to all prose content as the default
# closed vocabulary. Domain-specific source types additionally read their EG core
# module.
_CORE_SOURCES: tuple[str, ...] = ("core:foundation@1",)

# Source-type → extra EG core ontology source(s), merged with the core. Keyed on
# the substring that identifies the domain in the source_type/connector name, so
# both ``"medical"`` and ``"connector:medical"`` resolve. Static map, not a flag
# (Configuration discipline). Unmatched prose content uses the core only; a
# connector's own vocabulary lives in its connector pack, not in this table.
_DOMAIN_SOURCES: dict[str, tuple[str, ...]] = {
    "medical": ("core:medical@1",),
    "hr": ("core:hr@1",),
    "government": ("core:government@1",),
    "enterprise": ("core:enterprise@1",),
    "infrastructure": ("core:infrastructure@1",),
    "calendar": ("core:calendar@1",),
    "energy": ("core:energy_geopolitics@1",),
}


def _camel_to_snake(name: str) -> str:
    """``decidedBy`` → ``decided_by`` (the extractor's snake_case predicate form)."""
    s = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s)
    return s.lower()


@dataclass(frozen=True)
class EntityType:
    """One ``owl:Class`` rendered as a closed-vocabulary entity type."""

    name: str  # class local name, e.g. "Organization"
    description: str = ""
    synonyms: tuple[str, ...] = ()


@dataclass(frozen=True)
class Relation:
    """One ``owl:ObjectProperty`` with its declared direction (domain → range)."""

    predicate: str  # snake_case form, e.g. "decided_by"
    label: str = ""
    domain: tuple[str, ...] = ()  # subject class local names
    range: tuple[str, ...] = ()  # object class local names
    symmetric: bool = False


@dataclass(frozen=True)
class ExtractionSchema:
    """A compact, prompt-ready view of an ontology subset (CONCEPT:AU-KG.retrieval.mmr-diversification)."""

    name: str
    entity_types: tuple[EntityType, ...] = field(default_factory=tuple)
    relations: tuple[Relation, ...] = field(default_factory=tuple)

    @property
    def is_empty(self) -> bool:
        return not self.entity_types and not self.relations

    @property
    def closed_predicate_set(self) -> frozenset[str]:
        """The typed predicates this schema knows (for post-validation in F)."""
        return frozenset(r.predicate for r in self.relations)

    def relations_by_predicate(self) -> dict[str, Relation]:
        return {r.predicate: r for r in self.relations}

    def prompt_block(self) -> str:
        """Render the schema as a prompt section spliced into the extractor prompt.

        Soft-closed by design (Wire-First recall guard): the prompt *prefers* the
        typed vocabulary but explicitly permits coining a new predicate when none
        fits, so we exceed sift-kg's hard-closed vocabulary while keeping recall.
        """
        if self.is_empty:
            return ""
        lines: list[str] = [
            "ONTOLOGY SCHEMA — you are populating THIS knowledge graph. Prefer its",
            "types and relations so entities/edges merge; coin a new term ONLY when",
            "none fits (controlled overflow, not a hard menu).",
            "",
            "Entity types (set subject/object to the closest type's canonical name):",
        ]
        for et in self.entity_types[:_MAX_ENTITY_TYPES]:
            syn = f" (aka {', '.join(et.synonyms[:6])})" if et.synonyms else ""
            desc = f" — {et.description}" if et.description else ""
            lines.append(f"- {et.name}{desc}{syn}")
        lines.append("")
        lines.append(
            "Typed relations (prefer these predicates; the subject MUST be the type "
            "on the LEFT of →):"
        )
        for rel in self.relations[:_MAX_RELATIONS]:
            dom = "|".join(rel.domain) if rel.domain else "Thing"
            rng = "|".join(rel.range) if rel.range else "Thing"
            sym = " [symmetric]" if rel.symmetric else ""
            lines.append(f"- {rel.predicate}: {dom} → {rng}{sym}")
        lines.append("")
        return "\n".join(lines)


def _source_ids(source_type: str) -> tuple[str, ...] | None:
    """Resolve the EG core source ids for ``source_type`` (or None to skip)."""
    st = (source_type or "").strip().lower()
    if not st or st in _SKIP_TYPES:
        return None
    sources: list[str] = list(_CORE_SOURCES)
    for key, ids in _DOMAIN_SOURCES.items():
        if key in st:
            sources.extend(source for source in ids if source not in sources)
    return tuple(sources)


def _synonyms_for(class_local: str) -> tuple[str, ...]:
    """Reverse the grounding lexicon: class local name → its surface synonyms.

    Reuses ``ontology_grounding._RAW_CLASS_SYNONYMS`` (the existing convergence
    table) rather than a second lexicon. Best-effort: returns ``()`` if grounding
    is unavailable or the class has no registered synonyms.
    """
    try:
        from .ontology_grounding import _RAW_CLASS_SYNONYMS
    except Exception:  # noqa: BLE001
        return ()
    key = class_local.lower()
    syns = sorted(
        {surface for surface, target in _RAW_CLASS_SYNONYMS.items() if target == key}
        - {key}
    )
    return tuple(syns)


def _local(iri: str) -> str:
    text = str(iri)
    return text.rsplit("#", 1)[1] if "#" in text else text.rsplit("/", 1)[-1]


def _describe(term: Any) -> str:
    text = term.comment or term.label or ""
    return str(text).strip().replace("\n", " ")[:160]


def _tbox_locals(iris: Any) -> tuple[str, ...]:
    return tuple(_local(iri) for iri in iris if str(iri).startswith(_TBOX_NS))


def _entity_types(view: Any) -> list[EntityType]:
    entity_types: dict[str, EntityType] = {}
    for term in view.classes:
        local = _local(term.iri)
        if str(term.iri).startswith(_TBOX_NS) and local not in entity_types:
            entity_types[local] = EntityType(
                name=local, description=_describe(term), synonyms=_synonyms_for(local)
            )
    return list(entity_types.values())


def _relations(view: Any) -> list[Relation]:
    relations: dict[str, Relation] = {}
    for prop in view.object_properties:
        predicate = _camel_to_snake(_local(prop.iri))
        if not str(prop.iri).startswith(_TBOX_NS) or predicate in relations:
            continue
        relations[predicate] = Relation(
            predicate=predicate,
            label=str(prop.label or "").strip(),
            domain=_tbox_locals(prop.domains),
            range=_tbox_locals(prop.ranges),
            symmetric=bool(prop.symmetric),
        )
    return list(relations.values())


def _schema_from_view(name: str, view: Any) -> ExtractionSchema | None:
    """Build the schema from EG's typed vocabulary view (EH-471)."""
    entity_types = _entity_types(view)
    relations = _relations(view)
    if not entity_types and not relations:
        return None
    # Relevance ordering: relations with BOTH endpoints typed first (they carry
    # direction constraints F uses); classes referenced by a relation first.
    referenced = {local for r in relations for local in (*r.domain, *r.range)}
    entity_types.sort(key=lambda e: (e.name not in referenced, e.name))
    relations.sort(key=lambda r: (not (r.domain and r.range), r.predicate))
    return ExtractionSchema(
        name=name, entity_types=tuple(entity_types), relations=tuple(relations)
    )


def _inspect_sources(source_ids: tuple[str, ...]) -> Any:
    from ..core.graph_compute import GraphComputeEngine

    return GraphComputeEngine.get_or_create().ontology_inspect(
        source_ids=list(source_ids)
    )


@lru_cache(maxsize=64)
def load_extraction_schema(source_type: str) -> ExtractionSchema | None:
    """Return the ontology-guided extraction schema for ``source_type`` (cached).

    ``None`` means *no ontology guidance* — the extractor falls back to its
    free-vocab prompt unchanged (non-prose content, EG unavailable, or an empty
    read). Never raises: ingestion must not break on a schema-load failure.
    """
    sources = _source_ids(source_type)
    if not sources:
        return None
    try:
        view = _inspect_sources(sources)
    except Exception as exc:
        logger.warning(
            "load_extraction_schema(%s): EG OntologyInspect failed: %s",
            source_type,
            exc,
        )
        return None
    schema = _schema_from_view("+".join(sources), view)
    if schema is None or schema.is_empty:
        return None
    return schema


__all__ = [
    "EntityType",
    "Relation",
    "ExtractionSchema",
    "load_extraction_schema",
    "CONTENT_TYPE_TO_ONTOLOGY",
]

# Public alias for the content→ontology map (referenced in docs/tests).
CONTENT_TYPE_TO_ONTOLOGY = _DOMAIN_SOURCES
