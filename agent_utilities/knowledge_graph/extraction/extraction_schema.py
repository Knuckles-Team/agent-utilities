"""Ontology-guided extraction schema (CONCEPT:AU-KG.retrieval.mmr-diversification).

Loads the OWL **TBox** (``owl:Class`` + ``owl:ObjectProperty`` with
``rdfs:domain``/``rdfs:range`` + labels) from the canonical ontology ``.ttl``
modules into a compact, prompt-ready :class:`ExtractionSchema`, so the LLM fact
extractor (:mod:`agent_utilities.knowledge_graph.extraction.fact_extractor`)
extracts **ontology-typed** entities and **direction-constrained** relations
instead of free snake_case predicates.

The schema *is* the ontology. sift-kg injects a flat YAML schema into its prompt;
we inject our formal OWL classes + ``rdfs:domain/range``, then keep the post-hoc
grounding (:mod:`.ontology_grounding`) and the engine's OWL reasoning downstream —
generation-time guidance *and* reasoning, which a flat schema cannot give.

Design notes:

* **rdflib lives in the ``[owl]`` extra, NOT the serving plane** (KG-2.242). This
  module import-guards rdflib and degrades to ``None`` (free-vocab extraction)
  when it is absent, so the lean serving image is unaffected and ontology
  guidance auto-activates wherever the owl stack is installed (host daemon /
  enterprise profile). This is auto-detection, not a flag (Configuration
  discipline): the enhancement runs when the resource is present.
* The ``.ttl`` files themselves are always-present package data; only the rdflib
  *parser* is optional.
* The TBox default namespace is ``http://knuckles.team/kg#`` (the ``:`` prefix in
  the modules) — distinct from the engine LPG-projection ``au:`` namespace
  (``_AU_NS``), which is for *instance* data (KG-2.240/2.242). We read class/
  property *definitions* here, so the static ``.ttl`` parse is the right path.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)

# knowledge_graph/ — the directory the ontology_*.ttl modules live in.
_KG_DIR = Path(__file__).resolve().parent.parent
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

# The foundational core module — applies to all prose content as the default
# closed vocabulary. Domain-specific source types additionally load their module.
_CORE_MODULES: tuple[str, ...] = ("ontology",)

# Source-type → extra domain ontology module(s), merged with the core. Keyed on
# the substring that identifies the domain in the source_type/connector name, so
# both ``"servicenow"`` and ``"connector:servicenow"`` resolve. Static map, not a
# flag (Configuration discipline). Unmatched prose content uses the core only.
_DOMAIN_MODULES: dict[str, tuple[str, ...]] = {
    "servicenow": ("ontology_servicenow",),
    "leanix": ("ontology_leanix",),
    "legal": ("ontology_legal",),
    "medical": ("ontology_medical",),
    "wellness": ("ontology_wellness",),
    "hr": ("ontology_hr",),
    "finance": ("ontology_banking", "ontology_trading"),
    "banking": ("ontology_banking",),
    "trading": ("ontology_trading",),
    "government": ("ontology_government",),
    "enterprise": ("ontology_enterprise",),
    "infrastructure": ("ontology_infrastructure",),
    "grafana": ("ontology_grafana",),
    "observability": ("ontology_observability",),
    "media": ("ontology_media",),
    "social": ("ontology_social",),
    "calendar": ("ontology_calendar",),
    "energy": ("ontology_energy_geopolitics",),
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


def _module_paths(source_type: str) -> tuple[str, ...] | None:
    """Resolve the ontology module basenames for ``source_type`` (or None to skip)."""
    st = (source_type or "").strip().lower()
    if not st or st in _SKIP_TYPES:
        return None
    modules: list[str] = list(_CORE_MODULES)
    for key, mods in _DOMAIN_MODULES.items():
        if key in st:
            for m in mods:
                if m not in modules:
                    modules.append(m)
    return tuple(modules)


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


def _parse_modules(modules: tuple[str, ...]) -> ExtractionSchema | None:
    """Parse the given ontology modules into an ExtractionSchema (rdflib-guarded)."""
    try:
        import rdflib
    except ImportError:
        # Lean serving plane has no rdflib (KG-2.242) → free-vocab extraction.
        return None

    g = rdflib.Graph()
    parsed_any = False
    for mod in modules:
        path = _KG_DIR / f"{mod}.ttl"
        if not path.exists():
            continue
        try:
            g.parse(str(path), format="turtle")
            parsed_any = True
        except Exception as e:  # noqa: BLE001 — a malformed module never breaks ingest
            logger.debug("extraction_schema: failed to parse %s: %s", path, e)
    if not parsed_any:
        return None

    OWL = rdflib.OWL
    RDFS = rdflib.RDFS
    SKOS = rdflib.Namespace("http://www.w3.org/2004/02/skos/core#")

    def _local(uri: object) -> str:
        s = str(uri)
        if "#" in s:
            return s.rsplit("#", 1)[1]
        return s.rsplit("/", 1)[-1]

    def _label(subj: object) -> str:
        for pred in (RDFS.comment, RDFS.label, SKOS.prefLabel):
            val = g.value(subject=subj, predicate=pred)
            if val:
                return str(val).strip().replace("\n", " ")[:160]
        return ""

    # --- classes -> entity types ---
    entity_types: list[EntityType] = []
    seen_classes: set[str] = set()
    for cls in g.subjects(rdflib.RDF.type, OWL.Class):
        if not str(cls).startswith(_TBOX_NS):
            continue
        local = _local(cls)
        if local in seen_classes:
            continue
        seen_classes.add(local)
        entity_types.append(
            EntityType(
                name=local,
                description=_label(cls),
                synonyms=_synonyms_for(local),
            )
        )

    # --- object properties -> typed relations ---
    relations: list[Relation] = []
    seen_preds: set[str] = set()
    symmetric_props = {
        _local(s) for s in g.subjects(rdflib.RDF.type, OWL.SymmetricProperty)
    }
    for prop in g.subjects(rdflib.RDF.type, OWL.ObjectProperty):
        if not str(prop).startswith(_TBOX_NS):
            continue
        local = _local(prop)
        pred = _camel_to_snake(local)
        if pred in seen_preds:
            continue
        seen_preds.add(pred)
        domain = tuple(
            _local(d)
            for d in g.objects(prop, RDFS.domain)
            if str(d).startswith(_TBOX_NS)
        )
        rng = tuple(
            _local(r)
            for r in g.objects(prop, RDFS.range)
            if str(r).startswith(_TBOX_NS)
        )
        relations.append(
            Relation(
                predicate=pred,
                label=str(g.value(prop, RDFS.label) or "").strip(),
                domain=domain,
                range=rng,
                symmetric=local in symmetric_props,
            )
        )

    if not entity_types and not relations:
        return None

    # Relevance ordering: relations with BOTH endpoints typed first (they carry
    # direction constraints F uses); classes referenced by a relation first.
    referenced: set[str] = set()
    for r in relations:
        referenced.update(r.domain)
        referenced.update(r.range)
    entity_types.sort(key=lambda e: (e.name not in referenced, e.name))
    relations.sort(key=lambda r: (not (r.domain and r.range), r.predicate))

    return ExtractionSchema(
        name="+".join(modules),
        entity_types=tuple(entity_types),
        relations=tuple(relations),
    )


@lru_cache(maxsize=64)
def load_extraction_schema(source_type: str) -> ExtractionSchema | None:
    """Return the ontology-guided extraction schema for ``source_type`` (cached).

    ``None`` means *no ontology guidance* — the extractor falls back to its
    free-vocab prompt unchanged (non-prose content, rdflib absent, or an empty
    parse). Never raises: ingestion must not break on a schema-load failure.
    """
    try:
        modules = _module_paths(source_type)
        if not modules:
            return None
        schema = _parse_modules(modules)
        if schema is None or schema.is_empty:
            return None
        return schema
    except Exception as e:  # noqa: BLE001
        logger.debug("load_extraction_schema(%s) failed: %s", source_type, e)
        return None


__all__ = [
    "EntityType",
    "Relation",
    "ExtractionSchema",
    "load_extraction_schema",
    "CONTENT_TYPE_TO_ONTOLOGY",
]

# Public alias for the content→ontology map (referenced in docs/tests).
CONTENT_TYPE_TO_ONTOLOGY = _DOMAIN_MODULES
