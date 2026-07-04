"""Ontology-aware schema discovery (CONCEPT:AU-KG.ontology.do-not-auto-merge).

Beyond sift-kg's flat-YAML discovery (``domains/discovery.py``): sample documents,
ask the LLM to propose entity/relation types, then **diff against the existing OWL
ontology** and emit proposed ``.ttl`` *extensions* for the genuinely-missing
classes/relations — feeding the concept-reservation + evolution pipeline. We never
auto-merge: a new top-level ``.ttl`` is a build break (sprawl rule), so the output
is a *proposal* fragment with ``RESERVE-PENDING`` concept placeholders that a human
(or the evolution loop) reviews, reserves, and lands into the existing domain
module after the valid/connected/SHACL gate (KG-2.112).

Each candidate is classified against the live ontology (via the KG-2.255 schema +
the grounding synonym lexicon) as **covered** (already an OWL class/property),
**synonym** (maps to an existing class), or **missing** (a real extension
candidate). Only ``missing`` candidates reach the ``.ttl`` fragment.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

LLMFn = Callable[[str], str]

_TBOX_NS = "http://knuckles.team/kg#"
_MAX_SAMPLE_CHARS = 3000
_MAX_SAMPLES = 5


@dataclass
class DiscoveredType:
    """One LLM-proposed type, classified against the live ontology."""

    name: str
    kind: str  # "class" | "property"
    description: str = ""
    domain: str = ""
    range: str = ""
    classification: str = "missing"  # "covered" | "synonym" | "missing"
    existing_match: str = ""


def build_discovery_prompt(sample_texts: list[str]) -> str:
    """Render the schema-discovery prompt (sift-kg ``discovery.py`` parity)."""
    samples = "\n\n---\n\n".join(
        t[:_MAX_SAMPLE_CHARS] for t in sample_texts[:_MAX_SAMPLES]
    )
    return (
        "You are a knowledge-graph ontologist. From the document samples below, "
        "design the entity and relation types a graph of this domain needs.\n"
        "- 5-15 entity types (UpperCamelCase nouns: Organization, ClinicalTrial).\n"
        "- 8-20 relation types (snake_case verbs with an expected subject→object "
        "direction: employs: Organization→Person).\n\n"
        f"Samples:\n{samples}\n\n"
        "Return ONLY JSON:\n"
        '{"entity_types": [{"name": "...", "description": "..."}], '
        '"relation_types": [{"name": "...", "domain": "...", "range": "...", '
        '"description": "..."}]}'
    )


def parse_discovery_response(
    raw: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Parse ``(entity_types, relation_types)`` from the LLM JSON (lenient)."""
    if not raw:
        return [], []
    text = raw.strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return [], []
    try:
        obj = json.loads(text[start : end + 1])
    except (json.JSONDecodeError, TypeError):
        return [], []
    ents = obj.get("entity_types") or []
    rels = obj.get("relation_types") or []
    return (
        [e for e in ents if isinstance(e, dict) and e.get("name")],
        [r for r in rels if isinstance(r, dict) and r.get("name")],
    )


def _camel_to_snake(name: str) -> str:
    s = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s)
    return s.lower().replace(" ", "_")


def classify_candidate(
    name: str,
    kind: str,
    existing_class_names: set[str],
    existing_predicates: set[str],
    synonyms: dict[str, str],
) -> tuple[str, str]:
    """Classify a candidate as covered / synonym / missing against the ontology."""
    if kind == "class":
        low = name.strip().lower()
        if low in existing_class_names:
            return "covered", name
        target = synonyms.get(low)
        if target:
            return "synonym", target
        # a synonym surface whose canonical maps to an existing class
        if target in existing_class_names:
            return "synonym", target or ""
        return "missing", ""
    # property
    pred = _camel_to_snake(name)
    if pred in existing_predicates:
        return "covered", pred
    return "missing", ""


def discover_schema_extensions(
    sample_texts: list[str],
    source_type: str,
    llm_fn: LLMFn | None,
) -> list[DiscoveredType]:
    """Propose types from samples, classified against the live ontology.

    Returns all candidates with their classification; callers filter to
    ``classification == "missing"`` for the ``.ttl`` proposal. Never raises.
    """
    if llm_fn is None or not sample_texts:
        return []
    try:
        raw = llm_fn(build_discovery_prompt(sample_texts))
    except Exception:  # noqa: BLE001
        return []
    ents, rels = parse_discovery_response(raw)
    if not ents and not rels:
        return []

    # Existing vocabulary from the KG-2.255 schema + the grounding synonym lexicon.
    existing_class_names: set[str] = set()
    existing_predicates: set[str] = set()
    try:
        from .extraction_schema import load_extraction_schema

        schema = load_extraction_schema(source_type) or load_extraction_schema(
            "document"
        )
        if schema is not None:
            existing_class_names = {e.name.lower() for e in schema.entity_types}
            existing_predicates = set(schema.closed_predicate_set)
    except Exception:  # noqa: BLE001
        pass
    synonyms: dict[str, str] = {}
    try:
        from .ontology_grounding import _RAW_CLASS_SYNONYMS

        synonyms = dict(_RAW_CLASS_SYNONYMS)
    except Exception:  # noqa: BLE001
        pass

    out: list[DiscoveredType] = []
    for e in ents:
        cls, match = classify_candidate(
            str(e["name"]), "class", existing_class_names, existing_predicates, synonyms
        )
        out.append(
            DiscoveredType(
                name=str(e["name"]),
                kind="class",
                description=str(e.get("description", "")),
                classification=cls,
                existing_match=match,
            )
        )
    for r in rels:
        cls, match = classify_candidate(
            str(r["name"]),
            "property",
            existing_class_names,
            existing_predicates,
            synonyms,
        )
        out.append(
            DiscoveredType(
                name=str(r["name"]),
                kind="property",
                description=str(r.get("description", "")),
                domain=str(r.get("domain", "")),
                range=str(r.get("range", "")),
                classification=cls,
                existing_match=match,
            )
        )
    return out


def to_ttl_fragment(discovered: list[DiscoveredType]) -> str:
    """Emit a proposal Turtle fragment for the *missing* candidates only.

    Concept ids are emitted as ``RESERVE-PENDING`` placeholders (NEVER hardcoded) —
    the evolution pipeline reserves a real id via the flock ledger (OS-5.42) before
    the class lands in the domain module. The fragment is a *proposal artifact*, not
    a new ontology file (sprawl rule).
    """
    missing = [d for d in discovered if d.classification == "missing"]
    if not missing:
        return ""

    def _local(text: str) -> str:
        return re.sub(r"\W", "", text or "")

    lines = [
        "# PROPOSED ontology extension — review + reserve concept ids before landing.",
        "# Generated by schema_discovery (CONCEPT:AU-KG.ontology.do-not-auto-merge). Do NOT auto-merge.",
        f"@prefix : <{_TBOX_NS}> .",
        "@prefix owl: <http://www.w3.org/2002/07/owl#> .",
        "@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .",
        "",
    ]
    for d in missing:
        if d.kind == "class":
            local = _local(d.name)
            if not local:
                continue
            lines.append("# CONCEPT:RESERVE-PENDING")
            lines.append(f":{local} a owl:Class ;")
            lines.append(f'    rdfs:label "{d.name}" ;')
            if d.description:
                lines.append(f'    rdfs:comment "{d.description}" ;')
            lines.append("    rdfs:subClassOf owl:Thing .")
            lines.append("")
        else:
            pred = _local(d.name[:1].lower() + d.name[1:])
            if not pred:
                continue
            stmt = [f":{pred} a owl:ObjectProperty ;", f'    rdfs:label "{d.name}"']
            if d.domain:
                stmt[-1] += " ;"
                stmt.append(f"    rdfs:domain :{_local(d.domain)}")
            if d.range:
                stmt[-1] += " ;"
                stmt.append(f"    rdfs:range :{_local(d.range)}")
            stmt[-1] += " ."
            lines.append("# CONCEPT:RESERVE-PENDING")
            lines.extend(stmt)
            lines.append("")
    return "\n".join(lines)


def discovery_report(discovered: list[DiscoveredType]) -> dict[str, Any]:
    """A JSON-able summary: counts by classification + the proposed .ttl fragment."""
    by_class: dict[str, int] = {}
    for d in discovered:
        by_class[d.classification] = by_class.get(d.classification, 0) + 1
    return {
        "candidates": [
            {
                "name": d.name,
                "kind": d.kind,
                "classification": d.classification,
                "existing_match": d.existing_match,
                "domain": d.domain,
                "range": d.range,
            }
            for d in discovered
        ],
        "counts": by_class,
        "ttl_proposal": to_ttl_fragment(discovered),
    }


__all__ = [
    "DiscoveredType",
    "build_discovery_prompt",
    "parse_discovery_response",
    "classify_candidate",
    "discover_schema_extensions",
    "to_ttl_fragment",
    "discovery_report",
]
