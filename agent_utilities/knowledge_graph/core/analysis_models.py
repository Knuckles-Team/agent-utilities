#!/usr/bin/python
"""Pydantic models for KG analysis structured output.

These models enforce strict schema compliance for LLM extraction results
using pydantic-ai's ``result_type`` feature, which uses grammar-constrained
decoding (``response_format: json_schema``) to guarantee valid JSON output.

Standards Applied:
    - SKOS (Simple Knowledge Organization System) — W3C taxonomy standard
    - SSSOM (Standardized Semantic Sets of Mappings) — Alignment metadata
    - CodeTaxo pattern — LLM-driven code taxonomy expansion (ACL 2024)
"""

from __future__ import annotations

from pydantic import BaseModel, Field

# ──────────────────────────────────────────────────────────────────────
# L2 Synthesis Models
# ──────────────────────────────────────────────────────────────────────


class FeatureRecommendation(BaseModel):
    """A single actionable feature recommendation from L2 synthesis."""

    feature_name: str = Field(description="Name of the recommended feature or concept")
    target_concepts: list[str] = Field(
        default_factory=list,
        description="Concepts this feature enhances or is analogous to",
    )
    implementation_sketch: str = Field(
        default="",
        description="Key classes, methods, and architectural approach",
    )
    expected_impact: str = Field(
        default="", description="Expected benefit and impact assessment"
    )
    integration_complexity: str = Field(
        default="medium",
        description="Integration difficulty: low, medium, or high",
    )
    priority: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Priority ranking from 1 (lowest) to 10 (highest)",
    )


class SynthesisResult(BaseModel):
    """Structured output from L2 LLM synthesis.

    Used as ``result_type`` for pydantic-ai Agent to enforce
    grammar-constrained JSON output from the LLM.
    """

    recommendations: list[FeatureRecommendation] = Field(
        default_factory=list,
        description="List of actionable feature recommendations",
    )


# ──────────────────────────────────────────────────────────────────────
# L3 Deep Extraction Models
# ──────────────────────────────────────────────────────────────────────


class DeepExtraction(BaseModel):
    """Structured knowledge extraction for a single high-weight match."""

    source_name: str = Field(
        default="", description="Name of the source paper/codebase analyzed"
    )
    algorithms: list[str] = Field(
        default_factory=list,
        description="Key algorithms or techniques identified",
    )
    data_structures: list[str] = Field(
        default_factory=list,
        description="Novel data structures introduced",
    )
    patterns: list[str] = Field(
        default_factory=list,
        description="Architectural patterns adoptable for the target system",
    )
    integration_blueprint: str = Field(
        default="",
        description="Concrete integration plan for the target agent framework",
    )


class DeepExtractionResult(BaseModel):
    """Structured output from L3 deep extraction (batched).

    Supports batching multiple high-weight matches in a single LLM call
    to minimize inference requests.
    """

    extractions: list[DeepExtraction] = Field(
        default_factory=list,
        description="Deep extraction results for each analyzed match",
    )


# ──────────────────────────────────────────────────────────────────────
# SKOS-Inspired Concept Models (Universal Concept Extraction)
# ──────────────────────────────────────────────────────────────────────


class ConceptNode(BaseModel):
    """A concept within a SKOS-inspired concept scheme.

    Follows W3C SKOS vocabulary:
    - prefLabel: Primary human-readable label
    - broader: Parent concept(s) in the hierarchy
    - narrower: Child concept(s)
    - related: Non-hierarchical related concepts
    """

    concept_id: str = Field(description="Unique identifier for the concept")
    pref_label: str = Field(description="Primary human-readable label (skos:prefLabel)")
    description: str = Field(default="", description="Brief description of the concept")
    broader: list[str] = Field(
        default_factory=list,
        description="Parent concept IDs (skos:broader)",
    )
    narrower: list[str] = Field(
        default_factory=list,
        description="Child concept IDs (skos:narrower)",
    )
    related: list[str] = Field(
        default_factory=list,
        description="Related concept IDs (skos:related)",
    )
    level: int = Field(
        default=0,
        description="Hierarchy depth (0 = top-level pillar/theme)",
    )
    extraction_method: str = Field(
        default="structural",
        description="How this concept was extracted: structural, llm, manual",
    )


class ConceptScheme(BaseModel):
    """A SKOS ConceptScheme representing a project's concept taxonomy.

    Each ingested codebase produces one ConceptScheme with hierarchical
    broader/narrower relationships auto-inferred from package structure
    and semantically clustered by LLM.

    Standards:
        - SKOS: https://www.w3.org/TR/skos-reference/
        - SSSOM: https://mapping-commons.github.io/sssom/
        - CodeTaxo: ACL 2024 taxonomy expansion
    """

    scheme_id: str = Field(
        description="Unique scheme identifier (typically project name)"
    )
    title: str = Field(description="Human-readable scheme title")
    concepts: list[ConceptNode] = Field(
        default_factory=list,
        description="All concepts in this scheme",
    )
    top_concepts: list[str] = Field(
        default_factory=list,
        description="IDs of top-level concepts (skos:hasTopConcept)",
    )
