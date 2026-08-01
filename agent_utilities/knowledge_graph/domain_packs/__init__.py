"""Domain-pack framework + declarative mapping DSL (CONCEPT:AU-KG.ingest.domain-pack-framework,
CONCEPT:AU-KG.ingest.mapping-dsl).

Makes a corpus **configured, not coded**: a versioned, installable/removable
*domain pack* (ontology extension + SHACL shapes + mappings + evaluation
cases, see :mod:`domain_pack`) plus a *declarative mapping DSL* (:mod:`dsl`)
that turns predictable source structure — frontmatter, tables, headings,
links, JSON fields/API records — into graph facts with **no custom code per
corpus**.

Reuses rather than reinvents (see each module's docstring for detail):

* the connector-manifest primitives (:mod:`..ontology.connector_manifest`,
  ``manifest_compiler``, ``ontology_integrity``) for the pack's ontology
  extension, integrity hash, and anti-sprawl enforcement;
* the ``mcp_tool`` declarative-preset *pattern* (never its code) as the model
  for a mapping rule: data, not a bespoke connector module;
  the existing governed promotion path (``ingestion.envelope_ingest.ingest_graph_slice``)
  as the ONLY way a pack's proposed facts can ever reach the graph — a pack
  has no write authority of its own.

**Surveyed and deliberately NOT unified with** ``models.schema_pack.SchemaPack``
— that primitive tunes retrieval/type-activation over the fixed, hardcoded
``RegistryNodeType``/``RegistryEdgeType`` enum as ONE process-wide active
profile; this package is the orthogonal axis (a genuinely new, versioned,
hash-verified, MULTI-COEXISTING per-corpus ontology + extraction layer). See
:mod:`domain_pack`'s module docstring for the full reasoning, including why
entity-identity rules stay on ``SchemaPack`` (``IdentityRule``,
Track 5/entity-resolution) rather than being re-declared here.
"""

from __future__ import annotations
