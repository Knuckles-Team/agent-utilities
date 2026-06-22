---
name: concept-node-extractor
description: >
  Knowledge Graph concept extraction atomic skill. Parses text and structures
  definitions and relations for Graph-OS.
domain: research
license: MIT
tags: [graph-os, semantic, parsing, concept, research]
metadata:
  author: Genius
  version: '0.1.0'
requires:
  - graph_write
---

# Concept Node Extractor Skill

Stateless atomic operation to ingest unstructured literature reviews, academic transcripts, or enterprise systems specifications, parse out core terms and relationships, map them to ontological definitions, and compile transaction-safe Cypher queries for ingestion into the Graph-OS Knowledge Graph.

## Prerequisites

- `graph_write` — for transactional database writes and node/edge merging within the unified Graph-OS ecosystem.

## Steps

### Step 1: parse_unstructured_input
Analyze unstructured input text to locate relevant architectural or scientific domain concepts:
- Scan raw text inputs:
  - Process textual payloads (e.g., transcripts, academic papers, system design documents).
  - Use regex patterns, keyword match dictionaries, or natural language semantic parsing to identify candidates for new concepts, ontology items, systems definitions, or acronyms.
- Extract descriptive attributes:
  - Extract the term name, associated definition snippets, parent domain category, context references, and confidence rating.
- Output parameters:
  - `extracted_terms`: List of parsed concept dictionaries, each containing `name`, `definition`, `domain`, and `raw_context`.

### Step 2: extract_conceptual_triples [depends_on: parse_unstructured_input]
Structure the extracted concepts into standardized ontological nodes and directed relationships:
- Define node schemas:
  - **Concept Node**:
    - `id`: Unique UPPERCASE term identifier string (e.g., `CONCEPT:ORCH-1.2`, `CONCEPT:SYS-2.0`).
    - `name`: Clean term label.
    - `description`: Formal synthesized semantic definition.
    - `domain`: Research or technical area identifier.
  - **Reference Node**:
    - `id`: Hashed source context ID (e.g., `REF:abc123xyz`).
    - `context`: Original raw text snippet.
- Map directed edges/relationships:
  - Define taxonomic lineage: `(Concept)-[:SUB_CLASS_OF]->(ParentConcept)`.
  - Define semantic support: `(Concept)-[:DEFINED_IN]->(Reference)`.
- Output parameters:
  - `nodes`: List of formatted Node objects.
  - `relationships`: List of formatted Edge/Relationship objects.

### Step 3: compile_cypher_transaction [depends_on: extract_conceptual_triples]
Transform the structured graph components into standardized, idempotent Cypher queries:
- Draft transaction queries:
  - For each concept node, construct a parameterized `MERGE` query:
    `MERGE (c:Concept {id: $id}) ON CREATE SET c.name = $name, c.description = $description, c.domain = $domain ON MATCH SET c.description = $description`
  - For each reference node, construct a parameterized `MERGE` query:
    `MERGE (r:Reference {id: $id}) ON CREATE SET r.context = $context`
  - For each relationship, construct a parameterized `MERGE` query:
    `MATCH (a:Concept {id: $source_id}), (b:Concept {id: $target_id}) MERGE (a)-[r:SUB_CLASS_OF]->(b)` (or other relationship types).
- Validate transaction security:
  - Verify queries are read-only-safe in execution templates and strictly construct parameter-backed bindings to prevent Cypher injection vulnerabilities.
- Output parameters:
  - `status`: "SUCCESS" or "FAILED"
  - `cypher_payload`: Structured payload containing lists of Cypher query strings and their corresponding argument parameters ready for execution via `graph_write`.
