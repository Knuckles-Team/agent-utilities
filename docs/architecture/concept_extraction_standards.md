# Knowledge Graph Ingestion — Concept Extraction Standards

This document describes the industry-standard methodologies used by the
`agent-utilities` Knowledge Graph to automatically extract, organize, and
align concepts from ingested codebases and research papers.

## Standards Applied

### SKOS — Simple Knowledge Organization System (W3C)

**Specification**: [W3C SKOS Reference](https://www.w3.org/TR/skos-reference/)

SKOS is the W3C standard for representing taxonomies, classification schemes,
and controlled vocabularies in RDF. We use SKOS vocabulary semantics for
concept relationships within the Knowledge Graph:

| SKOS Property | KG Edge Type | Meaning |
|--------------|-------------|---------|
| `skos:prefLabel` | `name` property | Primary human-readable label |
| `skos:broader` | `BROADER` edge | Parent concept in hierarchy |
| `skos:narrower` | `NARROWER` edge | Child concept in hierarchy |
| `skos:related` | `RELATED_CONCEPT` edge | Non-hierarchical association |
| `skos:ConceptScheme` | `scheme` property | The project this concept belongs to |
| `skos:hasTopConcept` | Top-level concept nodes | Root concepts of a scheme |

#### ConceptScheme Per Project

Each ingested codebase produces one `ConceptScheme` identified by the project name.
For example, `agent-utilities` produces the scheme `agent-utilities` with its 5 pillar
top concepts: `ORCH`, `KG`, `AHE`, `ECO`, `OS`.

External codebases (e.g., `networkx`, `pydantic-ai`) produce their own concept schemes
with hierarchies auto-generated from package structure.

### SSSOM — Standardized Semantic Sets of Mappings

**Specification**: [SSSOM GitHub](https://mapping-commons.github.io/sssom/)

SSSOM is the standard for representing concept mappings between different ontologies
or concept schemes. We use SSSOM-style metadata on `ANALOGOUS_TO` edges:

| SSSOM Field | KG Property | Purpose |
|------------|------------|---------|
| `subject_id` | Source node ID | Concept from source scheme |
| `object_id` | Target node ID | Concept from target scheme |
| `predicate_id` | Edge type | Relationship type (e.g., `ANALOGOUS_TO`) |
| `mapping_justification` | `source` property | How the mapping was discovered |
| `confidence` | `score` property | Similarity/confidence score |

### CodeTaxo — LLM-Driven Code Taxonomy Expansion (ACL 2024)

CodeTaxo is a methodology for using LLMs to expand taxonomies from codebases.
The key insight is that code-language prompts (docstrings, function names,
module structure) provide strong signals for concept hierarchies.

Our 3-tier extraction pipeline implements the CodeTaxo approach:

## Concept Extraction Pipeline

### Tier 1: Structural Extraction (No LLM Required)

Parses the codebase structure to extract initial concept nodes:

- **Package tree** → Top-level hierarchy (e.g., `agent_utilities/` → `agent-utilities` scheme)
- **Module names** → Mid-level concepts (e.g., `knowledge_graph/` → "Knowledge Graph")
- **Class/function names** → Leaf concepts from public API
- **README sections** → Theme extraction
- **`CONCEPT:ID` tags** → Explicit concept declarations in code comments

**Output**: `Concept` nodes with `extraction_method: "structural"`, connected by
`BROADER`/`NARROWER` edges reflecting the package hierarchy.

### Tier 2: Semantic Clustering (LLM-Powered)

For codebases with >20 modules, uses the LLM to cluster related modules into
higher-level domain concepts:

1. Embed all module names + docstrings
2. Cluster via semantic similarity
3. LLM names each cluster (e.g., "authentication", "data pipeline", "orchestration")
4. Creates `Concept` nodes at `level=0` (top-level themes) with `extraction_method: "llm"`

**Pydantic Schema Enforcement**: Uses `pydantic-ai` `output_type` for grammar-constrained
JSON decoding — zero regex parsing, zero JSON errors.

### Tier 3: Cross-Reference Alignment (LLM-Powered)

Compares extracted concepts against existing KG concepts and research paper topics:

1. For each new concept, find similar existing `Concept` and `Article` nodes
2. Create `ANALOGOUS_TO` edges with SSSOM-style metadata
3. Discover cross-domain innovations via analogy matching

## OWL Reasoning Integration

After concept extraction and alignment, the OWL reasoning engine runs a
promote → reason → downfeed cycle:

1. **Promote**: Exports KG edges to the OWL backend
2. **Reason**: Applies transitive and symmetric closure rules
3. **Downfeed**: Writes inferred relationships back to the KG

This discovers implicit relationships that weren't directly extracted. For
example, if `A ANALOGOUS_TO B` and `B ANALOGOUS_TO C`, the reasoner infers
`A ANALOGOUS_TO C`.

### Triggering OWL Reasoning

```
# Via MCP
kg_inspect(view="owl_cycle")

# Automatically as part of full_pipeline
kg_analyze(query="...", action="full_pipeline")
```

## Pydantic Models

All LLM extraction uses strict Pydantic models for guaranteed valid output:

```python
# L2 Synthesis
class FeatureRecommendation(BaseModel):
    feature_name: str
    target_concepts: list[str]
    implementation_sketch: str
    expected_impact: str
    integration_complexity: str  # low/medium/high
    priority: int               # 1-10

class SynthesisResult(BaseModel):
    recommendations: list[FeatureRecommendation]

# L3 Deep Extraction
class DeepExtraction(BaseModel):
    source_name: str
    algorithms: list[str]
    data_structures: list[str]
    patterns: list[str]
    integration_blueprint: str

class DeepExtractionResult(BaseModel):
    extractions: list[DeepExtraction]
```

These models are defined in:
`agent_utilities/knowledge_graph/core/analysis_models.py`

## MCP Tool Reference

| Tool | Action/View | Layer | LLM Required |
|------|-------------|-------|-------------|
| `kg_search` | `mode='discover'` | L1 | No |
| `kg_analyze` | `action='synthesize'` | L1+L2 | Yes |
| `kg_analyze` | `action='deep_extract'` | L1+L2+L3 | Yes |
| `kg_analyze` | `action='full_pipeline'` | L1+L2+L3+OWL | Yes |
| `kg_analyze` | `action='background_research'` | Async L1+L2+L3 | Yes |
| `kg_inspect` | `view='owl_cycle'` | OWL only | No |
| `kg_inspect` | `view='build_indexes'` | HNSW index mgmt | No |

> **See also**: [Vector Index Lifecycle](vector_index_lifecycle.md) for details
> on HNSW index management, the drop→ingest→build cycle, and search performance tiers.

## Environment Configuration

The KG MCP server relies on the unified `agent-utilities` configuration for LLM connectivity. Ensure your `~/.config/agent-utilities/config.json` is configured:

```json
{
  "chat_models": [
    {
      "id": "qwen/qwen3.5-9b",
      "provider": "openai",
      "intelligence_level": "normal"
    }
  ],
  "embedding_models": [
    {
      "id": "text-embedding-nomic-embed-text-v2-moe",
      "provider": "openai",
      "base_url": "http://10.0.0.18:1234/v1"
    }
  ]
}
```

Authentication is handled via the `.env` file (e.g., `LLM_API_KEY`).
