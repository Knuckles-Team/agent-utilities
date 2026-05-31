# KG Schema Extensions: Research Assimilation

> **CONCEPT:KG-2.6** — Research Intelligence Extensions

This document describes the node and edge types added to support research-to-feature linkage and assimilation tracking across any codebase.

## New Node Types

### SDDFeature

Represents a feature specification derived from research analysis.

| Property | Type | Description |
|----------|------|-------------|
| `id` | string | Unique identifier (e.g., `trace-abc1234567`) |
| `name` | string | Feature name (e.g., `kg-2.1-memory-synthesis`) |
| `concept_ids` | string[] | Related CONCEPT:IDs (e.g., `["KG-2.1"]`) |
| `research_sources` | string[] | Source paper paths |
| `status` | enum | `DRAFT` → `SPEC` → `IN_PROGRESS` → `COMPLETED` |
| `sdd_path` | string | Path to `.specify/` directory |
| `codebase` | string | Target codebase path |

### Codebase

Anchor node representing an ingested codebase.

| Property | Type | Description |
|----------|------|-------------|
| `id` | string | SHA-256 hash prefix of codebase path |
| `codebase` | string | Full filesystem path |
| `name` | string | Display name |

## New Edge Types

### ASSIMILATED_INTO

**Direction**: Article → Codebase

Indicates that a research paper's findings have been reviewed and integrated into a codebase's features.

| Property | Type | Description |
|----------|------|-------------|
| `status` | enum | `reviewed` → `assimilated` → `implemented` |
| `assimilation_date` | ISO8601 | When the assimilation occurred |
| `sdd_feature_id` | string | Related SDDFeature node ID |
| `concept_ids` | string | JSON-encoded list of related concept IDs |
| `codebase` | string | Target codebase path (for disambiguation) |

### DERIVED_FROM_RESEARCH

**Direction**: SDDFeature → Article

Links an SDD feature specification back to the research paper(s) that inspired it.

| Property | Type | Description |
|----------|------|-------------|
| `sdd_feature` | string | Feature name for readability |

### IMPLEMENTS_FINDING (Future)

**Direction**: Code → SDDFeature

Links implementation code back to the feature spec it implements. Created during code-enhancer audits.

## Auto-Detection Mechanics

When `kg_trace(action='submit_sdd')` receives a payload with `status='COMPLETED'`:

1. All `research_sources` paths are resolved to Article nodes
2. `DERIVED_FROM_RESEARCH` edges are created (SDDFeature → Article)
3. For each linked Article, an `ASSIMILATED_INTO` edge is created (Article → Codebase)
4. Edge `status` is set to `implemented`
5. The process is **idempotent** — re-submitting COMPLETED doesn't create duplicates

## Multi-Codebase Support

The `codebase` property on `ASSIMILATED_INTO` edges enables multi-codebase tracking. The same paper can be:
- `implemented` for `agent-utilities`
- `reviewed` for `scholarx`
- Not yet linked for `repository-manager`

The `exclude_assimilated` filter in `discover_innovations()` is per-codebase.
