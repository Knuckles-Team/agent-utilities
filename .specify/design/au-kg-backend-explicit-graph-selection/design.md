# Design Document: Explicit graph selection, separated from backend-connection routing

CONCEPT:AU-KG.backend.explicit-graph-selection

> `agent_utilities/mcp/kg_server.py`, `agent_utilities/mcp/tools/query_tools.py`,
> `agent_utilities/mcp/tools/write_ingest_tools.py`,
> `agent_utilities/security/error_surface.py`, pinned by
> `tests/unit/mcp/test_graph_explicit_selection.py` and
> `tests/unit/mcp/test_graph_tool_targeting.py`.

## KG Analysis (Required)

### Nearest Existing Concepts

| Concept ID | Name | Similarity | Pillar |
|---|---|---|---|
| `AU-KG.backend.multi-connection-registry` | named multi-connection registry (WHICH backend connection serves a call) | med | KG |
| `AU-KG.backend.connection-registry` | role-aware registry + live config mutation for those connections | med | KG |

### Extension Analysis

- **Primary Extension Point**: the multi-connection registry's `target`
  parameter, which already lets a call choose WHICH backend connection to run
  against.
- **Extension Strategy**: specialize — a call needs a second, orthogonal
  choice (WHICH graph on the chosen connection), which the registry's
  `target` alone does not express.
- **New Concept Required?**: Yes — the commit introducing this concept names
  the split explicitly: "separate physical-graph selection from
  backend-connection routing".

## Problem

`target=` (the multi-connection registry) answers "which backend connection"
(e.g. `prod-neo4j` vs `pg-main`). It does not answer "which physical graph on
that connection" — a single connection can host more than one named graph,
and tools that read/write graph data (`query_tools.py`, `write_ingest_tools.py`)
need a second, independent selector for that axis. Conflating the two would
make a caller's connection choice silently also decide (or fail to decide)
its graph choice.

## Decision

Give graph selection its own explicit parameter/resolution path, kept fully
separate from backend-connection `target` routing. `kg_server.py` resolves
the requested graph before dispatch; `query_tools.py`/`write_ingest_tools.py`
thread it through every read/write entrypoint; `error_surface.py` reports an
unresolvable graph as its own distinct error rather than folding it into a
generic connection error. This keeps "which connection" and "which graph on
that connection" as two independently answerable questions, matching how the
multi-connection registry itself models one connection hosting several named
graphs.

## Wire-First

Landed in `fix(kg): separate physical-graph selection from
backend-connection routing` (`800c86b1`); pinned by
`tests/unit/mcp/test_graph_explicit_selection.py` and
`tests/unit/mcp/test_graph_tool_targeting.py`.
