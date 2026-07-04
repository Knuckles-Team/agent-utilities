---
name: kg-coverage-doctor
description: >-
  Audits that the graph-os MCP verb surface and the kg-* skill suite stay 1:1 — every
  graph_* / engine_* / ontology_* verb registered by the graph-os server is wrapped by a
  discoverable kg-* skill, and no kg-* skill points at a dead verb. Use when adding a new
  MCP tool/verb, authoring or renaming a kg-* skill, or when asked to "check skill coverage",
  "audit mcp tool coverage", "run the coverage doctor", or verify the skill↔verb contract.
  Replaces the former mcp_tool_coverage_audit workflow.
license: MIT
tags: [graph-os, mcp, skills, coverage, gate, audit, parity]
tier: meta
metadata:
  author: Genius
  version: '0.1.0'
---

# kg-coverage-doctor

Keeps the operator-facing **kg-\* skill suite** in lockstep with the **graph-os verb
surface**. This is the *third parity leg* (CONCEPT:OS-5.80), complementing the two
tool⇄REST-route legs already enforced by `tests/unit/test_gateway_mcp_parity.py`:

1. **Coverage** — every verb in `kg_server.REGISTERED_TOOLS` (minus a tiny, justified
   `INTENTIONALLY_UNSKILLED` set) is wrapped by at least one `kg-*` skill.
2. **No orphans** — every `tier: core|modality` `kg-*` skill maps to a real verb, via its
   slug (`kg-<x>` → `graph_<x>`) or an explicit `wraps: [verb, ...]` frontmatter list.
3. **Exemptions** — skills tagged `tier: meta|surface` (routers, builders, webui) are not
   verb wrappers and are excluded from both checks.

## Naming contract
`kg-<capability>` where `<capability>` = the MCP verb minus `graph_`, with `_`→`-`
(`graph_ontology` → `kg-ontology`). A skill fronting several verbs declares them
explicitly, e.g. `kg-ingest` → `wraps: [graph_ingest, source_sync, source_drain,
source_connector, document_process]`; each `kg-modality-*` wraps its `engine_*` domains.

## Execution
Run the doctor directly (exit 0 = green, exit 1 = drift with a per-verb/skill report):

```bash
python -m agent_utilities.mcp.skill_coverage
```

Or as the CI/pre-commit gate (same core):

```bash
pytest tests/unit/test_gateway_mcp_parity.py -q      # legs 1–3
pre-commit run guardrail-kg-skill-coverage --all-files
```

The logic lives in `agent_utilities/mcp/skill_coverage.py` (`compute_coverage()` →
`CoverageReport(uncovered, orphans, bad_tiers)`), reused by both the test and this skill.

## Fixing drift
- **Verb with no skill** → author `kg-<slug>` (use `skill-builder`), or add it to an
  existing skill's `wraps:` list, or (rarely) add it to `INTENTIONALLY_UNSKILLED` with a reason.
- **Orphan skill** → correct the slug/`wraps:`, or tag it `tier: meta|surface` if it is not
  a verb wrapper.

> **Execution:** If graph-os is reachable, this is a read-only audit; run the CLI above
> natively. No orchestration needed.
