# Agent-utilities architecture reference

Deep reference for `agent-utilities-development`: the six target layers with the
existing packages assigned to each and their measured violation counts, the API
gateway / MCP surface note, the per-package size tables, the measured import SCC
(both scopes), and the named reverse edges the refactor program is cutting. The
parent [`SKILL.md`](../SKILL.md) keeps the guardrails, the anti-sprawl checklist
and the workflow; this file is the lookup you open before creating a module.

### The six target layers

The dependency order is `contracts -> domain -> ports -> application -> adapters
-> composition`, per
`plans/refactor/evidence/reports/AU-SCC-DECOMPOSITION.md` "Cycle-break target"
(:40-46).

★ **The two program documents disagree about where `ports` sits.**
`plans/refactor/DESIGN.md` "AU target layers" (:61-72) numbers **Application
third and Ports fourth** — `contracts -> domain -> application -> ports ->
adapters -> composition`. This skill follows the **SCC report**; cite that one,
not `DESIGN.md`, until the disagreement is resolved upstream.

★ **Every `plans/…` path cited in this skill resolves at the WORKSPACE root**
(`<workspace>/plans/…`), **not inside this repository.** `agent-utilities/plans/`
holds only an untracked, near-empty `complex/` stub, so `git log` here returns
zero commits for any of them — expected, and *not* evidence a citation is
fabricated. That exact absence-of-evidence was misread on 2026-08-25 and valid
provenance was deleted from ~11 sites before it was caught (this repo's
`AGENTS.md`, "Provenance citations — `reports/*.md` resolves OUTSIDE this repo").

★ **These are target layers, not directories.** Verified by listing
`agent_utilities/`: there is **no** `contracts/`, `ports/`, `adapters/`, or
`composition/` package today. The mapping below assigns *existing* packages by
responsibility; treat the "may NOT import" column as the rule you are held to,
not as something the tree already expresses.

| Layer | Owns | Existing AU packages (measured) | May NOT import |
|---|---|---|---|
| **Contracts** | Versioned models, identifiers, errors, events, query/ingest types. Pure construction, no side effects. | `models/` (37), parts of `protocols/` | Anything above. No model-side registration, no `knowledge_graph` (C3 target: zero production `models -> knowledge_graph`) |
| **Domain** | Invariants and policies. Identity, RBAC decisions, guardrails, governance rules. | `security/` (44), `governance/` (8), `policies/` (1), parts of `core/` | Transport, process, filesystem, vendor SDKs. C4 target: zero `security -> knowledge_graph` production edges |
| **Ports** | Protocols for engine, sources, identity, secrets, clock, messaging, telemetry. | `protocols/source_connectors/`, `core/execution/` (`ExecutionEngine` Protocol) — ★ `protocols/` is **mixed**: `a2a*.py`, `acp_adapter.py`, `agui_emitter.py` are adapters, not ports | Concrete backends |
| **Application** | Use cases coordinating domain + ports. The graph facade and the orchestrator. | `knowledge_graph/` (607 — `core/` 99, `enrichment/` 92, `retrieval/` 48, `ontology/` 43), `orchestration/` (54), `graph/` (78), `harness/` (86), `capabilities/` (22), `domains/` (55) | **TARGET, not today's tree, and ungated:** no application module should import `mcp/`, `gateway/`, or `server/`. Measured 2026-09-03: **62 application→adapter import statements across 14 package pairs** — `knowledge_graph -> mcp` 16, `orchestration -> mcp` 12, `orchestration -> messaging` 9, `orchestration -> runtime` 7, `capabilities -> mcp` 3, `graph -> mcp` 3, then 2 or fewer each for `knowledge_graph -> messaging`/`runtime`/`gateway`/`deployment`/`ingestion` and `harness -> server`/`mcp`/`runtime` |
| **Adapters** | REST, MCP, CLI, source connectors, messaging surfaces, deployment integrations. | `gateway/` (94), `mcp/` (73), `server/` (22), `messaging/` (42), `cli/` (3), `ingestion/` (10), `deployment/` (17), `runtime/` (19) | **TARGET, not today's tree, and ungated:** each other — no adapter-to-adapter call, no adapter as a second graph authority. Measured 2026-09-03: **85 adapter→adapter import statements across 19 package pairs**, led by `gateway -> mcp` 19, `deployment -> mcp` 12, `mcp -> messaging` 8, `mcp -> deployment` 8, `server -> mcp` 7, `mcp -> runtime` 6, `server -> gateway` 5, `mcp -> gateway` 3 |
| **Composition** | The only place that selects concrete adapters and reads deployment config. | `__main__.py`, `server/app.py` (`build_agent_app`), `core/config.py` | Nothing may import composition |

★ **The API gateway and the graph-os MCP surface live INSIDE this repository.**
They are not separate repos and there is nothing to go looking for:

- `agent_utilities/gateway/` — **94 tracked `.py` files**. `graph_api.py`,
  `ontology_api.py`, `registry_api.py`, `research_api.py`, `usage_api.py`,
  `artifacts_api.py`, `remote_oauth_api.py`, plus `aggregator.py`, `fleet.py`,
  `rate_limit.py`, `registry.py`, `schemas/`, `widgets/`.
- `agent_utilities/mcp/` — **73 tracked `.py` files**, plus `mcp/tools/` holding
  the action-routed tool modules (`engine_tools.py`, `graph_tools.py`,
  `ontology_tools.py`, `intent_tools.py`, `compliance_tools.py`, …).
  `kg_server.py` is the MCP server; `multiplexer.py` fronts the fleet.

Both dispatch into the **same** `_execute_tool()` action core — REST twins are
registered in `kg_server.ACTION_TOOL_ROUTES`. A capability added to one and not
the other is a build break, not a follow-up (`guardrail-surface-parity`).

### The subsystems by measured size

`find <pkg> -name '*.py' | wc -l`, 2026-09-03. Size is a review signal, not a
split decision — but it tells you where an owner probably already exists.

| Package | Files | Package | Files |
|---|---:|---|---:|
| `knowledge_graph/` | 607 | `security/` | 44 |
| `gateway/` | 94 | `rlm/` | 43 |
| `harness/` | 86 | `messaging/` | 42 |
| `graph/` | 78 | `protocols/` | 39 |
| `mcp/`, `core/` | 73 | `tools/` | 38 |
| `domains/` | 55 | `models/` | 37 |
| `orchestration/` | 54 | `observability/` | 32 |

The heaviest single modules — where an unowned addition will land if you are not
careful (`git ls-files … | xargs wc -l`, 2026-09-03):

`mcp/multiplexer.py` 8,151 · `core/config.py` 7,843 ·
`knowledge_graph/core/engine_tasks.py` 7,736 ·
`knowledge_graph/core/source_sync.py` 6,793 · `mcp/kg_server.py` 6,430 ·
`orchestration/agent_runner.py` 6,348 · `deployment/doctor.py` 6,304 ·
`models/knowledge_graph.py` 5,827.


## Dependency direction — the measured SCC

| Measurement | Value | Source |
|---|---|---|
| Modules / raw unique internal edges | 1,701 / 6,066 | `plans/refactor/evidence/reports/AU-SCC-DECOMPOSITION.md` "Scope and evidence boundary" (:11-14) |
| Raw all-import SCC | **847 modules** | same |
| Eager (module-load) graph | 2,880 edges, **zero** non-trivial SCCs | same |
| First-level package projection | 59 nodes, 432 edges, **one 44-package SCC** | same |

★ **The gate's own docstring (`scripts/check_import_cycles.py`) carries a
second, differently-scoped measurement — and it is LIVE rationale, not history.**
It cites `plans/complex/reports/ARCH-import-cycles.md`'s **821-module SCC
("48.5% of the codebase", confirmed three ways: a hand-written AST parser, `tach
map`, `ast-metrics` coupling communities)** and records that restricting the
graph to load-time-executing edges collapses that SCC to **3 modules plus one
separate 2-module cycle** (:16-22). Its live eager-graph reading (:53-56, au
`5bfb6d0ee`) is **cycle_size 0, indirect_dependencies p50 2 / max 192,
dependency_depth p50 1 / max 8**, against kiss's all-imports p50 1133 / p50 11.
The 821/48.5% figure is the standing *reason* the eager/deferred/TYPE_CHECKING
split must not be "simplified" away — remove the split and the gate refires
against that raw-SCC debt and has to be reverted or grandfathered behind a
ratchet the workspace forbids. It is not a target and it does not conflict with
the refactor-program 847-module raw SCC above: different scope, different
snapshot, different graph.


## The rules — target direction

**The rules — TARGET direction you are held to in review, NOT a property the
tree has or a gate enforces.** ★ Nothing enforces the layering. `check-import-cycles`
refuses only *eager cycles*, in any direction; `scripts/check_coupling.py`
covers only geniusbot→`agent_utilities` and its hook wraps it in `|| true`. Rules
1 and 2 are measurably violated on `main` today (counts repeated in rules 1 and 2
below, re-measured 2026-09-03). Re-measure before quoting a number — one AST walk over
`git ls-files agent_utilities`, counting top-level and `from` import statements
whose first-level package differs from the importing file's, is what produced
them:

1. **Dependencies should point inward.** Contracts ← domain ← ports ←
   application ← adapters ← composition. **Violated today:** 62
   application→adapter import statements over 14 package pairs.
2. **Adapters should not call each other.** REST does not call MCP; MCP does not
   call REST. Both call the same use-case object. **Violated today:** 85
   adapter→adapter import statements over 19 package pairs, including the
   reciprocal `gateway -> mcp` (19) / `mcp -> gateway` (3).
3. **REST and MCP bind the SAME use-case object** — `_execute_tool()`. The tool
   function carries argument marshalling, never logic.
4. **Production code never imports tests, dev tooling, generated output, or
   deployment composition** (RF-ADR-003). Dev/test edges are inventoried
   separately and may not pull production upward.
5. **One capability, one owner, one implementation** (`plans/refactor/DESIGN.md` "Anti-sprawl
   invariants"). No new package/repository without a cohesive lower-level owner,
   **two or more real live consumers**, and a *measured* net reduction in cycles,
   duplicate implementations, public surface, or edges. Splitting by file count,
   scan score, or aesthetics is rejected.

## Named reverse edges — do not add to them

The named reverse edges the program is cutting — do not add to them:
`core -> knowledge_graph` (33), `models -> knowledge_graph` (unquantified),
`security -> knowledge_graph` (11), `core -> orchestration` (8),
`knowledge_graph -> mcp` (21), `knowledge_graph -> graph` (6), and the reciprocal
`knowledge_graph <-> orchestration` pair (44 / 39). The directions that are
*intended* and must stay one-way: `mcp -> knowledge_graph` (193),
`knowledge_graph -> security` (125, **not** the stale 133), `graph ->
knowledge_graph` (42), `orchestration -> core` (51).
