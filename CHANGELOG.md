# Changelog

All notable changes to agent-utilities will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.4.0] - 2026-07-03

### Added — per-package ontology federation (CONCEPT:KG-2.320..323)
- **Per-package ontology federation foundation + runtime verbs.** Every fleet package can now
  contribute its own ontology fragment, federated into the canonical library at load time;
  `sync_packages` reconciles contributed fragments, `graph_fork` forks a working ontology/graph
  view, and `graph_memory` is unified onto the one memory path. The federation binds
  package-declared `.ttl` fragments into the live facade (store / `owl_bridge` / retrieval).

### Added — unified XDG install (CONCEPT:OS-5.77/78/79)
- **One XDG tree for skills + prompts + ontologies.** `agent_utilities/core/unified_install.py`
  materializes all three provider types (skills, prompts, ontologies) into a single `$XDG`
  install tree with the matching runtime read-paths, so `install` drops everything an agent
  auto-loads in one place. Added a `doctor` check for the unified install and wired the read
  paths through `core/paths.py` / `core/providers.py`.

### Changed — the HARD numpy/scipy drop (CONCEPT:KG-2.324, Analytics P5 final)
- **`agent_utilities.numeric` is kernel-or-raise.** The `xp` shim now imports the compiled
  `epistemic_graph.numeric` kernel and **raises `ImportError` when it is absent** — the old
  `HAVE_KERNEL`-false path and the `import numpy as _np` binding are removed. There is **no
  numpy fallback**; the kernel is the **sole numeric backend**. `std`/`var` are called with
  the kernel's `ddof=` **keyword** (its signature is `(a, axis=None, ddof=0, keepdims=False)`).
- **numpy/scipy removed from agent-utilities entirely.** They are gone from
  `requirements.txt`, base `dependencies`, and every extra (`embeddings`/`ann`/`finance`);
  the `numeric-fallback` extra is **deleted**. Base now declares
  `epistemic-graph[numeric]>=2.7.0` (the kernel is a hard base dependency). numpy survives
  ONLY as a **dev/test-only** ground-truth reference in the `[test]` extra
  (`tests/test_numeric_parity.py`, `pytest.importorskip`-gated). Grep for `import numpy` /
  `import scipy` across `agent_utilities/` + `scripts/` is **zero**.
- **The four scipy ops are native kernel exports** (engine CONCEPT:EG-356): `xp.eigsh`,
  `xp.spearmanr`, `xp.ks_2samp`, `xp.norm_ppf` / `xp.norm_pdf`. Migrated off direct scipy:
  `spectral_navigator` (eigsh), finance `alpha_factors` (spearmanr) / `execution` (ks_2samp)
  / `risk_manager` (norm.ppf/pdf), and the `check_designation_eval` / `check_retrieval_quality`
  scripts (numpy → `xp`).
- numpy remains an **internal detail of the kernel** (rust-numpy), reached through the kernel
  — never imported or declared by agent-utilities — for the long-tail array ops the compiled
  kernel does not yet expose (the `random` Generator API, `cov`/`corrcoef`, `save`/`load`,
  axis norms, N-D element-wise, pandas-wrapped inputs). The shim restricts the kernel
  fast-path to raw `ndarray`/`list`/`tuple` so pandas `Series`/`DataFrame` wrappers are
  preserved. See `docs/guides/numeric-kernel.md`.

## [1.3.0] - 2026-07-03

### Summary — the full numpy drop (Analytics P5 final)
Finishes the numeric axis the honest way: the published `eg-numeric` kernel wheel
(cp39-abi3, shipped with `epistemic-graph` 2.6.0) becomes a **hard dependency** of the
`numeric-kernel` extra, so the `xp` shim is kernel-primary with numpy demoted to a
fallback-only extra. numpy is no longer a base or primary dependency — it survives only as
the `numeric-fallback` extra and in the leaf `finance`/`embeddings`/`ann` extras that use
genuinely scipy-specific ops. Dev stays editable and non-publishing.

### Added
- **KG-2.319 — `eg-numeric` hard dependency (loose floor).** `numeric-kernel` now declares
  `eg-numeric>=0.1.0` directly (alongside `epistemic-graph>=2.6.0`), both loose floors —
  never exact pins. Prod pulls the published kernel wheel; the `xp` shim's numpy fallback
  branch is deliberately kept for kernel-absent environments.
- **KG-2.319 — editable, non-publishing dev path documented.** `maturin develop -m
  crates/eg-numeric/Cargo.toml --features python` (with `PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1`)
  installs an editable kernel from local source over `PYTHONPATH=/au:/eg` — dev never depends
  on the published wheel. See `docs/guides/numeric-kernel.md`.

### Changed
- **KG-2.319 — `epistemic-graph` floor bumped to `>=2.6.0`** in the `engine` and
  `numeric-kernel` extras (the release that ships the abi3 kernel wheel). Still floors.
- **Version bumped 1.2.0 → 1.3.0** consistently across `pyproject.toml`,
  `agent_utilities/__init__.py`, `agent_utilities/agent_utilities.py`, `README.md`, and
  `.bumpversion.cfg`.

## [1.2.0] - 2026-07-03

### Summary — analytics kernel axis: numpy replaced by a parity-gated Rust kernel
Completes the analytics program on the agent-utilities side: the `xp` numeric shim
is now backed by the compiled `eg-numeric` kernel (faer/ndarray) as the **primary**
backend, with numpy demoted to an explicit fallback extra — and a CI gate proves the
kernel matches numpy op-for-op. Also adds the memory→weights distillation export path.

### Added
- **KG-2.314 — `xp` ufunc-method surface.** `xp.maximum`/`xp.minimum` expose
  `.accumulate`/`.reduce`/`.outer`/`.at`, closing the last 3 raw-numpy call sites.
- **KG-2.315 — kernel-live `xp` shim.** When the `eg-numeric` pyo3 wheel is installed
  the shim binds the compiled kernel (`HAVE_KERNEL=True`); clean numpy fallback otherwise.
- **KG-2.316 — memory→weights distillation.** Torch-free exporter turning consolidated
  agent-memory into a deterministic SFT/preference JSONL corpus + `DistillationTargetSpec`,
  with the LoRA training handed off to `agents/data-science-mcp` (MCP + REST surfaces).
- **KG-2.317 — kernel-primary numeric backend (P5).** `numeric-kernel` extra (primary)
  + `numeric-fallback` extra (numpy/scipy, degraded); numpy is out of base deps.

### Changed
- **KG-2.313 — numpy→`xp` migration.** 34 files migrated from `import numpy as np` to the
  `xp` shim (P2 light-ops + P3 faer-backed linalg); no numerical regressions.

## [0.53.0] - 2026-06-28

### Summary — the saturate-and-delegate program
This release lands the program that makes the local LLM + graph-os do the work
while the harness orchestrates and resolves exceptions, and hardens the engine to
run it at saturation:
- **Engine scaling & the resource-priority edict (CONCEPT:ORCH-1.98/1.99).**
  Interactive/orchestration work outranks background ingestion so orchestration is
  never starved; the LLM-server capacity guard (CONCEPT:KG-2.298) and split-GPU
  embedding keep the shared model responsive under load.
- **Orchestration/execution seam + the `agent-utilities-expert` agent
  (CONCEPT:KG-2.296).** Every delegated run writes `:ToolCall`/`RunTrace`
  provenance to the epistemic-graph for full visibility and steerability.
- **Chunked async drain (CONCEPT:KG-2.301/2.302).** A single full sync of a large
  corpus normalizes into capacity-guarded paginated background batch-tasks watched
  via the new `source_drain` tool / `/source/drain` REST twin — no per-task timeout
  hangs (paired with the KG-2.286 soft-timeout watchdog).
- **Self-evolution flywheel + genesis private repos** drive the AHE-3.x hardening
  loop and stand the ecosystem up from bare hosts.
- Plus the durable backend work merged in this cycle: bounded PostgreSQL
  authority-write tail (KG-2.152), a dedicated OWL-enrichment lane (KG-2.153), the
  MemoryData bake-off harness (AHE-3.71–3.74), fleet-wide verbose auto-wire
  (ECO-4.89/4.90), and atlassian-mcp streamable-http wiring (KG-2.123/2.124).

### Changed
- **Async mirror fan-out — the durable-write ack no longer waits on the mirror
  enqueue (CONCEPT:KG-2.273).** `FanOutBackend` mirror writes were synchronously
  appended to the sqlite outbox before the authority ack returned, so a slow/locked
  mirror outbox throttled ingestion (the `busy_timeout` fix removed the lock errors
  but not the coupling). The mirror enqueue is now a **non-blocking hand-off**: the
  write puts the mutation onto a bounded, auto-sized in-memory ring and returns
  immediately; a single persister thread drains the ring into the durable outbox
  (batched) and wakes the per-mirror drainers. Bounded + backpressure: on ring
  overflow the producer falls back to a synchronous durable-outbox append (loud,
  reconcilable — a mirror write is never silently dropped, memory never grows
  unbounded). Crash-safety is preserved (the durable outbox replays from its cursor
  on restart; `reconcile` covers the brief ack→persist window). Implements roadmap
  item D of `reports/north-star-agent-compute-architecture-2026-06-27.md`.

### Added
- **Agentic Resource Discovery (ARD) interop — publish + consume + federate
  (CONCEPT:ECO-4.95/4.96/4.97, KG-2.188, OS-5.60).** agent-utilities becomes a
  peer in the ARD discovery commons (draft spec from Hugging Face, Microsoft,
  Google, GoDaddy). *Publish:* serve a signed `ai-catalog.json` at
  `/.well-known/ai-catalog.json` and a ranked `POST /search` (gateway router +
  graph-os `@mcp.custom_route` mirrors), mapping fleet MCP servers and KG skills
  onto ARD media types via `MCPMultiplexer.discover_tools`. *Consume:* a new
  `@register_source("ard")` connector ingests external registries (HF preset
  built-in), Ed25519-verified, materialized as typed `:MCPServer`/`:Skill` nodes
  linked to a `:ResourceRegistry` (`_sync_ard`). *Federate:* `ArdFederationRelay`
  fans `/search` out to peer registries (`auto`/`referrals`/`none`) with loop-break
  and dedup. Ed25519 publisher signing (`security/ard_signing`, import-guarded).
  Docs: `docs/architecture/ard-interop.md`.
- **KG-2.9 connector-expansion: enterprise/EA/governance/CRM/finance/legal
  bidirectional connectors (CONCEPT:KG-2.9).** A large wave of source connectors
  that both ingest *and* write back over one fail-closed writeback core. Ops +
  observability (Phase 1), `spec<->ticket<->agent` bidirectional linking and
  full Twenty CRM (Phase 2A/2B), EA + governance bidirectional sinks
  (ArchiMate + Egeria, Phase 3), and finance/legal/personal high-stakes
  connectors (Phase 4: emerald + legal as propose-only, plus wger + Mealie).
  Cross-domain batches add Nextcloud (docs/calendar/contacts), Identity
  (Okta + Keycloak), Salesforce CRM, Ansible (action), and Home Assistant
  (device inventory) bidirectional. Closing deferrals: Mealie doc-source preset
  and Microsoft 365 (M365) async ingestion. Registry grows 53 → 54 connectors
  (`genesis.yaml` regenerated).
- **Risk-tier approval queue for high-stakes write-backs (CONCEPT:KG-2.9).**
  Outbound mutations carry a per-sink `risk_tier`; high-stakes writes
  (finance/legal/identity) are propose-only and route through a durable approval
  queue before execution, while low-risk sinks auto-apply. The three legacy
  writeback modules are unified into one fail-closed writeback core that every
  connector dispatches through.
- **ServiceNow + ERPNext bidirectional (CMDB/ERP) (CONCEPT:KG-2.9/KG-2.53).**
  ServiceNow and ERPNext become first-class `materialize` sources (TRM + risk
  read), with KG-derived inventory pushed back as a reconciled set into the
  CMDB/ERP (cross-source inventory push), plus scheduling, docs, and both MCP +
  REST surfaces.
- **Native LeanIX, Camunda, and ARIS EA/BPM integration (CONCEPT:KG-2.9/KG-2.8/KG-2.53).**
  A shared EA client (`ecosystem/ea_clients.py`); a LeanIX metamodel→OWL compiler
  with a dynamic promotable set, a real OWL mirror, delta sync (watermark poll +
  webhook narrowing + reconcile), and fail-closed dry-run-first backfeed of
  KG-derived knowledge. Camunda + ARIS gain `<->` KG bidirectional integration
  with an EPC step-lift extractor and outbound process-intelligence writeback.
  All sources route through the generic `source_sync` (delta/reconcile for any
  source) and shared materialize core; `aris-mcp` registered in the fleet.
- **Loop engine — successor to the golden loop (CONCEPT:KG-2.78).** Collapses the
  many ad-hoc loops into one `LoopController` over a single `Loop` node model (a
  long-running-objective unit). Folds the develop + skill loops and the durable
  goal-runner into the controller (checkpointing cross-cutting), collapses the
  goals table onto the KG Loop node (one persistence model), and exposes one
  `graph_loops` entrypoint (L3/L4). Engine intake now advances active research-kind
  Loops. Adds a Loop Engine runbook.
- **Agent-Native Research Artifacts — ARA (CONCEPT:KG-2.80).** ARA modeled as an
  OWL-native ontology object with a compiler, seal, exploration producer, and
  live manager (A2–A5), exposed via a `research_artifact` MCP tool and REST
  gateway twin (single source of truth). Includes the ARA comparative-analysis
  report and one-ontology framing.
- **Reasoning as the research engine (CONCEPT:KG-2.79).** `OntologyReasoningDriver`
  turns OWL/RDF reasoning into the engine that drives the research cycle; the
  reason stage is wired into the cycle natively (best-effort).
- **Unified research-intelligence cycle + robust ConceptMatcher (CONCEPT:KG-2.75/2.76/2.77).**
  A multi-signal `ConceptMatcher` (id + embedding-recall + LLM-judge + fusion)
  wired live into golden-loop assimilate, with matcher-driven tiering (covered
  papers demoted to memory). The golden loop becomes a unified research-pipeline
  runner via a single intake stage.
- **Action-conditioned world model + pluggable reasoning paradigms (CONCEPT:KG-2.67/2.68/2.73).**
  A learned action-conditioned world model over the Markov kernel (KG-2.67), a
  pluggable reasoning-paradigm registry with an outcome-learning router (KG-2.68),
  and a learned world-model SAI track (KG-2.73).
- **SAI factory: closed-loop self-improvement (CONCEPT:AHE-3.27/3.28/3.29, SAFE-1.6/1.7).**
  Adaptation-speed metric (time-to-target / sample-complexity, AHE-3.27), a
  Specialization task/verifier contract (AHE-3.28), and a closed-loop
  `SaiFactoryController` (AHE-3.29) wired into the gateway hot path
  (`graph_analyze specialize`) and a native-on daemon tick, plus superhuman
  certification + benchmark (SAFE-1.6/1.7).
- **RLM long-context benchmark + drop-in client (CONCEPT:AHE-3.32, ORCH-1.54).**
  A long-context benchmark with a scoreboard (and `RunTrace.usage` cost capture)
  and a drop-in, family-aware `RLM.completion()` client.
- **Recursive-distillation loop (CONCEPT:AHE-3.31).** A recursive knowledge-
  distillation loop realizing the AHE-3.25 intent.
- **Universal CredentialProvider + PulseLink open-web/social source presets
  (CONCEPT:OS-5.38/OS-5.39, ECO-4.46).** A unified outbound `CredentialProvider`
  with a typed source-credential registry (OS-5.38/5.39); PulseLink `mcp_tool`
  source presets (ECO-4.46) for keyless open-web/social research, with X
  integration externalized to the optional `pulselink-mcp` (registered in the
  fleet).
- **Ecosystem capability registry → Concept nodes (CONCEPT:KG-2.7).** The
  golden-loop bridges the ecosystem capability registry into the KG as Concept
  nodes, and breadth self-configures from `workspace.yml` (default on).
- **Per-repo baseline deltas in the code-health sweep (CONCEPT:CE-039).** The
  `code_health` sweep tracks per-repo baseline deltas (new / fixed / graceful),
  a new-debt gate distilled from VibeDoctor.
- **Declarative skill / skill-workflow scheduler (CONCEPT:OS-5.30).** A
  declarative scheduler replaces hardcoded daemon ticks.
- **Finance microstructure / Kyle-surveillance ontology (CONCEPT:KG-2.81).**
  `knowledge_graph/ontology/finance_objects.py` registers `MicrostructureSignal`
  and `SurveillanceSignal` (extends) OWL interfaces plus typed links (signal
  `grounded_in` Article / `relates_to` Concept) into the default registries, and
  promotes the `microstructure_signal`/`surveillance_signal` node types in the
  OWL bridge so the Kyle insider/stealth-surveillance signal (distilling
  arXiv:2605.27684) reasons transitively over `grounded_in`/`supports`. Pairs
  with the emerald-exchange detector (EE-042) and adverse-selection gate (EE-043)
  over the engine `surveillance_risk` kernel (KG-2.20k). Defensive use only.

### Changed
- **Loop stage flags renamed `KG_GOLDEN_*` → `KG_LOOP_*` (CONCEPT:KG-2.78).** The
  four loop-stage env flags follow the golden-loop → Loop-engine rename; no
  back-compat aliases (No-Legacy).
- **`kg_server._build_server` tools split into `mcp/tools/` modules.** The
  monolithic tool-registration block in `mcp/kg_server.py` is decomposed into
  per-domain modules under `mcp/tools/` — a structural refactor with no behavior
  change.
- **Connector sources routed through one materialize core.** Camunda / ARIS /
  Egeria (and LeanIX as first adopter) standardized onto the generic
  `source_sync` (delta/reconcile for any source) and the shared materialize core
  instead of bespoke per-source paths.

### Fixed
- **PG lock-contention check false-matched "idea_block" → schema heal starved.**
  The Postgres lock-contention detector matched the substring `idea_block`
  (e.g. in unrelated identifiers), wedging the schema-heal path into starvation;
  the check is now precise so schema healing proceeds.
- **Neo4j / FalkorDB fan-out mirrors unstalled (CONCEPT:KG-2.74).** Map / nested
  properties are now serialized for the Neo4j and FalkorDB mirror writers, so the
  fan-out mirror no longer stalls on structured property values.

## [0.49.0] - 2026-06-14

> Note: per-release CHANGELOG sections for 0.4.0–0.48.x were not maintained; this
> section consolidates the accumulated unreleased work up to 0.49.0 (the current
> `pyproject.toml` / bumpversion version). Full history is in git.

### Changed
- **Dependency floors refreshed to current PyPI latest.** Raised the declared
  minimum versions across `pyproject.toml` and `requirements.txt` to the latest
  compatible releases (e.g. `pydantic[email]>=2.13.4`, `urllib3>=2.7.0`,
  `dspy-ai>=3.2.1`, `platformdirs>=4.10.0`, `filelock>=3.29.4`, `fastapi>=0.137.0`).
  The resolved `uv.lock` is unchanged — the lock already used versions at or above
  the new floors — so this is declaration hygiene with no runtime change. Five
  constraints are held below latest because workspace siblings cap them:
  `pydantic-ai-slim`/`pydantic-graph` at `1.106.0` (pinned exactly by
  `pydantic-acp 0.9.6`), `pandas<3` (`llama-index-readers-file`), `packaging<26`
  and `cryptography<49` (both capped by `mlflow` via `data-science-mcp[tracking]`).

### Added
- **Concurrent N-way mirrored writes (CONCEPT:KG-2.74).** `GRAPH_BACKEND=fanout`
  turns mirroring into the default for every write: one configurable **authority**
  store (`GRAPH_AUTHORITY`) serves reads and acks writes, and each mutation is
  replicated — losslessly and asynchronously — to any set of durable mirrors
  (`GRAPH_MIRROR_TARGETS`, named against `KG_CONNECTIONS`). New
  `knowledge_graph/backends/fanout_backend.py` (`FanOutBackend`) generalizes the
  two-tier `TieredGraphBackend` into one-authority-N-mirror; new
  `backends/outbox.py` (`GraphOutbox`) is a durable sqlite/WAL per-mirror append
  log, so a mirror that is offline or slow **keeps its unapplied tail and replays
  from a persisted cursor on reconnect / restart — no write is lost**. One drainer
  thread per mirror serialises a single-writer store (LadybugDB) for free.
  `reconcile()` (reusing `TieredGraphBackend.reconcile_to_durable`) is the
  drift-repair backstop. Operated on both surfaces:
  `graph_configure(action="mirror_status"|"reconcile")` (MCP) and the
  `/graph/configure` REST twin. Default-on once configured; the zero-infra default
  is unchanged. Deploy stacks: `services/{neo4j,falkordb}` (R710). Docs:
  `docs/architecture/graph_backends_architecture.md` (“Mirror every write to N
  stores at once”).
- **Shortcut-resistant search-task synthesis (CONCEPT:KG-2.70/2.71/2.72, AHE-3.30).**
  Distills FORT-Searcher (arXiv:2606.12087) onto the epistemic graph. New
  `knowledge_graph/search_synthesis/` package: `evidence_subgraph.py` builds a
  bounded evidence-graph workspace around an answer entity (KG-2.70);
  `shortcut_risks.py` runs the four shortcut detectors — single-clue selectivity,
  evidence co-coverage, exposed constants, prior-knowledge binding — as queries over
  that workspace (KG-2.71); `question_formulation.py` formulates a verifiable
  question (intermediate-name withholding) and adversarially refines it until no
  shortcut trips (KG-2.72). The reward spine `graph/training_signals.py` gains the
  realized-difficulty trajectory signatures `solving_cost` / `answer_hit_time` /
  `prior_shortcut_rate` + a `search_heavy` gate (AHE-3.30). Wired live as the
  `graph_search_synthesis` MCP tool (synthesize/diagnose) and an opt-in
  `GoldenLoopController.run_one_cycle(synthesize_search=True)` self-play stage that
  persists `SearchTask` proposal nodes + a propose-only corpus draft. Docs:
  `docs/architecture/shortcut_resistant_search_synthesis.md`.
- **Git issue/PR → SWE task resolver (CONCEPT:ECO-4.43).** `integrations/git_resolver.py`
  parses GitHub/GitLab issue+PR webhooks, classifies a suggested-task taxonomy
  (open-issue / open-PR / failing-checks / merge-conflicts / unresolved-comments),
  ingests the task as a `GitTask` KG object linked to its repo (so suggested-tasks are a
  graph query, not per-platform code), and enqueues a `swe_engineer` turn on the durable
  dispatch queue (ORCH-1.45). Exposed at `/api/git/webhook` + `/api/git/suggested-tasks`.
- **Optional browser-agent tier (CONCEPT:ECO-4.44).** A pluggable `BrowserDriver`
  (`runtime/browser_tier.py`) + typed `BrowseAction`/`BrowserObservation`, kept off the
  core edit→test critical path: the `NullBrowserDriver` floor advertises the tier
  everywhere while a real driver (e.g. agent-browser) is attached only where needed.
  Gated by `ActionPolicy` as `workspace.browse`.
- **SWE provenance panel surface (CONCEPT:OS-5.34).** `/api/runtime/sessions/{id}/provenance`
  returns the run's action/observation trajectory plus the `Code` symbols each edit mutated
  (KG-2.64) — the data the agent-webui SWE view renders beside the live SSE event stream
  (`/api/runtime/sessions/{id}/events`). The React components are the remaining agent-webui
  follow-up; the backend contract is complete.
- **SWE-bench harness + failure-driven self-improvement (CONCEPT:AHE-3.22 / AHE-3.23) —
  the "surpass" lever.** `harness/swebench_{corpus,harness,remediation}.py` clone a repo
  @ base_commit into a workspace, run the KG-grounded SWE agent, apply the gold
  `test_patch`, run FAIL_TO_PASS/PASS_TO_PASS, and score "resolved" (aggregation mirrors
  the LongMemEval router). Unlike a static benchmark number, every *unresolved* instance
  becomes a `FailureRecord` → clustered → filed as a `failure_gap` Concept via the shared
  AHE-3.18 path, which the golden loop picks up to self-improve; a SWE-specific regression
  gate re-runs the *exact* failed instance and only passes when it now resolves (gating the
  AHE-3.14 governed merge). Exposed at `/api/swebench/run` + `/report`. The solver is
  injectable, so the suite + remediation loop test end-to-end without an LLM.
- **KG-grounded SWE agent (CONCEPT:ORCH-1.47 / KG-2.65).** A knowledge-grounded
  software-engineering agent that runs the edit→run→test loop inside the
  developer-workspace runtime. Its differentiator: it grounds in the live code
  ontology via graph queries — `find_definition`/`who_calls`/`impacted_tests`/
  `call_graph`/`dependencies` (`tools/code_intelligence_tools.py`) — instead of
  context-stuffing, and acts via workspace tools (`tools/swe_workspace_tools.py`)
  that execute in `deps.workspace` and mirror to the KG. Tools register natively
  through `register_agent_tools` behind a `SWE_TOOLS` gate (shared
  `register_swe_tools`); the agent is declared as a `prompts/swe_engineer.json`
  blueprint using a `capabilities` (intent) list, not hard-coded tool names; and a
  new `swe` orchestration mode (`engine.execute_swe`, `orchestration/swe_agent.py`)
  drives it.
- **Reference-integrity gate for tool/skill references (CONCEPT:ECO-4.45, Layer 1).**
  `scripts/check_tool_refs.py` resolves every concrete `tools`/`skills` reference in
  `prompts/*.json` against the live tool-function + skill-slug universe (MCP-prefix
  tolerant) and reports drift — making rename/refactor and mcp-multiplexer-prefix
  breakage loud instead of silent. `capabilities` intent lists are exempt (the target
  model: bounded intents resolved by the KG at construction, rename- and prefix-proof).
  Report mode by default; `--strict` fails CI. (KG-capability resolver = next workstream.)
- **Developer-Workspace Runtime — OpenHands-parity SWE substrate (CONCEPT:OS-5.33 /
  ORCH-1.46 / KG-2.64).** A new `agent_utilities/runtime/` package gives the agent a
  persistent, sandboxed workspace (stateful shell with cwd-persistence, file
  read/write/edit, a pytest test-runner, port-expose) driven by a *typed
  action/observation protocol* — distinct from the snippet-against-a-namespace RLM
  sandbox. `LocalWorkspace` is the zero-infra floor; `DockerWorkspace` reuses the RLM
  container hardening flags but keeps the container **long-lived** (`docker exec` +
  bind-mount file ops) with an idle-reaper. Every action/observation is mirrored into
  the KG as provenance grounded to the `Code` symbols an edit mutated
  (`(:WorkspaceAction)-[:MUTATED]->(:Code)`), making runs replayable and failures
  attributable. Mutating actions (`workspace.cmd|write|edit`) are governed by the
  fail-closed `ActionPolicy` (OS-5.24, default tier `auto` — the sandbox is the
  boundary, overridable to `approval_required`). Exposed over `/api/runtime/*`
  (session → act → SSE event stream). New ontology classes `:WorkspaceAction` /
  `:WorkspaceObservation` in `ontology_software.ttl`.
- **Durable execution wired into the live path (CONCEPT:OS-5.16).** The
  `DurableExecutionManager` was built but uninvoked; it now runs on the real async
  paths via the new `arun_durable_action`. `run_goal_loop` resumes from the last
  in-flight checkpoint on restart and wraps each iteration's validation under an
  idempotency key `{goal_id}:{iteration}`; the dispatch worker wraps the agent
  invocation under the turn `job_id` — a crash-resume or at-least-once redelivery
  never re-applies an effect that already ran.
- **Queryable cross-agent correlation / blast-radius (CONCEPT:OS-5.11).**
  `persist_event` stamps `correlation_id` (+ actor/tenant) onto `FleetEvent`
  nodes, and the dispatch worker stamps it onto executed `Task` nodes. New
  supervisory reads: `GET /api/fleet/trace?correlation_id=…` (full run trace) and
  `GET /api/fleet/touched?resource=…` (who touched X).
- **Granular typed OpenAPI surface for the ontology/object layer**
  (`gateway/ontology_api.py`): resource-style GET routes
  (`/api/ontology/value-types/{name}`, `/ontology/interfaces/{name}`,
  `/ontology/functions/{name}`, `/objects/{id}`, `/objects/{id}/history`,
  `/objects/{id}/as-of`) that appear in `/openapi.json` and dispatch through the
  same `_execute_tool` single source of truth as the collapsed routes/MCP tools.
- **Fleet chaos coverage** (`tests/integration/test_fleet_chaos.py`): whole-domain
  pause containment + concurrent goal loops honoring pause with zero side effects.

### Changed
- **Configuration discipline — the env-read fold is COMPLETE: zero bare
  `os.environ`/`os.getenv` reads remain anywhere in `agent_utilities/`** (every
  prefix, not just KG/graph — `AGENT_*`, `VAULT_*`, `OTEL_*`, connector creds, …).
  ~310 remaining reads across ~107 modules were routed through `config.setting()`
  behavior-preservingly. `setting()` moved to a dependency-free `core/_env.py`
  (re-exported by `config`) so it's importable while `config` is still
  initializing — fixing the circular-import deadlock that a package-wide fold would
  otherwise hit. The `check_no_env_sprawl.py` baseline is now **empty**; any new
  bare read (any prefix) fails CI. Also hardened `ConnectionRegistry.resolve_names`
  to route a non-str/list `target` (an unresolved pydantic `FieldInfo` when a tool
  fn is called directly in tests) to the default instead of a spurious fan-out.
- **Configuration discipline — every `KG_*`/`GRAPH_*`/`EPISTEMIC_*` env read folded
  off `os.environ`.** All 100+ governed bare reads across ~35 modules now go through
  one of two centralized, `config.json`-driven paths instead of scattered
  `os.environ.get`/`os.getenv`:
  - New **`config.setting("VAR", default, cast=…)`** accessor (`core/config.py`) — the
    sanctioned way to read an env var outside `config.py`. Reads `os.environ` **live**
    (so daemon cadences, `monkeypatch.setenv` tests, and runtime toggles still work),
    with a declared default and type coercion inferred from the default (or explicit
    `cast`). config.json keys are injected into the environment first, so both fields
    and `setting()` are config.json-driven.
  - Deployment-varying / behavioral / test-varied flags → `setting()`; pure load
    tunables (`KG_LLM_CONCURRENCY`, `KG_CHAT_CONCURRENCY`, `KG_ENRICH_BATCH`,
    `GRAPH_POOL_*`, `KG_BACKGROUND_MAX_CONCURRENT`) → **auto-sized** via
    `compute_ingest_worker_count()` or named constants; single-value cadences/limits
    (`KG_*_INTERVAL`, `KG_TASK_*`, `KG_LLM_TIMEOUT`/`_MAX_RETRIES`,
    `GRAPH_SERVICE_CHECKPOINT_INTERVAL`) → named module constants — honoring the
    AGENTS.md *Configuration discipline* rule (knobs auto-sized, not proliferated).
  - **Gate broadened** (`scripts/check_no_env_sprawl.py`): now ratchets bare env
    *reads* of **every** prefix (not just KG/graph), excluding writes; governed reads
    removed from the frozen baseline (now a non-KG burn-down list).
  - Docs: `AGENTS.head.md` *Configuration discipline* rewritten (field-vs-`setting()`
    decision, all-prefix gate) + regenerated `AGENTS.md`/`CLAUDE.md`;
    `docs/architecture/configuration.md` + `docs/examples/config.json` updated.

### Added
- **Named multi-connection graph registry (CONCEPT:KG-2.63)** — register several
  live graph backends side by side (e.g. `prod-neo4j`, `team-falkor`, `pg-main`)
  and run the *same* `graph_*` tools against any one or fan out to all, with the
  backend choice abstracted behind an optional `target` parameter. No code is
  forked into a separate server — every existing MCP tool and its REST twin gains
  this for free.
  - New `ConnectionRegistry` (`knowledge_graph/core/connection_registry.py`):
    lazy-connect, per-connection `IntelligenceGraphEngine`, the reserved
    `default` name aliases the existing active engine (never duplicated),
    thread-safe with double-checked caching, `status()` health surface.
  - `target` on `graph_query` / `graph_search` / `graph_write`: omitted/`default`
    = legacy behaviour (backward compatible); a single name targets one
    connection; `all`/list/`a,b` fans out to per-connection labeled results with
    partial success. **Writes only fan out on an explicit multi-target value.**
  - `graph_configure` actions: `add_connection`, `remove_connection`,
    `list_connections`, `set_default_connection`.
  - `KG_CONNECTIONS` (`AgentConfig.kg_connections`): declarative JSON list of
    named connections, seeded into the registry at first use. Zero-infra default
    fully preserved.
  - Per-backend `cypher_support` tier on `GraphBackend` (`full` for
    neo4j/falkordb/Apache AGE; `subset` for the regex Postgres transpiler and the
    in-memory epistemic graph) so portability is honest in fan-out. **Register
    Postgres as `age`** for one-query-runs-everywhere openCypher.
- Closed a pre-existing gateway/MCP parity gap: `usage_query` and
  `ingest_sessions` now have REST twins (`/usage/query`, `/usage/ingest-sessions`).

### Changed
- **Resilience/HTTP unification** — forked circuit breakers, hand-rolled retry
  loops, and inline httpx client constructions collapsed onto the canonical
  primitives, behavior-preserving per site:
  - **One HTTP client factory** (`agent_utilities/core/http_client.py`):
    `create_http_client` / `create_async_http_client` with unified defaults —
    finite timeout always (30s default; `timeout=None` rejected), `verify=True`
    default, a standard `agent-utilities/<version>` User-Agent (caller headers
    win), and optional `ResiliencePolicy`-backed transport retry
    (`http_retry_policy()`). The 9 inline construction sites from the
    consolidation audit now build through it: `core/embedding_utilities`,
    `server/dependencies`, `knowledge_graph/backends/contrib/ladybug_backend`
    (keepalive pool), `protocols/a2a` (×3), `knowledge_graph/core/engine_ingestion`
    (×2), `server/routers/proxy`, `tools/x_search_tool`, `security/auth` (JWKS),
    `orchestration/scaling_signals` (Prometheus).
  - **Breaker unification (CONCEPT:ORCH-1.8)** — the parallel engine's forked
    per-agent-type `_CircuitBreaker` is deleted; `AgentTypeCircuitBreaker` now
    subclass-parameterizes the canonical OS-5.23 `engine_breaker.CircuitBreaker`
    (the ECO-4.34 per-child-breaker pattern) with infinite cooldown to preserve
    the historical open-until-success semantics. One deliberate divergence:
    `CIRCUIT_BREAKER_THRESHOLD=0` now disables the breaker (canonical
    convention) instead of permanently disabling every agent.
  - **Retry loops onto `ResiliencePolicy` (CONCEPT:ORCH-1.36)** with
    byte-identical per-site delays: LadybugDB `execute` lock contention
    (exponential + additive jitter), PostgreSQL `execute` (lock backoff + the
    zero-delay auto-DDL heal retry via the new `RetryableError` delay hint),
    background-learning `with_backoff`, `x_search` (new linear backoff
    strategy; retry sleep is now non-blocking `asyncio.sleep` instead of
    `time.sleep` inside an async tool), prompt-chain steps, the specialist
    dispatch outer loop in `graph/executor`, and the parallel engine's in-wave
    SWARM-5 retry. The multiplexer child restart backoff
    (`mcp/child_resilience`) is intentionally left distinct.
  - **`ResiliencePolicy` extensions**: `backoff_strategy="linear"`,
    `jitter_strategy="additive"`, and `RetryableError(backoff_s=...)`
    per-exception delay hints — minimal additions so migrated sites keep
    identical timing rather than leaving forks.
- **`X_TOOLS` now defaults OFF** — the X/Grok social-search toolset requires
  xAI credentials (optional infra), so it joins `MEDIA_TOOLS`/`DB_TOOLS` as an
  explicit opt-in: production X/Grok deployments must set `X_TOOLS=1`.
  Documented in `.env.example` and `docs/architecture/configuration.md`; full
  extraction to a fleet `x-mcp` service remains the audit's preferred
  long-term home.

### Security
- **TLS verification is centrally governed** — the two `verify=False` client
  constructions flagged by the audit (`core/embedding_utilities`,
  `server/dependencies.get_http_client`) were in fact already gated behind the
  `SSL_VERIFY` / `ssl_verify` opt-out (default secure); they now construct via
  the canonical factory whose default is `verify=True`, so an accidental
  insecure default can no longer be introduced at a call site.

### Added
- **Fleet-scale MCP multiplexer hardening (CONCEPT:ECO-4.34)** — every aggregated
  child server now runs behind a per-child `ChildRuntime`
  (`agent_utilities/mcp/child_resilience.py`) instead of one bare shared session:
  - **Per-child concurrency limits + bounded queue**: `MCP_CHILD_MAX_CONCURRENCY`
    (default 8; per-server `max_concurrency` in `mcp_config.json`; 0 = unlimited)
    caps in-flight calls; excess calls queue at most `MCP_CHILD_QUEUE_TIMEOUT`
    (default 30s; per-server `queue_timeout`) then fail with the typed
    `MCPChildBusyError` — no head-of-line hangs behind one slow child.
  - **Session pools for HTTP children**: remote (streamable-http/SSE) children
    hold `MCP_CHILD_POOL_SIZE` round-robin connections (default 1 preserves the
    historical resource profile; per-server `pool_size`); stdio stays single-pipe.
  - **Cancellation-safe dispatch**: the child-side call runs in a shielded task —
    a caller timeout/cancel detaches cleanly (typed `MCPChildCallTimeoutError`,
    per-call ceiling from the server entry's existing `timeout` / new
    `call_timeout` key) and the slot is held until the child truly finishes, so a
    wedged child applies backpressure instead of corrupting shared session state.
  - **Restart-on-crash**: each connection generation is owned by a supervisor
    task; transport failures (stdio process exit, HTTP connect/reset) recycle the
    generation with exponential backoff (0.5s→30s, jittered). More than
    `MCP_CHILD_MAX_RESTARTS` (default 5) inside `MCP_CHILD_RESTART_WINDOW`
    (default 300s) parks the child as `failed`; calls to a restarting child wait
    briefly then fail with the typed `MCPChildUnavailableError` naming the child
    and its restart state. Generations also fix shutdown: child transports are
    now opened AND closed in the same task (anyio cancel-scope discipline).
  - **Per-child circuit breaker**: consecutive transport failures/timeouts open a
    breaker (`MCP_CHILD_BREAKER_THRESHOLD`/`MCP_CHILD_BREAKER_COOLDOWN`,
    per-server overrides) that short-circuits with the typed
    `MCPChildCircuitOpenError` until a half-open probe succeeds — the shared
    OS-5.23 engine-client state machine, subclassed (wording + per-child gauge).
  - **Health surface + metrics**: new `multiplexer_status` tool /
    `MCPMultiplexer.status_snapshot()` (per-child up/restarting/failed, restart
    count, breaker state, pool size, in-flight, queued) and new OS-5.23-registry
    series — `agent_utilities_mcp_child_calls_total{server,outcome}`,
    `..._mcp_child_breaker_state{server}`, `..._mcp_child_restarts_total{server}`,
    `..._mcp_child_queue_depth{server}` — all no-op without `prometheus_client`.
  - Deployment follow-up (out of scope here): all callers still share each
    child's credentials; per-caller credential injection is not yet wired.
- **Queue-driven agent dispatch (CONCEPT:ORCH-1.45)** — agent turns (goal runs /
  orchestrator jobs) dispatched via a session-partitioned durable queue consumed
  by a stateless scheduler fleet, replacing the in-process-only asyncio
  scheduler ceiling:
  - **Envelope + enqueue seams**: typed `AgentTurnEnvelope` (job id = idempotency
    key, session id, kind, payload *reference* — bodies stay in the state store);
    `graph_orchestrate action=dispatch` and the goal machinery return a job
    handle in queue mode (`AGENT_DISPATCH_BACKEND=queue`; `inline` default is
    byte-for-byte the previous behavior).
  - **Session-first ordering**: `partition_key_for` gains `session:<id>` above
    the KG-2.56 tenant key — per-session serial execution is a turn-coherence
    requirement; distinct sessions parallelize across `AGENT_TURNS_PARTITIONS`.
  - **Transport composes the KG-2.55 stack**: Kafka `agent_turns` topic /
    `agent-dispatch` group, Postgres SKIP LOCKED `agent_dispatch_queue` table,
    or per-host SQLite — same fail-loud selection contract.
  - **`agent-dispatch-worker`** (console script): claims under per-session
    mutual exclusion (process lock + `agent-session:<id>` advisory lock),
    rehydrates from the shared state store, executes the EXISTING
    `run_goal_loop` / orchestration-manager bodies, writes back durably, acks
    after. At-least-once + idempotent stale-claim-aware re-claims = crash
    recovery without a separate scheduler.
  - **Fleet visibility**: workers heartbeat into the sessions store's
    `dispatch_workers` registry; `/api/fleet/topology` lists them;
    `graph_orchestrate job/{id}` reports the executing worker/host; new
    `agent_utilities_dispatch_queue_depth` / `_dispatch_turns_total{outcome}` /
    `_dispatch_workers` metrics on the OS-5.23 registry.
  - Docs: `docs/architecture/agent_dispatch.md`; capacity model marks the
    queue-driven dispatch stage implemented.
- **Tenant-partitioned engine sharding (Stage 2, CONCEPT:KG-2.58 / OS-5.28)** — N
  epistemic-graph engine shards behind the existing client-side HRW router,
  partitioned by tenant/graph (`tenant → named graph → HRW → shard`):
  - **Shard-aware client path**: with 2+ `GRAPH_SERVICE_ENDPOINTS` (comma or JSON
    list; new before-validator), `GraphComputeEngine` routes each named graph to its
    owning shard via the exact `epistemic_graph.pool.ShardRouter` HRW hash (sync and
    async callers agree by construction). Routing key = explicit graph name → ambient
    `ActorContext` tenant (mapped through `tenant_graph_name`) → new `KG_DEFAULT_GRAPH`
    flag. Single-endpoint deployments are byte-for-byte unchanged (zero-infra default).
  - **Fail-loud shard contract**: `EPISTEMIC_GRAPH_AUTOSTART` applies only to the
    local `unix://` endpoint; an unreachable remote shard raises a `ConnectionError`
    naming the shard, its graph, and the remediation (KG-2.55 convention). The flock
    host role governs only the LOCAL engine.
  - **Tenant→graph naming discipline**: `tenant_graph_name(tenant, base)` →
    `tenant__<t>__<base>` in `knowledge_graph/core/shard_topology.py`, exported from
    `agent_utilities.knowledge_graph` and as `KnowledgeGraph.tenant_graph()`.
  - **Topology visibility (OS-5.28)**: `shard_topology_status()` (per-shard
    transport-level reachability + breaker state) on the unified daemon status,
    new gateway `GET /daemon/shards`, and a config-only summary on graph-os
    `GET /health`; new `agent_utilities_engine_shard_up{endpoint}` gauge and
    `agent_utilities_engine_shard_requests_total{endpoint,outcome}` counter on the
    OS-5.23 metrics registry.
  - **Deployment recipe**: worked 3-shard compose (`docker/engine-shards.compose.yml`,
    distinct ports/persist dirs/metrics listeners + one shared secret) and
    `docs/architecture/engine_sharding.md` (incl. the honest re-sharding caveat:
    HRW minimizes movement but data migration is a manual snapshot export/import);
    enterprise recipe + capacity model now reference the real sharding path.
- **Kafka ingest scale-out (Tranche 3, CONCEPT:KG-2.55/KG-2.56/KG-2.57)** — the durable
  ingest task queue becomes a production-grade, fail-loud, horizontally scalable system:
  - **Fail-loud queue selection (KG-2.55)**: new `TASK_QUEUE_BACKEND` flag
    (`sqlite|postgres|kafka`, default auto = postgres when `STATE_DB_URI` is set else
    sqlite, mirroring OS-5.16). Explicit `kafka`/`postgres` raise `TaskQueueUnavailable`
    at startup with the endpoint + fall-back instructions — never a silent SQLite
    degrade. The old `QUEUE_BACKEND` env is a deprecation-shimmed alias keeping its
    legacy graceful semantics. One construction path (`create_task_queue`) now serves
    the engine and the `--stage-to-queue` CLI.
  - **Keyed partitions (KG-2.56)**: `kg_tasks` messages are produced with a partition
    key — tenant id (ambient `ActorContext`) → repo/corpus identifier (batch-ingest
    provenance `full_path`, else path-derived root) → task type — giving per-tenant /
    per-repo ordering without global serialization. Idempotent ensure-topic at startup
    creates/grows `kg_tasks` to `KG_TASKS_PARTITIONS` (default 6, grow-only). The
    backend was rewritten onto `confluent_kafka` (already a core dep; the old code
    imported the undeclared `kafka-python`).
  - **Decoupled `kg-ingest` consumer group (KG-2.57)**: new
    `kg-ingest-worker` console script / `python -m
    agent_utilities.knowledge_graph.ingest_worker` runs ingest workers as engine
    *clients* (Rust daemon over UDS/TCP + OS-5.14 HMAC secret, `KG_DAEMON_ROLE=client`,
    no host flock). In Kafka mode the host engine's pool joins the same group, so
    partitions spread across the host AND any number of external workers; processing
    reuses the extracted `_execute_claimed_task` worker body (not duplicated) and the
    shared CPU/memory autosizer (`compute_ingest_worker_count`). At-least-once delivery
    with idempotent `job_id`-keyed claims (status-checked MERGE, cross-host
    `state_claim_guard`); the task reaper re-publishes reaped orphans in Kafka mode.
    In-process workers remain the default when Kafka isn't selected (zero-infra
    preserved).
  - **Backpressure + lag visibility**: `agent_utilities_kg_ingest_queue_depth{backend}`
    and `agent_utilities_kg_ingest_consumer_lag{topic,group}` gauges on the OS-5.23
    gateway metrics registry, sampled by the maintenance scheduler; the batch
    orchestrator's deferral now reads the uniform `engine.ingest_queue_depth()`
    (queue backlog + pending/running `:Task` nodes) across all backends.
  - Compose: `docker/kafka-kraft.compose.yml` provisions `kg_tasks`
    (`${KG_TASKS_PARTITIONS:-6}`) + `kg_staging`; docs: new "Ingest Task Queue
    Scale-Out" section in `docs/architecture/event_backbone_architecture.md`.
- **Fleet autonomy control plane (CONCEPT:OS-5.24 — OS-5.27)** — the Tranche-3 build that
  lets the platform act on its fleet without ever acting outside policy
  (`docs/architecture/fleet_autonomy.md`):
  - **ActionPolicy decision point** (`orchestration/action_policy.py`, OS-5.24): the single
    gate consulted before ANY autonomous mutating operational action. Per-action tiers
    (`auto` / `auto_notify` / `approval_required` / `forbidden`), durable per-action+target
    rate limits, blast-radius caps, and UTC maintenance windows — replacing the binary
    env-flag autonomy cliff. Policies load from YAML (`ACTION_POLICY_PATH`, default = the
    shipped conservative `deploy/action-policy.default.yml`: everything mutating requires
    approval) plus runtime KG `governance_rule` overrides (scope `action_policy`) that win
    over file rules. Decisions fail CLOSED, are audit-logged as `ActionDecision` nodes (also
    the durable rate/blast ledger), and queue-approval reuses the existing fleet approvals
    flow via `ActionApproval` nodes (`GET /api/fleet/approvals` lists them, `.../grant`
    resolves them in place).
  - **Desired-state fleet reconciler** (`orchestration/fleet_reconciler.py`, OS-5.25):
    opt-in leader-only maintenance tick (`FLEET_RECONCILER`, default off) that diffs
    `deploy/mcp-fleet.registry.yml` (+ optional `FLEET_DESIRED_STATE_PATH` override) against
    a pluggable `FleetObserver` (default: OS-5.15 FleetEvent stream + docker CLI when
    present) and converges each divergence through the ActionPolicy gate and the injectable
    `FleetActuator` protocol (`orchestration/fleet_actuation.py`). The default actuator is
    DRY-RUN — it records intended actions as `ActionExecution` nodes and notifies, mutating
    nothing; `FLEET_ACTUATOR=docker` selects the reference docker actuator; Portainer/Swarm
    actuation is deployment-wired via `set_fleet_actuator()`. Storm guard
    (`FLEET_RECONCILER_MAX_ACTIONS`/tick), human-granted approval drain, and a
    `ReconcileReport` node per pass.
  - **Remediation playbooks** (`knowledge_graph/adaptation/remediation_playbooks.py`,
    OS-5.26) on the OS-5.15 `register_playbook()` seam: `service_down` (confirm via observer
    → policy-gated restart → durable verification watch → escalate on deny/failure),
    `service_flapping` (back off + escalate to a human), and `resource_pressure`
    (notify + propose, never auto-act). Every step outcome lands on the originating
    FleetEvent node (`remediation_log` / `remediation_status`); escalation = approval-queue
    entry + notification through the KG-2.42 notifier seam.
  - **Health-gated deploy + rollback** (`orchestration/deploy_watch.py`, OS-5.27):
    `watch_deploy()` schedules a durable `deploy_watch` task after every autonomy-triggered
    deploy/restart; the worker probes the FleetObserver until the recorded deadline
    (`DEPLOY_WATCH_WINDOW` / `DEPLOY_WATCH_POLL`; a reaper-requeued watch resumes its
    ORIGINAL window). Sustained green ⇒ `DeployWatch` success node; failure ⇒ default
    `on_fail` = ActionPolicy-gated `rollback_service` + operator escalation; zero
    observations ⇒ notify only (never roll back on zero evidence).
  - **Strangled `capabilities/auto_healing.py`** — the dormant `AutoHealingEngine` shell
    (disabled by default, never-wired skill_evolver/fallback_router hooks) is deleted. Its
    useful bit — threshold-counted repeated-failure escalation — is absorbed into
    `graph/parallel_engine.py::_escalate_repeated_failure`, which files a `failure_gap`
    Concept topic through the live AHE-3.18 propose-only remediation chain.
- **Gateway middle-tier hardening (CONCEPT:OS-5.23)** — Python-tier observability and
  backpressure for the API gateway (Tranche 2):
  - **Prometheus metrics** (`observability/gateway_metrics.py`): pure-ASGI middleware +
    `GET /metrics` mounted by `register_graph_routes` (gateway AND agent-webui), emitting
    `agent_utilities_gateway_requests_total{route,method,status}` (route = TEMPLATE, bounded
    cardinality), `_request_duration_seconds{route}`, `_in_flight_requests`,
    `_rate_limited_total{tenant}`, `_engine_requests_total{op,outcome}` and
    `_engine_breaker_state{endpoint}` — naming mirrors the Rust engine's `epistemic_graph_*`
    series. `prometheus-client` is the new optional `metrics` extra with a graceful no-op
    fallback; `/metrics` is exempt from the OS-5.14 identity middleware (scrapers can't mint
    JWTs). Flag: `GATEWAY_METRICS` (default on).
  - **Per-tenant token-bucket rate limiting** (`gateway/rate_limit.py`): ASGI middleware
    mounted INSIDE the identity middleware so the bucket key uses the server-minted
    ActorContext (tenant → authenticated actor id → client IP). 429 + `Retry-After` + JSON
    body; `/metrics` and health routes exempt. Flags: `GATEWAY_RATE_LIMIT` (req/s, default 0 =
    off) and `GATEWAY_RATE_BURST` (default 2× rate). Buckets are per-process (documented:
    N workers → N× the configured rate).
  - **Engine circuit breaker** (`knowledge_graph/core/engine_breaker.py`): every
    `GraphComputeEngine` call is guarded by ONE shared breaker per endpoint —
    `ENGINE_BREAKER_THRESHOLD` (5) consecutive connect/timeout failures open the circuit,
    callers fail fast with the typed `EngineCircuitOpenError` (a `ConnectionError`), and a
    half-open probe after `ENGINE_BREAKER_COOLDOWN` (15s) heals it. Application-level errors
    never trip it.
  - **Multi-worker readiness**: `GATEWAY_WORKERS` (default 1 = historical single-process
    behaviour) pre-forks workers on one shared listen socket, forking BEFORE app build so the
    flock host-lock elects exactly ONE KG host daemon among the workers (verified against
    `host_lock.py`); per-process state audited and documented in
    `docs/architecture/gateway_scaling.md`; dashboard `ConfigManager.get_all_services` now
    always re-reads the shared YAML (no stale per-worker cache).
- **Tiered RLM code sandbox with an intelligent capability router (CONCEPT:ORCH-1.38)** —
  the RLM REPL's hardcoded `use_wasm/use_container/local` if-elif (`agent_utilities/rlm/repl.py`)
  is replaced by a uniform `Sandbox` contract (`agent_utilities/rlm/sandboxes/`) with **four real
  backends** and a deterministic, in-process `ast`-based router that picks the cheapest backend a
  snippet can run on (`monty > wasm > docker > local`) and **escalates on rejection** (a
  `SandboxRejected` advances to the next tier; `SandboxFatalError` still fast-fails per ORCH-1.29).
  - **monty** (`pydantic-monty`, new core dep) — the fast (~0.5ms) isolated **default** tier; the
    only isolating backend that still serves the RLM host helpers (`rlm_query` etc.) natively via
    pause/resume external functions. No daemon, no root; rejects classes/3rd-party libs → escalate.
  - **docker/podman** — hardened from the old stub: `--network none`, memory/cpu/pids caps,
    `--cap-drop ALL`, timeout, full-namespace context, **and a per-run UDS host-callback bridge** so
    an isolated container can still call the RLM helpers. `SandboxFatalError` on dead container.
  - **wasm** — real **CPython-on-WASI** under wasmtime (replaces the 3-task emulation stub):
    isolated full-stdlib compute, epoch-timeout, payload via `scripts/provision_rlm_wasm.py` or
    `$RLM_WASM_PYTHON` (`[sandbox]` extra). v1 has no host bridge → self-contained compute only.
  - **local** — the legacy restricted `exec()`, kept as the always-available floor.
  - Selected via `RLMConfig.sandbox` (`auto`|`local`|`monty`|`wasm`|`docker`) / `RLM_SANDBOX`;
    legacy `use_monty`/`use_wasm`/`use_container` honored as overrides. Default flips to `auto`,
    so out-of-the-box RLM gets monty's fast isolated path with helpers instead of raw `exec()`.
- **Orchestration flow-diagram surfacing (CONCEPT:ORCH-1.37)** — the execution-flow
  Mermaid diagram generated by the ORCH-1.8 `WorkflowVisualizer` (and the per-run graph
  diagram from `GraphResponse`/`WorkflowDefinition`) is now returned by the `graph_orchestrate`
  MCP tool instead of only being logged to stdout or left as a KG node property. `swarm`,
  `compile_workflow` and `execute_workflow` gain an additive null-safe `"mermaid"` JSON key;
  `execute_agent` returns a JSON wrapper `{"output","mermaid"}` when a diagram is produced
  (bare output string otherwise, preserving backward compatibility). Implemented additively in
  `agent_utilities/mcp/kg_server.py` (4 handlers), `agent_utilities/orchestration/agent_runner.py`
  (`run_agent(return_mermaid=...)` + `streamdown=True` on the routed-graph execution), and
  `agent_utilities/orchestration/manager.py` (flag pass-through). Internal `run_agent` callers
  keep their bare-string contract (`return_mermaid` defaults to `False`). Validated by
  `tests/unit/test_orchestrate_mermaid_surfacing.py` and the `test_workflow_e2e.py` harness.

### Performance
- **Execution-loop optimization (CONCEPT:ORCH-1.37 perf)** — collapsed the
  plan→execute→verify→re-plan loop for simple tasks from dozens of LLM round-trips (with
  two `request_limit=50` thrashes + 32K context overflows observed) down to a single
  execution call. Changes: (1) **direct-dispatch fast-path** in `router_step` — when the
  task resolves to a single connected MCP server, run the agent once with that toolset and
  return, skipping planner + memory_selection + verifier (`GRAPH_DIRECT_DISPATCH=false` to
  disable); (2) **capability-gated tool injection** in `executor._get_domain_tools` — the 23
  generic developer/sdd tools are only injected for code/spec nodes, so an MCP-server agent
  sees only its server's tools (fixes wrong-tool selection, reclaims 10–28% of a 32K window);
  (3) **proportional verification** in `verification.verifier_step` — skip the LLM quality
  gate for trivial/≤1-step read-only results, constrain the verifier to the literal query
  (no demanding unrequested fields/pytest), and cap re-planning to once-when-nothing-salvageable
  (both scoring branches); (4) **explicit `UsageLimits`** on executor/planner/verifier agent
  runs (default pydantic-ai cap is 50; now 8/6/4 via `AGENT_/PLANNER_/VERIFIER_REQUEST_LIMIT`);
  (5) routing now prefers a structured-output-capable model (config `can_route` flipped to the
  9B in the deployment).

### Fixed
- **Dynamic/specialist/planner/verifier agents ran injected `RunContext[AgentDeps]` tools with
  `deps=None` (or raw `GraphDeps`)** → `'NoneType'/'GraphDeps' object has no attribute
  'workspace_path'`, aborting any graph node that called a dev/sdd tool. Added
  `executor.agent_deps_from_graph(...)` adapter and pass proper `AgentDeps` at all four spawn
  sites (`_router_impl` dynamic agent, `executor.execute_specialist_step`,
  `hierarchical_planner.planner_step`, `verification.verifier_step`).
- **`asyncio.wait_for(stack.enter_async_context(mcp_toolset), ...)` entered an MCP stdio
  toolset's anyio cancel scope in a child task** while the `AsyncExitStack` exited it in the
  outer task → "Attempted to exit cancel scope in a different task than it was entered in"
  (and wedged processes on teardown). `orchestration/engine.py` now uses `asyncio.timeout()`
  (current-task) instead of `asyncio.wait_for()` (new-task).
- **`_execute_tool` passed unset `Field()` defaults through as raw `FieldInfo` objects**
  (`'FieldInfo' object has no attribute 'replace'` / "not JSON serializable") for internal
  callers, the REST gateway, and tests. It now resolves `FieldInfo` defaults for omitted
  params, matching the FastMCP schema layer. Regression test in
  `tests/unit/test_execute_tool_field_defaults.py`.
- **Native database traversal + full Onyx connector parity (CONCEPT:ECO-4.33)** —
  agent tools (`db_tables` / `db_schema` / `db_query` in `agent_utilities/tools/db_tools.py`,
  gated `DB_TOOLS`) that let an agent (incl. RLM recursive agents, ORCH-1.1) natively
  traverse a database — list tables, inspect schema, run live read queries — **universally
  across PostgreSQL, MySQL/MariaDB, MS SQL Server, Oracle, SQLite, and MongoDB** via the one
  `UniversalConnector` abstraction (KG-2.9); the DSN scheme selects the backend. Read-only by
  default (DDL/DML deny-list; writes require `DB_TOOLS_ALLOW_WRITE=1` and commit via `write()`),
  with `{ALIAS}_DSN` secret resolution. A database is thus both an *ingestion* source (the
  `database` document-source connector, ECO-4.25) and an *interactive* tool — a capability Onyx
  lacks entirely (Onyx ships **zero** database connectors). Also **completes the Onyx
  connector-parity catalog to 48/48** (added `blob`, `drupal_wiki`, fixed `google_site`).

### Fixed
- **Per-model `base_url` collapsed onto one endpoint in split vLLM deployments** — the graph
  engine threads a single graph-level `base_url` into `create_model` for every role
  (`engine.py` router/agent/embedding deps), and `create_model` only applied a model's
  registered `base_url` when the caller passed none — so a router model served on a *separate*
  host (e.g. `qwen-lite` on `vllm-lite.arpa`) was sent to the KG model's endpoint
  (`vllm.arpa`), 404ing on every routing/planning call. `create_model`
  (`agent_utilities/core/model_factory.py`) now treats the model registry as the source of
  truth for *where* a model is served: a registered per-model `base_url` wins over a
  caller-supplied default; unregistered models still honor an explicit `base_url`. Regression
  test in `tests/unit/test_model_factory_base_url.py`.
- **Ingestion/evolution daemon bugs surfaced by a full re-ingest (CONCEPT:KG-2.7 / KG-2.8 / KG-2.12):**
  - *Document ingest "No files found"*: the worker's `SimpleDirectoryReader` excluded every file
    because the research store lives under `~/.local/share/...` (a dot-dir parent makes all files
    "hidden"). Pass `exclude_hidden=False` (+ `recursive=False`, `required_exts`) so PDFs ingest.
  - *Parallel multi-repo `enrich_comm` race*: the community-detection step used one shared
    transient tenant `{graph}__enrich_comm` across all concurrent codebase jobs, so a finishing
    job deleted a sibling's tenant mid-run → "Graph not found". Now a unique per-job tenant name.
  - *`deep_analysis` AttributeError*: `search_hybrid` referenced `self.hybrid_retriever` which was
    unset on the background-task host path. Lazy-ensure it on first use.
- **Directory-of-documents misclassified as a codebase (CONCEPT:KG-2.7)** —
  `ContentType.classify` blindly mapped *every* directory to `CODEBASE`, so ingesting a folder
  of papers (`~/.local/share/.../research/papers`, a ScholarX corpus) routed through the codebase
  adaptor and produced no `Document`/`Concept` nodes — the only workaround was forcing
  `content_type="document"`. Detection is now composition-aware (`ContentType._classify_dir`): a
  packaging/VCS marker (`pyproject.toml`, `package.json`, `.git`, …) is a definitive codebase
  signal, otherwise the directory's non-vendored files are sampled and a document-dominant folder
  classifies as `DOCUMENT`. Vendored/build subtrees (`.venv`, `node_modules`, …) are pruned so a
  doc corpus carrying a bundled virtualenv isn't misread as code, and sampling is capped for huge
  trees. Empty/ambiguous dirs still default to `CODEBASE` (unchanged). A paper directory now
  ingests correctly out of the box — no `content_type` override needed.

### Changed
- **Faster, throttled codebase ingestion (CONCEPT:KG-2.7 / KG-2.8)** — five changes so
  large-repo (re-)ingest stays cheap and bulk loads can't saturate the engine:
  - *Pre-hash skip* (`enrichment/pipeline.py`): files are content-hashed **before** parsing,
    so an unchanged file costs one local sha256 instead of a Rust-engine `parse_file`
    round-trip. Re-ingest of an unchanged repo no longer pays the full per-file RPC.
  - *Git-aware delta* (`ingestion/engine.py`): the codebase adaptor records the repo HEAD sha
    as a delta watermark and, on re-ingest of a git work-tree, enriches only the `*.py` files
    `git diff` reports changed — turning a whole-tree walk into a single `git diff` + a handful
    of parses. First ingest / non-git / git failure fall back to the full walk.
  - *Lite-model card backfill* (`core/engine_tasks.py`): the background capability-card daemon
    defaults to the lite chat model (`KG_CARD_MODEL=heavy` to override) and acquires a shared
    `background_throttle` slot, so card generation can't monopolize the engine.
  - *Read/ingest plane isolation* (`core/engine_tasks.py`): heavy task types
    (codebase/document/deep_analysis/synthesize/…) execute inside the shared `background_slot`,
    yielding to interactive (foreground) work and staying within the global concurrency cap.
  - *deep_analysis gating* (`core/engine_tasks.py`): while a bulk codebase ingest is draining,
    `deep_analysis` runs flat (`max_depth=0`) so its recursive, 0-node, blocking-LLM fan-out
    can't flood the queue ahead of structural ingest.
  - *Batched parse over the wire (CONCEPT:KG-2.16)*: the engine now has a `ParseFiles` op
    (epistemic-graph ≥ 0.27.0) that parses N files in one round-trip (rayon-parallel,
    fault-tolerant per file, ordered results). `enrich_files` sends changed files via
    `parse_files` in chunks (`KG_PARSE_BATCH`, default 128) instead of one RPC per file.
    Capability-gated: the client probes `Health.ops` and falls back to per-file `parse_file`
    against an engine that predates the op, so the cutover is zero-flag-day.

### Added
- **Document-source connector framework + Onyx parity (CONCEPT:ECO-4.25–4.29)** — a
  `load`/`poll`/`slim` connector abstraction (`agent_utilities/protocols/source_connectors/`)
  that ingests external documents — websites, filesystems, **databases (PostgreSQL, MySQL/
  MariaDB, MS SQL Server, Oracle, SQLite, MongoDB via `UniversalConnector`)**, and the entire
  `agent-packages/agents/*` MCP fleet — into the KG as first-class `Document` + `Chunk`
  ontology objects. Provenance: the Onyx/Danswer connector surface
  (`LoadConnector`/`PollConnector`/`SlimConnector`), ported onto the *semantic* core so
  ingested documents inherit OWL semantics, bitemporal slicing, reified `HAS_CHUNK`/`CHUNK_OF`
  links, and the entailment-aware ACLs of KG-2.46 — capabilities a flat vector index cannot
  offer. Includes resumable **checkpointed incremental poll** (ECO-4.26) round-tripping through
  the existing `DeltaManifest` (KG-2.8); a self-registering **registry + factory** (ECO-4.27);
  **external permission sync** (ECO-4.28) mapping source ACLs onto KG-2.46 markings/`read_roles`;
  and a generic **MCP agent-package adapter** (ECO-4.29) with a preset catalog that encodes an
  explicit **Onyx connector-parity map** (every Onyx source routes to a native package or a
  generic web/rest/database/filesystem connector). Wired end-to-end: `ContentType.CONNECTOR`
  ingestion adaptor, `kg.ontology.run_connector(...)`, the `source_connector` MCP tool, and
  `/connector/*` REST routes.
- **Contextual-retrieval enrichment (CONCEPT:KG-2.50)** — per-chunk situating context computed
  before embedding and prepended to the embedding input (Anthropic "Contextual Retrieval"),
  with an LLM path and a deterministic offline heuristic; wired into the KG-2.48
  `DocumentProcessor` (default OFF; on for connector ingest) and stored on chunk nodes.
- **Query analysis (CONCEPT:ECO-4.32)** — source-type + time-window filter derivation from a
  natural-language query (LLM or deterministic fallback) plus a citation processor, wired as an
  opt-in pre-filter on `HybridRetriever.retrieve_hybrid(query_analysis=True)`.
- **Media generation + transcription gateway (CONCEPT:ECO-4.30 / ECO-4.31)** — self-hosted
  image (`flux.2` + Stable Diffusion 3.5), video (`hunyuanvideo`), speech (`xtts`), and
  transcription (`faster-whisper`) via lazy-`httpx` clients in `agent_utilities/ecosystem/media/`,
  exposed as agent tools under the `MEDIA_TOOLS` gate. The `flux.2`/`hunyuanvideo` stale
  compose templates were rewritten and a `stable-diffusion` (SD3.5) service added, all targeting
  the GB10 host via the swarm-launcher pattern with a light, on-demand footprint.

- **Ontology System — Palantir-Foundry-parity, graph-native (CONCEPT:KG-2.26 / KG-2.38–KG-2.48 + KG-2.42)** —
  a first-class object/link/function/action layer at `agent_utilities/knowledge_graph/ontology/`,
  reached through `kg.ontology` (`KnowledgeGraph.ontology` → `OntologySystem`). It binds real,
  import-populated registries to the *live* epistemic backend the rest of the KG uses (store /
  semantic / retrieval), so interface targeting, derived-property compute, Functions-on-Objects,
  and ACL enforcement resolve against the actual graph — not a parallel shell. Provenance: the
  Palantir Foundry / AIP docs (`ontology/object-types`, `interfaces`, `value-types`,
  `functions/overview`, `action-types/overview`, `object-edits`, object-set `SEARCH_AROUND`,
  document-processing), cited per-module in docstrings; identifiers are named from purpose, not
  the vendor. New concepts:
  - **Property types (CONCEPT:KG-2.47, `ontology/property_types.py`)** — the scalar/geo/
    vector-embedding/array/struct type vocabulary that drives node-table column DDL and
    write-path coercion (`column_type_for`).
  - **Value types (CONCEPT:KG-2.39, `ontology/value_types.py`)** — constrained semantic types
    (EmailAddress, Percentage, …) compiled to reusable SHACL `sh:PropertyShape` / named
    `rdfs:Datatype` and gated by the SHACL validator on write.
  - **Interfaces (CONCEPT:KG-2.38, `ontology/interfaces.py`)** — abstract shapes a concrete
    object type implements; the programmatic-targeting resolver expands an interface to its
    implementers.
  - **First-class links (CONCEPT:KG-2.26, `ontology/links.py`)** — named directed link types +
    many-to-many **junction reification** onto the existing graph-write path, with reverse
    traversal.
  - **Functions (CONCEPT:KG-2.41, `ontology/functions/`)** — typed, versioned, governed user
    functions (`PLAIN | ON_OBJECTS | QUERY`) over one audited runtime.
  - **Derived properties (CONCEPT:KG-2.40, `ontology/derived_properties.py`)** — read-time
    computed properties dispatched across `FUNCTION / CYPHER / SPARQL / EMBEDDING`.
  - **Action types (CONCEPT:KG-2.42, `knowledge_graph/actions/`)** — submission-criteria-gated,
    typed-side-effecting, batchable, notification/webhook-dispatching, and **revertable**
    actions over the edit ledger.
  - **Durable object edits (CONCEPT:KG-2.43, `ontology/edits/`)** — a bitemporal edit ledger
    (`EditLedger`, `JsonlEditSink`, `WriteBackRouter`) with per-object history and
    `revert_edit`/`revert_object`.
  - **Indexing lifecycle (CONCEPT:KG-2.44, `ontology/indexing/`)** — content-hashed
    `ObjectIndexFunnel` + `StalenessLedger` driving the SAME live search index.
  - **Object Set service (CONCEPT:KG-2.45, `ontology/object_set.py`)** — composable
    STATIC/derived handles with `filter` / `search` / `search_around` / `pivot` / `aggregate`
    and set algebra (union/intersection/difference).
  - **Fine-grained permissioning (CONCEPT:KG-2.46, `ontology/permissioning.py`)** —
    entailment-aware ACL **marking propagation** + `restricted_view` row-drop and `enforce`
    on the read path.
  - **Document processing (CONCEPT:KG-2.48, `ontology/document_processing.py`)** —
    extract→chunk→embed→link (`DocumentProcessor`, `process_document`).

  Wired into `knowledge_graph/facade.py` (`kg.ontology`), the `ontology_*` MCP tools in
  `mcp/kg_server.py` (`ontology_property_types`, `ontology_value_types`, `ontology_interface`, …),
  and an operator **UI** in agent-webui — `/api/enhanced/ontology/*` routes (object-types,
  property-types, interfaces, object-set search/aggregate/pivot/search-around, object edit/revert,
  function invoke, derive, document process) plus the **ObjectExplorerView / ObjectView /
  VertexView** React views. Unique value-adds vs Foundry: OWL/SHACL-backed interfaces + value
  types (reasoning + validation), embedding/cypher/sparql-backed derived properties, reified
  junction links, entailment-aware ACL marking propagation, a bitemporal edit history, a
  self-evolving ontology, and the Rust epistemic engine underneath.
- **2026 reasoning-RL gap closure (CONCEPT:AHE-3.15 / 3.16 / 3.17 + AHE-3.1)** — implements the
  high-leverage gaps from `.specify/specs/reasoning-rl-2026/` (the agentic adaptations, not
  re-implementing GRPO which the AHE-3.1 spine already covers):
  - **AHE-3.15 Agent-Step Policy Optimization (ARPO, arXiv:2507.19849)** — `graph/agent_step_po.py`
    (`step_entropy`, `should_branch`, `write_back_step_credit`) + `RewardDecomposer.step_advantages`;
    `SubagentLifecyclePolicy.determine_route` now branches to `fan_out` on a high-entropy decision
    step (bounded by `ARPO_MAX_BRANCHES`) and per-step advantage is written back into the capability
    reward-EMA.
  - **AHE-3.16 Test-Time Diversity (VPO, arXiv:2605.22817)** — `graph/test_time_diversity.py`
    (`mean_pairwise_distance`, `select_diverse` MMR best-of-k) + an effort-derived
    `ReasoningBudget.diversity_width` so harder queries fan out wider/more diverse.
  - **AHE-3.17 Preference-Corpus Reliability (RAPPO/TI-DPO/InSPO/DPO)** — `harness/preference_pairs.py`
    (`PreferencePair`, `PreferencePairExporter`, `reliability_filter`, `attach_token_weights`,
    `with_reflection`) consolidating the eval corpus + distilled episodes + corrections into a
    DPO-ready pair store; wired live via `FeedbackService.export_preference_pairs`.
  - **AHE-3.1 reward-primitive hardening** — `graph/training_signals.py`:
    `batch_normalized_advantage(length_unbiased=…, mode=…, group_ids=…)` (Dr.GRPO σ-bias removal +
    GRPO/REINFORCE++ grouping), `dynamic_sample` (DAPO), `entropy_progress_weights` (EP-GRPO,
    consumed by `step_advantages`), `token_regulation` (TR-GRPO). All opt-in, defaults unchanged;
    GSPO/DPPO trainer mechanics deferred until a trainer consumes them. Docs: AHE pillar page + C4
    diagram; concepts.yaml regenerated (115 concepts). Tests: `tests/test_training_signals.py`,
    `tests/test_preference_pairs.py`, `tests/test_agent_step_po.py`, `tests/test_time_diversity.py`.
- **Job-queue controls on `graph_ingest` (CONCEPT:KG-2.8 queue control)** — operators can now
  manage the ingestion queue over MCP: `action="cancel"` (job_id → terminal `cancelled`),
  `action="clear"` (`target_path` = status filter `pending|running|completed|failed|cancelled|
  zombie|all`, default `completed`; `zombie` clears only `running` tasks not owned by the live
  host token), and `action="prioritize"` (job_id, `target_path`=`high|normal`). The worker poll
  is now priority-aware — it claims `priority='high'` pending tasks before others (two-tier poll,
  since the L1 interpreter strips `ORDER BY`). Backed by `cancel_task`/`clear_tasks`/
  `prioritize_task` on the task manager + tests in `test_task_queue_controls.py`.
- **Zombie/stuck task reaper for the KG ingestion queue (CONCEPT:KG-2.8 durability)** —
  when a worker/host process dies mid-task (crash / SIGKILL / redeploy), the `Task` was
  stranded in `running` forever and never re-claimed, silently wedging that ingestion (we hit
  exactly this: 43 `running` vs 8 workers after host hand-offs). The host daemon now runs a
  `task_reaper` maintenance job (`engine_tasks.py`, default every 120s) that uses the singleton
  host lock as ground truth: each claim stamps `claimed_by = <host-token>` + `claim_unix`, and
  since exactly one host runs workers, any `running` task **not** stamped with the *live* token
  (past a 90s hand-off grace) is an orphan from a dead host → reset to `pending` for re-claim.
  An absolute-runtime backstop (`KG_TASK_MAX_RUNTIME_SEC`, 2h) catches a wedged-but-alive worker,
  and a poison-pill cap (`KG_TASK_MAX_REQUEUE`, 3) fails a task that repeatedly kills its worker
  instead of looping. Host-gated; configurable via `KG_TASK_REAPER_DAEMON`/`_INTERVAL`/
  `_ORPHAN_GRACE_SEC`. Tests in `tests/unit/knowledge_graph/test_task_reaper.py`.

### Fixed
- **Document ingestion is ~16–50× faster — embeddings are batched, not per-chunk** —
  the async document worker (`knowledge_graph/core/engine_tasks.py`) embedded chunks one at a
  time inside the ingest loop (`embed_model.get_text_embedding(chunk)`), i.e. one network
  round-trip to the embedding service **per chunk**, which made a single PDF take minutes and
  contradicted the project's "batch over the wire, never per-element" rule. The loop is now two
  passes: pass 1 dedups (O(1) id-keyed) and collects new chunks; pass 2 embeds them all via
  `get_text_embedding_batch` in sub-batches of 64 (with a per-chunk fallback when the model
  lacks the batch API). Dedup, stale-delete, node properties, and metrics are unchanged.
- **L1 epistemic-graph Cypher: `WHERE … OR …` and inline-literal relationship ids now work** —
  the in-memory interpreter's `_parse_where` split only on `AND`, and `_exec_rel_match` required
  `{id:$param}`; both silently fell through to the read-only legacy reader and returned `[]` for
  any `OR` clause or a relationship anchored by a quoted literal id — a footgun where "I can't
  parse this" masqueraded as "no rows". WHERE is now parsed into DNF (OR of AND-groups) via
  `_parse_where_or`, relationship anchors accept `$param` **or** a quoted literal, and a
  genuinely unsupported shape is now **logged loudly** before deferring to legacy. Backed by
  new tests in `tests/unit/knowledge_graph/test_epistemic_backend_cypher.py`. (Note: this was a
  query-interpreter limitation, **not** a persistence bug — writes always landed; only certain
  read shapes under-matched.)
- **`graph_ingest` no longer blocks on document/codebase ingestion (footgun removal)** —
  passing `content_type` previously routed ingestion through the *synchronous* `IngestionEngine`,
  so a single PDF/markdown could hang the MCP caller for many minutes with no job id to poll.
  `graph_ingest(action="ingest")` now **auto-detects** the content type per path via
  `ContentType.classify` (the single source of truth, CONCEPT:KG-2.7) and **always routes the heavy
  categories (document, codebase) through the async durable job queue** — even when `content_type`
  is given explicitly. `content_type` is demoted to an internal override the agent never needs to
  set; the lightweight special categories (config/prompt/skill/mcp_server/kb/conversation/policy)
  still fold through the unified engine inline. Live-path tests in `tests/unit/mcp/test_kg_server.py`
  assert a `.pdf` (and an explicit `content_type="document"`) enqueue an async `submit_task` rather
  than running inline.
- **Document ingestion jobs reported `nodes: 0` in per-category metrics** — the async document worker
  recorded persisted-Article counts under `chunks_added`, but `aggregate_ingest_metrics` reads
  `nodes_added`/`edges_added`. Completed document jobs therefore always showed 0 nodes despite
  persisting one Article node per new chunk. The worker now also surfaces `nodes_added`/`edges_added`
  (`knowledge_graph/core/engine_tasks.py`), so document-ingest node counts are visible.

### Added
- **Structured-output contracts on the RLM subagent fan-out (CONCEPT:ORCH-1.12)** —
  extends the Predict-RLM runtime so subagents return *schema-constrained, typed* values instead of
  free-form prose (the "external attention mask" pattern from the RLM-structured-outputs writeup).
  Previously the contract only existed at the **root** signature; `rlm_query`/`run_parallel_sub_calls`
  returned bare strings, forcing the parent to re-read and re-classify many unstructured blurbs.
  - **`SchemaContract` normalizer** (`rlm/schema.py`) — `from_spec()` accepts a Pydantic `BaseModel`,
    a primitive (`int`/`bool`/`str`/`float`), a typing generic (`list[Model]`, `dict[...]`), or a raw
    JSON-Schema `dict` (e.g. `{"type": "boolean"}`) and normalizes all to plain JSON Schema via
    `model_json_schema()`/`TypeAdapter`. `.validate()` returns `(ok, coerced_value, error)` with
    `path: message` errors; raw-dict validation uses `jsonschema` when present, with a non-silent
    shallow fallback otherwise.
  - **Per-subagent `schema=`** — `rlm_query(prompt, context, schema=…)` and a per-call `"schema"` key
    in `run_parallel_sub_calls` (incl. the depth-floor `pydantic_ai` fallback via `output_type`) now
    return the **coerced typed value**, letting the parent route on a clean boolean/model/list.
  - **Validate-on-FINAL with retry-don't-restart** — the existing `run_full_rlm` loop is reused: a
    sub-RLM whose `FINAL` violates the contract is shown the JSON Schema + specific errors and retries
    with REPL state intact (`rlm/repl.py`); the contract is injected into the sub-REPL prompt at
    startup and `schema=` is advertised in the helper docs (Wire-First).
  - **Root contract generalized** — `run_rlm(..., output_type=…)` (`rlm/runner.py`) and
    `_generate_instruction_prompt` (`rlm/predict_rlm.py`) accept/show primitive/generic/model output
    specs, not just a free-form string.
- **Schema-Pack 2.0 — domain retrieval+extraction+reasoning profiles (CONCEPT:KG-2.22–KG-2.37)** —
  turns the domain Schema Pack from a type-selection profile into a fully-wired domain profile,
  closing gbrain-class gaps while leveraging our OWL reasoner and bi-temporal store for capabilities a
  flat brain layer cannot match:
  - **KG-2.22 Pack-Driven Retrieval Signals** — declarative per-type recency decay (over bi-temporal
    `event_time`, with `graph_search(as_of=…)` for "knowledge state as of date D"), per-source trust
    weighting, and score-discontinuity **autocut**, applied in `HybridRetriever.retrieve_hybrid`
    (`retrieval/autocut.py`). No-op under the default `core` pack (bit-for-bit backward compatible).
  - **KG-2.33 Zero-LLM Pack-Driven Link Inference** — `knowledge_graph/kb/link_inference.py`
    materialises typed edges (supports/weakens/cites/uses-dataset) from pack-declared **ReDoS-bounded**
    regex rules on write; wired into `EntityClaimExtractor.extract_and_persist`.
  - **KG-2.34 Relational-Intent Retrieval** — `knowledge_graph/retrieval/relational_intent.py` parses
    "which papers support X" / "what is cited by Y" deterministically and walks typed edges; merged
    additively into hybrid retrieval (no-op for non-relational queries).
  - **KG-2.35 Schema-Pack Lifecycle & Audit** — `models/schema_pack_loader.py` resolves the active pack
    (`GRAPH_SCHEMA_PACK` > config > `core`) and threads it into the engine/retriever (previously
    pack-blind); `models/schema_pack_audit.py` records out-of-pack candidate types under EXCLUSIVE packs
    (observe-only, privacy-hashed). Exposed via `graph_configure(action="schema_pack"|"schema_candidates")`.
  - **KG-2.36 Pack-Driven OWL Closure** — packs declare edge types as transitive/symmetric/inverse OWL
    object-properties, unioned into the `owl_bridge` reasoning sets so multi-hop support chains and
    `cites`/`cited_by` inverses are inferred **for free** (idempotent fixpoint).
  - **KG-2.37 Research-State Domain Pack** — flagship `research-state` pack realising the
    [garrytan/gbrain#587](https://github.com/garrytan/gbrain/issues/587) "academic literature state"
    use case; adds dedicated `WEAKENS` / `USES_DATASET` edge types.
- **Tool gap-fill resolver for workflow materialization (CONCEPT:ECO-4.0)** — `graph/tool_resolver.py`
  `resolve_tools`: when a workflow template needs a tool that isn't bound (e.g. a gitlab-pr tool), substitute
  an available tool providing the same capability (via the capability index) or surface a precise gap
  (`resolved`/`filled`/`missing`). Pure + testable (inject `available`/`designate_fn`); `resolve_agent_tools`
  derives them from a live engine best-effort and **passes tools through unchanged when availability is
  undeterminable** (no hot-path regression). Wired defensively into `ParallelEngine._execute_agent` before
  `create_agent(tool_tags=…)`. Full live tag↔capability binding is staged (see WORKFLOW_ABSTRACTION_STRATEGY).
- **Cheap input-scoped cycle watermark (CONCEPT:KG-2.7)** — `golden_loop._state_watermark` now uses an
  input-scoped node **count** (one Cypher query, no embedding transfer) instead of fetching all ~5k embedded
  nodes; the unchanged-graph skip path is **~13× faster** (live-measured 92.9s → 7.0s). Falls back to the full
  `(id,status,content_hash)` hash when `query_cypher` is unavailable. (Caveat: a pure in-place content update
  with no count change isn't detected by the count alone — use `force_assimilate`.)
- **Golden-loop breadth ingest + cycle monitoring + live validation (CONCEPT:KG-2.7)** —
  `run_one_cycle` gains an env-gated `breadth` stage (`KG_GOLDEN_BREADTH` + `KG_BREADTH_*_ROOTS`) that ingests
  the OSS/repos/docs corpus before assimilate (idempotent), and a `metrics` block (per-stage timings,
  `error_count`, `open_gaps`, duration) + a structured health log + a persisted `EvolutionCycle`
  (`orchestration_cycle`) node for monitoring. **Live-validated** against the running engine (5048 nodes): a
  full `run_one_cycle` completed error-free (`errors: []`), the assimilate watermark made the 2nd/3rd passes
  idempotently skip (`skipped: true`, unchanged), and the monitoring node persisted. Type matching across the
  assimilation engine is now **case-insensitive** (live graph stores capitalized labels like `Article`/`Concept`;
  enum values are lowercase) — caught by the live run.
- **Breadth-ingest orchestration + acceptance pilot (CONCEPT:KG-2.7, VU-10)** —
  `knowledge_graph/assimilation/breadth_ingest.py` (`discover_projects`/`classify_project`/
  `organize_libraries`/`run_breadth_ingest`) brings the whole corpus — OSS library categories, our ~62
  repos, and a docs batch — into the assimilation graph: pure-filesystem classification (language + target
  pillars) + injectable codebase/doc ingest (default = the content-addressed `IngestionEngine`, so unchanged
  sources skip). `knowledge_graph/assimilation/pilot.py` (`run_pilot`/`summarize`) is the acceptance harness:
  it runs the assimilation pass and asserts the hard invariant — **no already-built feature is re-proposed** —
  while emitting the ranked gaps for human comparison against the known Waves A/B/C. Thin CLI
  `scripts/run_assimilation_breadth.py` (`organize`/`ingest`/`pilot`) wires the live engine. Completes the
  Phase-0 graph-native assimilation substrate.
- **Assimilation MCP action + public pass entrypoint (CONCEPT:KG-2.7, VU-9)** —
  `graph_orchestrate(action="assimilate")` runs the graph-native assimilation pass (dedup → gap → synergy →
  rank; `task="synthesize"` also proposes grounded SDD plans; `task="force"` ignores the watermark), backed by
  the new `research/golden_loop.run_assimilation_pass`. The background golden-loop tick already runs this each
  cycle (daemon mode). The `agent-utilities-evolution` skill SOP is updated (separate `universal-skills` repo)
  to drive the graph-native stages, multi-source ingest, and idempotency.
- **Plan synthesis from KG neighborhood (CONCEPT:KG-2.7 / KG-2.10, VU-8)** —
  `knowledge_graph/assimilation/plan_synthesis.py`: `synthesize_plans` turns the top-ranked **open** gaps
  into grounded SDD plan proposals. `hydrate_feature` pulls each feature's neighborhood (sources, synergy
  partners, pillar); `synthesize_plan_for_feature` synthesizes a plan (injectable `synth_fn`; default tries
  the ORCH-1.27 `planner` role, falls back to a deterministic grounded template so it never hard-fails),
  persists it as an `SDDPlan` proposal + `feature -[ADDRESSED_BY{proposed}]-> plan`, and flips the feature to
  `proposed` so it is not re-proposed (idempotent). Grounded + deduped by construction — replaces the
  first attempt's per-paper plan generation. Propose-only (promotion reuses the AHE-3.14 gate at apply time).
- **Golden-loop assimilation stage + watermark idempotency (CONCEPT:KG-2.7, VU-7)** —
  `knowledge_graph/research/golden_loop.py` `run_one_cycle` now runs the graph-compute middle
  (`_run_assimilate`: dedup → auto-satisfy → synergy → rank) before topic intake, and reports the
  exclusion-filtered, leverage-ranked `ranked_gaps` (only `open_features` — satisfied/superseded/implemented
  features are never re-proposed). A **state watermark** over the assimilation input nodes
  ((id, status, content_hash)) makes the stage idempotent — an unchanged graph skips the work, so a re-run
  over the same corpus is a no-op (`force_assimilate=True` overrides). This turns the substrate into a live,
  self-running, delta-only loop.
- **Multi-source ingest adapters + granular idempotency (CONCEPT:KG-2.7, VU-6)** —
  `knowledge_graph/assimilation/ingest.py`: `ingest_documents` (PRD/BRD/SOW/tasks → `Requirement` nodes; new
  `RegistryNodeType.REQUIREMENT`) and `ingest_conversations` (chat/SDD transcripts → `Decision` nodes), with
  content-addressed idempotency — `canonical_source_id` collapses the same source ingested from different URIs
  (arxiv abs/pdf/version, DOI variants, URL, file path) onto one node, and `content_fingerprint` makes an
  unchanged re-ingest a no-op (skipped) while changed content updates in place. Granular: each item is hashed
  independently so a changed batch skips its unchanged members (per-paper skip) — cost grows with the delta,
  not the corpus.
- **Feature lifecycle ledger + assimilation close-out (CONCEPT:KG-2.7, VU-5)** —
  `knowledge_graph/assimilation/ledger.py`: `record_feature`/`set_status` maintain `SDDFeature` lifecycle
  nodes; `close_out` records an implemented feature by writing `feature -[DERIVED_FROM_RESEARCH]-> source` +
  `source -[ASSIMILATED_INTO]-> codebase` and flipping status to `implemented` (KG-2.7 US-1/3), closing the
  research→code provenance loop so it is never re-opened; `promote_feature_ledger` lifts the YAML
  feature/capability ledger into `SDDFeature` nodes; `ledger_state` gives an open/closed/by-status summary.
  `is_closed` now also consults the node's stored `status` (self-sufficient) and treats `DERIVED_FROM_RESEARCH`
  as a closing edge.
- **Synergy bundles + leverage ranking (CONCEPT:KG-2.7 / KG-2.5, VU-4)** —
  `knowledge_graph/assimilation/synergy.py`: `synergy_bundles` community-detects the feature graph (engine
  Louvain `community_detection` fast path, local connected-components fallback) and flags **cross-pillar**
  communities (spanning ≥2 of ORCH/KG/AHE/ECO/OS) as synergy bundles, linking members with
  `HAS_SYNERGY_WITH` — where the novel combinations live. `rank_features` scores the **open** gaps by
  leverage `source_count × (1 + centrality)` (engine PageRank fast path, local degree fallback) so the
  golden loop spends budget on the highest-impact gaps first. Duplicate (`SUPERSEDES`) edges are excluded
  from the synergy graph.
- **Auto gap analysis — `SATISFIED_BY` + `open_features` (CONCEPT:KG-2.7, VU-3)** —
  `knowledge_graph/assimilation/gap_analysis.py`: `auto_satisfy` embedding-matches every extracted feature
  (`SDD_FEATURE`/`CAPABILITY`/`ARTICLE`) against existing `CONCEPT` nodes and writes a candidate
  `feature -[SATISFIED_BY]-> concept` edge above threshold; `open_features` returns the features with no
  closing edge / closed status — the durable, queryable answer to "what have we NOT already hit?" and the
  only set the golden loop proposes against. This is the **"stop rediscovering already-built features"** fix.
  Closing edges are detected backend-portably via a `_rel` property marker (now also stamped on VU-2's
  `SUPERSEDES` edges) plus the node `status`. Incremental (`restrict_to`) and dry-run capable.
- **Cross-source feature dedup (CONCEPT:KG-2.7, VU-2)** — `knowledge_graph/assimilation/dedup.py`
  `dedup_features`: collapses the same capability appearing across a paper + an OSS library + our own code
  into one node with multi-source provenance. Pairwise cosine over embedded `SDD_FEATURE`/`CAPABILITY`/
  `ARTICLE` nodes (preferring the engine's batched `compute_similarity_edges`, local-numpy fallback) →
  `SIMILAR_TO` edges (score); union-find clusters above the duplicate threshold → highest-importance survivor
  with `survivor -[SUPERSEDES]-> duplicate`. Idempotent (MERGE on write) and incremental (`restrict_to` =
  newly-ingested ids → O(new·N)). The first graph-compute stage of the assimilation pipeline.
- **Assimilation engine schema foundation (CONCEPT:KG-2.7, VU-1)** — `models/knowledge_graph.py`: new
  `RegistryNodeType.SDD_FEATURE` + `SDDFeatureNode` (lifecycle-tracked feature: `concept_ids`,
  `research_sources`, `status` open→implemented/rejected/superseded, `sdd_path`, `codebase`) and registered
  assimilation edges `ADDRESSES` / `ADDRESSED_BY` / `RELEVANCE_SCORED` / `ASSIMILATED_INTO` /
  `DERIVED_FROM_RESEARCH` / `SATISFIED_BY` (UPPER_SNAKE values matching the live Cypher labels the golden-loop
  / relevance-sweep subsystem already writes — so the enum becomes the source of truth without splitting
  existing edges). Dedup/synergy/supersede reuse the existing `SIMILAR_TO` / `HAS_SYNERGY_WITH` / `SUPERSEDES`
  edges (no sprawl). The foundation for the graph-native assimilation pipeline (see
  `.specify/specs/ecosystem-evolution/PHASE0_IMPLEMENTATION_PLAN.md`).
- **MASS — Multi-Agent Social System swarm model (CONCEPT:ORCH-1.32)** — `graph/social_system.py`
  `MultiAgentSocialSystem` models the swarm as ``S=(f,g,G)``: archetype-tagged agents over an explicit
  interaction graph with local (neighborhood-scoped) observability, a co-evolution edge-update loop, and a
  P1–P4 **swarm-health** snapshot (degree-partition heterogeneity, topology variance, neighbor co-evolution
  OLS slope, Wasserstein-1 drift — reusing `population_drift.wasserstein1`). Wired live into
  `ParallelEngine.execute` (`_social_swarm_health`, built from agent roles + `depends_on` edges; surfaced in
  `ExecutionResult.telemetry["social_system"]`). Source b2-01.
- **MEMO — merge-generalize reconciliation + prioritized replay (CONCEPT:KG-2.1 / AHE-3.0)** —
  `evolving_memory.EvolvingMemoryStore.reconcile_similar` collapses *near-duplicate* insights (not just exact
  signatures) into a canonical survivor via `merge(..., generalize=True)`, which records absorbed variants under
  `metadata['generalized_from']`. New `harness/replay_buffer.PrioritizedReplayBuffer` (inverse-frequency
  priority, seed-faithful sampling, FIFO-tiebreak eviction). Both wired into
  `AgenticEvolutionEngine.run_evolution_cycle` (per-cycle generalize + replay-state push) with a new
  `sample_replay` accessor; cycle report gains `insights_generalized` / `replay_buffer_size`. Source b4-03.

### Fixed
- **`RELEVANCE_SCORED` edge type was written but unregistered (CONCEPT:KG-2.7, VU-1)** — the relevance sweep
  (`knowledge_graph/core/engine_tasks.py`) wrote a `"RELEVANCE_SCORED"` string literal with no matching
  `RegistryEdgeType` member (a type-safety gap). Registered the member and switched the write to
  `RegistryEdgeType.RELEVANCE_SCORED` (value unchanged → no edge/query break).
- **Revived the Global Workspace Attention loop (CONCEPT:ORCH-1.2 / KG-2.1)** — the GWT loop was entirely
  dead: `WorkspaceAttention.__init__` took no `engine` (so `executor.py`'s `WorkspaceAttention(engine)`
  assigned the engine to an int slot), `get_attention_score` was **never implemented**, and `executor.py`
  imported `knowledge_graph.workspace_attention` which **does not exist** — all three failures silently
  swallowed by bare `except`. Now: `__init__` accepts `engine`; `collect_proposals`/`broadcast_to_kg`
  default to it; new `get_attention_score` reads back the most-recent *selected* `ProposalNode` composite
  score; new `select_and_broadcast` one-call loop (collect→score→select→broadcast). `executor.py` imports
  the real `.workspace_attention`. `ParallelEngine.execute` now drives `_broadcast_workspace_attention`
  after each multi-agent wave (≥2 successful outputs, shared engine; non-fatal), closing the write→read
  loop so specialist standings actually feed routing/confidence. Tests: `tests/test_workspace_attention.py`
  (+9), `tests/integration/core/test_parallel_engine_advanced.py` (+2).
- **EvolvingMemoryStore now records workspace winners (CONCEPT:KG-2.1)** — with the GWT loop revived,
  `ParallelEngine._record_winners_to_memory` routes broadcast winners into the `EvolvingMemoryStore` INSIGHT
  bank (deduped per specialist → repeat wins reinforce), completing the previously-deferred adoption now
  that there is a genuine live driver. Best-effort; persisted to the shared engine.

### Added
- **GWT loop telemetry / engine-mismatch guard (CONCEPT:ORCH-1.2)** — `workspace_attention.py` tracks
  process-wide write/read counters (`workspace_attention_telemetry()`); `get_attention_score` flags
  `suspected_engine_mismatch` and warns once (or raises under `AGENT_UTILITIES_GWT_STRICT`) when proposals
  are broadcast but reads never resolve one — the signature of the writer/reader holding different engine
  instances. Surfaced in `ExecutionResult.telemetry["workspace_attention"]`.

### Added
- **LLM plan-synthesizer for executable RAG (CONCEPT:KG-2.12)** — `retrieval/executable_rag.py`
  `parse_executable_plan` (parse-or-fallback: malformed/partial LLM output degrades to `build_linear_plan`,
  always yields a runnable plan) + `HybridRetriever._synthesize_executable_plan` (ORCH-1.27 `planner` role).
  `retrieve_executable(..., use_planner=True)` now optionally synthesizes a richer/non-linear plan with the
  planner role instead of the deterministic linear plan; a planner failure degrades transparently. Source b2-03.
- **Self-bootstrapping ontology in the OWL ingest phase (CONCEPT:KG-2.2)** — `pipeline/phases/owl_reasoning.py`
  `bootstrap_ontology_path` derives the ontology from the graph's own sampled records (plateau-stopped
  `OntologyBootstrapper`, emits Turtle to a temp file) and `execute_owl_reasoning` reasons over it when the new
  `PipelineConfig.enable_ontology_bootstrap` flag (env `ENABLE_KG_ONTOLOGY_BOOTSTRAP`, default off) is set and
  no explicit `owl_ontology_path` is given; falls back to the bundled `ontology.ttl` if nothing is derived.
  Adds `ontology_bootstrap_plateau_patience` / `ontology_bootstrap_sample_limit` config. Source b7-05.
- **Hierarchical (global→local) GraphRAG retrieval (CONCEPT:KG-2.5)** — `core/hierarchical_retrieval.py`
  `HierarchicalCommunityRetriever`: ranks communities by query relevance, drills into the top-k, ranks
  entities with a parent-community context boost; wired live as `TopologicalAnalysisEngine.hierarchical_retrieve`. Source b2-04.
- **Quality-budget / fidelity-gated compaction (CONCEPT:KG-2.1)** — `ContextCompactor` gains `fidelity(P)`,
  `record_processed`, `divergence_report`, and a fidelity gate in `should_compact` (compacts before the
  cumulative-context cliff; dormant until tokens are recorded → capacity-only callers unaffected). Source b7-01.
- **LCM convergence guarantee + summary-DAG recovery (CONCEPT:KG-2.20)** — `_guarantee_shorter` makes
  `compress_to_memento` always shrink the block (deterministic truncation terminal guarantee); `recover_chain`
  walks the multi-level SUMMARIZES DAG; `link_parent_memento` builds summaries-of-summaries. Source b1-05.
- **Coordination named-aggregation registry (CONCEPT:ORCH-1.3)** — `AggregationOperator` + `aggregate_scores`
  (mean/median/max/min/log-pool) + `CoordinationLayer.aggregate`/`rank` (rank delegates to the unified
  `selection_operators` → synergy #2); consumed live by `WorkspaceAttention.consensus_score`/`select_winners`. Source b1-02.
- **Self-bootstrapping ontology agent (CONCEPT:KG-2.2)** — `core/ontology_bootstrap.py`
  `OntologyBootstrapper`: derives classes/typed-properties from a sample corpus with plateau-based
  stopping, emits RDF/Turtle, and populates grounded (explicit-value, unit-normalised) triples; schema-free
  KG construction behind a flag. Source b7-05.
- **RLM long-context selectors (CONCEPT:ORCH-1.12)** — `RLMConfig.max_turns` (configurable; wired into
  `RLMEnvironment.run_full_rlm`, was hardcoded) + `RLMConfig.select_long_context_strategy()` explicit
  `rlm_lossless` / `memento_compaction` / `none` decision + `compaction_threshold`. Source b2-05.
- **Conductor per-step model routing (CONCEPT:ORCH-1.27)** — `Task.model_id` field; `pick_specialist_model`
  honors a Conductor-assigned `step_model_id` (highest precedence over the per-turn override and tier
  routing); executor passes `ctx.inputs.model_id`; planner instructed to emit per-step `model_id`. Source b5-07.
- **Skill evolution routed through EvolvingMemoryStore (CONCEPT:KG-2.1)** — `AgenticEvolutionEngine`
  skill create/merge now mirror into the unified SKILL bank (`_record_skill`), converging skill memory
  onto the single graph-native store (synergy #2 adoption).
- **DW-GRPO dynamic reward weighting (CONCEPT:ORCH-1.30)** — `rlm/dynamic_reward.py`
  `DynamicRewardWeighter`: tracks each objective's improvement slope across generations and shifts
  weight toward *lagging* objectives (anti-seesaw), so multi-reward optimization stops collapsing onto
  the easiest reward. Wired into `ParetoCandidatePool` (`dynamic_weighting=` flag, `observe()`,
  `weighted_best()`, `reward_weights`) and **default-on in `GEPAOptimizer`** (observes each generation,
  weighted final selection in the no-held-out branch; falls back to prior behaviour until the slope
  signal is meaningful). Deterministic core of plan b2-04; STRATEGY synergy #8.
- **Deterministic reward/dataset training spine (CONCEPT:AHE-3.1)** — `graph/training_signals.py`:
  `batch_normalized_advantage` (GRPO group-normalized advantage), `failure_point` (first-divergence
  step index for error-attributed preference pairs), `composite_reward` (weighted + conditionally-gated
  reward), `difficulty_floor_filter` (b3-02 data-quality floor). Wired into `RewardDecomposer`
  (`batch_advantages()`/`failure_points()`), surfaced through the live `get_distillation_insights`
  (`advantage_spread`/`localized_failures`/`mean_failure_point`). The "build-once" signal layer every
  Wave-C training-gated paper consumes — STRATEGY synergy #10 (sources b6-04/b7-03/b6-01/b3-02).
- **Executable multi-hop RAG spine (CONCEPT:KG-2.12)** — `retrieval/executable_rag.py`:
  `ExecutableRagProgram` runs a typed `retrieve`/`answer` plan (with `{{var}}` data-flow) via a
  deterministic interpreter giving two training-free grounded loops — execution-driven adaptive
  retrieval (boost `top_k`, then fall back vector→grep) and compiler-grounded self-repair (an
  insufficient answer re-runs the implicated retrieve) — plus an inspectable `StepTrace`. Wired live as
  `HybridRetriever.retrieve_executable` dispatching modes to `retrieve_hybrid` (vector) and
  `direct_search` (grep). Replaces ungrounded NL self-reflection; STRATEGY synergy #3, source b2-03 (PyRAG).
- **Graph-native CRUD evolving-memory store (CONCEPT:KG-2.1)** — `harness/evolving_memory.py`:
  `EvolvingMemoryStore` with typed banks (`MemoryBank`: error/skill/tool/guide/insight), full CRUD
  (add+dedup-by-signature, edit, merge=soft-retire+`MERGED_INTO`, remove=soft-retire), `query`,
  lexical-or-embedder `resolve`, and `reconcile` (signature de-dup). In-memory authoritative with
  best-effort durable mirror to the GraphBackend. Wired live into `AgenticEvolutionEngine.run_evolution_cycle`
  (writes an INSIGHT per cycle → `report["insight_id"]`). The "build-once" unification of plans b4-03
  (insight bank), b8-06 (skill banks), and b5-02 (typed workspace banks) — STRATEGY synergy #2.
- **External-verification + provenance on memento compaction (CONCEPT:KG-2.20)** —
  `memento_compressor.verify_memento` runs a deterministic, *independent* faithfulness check
  (AHE-3.1 `FaithfulnessScorer`, distinct from the LLM judge-refine loop) of each memento against its
  source block; the verdict is stamped as `provenance_verified`/`provenance_faithfulness`/
  `provenance_verifier` on the persisted node. Wired into the live `compress_to_memento` path; a failed
  gate never blocks persistence (the raw block is retained losslessly via `SUMMARIZES` for re-expansion).
  STRATEGY synergy #4 (provenanced/verified/recoverable compaction); source b4-04 + b7-01.
- **Harness self-attribution reliability metric (CONCEPT:AHE-3.0)** — `ManifestVerifier.verify` now
  reports `random_baseline_precision` (fix base-rate among evaluated tasks), `attribution_lift`
  (fix_precision ÷ baseline), and `attribution_reliable` (lift ≥ `reliability_multiple`, default 3×) —
  so a harness whose fix predictions merely match chance is flagged unreliable. Source b7-04 F7 (first
  Wave B delta).

### Fixed
- **`MemoryEngine.consolidate()` / `.compact_traces()` were dead (broken imports)** — both imported
  non-existent modules (`.consolidation`, `.memory_compaction`). Rewired to the real implementations:
  `consolidate()` → `SynthesisEngine` + the standard rules (Episode→Preference, Decision→Principle,
  Trace→Skill, KG-2.1); `compact_traces()` → `MemoryHygiene` decay+semantic-merge (KG-2.17). Regression
  test added.

### Added
- **Population-drift monitor (CONCEPT:AHE-3.2)** — `graph/population_drift.py`: 1-D Wasserstein-1
  (`wasserstein1`), `population_spread`, and `PopulationDriftMonitor` (distributional diversity-collapse
  detection across generations). Wired into `VariantPool.population_health()` and surfaced in the live
  `AgenticEvolutionEngine.run_evolution_cycle` report (`population_health` + `early_stop_recommended`).
  STRATEGY synergy #6 — the shared collapse detector for verifier-free evolution (b6-67) / tournaments
  (b4-03); source b2-01 MASS.
- **Unified selection / aggregation operators (CONCEPT:ORCH-1.30)** — `harness/selection_operators.py`:
  `bradley_terry_scores` (verifier-free pairwise→global ranking via MM iteration, previously absent —
  completes plan b6-67), `conservative_rating` (uncertainty-aware LCB μ−κσ, b4-03 TrueSkill-LCB spirit),
  `contribution_weighted_vote` (b5-02), and scalar `select_top_k` (score/LCB). Wired into
  `VariantPool.tournament_select(strategy="score"|"lcb")` (default `"tournament"` unchanged). First
  build-once foundation from `STRATEGY.md` synergy #1 (one selection registry for VariantPool /
  EvolutionaryAggregator / CoordinationLayer / TeamConfig promotion).
- **Direct Corpus Interaction (CONCEPT:KG-2.12)** — `retrieval/direct_corpus.py`:
  `DirectCorpusSearcher` (literal/regex `grep`, line-range `read`, ranked `search` with term
  coverage + line-level localization) — a precise, auditable retrieval mode complementing dense
  vectors. Wired live as `HybridRetriever.direct_search`. Source: research-evolution plan b2-02.
- **Agentic red-team harness (CONCEPT:AHE-3.1)** — `harness/red_team.py`: static attack catalog
  (prompt injection, jailbreak, role override, data exfiltration, sandbagging; OWASP-LLM mapped) +
  `RedTeamRunner` that scores a target's responses (reusing the reliability safety/deception scorers)
  into a severity-ranked `RedTeamReport`. Wired as `red_team_catalog` / `red_team_assess` MCP tools.
  Source: plans b8-04 / b7-07.
- **Provenance-completeness critic gate (CONCEPT:AHE-3.13)** — `harness/provenance_gate.py`:
  deterministic pre-emit gate that checks numeric claims trace to tool values and substantive
  sentences carry valid citations, returning accept / revise / escalate under a revise budget
  (complements the LLM-based `adversarial_verifier`). Wired as the `provenance_check` MCP tool.
  Source: plan b4-04 (MAKA).
- **Reasoning-aware reranking (CONCEPT:KG-2.6)** — `retrieval/reasoning_reranker.py`:
  `ReasoningAwareReranker` reorders an over-fetched candidate pool by a prior-blended,
  calibrated (five-level), instruction-aware relevance via a pluggable `RerankScorer`
  (deterministic `LexicalRelevanceScorer` default; cross-encoder seam for later). Wired
  **default-on** into `HybridRetriever.retrieve_hybrid` (new `_rerank_candidates`: over-fetch →
  rerank → cap), toggle via `enable_rerank`. Source: research-evolution plan b4-02; retrieval-quality
  gate unaffected.
- **Reliability evaluation scorers (CONCEPT:AHE-3.1)** — `harness/reliability_scorers.py`: nine
  pluggable `EvalScorer`s distilled from recent research (faithfulness/grounding, safety⊥accuracy
  decoupling, topic-coverage T-P/R/F1, tool-necessity knowing-doing gap, deception/sandbagging/
  sycophancy probes, citation coverage/precision/recall, abstention-aware Brier skill score,
  retrieval recall@k/nDCG@k, and content-injection/prompt-trap guardrail). Bundled via
  `build_reliability_suite()` and exposed on the live harness MCP server as the `eval_reliability`
  tool. Sources: `.specify/specs/research-evolution-20260606/` (b1-06, b1-01, b4-06, b5-05, b2-06,
  b5-06, b8-03, b3-01, b7-07).
- **Reliability seed corpus + gate (CONCEPT:AHE-3.1)** — `harness/reliability_corpus.py` seeds the
  suite with deterministic grounded/adversarial cases run through the real `EvalCorpus`, scored by
  `run_reliability_corpus()`. New `scripts/check_reliability_corpus.py` CI gate (registered in
  `guardrails.yml`, with a `tests/gates` meta-test) fails if the match-rate drops below floor.
  `EvalCorpus.add_case`/`load_cases` gained backward-compatible `metadata` passthrough so a corpus
  case can carry per-scorer context (evidence, gold topics, retrieved ids, …).
- **Sentiment Fusion Signals (CONCEPT:KG-2.29)** — credibility-weighted sentiment fusion
  (lexicon polarity + source-credibility prior + recency decay) emitted as a SENTIMENT_ANALYST
  `AgentSignal` consumed by `SwarmConsensus` and registered into `BayesianSignalFusion`;
  `:SentimentFact` KG provenance nodes.
- **Geopolitical Risk Scoring (CONCEPT:KG-2.30)** — risk factors × sector/region exposure →
  portfolio geopolitical-risk score; wired into `StressTestEngine` (shock vector) and
  `RegimeDetector` (regime flag); `:GeopoliticalRisk`/`affectsSector`/`exposedTo` OWL facts so
  "which holdings are exposed to risk X" is a graph query.
- **Dividend Sustainability & Credit/Fixed-Income Quality (CONCEPT:KG-2.31)** — payout/coverage/
  growth + yield-trap detection, and Merton distance-to-default (PD = Φ(−DD)) + interest-coverage/
  leverage credit grading; folded into the debate via `DebateContext.fundamentals_report` so the
  Bear/Risk personas argue against real solvency numbers.
- **Multi-Market Composite Backtester (CONCEPT:KG-2.32)** — path-dependent, look-ahead-safe
  (signals shifted 1 bar) multi-market backtest with a shared capital pool + per-market
  attribution + engine-or-local Sharpe/DSR/drawdown.
- **SABR Volatility Surface (epistemic-graph CONCEPT:KG-2.20j)** — Hagan (2002) implied-vol,
  smile, and (α,ρ,ν) calibration kernels; exposed via emerald `emerald_derivatives` (+vol-arb)
  and data-science `quant_derivatives`.
- **Execution Bridge (emerald CONCEPT:EE-032)** — routes decisions to orders through the
  ExchangeBackend Protocol; live orders blocked behind `require_human_approval_live` + RiskGuard,
  paper executes freely. **Cockpit CLI (EE-033)** — text-mode live engine/account/risk/positions
  snapshot (`emerald-cockpit`).
- **Trade-Journal Bias Auditor & Shadow Account (CONCEPT:KG-2.26)** — From Vibe-Trading's
  shadow account. `TradeJournalAuditor.audit()` builds a `TraderProfile` (win rate, avg
  holding period, PnL ratio, max drawdown) + 4 behavioural-bias diagnostics (disposition,
  overtrading, momentum-chasing, anchoring) and persists them as queryable `:TraderProfile`/
  `:BehavioralBias` KG nodes a future debate/risk-officer can cite.
- **Agent Calibration / Reputation Tracking (CONCEPT:KG-2.27)** — `CalibrationTracker` records
  each persona's past directional calls + outcomes and computes a Brier-based calibration
  score (via engine `brier_score`), then `apply_calibration_to_swarm()` rewrites
  `SwarmConsensus` role weights so miscalibrated voices are down-weighted — a reputation
  feedback loop persisted to the KG (`:AgentCalibration`).
- **Persona Decision-Heuristic Enrichment (CONCEPT:KG-2.28)** — From Fincept's persona registry.
  Attaches typed, evaluable rules (Graham P/E<15·P/B<1.5·MoS≥30%, Buffett ROIC/moat, Burry
  forensic short triggers, Lynch PEG, …) to each investor persona; `evaluate_persona()` returns
  pass/fail + rationale and the `DebateEngine` folds the verdict into each side's prompt when
  `DebateContext.metrics` is present. Seeded as `:DecisionHeuristic` OWL facts.
- **HITL Escalation Matrix (CONCEPT:OS-5.12)** — `EscalationMatrix` (risk × value tier →
  required approver/timeout/fallback) consulted by the Ontology Action `ActionExecutor` after
  authorization: high-tier actions require human approval (driven through `ApprovalManager`),
  audited + KG-persisted; conservative default matrix.
- **Release-Channel System (CONCEPT:OS-5.13)** — `ReleaseChannel` (stable/beta/edge) +
  `@release_channel` guard + `ChannelRegistry`, wired into capability designation so
  edge-tagged agents/skills/tools are genuinely unroutable on `stable` (default).
- **Langfuse Exporter (CONCEPT:ECO-4.24)** — lazy/optional Langfuse sink wired into
  `AgentOrchestrationEngine.run_graph`; exports spans/token-usage/traces when
  `LANGFUSE_*` keys + dep are present, clean no-op otherwise.
- **Golden-Loop Auto-Merge (CONCEPT:AHE-3.14)** — governed promotion path: golden-loop
  proposals passing a conservative quality threshold + governance validity auto-promote to
  active skills/prompts; below-threshold stay propose-only (the safe default; opt-in via
  `KG_GOLDEN_AUTO_MERGE`), every decision audited.
- **Quant-framework comparative analysis** — `docs/comparative_analysis_quant_frameworks.md`
  (Vibe-Trading / AutoHedge / FinceptTerminal vs agent-utilities) seeding KG-2.26/27/28.
- **Investor-persona debate voices (CONCEPT:KG-2.6)** — `DebateEngine.with_personas()` loads
  persona prompt bodies (default Buffett vs Burry) so each side argues in-voice; archetype
  stamped on each `DebateArgument.role`.
- **Ontology Action System (CONCEPT:KG-2.25)** — First-class, governed *verbs* on the
  ontology, closing the one genuine gap vs Palantir AIP's "data+logic+**actions**+security"
  ontology. Adds `OntologyAction`/`ActionParameter`/`ActionInvocation` models, an
  `ActionRegistry`, and a permission-gated + audited + KG-persisted `ActionExecutor`
  (`agent_utilities/knowledge_graph/actions/`). Authorization reuses the existing
  `PermissionsKernel`; every invocation is audited via `AuditLogger` and persisted as a
  queryable `action_invocation` KG node. New OWL module `ontology_action.ttl` adds
  `:OntologyAction` + `:actsOn`/`:hasParameter`/`:producesEffect` and a property chain
  `:mayBeInvokedBy ← (:requiresCapability ∘ :providedBy)` so an agent's eligibility to invoke
  an action is **reasoned by OWL, not hand-wired**. Governed by a new SHACL shape; two real
  built-in actions (`kg.search`, `finance.forensic_screen`) registered on the live path.
- **Declarative Resilience Policy (CONCEPT:ORCH-1.36)** — Closes the L7 reliability gap.
  `ResiliencePolicy` (max attempts, exponential backoff + seedable jitter, `retry_on`
  predicate/allow-list with a non-retryable hard-deny set, per-attempt timeout, fallback
  chain) + `run_with_resilience`/`_sync` (`agent_utilities/orchestration/resilience.py`).
  Wired **default-ON** into the live specialist-execution path
  (`graph/executor.py:_execute_dynamic_mcp_agent`), composing with — not replacing — the
  existing circuit breaker, per-attempt timeout, and sibling-specialist fallback.
- **Investor-persona debate voices (CONCEPT:KG-2.6)** — The Bull/Bear `DebateEngine` now loads
  investor-persona prompt bodies (`DebateEngine.with_personas(bull, bear)`, default Buffett vs
  Burry) so each side argues in a specific investor's voice; the persona archetype is stamped
  on every `DebateArgument.role` for the audit trail. Generic voices preserved as fallback.
- **Palantir AIP comparative analysis** — `docs/comparative_analysis_palantir_aip.md`: a
  capability-by-capability (12) + layer-by-layer (9) mapping showing agent-utilities meets
  ~10/12 AIP capabilities in full and exceeds it on OWL+SHACL reasoning, graph-native compute,
  self-evolution, and reward-weighted routing; documents the closed gaps above and the backlog.
- **KG-Governed Agent Swarm (CONCEPT:ORCH-1.32, extends ORCH-1.8/1.1/1.27)** — Assimilated from Kimi
  K2.6 Agent Swarm (Moonshot AI) + PARL/Mooncake. Adds governance/quality deltas on top of the
  existing `ParallelEngine` (which already did dependency-ordered parallel waves + synthesis):
  **SWARM-1** one-shot `graph_orchestrate(action="swarm")` (goal→decompose→parallel-waves→verify→
  synthesize, governance ON by default); **SWARM-2** planner→execute→**verify** loop (per-leaf
  `success_criteria` judged + bounded re-dispatch); **SWARM-3** critical-path metric (longest
  dependency chain — the PARL insight) + parallelism ratio; **SWARM-4** per-agent `output_schema`
  enforcement (prose → soft failure); **SWARM-5** retry-with-backoff (distinct from the circuit
  breaker); **SWARM-6** heterogeneous-model swarm via `AgentSpec.model_role`; **SWARM-7** per-wave
  telemetry on `ExecutionResult`. Does NOT adopt PARL training or Mooncake serving (documented).
  Analysis: `.specify/reports/kimi-swarm-comparative-analysis.md`.
- **Mementified Context Management (CONCEPT:KG-2.20, extends KG-2.1)** — Assimilated from *Memento:
  Teaching LLMs to Manage Their Own Context* (Kontonis et al., MSR AI Frontiers 2026). Brings the
  paper's block-compress-evict pattern to the orchestration layer (the paper's flagged "agents"
  application). **MEM-0:** strangled the memento compressor into a canonical `memento_compressor.py`,
  fixing a silently-broken `.memento_compressor` import that had left the memento write path and the
  whole `MemoryEngine` compaction facade dead. **MEM-1:** new default-ON `MementoCompaction`
  capability that, on `before_model_request`, segments the running history and evicts old completed
  blocks as dense mementos (live-path test: −77% tokens on a synthetic trajectory). **MEM-2:**
  judge-refine compression loop (rubric, τ=8/10, ≤2 iters — the paper's 28%→92% quality step).
  **MEM-3:** semantic-boundary segmentation (`boundary_score`/`segment_into_blocks`, never cuts
  mid-derivation) + a `memento_blocks` `ContextCompactor` strategy. **MEM-4:** lossless
  recoverability (`Memento -[:SUMMARIZES]-> EvictedBlock`, `recover_evicted_block`) — the
  orchestration-layer substitute for the paper's in-engine implicit KV channel (which we cannot
  reproduce; documented honestly). Comparative analysis: `.specify/reports/memento-comparative-analysis.md`.
- **RLM-GEPA synergy features (CONCEPT:ORCH-1.28–1.31 + ORCH-1.12/1.13/1.27 wiring)** — Assimilated
  from the GEPA paper (Agrawal et al., ICLR 2026, `2507.19457`), `Trampoline-AI/predict-rlm@edaddfe`,
  and the AppWorld RLM-GEPA work (claims verified against source before implementation):
  - **P0 — RLM-GEPA live entry point.** `rlm/runner.py` (`run_rlm`, `optimize_rlm_skill`) +
    `graph_orchestrate(action="rlm_run"|"rlm_optimize")` make `PredictRLM`/`GEPAOptimizer` reachable
    ≤3 hops (they were library-only — a Wire-First violation).
  - **ORCH-1.28 Composable Skills + Generic Adapter** — `rlm/skills.py` (`Skill`, `merge_skills` with
    conflict detection, `RegistryEnvironmentAdapter`); `PredictRLM.mount_skill_unit`. Extends ORCH-1.12.
  - **ORCH-1.29 RLM Resilience + Telemetry** — `rlm/telemetry.py` (`RunTrace`, `FailureClass`,
    `classify_failure`, `with_tool_timeout`, `SandboxFatalError` wired into `repl.py`). Extends ORCH-1.12.
  - **ORCH-1.30 Generalizing GEPA** — held-out feedback/Pareto split (`split_dataset`,
    `optimize(dev_fraction=...)`, `select_best_on_heldout`) + `AgentSpec` anti-overfit grounding in
    `rlm/gepa.py`. Extends ORCH-1.13.
  - **ORCH-1.27 (RLM extension)** — added `rlm-executor`/`rlm-sublm`/`rlm-proposer` roles to the model
    registry; the GEPA proposer now resolves the strong `rlm-proposer` role (`rlm/roles.py`).
  - **ORCH-1.31 Graph-Native Optimization State** — `ParetoCandidatePool.to_snapshot`/`load_snapshot` +
    `GEPAOptimizer.persist_frontier`/`resume_frontier` for resumable, cross-session GEPA on the
    durable epistemic-graph. Extends ORCH-1.13 (+KG-2.7).
- **Memory-OS synergy features (CONCEPT:KG-2.14, KG-2.15, KG-2.17, KG-2.18, KG-2.19 + KG-2.13
  enhancement)** — Assimilated from `ClaudioDrews/memory-os@a4ca094` (all claims verified against
  source before implementation):
  - **KG-2.14 Ground-Truth Context Authority** — `StartupChunk.source_authority` tier + priority
    boost + a Ground-Truth Hierarchy preamble in `build_payload` so injected memory is treated as
    authoritative (stops re-fetching). `knowledge_graph/memory/memory_engine.py`. Extends KG-2.1.
  - **KG-2.15 Resilient Retrieval** — 4-level fallback cascade (`_lexical_fallback`) + social-closer
    triviality gate (`hyde_planner.is_trivial_query`) in `plan_and_retrieve`. Extends KG-2.12.
  - **KG-2.17 Memory Hygiene** — decay scanner (importance half-life, archive via `valid_to`, never
    delete; confidence alert) + semantic-merge dedup (cosine 0.92, length pre-filter) in
    `knowledge_graph/memory/hygiene.py`; `agent-utilities-memory hygiene` CLI. Extends KG-2.1/2.3.
  - **KG-2.18 Evidence-Weighted Memory** — Bayesian `trust_score` feedback loop + recall/usage
    `UsageTelemetry` + generation `LineageRecord` in `retrieval/retrieval_quality.py`. Extends KG-2.6.
  - **KG-2.19 Self-Curating Wiki** — SHA-256 delta-skip continuous ingest of a markdown vault
    (`ingestion/wiki_curator.py`, crash-safe state) reusing `IngestionEngine`; MCP
    `graph_ingest(action="curate_wiki")`. Extends KG-2.7.
  - **KG-2.13 enhancement** — typed, outcome-grounded learning: `MemoryEdit` gains `entry_type`,
    `training_value`, `outcome_gate`, `evidence_ids`; un-grounded decisions are dropped; persisted
    decisions get `GROUNDED_BY` edges.
- **Docs** — fixed two misnamed per-concept files (restored KG-2.7 Speculative-Brancher /
  Semantic-Compactor to their correct `KG-2.7-*` names) and added per-concept deep-dives for
  KG-2.11–2.15/2.17–2.19, ORCH-1.27, and AHE-3.12; refreshed `concept_map.md` + `overview.md`.
- **LongMemEval-S Validation Harness (CONCEPT:AHE-3.12)** — A FastAPI `/benchmark` surface
  (`session` / `query` / `report` / `health`) compatible with Quarq's HTTP benchmark runner
  (`quarqlabs/benchmarks`), proving the memory-first synergy (ORCH-1.27 + KG-2.11/2.12/2.13) meets
  or beats 98.2% on LongMemEval-S. Haystack messages ingest as episodic memory into a **frozen,
  versioned** `EvaluationCorpus` (reproducible — Quarq re-derives FAISS each run); questions run the
  HyDE + two-pass pipeline; answers synthesize via the `generator` role and are scored by the
  `judge` role with a deterministic pure-Python fallback (`normalize_answer`, `judge_binary`,
  `aggregate_report`). New `server/routers/benchmark.py` (mounted in `build_agent_app`) and
  `scripts/check_longmemeval.py` CI floor gate (default 95% on a frozen subset; full 500-q run is
  nightly/on-demand). Extends AHE-3.
- **Background Learning Engine (CONCEPT:KG-2.13)** — Assimilated Quarq Agent's asynchronous
  targeted-edit learner (`agent-oss/agent.py:99-160, 2951-3007, 3303/3646`): a concurrency-bounded
  (`Semaphore(4)`), backoff-retried, sync-barriered loop that emits targeted **ADD/UPDATE/DELETE**
  fact edits rather than raw dumps. New `knowledge_graph/memory/learning_engine.py`
  (`MemoryEdit`, `resolve_relative_dates`, `parse_memory_edits`, `with_backoff`, `BackgroundLearner`,
  `extract_edits` via the ORCH-1.27 `learner` role, `run_learner`) and an `agent-utilities-memory
  learn` CLI subcommand. Edits are written as **bi-temporal graph mutations** (KG-2.11): UPDATE
  re-stamps event/storage time; DELETE is **soft** (`status=REMOVED` + `valid_to`), preserving
  history — unlike Quarq's JSON-line overwrite / hard delete. Backoff is bounded (not Quarq's
  infinite loop) to honor the ≤60s test gate. Extends KG-2.1 (+AHE-3).
- **Memory-First Retrieval (CONCEPT:KG-2.12)** — Assimilated Quarq Agent's retrieval stack —
  HyDE query expansion, dual-threshold hybrid search, self-correcting two-pass retrieval, and a
  quantitative-fidelity evidence ledger (`agent-oss/agent.py:1817-2825, 2435, 3211`) — onto the
  graph-native hybrid retriever. New pure helpers in `knowledge_graph/retrieval/hyde_planner.py`
  (`HydePlan`, `HYDE_THRESHOLDS` 0.38/0.28, `parse_hyde_plan`, `merge_retrievals`,
  `build_evidence_ledger`) and `HybridRetriever.plan_and_retrieve()` (HyDE multi-query via the
  ORCH-1.27 `planner` role → dual threshold → **evidence-gated** second pass on
  `RetrievalQualityReport.gate_passed=False` → optional ledger). Exposed via `search_hybrid(mode,
  self_correct)` and `graph_search(mode="hyde"|"deep", self_correct=True)`. The second pass is
  triggered by a *measured* quality-gate failure, not just model self-report. Extends KG-2.3.
- **Bi-Temporal Memory Layers (CONCEPT:KG-2.11)** — Assimilated Quarq Agent's three memory
  layers (semantic / episodic / procedural; `agent-oss/agent.py:1058-1466,3587`) and Temporal
  Truth Protocol (`agent-oss/agent.py:2370-2477,3114-3161`) as **structural graph metadata**
  instead of prompt-only date discipline. Adds a first-class procedural memory layer
  (`MemoryNode.memory_type` + `target_entity`), a pure bi-temporal core
  (`knowledge_graph/core/bitemporal.py`: `stamp_bitemporal`, `is_valid_as_of`, `filter_as_of`,
  `resolve_precedence`, `supersede`) auto-wired into `engine.link_nodes`, **as-of queries**
  (`query_cypher(as_of=...)` + `graph_query(as_of=...)`) answering "what was true on date T",
  and **event-time contradiction precedence** (`resolve_temporal_contradiction` writes a
  `SUPERSEDES` edge and closes the loser's `valid_to` without deleting history). Extends KG-2.1.
- **Role-Specialized Model Routing (CONCEPT:ORCH-1.27)** — Assimilated Quarq Agent's
  three-specialized-model pattern (planner / generator / learner; `agent-oss/agent.py:58-92`)
  as portable **role→(tier,tags) bindings** over the existing model registry, rather than
  hardcoded model ids. New `ModelRole`, `RoleSpec`, `_DEFAULT_ROLE_ROUTING`, optional
  `ModelRegistry.role_routing`, and `ModelRegistry.pick_for_role()` / `resolve_role()` in
  `models/model_registry.py`; `create_model(role=...)` in `core/model_factory.py`;
  `AgentConfig.role_routing` override; and a `graph_configure(action="set_role_routing")`
  MCP action. Roles degrade gracefully via the existing `pick_for_task` tier-fallback so any
  provider pool works. First feature of the memory-first synergy assimilation (extends ORCH-1.2).
- **GeniusBot Desktop Cockpit Integration (CONCEPT:GBOT-6.0 – 6.6)** — Built complete systems-level integration and visual design systems mapping for `geniusbot`, a premium Qt/PySide6-based Multi-Platform Systems and Finance Cockpit:
  - **GBOT-6.0 (Desktop Cockpit Orchestrator)**: Native Python/PySide6 interface orchestration for autonomous agent interactions.
  - **GBOT-6.1 (Ecosystem Dynamic Tab Matrix)**: Tabular matrix manager for swappable multi-plugin ecosystem cockpits.
  - **GBOT-6.2 (Embedded Terminal Sandbox)**: High-performance terminal emulator directly within the Qt GUI for sandboxed shell interactions.
  - **GBOT-6.3 (Universal Tool Approval Gate)**: Interactive, secure popup dialogs for human-in-the-loop tool approvals.
  - **GBOT-6.4 (Topological Cockpit Memory)**: Real-time Virtual Context Block and memory-in-view topological visualizer.
  - **GBOT-6.5 (Multi-Tenant Daemon & Tray)**: System tray daemon for background watcher execution and background ingestion notifications.
  - **GBOT-6.6 (High-Performance Visual Finance Cockpit)**: Snappy trade visualization dashboard with real-time price feeds, Kelly position sizing, and historical comparison charts.
- **Ecosystem Architecture Mapping** — Fully integrated `geniusbot` into the core systems context and container architecture:
  - Updated `overview.md` systems table and ecosystem architecture catalogs.
  - Added `geniusbot` to the canonical System Context C4 Context diagram in `architecture_c4.md`.
  - Added Qt/PySide6 dependencies and interface links to the Ecosystem Dependency Graph in `architecture_c4.md`.
  - Extended the **Canonical Concept Registry** (`concept_map.md`) to formally track all 7 `GBOT` concepts with 1:1:1 traceability across code and documentation.

### Changed
- **BREAKING: Registry-Based Configuration Migration** — Fully deprecated all legacy per-tier LLM environment variable fields (`LLM_PROVIDER`, `LLM_MODEL_ID`, `LLM_BASE_URL`, `LLM_API_KEY`, `LITE_LLM_*`, `SUPER_LLM_*`, `EMBEDDING_*`) from `AgentConfig`. All LLM and embedding configuration now routes exclusively through the `chat_models` and `embedding_models` registries in `config.json`. Per-model `base_url` and `api_key` overrides are supported natively within each registry entry.
  - New `AgentConfig` properties: `default_chat_model`, `lite_chat_model`, `super_chat_model`, `default_embedding_model` (registry-derived)
  - `DEFAULT_LLM_*` constants in `core/config.py` now source from the registry instead of environment variables
  - Updated consumers: `model_factory.py`, `server_factory.py`, `mcp_utilities.py`, `embedding_utilities.py`, `analyzer.py`, `routing.py`, `extractor.py`

### Added
- **Native xAI OAuth 2.0 (PKCE) Authentication Loop (CONCEPT:OS-5.1)** — RFC 7636-compliant authentication flow with code verifier and code challenge (S256). Includes an automatic temporary callback server on `http://127.0.0.1:56121/callback` and manual CLI fallback. Fully supports remote loopback and headless server environments using IDE or SSH port forwarding.
- **X Post Search & Browse Tools** — New high-performance agents and tools (`x_search`, `browse_x_post`) for searching recent posts and parsing specific tweet status payloads using Grok models.
- **Background Workflow Orchestration Polling** — Extended the `graph-os` `graph_orchestrate` tool with `dispatch_workflow` and `workflow_status` actions, allowing non-blocking background executions of complex multi-agent workflows and real-time state polling.
- **Dynamic Ingestion Workflows & Templates** — Workflow catalog integration under `presets/social.yaml` with dynamic parameter substitution (`{{task}}`) to automate social browsing and hydrations directly into the Knowledge Graph.

### Changed
- **`graph_configure` Token Write-back** — Implemented the `set_secret` action in the `graph_configure` tool of the `graph-os` MCP server to securely commit credentials and xAI OAuth tokens back into the persistent `SecretsClient` store.
- **Test Infrastructure: Daemon Thread Guard** — Added `AGENT_UTILITIES_TESTING` guard in `engine_tasks.py` to skip spawning background daemon threads (compaction, evolution, telemetry, analysis, graph writer) during test runs. Prevents pytest-xdist worker hangs from orphaned threads hitting closed backends.
- **Build Artifact Cleanup Fixture** — Session-scoped `cleanup_build_artifacts` fixture in `conftest.py` that removes stale `build/`, `dist/`, `*.egg-info`, rogue `.db` files, and temporary test DB directories after test sessions.
- **Ruff Unsafe Fixes** — Enabled `--unsafe-fixes` in ruff pre-commit hook to auto-fix UP038 (`isinstance` tuple → union) lint errors.

### Fixed
- Fixed broken xAI credential verification by correcting context parameter typing from `RunContext[AgentDeps] | None` to `RunContext[AgentDeps]` for Pydantic AI compatibility.
- Fixed 10 UP038 ruff violations across `research_pipeline.py`, `config.py`, `distributed_state_manager.py`, `owl_bridge.py`, and `graph_validator.py`.
- Fixed broken imports in `mcp_utilities.py` and `server_factory.py` caused by moving `DEFAULT_LLM_*` constants to `core.config`.
- Resolved mypy type errors across the graph module (0 errors achieved).
- Removed leftover `fix_env_vars.py` migration script.


- **Super-Assimilation Evolution Pipeline (CONCEPT:KG-2.0 + AHE-3.2)** — Constitution-governed, KG-driven feature assimilation framework for ingesting 60+ external repositories and research papers:
  - **Assimilation Governance** section added to `constitution.md`: Wire-First (≤3 hops), Extend Don't Duplicate (≥0.7 similarity), No Dead Code, Constitution Preservation, Unified Downstream.
  - **Constitution Preservation**: External codebases' `constitution.md`/`CONSTITUTION.md` ingested as tagged `PolicyNode` entries for cross-project rule synthesis.
  - **4-Phase Pipeline**: Ecosystem Ingestion → Assimilation Codification → Parallel Comparative Analysis (5 pillar-parallel background_research jobs) → SDD Plan Generation.
  - **34-Concept Cross-Reference Matrix**: Every canonical concept cross-referenced against all ingested codebases for targeted innovation extraction.
  - **Evolution Pipeline docs** added to `docs/overview.md` with Mermaid architecture diagram, integration point table, and KG node type catalog.
  - **README updated** with feature entry and documentation reference link.

- **CONCEPT:KG-2.6: Enterprise Knowledge Architecture** — Full enterprise-grade SPARQL federation, modular ontology, and governance validation:
  - **SDD Ontology Layer** (`ontology_sdd.ttl`) — 10 new OWL classes (Specification, SoftwareFeature, Requirement, UserStory, AcceptanceCriteria, SoftwareComponent, APIContract, TestCase, DesignGuideline, ComplianceConstraint) with 9 object/datatype properties. All mapped to ArchiMate 3.2.
  - **Enterprise Core Ontology** (`ontology_enterprise.ttl`) — Standalone importable module with ArchiMate 3.2 layer hierarchy, ADR decision traces, governance properties, and enterprise integration points (EAR, BPM).
  - **SHACL Governance Validation** (`core/shacl_validator.py`) — Enterprise governance validation via `pyshacl`. Supports layered shapes (global + domain overrides). Default shapes in `shapes/governance.shapes.ttl` enforce ADR, Agent, Policy, Specification, and Requirement constraints. MCP: `kg_inspect(view="shacl_validate")`.
  - **SPARQL HTTP Endpoint** (`core/sparql_http.py`) — W3C SPARQL Protocol HTTP endpoint backed by rdflib. Supports GET/POST, content negotiation, W3C Results JSON format. Mountable as Starlette ASGI app.
  - **Ontology Publisher** (`core/ontology_publisher.py`) — Export and distribute ontologies to Stardog (pystardog) and Apache Jena Fuseki (REST API). Supports versioned publishing. MCP: `kg_inspect(view="export_ontology")`.
  - **Ontology Loader** (`core/ontology_loader.py`) — Resolves `owl:imports` declarations at runtime. Supports file://, HTTP, and cached remote fetching. Enables "inherit from central, extend locally" enterprise pattern.
  - **Modular Ontology Architecture** — Main `ontology.ttl` now declares `owl:imports` for enterprise and SDD modules. 13 new SDD ArchiMate mappings added to `archimate_layer.py`.
  - **`pyshacl>=0.29.0`** added to `[owl]` optional dependency group.
  - **55 tests** covering SPARQL, ArchiMate, ADR, SDD mappings, SHACL, publisher, loader, and enterprise modules. Test suite runs in 0.82s.
  - **C4 Diagram**: KG component diagram updated with SDD Ontology, SHACL Validator, Ontology Publisher, and Ontology Loader. New Enterprise Federation data flow added.
- **LLM Configuration Standardization** — Fully migrated all legacy `DEFAULT_MODEL_ID` and `DEFAULT_PROVIDER` references to `DEFAULT_LLM_MODEL_ID` and `DEFAULT_LLM_PROVIDER` across 5 files (`config.py`, `factory.py`, `mcp_utilities.py`, `server/app.py`, `server/routers/agent_ui.py`, `mcp/server_factory.py`). No backward-compatibility aliases — single standard enforced.
- **CONCEPT:KG-2.5: Evolution Cycle Daemon** — Background daemon (`KG-Evolution-Daemon`) in `engine_tasks.py` that triggers every 60 minutes (configurable via `KG_EVOLUTION_INTERVAL` env var). Scans the KG for unresolved `ResearchTopic` / `Concept` nodes, runs relevance sweeps against the primary codebase, and logs each cycle as an `EvolutionCycle` node.
- **`agent-utilities-evolution` Skill** — Meta-orchestration skill in `universal-skills` that chains: KG topic detection → `research-scanner` paper discovery → `kg_ingest` paper ingestion → `comparative-analysis` gap extraction → SDD implementation plan generation. Constitution artifact mandate auto-injected into every SDD plan.
- **Post-Modification Artifact Mandate** — Constitution section mandating 7 artifact types (docs, AGENTS.md, CHANGELOG.md, README.md, .specify/, C4 diagrams, pytests) after every code change. Enforced via `Policy` nodes in the KG with `enforcement: MANDATORY`.
- **LCM Documentation** — New guide `docs/guides/lcm_memory.md` documenting the Summary DAG architecture, MCP tool usage, partition-aware memory, compaction daemon, and evolution daemon.
- 10 new tests in `test_lcm_integration.py` covering ContextCompactor persistence (persist_compaction, escalation, DAG retrieval) and ElasticContextManager LCM operations (compact_thread, expand, grep, describe).

### Added
- **CONCEPT:KG-2.6: Background Concept Research Daemon** — Native, persistent background intelligence integration within `agent-utilities`.
  - `kg_analyze(action="background_research")` submits a persistent `deep_analysis` task to the `SQLiteTaskQueue`.
  - Configurable via `KG_INFERENCE_MODEL`, `KG_LLM_CONCURRENCY` (respects local compute slots), and `KG_ANALYSIS_MAX_DEPTH` (default 2).
  - Background workers natively load `IntelligenceGraphEngine.execute_deep_analysis()`, recursively inferring analogical relationships (`ANALOGOUS_TO`) without requiring an active FastMCP context.
- **BrowseComp-Plus Innovation Transfer (arXiv:2508.06600)** — 5 research-driven enhancements for deep-research benchmarking and evaluation:
  - **CONCEPT:KG-2.3: Evaluation Corpus** — Fixed corpus evaluation mode with `EvaluationCorpus` model, `CorpusManager` CRUD, freeze semantics, and `HybridRetriever` constrained search via `corpus_id` parameter. MCP tools: `kg_create_corpus`, `kg_list_corpora`, `kg_freeze_corpus`. New module: `agent_utilities/knowledge_graph/retrieval/evaluation_corpus.py`.
  - **CONCEPT:AHE-3.1: Adaptive Reasoning Budget** — Continuous 0.0–1.0 float scale for test-time compute scaling with `get_budget()` smooth interpolation and `estimate_query_complexity()` lightweight heuristic. Integrated into `EvaluationEngine.evaluate_and_decompose()`. New module: `agent_utilities/harness/reasoning_effort.py`.
  - **CONCEPT:AHE-3.1: Disentangled Evaluation** — Separates retriever quality (precision, recall, nDCG, MRR) from LLM reasoning quality (step accuracy, goal achievement) and citation quality (precision, recall, F1). `EvaluationEngine.evaluate_disentangled()` returns three independent metric groups. New method: `RetrievalQualityGate.compute_ndcg()`.
  - **CONCEPT:KG-2.3: Hard Negative Mining** — Mines challenging distractors via query decomposition for retriever calibration. `HardNegativeMiner` with penalty application. Gated behind `KG_ENABLE_HARD_NEGATIVE_MINING` env var. New module: `agent_utilities/knowledge_graph/retrieval/hard_negative_miner.py`.
  - **CONCEPT:AHE-3.1: Citation Quality Tracking** — Extracts and evaluates citations (KG refs, concept IDs, external URLs, file paths, arXiv IDs) in agent responses. `CitationTracker` with precision/recall/F1 computation. Lazy-loaded in `EvaluationEngine`. New module: `agent_utilities/harness/citation_tracker.py`.
- 33 new tests in `test_browsecomp_innovations.py` covering all 5 components.
- **CONCEPT:ECO-4.0: Terminal Agent Launcher** — `kg_launch_terminal_agent` MCP tool to spawn CLI-based agents (`agent-terminal-ui`, `claude`, `opencode`, `devin`) in managed tmux sessions. Supports configurable `--prompt` and `--override` (yolo) mode flags per agent type. Tmux auto-detects whether to create a new window (inside tmux) or a detached session (outside tmux).
- **CONCEPT:ECO-4.1: KG Agent Execution** — `kg_execute_agent` MCP tool to trigger the Pydantic AI agent graph execution natively from the KG MCP server, dynamically initializing the orchestration graph and routing queries to specialist agents.
- **CONCEPT:KG-2.0: Document Retrieval** — `kg_get_document` MCP tool to retrieve full documents from the Knowledge Graph by target path, with chunk reassembly and sorting.
- **Configurable Default Terminal Agent** — Added `default_terminal_agent` to `AgentConfig` with `DEFAULT_TERMINAL_AGENT` constant export. Users can override via XDG `config.json` to switch the default CLI agent (e.g., `claude-code`, `opencode`, `devin`).

### Fixed
- **`kg_ontology_validate`** — Aligned schema bindings to use `SCHEMA.nodes[].name` and `node_def.columns.keys()` instead of deprecated `.label` and `.properties` accessors from `schema_definition.py`.

### Changed
- **Ecosystem API Client Standardization** — Standardized all `*_api.py` wrapper files across the agent ecosystem to the unified `api_client.py` convention to simplify downstream imports and skill tooling.
- **Model Registry Loading** — Updated `AgentConfig` to include `model_registry_path` to support dynamic loading of custom JSON/YAML model registries.
- **`kg_ingest` refactored** — Replaced inline document ingestion logic with a cleaner phase-based approach using `kg_get_document` for retrieval.

### Previously Added
- **5-Stage / 17-Phase Intelligence Pipeline** — Evolved the graph ingestion architecture from 15 phases to 5 logical stages: Context Hydration, Structural Extraction, Topological Enrichment, Epistemic Consolidation, and Governance & Evolution.
- **Phase 16: Experience Distillation** & **Phase 17: Decision Evolution** — Added observational hooks to the IntelligencePipeline for self-improving reasoning capabilities.
- **Structural Isomorphism** — Implemented semantic deduplication via AST/content hashing (`AST_hash`, `content_hash`) instead of naive ID-based node creation, using Cypher `MERGE` logic. Replaced `CONTAINS` edges with `IMPLEMENTS` edges for symbols to accurately reflect file-to-symbol relationships.
- **CONCEPT:KG-2.6: Deterministic Mark-and-Sweep Garbage Collection** — Implemented temporal heartbeat mechanism (`last_seen_timestamp`) to atomically purge stale `:Code` and `:Article` nodes from the Knowledge Graph during ingestion runs, maintaining 1:1 codebase parity natively without MD5 overhead.
- **CONCEPT:ORCH-1.4: Dynamic Subgraph Orchestrator** — Dynamically synthesizes subgraph transition logic from the Knowledge Graph on the fly without using predefined templates. Uses KG-2.41 Formal Graph Theory.
- **CONCEPT:AHE-3.4: Long-Running Background Context Spawner** — Polling module that checks background tasks via KG state and autonomously spawns specialized sub-agents based on context shifts and impact scores.
- **CONCEPT:ECO-4.0: Dynamic Tool Assignment Orchestration** — Matches tool ontology to agent tasks dynamically at runtime based on task context and KG embeddings.
- **CONCEPT:KG-2.6: Ontological Team Sharing** — Serializes dynamically created and successful subgraphs into OWL/Turtle ontology formats so they can be exported/imported as shareable team compositions.
- **Multi-Domain Architectural Pattern** — Transitioned `agent-utilities` to a Multi-Domain Expert System supporting modular expansion into `finance`, `medical`, `law`, and `science`. Domain integrations leverage Vectorized Topological Memory and the core Knowledge Graph, with heavy domain-specific dependencies optionally loaded via tags (e.g., `agent-utilities[finance]`).
- **Quantitative Finance Framework** — Production-grade, KG-native financial framework designed for global asset classes (Crypto, Equities, Forex, Derivatives). Features:
  - **Stationary Feature Engineering**: ADF stationarity testing across multi-asset OHLCV market data.
  - **Topological TradingLSTM**: Neural network architectures designed to process sequence data and networkx topological market regimes.
  - **Execution & Evaluation**: Walk-forward validation loops preventing lookahead bias, automated Kelly Criterion position sizing, and Kolmogorov-Smirnov regime shift detection.
- **FIBO & Quant Ontology Alignment** — Extended `ontology.ttl` with `DomainEntity`, `ScientificEntity`, `LegalEntity`, and specialized finance classes (`FinancialInstrument`, `TradingStrategy`, `StationaryFeature`, `LSTMNetwork`, `MarketRegime`, `ExecutionSignal`, `KellySizing`).
- **CONCEPT:ORCH-1.2: Ontological Fallback Chains** — Uses the Knowledge Graph to find fallback models dynamically rather than relying on static lists during rate limits or server errors.
- **CONCEPT:KG-2.6: Vectorized Context-Window Filtering** — Semantically prunes non-relevant subgraph context before swapping models on token overflow, ensuring only contextually distant nodes are dropped. New module: `agent_utilities/graph/context_filter.py`.
- **CONCEPT:AHE-3.4: KG-Native Agentic Task Detection** — Evaluates topological complexity via KG subgraphs to route dense API toolchains to complex models automatically.
- **CONCEPT:AHE-3.4: Topological Reasoning Detection** — Maps user queries to `MathematicalFoundationNode` or quantitative financial concepts to trigger reasoning models natively.
- **CONCEPT:OS-5.0: Topological Session Persistence** — Pins the model for multi-turn conversations directly to the SessionNode to avoid jarring mid-thread model bouncing.
- **CONCEPT:ECO-4.0**: Graph-Native Durable Execution Engine for fault-tolerant multi-leg algorithmic trading.
- **CONCEPT:ECO-4.0**: Secure Jupyter Sandbox with State Machine Invariant checks for code generation.
- **CONCEPT:AHE-3.4**: AgentSpecs Catalog Generator for shareable OWL-driven agent blueprints.
- **CONCEPT:KG-2.6**: Latent Topology RAG (Latte) for hierarchical routing.
- **CONCEPT:KG-2.5**: Single-Shot SIRA for retrieval sparsity optimization.
- **CONCEPT:KG-2.5**: VOI Budget Controller for scaling law enforcement.
- **CONCEPT:KG-2.3**: Cognitive Trap Defense for topological vulnerability mitigation.
- **CONCEPT:KG-2.3**: Experience Alignment for natively managed few-shot adaptation.
- Domain-driven knowledge_graph refactoring with zero-stub parity.
- **CONCEPT:ECO-4.0: Graph-Native Durable Execution** — Fault-tolerant, resumable state execution by persisting graph execution traces natively into the Knowledge Graph (LadybugDB) for high-assurance multi-leg trading. New module: `agent_utilities/orchestration/durable_execution.py`.
- **CONCEPT:ECO-4.0: Secure Jupyter Sandbox** — Isolated code generation sandbox with Vectorized Topology AST validation and State Machine Invariant checks (MCS Ch 6). Blocks unsafe OS commands. New modules: `agent_utilities/tools/jupyter_adapter.py`, `agent_utilities/tools/sandbox_executor.py`.
- **CONCEPT:AHE-3.4: OWL-Driven AgentSpecs** — Compiles dynamic agent topologies into exportable JSON AgentSpec catalogs, strongly typed by OWL ontologies for reproducibility. New module: `agent_utilities/core/agentspec_catalog.py`.
- **CONCEPT:KG-2.6: Research Intelligence Sub-Agent** — Isolated research context with citation graph traversal (Semantic Scholar API), doom-loop detection, and KG persistence. Adapted from ml-intern's research_tool.py sub-agent pattern. New module: `agent_utilities/knowledge_graph/orchestration/research_subagent.py`.
- **CONCEPT:KG-2.5: Spectral Cluster Navigator** — Tuning-free spectral clustering using normalized Laplacian eigengap heuristics for automatic k-selection. OWL-integrated via `skos:Concept` alignment with financial regime detection extension. Adapted from contextplus. New module: `agent_utilities/knowledge_graph/core/spectral_navigator.py`.
- **CONCEPT:KG-2.5: Symbol Blast Radius Analyzer** — Regex-based symbol usage tracking across Python codebases with definition-line exclusion and KG integration. Adapted from contextplus. New module: `agent_utilities/knowledge_graph/core/blast_radius.py`.
- **CONCEPT:KG-2.3: Auto-Similarity Memory Graph** — Auto-creates `SIMILAR_TO` edges between KG memory nodes when cosine similarity ≥ threshold (default 0.72). Exponential decay scoring with stale edge pruning and hub control. Adapted from contextplus. New module: `agent_utilities/knowledge_graph/memory/auto_similarity.py`.
- **CONCEPT:KG-2.3: Hybrid Search Index** — Weighted semantic+keyword search scoring (72%/28% default) with CamelCase/snake_case token splitting and phrase boost. Uses existing embedding infrastructure. Adapted from contextplus. New module: `agent_utilities/knowledge_graph/retrieval/hybrid_search_scorer.py`.
- **CONCEPT:OS-5.0: Enhanced Doom-Loop Detector** — Pattern-aware doom-loop detection with result-aware tool call signatures, repeating sequence detection, and corrective prompt generation. Adapted from ml-intern. New module: `agent_utilities/security/doom_loop_detector.py`.
- **CONCEPT:KG-2.3: RAG-KG Unification** — Collapses separate RAG vector index into KG-native retrieval using three acceleration layers: similarity-edge shortcuts (O(degree) vs O(N)), spectral cluster scoping, and hybrid semantic+keyword scoring. New module: `agent_utilities/knowledge_graph/retrieval/unified_rag_kg.py`.
- **CONCEPT:KG-2.6: Research Orchestration Integration** — Connects ResearchSubagent to ResearchPipelineRunner and UnifiedRAGKGRetriever for automated 7-phase daily research cycles. MCP-compatible for tool registration. New module: `agent_utilities/knowledge_graph/orchestration/research_orchestrator.py`.
- **CONCEPT:KG-2.6: Graph Distillation Migration** — Migrates standard RAG retrieval to pre-computed SimilarityEdgeNode shortcuts for O(degree) retrieval. Manages distillation index lifecycle: batch creation, incremental updates, stale edge pruning, and coverage health monitoring. New module: `agent_utilities/knowledge_graph/retrieval/graph_distillation.py`.
- 10 new `RegistryNodeType` entries, 8 new `RegistryEdgeType` entries, and 11 new Pydantic models for comparative analysis and RAG-KG unification.
- 55 new tests in `test_comparative_analysis.py` and `test_rag_kg_unification.py` covering all 9 new modules.
- Updated Concept Galaxy to 88 concepts (from 79).
- **CONCEPT:KG-2.6: Formal Graph Theory Primitives** — Mathematically rigorous graph operations from MCS (MIT 6.042J): DAG critical path analysis (O(V+E) makespan), k-connectivity certificates (Whitney's theorem), Euler tour serialization, chromatic scheduling (conflict-free parallelism), personalized PageRank (random walk with restart), adjacency matrix power path counting (A^k theorem), and curated MCS seed taxonomy (20 mathematical foundation nodes for KG preloading). New module: `agent_utilities/knowledge_graph/core/graph_theory_primitives.py`.
- **CONCEPT:KG-2.6: Embedding Alignment Diagnostics** — Multi-layer embedding quality analysis from MINER (arXiv:2605.06460v1): Centered Kernel Alignment (CKA) for structural space comparison, Alignment Ratio (AR) diagnostics, adaptive sparse fusion with neuron-level masking and cross-layer weighting, continuous embedding health monitoring via SVD effective dimensionality and CKA drift detection. New module: `agent_utilities/knowledge_graph/retrieval/embedding_diagnostics.py`.
- **CONCEPT:KG-2.6: Structural Causal Reasoning Engine** — Explicit causal chain modeling from MedCausalX (arXiv:2603.23085v1): Structural Causal Models (SCMs) with do-calculus interventions (graph mutilation), d-separation conditional independence testing (nx.is_d_separator), causal verification protocol for reasoning chains, counterfactual generation (proximity-sorted ancestor queries), spuriousness detection via conditional d-separation, and trajectory-level causal alignment scoring. New module: `agent_utilities/knowledge_graph/core/causal_reasoning.py`.
- **CONCEPT:KG-2.6: Latent Space Anti-Collapse Regularizer** — Formal anti-collapse guarantees from LeWorldModel (arXiv:2603.19312v2): SIGReg normality testing via random 1D projections with Jarque-Bera statistics, SVD-based collapse detection (effective dimensionality ratio), diversity-preserving EWC synthesis (extends AHE-3.6 with participation ratio constraints), embedding diversity metrics (isotropy, participation ratio, Shannon entropy), and predictive consistency scoring for agent trajectory validation. New module: `agent_utilities/knowledge_graph/memory/latent_space_regularizer.py`.
- **CONCEPT:KG-2.6: Probabilistic Knowledge Graph Reasoning** — Probabilistic reasoning from MCS Ch 17–21: Bayesian belief propagation with loopy BP over graph topology (odds-form updates with exponential distance decay), random walk exploration with surprise scoring (frequency × distance), Law of Total Probability aggregation (anti-Simpson's Paradox multi-source combination), Birthday Paradox collision detection (n ≈ 1.2√d threshold), and d-separation conditional independence testing. New module: `agent_utilities/knowledge_graph/core/probabilistic_reasoning.py`.
- **CONCEPT:KG-2.6: Optimal Execution Engine** — Mathematical optimal execution from Oxford HFT (Drissi, 2024): Almgren-Chriss discrete (hyperbolic scheduling via κ = arccosh(...)), Almgren-Chriss continuous (HJB-based smooth trajectories), Cartea-Jaimungal (running inventory penalty with Riccati ODE), Avellaneda-Stoikov market making (reservation price + optimal spread), cointegration pairs trading (OU process parameter estimation with AR(1) regression), and signal-adaptive execution (tanh-gated urgency adjustment). New module: `agent_utilities/knowledge_graph/core/optimal_execution.py`.
- **CONCEPT:KG-2.6: Formal Relations Engine** — Mathematical relation properties (Reflexive, Symmetric, Transitive closures) and Equivalence Classes from MCS Ch 4 for zero-shot entity resolution. New module: `agent_utilities/knowledge_graph/core/formal_relations.py`.
- **CONCEPT:KG-2.6: State Machine Invariant Engine** — Deterministic Finite Automata (DFA) abstractions and provable state invariants from MCS Ch 6 to prevent infinite loops. New module: `agent_utilities/knowledge_graph/core/state_machines.py`.
- **CONCEPT:KG-2.6: Markov Transition Forecasting** — Markov Chain transition matrices over agent interaction traces (Vectorized Topologies) from MCS Ch 21 to predict statistical failure nodes via stationary distribution. New module: `agent_utilities/knowledge_graph/core/markov_transitions.py`.
- 7 new `RegistryNodeType` entries (`MATH_FOUNDATION`, `CRITICAL_PATH_RESULT`, `CAUSAL_FACTOR`, `CAUSAL_MODEL`, `EXECUTION_PLAN`, `MARKET_MAKING_QUOTE`, `PAIRS_TRADE_SIGNAL`), 9 new `RegistryEdgeType` entries for KG-2.41–KG-2.46.
- 116 new tests across 6 test suites (`test_graph_theory_primitives.py`, `test_embedding_diagnostics.py`, `test_causal_reasoning.py`, `test_latent_space_regularizer.py`, `test_probabilistic_reasoning.py`, `test_optimal_execution.py`) plus 3 new suites for formal relations, state machines, and Markov transitions.
- New documentation: `docs/mathematical_foundations.md` — comprehensive 300+ line glossary of all mathematical, probabilistic, and financial engineering terms.
- Updated Concept Galaxy to 97 concepts (from 94).
- **CONCEPT:KG-2.6: Alpha Factor Library** — 20 battle-tested alpha factors (momentum, mean-reversion, volatility, volume, value) with IC/IR analysis for factor selection. Sourced from Qlib Alpha158. New module: `agent_utilities/domains/finance/alpha_factors.py`.
- **CONCEPT:KG-2.6: Risk Management Engine** — Risk-first guard pipeline with VaR (Historical/Parametric/Monte Carlo), stress testing (5 predefined scenarios), and pre-trade validation. Sourced from AutoHedge/OpenAlice. New module: `agent_utilities/domains/finance/risk_manager.py`.
- **CONCEPT:KG-2.6: Portfolio Optimization Suite** — Mean-Variance (Markowitz), Risk Parity, and Black-Litterman portfolio optimization with KG-backed allocation tracking. New module: `agent_utilities/domains/finance/portfolio_optimizer.py`.
- **CONCEPT:KG-2.6: Versioned Order System ("Trading-as-Git")** — Immutable order snapshots with version chains, atomic commit promotion, and pre-commit guard hooks. Sourced from OpenAlice. New module: `agent_utilities/domains/finance/versioned_orders.py`.
- **CONCEPT:KG-2.6: Market Data Abstraction Layer** — Protocol-based data provider system with auto-fallback chains, OHLCV normalization, synthetic GBM data generator, and KG data provenance. New module: `agent_utilities/domains/finance/market_data.py`.
- **CONCEPT:KG-2.6: x402 AI Payment Protocol** — First-class HTTP 402 challenge-response handler for autonomous AI agent payments with budget guards, daily/monthly spend limits, and manual approval thresholds. New module: `agent_utilities/domains/finance/payments.py`.
- **CONCEPT:KG-2.6: Profit Attribution Engine** — P&L decomposition into alpha/beta/residual via OLS regression, plus Sharpe, Sortino, Calmar, Information Ratio, win rate, and profit factor analytics. Sourced from Qlib. New module: `agent_utilities/domains/finance/profit_attribution.py`.
- **CONCEPT:KG-2.6: Universal Real-Time Streaming** — Domain-agnostic pub/sub message bus with wildcard topic matching, sequence numbering, message history, and WebSocket transport adapter. New module: `agent_utilities/domains/finance/streaming.py`.
- Extended `ontology.ttl` with 11 new OWL classes (`AlphaFactor`, `RiskLimit`, `VaREstimate`, `PortfolioAllocation`, `OrderVersion`, `OrderCommitRecord`, `MarketDataSource`, `PaymentProofEntity`, `PaymentBudget`, `ProfitAttribution`, `StreamChannel`) and 7 new object properties (`hasAlphaFactor`, `hasRiskLimit`, `hasAllocation`, `hasOrderVersion`, `paidVia`, `attributedBy`, `streamsTo`).
- Added `[finance-kronos]` optional extra for GPU-accelerated foundation model inference.
- 123 new tests across 8 test suites (`test_alpha_factors.py`, `test_risk_manager.py`, `test_portfolio_optimizer.py`, `test_versioned_orders.py`, `test_market_data.py`, `test_payments.py`, `test_profit_attribution.py`, `test_streaming.py`).
- Updated Concept Galaxy to 105 concepts (from 97).
- **CONCEPT:KG-2.6: Kronos Foundation Model Forecaster** — K-line candlestick tokenizer (1200-token vocabulary) and autoregressive transition matrix predictor for time series forecasting. New module: `agent_utilities/domains/finance/kronos_forecaster.py`.
- **CONCEPT:KG-2.6: Multi-Agent Trading Swarm** — 8 specialized agent roles (Director, Quant, Risk, Execution, Indicator, Pattern, Trend, Sentiment) with weighted consensus aggregation and risk manager veto power. Sourced from QuantAgent/AutoHedge. New module: `agent_utilities/domains/finance/trading_swarm.py`.
- **CONCEPT:KG-2.6: Visual Technical Analysis Engine** — Chart pattern detection (double top/bottom, breakout), support/resistance levels via local extrema clustering, and trend analysis via linear regression. Sourced from QuantAgent. New module: `agent_utilities/domains/finance/visual_ta.py`.
- **CONCEPT:KG-2.6: Real-Time Market Feeds** — Finance-specific StreamBus adapter with tick-to-bar aggregation, multi-symbol topic routing, and price alert triggers. Sourced from FinceptTerminal. New module: `agent_utilities/domains/finance/market_feeds.py`.
- **CONCEPT:KG-2.6: Multi-Platform Strategy Export** — Code generation engine producing Pine Script v6 (TradingView), MQL5 (MetaTrader 5), and TDX (通达信) from universal StrategySpec. Sourced from Vibe-Trading. New module: `agent_utilities/domains/finance/strategy_export.py`.
- **CONCEPT:KG-2.6: Research Autopilot** — Automated hypothesis → backtest → report loop with configurable pass criteria (Sharpe, drawdown, win rate, profit factor). Sourced from Vibe-Trading. New module: `agent_utilities/domains/finance/research_autopilot.py`.
- **CONCEPT:KG-2.6: Strategy Sharing System** — Strategy cards with metadata, configuration presets, registry search, fork capability, and community marketplace. Sourced from Vibe-Trading. New module: `agent_utilities/domains/finance/strategy_sharing.py`.
- Extended `ontology.ttl` with 7 new OWL classes (`KronosModel`, `TradingSwarmEntity`, `ChartPattern`, `MarketFeed`, `StrategyExport`, `ResearchHypothesis`, `StrategyCardEntity`) and 6 new object properties (`forecasts`, `swarmDecidedBy`, `detectedPattern`, `exportedAs`, `testedHypothesis`, `sharedAs`).
- 95 new tests across 7 test suites (`test_kronos_forecaster.py`, `test_trading_swarm.py`, `test_visual_ta.py`, `test_market_feeds.py`, `test_strategy_export.py`, `test_research_autopilot.py`, `test_strategy_sharing.py`).
- Updated Concept Galaxy to 112 concepts (from 105).


#### Added
- **CONCEPT:ORCH-1.4: Learned Agent Routing** — Jointly optimizes decomposition depth, worker choice, and inference budget from execution traces. Three routing policies: `RuleBasedPolicy` (keyword pattern matching to primitives), `TraceLearnedPolicy` (softmax scoring from historical `ExecutionTrace` with exponential moving average quality/success tracking), and `CostAwareRouter` (Pareto-optimal cost/accuracy wrapping any policy with budget filtering). Based on Uno-Orchestra research (arXiv:2605.05007v1, relevance score 31.2). New module: `agent_utilities/graph/routing_policy.py`.
- **CONCEPT:KG-2.2: Elastic Context Operators** — 5 atomic operators for elastic context orchestration: `SKIP` (exclude irrelevant messages), `COMPRESS` (replace messages with summary), `ROLLBACK` (revert to checkpoint), `SNIPPET` (extract focused evidence from verbose content), `DELETE` (permanent removal). Compress is expressively complete (any operation expressible as compression) while specialized operators reduce generation cost and hallucination risk. Extends `ContextCompactor` (KG-2.7) with `ElasticContextManager`, `ContextCheckpoint`, and `OperatorResult`. Based on LongSeeker's Context-ReAct paradigm (arXiv:2605.05191v1, relevance score 25.5). Extended module: `agent_utilities/knowledge_graph/context_compactor.py`.
- **CONCEPT:KG-2.2: Multi-Timescale Memory Dynamics** — Three-tier `TimescaleMemoryStore` with timescale-aware exponential decay: Working (5min half-life, promotes at 3+ accesses), Episodic (4hr half-life, promotes at 5+ accesses), Semantic (30-day half-life, permanent). Consolidation engine promotes high-activation memories up tiers. Content-hash deduplication, keyword-scored retrieval with activation weighting, and configurable decay floor pruning. Based on Continual Knowledge Updating (arXiv:2605.05097v1, relevance score 11.2). New module: `agent_utilities/knowledge_graph/timescale_memory.py`.
- **CONCEPT:KG-2.3: Versioned KG Mutations** — Git-like transactional mutation semantics for Knowledge Graph evolution. `KGTransaction` (batches add_node/update_node/delete_node/add_edge/delete_edge mutations), `KGCommit` (atomic application with rollback data and parent-commit chaining), `KGVersionEngine` (commit/rollback/diff with full history), `KGDiff` (structural diff between graph versions: nodes_added/removed/modified, edges_added/removed). Based on Evolving Idea Graphs (arXiv:2605.04922v1, relevance score 11.2). New module: `agent_utilities/knowledge_graph/kg_versioning.py`.
- **CONCEPT:ECO-4.0: Dynamic Skill Evolution** — On-the-fly skill creation and synthesis to avoid catastrophic forgetting. `SkillNeologismDetector` (identifies when existing skills don't cover a new capability via Jaccard keyword similarity below configurable threshold), `SkillFactory` (creates new `SkillNode` instances from detected gaps or execution traces with provenance tracking), `SkillMerger` (detects overlapping skills and consolidates them, combining keywords, patterns, and confidence scores). Based on Skill Neologisms (arXiv:2605.04970v1, relevance score 11.9). New module: `agent_utilities/knowledge_graph/skill_evolver.py`.
- **CONCEPT:OS-5.1: Jailbreak Robustness Hardening** — Extends Prompt Injection Scanner (OS-5.4) with 4-category jailbreak attack taxonomy from SoK research. Template-based (DAN/Developer Mode, AIM persona, UCAR unrestricted, Grandma exploit), optimization-based (GCG adversarial suffix detection, token smuggling via encoding), LLM-based (context boundary confusion with `[/INST]`/`[/SYS]` markers, multi-turn escalation), manual (role-play/hypothetical framing, false authority claims). 12 new `ThreatPattern` entries, `JailbreakCategory` enum. Based on SoK: Robustness against Jailbreak Attacks (arXiv:2605.05058v1, relevance score 16.2). Extended module: `agent_utilities/security/prompt_scanner.py`.
- 32 new unit tests in `test_research_enhancements.py` covering all 6 enhancements (routing policies, elastic operators, skill evolution, timescale memory, KG versioning, jailbreak patterns).
- Updated Concept Galaxy to 79 concepts (from 73).

### Added
- **CONCEPT:AHE-3.3: Agent-Interpretable Model Evolver** — Autoresearch loop that evolves scikit-learn-compatible model classes optimized for dual objectives: predictive accuracy and LLM readability via `__str__()`. Features Pareto frontier tracking with O(n²) dominance checking, reward decomposition integration (AHE-3.10), display strategy auto-selection, and KG-native evolutionary lineage via `EVOLVED_MODEL` transitive edges. Actual model fitting delegated to `data-science-mcp` via MCP tool calls. Based on Microsoft Research's Agentic-iModels (arXiv:2605.03808). New module: `agent_utilities/harness/imodel_evolver.py`.
- **CONCEPT:AHE-3.3: LLM-Graded Interpretability Tests** — 6-category, 200-test protocol measuring whether an LLM can simulate model behavior from `__str__()` alone. Categories: feature attribution (32), point simulation (43), sensitivity analysis (32), counterfactual (32), confidence calibration (32), data attribution (29). Includes numerical tolerance grading, reward hacking detection, and EvalRunner (AHE-3.12) integration. Results persist as `InterpretabilityTestNode` in the KG. Based on arXiv:2605.03808. New module: `agent_utilities/harness/interpretability_tests.py`.
- **CONCEPT:KG-2.6: Topological Graph Visualization** — Scalable WebGL-based Knowledge Graph visualization engine using Sigma.js and ForceAtlas2 physics for the `agent-webui`. Implements intelligent mass assignment and radial clustering for high-mass structural nodes to prevent graph spaghetti at 100K+ scale. Provides full interactive CRUD capabilities via React overlay UIs.
- **CONCEPT:KG-2.6: Model Display Optimization** — Display-predict decoupling engine optimizing model `__str__()` for agent consumption independently of `predict()` logic. 5 strategies: `linear_collapse`, `piecewise_table`, `symbolic_equation`, `coefficient_summary`, `adaptive` (SmartAdditive pattern with per-feature R² gating). Includes linearization with R² threshold, hinge basis collapse, and bounded complexity budgets (`DisplayComplexityBudget`). Results persist as `ModelDisplayNode` with `DISPLAY_OF` edges. Based on arXiv:2605.03808. New module: `agent_utilities/knowledge_graph/model_display.py`.
- **Agentic-iModels KG Models** — New Pydantic models: `IModelNode`, `InterpretabilityTestNode`, `ModelDisplayNode`, `IModelCandidate`, `DisplayComplexityBudget`, `ParetoPoint`. 3 new `RegistryNodeType` entries (`IMODEL`, `INTERPRETABILITY_TEST`, `MODEL_DISPLAY`) and 4 new `RegistryEdgeType` entries (`EVOLVED_MODEL`, `TESTED_INTERPRETABILITY`, `DISPLAY_OF`, `PARETO_DOMINATES`). New module: `agent_utilities/models/imodel.py`.
- **Agentic-iModels OWL Ontology** — Extended `ontology.ttl` with 4 OWL classes (`IModel`, `InterpretabilityTest`, `ModelDisplay`, `ParetoFrontierEntry`), transitive `evolvedModel` property for model lineage inference, and 3 datatype properties. Updated `owl_bridge.py` with 7 new promotable types.
- 70 new unit tests across `test_imodel_evolver.py` (25), `test_interpretability_tests.py` (24), and `test_model_display.py` (21).
- **CONCEPT:KG-2.5: Topological Analogy Engine** — Leverages exact subgraph isomorphism (networkx VF2) and vectorized embeddings (`EncPI`) to find analogous subgraphs across different domains, enabling structural pattern matching and cross-domain innovation extraction within the Knowledge Graph. New module: `agent_utilities/knowledge_graph/analogy_engine.py`.
- **CONCEPT:KG-2.2: OWL-Driven Semantic Subsumption** — Enables hierarchy-aware zero-shot ontology alignment. Automatically computes topological embedding cosine similarity against OWL class prototypes to infer and inject newly discovered concepts directly into the correct class lineage. New module: `agent_utilities/knowledge_graph/semantic_subsumption.py`.
- **Finance Schema Pack Enhancements** — Integrated abstractions from OpenAlice (Trading-as-Git, `VERSIONED_TRADE_COMMIT`, `EXECUTION_GUARD`, `UNIFIED_TRADING_ACCOUNT`) and Kronos (`TIME_SERIES_FORECAST`) into `FinanceSchemaPack`. Added corresponding `RegistryNodeType` and `RegistryEdgeType` entries in `knowledge_graph.py`.
- **CONCEPT:OS-5.1: Topological Vulnerability Scanner** — Enhances security by scanning execution graphs for structural vulnerabilities (e.g., untrusted data flows, dependency deadlocks) by matching them against known risk subgraphs using the Analogy Engine. New module: `agent_utilities/security/topological_scanner.py`.
- **CONCEPT:KG-2.6: Research Intelligence Pipeline** — Automated end-to-end research ingestion cycle: ScholarX Discovery → 9-domain Relevance Scoring → Tiered Ingestion (full KG + SQLite for relevant papers ≥ 3.0, abstract-only for marginal ≥ 1.0) → OWL Enrichment → Digest Generation. Supports arXiv papers via ScholarX, local files (PDF/HTML/Markdown), and web URLs. KG-backed watchlists via PolicyNodes. Integrated into MaintenanceCron for continuous discovery. New module: `agent_utilities/automation/research_pipeline.py`.
- **CONCEPT:KG-2.6: KG Source Resolver** — Bridges the Knowledge Graph (indexing/discovery layer) to the comparative-analysis skill (analysis layer) by materializing KG-stored documents to filesystem paths with metadata enrichment. Optional — gracefully returns empty when no KG is available. New module: `agent_utilities/knowledge_graph/source_resolver.py`.
- **Research Artifact Generator** — Creates structured LLM artifacts from KG-ingested papers: key contributions extraction, method detection, concept linkage discovery, application mapping to existing CONCEPT IDs, and periodic digest generation with emerging theme detection. New module: `agent_utilities/knowledge_graph/research_artifacts.py`.
- **Comparative Analysis KG Integration** — Updated `discover_projects.py` with `--kg-query` flag enabling KG-backed source resolution for the comparative-analysis skill. KG sources are materialized to `~/.scholarx/analysis/` as enriched markdown.
- **Conceptual Registry Parity** — Formalized 6 existing features into the 5-Pillar ecosystem architecture to ensure full zero-stub tracking:
  - **CONCEPT:ORCH-1.6: SDD Pipeline**
  - **CONCEPT:KG-2.1: Cross-Session Chat Recall**
  - **CONCEPT:KG-2.1: Project-Aware Context**
  - **CONCEPT:AHE-3.2: Agentic Engineering Patterns**
  - **CONCEPT:OS-5.1: Telemetry & Observability**
  - **CONCEPT:OS-5.1: Policy & Prompt Governance**
- Updated Concept Galaxy to 67 concepts (from 61).
- 45 new unit tests across `test_research_pipeline.py`, `test_source_resolver.py`, and `test_research_artifacts.py`.

- **CONCEPT:AHE-3.1: Multi-Strategy EvalRunner** — Multi-strategy evaluation runner with three scoring modes: exact match (Jaccard-normalized), semantic similarity (embedding cosine), and LLM-as-Judge (structured JSON prompt). Composite mode combines all three with configurable weights. Integrates with existing `EvaluationMonitor` for trend tracking and alerting. OWL-promoted via `eval_run` node type and `evaluated_by` edge. Ported from MATE's `eval_runner.py`.
- **CONCEPT:OS-5.1: Token Usage Tracker** — 4-bucket granular token analytics (prompt/response/thoughts/tool_use) with session aggregation, per-agent breakdown, and configurable budget alerting. Includes `record_from_llm_response()` adapter for pydantic-ai integration. OWL-promoted via `token_usage_record` node type. Ported from MATE's `token_usage_service.py` and `token_usage_callback.py`.
- **CONCEPT:OS-5.1: Audit Logger** — Append-only compliance audit trail with 30+ action constants, never-raise semantics, FIFO eviction, configurable retention cleanup, and query filtering by actor/action/resource/session. OWL-promoted via `audit_log` node type and `audited_by` edge. Ported from MATE's `audit_service.py`.
- **CONCEPT:OS-5.1: Guardrail Callback Engine** — Push-based input/output guardrail interception with block/redact/warn/log actions, regex and keyword pattern matching, and PolicyEngine adapter for unified evaluation. OWL-promoted via `guardrail_trigger` node type and `triggered_guardrail` edge. Ported from MATE's `guardrail_callback.py`.
- **CONCEPT:AHE-3.2: Agent Config Versioning** — Immutable agent configuration snapshots with sequential versioning, forward-only rollback (creates new version copying target's config), structured diffs between versions, and SUPERSEDES edge chains for KG-native version traversal. OWL-promoted via `agent_config_version` node type and `config_version_of` edge. Ported from MATE's `AgentConfigVersion` model.
- **OWL Bridge Extension** — Added `token_usage_record`, `audit_log`, `guardrail_trigger`, `agent_config_version`, `eval_run` to `PROMOTABLE_NODE_TYPES`; `audited_by`, `triggered_guardrail`, `config_version_of`, `evaluated_by` to `PROMOTABLE_EDGE_TYPES`.
- **MATE Comparative Analysis** — Functional gap closure between agent-utilities and the MATE framework (Control Room patterns, Eval Framework, Token Analytics, Audit Logging, Guardrail Callbacks, Config Versioning).
- Updated Concept Galaxy to 59 concepts (from 54).
- **CONCEPT:OS-5.1: Prompt Injection Scanner** — Pattern-based prompt injection and command injection scanner with 25+ threat vectors adapted from Goose's `scanner.rs`. Provides `PromptInjectionScanner` with text, tool-call, and conversation scanning modes. Integrates with `PolicyEngine` via `PromptInjectionPolicy` adapter. Security findings persist as `SecurityFindingNode` in the KG, enabling OWL transitive risk propagation via `propagatesRiskTo`.
- **CONCEPT:OS-5.1: Tool Repetition Guard** — Detects infinite tool call loops via consecutive identical call tracking and per-session budgets (configurable via `MAX_TOOL_REPEATS`, `MAX_TOOL_CALLS_PER_SESSION`). Adapted from Goose's `tool_monitor.rs`/`tool_inspection.rs`. Denied repetitions distill into `ExperienceNode` tactical rules (AHE-3.5) for cross-session loop avoidance. `RepetitionPolicy` adapter for `PolicyEngine`.
- **CONCEPT:KG-2.1: Token-Aware Context Compaction** — Intelligent context window management replacing naive truncation. Three strategies adapted from Goose's `context_mgmt/mod.rs`: `summarize_tools` (default), `drop_middle`, `progressive`. Compaction summaries persist as `EpisodeNode` snapshots for cross-session context recall via `MemoryRetriever`. Backward-compatible `compact_messages()` wrapper in `chat_persistence.py`.
- **CONCEPT:ORCH-1.3: Structured Retry Manager** — Shell-based success checks, on-failure hooks, and configurable timeouts for structured retry logic. Adapted from Goose's `retry.rs`. `RetryManager` evaluates `SuccessCheck` commands and manages attempt state. Retry outcomes feed into `TeamConfigNode.record_team_outcome()` reward signaling (AHE-3.3) for routing improvement.
- **Cross-Session Chat Recall** — `search_chat_history()` function in `chat_persistence.py` for keyword-based search across stored `Thread`/`Message` nodes. Adapted from Goose's `ChatHistorySearch` (Rust/SQLite). Uses KG Cypher backend with relevance scoring.
- **OWL Bridge Extension** — Added `security_finding` and `experience` to `PROMOTABLE_NODE_TYPES`; `detected_threat` and `triggered_retry` to `PROMOTABLE_EDGE_TYPES` for transitive risk inference.
- **CONCEPT:KG-2.6: Financial Trading Pipeline** — Added 5 new KG node types (`TradingSignalNode`, `OrderNode`, `PositionNode`, `PortfolioNode`, `StrategyNode`) and 6 edge types for modeling complete trading pipeline lifecycle. OWL-promoted with FIBO alignment for transitive provenance chains (e.g., Strategy → Signal → Order → Position → Portfolio).
- **CONCEPT:ECO-4.0: Market Data Connector Protocol** — Generic `DataConnectorProtocol` with auto-fallback chain and provenance tracking. Includes `DataConnectorRegistry` with prioritized failover, rate-limit awareness, and `DataFetchRecordNode` for immutable audit trails. OWL `fallsBackTo` declared as transitive for automated connector chain inference.
- **CONCEPT:ORCH-1.4: Swarm Preset Template Engine** — YAML-driven declarative multi-agent workflow engine with DAG-based dependency resolution, parallel dispatch identification, and variable substitution. Includes `SwarmPresetEngine` with topological sort, cycle detection, and layer-based execution ordering. KG-persisted via `SwarmPresetNode`, `SwarmRunNode`, `SwarmTaskRecordNode`.
- **CONCEPT:ORCH-1.3: Multi-Level Abstraction Layering** — Planners emit coarse-grained abstraction steps and delegate fine-grained execution to specialist nodes, reducing upfront planning token overhead.
- **Adaptive Model Routing & Reward-Driven Routing** — Included adaptive fast-path model selection (`gpt-4o-mini` fallback) for simple queries. Leverages ACO `pheromone_trails` to down-weight specialists with historically low success rates.
- **CONCEPT:KG-2.6: Risk Scoring Ontology Extension** — Domain-agnostic risk assessment framework with `RiskAssessmentNode`, `RiskFactorNode`, `RiskMitigationNode`. OWL `propagatesRiskTo` declared as transitive property enabling automated upstream risk chain inference via the OWL reasoner.
- **CONCEPT:AHE-3.4: Backtest Evaluation Harness** — Strategy evaluation harness with SQLite storage (separate from KG), walk-forward validation windows, benchmark comparison, and KG integration via `BacktestRunNode` and `BacktestMetricNode`. Connects to `StrategyNode` (KG-2.6) for full provenance chains.
- **CONCEPT:AHE-3.4: Horizon-Aware Task Curriculum** — Progressive horizon scheduling derived from Long-Horizon Training research (Kim et al., ICML 2026). Implements macro-action composition to reduce effective interaction steps, subgoal checkpoints for intermediate credit assignment, and configurable promotion policies (threshold/plateau/adaptive EMA). Integrates with `CognitiveScheduler` and `SwarmManager` for automatic horizon reduction during swarm execution.
- **CONCEPT:AHE-3.1: Decomposed Reward Signals** — Separates step-level reward (local constraint satisfaction) from trajectory-level reward (goal achievement) to prevent penalizing correct intermediate steps in failed trajectories. Implements `RewardDecomposer` engine with `R_total = R_trajectory + α·ΣR_step` formula, distillation insight extraction (correct-in-failures, incorrect-in-successes patterns), and integration with `ExperienceNode` pipeline.
- **Finance Schema Pack Expansion** — Expanded `FinanceSchemaPack` with all trading pipeline, risk scoring, data connector, and backtest types. Added retrieval boosts for `propagates_risk_to` (1.6×), `generated_signal` (1.5×), and `evaluated_strategy` (1.5×).
- **OWL Ontology Extension** — Added 18 new OWL classes and 17 new object/datatype properties to `ontology.ttl` including 3 transitive properties (`fallsBackTo`, `taskDependsOn`, `propagatesRiskTo`).
- **Backward Compatibility** — Added `SelfModelNode` alias for renamed `MemoryRetrieverNode` and `self_model.py` shim module for import compatibility.
- **CONCEPT:KG-2.5: Topological Mincut Partitioning** - Uses NetworkX Louvain detection to dynamically partition the Knowledge Graph into emergent topological clusters. Includes Label Propagation fallback for failed partitioning loops. Stable communities are persisted back to the Cypher backend via `maintenance_cron`, providing hierarchical waypoints for graph traversal.
- **CONCEPT:AHE-3.4: Temporal Drift & EWC Consolidation** - Tracks concept drift across node embeddings via coefficient of variation and cosine distance. Mitigates catastrophic forgetting by applying a lightweight Fisher-proxy Elastic Weight Consolidation (EWC++) when modifying established knowledge graph representations.
- **CONCEPT:AHE-3.4: Heavy Thinking Orchestration** - Two-stage parallel-then-deliberate reasoning pipeline adapted from HEAVYSKILL research. Features include: tiered hybrid complexity gating (heuristic → confidence → LLM fallback), configurable K parallel thinkers (default 4), a Serialized Memory Cache with thinking token pruning and trajectory shuffling, iterative convergence refinement, KG-native `TrajectoryNode` and `DeliberationNode` persistence, EncPI hyperedge mapping, and `WorkspaceAttention.deliberation_score()` for cross-trajectory consensus analysis.
- **CONCEPT:AHE-3.4: Distributed Agentic Evolution** - Transitioned the harness into an open-source hive mind where autonomous agents evolve globally. Features include:
  - **Evolutionary Vector (`genius-agent`)**: A background daemon (`--evolve` flag) that runs `SelfImprovementCycle` indefinitely, creating synthetic tasks and writing new skills to close gaps.
  - **Autonomous PR Generator**: A central `autonomous-contribution` skill in `universal-skills` that formats local KG breakthroughs (TeamConfigs, Skills) into Git branches and upstream PRs.
  - **Community Telemetry & Identity**: `TeamConfigNode` and `CallableResourceNode` metadata now include `origin` (`local` | `community` | `upstream`), deterministic hash identifiers, precise timestamps, and author fields to prevent duplication and establish primacy.
  - **Human-in-the-loop Guardrails**: All autonomously generated skills are explicitly flagged with `Author: Autonomous` in the SKILL.md frontmatter, requiring maintainer approval before centralized CI merge and global distribution.
- **CONCEPT:KG-2.4: Inductive Knowledge Hypergraphs** - Implemented Positional Interaction Encodings (`EncPI`) allowing the `HybridRetriever` to map n-ary hyperedges and perform zero-shot inductive reasoning across novel graph topologies by vectorizing relational intersections.
- **CONCEPT:KG-2.4: Offline/Async Knowledge Compression** — Added `TraceDistiller` to periodically run `SynthesisEngine` background tasks, abstracting episode-level execution traces into generalized `PreferenceNode` and `PrincipleNode` knowledge points.
- **CONCEPT:AHE-3.4: Memory-Aware Test-Time Scaling** - Integrated batch-parallel trajectory generation into the orchestration planner. Rather than distilling memory from a single sequential failure, the system scales inference across parallel Siblings, extracts reasoning across all paths, and maps them to hyperedges for zero-shot generalization and graph-native topological feedback.
- **CONCEPT:AHE-3.4: Decomposed Context Retrieval** - Modified HybridRetriever to decompose complex queries into abstract technical sub-queries for targeted multi-vector retrieval.
- **CONCEPT:AHE-3.4: Cross-Rollout Critique** - Added contrastive self-correction distillation to the `verifier`. When a failure is followed by a successful retry, the system contrasts the states to distill an action-level tactical fix.
- **CONCEPT:AHE-3.4: Experience Node Architecture** - Introduced `ExperienceNode` schema to store specific `Condition -> Action` tactical insights in the Knowledge Graph for continual learning.
- **CONCEPT:AHE-3.4: HTN & LATS** - Integrated Hierarchical Task Networks (`Task.subtasks`) and `LATSPlanner` (Monte Carlo Tree Search fallback) into the orchestration planner for recursive goal decomposition.
- **CONCEPT:AHE-3.4: Tiered Virtual Context Blocks** - Introduced `VirtualContextBlockNode` for tiered working/episodic memory scaling.
- **CONCEPT:AHE-3.4: Multi-Agent BFT Consensus** - Integrated `execute_bft_consensus` into `A2AClient` for Byzantine Fault Tolerance across agent peers.
- **CONCEPT:ORCH-1.3: Execution Budgets & Cost Governors** - Implemented `ExecutionBudget` and integrated enforcement inside `dispatcher_step` to prevent infinite LLM loops and cap USD costs.
- **CONCEPT:KG-2.1: Quiet-STaR Rationale Persistence** - Created `QuietStarRationaleNode` to persist internal chain-of-thought traces with reward gradients for self-improvement.
- **Context-Aware Entity Representations (CONCEPT:KG-2.2)** — Injects topological graph structure directly into node vector embeddings. Features include:
  - Expands Deep GraphRAG principles by appending multi-hop contexts (up to 2 levels of parents/children) directly into the stringified node description before embedding generation.
  - Automatically fetches and appends OWL-inferred relationships (e.g., transitive subclasses) into the node's context space.
  - Enables "topology-aware" vector semantic searches for free, drastically improving multi-hop accuracy.
  - `ContextualRepresentationBuilder` dynamically controls the depth and breadth of injected structural logic.
  - `re_embed_node` pipeline immediately re-embeds nodes when the OWL bridge downfeeds new inferred facts.
- **Wide-Search Orchestration (CONCEPT:ORCH-1.1)** — Pydantic-native Graph node architecture for orchestrating large-scale extractions. Features include:
  - Automates batch decomposition within the SDD pipeline by instructing orchestrators (planners/routers) to partition large extractions into parallel `ExecutionStep`s (`is_parallel=True`).
  - Implements a hybrid validation strategy inside `join_step` using `WideSearchWorkboard`.
  - Fast-path: Native Pydantic schema validation for expected row counts and schema conformity.
  - Slow-path: `wide_search_joiner` LLM repair node to standardization data, fix schema mismatches, or signal re-plans on fast-path failures.
- **Trace Distillation Error Categorization (CONCEPT:AHE-3.1)** — Categorizes orchestrator (`ORCHESTRATOR_SKILL`) vs worker (`WORKER_SKILL`) failure modes through AHE skill distillation to enable self-evolving updates. RLM prompt updated to support proposing targeted ComponentEdits for orchestration logic vs worker tools.
- **A2A Config File (CONCEPT:ECO-4.0)** — File-based external A2A agent discovery via `a2a_config.json`. Mirrors `mcp_config.json` symmetry. Features include:
  - `secret://`, `env://`, `vault://` URI-based auth token resolution via `SecretsClient.resolve_ref()`.
  - Soft-fail startup: unreachable `.well-known/agent-card.json` endpoints log a warning and skip, never blocking server startup.
  - Periodic background re-fetch (`A2A_REFRESH_INTERVAL`, default 300s) to detect capability changes from remote agents.
  - Full KG ingestion: agent cards are registered as `CallableResource` nodes with embeddings, making them eligible for affinity-based swarm selection.
  - Cache invalidation: bulk ingestion triggers `invalidate_registry_cache()` (CONCEPT:ORCH-1.2) to keep the hot cache synchronized.
- **Unified Specialist Model (CONCEPT:ORCH-1.2)** — Collapses the artificial `prompt` / `mcp` agent type distinction into a single `specialist` type. Any specialist can now host any combination of MCP tools and/or agent skills. A2A agents remain their own type (`a2a`) due to the fundamentally different remote execution protocol. Legacy `agent_type` values are normalized at read time for full backward compatibility.
- New `A2A_CONFIG` and `A2A_REFRESH_INTERVAL` environment variables.
- New module: `agent_utilities/protocols/a2a_config.py` (config loader, auto-discovery, periodic refresh).
- Updated Concept Galaxy to 29 concepts (from 27).
- **Confidence-Gated Model Router (CONCEPT:ORCH-1.2)** — Adaptive model tier routing using runtime confidence signals from specialist consensus. When WorkspaceAttention scores indicate high agreement across specialist outputs, the model tier is automatically downgraded to reduce cost. Low confidence triggers escalation to heavier tiers. Based on the Squeeze Evolve multi-model orchestration framework (Maheswaran et al., 2026). Features include:
  - `ModelRegistry.pick_for_task_adaptive()` with `confidence_signal` and `routing_percentile` parameters.
  - Integration with `pick_specialist_model()` in the graph executor for automatic confidence-gated routing.
  - Composition with CONCEPT:OS-5.2 (Homeostatic Downgrade): budget pressure routes first, then confidence adjusts within the budget-allowed tier range.
  - `routing_confidence_log` on `GraphState` for per-specialist routing decision observability.
  - Soft SelfModel (CONCEPT:KG-2.1) integration: blends 70% runtime confidence + 30% historical proficiency when available; degrades gracefully when absent.
- **Evolutionary Aggregation Engine (CONCEPT:ORCH-1.2)** — Group-level diversity scoring and three-tier aggregation for specialist outputs. Extends WorkspaceAttention (CONCEPT:ORCH-1.2) with group fitness computation (confidence and diversity signals) and routes aggregation to the most cost-effective strategy:
  - `MAJORITY_VOTE`: Free — no LLM call when all specialists agree.
  - `LIGHT_MODEL`: Cheap model synthesis for moderate-confidence groups.
  - `HEAVY_MODEL`: Deep aggregation with reasoning-tier models for low-confidence, high-diversity groups.
  - `ConvergenceMonitor` in the CognitiveScheduler (CONCEPT:OS-5.2) detects diversity collapse and triggers early loop termination.
  - Configurable `population_size` (N=4) and `group_size` (K=2) for fully adjustable evolutionary loops.
- New module: `agent_utilities/graph/evolutionary_aggregation.py`.
- New documentation: `docs/squeeze-evolve-routing.md`.
- `routing_percentile` field on `GraphDeps` (env var: `ROUTING_PERCENTILE`, default 50.0).
- `ROUTING_DECISION` node type and `ROUTED_BY`/`AGGREGATED_FROM` edge types in Knowledge Graph.
- Updated Concept Galaxy to 40 concepts (from 38).
- 46 new unit tests across `test_confidence_routing.py` and `test_evolutionary_aggregation.py`.
- **Schema Packs (CONCEPT:KG-2.2)** — Domain-configurable KG profiles that scope the active node types, edge types, retrieval boosts, and OWL extensions to a specific domain. Inspired by gbrain#587 schema-pack proposal. Features include:
  - `SchemaPack` base model with dual operating modes: `ADDITIVE` (layer on top of core) and `EXCLUSIVE` (only pack + protected core types).
  - Protected core types (memory, episode, person, concept, etc.) that are always active regardless of mode.
  - Per-pack retrieval boost multipliers for domain-specific edge weighting.
  - Schema pack registry with `get_schema_pack()` factory and `register_schema_pack()` for runtime extensions.
  - Four pre-built packs: `core` (default), `research-state`, `biomedical`, `finance`.
  - `SchemaPackNode` for KG persistence of active pack configuration.
  - OWL Bridge integration: `PROMOTABLE_NODE_TYPES` and `PROMOTABLE_EDGE_TYPES` filtered through active pack.
- **Backlink-Density Retrieval Boost (CONCEPT:KG-2.2)** — Logarithmic in-degree-based scoring modifier in `HybridRetriever` that boosts hub entities with many inbound edges. Pack-configurable via `backlink_boost_strategy` (`global`, `context_only`, `disabled`) and `backlink_boost_factor` (default 0.1). Based on gbrain's observed +31% P@5 improvement.
- **KG Eval Capture (CONCEPT:KG-2.2)** — Lightweight regression testing harness for Knowledge Graph retrieval. Records query-result pairs to a separate SQLite database (`eval_log.db`) to prevent KG contamination. Features include:
  - `KGEvalCapture.capture()` — append-only recording of queries, results, scores, and latency.
  - `KGEvalCapture.replay()` — re-runs captured queries and reports Jaccard@k, top-1 stability, and latency delta.
  - `export()` / `purge()` for maintenance. Controlled by `KG_EVAL_CAPTURE` env var (disabled by default).
- New module: `agent_utilities/models/schema_pack.py`.
- New package: `agent_utilities/models/schema_packs/` with 4 pre-built domain profiles.
- New module: `agent_utilities/knowledge_graph/eval_capture.py`.
- `SCHEMA_PACK` node type and `USES_SCHEMA_PACK` edge type in Knowledge Graph.
- Updated Concept Galaxy to 43 concepts (from 40).
- 47 new unit tests across `test_schema_packs.py`, `test_backlink_boost.py`, and `test_eval_capture.py`.
- **Conductor Workflow Specification (CONCEPT:ORCH-1.1)** — Refined natural-language subtask instructions per specialist step. The router/planner now generates a focused `refined_subtask` field on each `ExecutionStep`, crafting targeted sub-goals instead of forwarding the raw user query. Inspired by the RL Conductor's per-step subtask specification (Nielsen et al., ICLR 2026).
- **Execution Visibility Graph (CONCEPT:ORCH-1.1)** — Per-step `access_list` controlling which prior step results are visible to each specialist. `_resolve_access_context()` helper filters `results_registry` before injection. Supports `["all"]`, specific node_ids, or empty for no context sharing.
- **Model Synergy Tracker (CONCEPT:AHE-3.3)** — Per-model-combination EMA success tracking in SelfModel (CONCEPT:KG-2.1). `model_synergies` dict on `SelfModelNode` tracks sorted pipe-delimited model combination keys. `SelfModel.get_best_synergies()` filters by available models for intelligent recombination.
- **Recursive Graph Orchestration (CONCEPT:ORCH-1.1)** — Nested `run_graph()` calls for self-referential test-time scaling. `recursive_orchestrator` specialist spawns inner graph executions with parent context. `RecursiveContext` dataclass and `MAX_RECURSION_DEPTH` env var (default 2) for depth control.
- New module: `agent_utilities/graph/recursive_executor.py`.
- New documentation: `docs/conductor-orchestration.md`.
- Updated Concept Galaxy to 47 concepts (from 43).
- ~60 new unit tests across `test_conductor_workflow.py`, `test_visibility_graph.py`, `test_model_synergy.py`, and `test_recursive_orchestration.py`.
- **Structural Fingerprint Engine (CONCEPT:KG-2.3)** — AST-based signature extraction and three-level change classification (NONE/COSMETIC/STRUCTURAL) for incremental KG updates. Generic capability that avoids costly full re-ingestion when only cosmetic changes occur. Includes `FingerprintManager` for workspace-level scanning and `detect_stale_files()` for git-based staleness detection.
- **Graph Integrity Validator (CONCEPT:KG-2.3)** — Non-blocking 4-tier graph validation inspired by Understand-Anything's `graph-reviewer`. Features include:
  - Tier 1 (Auto-fix): LLM type alias normalization (30+ node aliases, 30+ edge aliases), score clamping, missing name defaults.
  - Tier 2 (Integrity): Dangling edges, missing node types, untyped edges, duplicate IDs.
  - Tier 3 (Quality): Orphan nodes, self-referencing edges, generic descriptions, underscored hub detection.
  - Tier 4 (Fatal): Zero-node graphs, graph fragmentation below 50% threshold.
  - Integrated as 15th pipeline phase (`validate`) with `KGEvalCapture` (CONCEPT:KG-2.2) trend storage.
- **Entity-Claim Extraction / MAGMA Completion (CONCEPT:KG-2.2)** — Two-phase entity-claim extraction that fills the MAGMA epistemic view with real data. Features include:
  - Deterministic Phase 1: Regex-based extraction of citations, wikilinks, and assertion patterns.
  - `ClaimNode` model with confidence scoring, claim types, and epistemic metadata.
  - New edge types: `BUILDS_ON`, `EXEMPLIFIES`, `AUTHORED_BY` (joining existing `CONTRADICTS` and `CITES`).
  - `retrieve_epistemic_view()` fully implemented with real Cypher queries (replacing stub) and NetworkX fallback.
- New module: `agent_utilities/knowledge_graph/fingerprint.py`.
- New module: `agent_utilities/knowledge_graph/graph_validator.py`.
- New module: `agent_utilities/knowledge_graph/kb/entity_claim_extractor.py`.
- New pipeline phase: `validate` (15th phase, runs after `knowledge_base`).
- `CLAIM` node type and `BUILDS_ON`/`EXEMPLIFIES`/`AUTHORED_BY` edge types in Knowledge Graph.
- Updated Concept Galaxy to 54 concepts (from 47).
- 43 new unit tests across `test_graph_validator.py`, `test_entity_claim_extractor.py`, and `test_fingerprint.py`.

### Changed
- `MCPAgent.agent_type` default changed from `"prompt"` to `"specialist"`.
- `DiscoveredSpecialist.source` values unified to `"specialist"` or `"a2a"`.
- `discover_agents()` collapsed from 3 type branches to 2 (specialist + a2a).
- Executor `_execute_agent_package_logic()` simplified from 3 execution paths to 2 (remote A2A vs. unified specialist).
- `build_agent_app()` and `create_agent_server()` accept new `a2a_config` parameter.
- `initialize_graph_from_workspace()` accepts `a2a_config` parameter and syncs A2A agents alongside MCP agents during startup.

## [0.3.0] - 2026-05-02

### Added
- **First Principles Architecture** — Four new foundational concepts (CONCEPT:ORCH-1.2 through CONCEPT:ECO-4.0) that rewire the routing, dispatch, and feedback layers from first principles.
- **Registry Hot Cache (CONCEPT:ORCH-1.2)** — Session-scoped `_RegistryCache` singleton providing O(1) specialist lookups. Replaces full registry scans on every routing call, reducing prompt bloat by filtering to only the top-7 relevant specialists per query.
- **Event-Driven Cache Invalidation** — Cache auto-invalidates on MCP reload (`/mcp/reload`), pipeline completion, Self-Model session updates, and TeamConfig promotions. No stale-cache risk.
- **TeamConfig Promotion (CONCEPT:AHE-3.3)** — `promote_coalition_to_template()` persists successful specialist coalitions as reusable `TeamConfigNode` templates in the Knowledge Graph. `find_matching_team_config()` enables 3-stage hybrid routing: TeamConfig → Self-Model bias → LLM planning.
- **TeamConfig Reward Tracking** — `record_team_outcome()` records success/failure outcomes against team templates, enabling reward-weighted team selection over time.
- **RLM + TeamConfig Synergy** — When a `TeamConfig` is selected and input exceeds size thresholds, RLM capability is auto-attached to specialists via `capability_overrides`.
- **AgentCapability Type System (CONCEPT:ORCH-1.2)** — `AgentCapabilityNode` formalized as a first-class KG node with `auto_activate`, `trigger_conditions`, and `handler_module` fields. Capabilities are auto-activated in the executor based on input constraints (e.g., RLM for large payloads, critic for code).
- **PlannerGraphSkill (CONCEPT:ECO-4.0)** — A2A-native routing entry point via `PlannerGraphSkill` registered in `server/app.py`. When a `graph_bundle` is available, A2A requests bypass LLM orchestration and route directly through the graph planner.
- **Self-Model Feedback Loop** — Post-execution verification in `synthesizer_step` now feeds outcomes back to `SelfModel.update_after_session()` and `record_team_outcome()`, enabling recursive learning.
- **WorkspaceAttention Scoring** — `WorkspaceAttention` (GWT) scores are computed and logged per-specialist during execution for data-driven specialist prioritization.
- **Process Lifecycle Management** — `atexit` and `SIGTERM`/`SIGINT` handlers in `server/__init__.py` ensure all child processes (MCP servers, TUI, background threads) are gracefully killed on server exit. Uses child-only `pgrep` pattern instead of `killpg` to avoid self-termination.
- **USES_PROMPT Edges** — `link_prompt_to_agent()` creates `USES_PROMPT` edges between specialist nodes and their JSON prompt templates for full traceability.
- 33 new unit tests across `test_config_helpers.py`, `test_team_config.py`, and `test_capability_nodes.py`.
- 3 new documentation files: `first-principles.md`, `registry-cache.md`, `process-lifecycle.md`.

### Changed
- Router now performs 3-stage hybrid routing: (1) TeamConfig match → (2) Self-Model proficiency bias → (3) LLM planning fallback.
- Specialist filtering reduced from O(N) full registry scan to O(7) via `get_relevant_specialists()` using query-keyed caching.
- Verification synthesizer now feeds execution outcomes back to both Self-Model and TeamConfig for continuous improvement.
- `RegistryNodeType` enum extended with `TEAM_CONFIG` and `AGENT_CAPABILITY` types.
- `RegistryEdgeType` enum extended with `HAS_CAPABILITY`, `REUSED_TEAM`, and `USES_PROMPT` types.
- Server process cleanup uses targeted `pgrep -P` child enumeration instead of `os.killpg()` to avoid killing the process group (which would terminate test runners).

### Fixed
- mypy `call-arg` error: `SelfModel.update_after_session()` now correctly receives `GraphState` instead of kwargs.
- mypy `assignment` error: `log_file_path` now guards against `None` before assigning to env dict.
- ruff `F401`: Removed unused `contextlib.suppress` import from `server/__init__.py`.
- Bandit `B110` suppressions added for defensive `try/except pass` patterns in cleanup handlers.

## [0.2.41] - 2026-04-29

### Added
- Direct graph execution fast-path in AG-UI endpoint — bypasses LLM tool-call overhead
- `AGUIGraphEmitter` module for translating graph events to AG-UI wire format (0:/2:/8:/9: prefixes)
- `run_graph_iter()` and `execute_graph_iter()` — step-by-step graph execution using `graph.iter()` beta API
- Per-step state snapshots and elicitation hooks in the iter-based execution path
- `GRAPH_DIRECT_EXECUTION` env var (default: `true`) to toggle direct dispatch
- `SecretsClient` with pluggable backends: InMemory (Fernet-encrypted), SQLite (persistent + encrypted), HashiCorp Vault (enterprise)
- URI-style secret references: `vault://`, `env://`, `sqlite://` schemes
- `SECRETS_BACKEND`, `SECRETS_SQLITE_PATH`, `SECRETS_VAULT_URL` configuration
- `secrets_client` field on `GraphDeps` for graph execution credential resolution
- `docs/secrets-auth.md` comprehensive documentation (CONCEPT:OS-5.1)
- Concept marker backfill: CONCEPT:OS-5.0, CONCEPT:ORCH-1.1, CONCEPT:OS-5.2, CONCEPT:AHE-3.0, CONCEPT:ECO-4.0, CONCEPT:OS-5.1 across tests and source
- `auth.py` JWT Bearer token validation using `authlib` + JWKS caching (CONCEPT:OS-5.1)
- Combined auth dependency: accepts API key OR JWT Bearer token (gradual migration)
- `AUTH_JWT_JWKS_URI`, `AUTH_JWT_ISSUER`, `AUTH_JWT_AUDIENCE` configuration
- `ALLOWED_ORIGINS` and `ALLOWED_HOSTS` for configurable CORS/host restriction
- MCP subprocess token forwarding via `AGENT_USER_TOKEN` env injection
- `auth = ["authlib>=1.4.0"]` optional extra in `pyproject.toml`
- 69 new unit tests (30 secrets, 20 auth, 19 emitter/iter)

### Changed
- ACP adapter refactored to use pydantic-acp's `agent_factory` callback for per-session agent creation
- Removed `REQUESTED_MODEL_ID_CTX` workaround from ACP's `run_graph_flow` closure
- Unified execution layer (`graph/unified.py`) now exports `execute_graph_iter` as a first-class entry point
- `cryptography>=44.0.0` added as core dependency for Fernet encryption
- `vault` optional extra added to `pyproject.toml` for `hvac>=2.3.0`
- CORS middleware now reads `ALLOWED_ORIGINS` instead of hardcoded `["*"]`
- TrustedHostMiddleware now reads `ALLOWED_HOSTS` instead of hardcoded `["*"]`

## [0.2.40] - 2026-04-28

### Added
- LLM Council integration with 7 role-based advisor prompts
- Hybrid OWL Reasoning sidecar with HermiT/Stardog inference
- Standard ontology schemas (BFO, Schema.org, PROV-O, Dublin Core, SKOS)
- Concept traceability markers (`@pytest.mark.concept`) for doc-test alignment
- `.env.example` template for developer onboarding
- Project Structure section in AGENTS.md

### Changed
- Restructured test suite into domain-driven subdirectories (core, graph, integration)
- Deprecated `MEMORY.md` in favor of Knowledge Graph native storage
- Bumped pre-commit hooks: ruff 0.15.12, mypy 1.20.2, bandit 1.9.4

### Fixed
- Git merge conflict artifacts in atlassian-agent, langfuse-agent, repository-manager
- Duplicate import errors in protocol adapters
- TOML configuration errors in .bumpversion.cfg
- Broken file references in AGENTS.md

## [0.2.39] - 2026-04-26

### Added
- AG-UI protocol adapter for web and terminal frontends
- Human-in-the-loop tool approval with `ApprovalManager`
- MCP elicitation callback support via `global_elicitation_callback()`

### Changed
- Unified protocol layer: ACP, A2A, MCP, AG-UI all served from single FastAPI server
- Migrated flat-file state (MEMORY.md, USER.md, HEARTBEAT.md) to Knowledge Graph

## [0.2.38] - 2026-04-24

### Added
- 14-phase Intelligence Pipeline
- LadybugDB as default embedded graph backend
- Knowledge Base layer with LLM-maintained wiki
- MAGMA-inspired orthogonal reasoning views (Semantic, Temporal, Causal, Entity)

### Changed
- Replaced Neo4j-only backend with pluggable graph abstraction (LadybugDB, FalkorDB, Neo4j)
