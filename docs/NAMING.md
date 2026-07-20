# Naming Conventions

## Class Naming Rules

This document establishes naming conventions to prevent further **Engine proliferation** (50+ `*Engine` classes observed at audit time).

### `*Engine` — Standalone Systems

Reserved for **top-level entry points** that own their own lifecycle (startup → operation → shutdown). These are the primary interfaces consumers import.

**Examples:**
- `IntelligenceGraphEngine` — KG core lifecycle owner
- `ParallelEngine` — Canonical parallel task execution
- `GraphComputeEngine` — Rust-backed graph algorithms

**Rule:** If your class is a mixin, strategy, or helper, it is **not** an Engine.

---

### `*Mixin` — Composed via MRO

For classes that are mixed into an Engine via multiple inheritance. They add behavior but do not function standalone.

**Examples:**
- `QueryMixin` — Adds query methods to IntelligenceGraphEngine
- `PersistenceMixin` — Adds save/load lifecycle hooks
- `ObservabilityMixin` — Adds metrics/tracing

**Rule:** Mixins must not have `__init__` (or only call `super().__init__`). They must not be instantiated directly.

---

### `*Strategy` — Pluggable Algorithms

For classes implementing a specific algorithm or policy that can be swapped. These follow the Strategy pattern.

**Examples:**
- `CompactionStrategy` — Token-aware compaction policy
- `RoutingStrategy` — Task-to-specialist routing logic
- `ScoreStrategy` — Relevance scoring algorithm

**Rule:** Strategies are stateless or nearly stateless. They implement a common interface and are injected into Engines.

---

### `*Coordinator` — Cross-System Orchestration

For classes that coordinate across multiple systems without owning execution themselves.

**Examples:**
- `DistributedCoordinator` — NATS/local task routing
- `MemoryEngineManager` — Memory lifecycle coordination

**Rule:** Coordinators delegate execution to other components. They are orchestration glue, not execution engines.

---

### `*Protocol` — Structural Typing Interfaces

For `typing.Protocol` classes that define structural contracts.

**Examples:**
- `OrchestratorProtocol` — Dispatch + status interface
- `CapabilityHandlerProtocol` — Capability execution interface

**Rule:** Protocols must use `@runtime_checkable` when isinstance checks are needed at runtime.

---

## Fleet skill & prompt names — globally unique (CONCEPT:AU-OS.deployment.agent-factory-autoload)

Every agent-package contributes skills via `agent_utilities.skill_providers`. When the
whole fleet's skills are installed together with `universal-skills` and the hub's own
`agent_utilities/skills`, the current runtime resolver **rejects two skills sharing a
`name:` frontmatter value before ingestion**. Skill names therefore MUST be **globally
unique across the installable namespace** (agents/* + universal-skills +
agent-utilities skills; `skill_graphs`/`skill-graphs` KG-ingestion corpora are
excluded). There is no precedence or shadowing compatibility path.

- **Hard rule (enforced):** no duplicate skill `name:`; no duplicate prompt `task`
  within a package's `prompts/`. The runtime resolver fails closed, and
  `scripts/check_skill_name_collision.py` provides the pre-commit/CI gate
  (`guardrails.yml`), baseline-gated via `scripts/skill_collision_baseline.txt`.
- **Recommended convention (advisory):** prefix a package's skills with its slug —
  `<pkg-slug>-<capability>` (e.g. `servicenow-cmdb`, `fan-manager-thermal`). The
  `<short>-starter` scaffolded name uses `slug.rsplit("-",1)[0]` (e.g. `arr-mcp` → `arr`).
- **Capability-domain names** (e.g. `dns-record-manager`, `ipmi-bmc-manager`, and
  the deliberately cross-package `caddy-uptime-sync`) are allowed
  **only while they stay globally unique** — the gate flags them as advisories, not failures.
- Prompt specialists are namespaced `prompt:<source>/<task>` so cross-package `task` reuse is
  safe; keep `<source>` = the package slug.

## Renaming an Existing Type

Renames are atomic across the ecosystem. Search every consumer under
`agent-packages/`, update imports and annotations in the same change, then delete the
old symbol. Do not publish aliases, dual names, or a deprecation window. If a type does
not satisfy the rules above, rename it when touched rather than carrying its old name.
