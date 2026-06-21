# The Agent-Operator Program — closing the loops

> How an agent (with local LLMs) runs an enterprise on agent-utilities, and how we
> make agent-utilities easy for that agent to evolve. The capabilities exist; the
> work is **closing two loops** — the *operating* loop (sense → decide → act →
> **verify → learn**) and the *build* loop (build → **deploy → observe → improve**).
> Almost every item is an instance of two abstractions: **synthesize-from-KG
> context** (`context(domain, intent)`) and **measured outcomes** (`record_action_
> outcome`). Build those keystones and the rest are small plug-ins.

Status legend: ✅ shipped · 🔧 designed (next phase) · 🔭 future.

---

## Phase 1 — the keystones (✅ shipped this change)

| Item | Concept | What |
|---|---|---|
| **Context plane** | KG-2.136 | `context_plane.synthesize_context(domain, query, intent)` — a provider registry generalizing `code_context`. `graph_analyze action=explain` + REST `/graph/analyze/explain`. Providers: `code`, `ops`. The cockpit is *more providers here*, not new subsystems. |
| **Ops diagnosis** | KG-2.137 | `ops_context.diagnose_ops` — synthesizes the live `:Task`/lane/queue state into "is it healthy / why is the maint lane backing up / what's poisoned" with task+lane citations and a remediation hint. |
| **Universal action-outcome** | AHE-3.62 | `FeedbackService.record_action_outcome` + `graph_feedback correction_type=action_outcome` — ANY action (a context answer, a deploy, a ticket close, a routing choice) reports `{success, reward?, expected?, observed?}` → reward-EMA + eval case. The back-half of the operating loop. |
| **Shared source paths** | KG-2.136 | `core/source_paths.py` — one home for mount-alias (`/au`→canonical) normalization + repo labeling (was duplicated in `code_context`). |
| **Gen-script self-resolution** | — | `gen_graphos_manifest.py` inserts its own repo root on `sys.path` so regenerating from a worktree reflects the worktree, not the editable-installed copy (a real trap that silently dropped new actions). |

---

## Phase 2 — finish the operating loop (🔧)

### 2.1 `deployment_status` provider — "where does this run / is my change live?"
**Problem:** the #1 build-loop papercut — which code actually executes (worktree vs
canonical vs editable-install vs `/au` mount vs served container), and is my merge
live. Today: `ps | grep` and guess.
**Design:** a `deploy` context provider that synthesizes from git (canonical HEAD +
dirty), the `MOUNT_ALIASES` map, the worktree list, and the KG `serves`/`servedBy`
routes; plus a small addition to `system_doctor`/health that makes graph-os **report
its loaded git rev at startup**, so the provider can say "served rev R₀ vs canonical
R₁ → your change is/ isn't live; restart to guarantee." Surface: `graph_analyze
action=explain target=deploy:status`.

### 2.2 Per-connector coverage + freshness (generalize OS-5.47)
**Problem:** the world-model is only trustworthy for *code* today. Every domain
(tickets, deploys, processes) needs the same coverage/freshness SLA or the agent
falls back to hitting source systems.
**Design:** lift `ingestion/coverage.py` into a connector-driven check — for each
registered `mcp_tool` source, assert "ingested ≥1 node, last `DeltaManifest`
watermark < SLA". One new doctor check `connector_coverage` iterating the connector
registry; surfaces per-connector freshness in `system_doctor` and as an `ops`
sub-answer.

### 2.3 Goals as durable contracts (SLA + escalation)
**Problem:** to run an enterprise I set intent ("triage every P1 within 1h") and the
system pursues it, escalating only on a boundary. `graph_goals` + `schedule_engine` +
`ActionPolicy` are the parts; the missing object is the *contract*.
**Design:** extend the `:Goal` node with `sla` (deadline/cadence), `playbook` (the
workflow/agent that pursues it), `escalation` (policy condition → notify via
`graph_reach`), and `outcome` (fed by `record_action_outcome`). A maintenance tick
evaluates open goals against their SLA, runs the playbook, escalates on breach, and
records the outcome — closing sense→act→verify→learn for goals.

### 2.4 Autonomy ramp — earned ActionPolicy scope
**Problem:** I shouldn't get full autonomy on day one, and shouldn't ask forever.
**Design:** wire the capability reward-EMA (already maintained by
`record_action_outcome`) into `ActionPolicy.classify`: a per-(actor, action-class)
trust score; when verified-success EMA over N outcomes clears a threshold, the action
class graduates from `ask` → `allow` for that actor (with an audit trail and a
one-way ratchet the operator can reset). "200 P3s closed correctly → close P3s
without approval."

---

## Phase 3 — economics & enterprise breadth (🔧/🔭)

### 3.1 Local-LLM adaptive routing with a quality floor
**Problem:** lane→model routing is a no-op under one model; running an enterprise on
local compute needs cheapest-model-that-clears-the-bar with a fallback ladder.
**Design:** a router over `SamplingProfile` + lane `model_role` + the capability
reward-EMA: route each task to the cheapest local model whose measured success-EMA
for that task-class clears a floor; on low self-reported confidence or a failed
verify, escalate one rung up the ladder (local-small → local-large → frontier).
Every routing decision is an `action_outcome`, so the router *learns* the cheapest
model that works per task-class. Measure: cost/task and escalation rate.

### 3.2 Enterprise domain providers (tickets, deploys, processes, finance)
Each is just another context-plane provider over an existing connector's ingested
data: `domain_context("tickets", intent=health)` = the open-P1 picture with
citations; `("deploys")` = drift/rollout health; `("process")` over the Camunda/ARIS
subgraph. The cockpit emerges as the union of providers — no new subsystem.

### 3.3 Diagnose-everything (`explain target=<domain>:why`)
Generalize the ops `why` synthesis: every number in any domain gets a one-query
causal explanation grounded in the KG lineage (the data is already there).

---

## Phase 4 — developer experience for the agent (🔧)

### 4.1 Dogfood onboarding + gotchas-in-KG
Make the KG the reliable first stop (freshness SLA green + the task-start prime
live), and add a first-class **`:Gotcha`** node attached to code/modules, surfaced
in `code_context` — so hard-won traps ("`gen_*` scripts import the canonical copy",
"`_get_engine()` hangs in a one-off host process", "AGE `()-[r:calls]->()` count
returns 0") are inherited, not rediscovered. Capture: a `graph_feedback
correction_type=gotcha` that pins a note to a module node.

### 4.2 "Add one graph action" scaffolder
A skill/CLI that, given a tool + action name, generates the dispatch branch, the REST
twin, the manifest entry, the description stub, and a test stub — the ~6-file +
3-generator wiring I did by hand each feature, so adding a surface is one command and
can't drift.

### 4.3 Diff-scoped "validate my change" command
One entry point running the pinned mypy + touched tests + the guardrail gates on the
diff only — turning the minutes-long full pre-commit into a seconds-long inner loop.

### 4.4 Invisible coordination
Writing a `CONCEPT:` marker auto-reserves it (a pre-commit hook calling the
allocator); starting work auto-takes a worktree. Remove the reserve-via-CLI-because-
MCP-is-read-only papercut by having the reservation MCP detect a read-only ledger and
fall back to the CLI path transparently.

### 4.5 Close the deploy loop safely
A policy-gated self-deploy: `restart served graph-os behind a health-gate +
auto-rollback`, gated by `ActionPolicy` (`deploy.restart`), so a verified-green change
can go live without a human on the critical path — the missing last mile of the build
loop.

---

## The shape when the loops are closed

A session/operator does: **ask the plane** (`explain` any domain) → act through the
**governed** action core → **record the outcome** → the world-model + playbooks +
router improve → the next loop is cheaper and more autonomous. Code was the proving
ground (`code_context`); the enterprise is the same pattern, more providers, more
outcomes, more earned trust.
