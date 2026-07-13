---
name: ecosystem-validation
skill_type: workflow
description: >-
  Turns the manual "validate a fleet package/skill, run its tests, delegate a
  grounded task to it, hunt for known bug-classes, and feed findings back into
  the KG" process into a SELF-HOSTED, delegation-ready capability the local LLM
  runs against ANY target — a fleet agent under agent-packages/agents/*, the
  agent-utilities package itself, or a universal-skill / agent-utilities skill.
  Composes existing primitives (graph_orchestrate execute_agent, graph_write,
  graph_feedback, pytest via the tier4 pattern, deterministic structural/bug-class
  scans) rather than re-implementing them. Use when asked to "validate <target>",
  "test this agent/skill end to end", "bug-hunt the fleet", "run ecosystem
  validation", or to set up continuous automated fleet validation via the loop
  engine or the scheduler.
domain: infrastructure
tags:
  - validation
  - regression
  - bug-hunt
  - delegation
  - evolution
  - grounding
  - fleet
  - continuous-validation
requires:
  - graph-os
metadata:
  author: Genius
  version: '0.1.0'
---

# ecosystem-validation

The single entrypoint that exercises **agent-skill execution + validation +
evolution/bug-hunting** against agent-utilities itself — turning the "run the
fleet/skill validation matrix by hand" process the operator was running
manually (see `reports/delegation-validation-matrix.md`,
`reports/tier4-fleet-regression.md`) into something **agent-utilities
delegates to itself**, fully automated with local LLMs.

## Why a new skill, not an existing one

Three ecosystem-shaped skills already exist in `universal-skills` —
`ecosystem-validation-sweep`, `full-ecosystem-health`, `ecosystem-standardizer`
— and were checked before writing this one. `ecosystem-standardizer` is a real,
non-trivial workflow (structural drift/compliance auditing — file presence, env
var naming, CONCEPT registry). But `ecosystem-validation-sweep` and
`full-ecosystem-health` are **auto-generated template stubs**: every step body
is the generic placeholder sentence "Execute `<step>` operations for the
`<Workflow Name>` workflow", the `Expected:` values are placeholder tokens
(`install_artifacts`, `lint_artifacts`, `memory, cpu`), and neither one runs a
**delegated, grounded, per-target validation** (an `execute_agent` probe with a
grounding assertion) or scans for the fleet's **known bug-classes**. Neither
takes a `target`/`mode` pair or resolves target *type* (package vs skill).
Enhancing them would mean replacing their entire body — so this ships as a
new, purpose-built skill instead, and lives **here** (agent-utilities' own
`kg-*`-family skill directory, `agent_utilities/skills/`) rather than in
universal-skills, because it directly drives agent-utilities' own
`graph_orchestrate`/`graph_write`/`graph_feedback`/`graph_loops`/`graph_schedules`
surface and is how agent-utilities validates *itself* and the fleet it hosts
(CONCEPT:AU-OS.deployment.agent-factory-autoload — package-owned skills live with their owning package).

## Inputs

| Param | Values | Meaning |
|---|---|---|
| `target` | a path under `agent-packages/agents/*`, the literal string `agent-utilities`, or a universal-skill / agent-utilities skill name | what to validate |
| `mode` | `structural` \| `delegation` \| `bug_hunt` \| `full` | which phases to run (`full` = all four) |

## Step 1 — Resolve target type

Classify `target` before doing anything else:

1. **fleet-agent package** — a directory under `agent-packages/agents/<name>` (has `pyproject.toml`).
2. **agent-utilities itself** — the literal string `agent-utilities`.
3. **agent-utilities `kg-*` skill or universal-skill** — a directory with a `SKILL.md`, found under
   `agent_utilities/skills/` or `skills/universal-skills/universal_skills/**`.

`scripts/validate_target.py` (shipped with this skill) does this resolution
deterministically — see [Script usage](#script-usage) below — and every other
step below takes the resolved `(path, kind)` as input.

## Step 2 — `structural`

Deterministic, no LLM. Dispatches on `kind`:

- **package** → run its pytest, following the exact pattern
  `agent-packages/scripts/tier4_fleet_regression.sh` uses per package:
  `cd <pkg-dir> && timeout <N>s python -m pytest -q --no-header -p no:cacheprovider`,
  parsed for `passed`/`failed`/`errors`/`skipped`. No tests found → `no-tests`
  verdict (not a failure, matches tier4's convention).
- **skill** → frontmatter parses; `name` == directory name; every file the
  `SKILL.md` body references under `scripts/` or `references/` actually
  exists; no `gith__`/`go__`/`cm__`/`tm__` multiplexer-prefix tool name
  appears without a "dual-context" note explaining why (a skill that names a
  multiplexer-prefixed tool is usually describing cross-agent routing on
  purpose — the note is what distinguishes intentional from drifted).

Run: `python scripts/validate_target.py --target <X> --mode structural`.

## Step 3 — `delegation` (grounding)

For a package/agent target, delegate a **representative task** to it and
assert the result is genuinely grounded — this is the part that cannot be
scripted; it needs the local LLM and the fleet:

```
graph_orchestrate(action="execute_agent", agent_name="<pkg>",
                   task="<a representative read (and, where safe, create/search) task for <pkg>>")
```

**Concurrency cap: ≤2 in flight at once** — the GB10 inference host is
GPU-serialized; a wider fan-out starves every other in-flight run (see
`gb10-power-fault-and-vllm-topology` operational note). Batch targets in pairs.

Assert the result is **NOT** one of the three known failure patterns, and
record which:

| Failure pattern | What it looks like |
|---|---|
| Ungrounded degrade | `"could not produce a tool-grounded result"` or equivalent — the agent gave up on tool use |
| Self-refusal | `"I cannot access…"` / `"I don't have the ability to…"` when the tool plainly exists |
| Hallucinated generic data | Plausible-looking IDs/data with **no** matching `:ToolCall` provenance in the run's `RunTrace` |

A **PASS** verdict requires: real IDs/data in the response **and** matching
`:ToolCall` provenance (query the `RunTrace` — see
`delegation-tracing-hardening-program` for the provenance query pattern).
Record `run_id` + verdict either way — a documented failure is as valuable an
output as a pass (it is exactly what Step 5 turns into a `:ValidationFinding`).

## Step 4 — `bug_hunt`

Deterministic regex/AST scan for the **known fleet bug-classes** — every one
of these has actually been hit and root-caused in this ecosystem (see the
memory topics cited per row). `scripts/validate_target.py`'s `BUG_CLASSES`
table implements the detection heuristic for each:

| Bug-class | How it's detected | Where it's hit |
|---|---|---|
| Unbounded `object_set(of_type=...)` / unbounded graph scans (OOM) | grep `.py` for `object_set(of_type=` with no `limit=`/`page_size=`/`max_results=`/`top_k=` on the same line | `object_set.py` ontology surface |
| Fake-success stubs (`return "...executed successfully..."` doing nothing) | grep `.py` for a `return` statement containing "executed/completed/done successfully" | delegation-tracing-hardening-program (toolless runs stamped "completed") |
| `nl_query` event-loop-already-running fallback | grep `.py` for `get_event_loop()`/"already running" near `nl_query`/`asyncio` | skill-validation-regression-program Session-5 (`ask`-NL-planner no-LLM defect) |
| Multiplexer-prefix naming drift in skills | a `SKILL.md` referencing `gith__`/`go__`/`cm__`/`tm__` with no "dual-context" note | cross-agent-skill-standard program |
| Missing skill script/reference files | a `SKILL.md` body references `scripts/X` or `references/X` that doesn't exist on disk | skill-validation-regression-program |
| Stale swarm hostnames in env (`*_*:port`) | grep `.env`/`.env.example`/compose/`mcp_config.json` for a `service_name:port` URL that isn't `.arpa`/`localhost` | k8s-swarm-cutover (Swarm dissolved, docker-free; any `service_name:port` left over is stale) |
| Aggressive `timeoutSeconds: 1` k8s probes | grep `*.yaml`/`*.yml` for `timeoutSeconds: 1` | k8s-swarm-cutover follow-ups (probe tuning) |
| Unpinned CI Python (allows 3.14+ → native-build breaks) | grep `.github/workflows/*.yml` `python-version:` for anything outside a pinned 3.10–3.13 floor/ceiling | dep-security-hardening-fleet program |

Run: `python scripts/validate_target.py --target <X> --mode bug_hunt`.
Findings are emitted with `category`/`severity`/`file:line`/`detail` — treat
`critical` findings (fake-success stubs) as release-blockers, the rest as
triage candidates for Step 5.

## Step 5 — `evolution` (persist + hand off)

For every structural failure, non-PASS grounding verdict, or bug-hunt finding,
persist it to the KG as a typed node and close the loop:

```
graph_write(action="add_node", node_id="finding:<target>:<category>:<n>",
            node_type="ValidationFinding",
            properties='{"target": "<target>", "category": "<category>",
                         "severity": "<severity>", "evidence": "<file:line / run_id>",
                         "detected_at": "<passed-in ISO timestamp>"}')
graph_write(action="add_edge", source_id="finding:<target>:<category>:<n>",
            target_id="<target-node-id>", rel_type="FOUND_IN")
```

Then, for anything that warrants a code fix: either propose an SDD spec
(`graph_loops(action="submit", kind="develop", objective="fix <finding>", ...)`)
or hand off to the **`github-org-remediation-loop`** skill so the fix flows
through the spec-driven remediation loop rather than being hand-patched.

Record the outcome of the validation run itself so routing/scheduling learns
from it:

```
graph_feedback(correction_type="action_outcome", target_id="ecosystem-validation:<target>",
               corrected_value='{"success": <bool>, "mode": "<mode>", "findings_count": <n>}')
```

## Step 6 — Report

A structured summary: per-check pass/fail (Step 2), grounding verdict + run_id
(Step 3), findings list (Step 4), and what was persisted/handed off (Step 5).
`scripts/validate_target.py --json` emits the machine-readable half of this
(Steps 2 + 4); the delegating agent appends Steps 3 + 5 and renders the final
report.

## Delegation-ready invocation

**Primary invocation — the local LLM runs the whole thing:**

```
graph_orchestrate(action="execute_agent", agent_name="agent-utilities-expert",
                   task="run ecosystem-validation target=<X> mode=full")
```

This is deliberately the *documented* entrypoint, not a fallback: the goal is
that an operator (or the loop engine) never hand-runs these steps — they hand
the target + mode to the local model and it executes Steps 1–6 itself,
invoking `scripts/validate_target.py` for the deterministic half and
`graph_orchestrate`/`graph_write`/`graph_feedback` for the delegated half.

**Continuous automated validation — the "fully automated with local LLMs" goal:**

- **Scheduled sweep** — register a recurring `graph_schedules` entry that
  fans this skill out over every `agents/*` package + agent-utilities +
  every shipped skill on a cadence (e.g. nightly), so drift/regressions/bugs
  are caught before a human asks:
  `graph_schedules(action=..., name="ecosystem-validation-sweep", ...)` (see the
  `kg-schedules` skill for the full action set).
- **Loop-engine-driven** — submit it as a `develop`-kind Loop so the AHE
  evolution flywheel treats "keep the fleet green" as a standing objective,
  not a one-off task: `graph_loops(action="submit", kind="develop",
  objective="continuously validate the fleet with ecosystem-validation",
  validation_cmd="python scripts/validate_target.py --target <X> --mode full")`
  (see the `kg-loops` skill). Every run's findings feed Step 5's
  `:ValidationFinding` nodes, which is exactly the substrate the evolution
  flywheel (AHE) mines for what to fix next — this closes the loop from
  "validate" to "evolve" without a human in the middle.

## Composition-first — what's genuinely new here

This skill is composition, not reimplementation: it orchestrates
`graph_orchestrate`/`graph_write`/`graph_feedback`/`graph_loops`/
`graph_schedules` (all pre-existing) and the tier4 pytest pattern (pre-existing
script, referenced not copied). The **only new code** is
`scripts/validate_target.py` — a small, deterministic script that does target
resolution + structural checks + the bug-class regex scan, because those parts
are cheap and 100% mechanical; delegation and evolution are left to the LLM +
existing graph-os tools rather than scripted, because grounding assertions and
KG writes are exactly the kind of judgment call this skill exists to delegate.

## Script usage

```bash
python agent_utilities/skills/ecosystem-validation/scripts/validate_target.py \
    --target agents/gitlab-api --mode full

python agent_utilities/skills/ecosystem-validation/scripts/validate_target.py \
    --target agent-utilities --mode structural --timeout 300

python agent_utilities/skills/ecosystem-validation/scripts/validate_target.py \
    --target kg-write --mode bug_hunt --json
```

`--json` / `--output <path>` emit the machine-readable report (`checks`,
`findings`, `overall_pass`) for Step 5/6 to consume; without `--json` it prints
a markdown summary. Exit code is non-zero on any failing structural check or
any `critical`-severity bug-hunt finding.

## Execution

Run this workflow as a dependency-ordered DAG. Steps with no unmet
`depends_on` run in parallel; dependents run after their prerequisites
complete.

- **Run first:** Step 1 — resolve target type
- **After level 0 (in parallel):** Step 2 — structural; Step 3 — delegation (grounding, ≤2 concurrent); Step 4 — bug_hunt
- **After level 1:** Step 5 — evolution (persist + hand off)
- **After level 2:** Step 6 — report

**Execution:** If graph-os is reachable, offload the whole thing via
`graph_orchestrate(action="execute_agent", agent_name="agent-utilities-expert",
task="run ecosystem-validation target=<X> mode=full")` — see *Delegation-ready
invocation* above — for true self-hosted execution. Otherwise run
`scripts/validate_target.py` for Steps 1/2/4 natively and hand-run Steps 3/5
via the `graph_orchestrate`/`graph_write`/`graph_feedback` calls shown above.
