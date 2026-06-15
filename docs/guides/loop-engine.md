# Loop Engine — running the self-improvement / research / goal loop

> **What this is.** The "golden loop" was renamed to the **Loop engine**
> (`LoopController`). It is the one engine that advances every long-running
> objective — a **Loop** of kind `research`, `develop`, or `skill` — through a
> single hot path. This guide is the runbook: how to trigger one cycle, drive a
> single objective to completion, run it autonomously, and the rename/migration
> notes. Architecture lives in
> [OWL/RDF Layer](../architecture/owl_rdf_layer.md) (the "Reasoning as the
> research engine" section).

## The model in one paragraph

A **Loop** is one long-running objective, stored as a develop/skill/research
`Concept` node (CONCEPT:KG-2.78). `submit_loop` is the single creation path for
goals, research topics, failure gaps, and skill executions. The **`LoopController`**
advances active Loops: research loops acquire sources → reason (OWL/RDF over the
whole ecosystem) → assimilate → distill → synthesize (propose-only); `develop`
loops run act→validate (`validation_cmd`, done on exit 0); `skill` loops execute
their skill / skill-workflow. Goal state and the durable iteration record live on
the Loop node itself — there is no separate `goals` table and no separate
goal-runner. There is **one entrypoint**: the `graph_loops` MCP tool (and its REST
twin).

## Trigger it — pick the surface

| You want to… | Surface | Call |
|---|---|---|
| Advance all active Loops one cycle (the classic "golden loop" run) | **MCP** | `graph_loops(action="run", max_topics=5)` |
| Same, inside the orchestrate tool | **MCP** | `graph_orchestrate(action="loop_cycle", max_fan_out=5)` |
| Drive ONE objective to completion, durably (resume / checkpoint / corrigible) | **MCP** | `graph_loops(action="drive", loop_id="loop:research:…")` |
| Create a Loop (goal / research topic / skill run) | **MCP** | `graph_loops(action="submit", objective="…", kind="develop", validation_cmd="pytest -q")` |
| Inspect / cancel | **MCP** | `graph_loops(action="list")` · `graph_loops(action="cancel", loop_id="…")` |
| One cycle over HTTP | **REST** | `POST {prefix}/graph/loops` `{"action":"run","max_topics":5}` |
| Run autonomously on a throttle | **daemon** | set `KG_LOOP=1` (see below) |
| From Python | **library** | `LoopController(engine).run_one_cycle(max_topics=5)` |

`{prefix}` defaults to `/api` (e.g. `POST /api/graph/loops`). The REST routes are
thin twins of the MCP tools dispatching the same in-process core — they never
drift.

### 1. MCP — `graph_loops` (the one entrypoint)

```jsonc
// advance every active Loop one cycle — research/develop/skill
graph_loops(action="run", max_topics=5)

// submit a develop goal, then drive it to completion durably
graph_loops(action="submit", objective="make CI green",
            kind="develop", validation_cmd="pytest -q")
graph_loops(action="drive", loop_id="loop:develop:make-ci-green")
```

`action`:

- **`run`** — advance all active Loops one cycle (`LoopController.run_one_cycle`).
- **`drive`** — run ONE Loop (by `loop_id`) to completion, durably: it resumes from
  the last checkpoint, runs each iteration under an idempotency key (at-least-once
  retries, exactly-once effect), and honors a fleet pause/kill signal (checkpoint &
  yield). Works for any kind.
- **`submit`** — create a Loop. `kind=research` (default) needs only `objective`;
  `kind=develop` takes `validation_cmd` / `end_state`; `kind=skill` takes
  `skill_ref`.
- **`list`** — the active Loops (intake view; in-flight `running` loops are
  excluded so a goal is never double-driven).
- **`cancel`** — terminate a Loop by `loop_id`.

### 2. MCP — `graph_orchestrate(action="loop_cycle")`

```jsonc
graph_orchestrate(action="loop_cycle", max_fan_out=5)
```

Same single-cycle run, exposed inside the orchestrate tool. This is the renamed
`golden_loop` action (the old name no longer exists — see migration below).

### 3. REST gateway

```bash
# one cycle
curl -sX POST "$GATEWAY/api/graph/loops" \
  -H 'content-type: application/json' \
  -d '{"action":"run","max_topics":5}'

# the ARA research surface (reason / compile / review) is its own router
curl -sX POST "$GATEWAY/api/research/reason" \
  -H 'content-type: application/json' -d '{"query":"long-context retrieval"}'
```

### 4. Autonomous daemon tick

The engine runs the cycle on a throttle when enabled — opt-in because it does
autonomous LLM work:

| Variable | Default | Meaning |
|---|---|---|
| `KG_LOOP` | `false` | enable the periodic Loop tick (`_tick_loop`) |
| `KG_LOOP_INTERVAL` | `3600` | seconds between ticks |
| `KG_LOOP_TOPICS` | `5` | max Loops advanced per tick |

```bash
KG_LOOP=1 KG_LOOP_INTERVAL=900 KG_LOOP_TOPICS=8 python -m agent_utilities
```

(Set these via `config.json` / `AGENT_UTILITIES_CONFIG_DIR` like any other config.)

### 5. Library

```python
from agent_utilities.knowledge_graph.research.loop_controller import LoopController

report = LoopController(engine).run_one_cycle(max_topics=5)
# report carries per-stage timings + an "executed" block for develop/skill loops
```

`run_one_cycle` stage toggles (each is best-effort; one failing stage never aborts
the cycle): `assimilate`, `reason`, `breadth`, `standardize`, `distill`,
`synthesize`, `discover`, `synthesize_search`. The four still-`GOLDEN`-prefixed
stage flags gate the heavier stages:

| Variable | Default | Stage |
|---|---|---|
| `KG_GOLDEN_BREADTH` | `true` | ingest the OSS/repos/docs corpus (idempotent) |
| `KG_GOLDEN_DISCOVER` | `false` | discover + ingest new papers (external calls) |
| `KG_GOLDEN_DISTILL` | `false` | write SpecDraft markdown under `.specify/` |
| `KG_GOLDEN_STANDARDIZE` | `false` | enterprise standardization pass |

## Migration — what changed from the "golden loop"

| Before | Now |
|---|---|
| `GoldenLoopController` | `LoopController` |
| `research/golden_loop.py` | `research/loop_controller.py` |
| `graph_orchestrate(action="golden_loop")` | `graph_orchestrate(action="loop_cycle")` — **no alias** |
| `KG_GOLDEN_LOOP` / `_INTERVAL` / `_TOPICS` | `KG_LOOP` / `KG_LOOP_INTERVAL` / `KG_LOOP_TOPICS` |
| separate goal-runner (`run_goal_loop`) + `goals` SQLite table | folded into `LoopController.run_loop`; goal state lives on the **KG Loop node** |
| `run_golden_loop_cycle()` facade | removed — call `LoopController(...).run_one_cycle()` |

> The `KG_GOLDEN_BREADTH/DISCOVER/DISTILL/STANDARDIZE` per-stage flags still carry
> the `GOLDEN` prefix; the loop-enable flag and controller are the renamed parts.

## Loop kinds at a glance

- **research** — acquire knowledge for a topic; *done* when it has `ADDRESSED_BY`
  sources. Reasoning over the one ecosystem ontology extrapolates cross-domain
  links and harvests them back as fresh topics.
- **develop** — iterate act→validate until `validation_cmd` exits 0 (or `end_state`
  holds). Autonomous goals are develop Loops.
- **skill** — run a skill / skill-workflow (`skill_ref`) to its completion state.

All three are advanced by the one `LoopController`, share the durable
checkpoint/resume machinery, and are reachable through the one `graph_loops`
entrypoint.
