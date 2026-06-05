# mypy Debt Remediation Plan

**Status:** Phase 0 landed (scoped baseline). Phases 1–3 tracked, not yet executed.
**Owner:** typing-cleanup track (separate from feature work).
**Baseline command:** `python -m mypy --ignore-missing-imports agent_utilities/`

## Why this exists

A full-repo mypy run reported **723 errors in 53 files**. None originate from the recent
RLM-GEPA / memory-first / enrichment work — every error pre-existed. Rather than fix 723 errors
across framework code in one risky sweep (or silently ignore them), this plan characterizes the
debt, removes the noise, and sequences the real fixes so the suite becomes **actionable** and
**new code stays strictly checked**.

## Characterization (what the 723 actually were)

| Error code | Count | Nature |
|---|---:|---|
| `attr-defined` | 596 | **Framework false positives** — see below |
| `type-arg` | 29 | bare generics (`dict`, `list`, `Callable` without params) |
| `arg-type` | 27 | genuine-ish signature mismatches |
| `assignment` | 16 | incompatible assignments |
| `override` | 13 | LSP signature drift in subclasses |
| `call-overload` | 10 | overload resolution (e.g. pydantic `create_model`) |
| others | ~32 | `list-item`, `call-arg`, `no-redef`, `abstract`, `var-annotated`, … |

**92% of all errors (665) lived in just 5 files:**

| File | Errors (before) |
|---|---:|
| `graph/_router_impl.py` | 231 |
| `graph/executor.py` | 148 |
| `graph/verification.py` | 112 |
| `core/checkpoint/manager.py` | 89 |
| `graph/hierarchical_planner.py` | 85 |

The 596 `attr-defined` errors are almost entirely **pydantic-graph generic state/deps access**.
mypy cannot resolve concrete fields through the framework's generic node/context types:

```
"NodeSnapshot[StateT, Any]" has no attribute "run_id"
ctx.state.query        # GraphRunContext[StateT, DepsT].state is StateT (unbound)
node.plan              # BaseNode[StateT, Any, Any] has no concrete fields
```

Top "missing" attributes — all real fields on the concrete State/Deps classes the nodes are
*actually* parameterized with at runtime: `event_queue` (66), `query` (63), `knowledge_engine`
(47), `plan` (41), `step_cursor` (27), `error` (26), `verification_attempts` (20),
`results_registry` (20), `validation_feedback` (19). **These are not bugs** — the code runs
correctly; mypy simply can't follow the generic binding.

## Phase 0 — Scoped baseline (LANDED)

Added a **narrowly scoped** `[[tool.mypy.overrides]]` block to `pyproject.toml` that disables
**only `attr-defined`** in **only those 5 framework files**:

```toml
[[tool.mypy.overrides]]
module = [
  "agent_utilities.graph._router_impl",
  "agent_utilities.graph.executor",
  "agent_utilities.graph.verification",
  "agent_utilities.graph.hierarchical_planner",
  "agent_utilities.core.checkpoint.manager",
]
disable_error_code = ["attr-defined"]
```

**Result: 723 → 170 errors (−76%).** Crucially this masks nothing real:
- `attr-defined` remains **active in every other module** (43 still flagged, all actionable).
- All other error codes remain active **even in the 5 listed files** (e.g. `manager.py` still
  reports its `type-arg`/`arg-type`/`override` issues).
- New code is held to the full standard; the override is a shrinking allow-list, not a global relax.

This makes the residual 170 a usable signal instead of being buried under 596 framework
false-positives. Each entry is removed as its file is properly annotated in Phase 2.

## Phase 1 — Drain the long tail (low risk, high signal)

**28 files have ≤2 errors each.** These are quick, isolated, mostly mechanical:
- `type-arg` (29): parameterize bare generics — `dict` → `dict[str, Any]`, `Callable` →
  `Callable[..., Any]`, `list` → `list[X]`.
- `var-annotated` (2): add the annotation mypy literally suggests (`_cached_modules: dict[...] = {}`).
- `no-redef` (5): usually a conditional-import shim — guard with `if TYPE_CHECKING:` or rename.

Do these file-by-file, one commit per cluster, `pre-commit` green each time. Target: 170 → ~110.
No framework risk — these are leaf modules.

## Phase 2 — Properly type the framework generics (removes the Phase 0 override)

The real fix for the suppressed 596. For each of the 5 graph/checkpoint files, give the
pydantic-graph nodes **typed State/Deps** so mypy resolves field access:

1. Define `class GraphState` / `class GraphDeps` (or `Protocol`s) with the concrete fields
   (`query`, `event_queue`, `knowledge_engine`, `plan`, `step_cursor`, …).
2. Parameterize nodes/context explicitly: `BaseNode[GraphState, GraphDeps, ResultT]`,
   `GraphRunContext[GraphState, GraphDeps]`.
3. For pydantic-graph persistence types (`NodeSnapshot`/`EndSnapshot`/`run_id`/`timestamp`),
   either upgrade pydantic-graph if a typed release exists, or add a thin typed wrapper /
   `cast()` at the 3–4 access sites rather than blanket-ignoring.
4. **Remove that file's entry from the override block** and confirm 0 `attr-defined`.

Sequence by leverage: `executor.py` and `_router_impl.py` share the same State/Deps, so typing
them together likely clears the bulk. Do **one file per PR** so a framework typing regression is
easy to bisect. Risk: medium (touches hot-path framework code) — gate on the full unit suite +
the graph integration tests, not just mypy.

## Phase 3 — Targeted real-bug fixes

A handful are worth fixing on their own merits regardless of the generics work:
- `core/model_factory.py` (11): the pydantic-ai import shim — add a typed fallback / `Protocol`
  for the optional `pydantic_ai` surface instead of `# type: ignore`.
- `knowledge_graph/backends/contrib/neo4j_backend.py`: `Cannot assign to a type` / `None` vs
  `type[GraphDatabase]` — the optional-dep `try/except ImportError: GraphDatabase = None` pattern;
  annotate `GraphDatabase: type[Any] | None`.
- `harness/distributed_state_manager.py`: `loads()` on `Awaitable[Any] | Any` — a missing `await`
  or a `cast`; verify it's not a latent runtime bug.
- `override` (13) in `checkpoint/manager.py`: align subclass signatures with the base (real LSP
  drift — worth fixing for correctness).

## Gating policy (so debt doesn't regrow)

- mypy is **not** in the blocking CI gate today and this plan does **not** add it globally yet —
  flipping it on at 170 errors would just wedge CI.
- Recommended ratchet: once Phase 1 lands (~110), add mypy as a **non-blocking** informational CI
  step that fails only on a count **increase** vs a committed baseline number. After Phase 2,
  promote to blocking for the cleaned modules.
- Keep `disallow_untyped_defs = false` for now; revisit per-module after Phase 2.

## Scope note

Phase 0 is done. Phases 1–3 touch code outside the memory/RLM feature work (mostly the graph
framework owned by others), so they are intentionally **tracked, not bundled** into the
pre-live-testing checkpoint — to keep that checkpoint reviewable and low-risk.
