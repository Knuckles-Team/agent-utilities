# Tiered RLM Code Sandbox + Capability Router (CONCEPT:AU-ORCH.sandbox.tiered-rlm-sandbox)

## Overview

The RLM REPL (ORCH-1.1) executes LLM-generated Python glue code to decompose long contexts.
Historically that ran through a hardcoded `if use_wasm / elif use_container / else local` switch
in `rlm/repl.py`, where the **WASM path was a stub** (it never loaded a `python.wasm` — it emulated
three hardcoded tasks), the **Docker path could not serve the RLM host helpers** (it passed only a
JSON `context` string and dropped every other var), and the only working path was **unsandboxed
`exec()`** (CWE-94). So the real choice was "host helpers *or* isolation", never both.

ORCH-1.38 replaces that with a uniform **`Sandbox` contract** and four *real* backends behind a
deterministic **capability router** that picks the cheapest backend a snippet can run on and
**escalates on rejection**. The keystone is **monty** (Pydantic's pure-Rust Python interpreter):
it is the only *isolating* backend that can still serve the host helpers, because its async
external functions suspend the VM and let the host fulfil `await rlm_query(...)` — at ~0.5ms vs a
container's hundreds of ms. Extends **ORCH-1.1**, reuses the resilience semantics of **ORCH-1.29**.

## How it works

`RLMEnvironment.execute(code)` builds a `SandboxEnv` (the REPL `vars`, `tool_sources`, the
host-callback `helpers`, and local-only globals), asks the router for an **escalation chain**, then
runs each backend in order:

- **`SandboxRejected`** — this backend can't run this snippet (a capability gap or a parse-time
  rejection, e.g. monty meeting a `class`). The router advances to the next tier. Parse-level
  rejections fire *before* any host helper runs, so escalation has **no side effects**.
- **`SandboxFatalError`** — irreversible infra death (dead container, lost mount). The run
  fast-fails and does **not** escalate, preserving ORCH-1.29 semantics.
- **success** — the namespace is synced back and `(vars, stdout)` returned.

### Routing (deterministic, in-process)

`AstAnalyzer` does one `ast.walk` over the snippet (~50–200µs) to extract: third-party imports
(vs `sys.stdlib_module_names`), class/`@dataclass` definitions, `async` usage, and which RLM
helpers it calls. `SandboxRouter` turns that into required capabilities and returns the
capable-and-available backends ordered by `preference_rank` (monty → wasm → docker → local), with
the `local` floor always anchoring the tail so the chain is never empty.

> The hot path uses stdlib `ast`, **not** the epistemic-graph `ParseFile` kernel: that is a UDS
> round-trip (~0.2ms+), is not guaranteed running, and its own guidance is "batch, never
> per-element" — all wrong for per-snippet routing. The `Analyzer` Protocol leaves room for an
> engine-backed *batch* analyzer in a future offline pass.

### The backends (capability matrix)

| Backend | Startup | Host helpers | 3rd-party libs | classes / full stdlib | Isolation | Needs |
|---|---|---|---|---|---|---|
| **monty** | ~0.5ms | ✅ native `external_functions` | ❌ | ❌ / ~8 modules | language-level | `pydantic-monty` (core dep) |
| **wasm** | ~ms | ❌ (v1) | ❌ | ✅ / ✅ (CPython) | WASI | `wasmtime` + `python.wasm` |
| **docker** | 100–500ms | ✅ via UDS bridge | ✅ | ✅ / ✅ | OS-level | docker/podman daemon |
| **local** | ~0 | ✅ native | ✅ | ✅ / ✅ | **none** (floor) | — |

- **monty** — the fast, isolated **default**. Fresh `Monty` per call (matches local semantics).
  Rejects classes / `@dataclass` / third-party imports at *construction* time → escalate with zero
  side effects. Enforces memory/time/recursion limits; no daemon, no root.
- **docker/podman** — the heavy escalation tier, hardened: `--network none`, `--memory` /
  `--cpus` / `--pids-limit` caps, `--cap-drop ALL`, `--security-opt no-new-privileges`, a
  wall-clock timeout, and the full JSON-able namespace shipped in. A per-run **UDS host-callback
  bridge** (a socket inside the bind-mounted run dir) lets the `--network none` container still
  call `rlm_query` etc. over the filesystem socket; async helpers are awaited host-side and
  `FINAL_VAR` round-trips to host `vars`.
- **wasm** — real **CPython-on-WASI** under wasmtime (replaces the stub). Isolated full-stdlib
  compute; the only fs access is one preopened scratch dir, no network/env/subprocess. Wall-clock
  timeout via epoch interruption. v1 has `host_callbacks=False`, so the router sends only
  self-contained compute here; a WASI run has no side effects, so any failure is a **safe**
  `SandboxRejected` (escalate to docker).
- **local** — the legacy restricted `exec()`, kept as the always-available floor.

### Selection

`RLMConfig.sandbox` (`auto` | `local` | `monty` | `wasm` | `docker`, env `RLM_SANDBOX`) chooses the
strategy; `auto` engages the router, any concrete value pins one backend. The legacy
`use_monty` / `use_wasm` / `use_container` booleans are honored as overrides via
`resolved_sandbox()`. The default is `auto`, so out-of-the-box RLM gets monty's fast isolated path
*with* helpers instead of raw `exec()`.

```python
from agent_utilities.rlm.config import RLMConfig
from agent_utilities.rlm.repl import RLMEnvironment

env = RLMEnvironment(context=long_text, config=RLMConfig())  # sandbox="auto"
vars_, stdout = await env.execute("parts = [await rlm_query('s', context[:500])]\nFINAL_VAR('a', parts)")
# helper-driven glue → monty;  numpy/classes → escalate to docker;  pure compute → wasm
```

## Key files / API

| Piece | Location |
|---|---|
| Contract + capabilities | `rlm/sandboxes/base.py` (`Sandbox`, `SandboxCapabilities`, `SandboxEnv`, `SandboxResult`, `SandboxRejected`, `HELPER_NAMES`) |
| Analyzer | `rlm/sandboxes/analyzer.py` (`AstAnalyzer`, `CodeRequirements`, `Analyzer` Protocol) |
| Router | `rlm/sandboxes/router.py` (`SandboxRouter.select`) |
| Backends | `rlm/sandboxes/{local,monty,docker,wasm}_backend.py` |
| Registry | `rlm/sandboxes/registry.py` (`default_sandboxes`) |
| Wiring | `rlm/repl.py` (`execute`, `_build_sandbox_env`, `_get_sandbox_router`) |
| Config | `rlm/config.py` (`sandbox`, `use_monty`, `resolved_sandbox`) |
| Telemetry | `rlm/telemetry.py` (`sandbox_escalated` failure class) |
| WASM payload provisioning | `scripts/provision_rlm_wasm.py` |

## Provisioning the WASM payload

The wasm tier stays unavailable (and the router silently skips it) until a `python.wasm` is
present. It is ~25MB and kept out of the repo:

```bash
python scripts/provision_rlm_wasm.py          # download + sha256-verify into the platform cache
# or point at any python.wasm:
export RLM_WASM_PYTHON=/path/to/python.wasm
```

`wasmtime` itself is an optional dependency: `pip install 'agent-utilities[sandbox]'`.

## Wiring (≤3 hops)
`graph_orchestrate(rlm_run)` → `repl.execute` → `router` → `*_backend` (≤3 hops).

## Error contract (relation to ORCH-1.29)
`SandboxRejected` is *new* and benign (routing escalation, classified as `sandbox_escalated`);
`SandboxFatalError` retains its ORCH-1.29 meaning (irreversible sandbox death → fast-fail). The
host-tool wall-clock budget (`with_tool_timeout`) still governs the helper callbacks.
