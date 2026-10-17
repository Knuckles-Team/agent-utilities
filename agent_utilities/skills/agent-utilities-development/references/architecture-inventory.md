## Orient first — the component inventory (READ BEFORE ADDING ANYTHING)

Most sprawl here is not written deliberately; it is written by an author who did
not know an owner already existed. `agent_utilities/` is **1,712 tracked `.py`
modules** (`git ls-files agent_utilities | grep -c '\.py$'`) across **47
top-level importable packages** — first-level directories carrying their own
`__init__.py`:

```bash
git ls-files agent_utilities | awk -F/ 'NF==3 && $3=="__init__.py"{print $2}' | sort -u | wc -l   # 47
```

★ **State which definition you used** — three neighbouring ones give three
answers (measured 2026-09-03): **47** first-level dirs with their own
`__init__.py`, **48** first-level dirs containing a tracked `.py` anywhere below
(`agent_chat/` has no top-level `__init__.py`), **51** first-level tracked dirs at all
(`.agent_data/`, `data/`, `images/` hold no Python).

Nobody reads 1,712 modules. So read the *layer* table, then the *size* table
(both in the reference below), then `scripts/find_callers.py` — in that order —
before you create a module.

### Layers and sizes — where an owner already lives

[`references/architecture-reference.md`](references/architecture-reference.md) carries the
six target layers (`contracts -> domain -> ports -> application -> adapters ->
composition`) with the existing packages assigned to each and what each may NOT
import, the measured 62 application→adapter / 85 adapter→adapter violations, the
note that the API gateway and the graph-os MCP surface both live INSIDE this repo
and share one `_execute_tool()` core, the per-package file-count table, and the
heaviest single modules. Read it before you create a module.

## Dependency direction and the known SCC

The measured figures — 1,701 modules / 6,066 raw internal edges, the **847-module** raw
all-import SCC, the eager graph's **zero** non-trivial SCCs, the 59-node package projection
with its one 44-package SCC, and `check_import_cycles.py`'s own differently scoped
**821-module / 48.5%** measurement (the standing reason the eager/deferred/`TYPE_CHECKING`
split must not be "simplified" away) — are under "Dependency direction — the measured SCC"
in `references/architecture-reference.md`.

**The rules — TARGET direction you are held to in review, NOT a property the tree
has or a gate enforces.** ★ Nothing enforces the layering; rules 1 and 2 are
measurably violated on `main` today. All five rules, what refuses what, and the
measured violation counts are under "The rules — target direction" in
`references/architecture-reference.md`, and the named reverse edges the program is
cutting — with their measured edge counts and the directions that must stay one-way —
under "Named reverse edges" in the same file. Do not add to them; re-measure before
quoting a number.

`KnowledgeGraph` (`knowledge_graph/facade.py`) is the graph facade;
`Orchestrator.execute_agent` (`orchestration/manager.py:512`) owns agent/loop/workflow
dispatch. ★ `run_agent` is **not** a method on `Orchestrator` — it is a **module-level
function**, `orchestration/agent_runner.py:700`, which `Orchestrator.execute_agent`
imports and calls (`AGENTS.md` writes them together; that shorthand is not a class API).
Facade and orchestrator have **disjoint** authority — the facade does not dispatch
agents, the orchestrator does not implement graph persistence. Composition binds
exactly one of each.
