# Agent Utilities

**The in-run agent harness.** agent-utilities holds only the mechanism inside one
agent run. It applies the decisions it is given; it does not store, decide,
serve or connect on its own.

[![PyPI - Version](https://img.shields.io/pypi/v/agent-utilities)](https://pypi.org/project/agent-utilities/)
[![Build](https://github.com/Knuckles-Team/agent-utilities/actions/workflows/release.yml/badge.svg)](https://github.com/Knuckles-Team/agent-utilities/actions/workflows/release.yml)

*Version: 3.0.2*

## What it contains

| Module | What it does inside one run |
|---|---|
| `agent_utilities.graph` | Applies a received ordered plan or dependency DAG. It enforces task slots and shared budgets, and emits ordered execution events. |
| `agent_utilities.rlm` | Runs recursive model queries with shared request and tool limits and bounded concurrent model requests. |
| `agent_utilities.harness.edit_engine` | Parses model-proposed edits (search/replace blocks, unified diffs), applies them with graduated matching, and re-prompts on malformed or unmatched edits. |
| `agent_utilities.harness.contract_validator` | Checks a tool call's pre-conditions and its output's schema and verifier. |
| `agent_utilities.harness.constraint_engine` | Enforces hierarchical constraints on tool calls (prompt, tool description, middleware, implementation) and escalates repeated violations. |
| `agent_utilities.harness.quality_gates` | The pre-emit gate: antipattern detection, a preflight checklist and a multi-dimensional critique, in warn or block mode. |
| `agent_utilities.runtime` | The developer workspace: typed actions and observations, a stateful shell, local and container backends, a policy gate it is given, and the browser and computer-use tier ports. |
| `agent_utilities.host_ports` | The protocols the host fills: native and remote toolsets, tool clients, outbound credentials, delegated identity, the fleet catalog and the session keepalive. Without a host every binding port fails closed. |
| `agent_utilities.core.paths` | Resolves the platform config, data, cache and log directories (`ARBORA_CONFIG_DIR`, XDG). |

## Where everything else lives

- **[epistemic-graph](https://github.com/Knuckles-Team/epistemic-graph)**: facts, records,
  vocabulary, storage and queries, ingest, retrieval, reasoning, memory, decisions, and
  learning (evaluation, harness evolution, reward and trace records).
- **[arbora](https://github.com/Knuckles-Team/arbora)**: admission, auth, served
  operations (MCP, REST, A2A), messaging front-ends, the registry and the multiplexer.
  arbora hosts this harness and registers the `host_ports` providers.
- **[agent-connector-sdk](https://github.com/Knuckles-Team/agent-connector-sdk)**:
  connector contracts, transports, the A2A protocol library and connector helpers.

## Install and use

```bash
pip install agent-utilities
```

```python
from agent_utilities.runtime import create_workspace
from agent_utilities.runtime.events import CmdRunAction, TestRunAction

async with create_workspace(run_id="r1", prefer_docker=False) as ws:
    await ws.act(CmdRunAction(command="git status"))
    result = await ws.act(TestRunAction(selector="tests"))
```

## Received dependency plans

A host can pass `RunPlan(dependency_plan=DependencyPlan(...))` to the same
`run` or `iterate_run` entry point. Do not also set serial `steps`.
`DependencyPlan` contains ordered `RunTask(task_id, prompt, depends_on)` values,
an `output_task_id`, a `max_concurrency` task cap and an optional `plan_id`.
The host gets the plan from EG and admits it through Arbora. AU does not turn
free text into a plan or authorize a task.

A completed task frees a slot at once. A dependent task can start while an
unrelated task still runs. Every prerequisite must succeed. Each task gets a
fresh history with only its direct dependency outputs. The output task must
be the sole sink. Failed or stopped plans never return a partial child answer
as a completed parent result.

Request and tool reservations are shared across tasks and validation retries.
An unspecified parallel tool limit uses a finite fallback of fifty calls.
Token limits use reported usage, including fallback attempts. In-flight token
use is not known in advance. `RunBudget.max_concurrent_requests` also caps
model requests (fallback: four); the task cap can be lower. On cancellation,
AU drains child coroutines before invoking host-scope exit. Hosts own the
cancellation safety of their asynchronous scope exits; repeated cancellation
can interrupt that host cleanup. Host tools also own external work that cannot
stop with their coroutine. No elapsed cleanup bound is promised for handlers
that do not cooperate.

`RLMConfig.max_concurrent_requests` and `tool_calls_limit` bound recursive
model calls and tool fan-out (fallbacks: four and fifty). A parent releases its
model-request slot before waiting for a child, including at capacity one.
The parent model still chooses its next tool batch after the current batch
returns. That model-choice boundary is not a declared dependency DAG.

Public lifecycle events have a per-run `sequence` and stable `event_id`.
Task events carry `task_id`, `parent_run_id`, `plan_id` and `depends_on`.
They report execution state, not private model reasoning. Arbora owns their
safe replay projection and access checks.

A DAG driver owns the deadline after setup, using the same absolute deadline.
A paused event reader cannot delay task dispatch or timeout cleanup. Consumers
must finish or close the iterator. Pausing can delay the parent terminal event
and host-scope release, even after all task work has stopped. Request handlers
finish asynchronous cleanup before cancellation returns, including nested runs.

## Develop

```bash
uv sync --group dev
uv run ruff check . && uv run ruff format --check .
uv run mypy agent_utilities
uv run pytest -q
```

## License

MIT, see [LICENSE](LICENSE).
