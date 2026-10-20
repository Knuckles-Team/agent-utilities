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

## Run measurements

The graph run result and terminal event include `metrics` from a monotonic
client clock: `run_elapsed_ms` and ordered `model_requests` with
`client_request_elapsed_ms`, outcome, model/provider attribution and validated
per-response token counts. The sample list is bounded; `coverage: "partial"`
means some observed requests were omitted. Failed requests have no invented
token counts. These measurements are emitted through the host event sink for
shared backend consumption; AU stores no session totals.

Run time includes planning, tools and retries. Client request time includes
transport and provider waiting. Neither is provider generation time, and neither
is used to derive generation TPS. Provider queue wait, generation duration and
TTFT are absent unless a future adapter measures and identifies them explicitly.

## Develop

```bash
uv sync --group dev
uv run ruff check . && uv run ruff format --check .
uv run mypy agent_utilities
uv run pytest -q
```

## License

MIT, see [LICENSE](LICENSE).
