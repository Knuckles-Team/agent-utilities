# Agent Utilities engineering contract

## What this repository owns

agent-utilities is the **in-run agent harness** and nothing else. It owns the
mechanism inside one run that applies a decision it was given:

- edit application (`harness/edit_engine.py`);
- tool-call contracts and hierarchical constraints (`harness/contract_validator.py`,
  `harness/constraint_engine.py`);
- the pre-emit quality gate (`harness/quality_gates.py`);
- the developer-workspace action/observation loop and its backends (`runtime/`);
- the `host_ports` protocols that graph-os fills;
- the platform directory resolver (`core/paths.py`).

## What it does not own

Split every new requirement by the boundary rule before writing code here:

| Part of the requirement | Owner |
|---|---|
| A fact, record, vocabulary term, trace record, provenance mark; storage, ingest, retrieval, reasoning, memory | epistemic-graph |
| A choice: plan, routing, selection, ranking, thresholds, budgets, keep-or-revert verdicts; learning and evaluation | epistemic-graph |
| Admission, approvals, auth, served operations (MCP, REST, A2A), registry, multiplexer, messaging front-ends | graph-os |
| Connector contracts, transports, the A2A protocol library, connector helpers | agent-connector-sdk |

A harness module never decides, holds no store and serves no operation. Any
threshold it uses is a value it receives, with a fallback default only. When the
harness needs a capability another owner provides, it reaches it through a
`host_ports` protocol (graph-os supplies the implementation) or refuses clearly.

## Rules

- No new top-level packages. A new module needs a one-line reason that fits the
  harness row above.
- Never `git stash`, `git add -A` or `git add .`. Stage explicit paths.
- Every commit carries a `Spec:` trailer; specs live in `specs/` under the
  pipelines spec standard.
- Tests: one file per module, behaviour at the public surface only, parametrize
  near-duplicates, no sleeps or network, each test under 1 s, green under
  `pytest -n auto -p randomly --count=5`.

## Gates

`release.yml` runs the spec gates, `ruff check`, `ruff format --check`, `mypy
agent_utilities`, and the test suite, then builds the wheel. Publishing happens
only from an exact `vX.Y.Z` tag push.
