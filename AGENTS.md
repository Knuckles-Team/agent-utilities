# Agent Utilities engineering contract

<!-- agent-contract:start -->
## Agent contract

Tool-neutral rules for any AI agent or human working in this repository. This block is maintained by the fleet standard; edit outside the markers.

- **Owner boundary:** this repository owns the in-run harness only: run loop, context, fan-out, sandbox, replay and trace events. It owns no MCP, server, API, A2A or store (spec prefix `AU-`). Feature routing across repositories: [ecosystem boundaries](https://github.com/Knuckles-Team/pipelines/blob/main/reference/ecosystem-boundaries.md).
- **Setup:** run `scripts/bootstrap.sh`. It is idempotent and non-interactive: it pins uv and Python, syncs the locked environment, installs the git hooks and prints the gate command.
- **Gates:** `uvx pre-commit run --all-files` must pass before you push. Do not bypass hooks to get green.
- **Commits:** one logical change per commit with a `Spec: <ID>` trailer (or `Spec: none (<reason>)`), plus a `Co-Authored-By:` trailer naming the assisting agent. Stage explicit paths only; never `git add -A` or `git add .`. Use your own configured Git identity, never a tool or bot name.
- **Org rules:** [organization AGENTS.md](https://github.com/Knuckles-Team/.github/blob/main/AGENTS.md).
<!-- agent-contract:end -->

## Boundaries

- **Owns:** the in-run harness. That covers the run loop, context assembly, fan-out, sandbox runs, replay and trace events. It enforces a limit it receives.
- **Never builds:** an MCP server, server, gateway, API or A2A (route them to arbora). A store, planner, selector, router, retriever or policy goes to epistemic-graph. Contracts go to agent-connector-sdk.
- **Allowed dependencies:** agent-connector-sdk and the model client library. It imports no MCP, web server or storage package.

Route every feature with the [ecosystem boundaries standard](https://github.com/Knuckles-Team/pipelines/blob/main/reference/ecosystem-boundaries.md) before any code. A part this repository does not own goes to its owner's specs.

## What this repository owns

agent-utilities is the **in-run agent harness** and nothing else. It owns the
mechanism inside one run that applies a decision it was given:

- edit application (`harness/edit_engine.py`);
- tool-call contracts and hierarchical constraints (`harness/contract_validator.py`,
  `harness/constraint_engine.py`);
- the pre-emit quality gate (`harness/quality_gates.py`);
- the developer-workspace action/observation loop and its backends (`runtime/`);
- the `host_ports` protocols that arbora fills;
- the platform directory resolver (`core/paths.py`).

A harness module never decides, holds no store and serves no operation. Any
threshold it uses is a value it receives, with a fallback default only. When the
harness needs a capability another owner provides, it reaches it through a
`host_ports` protocol (arbora supplies the implementation) or refuses clearly.

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

Install the local hooks with `pre-commit install --config .config/pre-commit.yaml
--hook-type pre-commit --hook-type pre-push`. A hook script left by an older checkout
that calls `scripts/hook_python.sh` is stale; reinstall with that command.
