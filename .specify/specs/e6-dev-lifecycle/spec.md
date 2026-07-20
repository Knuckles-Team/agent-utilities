# Spec: EPIC 6 — Dev Lifecycle CLI + Ubiquitous Glossary (extends OS-5.1/5.2; OS-5.11 separate)

> Designs: `.specify/design/e6-dev-lifecycle-cli/design.md`, `.specify/design/os-5.11-run-scoped-tool-token/design.md`.
> Parallelizable across phases 1–3; OS-5.11 token lands with EPIC 2.

## Pre-Flight Checklist
- [x] Designs exist; CLI/glossary KG-nearest max 0.72 ≥0.70 → extend; OS-5.11 max 0.64 <0.70 → new.
- [x] Extension points: `pyproject.toml [project.scripts]`, existing `graph-os-daemon`/`graph-os`/`mcp-multiplexer`, `docs/CONTEXT.md`, CI.
- [x] Wire-First: ≤2 hops (CLI `run` → engine + OS-5.11 mint).

## User Stories
### US-1 — Unified lifecycle CLI
**As** an operator, **I want** one `agent-utilities` CLI with `start/stop/status/logs/inspect/run`, `--namespace`, `--json`, **so that** I control the whole system from one entry.
- **AC1**: `agent-utilities status --json` reports daemon/MCP/gateway states machine-readably.
- **AC2**: Two `--namespace` stacks isolate state under `.tmp/agent-utilities/<namespace>/` and don't collide (smoke test).
- **AC3**: `run <agent> <task>` dispatches through the orchestration entry and (AC6) mints an OS-5.11 token.

### US-2 — Ubiquitous-language glossary with CI check
**As** a maintainer, **I want** `docs/CONTEXT.md` (term + `Avoid:` + relationships) checked in CI against `concepts.yaml`, **so that** terminology doesn't drift.
- **AC4**: A CI guardrail fails when a defined term's `Avoid` synonym appears in new docs/code.
- **AC5**: Glossary terms reconcile with concept registry names (no orphan/contradiction).

### US-3 — Run-scoped tool token (OS-5.11)
- **AC6**: CLI `run` mints a token bound to runId/project/endpoints/expiry, injected into the run env; `tool_guard` enforces it (covered in EPIC 2 tests; surfaced here at the CLI).

## Non-Functional Requirements
- `@pytest.mark.concept(id="OS-5.11")` for the token; CLI smoke tests; ≤60s.
- Existing console-scripts remain functional (zero regression); CLI orchestrates them.
- Docs: `docs/pillars/5_agent_os_infrastructure/OS-5.11.md`, `docs/CONTEXT.md`; concepts.yaml regen.

## Tasks
- [ ] T1 `agent_utilities/cli/__init__.py`: argparse/typer app with subcommands + `--namespace`/`--json`. *(smoke)*
- [ ] T2 Lifecycle ops over `graph-os-daemon`/`graph-os`/`mcp-multiplexer`; namespaced `.tmp/`. *(smoke: isolation)*
- [ ] T3 `pyproject.toml`: `agent-utilities` console-script entry.
- [ ] T4 (OS-5.11) token mint in CLI `run`; integrate with EPIC 2 `tool_guard`. *(unit)*
- [ ] T5 `docs/CONTEXT.md` glossary + `scripts/check_context_glossary.py` CI guardrail + pre-commit hook. *(unit)*
- [ ] T6 Docs/concepts/CHANGELOG.
