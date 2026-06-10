# Contributing to agent-utilities

Thanks for contributing! This is the Python harness (the 5-pillar platform); the
high-performance graph compute lives in the separate Rust engine
[`epistemic-graph`](https://github.com/Knuckles-Team/epistemic-graph), reached
out-of-process over MessagePack/UDS (no PyO3).

## Development setup

```bash
pip install -e ".[all]"
pre-commit install
```

The default knowledge-graph backend is zero-infra (in-process), so most work needs
no external services. For the durable tier set `GRAPH_DB_URI` (Postgres/pggraph).

## Branch / worktree workflow

Multiple agents and people work this repo concurrently. **Do not edit the
canonical checkout** at `/home/apps/workspace/agent-packages/agent-utilities` — a
background sync can reset its working tree. Take your own git worktree on your own
branch (one branch per worktree keeps concurrent sessions from colliding):

```bash
rm_worktree add agent-utilities <your-branch>     # repository-manager MCP, or:
git worktree add /home/apps/worktrees/agent-utilities/<branch> -b <branch> main
```

Commit early and often (commits survive a working-tree reset); merge to `main`
locally when done. Push only when asked.

## Before you push

```bash
python -m pytest          # unit suite (keep it green)
pre-commit run --all-files
```

Note: the `pre-commit` pytest hook can fail repo-wide due to an unrelated
egeria/py3.12 dependency pin — validate with the system `python -m pytest` if so.

## Conventions

- **No stubs.** `raise NotImplementedError` only with `# ABSTRACT-OK`.
- **Strangler-then-delete** — never "v2 beside old".
- **Name from purpose, not process** — no `wave0`/`phase2`/`v2` in identifiers;
  provenance goes in the docstring/CHANGELOG.
- **Wire-First** — a feature isn't done until a live path invokes it; ship
  primitives with a real consumer and a live-path test.
- New `CONCEPT:` ids go in `docs/concepts.yaml` (run `scripts/check_concepts.py`).

See [AGENTS.md](AGENTS.md) for the full architecture reference and guardrails.
