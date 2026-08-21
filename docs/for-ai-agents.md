# For AI Agents

If you are an AI agent (Claude Code, Cursor, Codex, or any other tool) that has been
pointed at this repository — to use it, deploy it, or work on it — the canonical
instructions live outside this MkDocs site, in the repository root, so they load
automatically into agent context:

- **[`AGENTS.md`](https://github.com/Knuckles-Team/agent-utilities/blob/main/AGENTS.md)**
  — the single source of truth for working in this repo: zero-to-deployed genesis
  procedure, working discipline, architecture reference, and every guardrail gate.
  Claude Code loads it automatically via `CLAUDE.md`'s `@AGENTS.md` import; other
  tools should read it directly.
- **[`llms.txt`](https://github.com/Knuckles-Team/agent-utilities/blob/main/llms.txt)**
  — the machine-readable entry index: a flat, prioritized list of the docs an LLM
  should read first, without the human navigation chrome.
- **[`genesis.yaml`](https://github.com/Knuckles-Team/agent-utilities/blob/main/genesis.yaml)**
  — the machine-readable deployment manifest `AGENTS.md`'s zero-to-deployed
  procedure loops over (profiles, run plan, per-connector deploy strategy).

## The short version

- **Deploying this for an operator?** `AGENTS.md`'s
  [Zero-to-deployed (genesis)](https://github.com/Knuckles-Team/agent-utilities/blob/main/AGENTS.md#-zero-to-deployed-genesis--deploying-this-for-an-operator)
  section is the router — it asks one question (homelab or enterprise?) and takes
  it from there via the `agent-utilities-deployment` skill.
- **Integrating an existing agent with the knowledge graph?** Start at
  [Consumption Models](guides/consumption-models.md) for the trade-offs between the
  Python library, the MCP server, and the shared HTTP gateway.
- **Working on this codebase itself?** `AGENTS.md`'s working-discipline sections
  (query the KG before grepping, Wire-First, fail-closed, no-legacy, branching &
  isolation) are mandatory reading before your first edit.
- **Just need the facts fast?** [Start Here](start-here.md) is the same
  five-minute orientation a human gets, and [`docs/status.md`](status.md) is the
  generated concept/capability registry — the honest, drift-free source for "what
  actually works right now."

This page exists only as a site-nav pointer — the content it points to
(`AGENTS.md`, `llms.txt`, `genesis.yaml`) is intentionally kept at the repository
root, outside `docs/`, so agent tooling that auto-loads root-level convention files
picks it up without needing to know about MkDocs at all.
