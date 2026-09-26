# One-link self-deploy

agent-utilities is **self-deploying**: point any AI agent (Claude Code, Cursor,
Codex, Windsurf, …) at the repository and it knows how to set itself up — or run a
single command yourself. There is one path for two very different operators:

- **Homelabbers / self-hosters** — a zero-infra, all-local stack on a laptop or Pi.
- **Enterprises** — a multi-host Docker Swarm with *everything wired in* (Vault, SSO,
  DNS, ingress, observability, and all 50+ MCP connectors).

## One command

=== "macOS / Linux"

    ```bash
    curl -fsSL https://knuckles-team.github.io/agent-utilities/install.sh | sh
    ```

=== "Windows (PowerShell)"

    ```powershell
    irm https://knuckles-team.github.io/agent-utilities/install.ps1 | iex
    ```

Pick a profile (defaults to `tiny`):

```bash
curl -fsSL https://knuckles-team.github.io/agent-utilities/install.sh | sh -s -- --profile enterprise
```

```powershell
& ([scriptblock]::Create((irm https://knuckles-team.github.io/agent-utilities/install.ps1))) -DeployProfile enterprise
```

!!! note "If the pretty URL is unavailable"
    The scripts are also fetchable raw from GitHub:
    `https://raw.githubusercontent.com/Knuckles-Team/agent-utilities/main/scripts/install.sh`
    (and `install.ps1`).

## What the installer does

1. Checks your host (Python 3.11–3.14; **no Rust needed** — the engine ships as a
   prebuilt wheel; Docker only above `tiny`).
2. Installs agent-utilities (+ extras for your profile) and the skill toolkit.
3. Runs the **host preflight** for your profile and any UI components you choose.
4. Installs the skills into **every AI tool on your host** and wires the `graph-os`
   MCP server into each — so your agent immediately has the knowledge graph and the
   genesis skills.
5. Hands off to a guided deployment that finishes the wiring.

## Profiles

| You are… | Profile | What you get |
|---|---|---|
| A homelab / self-hoster | `tiny` | Zero-infra, all-local. No databases, no Docker. |
| One durable server | `single-node-prod` | Postgres/pg-age + the core MCP connector fleet. |
| An enterprise | `enterprise` | Multi-host Swarm — Vault, SSO, DNS, ingress, observability, all 50+ connectors. |

Optional UIs (add `--component`): `agent-webui`, `agent-terminal-ui`, `geniusbot`.

## For AI agents

The full deployment procedure lives in
[`AGENTS.md` → Zero-to-deployed](https://github.com/Knuckles-Team/agent-utilities/blob/main/AGENTS.md#-zero-to-deployed-genesis--deploying-this-for-an-operator),
and the machine-readable manifest you loop over is
[`genesis.yaml`](https://knuckles-team.github.io/agent-utilities/genesis.yaml)
(profiles · preflight · MCP fleet · UI components · IDE targets). Preflight a host
without installing anything via the MCP tool
`graph_configure action=preflight config_key=<profile>`.

## Verify

```bash
agent-utilities-doctor                       # full deployment health sweep
agent-utilities-doctor --preflight --profile enterprise   # host readiness only
```
