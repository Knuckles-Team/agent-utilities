# Per-package deploy README block

This is the canonical block `scripts/inject_package_deploy_readme.py` renders into each
`agents/*` `README.md` (between `<!-- BEGIN agent-os-genesis-deploy -->` markers), so any
connector can be stood up skill-guided via **agent-os-genesis** (the single-package
deploy mode). `{name}` = package/console-script base (e.g. `gitlab`), `{package}` = PyPI
name (e.g. `gitlab-api`), `{image}` = registry image. Rendered, it reads:

---

## Deploy with `agent-os-genesis`

This package can be provisioned for you — skill-guided — by the **`agent-os-genesis`**
universal skill (its *single-package deploy mode*). It picks your install method, seeds
secrets to OpenBao/Vault (or `.env`), trusts your enterprise CA, registers the MCP
server, and verifies it — the same machinery that stands up the whole Agent OS, narrowed
to just this package. Ask your agent to **"deploy `{package}` with agent-os-genesis"**.

**Install methods** (choose one; the skill drives it):

| Mode | Command |
|------|---------|
| Bare-metal, prod (PyPI) | `uvx {name}-mcp`  ·  or `uv tool install {package}` |
| Bare-metal, **dev** (editable) | `uv pip install -e ".[all]"`  ·  or `pip install -e ".[all]"` |
| Container, prod | deploy `{image}` via docker-compose / swarm / podman / podman-compose / kubernetes |
| Container, **dev** (editable) | deploy `docker/compose.dev.yml` (source-mounted at `/src`, edits live on restart) |

Secrets are read-existing + seeded via `graph_configure action=vault_sync` — you are only
prompted for what's missing. See the `agent-os-genesis` skill for the full flow.

---
