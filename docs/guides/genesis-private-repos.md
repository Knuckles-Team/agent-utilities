# Genesis standard private repos + CI (per profile)

> **CONCEPT:OS-5.74 / OS-5.75.** What private repos (and CI/runners) genesis provisions
> for you, per profile — so your environment lives in *your* repos + XDG config, never
> in the public agent-utilities repo.

Genesis Step 9b provisions a consistent, **abstract** set of operator-owned **private**
git repos. The public framework repo ships only placeholder skeletons; your concrete
inventory/networks/secrets/manifests are read from `~/.config/agent-utilities/`
(`inventory.yaml`, `workspace.yml`) at deploy time and committed into **your** private
repos. The standard is defined once in
[`agent_utilities/deployment/repo_templates.py`](https://github.com/) and surfaced in
`genesis.yaml → private_repos`.

## The standard repo set

| Repo | Purpose | Seeded from |
|------|---------|-------------|
| `inventory` | Host inventory — source of truth for hosts/roles/SSH | `~/.config/agent-utilities/inventory.yaml` |
| `config` | `workspace.yml` + secret-redacted `config.json`/`mcp_config.json` examples | XDG config |
| `networks` | Overlay/CNI bootstrap + CIDR allocations | deployment plan |
| `secrets-config` | Secrets *convention* — `service → vault://` / `engine://__secrets__` references (no plaintext) | secret-store reconcile |
| `infrastructure` | Deployment manifests (compose/swarm/k8s) | `workspace.yml` |
| `pipelines` | Generalized, reusable GitLab CI templates | the module templates |

## Profile scaling

| Profile | Repos | Git | CI | Runners |
|---------|-------|-----|----|---------|
| `tiny` | `inventory`, `config` | local (not pushed) | off | 0 |
| `single-node-prod` | + `networks`, `secrets-config`, `infrastructure` | remote | minimal | 1 |
| `enterprise` | + `pipelines` | remote | full | 2+ (group) |

Larger profiles are strict supersets of smaller ones. A Pi/tiny deploy needs no remote
git host or CI runner.

## How it runs

`provision_plan(profile, git_host=..., namespace=..., existing_repos=...)` builds an
**idempotent** plan (already-present repos → `action="skip"`). Each `create` repo is
made **private**, then its skeleton is rendered (`render_skeleton`) — resolving
`${GIT_NAMESPACE}` / `${RUNNER_TAG}` / `${CI_TEMPLATES_PROJECT}` / `${REGISTRY}` from
your config at deploy time — and committed. Generalized CI templates (`stages.yml`,
`agent-package-ci.yml`, `service-deploy.yml`) and `runner_plan(profile)` runners are
seeded for non-tiny profiles.

## The guarantee

No operator IPs, hostnames, secrets, runner tags, or inventory live in the public repo
or its templates — enforced by `tests/unit/deployment/test_repo_templates.py`.
Operator-specifics enter only via `render()` at deploy time, into the **private** repos.

The full operator recipe lives in the genesis skill reference
`agent_utilities/skills/workflows/agent-os-genesis/references/standard-repos-and-ci.md`.
