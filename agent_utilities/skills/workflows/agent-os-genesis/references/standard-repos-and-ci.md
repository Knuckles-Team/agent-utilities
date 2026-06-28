# Standard operator-owned private repos + generalized CI (genesis Step 9b)

> CONCEPT:OS-5.74 / CONCEPT:OS-5.75. The standard, **abstract** set of private repos
> every genesis deployment provisions — so an operator's environment lives in *their*
> repos + XDG config, **never in the public agent-utilities repo**.

The single source of truth is `agent_utilities/deployment/repo_templates.py`, surfaced
in `genesis.yaml → private_repos`. This page is the operator-facing explainer; the
skill (Step 9b) executes it.

## The contract (why this exists)

The operator's environment — hosts, networks, secrets, deployment manifests, resolved
config — is **operator-specific** and must not be committed to the public framework
repo. Genesis therefore provisions a consistent set of **operator-owned private repos**
from **abstract templates**: the public repo carries only placeholder skeletons
(`${GIT_NAMESPACE}`, `<host>`, `vault://…` references); the concrete values come from
`~/.config/agent-utilities/inventory.yaml` + `workspace.yml` at deploy time and are
committed into the **private** repos.

## The standard repo set

| Repo | Purpose | Seeded from (deploy time) |
|------|---------|---------------------------|
| `inventory` | Host inventory — source of truth for hosts, roles, SSH access | `~/.config/agent-utilities/inventory.yaml` |
| `config` | `workspace.yml` + **secret-redacted** `config.json`/`mcp_config.json` examples | `~/.config/agent-utilities/{workspace.yml,config.json,mcp_config.json}` |
| `networks` | Overlay/CNI bootstrap + CIDR allocations | deployment-planner network plan |
| `secrets-config` | Secrets **convention** — `service → reference` (no plaintext) | secret-store reconcile (`graph_configure vault_sync`) |
| `infrastructure` | Deployment manifests (compose/swarm/k8s stacks) | `workspace.yml` stacks |
| `pipelines` | Generalized, reusable GitLab CI templates | the templates in this module |

All repos are created **private**. `secrets-config` holds only `vault://` /
`engine://__secrets__` *references* — **never plaintext secret values**.

## Profile scaling (degrades cleanly)

| Profile | Repos | Git mode | CI | Runners |
|---------|-------|----------|----|---------|
| `tiny` | `inventory`, `config` | **local** (git init, not pushed) | off | 0 |
| `single-node-prod` | `+ networks, secrets-config, infrastructure` | remote | minimal | 1 |
| `enterprise` | `+ pipelines` | remote | full | 2+ (group) |

A Pi/tiny deploy never needs a remote git host or a CI runner; enterprise gets the
shared `pipelines` repo + runners. Larger profiles are strict supersets of smaller ones.

## How genesis runs it (idempotent)

```python
from agent_utilities.deployment import provision_plan, standard_repos, render_skeleton

plan = provision_plan(
    profile,                       # tiny | single-node-prod | enterprise
    git_host="gitlab",             # gitlab | github | local  (tiny forces local)
    namespace=operator_group,      # operator's GitLab group / GH org
    existing_repos=already_present # → action="skip" for these (idempotent)
)
```

For each repo with `action == "create"`:

1. **Create private repo** — `gitl__projects create visibility=private` (GitLab) or
   `github_repos create private=true auto_init=true` (GitHub). Reuse the
   `gitlab-repository-seeder` skill for create+seed+PAT.
2. **Render + commit the skeleton** — `render_skeleton(repo, context)` resolves the
   `${TOKEN}`s from the operator's resolved config **at deploy time**:
   `GIT_NAMESPACE`, `DEFAULT_BRANCH`, `CI_TEMPLATES_PROJECT`, `RUNNER_TAG`, `REGISTRY`.
   Commit via `gitl__commits create` (multi-file `actions[]`) or `rm_git commit/push`.
3. **Seed real operator data** — copy the resolved `inventory.yaml` / `workspace.yml`
   from the XDG config into the `inventory` / `config` private repos.

## Generalized CI + runners

When `ci.enabled` (non-tiny), seed the generalized templates into the `pipelines` repo
(enterprise) and point each repo's `.gitlab-ci.yml` at them via `include: project:`:

- `stages.yml` — shared pipeline stages.
- `agent-package-ci.yml` — lint/test/build/publish for an agent-package (Python).
- `service-deploy.yml` — orchestrator-agnostic deploy (Swarm **or** Kubernetes).

Runner registration follows `runner_plan(profile)` (`gitl__runners register` /
`enable_project`), tagged `${RUNNER_TAG}`. **No operator project-ids, registration
tokens, or runner tags are baked into the templates** — they are CI/CD variables /
placeholder tokens resolved at deploy time from the operator's secret store.

## The invariant

Nothing in `repo_templates.py` or the CI templates contains operator IPs, hostnames,
secrets, runner tags, or inventory. The test
`tests/unit/deployment/test_repo_templates.py` enforces this (a forbidden-substring
sweep + token-declaration checks). Operator-specifics enter **only** via `render()` at
deploy time, writing into the **private** repos.
