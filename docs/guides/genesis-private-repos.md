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

## Generalized, reusable CI (CONCEPT:OS-5.75)

For non-`tiny` profiles, genesis also seeds a set of **generalized** GitLab CI templates
(`CI_TEMPLATES` in `repo_templates.py`, generalized from `gitlab-pipelines/*`). They are
fully abstract — every operator specific is a `${TOKEN}` resolved at deploy time, never
baked in:

| Template | Stage(s) | What it does |
|----------|----------|--------------|
| `stages.yml` | — | The shared pipeline stage list (`prepare → scan → build → test → publish → deploy → verify`) every other template includes |
| `agent-package-ci.yml` | `test`, `build`, `publish` | Lint + `pytest` (`lint-and-test`), `docker build`, and `docker push` for a Python agent-package, tagged `${RUNNER_TAG}`, image at `${REGISTRY}/${PACKAGE_NAME}` |
| `service-deploy.yml` | `deploy` | **Orchestrator-agnostic** deploy — a `swarm-deploy` job (`docker stack deploy`) and a `k8s-deploy` job (`helm upgrade --install` or `kubectl apply -f k8s/`), selected by a `CLUSTER_TYPE` (`swarm`\|`kubernetes`) variable |

The `pipelines` repo holds these templates; every other repo's `.gitlab-ci.yml` consumes
them via `include: { project: '${CI_TEMPLATES_PROJECT}', file: [...] }`. The
`infrastructure` repo ships a minimal `.gitlab-ci.yml` that just includes `stages.yml` +
`service-deploy.yml`.

### Runners (`runner_plan(profile)`)

Runner registration scales with the profile and carries **no** operator tokens/tags:

| Profile | Runners | Scope |
|---------|---------|-------|
| `tiny` | 0 (no CI) | — |
| `single-node-prod` | 1 | `project` |
| `enterprise` | 2+ | `group` |

`runner_plan` returns `{register, count, tag_placeholder: "${RUNNER_TAG}", scope}`. The
operator registers shell/docker runners and tags them `${RUNNER_TAG}`; the project-ids
and registration tokens are read from the secret store at deploy time, never committed.

## How an operator gets these provisioned during genesis

This is **Step 9b** of the `agent-os-genesis` workflow, driven entirely off
`repo_templates.py` (the genesis skill executes the plan; `genesis.yaml → private_repos`
is the generated, machine-readable mirror via `manifest_summary()`). The flow:

```python
from agent_utilities.deployment import provision_plan, standard_repos, render_skeleton

plan = provision_plan(
    profile,                        # tiny | single-node-prod | enterprise
    git_host="gitlab",              # gitlab | github | local  (tiny forces local)
    namespace=operator_group,       # operator's GitLab group / GH org → ${GIT_NAMESPACE}
    existing_repos=already_present,  # → action="skip" for these (idempotent re-runs)
)
```

For each repo the plan marks `action="create"` (already-present repos are `"skip"`):

1. **Create it private** — `gitl__projects create visibility=private` (GitLab) or
   `github_repos create private=true` (GitHub); reuse the `gitlab-repository-seeder`
   skill for create + seed + scoped PAT. `tiny` forces `git_host="local"` (a local
   `git init`, never pushed).
2. **Render + commit the skeleton** — `render_skeleton(repo, context)` resolves the
   `${TOKEN}`s (`GIT_NAMESPACE`, `DEFAULT_BRANCH`, `CI_TEMPLATES_PROJECT`, `RUNNER_TAG`,
   `REGISTRY`) from the operator's resolved config **at deploy time** and commits the
   files. `render()` uses `Template.safe_substitute`, so a partial context never corrupts
   a skeleton.
3. **Seed real operator data** — copy the resolved `inventory.yaml` / `workspace.yml`
   from `~/.config/agent-utilities/` into the `inventory` / `config` private repos
   (`secrets-config` gets only `vault://` / `engine://__secrets__` references via the
   `graph_configure vault_sync` reconcile — never plaintext).
4. **CI + runners** (non-tiny) — seed `CI_TEMPLATES` into `pipelines` (enterprise) and
   register runners per `runner_plan(profile)`.

The plan is **idempotent** (`existing_repos` → `action="skip"`) so re-running genesis is
safe. The full operator runbook lives in the genesis skill reference
`agent_utilities/skills/workflows/agent-os-genesis/references/standard-repos-and-ci.md`.

## The guarantee

No operator IPs, hostnames, secrets, runner tags, or inventory live in the public repo
or its templates — enforced by `tests/unit/deployment/test_repo_templates.py` (a
forbidden-substring sweep + token-declaration checks). Operator-specifics enter only via
`render()` at deploy time, into the **private** repos.
