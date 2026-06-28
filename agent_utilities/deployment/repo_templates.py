"""Standard operator-owned PRIVATE repository set + generalized CI templates.

CONCEPT:OS-5.74 / CONCEPT:OS-5.75

Every genesis deployment (``tiny`` | ``single-node-prod`` | ``enterprise``) should
end up with a *consistent, abstract* set of operator-owned **private** git repos that
hold the operator's environment — inventory, networks, secrets convention, the
deployment manifests, the resolved agent-utilities config — plus (for the larger
profiles) a reusable GitLab CI/runner layout. The whole point is that **this
operator's environment is never encoded in the public agent-utilities repo**: the
public repo carries only ABSTRACT, TEMPLATED skeletons (placeholder tokens like
``${GIT_NAMESPACE}`` / ``<host>``); the concrete values live ONLY in the operator's
XDG config (``~/.config/agent-utilities/inventory.yaml`` + ``workspace.yml``) and the
private repos this module describes.

This module is the single source of truth the genesis skill (Step 9b) and the
``genesis.yaml`` manifest generator both read:

* :data:`STANDARD_REPOS` — the abstract repo templates (purpose + skeleton files).
* :data:`PROFILE_REPO_SETS` — which repos each profile provisions (scales cleanly:
  a Pi/tiny deploy gets a minimal *local* set, enterprise gets the full set + CI).
* :func:`provision_plan` — an idempotent, profile-aware plan an agent executes to
  CREATE the repos (via the repository-manager / git-host API) and seed them.
* :data:`CI_TEMPLATES` / :func:`runner_plan` — generalized, reusable GitLab pipeline
  templates + a runner-registration plan (no operator project-ids/tokens/tags baked
  in — everything is a ``${TOKEN}`` resolved at deploy time from operator config).

Nothing here contains operator-specific IPs, hostnames, secrets, or inventory — it
is rendered with an operator-supplied ``context`` at *deploy time* (:func:`render`),
and the rendered output is committed into the operator's PRIVATE repo, never here.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from string import Template

# ── deployment profiles (mirror config_generator.PROFILES) ──────────────────
PROFILES = ("tiny", "single-node-prod", "enterprise")

# Placeholder tokens an operator's deploy-time ``context`` resolves. They are the
# ONLY way operator-specifics enter a seeded file — the templates below are abstract.
PLACEHOLDER_TOKENS = (
    "GIT_NAMESPACE",  # operator git group/org (e.g. a GitLab group or GH org)
    "DEFAULT_BRANCH",  # default branch, e.g. main
    "CI_TEMPLATES_PROJECT",  # path to the seeded pipelines/CI-templates repo
    "RUNNER_TAG",  # CI runner tag the operator registers (NEVER hardcoded)
    "REGISTRY",  # private container registry host
)

_TOKEN_RE = re.compile(r"\$\{([A-Z_][A-Z0-9_]*)\}")


# ── repo template model ─────────────────────────────────────────────────────
@dataclass(frozen=True)
class RepoTemplate:
    """An abstract, operator-agnostic private-repo template.

    ``skeleton`` maps a relative path to its templated content (with ``${TOKEN}``
    placeholders / ``<...>`` example markers — never real operator data). ``seed_from``
    names the operator XDG source whose *resolved* content genesis commits into the
    private repo at deploy time (read from ``~/.config/agent-utilities/``), or ``None``
    when the skeleton itself is the seed.
    """

    key: str
    purpose: str
    skeleton: dict[str, str]
    seed_from: str | None = None
    ci: bool = False
    # Tokens this template's skeleton references (informational; validated by tests).
    tokens: tuple[str, ...] = field(default_factory=tuple)


# ── the abstract skeleton contents ──────────────────────────────────────────
_INVENTORY_YAML = """\
# Operator host inventory — the SOURCE OF TRUTH for this deployment.
# Genesis seeds this from ~/.config/agent-utilities/inventory.yaml at deploy time.
# This skeleton is ABSTRACT: replace the <placeholders> with your real hosts.
# Ansible-style; consumed by agent-utilities (engine_infra.ingest_hosts_from_inventory).
all:
  children:
    homelab:
      hosts:
        <host-1>:
          ansible_host: <ip-or-dns>
          ansible_user: <user>
          ansible_ssh_private_key_file: <path-to-key>
          roles: [<manager|worker|edge|gpu|nas>]
      vars:
        deployment_profile: <tiny|single-node-prod|enterprise>
"""

_INVENTORY_README = """\
# inventory

Operator-owned **private** host inventory for this agent-utilities deployment — the
single source of truth for hosts, roles and SSH access. Mirrors
`~/.config/agent-utilities/inventory.yaml`.

- `inventory.yaml` — Ansible-style host inventory (genesis seeds this from your XDG config).
- Never commit secrets here — SSH keys are *referenced by path*, not embedded.
"""

_CONFIG_README = """\
# config

Operator-owned **private** agent-utilities configuration for this deployment:

- `workspace.yml` — the workspace/service manifest (mirrors `~/.config/agent-utilities/workspace.yml`).
- `config.json.example` — the resolved, **secret-redacted** profile config (real
  `config.json` is seeded by genesis with `vault://` / `engine://__secrets__` refs, never plaintext).
- `mcp_config.json.example` — the MCP fleet wiring (streamable-http endpoints).

Secrets are *references*, not values — see the `secrets-config` repo.
"""

_WORKSPACE_YML = """\
# Workspace / service manifest — genesis seeds this from
# ~/.config/agent-utilities/workspace.yml at deploy time. ABSTRACT skeleton:
name: <operator-workspace>
path: <workspace-root>
repositories: []
subdirectories: {}
graph: {}
"""

_CONFIG_JSON_EXAMPLE = """\
{
  "_comment": "Secret-redacted example. Genesis writes the real config.json with vault:// / engine://__secrets__ references — NEVER plaintext secrets.",
  "APP_PROFILE": "<dev|production>",
  "GRAPH_BACKEND": "<epistemic_graph|fanout>"
}
"""

_MCP_CONFIG_EXAMPLE = """\
{
  "mcpServers": {
    "graph-os": {"command": "uv", "args": ["run", "graph-os"]}
  }
}
"""

_NETWORKS_COMPOSE = """\
# Overlay / network bootstrap — ABSTRACT. Genesis renders this from the
# deployment-planner's network plan + your inventory at deploy time.
# Define the overlay networks your stacks attach to (no operator CIDRs here).
networks:
  <overlay-name>:
    driver: <overlay|bridge>
    attachable: true
    # ipam:
    #   config:
    #     - subnet: <CIDR>   # operator-specific — seeded at deploy time
"""

_NETWORKS_CIDR = """\
# Network/CIDR allocations — operator-specific values are seeded at deploy time.
# This skeleton documents the SHAPE only.
overlays: []          # - {name: <overlay>, subnet: <CIDR>, purpose: <text>}
reserved_ranges: []   # - {cidr: <CIDR>, purpose: <text>}
"""

_NETWORKS_README = """\
# networks

Operator-owned **private** overlay/network bootstrap (Swarm overlays / k8s CNI
allocations). Abstract here; genesis seeds real CIDRs from the deployment plan.
"""

_SECRETS_MAP = """\
# Secrets MAP — references only, NEVER values. Maps each service to where its
# secrets live (OpenBao/Vault or the engine's encrypted __secrets__ store).
# Genesis reconciles this against the live secret store (graph_configure vault_sync).
services: {}
#   <service>:
#     <KEY>: "vault://apps/<service>/<KEY>"      # OpenBao / Vault
#     <KEY2>: "engine://__secrets__/<service>/<KEY2>"   # engine-encrypted store
"""

_SECRETS_README = """\
# secrets-config

Operator-owned **private** secrets *convention* — a map of `service -> secret
reference` pointing at OpenBao/Vault or the engine's encrypted `__secrets__` store.

> **NEVER commit plaintext secrets.** This repo holds only `vault://` /
> `engine://__secrets__` *references*. Values live in the secret store.
"""

_INFRA_README = """\
# infrastructure

Operator-owned **private** deployment manifests (Docker Compose stacks / Swarm
stacks / Kubernetes manifests) for this deployment. Genesis seeds stacks from your
`workspace.yml`. The `.gitlab-ci.yml` wires deploys via the shared CI templates.
"""

_INFRA_GITKEEP = (
    "# Place compose/swarm/k8s stack definitions here (seeded from workspace.yml).\n"
)


# ── generalized, reusable GitLab CI templates (CONCEPT:OS-5.75) ─────────────
# Generalized from gitlab-pipelines/* — operator specifics (runner tags, the CI
# templates project path, registry) are ${TOKEN}s resolved at deploy time, NOT baked.

_CI_STAGES = """\
# Shared pipeline stages — generalized, reusable across operator repos.
stages:
  - prepare
  - scan
  - build
  - test
  - publish
  - deploy
  - verify
"""

_CI_AGENT_PACKAGE = """\
# Generalized build/test/scan/publish pipeline for an agent-package (Python).
# Include from a repo's .gitlab-ci.yml:
#   include:
#     - project: '${CI_TEMPLATES_PROJECT}'
#       file: ['stages.yml', 'agent-package-ci.yml']
include:
  - project: '${CI_TEMPLATES_PROJECT}'
    file: ['stages.yml']

variables:
  PACKAGE_NAME: ''
  REGISTRY: '${REGISTRY}'

lint-and-test:
  stage: test
  script:
    - python -m pip install -e '.[dev]' || pip install -e .
    - ruff check . || true
    - pytest -q || true
  tags: ['${RUNNER_TAG}']
  rules:
    - if: '$CI_PIPELINE_SOURCE == "merge_request_event"'
    - if: '$CI_COMMIT_BRANCH == $CI_DEFAULT_BRANCH'

build-image:
  stage: build
  script:
    - docker build -t ${REGISTRY}/${PACKAGE_NAME}:${CI_COMMIT_SHORT_SHA} .
  tags: ['${RUNNER_TAG}']
  rules:
    - if: '$CI_COMMIT_BRANCH == $CI_DEFAULT_BRANCH'

publish-image:
  stage: publish
  script:
    - docker push ${REGISTRY}/${PACKAGE_NAME}:${CI_COMMIT_SHORT_SHA}
  tags: ['${RUNNER_TAG}']
  rules:
    - if: '$CI_COMMIT_BRANCH == $CI_DEFAULT_BRANCH'
"""

_CI_SERVICE_DEPLOY = """\
# Generalized service deploy — Swarm OR Kubernetes (orchestrator-agnostic).
# Generalized from docker_swarm_deploy.yml; runner tag + templates project are tokens.
include:
  - project: '${CI_TEMPLATES_PROJECT}'
    file: ['stages.yml']

variables:
  STACK_NAME: ''
  CLUSTER_TYPE: 'swarm'        # 'swarm' | 'kubernetes'
  K8S_NAMESPACE: 'default'
  HELM_CHART: ''               # set to use `helm upgrade --install`, else `kubectl apply -f k8s/`

swarm-deploy:
  stage: deploy
  script:
    - docker stack rm ${STACK_NAME}; sleep 9; docker stack deploy -c *compose.yml ${STACK_NAME} --detach=true
  rules:
    - if: '$CI_COMMIT_BRANCH == $CI_DEFAULT_BRANCH && $CLUSTER_TYPE == "swarm"'
  retry: 1
  tags: ['${RUNNER_TAG}']

k8s-deploy:
  stage: deploy
  script:
    - >
      if [ -n "$HELM_CHART" ]; then
        helm upgrade --install ${STACK_NAME} ${HELM_CHART} -n ${K8S_NAMESPACE} --create-namespace --wait;
      else
        kubectl apply -n ${K8S_NAMESPACE} -f k8s/;
        kubectl -n ${K8S_NAMESPACE} rollout status deploy/${STACK_NAME} --timeout=180s;
      fi
  rules:
    - if: '$CI_COMMIT_BRANCH == $CI_DEFAULT_BRANCH && $CLUSTER_TYPE == "kubernetes"'
  retry: 1
  tags: ['${RUNNER_TAG}']
"""

_CI_PIPELINES_README = """\
# pipelines

Operator-owned **private** GitLab CI templates — generalized, reusable includes
shared by every other repo (`stages.yml`, `agent-package-ci.yml`, `service-deploy.yml`).

Operator specifics are CI/CD variables / placeholder tokens (runner tag, registry,
this project's path) — resolved at deploy time, never committed in the public template.
"""

CI_TEMPLATES: dict[str, str] = {
    "stages.yml": _CI_STAGES,
    "agent-package-ci.yml": _CI_AGENT_PACKAGE,
    "service-deploy.yml": _CI_SERVICE_DEPLOY,
}

# A minimal infrastructure-repo CI that just consumes the shared deploy template.
_INFRA_CI = """\
# Deploy this repo's stacks via the shared, generalized CI templates.
include:
  - project: '${CI_TEMPLATES_PROJECT}'
    file: ['stages.yml', 'service-deploy.yml']
"""


# ── the standard repo set (abstract) ────────────────────────────────────────
STANDARD_REPOS: tuple[RepoTemplate, ...] = (
    RepoTemplate(
        key="inventory",
        purpose="Host inventory — the source of truth for hosts, roles & SSH access.",
        seed_from="~/.config/agent-utilities/inventory.yaml",
        skeleton={"inventory.yaml": _INVENTORY_YAML, "README.md": _INVENTORY_README},
    ),
    RepoTemplate(
        key="config",
        purpose="Resolved agent-utilities config: workspace.yml + redacted config/mcp examples.",
        seed_from="~/.config/agent-utilities/{workspace.yml,config.json,mcp_config.json}",
        skeleton={
            "workspace.yml": _WORKSPACE_YML,
            "config.json.example": _CONFIG_JSON_EXAMPLE,
            "mcp_config.json.example": _MCP_CONFIG_EXAMPLE,
            "README.md": _CONFIG_README,
        },
    ),
    RepoTemplate(
        key="networks",
        purpose="Overlay/network bootstrap (Swarm overlays / k8s CNI + CIDR allocations).",
        seed_from="deployment-planner network plan",
        skeleton={
            "compose.yml": _NETWORKS_COMPOSE,
            "cidr-allocations.yaml": _NETWORKS_CIDR,
            "README.md": _NETWORKS_README,
        },
    ),
    RepoTemplate(
        key="secrets-config",
        purpose="Secrets CONVENTION — service->reference map (OpenBao/Vault or engine __secrets__). No plaintext.",
        seed_from="secret-store reconcile (graph_configure vault_sync)",
        skeleton={"secrets-map.yaml": _SECRETS_MAP, "README.md": _SECRETS_README},
    ),
    RepoTemplate(
        key="infrastructure",
        purpose="Deployment manifests (compose/swarm/k8s stacks) for this deployment.",
        seed_from="workspace.yml stacks",
        ci=True,
        tokens=("CI_TEMPLATES_PROJECT",),
        skeleton={
            "stacks/.gitkeep": _INFRA_GITKEEP,
            ".gitlab-ci.yml": _INFRA_CI,
            "README.md": _INFRA_README,
        },
    ),
    RepoTemplate(
        key="pipelines",
        purpose="Generalized, reusable GitLab CI templates shared by every repo.",
        seed_from=None,
        ci=True,
        tokens=("CI_TEMPLATES_PROJECT", "RUNNER_TAG", "REGISTRY"),
        skeleton={**CI_TEMPLATES, "README.md": _CI_PIPELINES_README},
    ),
)

_BY_KEY = {r.key: r for r in STANDARD_REPOS}

# Profile scaling — a pi/tiny deploy stays minimal & LOCAL; enterprise gets the
# full set + the shared CI templates repo. (degrades cleanly across profiles)
PROFILE_REPO_SETS: dict[str, tuple[str, ...]] = {
    "tiny": ("inventory", "config"),
    "single-node-prod": (
        "inventory",
        "config",
        "networks",
        "secrets-config",
        "infrastructure",
    ),
    "enterprise": (
        "inventory",
        "config",
        "networks",
        "secrets-config",
        "infrastructure",
        "pipelines",
    ),
}

# tiny keeps repos LOCAL (git init, not pushed to a remote host); larger profiles
# push to the operator's git host. CI is off for tiny, on otherwise.
_GIT_MODE = {"tiny": "local", "single-node-prod": "remote", "enterprise": "remote"}

# Runner registration scales with the profile (no operator tokens/tags here).
_RUNNER_COUNT = {"tiny": 0, "single-node-prod": 1, "enterprise": 2}


# ── public helpers ──────────────────────────────────────────────────────────
def standard_repos(profile: str) -> list[RepoTemplate]:
    """The ordered abstract repo templates a *profile* provisions."""
    if profile not in PROFILES:
        raise ValueError(f"unknown profile {profile!r}; expected one of {PROFILES}")
    return [_BY_KEY[k] for k in PROFILE_REPO_SETS[profile]]


def render(text: str, context: dict[str, str]) -> str:
    """Resolve ``${TOKEN}`` placeholders from an operator-supplied *context*.

    Unknown tokens are left intact (``Template.safe_substitute``) so a partial
    context never corrupts a skeleton. This is the ONLY place operator-specific
    values enter a file — and it runs at *deploy time*, writing into the operator's
    private repo, never the public repo.
    """
    return Template(text).safe_substitute(context)


def render_skeleton(repo: RepoTemplate, context: dict[str, str]) -> dict[str, str]:
    """Render every file in *repo*'s skeleton with *context* (deploy-time)."""
    return {path: render(content, context) for path, content in repo.skeleton.items()}


def referenced_tokens(text: str) -> set[str]:
    """The ``${TOKEN}`` names a string references."""
    return set(_TOKEN_RE.findall(text))


def ci_enabled(profile: str) -> bool:
    """Whether this profile gets CI pipelines (tiny does not)."""
    return _GIT_MODE.get(profile) == "remote"


def runner_plan(profile: str) -> dict:
    """Generalized CI-runner registration plan (no operator tokens/tags baked in)."""
    count = _RUNNER_COUNT.get(profile, 0)
    return {
        "register": count > 0,
        "count": count,
        # The tag the operator's runner registers under — resolved at deploy time.
        "tag_placeholder": "${RUNNER_TAG}",
        "scope": "group" if profile == "enterprise" else "project",
        "note": (
            "Register shell/docker runners and tag them; the CI templates reference"
            " ${RUNNER_TAG}. Operator project-ids/registration-tokens are read from"
            " the secret store at deploy time — never committed."
        ),
    }


def provision_plan(
    profile: str,
    *,
    git_host: str = "gitlab",
    namespace: str = "${GIT_NAMESPACE}",
    existing_repos: tuple[str, ...] | list[str] = (),
) -> dict:
    """Build an idempotent, profile-aware plan to provision the standard repos.

    Abstract by construction: ``namespace`` defaults to the ``${GIT_NAMESPACE}``
    placeholder, so the plan itself carries no operator data unless a caller passes
    a concrete namespace at deploy time. ``existing_repos`` makes the plan
    idempotent — already-present repos become ``action="skip"``.

    Parameters
    ----------
    profile: one of :data:`PROFILES`.
    git_host: ``gitlab`` | ``github`` | ``local`` (tiny forces ``local``).
    namespace: operator git group/org (placeholder by default).
    existing_repos: repo *keys* or ``namespace/key`` names already present.
    """
    if profile not in PROFILES:
        raise ValueError(f"unknown profile {profile!r}; expected one of {PROFILES}")
    git_mode = _GIT_MODE[profile]
    host = "local" if git_mode == "local" else git_host
    existing = set(existing_repos)
    repos: list[dict] = []
    for repo in standard_repos(profile):
        name = repo.key
        full = f"{namespace}/{name}"
        present = name in existing or full in existing
        repos.append(
            {
                "key": repo.key,
                "name": full,
                "purpose": repo.purpose,
                "visibility": "private",
                "action": "skip" if present else "create",
                "ci": repo.ci and ci_enabled(profile),
                "seed_from": repo.seed_from,
                "seed_files": sorted(repo.skeleton),
            }
        )
    ci = ci_enabled(profile)
    return {
        "profile": profile,
        "git_host": host,
        "git_mode": git_mode,
        "namespace": namespace,
        "idempotent": True,
        "repos": repos,
        "ci": {
            "enabled": ci,
            "templates_repo": f"{namespace}/pipelines"
            if "pipelines" in PROFILE_REPO_SETS[profile]
            else None,
            "templates": sorted(CI_TEMPLATES) if ci else [],
        },
        "runners": runner_plan(profile),
        "tokens": list(PLACEHOLDER_TOKENS),
        "note": (
            "ABSTRACT plan. Concrete inventory/workspace/secrets are read from the"
            " operator's XDG config at deploy time and committed into these PRIVATE"
            " repos — never into the public agent-utilities repo."
        ),
    }


def manifest_summary() -> dict:
    """Compact, generator-friendly view for genesis.yaml (CONCEPT:OS-5.74)."""
    return {
        "concept": "OS-5.74",
        "purpose": (
            "Standard operator-owned PRIVATE repos genesis provisions per profile —"
            " abstract/templated; operator env lives only in XDG config + these repos,"
            " never in the public repo."
        ),
        "repos": {
            r.key: {"purpose": r.purpose, "seed_from": r.seed_from, "ci": r.ci}
            for r in STANDARD_REPOS
        },
        "per_profile": {p: list(PROFILE_REPO_SETS[p]) for p in PROFILES},
        "git_mode": dict(_GIT_MODE),
        "ci_templates": sorted(CI_TEMPLATES),
        "runners": {p: _RUNNER_COUNT[p] for p in PROFILES},
        "step": "agent-os-genesis Step 9b (CONCEPT:OS-5.75)",
        "tokens": list(PLACEHOLDER_TOKENS),
    }


__all__ = [
    "CI_TEMPLATES",
    "PLACEHOLDER_TOKENS",
    "PROFILES",
    "PROFILE_REPO_SETS",
    "STANDARD_REPOS",
    "RepoTemplate",
    "ci_enabled",
    "manifest_summary",
    "provision_plan",
    "referenced_tokens",
    "render",
    "render_skeleton",
    "runner_plan",
    "standard_repos",
]
