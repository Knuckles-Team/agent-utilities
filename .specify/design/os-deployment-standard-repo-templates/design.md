# Design Document: Every genesis profile provisions an abstract, templated set of operator-owned PRIVATE repos (+ generalized reusable CI templates) from ONE module — never bakes operator-specific values into the public agent-utilities repo

CONCEPT:AU-OS.deployment.standard-repo-templates (covers the cluster:
`AU-OS.deployment.concept-2` is the pointer for the CI-templates/runner-plan half
of this same module and decision)

> `agent_utilities/deployment/repo_templates.py:1-27` (module docstring,
> `STANDARD_REPOS`, `PROFILE_REPO_SETS`, `provision_plan`, `CI_TEMPLATES`,
> `runner_plan`).

## Decision — `repo_templates.py` is the single source of truth both the genesis skill (Step 9b) and the `genesis.yaml` manifest generator read: an ABSTRACT, profile-scaled set of operator-owned private repo templates (inventory, networks, secrets convention, deployment manifests, resolved config — scaling from a minimal local set on `tiny` to the full set + CI on `enterprise`), plus generalized, reusable GitLab CI/runner templates with no operator project-ids/tokens/tags baked in, rendered with an operator-supplied `context` only AT DEPLOY TIME into the operator's own private repo

Every genesis deployment needs a consistent way to keep the operator's actual
environment — infrastructure inventory, network topology, secrets conventions, the
resolved config — OUT of the public `agent-utilities` repo, while still giving
every profile a repeatable, versioned place for that environment to live. This
module is the ONE place that structure is defined: `STANDARD_REPOS` are abstract
skeletons with placeholder tokens (`${GIT_NAMESPACE}`, `<host>`) rather than
concrete values; `PROFILE_REPO_SETS` decides which repos each profile provisions,
scaling cleanly from a minimal `tiny`/Pi deployment to the full `enterprise` set
plus CI; `provision_plan` is the idempotent, profile-aware execution plan an agent
runs (via repository-manager / the git-host API) to create and seed the repos; and
`CI_TEMPLATES`/`runner_plan` extend the same pattern to reusable GitLab pipeline
templates and runner registration, with every operator-specific value (`${TOKEN}`)
resolved only at deploy time from operator config. "Nothing here contains operator-
specific IPs, hostnames, secrets, or inventory" (`repo_templates.py:27-28`) — the
rendered, concrete output is committed into the operator's private repo, never this
one.

## Rejected alternative — let each operator hand-roll their own private-repo layout and CI setup, or bake operator-specific values directly into agent-utilities for convenience

Two shapes are rejected by the "abstract, templated, profile-scaled" design. First,
no standard at all — every operator improvising their own private-repo structure
and CI layout — was rejected because it makes every deployment's environment
bespoke and undocumented, with no shared tooling (`provision_plan`, `runner_plan`)
able to act on it generically across profiles. Second, and the one the module's own
opening line warns against directly — baking concrete operator values (their actual
git namespace, host, project-ids, tokens) into the public repo for convenience —
was rejected as the core privacy/security boundary the whole module exists to
enforce: "this operator's environment is never encoded in the public agent-utilities
repo" (`repo_templates.py:9-10`). Templating with placeholder tokens, resolved only
at deploy time into a repo the operator controls, keeps the public repo's git
history free of any single operator's real topology while still giving every
operator the SAME reusable structure and CI templates to render into their own repos.

## Risk Assessment

- **Blast Radius**: `agent_utilities/deployment/repo_templates.py`,
  `scripts/gen_genesis_manifest.py`, `genesis.yaml`, the genesis skill's Step 9b.
- **Backward Compatible**: Yes — additive templating; does not change any
  already-provisioned operator repo's content.
- **Known weak point**: `provision_plan`/`runner_plan` describe an idempotent plan
  an agent EXECUTES via repository-manager/the git-host API — the templates
  themselves guarantee no operator secrets leak into the PUBLIC repo, but nothing
  in this module verifies the render step was actually run correctly against a
  given operator's real private repo, or that a stale rendered copy in that
  private repo gets refreshed when the template changes upstream.
