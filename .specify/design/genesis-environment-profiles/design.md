# Design Document: Named, explicitly-reviewable genesis k8s deployment-input profiles

CONCEPT:AU-OS.deployment.genesis-environment-profiles

> Realised by `agent_utilities/deployment/genesis_environments.py` (schema, loader,
> extension discovery, fail-loud validation), `deploy/environments/{dev,test,prod}.yaml`
> (the three default profiles), the `setup-config environments` subcommand
> (`agent_utilities/deployment/cli.py`), and the `environment_profiles` section of
> `genesis.yaml` (`scripts/gen_genesis_manifest.py`).

## Decision — a profile is ten typed, fully-explicit input categories plus a discovery-based extension mechanism; secrets are references only, and a missing one is a hard load-time failure

Genesis already had a "profile" axis — `tiny` / `single-node-prod` / `enterprise`
(`agent_utilities/deployment/config_generator.py:PROFILES`). That axis answers
*how big is this deployment* (topology, engine shape, connector count). It does not
answer *which named environment am I deploying into* — a `single-node-prod` shape
can legitimately be a `dev` box, a `test` box, or `prod`, and an `enterprise` shape
commonly runs `dev`/`test`/`prod`/`uat`/`sit` k8s namespaces side by side. Those are
two independent axes: **topology profile** (existing) × **named environment profile**
(this decision). Conflating them would have made every operator with a `uat` or `sit`
environment either misuse `enterprise` as a stand-in name or fork the codebase to add
an enum value — exactly the "some people have more envs... ability to update/modify/
extend" requirement this was written against.

**The environment set is data, not an enum.** `list_environment_profiles()` discovers
profiles by scanning two directories for `*.yaml` files: the repo-shipped
`deploy/environments/` (ships `dev.yaml`, `test.yaml`, `prod.yaml`) and the operator's
XDG config extension directory (`agent_utilities.core.paths.config_dir() /
"environments"`, i.e. `~/.config/agent-utilities/environments/` by default). A new
`uat.yaml` dropped in either directory is immediately a valid `--profile uat` — no
code change, no PR to this repo. `load_environment_profile("uat")` on an
undiscovered name fails with the exact list of what *was* found and where a new file
would go; it never silently falls back to a "closest" name.

**Every one of the ten input categories is a closed-schema section, not a bag of
`Any`.** `environment, target, release, runtime, filesystem, configuration, secrets,
network, identity, validation` are ten frozen dataclasses assembled into one
`EnvironmentProfile`. Parsing a profile file requires **every** field the dataclass
declares to be present under its section — an omitted field is a load error naming
the missing key, and an unrecognized key is also a load error (typo/unreviewed-value
protection). This is the direct implementation of the owner's first principle:
*"a profile is a thing a human reads before a deploy and can reason about... no
hidden defaults resolved at apply time the reader cannot see in the file."* A sparse
YAML that only sets the fields someone thought of, filled out with silent Python
defaults for the rest, fails that test even when every present value is correct —
the reader cannot tell, from the file, what the deployment will actually do. The
`_require_keys` helper enforces the section's declared key set exactly; there is
deliberately no `**kwargs`-style passthrough section anywhere in the schema.

**Secrets are the sharpest constraint, and they get their own closed vocabulary.**
`secrets.required` is a list of `SecretReference(name, ref, keys)`. `ref` must match
one of four schemes — `env://VAR`, `vault://path`, `secret://path`, or
`k8s-secret://<namespace>/<name>` (the first three already exist repo-wide, see
`agent_utilities/core/config.py:_RUNTIME_SECRET_REFERENCE_RE` and
`agent_utilities/core/credentials.py`; `k8s-secret://` is added here because genesis
k8s inputs need to say *"this Secret object must already exist in this namespace"*
distinctly from *"resolve this value from OpenBao at process start"* — the graph-os
production manifest's `envFrom: secretRef: name: graph-os-secrets` is exactly that
case). Anything that is not one of those four schemes — including a bare string that
looks like it could be a credential — is rejected at load time, not merely
discouraged by convention. `keys` names which fields the secret object is expected to
contain (e.g. `OIDC_CLIENT_SECRET`), so the profile documents *what* must be supplied
without ever holding *the value*. There is no field anywhere in the schema shaped
like a credential value; `configuration.env` (the one place free-form key/value pairs
are allowed, for genuinely non-secret settings) is validated against the same
secret-suffix heuristic `config_generator.py` already uses (`_is_secret` /
`_SECRET_SUFFIXES`) so a `*_TOKEN` or `*_PASSWORD` key cannot be smuggled into the
"this is fine to read out loud" section.

**A missing required secret reference is a load-time `MissingSecretReferenceError`,
never a default.** This is the direct implementation of the owner's second
principle: *"we do not infer credentials."* There is no code path in
`genesis_environments.py` that manufactures, guesses, or silently omits a secret
reference — `validate_environment_profile` raises naming the exact secret `name`
that has no `ref`, and `identity.client_secret_ref` (when set) is cross-checked
against `secrets.required` by name so an identity section cannot point at a secret
that was never declared. BUG-006 (an exposed graph-service secret) and BUG-038 (a
live cluster that only worked because of an undocumented out-of-band admission call)
are both instances of "the system proceeded past a place a human should have had to
look" — this schema makes that place unavoidable and machine-checked.

**Two cross-checks reuse `genesis.yaml` instead of re-declaring its enums.**
`target.orchestrator`, `target.authority`, and `identity.idp` are validated against
`genesis.yaml`'s own `run_plan.orchestrators` / `run_plan.substrate_authority` /
`run_plan.idp` lists (read as data — a `yaml.safe_load`, not a package import, so no
circular dependency) rather than a second hard-coded enum that could drift from the
one `genesis.yaml` already publishes. This is the same "compose sources that already
exist" discipline `gen_genesis_manifest.py`'s own docstring states for the file as a
whole.

**Tier-gated rules encode two existing prose safety invariants as code.** The
`agent-os-genesis` skill's safety invariants already say "pin production images by
digest... floating tags are development-only" and this repo's own working discipline
says "a real MCP `tools/list`, not merely `/health`." Those were prose before this
change — true, but not provable except by review. `validate_environment_profile` now
enforces both: `environment.tier == "prod"` requires `release.tag_policy ==
"digest-pinned"` with a non-null `release.revision`, and **every** profile
(non-prod included — the MCP-liveness rule is not a production-only concern) must
declare at least one `validation.functional_checks` entry of kind
`"mcp-tools-list"`. A profile that only checks `/health` fails to load, with the
exact reason named.

## Rejected alternative — extend `config_generator.PROFILES` with more string values

The obvious minimal change was adding `"uat"`, `"sit"` and any operator's future name
to the existing `PROFILES` tuple. Rejected for three reasons: (1) it re-conflates the
topology axis with the environment-name axis discussed above; (2) it requires a code
change (a PR to this repo) for every operator's local environment name, which is
exactly the closed-enum problem the request called out; (3) `PROFILES` currently
drives `_PROFILE_PRESETS`/`_PROFILE_REQUIRED` — deployment-topology behavior, not
ten independently-reviewable k8s input categories — so widening it would have
smuggled unrelated dev/test/prod semantics into a schema that already means
something else.

## Extension — `filesystem.runtime_paths` (BUG-ROFS-1 runtime-path configurability)

The original ten sections modeled a writable-path *exception* to
`read_only_root_filesystem` (`filesystem.writable_paths`: WHERE a volume is
mounted and WHY) but not WHICH env var a served process reads to find its home
inside that mount. That gap was discovered live: the graph-os `readOnlyRootFilesystem`
rollout (`services/graph-os` `fix/rofs-tmp-home`) had to empirically rediscover the
image's real `$HOME` (`/tmp`, not `/home/app` — that account does not exist) by
reading `/etc/passwd` inside a running container, then hand-wire `HOME`/
`XDG_CACHE_HOME`/`XDG_CONFIG_HOME`/`XDG_DATA_HOME`/`XDG_STATE_HOME`/
`AGENT_UTILITIES_DATA_DIR` as scattered literals split between a k8s ConfigMap and a
per-container `env:` block, with no schema forcing them to agree.

`filesystem.runtime_paths` closes that gap **inside the existing section** rather
than adding an eleventh top-level category or a parallel mechanism:
`RuntimePathBinding(env_var, path, writable_path_ref)` binds each of the closed
`RUNTIME_PATH_ENV_VARS` set (`HOME`, the three XDG dirs a served process actually
consults, `XDG_STATE_HOME`, and `AGENT_UTILITIES_DATA_DIR`) to an explicit path, and
`writable_path_ref` must name — and `path` must fall under — a `filesystem.
writable_paths[].mount_path` this same profile already declared and justified.
`validate_environment_profile` enforces: every one of the six vars is bound exactly
once (closed, like every other section — an env var this schema doesn't know about,
or a missing one, is a load error); every `path` is absolute; every binding is
anchored under a real, reviewed writable mount. A profile can no longer point
`AGENT_UTILITIES_DATA_DIR` (or any of the others) somewhere the `filesystem` section
never reviewed as writable, and the exact set BUG-ROFS-1 needed is now something a
reviewer reads in one place instead of reconstructing from a live container.

This section is an **input schema only** (same scope boundary as the rest of the
concept, below) — it declares what the k8s manifest's `ConfigMap`/`env:` blocks
*should* say, machine-checked; the graph-os manifest itself
(`services/graph-os/k8s/manifests.yaml`'s `graph-os-env` ConfigMap) is the actually
-applied source of truth for the live deployment and is kept in sync by hand today
(see that repo's own `AGENTS.md`), the same "input, not renderer" relationship the
Scope note below already describes for the other nine sections.

## Scope note

This concept covers the environment-profile schema, its loader/extension
discovery, and its fail-loud secret-reference validation. It does not cover
rendering the profile into live Kubernetes manifests — that remains
`agent-os-genesis`'s Phase 4 (`references/kubernetes-and-helm.md`) and the existing
`deploy/k8s/production-cell/` render pipeline
(`scripts/release/render_production_cell.py`); this schema is a typed, validated
**input** those phases can consume, not a second renderer (see "Sprawl boundaries —
extend before you add" in `AGENTS.md`).

## Risk Assessment

- **Blast Radius**: new module (`agent_utilities/deployment/genesis_environments.py`),
  new data files (`deploy/environments/*.yaml`), one new CLI subcommand group
  (`setup-config environments`), and an additive `environment_profiles` section in
  the generated `genesis.yaml`. No existing profile, config field, or deployment path
  is changed or removed.
- **Backward Compatible**: yes — `config_generator.PROFILES` (tiny/single-node-prod/
  enterprise) is untouched; this is a new, independent axis an operator opts into.
- **Known weak point**: the `k8s-secret://` scheme's namespace/name are structurally
  validated (DNS-1123-shaped) but existence in the live cluster is never checked here
  — that remains a deploy-time/Phase-4 concern (`kubectl get secret`), consistent
  with this module never touching a live cluster or a secret's value.
