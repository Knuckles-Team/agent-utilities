# Modular prompt & skill contribution

> CONCEPT:AU-OS.deployment.agent-factory-autoload (entry-point discovery) · CONCEPT:AU-ORCH.routing.resolve-body-single-canonical (canonical prompt
> schema) · CONCEPT:AU-KG.compute.user-override-prompt-library (KG prompt-library ingestion + XDG overlay)

## Why

`agent-utilities` and `universal-skills` used to be the only homes for system
prompts (~90 JSON blueprints in `agent_utilities/prompts/`) and skills (~330 in
`universal-skills`). The ~63 agent-packages under `agent-packages/agents/*`
carried almost nothing of their own. That made the hub heavy and coupled every
agent's prompt/skills to a central repo.

This subsystem inverts the topology: **each agent-package ships its own system
prompt(s) and skills inside its own wheel**, and the hub *discovers* them. The
hub stays lean (it gains discovery code, not assets); a package is modular and
self-contained.

## How discovery works

Any package opts in by declaring two setuptools entry-points pointing at
*data-only* subpackages:

```toml
[project.entry-points."agent_utilities.skill_providers"]
servicenow-api = "servicenow_api.skills"
[project.entry-points."agent_utilities.prompt_providers"]
servicenow-api = "servicenow_api.prompts"
```

The hub resolves each entry-point from its owning distribution's file manifest and
`RECORD`; provider modules are never imported. The target must be a data-package
directory owned by that same distribution. Duplicate names, including portable
case-fold collisions, fail closed before mutation. A provider-local missing or invalid
source is isolated and reported unhealthy without making stale XDG content current.

```mermaid
flowchart TD
    subgraph pkg["agent-package wheel (servicenow-api)"]
        SK["servicenow_api/skills/**/SKILL.md"]
        PR["servicenow_api/prompts/*.json"]
        EP["pyproject entry-points:\nskill_providers / prompt_providers"]
    end

    EP -. importlib.metadata .-> RES["core.providers\nvalidated current resolvers"]

    subgraph hub["agent-utilities (lean hub)"]
        RES --> MAT["core.unified_install"]
        RES --> INST["toolkit installer"]
        RES --> ING["registry_builder.ingest_prompts_to_graph()"]
    end

    SK --> MAT
    SK --> INST
    PR --> ING
    MAT --> XDG["XDG data: skills/provider/.generations/digest/skill\n(v2 activation marker)"]
    INST --> TOOLS["detected agent tools"]
    ING --> KG[("KG prompt library\nPromptNode prompt:&lt;pkg&gt;/&lt;name&gt;")]
    OVL["~/.config/agent-utilities/prompts/\n(operator XDG overlay)"] --> ING
    BASE["base prompts\nexact XDG generation or current source"] --> ING
```

### Skills → XDG skills library

`agent-utilities install` materializes every current provider into the XDG data tree
and then installs the selected toolkit into detected external agent tools. Each
`skills/<provider>/` root holds immutable content-addressed generations. A closed,
bounded, path-free schema-v2 marker atomically selects one generation and records the
registration digest, content digest, file count, and byte count. Writers are
serialized and build a private, fsynced stage before activation, so readers observe a
complete old or complete new generation.

An unmarked destination is operator-owned even when its name collides with a provider;
installation fails without replacing it. Flat `skills/<skill>/SKILL.md` directories
remain operator-owned and are never pruned. A root containing any provider marker
entry—current, inactive, corrupt, or retired—is never reinterpreted as a flat skill.
Duplicate declared skill identities fail closed. Sources containing links, junctions,
special files, reserved markers, escaping paths, or assets beyond the bounded manifest
limits are rejected. A zero-asset registration is atomically deactivated instead of
retaining its previous content.

At runtime the installed distribution registration and live source manifest are
revalidated against the marker and generation. Package upgrades, tampering, empty
generations, and unresolved sources therefore cannot make old XDG content current;
validated live source is used when available, otherwise that provider is unavailable.

### Prompts → KG prompt library

`ingest_prompts_to_graph()` ingests prompts in precedence order (later overrides
earlier on the namespaced id):

1. validated base — exact current XDG generation or packaged source → `prompt:<name>`
2. fleet-contributed — each `prompt_providers` dir → `prompt:<provider>/<name>`
3. operator overlay — `prompts_dir()` (`~/.config/agent-utilities/prompts/`)

## The canonical prompt schema

The single source of truth is the Pydantic model
`agent_utilities.prompting.structured.StructuredPrompt`. The body lives in
**`instructions.core_directive`**. Alternate flat body keys are rejected. The
canonical fields are `schema_version`, `prompt_version`, `source`,
`skills`, `extends` (+ `compose`). One resolver `resolve_body()` and one
validator `validate_canonical()` back every consumer:

- the three readers in `prompting/builder.py` + `agent/registry_builder.py`
  (this fixed a real bug where decomposed prompts extracted an **empty** body);
- the `prompt-builder` skill (`build_prompt.py` / `validate_prompt.py`);
- the CI gate `scripts/check_prompt_schema.py` + generated
  `prompting/prompt.schema.json` (`scripts/gen_prompt_schema.py`);
- per-package/scaffold parity tests.

A package prompt sets `extends: "agent-utilities:base"` + `compose: append` to
inherit the base prompt at render time (`build_system_prompt_from_workspace`).

## Authoring / scaffolding

`agent-package-builder` now scaffolds the whole contribution (canonical
`prompts/main_agent.json`, a starter skill, entry-points, package-data,
MANIFEST). `prompt-builder` authors/validates individual prompts. Existing
packages are brought up to standard idempotently by
`scripts/retrofit_fleet_contribution.py`.

## Keep-lean guarantee

Assets live in each contributor's wheel; the hub only resolves + indexes them.
Adding the Nth provider adds zero bytes to `agent-utilities`/`universal-skills`,
no new hub dependencies (stdlib `importlib.metadata`), and the heavy ML deps of
any agent never reach the hub serving path. Resolution is non-importing: the hub
accepts only selected files listed by the entry point's owning distribution
(`RECORD`/distribution file manifest) and then applies the bounded regular-file
manifest contract. An injected but unrecorded file fails closed.
