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

The hub resolves each entry-point to the contributor's installed data directory
via `importlib.resources` — it imports only the named data subpackage (no heavy
deps), never the agent's business logic. The single resolver is
`agent_utilities.core.providers.iter_provider_dirs(group)`. Discovery is
failure-isolated: an uninstalled/broken provider is skipped, never fatal.

```mermaid
flowchart TD
    subgraph pkg["agent-package wheel (servicenow-api)"]
        SK["servicenow_api/skills/**/SKILL.md"]
        PR["servicenow_api/prompts/*.json"]
        EP["pyproject entry-points:\nskill_providers / prompt_providers"]
    end

    EP -. importlib.metadata .-> RES["core.providers.iter_provider_dirs()"]

    subgraph hub["agent-utilities / universal-skills (lean hub)"]
        RES --> INST["skill-installer get_source_paths()"]
        RES --> ING["registry_builder.ingest_prompts_to_graph()"]
    end

    SK --> INST
    PR --> ING
    INST --> XDG["~/.config/agent-utilities/skills/\n(+ every detected agent tool)"]
    ING --> KG[("KG prompt library\nPromptNode prompt:&lt;pkg&gt;/&lt;name&gt;")]
    OVL["~/.config/agent-utilities/prompts/\n(operator XDG overlay)"] --> ING
    BASE["agent_utilities/prompts/*.json\n(packaged base)"] --> ING
```

### Skills → XDG skills library

`install-skills` (`universal_skills/core/skill_installer`) walks every
`agent_utilities.skill_providers` entry-point, rglobs `SKILL.md` under each
provider, applies the existing `--skills`/`--group`/`--layer`/`--install-skill-graphs`
gates, de-dups, and installs (copy or `--symlink`) into every detected agent
tool — including `~/.config/agent-utilities/skills/`. Provider skill-graphs
(under a `skill-graphs/` path segment) route into the `skill-graphs/` subfolder.

### Prompts → KG prompt library

`ingest_prompts_to_graph()` ingests prompts in precedence order (later overrides
earlier on the namespaced id):

1. packaged base — `agent_utilities/prompts/*.json` → `prompt:<name>`
2. fleet-contributed — each `prompt_providers` dir → `prompt:<provider>/<name>`
3. operator overlay — `prompts_dir()` (`~/.config/agent-utilities/prompts/`)

## The canonical prompt schema

The single source of truth is the Pydantic model
`agent_utilities.prompting.structured.StructuredPrompt`. The body lives in
**`instructions.core_directive`**; `content`/`input` are migration-only legacy
keys. New canonical fields: `schema_version`, `prompt_version`, `source`,
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
no new hub dependencies (stdlib `importlib.metadata`/`importlib.resources`), and
the heavy ML deps of any agent never reach the hub serving path.
