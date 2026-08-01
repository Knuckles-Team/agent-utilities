# Design Document: Model profiles are first-class, content-addressed graph resources — with an explicit preview/persist split

CONCEPT:AU-KG.ontology.model-profile-graph-resource

> `agent_utilities/models/knowledge_graph.py:80-83`,
> `agent_utilities/mcp/tools/ontology_tools.py:475-499`,
> `agent_utilities/gateway/ontology_api.py:390-413`.

## Decision — a provider+model's capability/cost/observability contract is a versioned `ArtifactVersionNode`, same as a skill or spec version

`models/knowledge_graph.py:80-83` places `MODEL_PROFILE = "model_profile"`
directly alongside `SKILL_VERSION` and `SPEC_VERSION` in the same enum, with
the comment stating the decision plainly: "Model profiles as first-class graph
resources ... a provider+model's capability/cost/observability contract,
content-addressed like every other `ArtifactVersionNode`." Making a model
profile an `ArtifactVersionNode` rather than plain config means it inherits the
same content-addressing, versioning, and queryability every other artifact
kind gets — a downstream decision point (cost governance, `ActionPolicy`
model-routing) can query "what model profiles exist / what changed" through the
graph rather than reading a separate config file with no version history.

**The rejected alternative is auto-writing on every read.** `ontology_model_profile`
(`ontology_tools.py:475-499`) deliberately splits `'list'` (preview: "the
configured registry's models as profile previews (**no KG write**)") from
`'sync'` (explicit: "upserts a `ModelProfileVersionNode` **per configured
model** into the KG"). A caller who only wants to see what models are
configured never triggers a graph write; a graph write happens only on the
explicit `sync` action. The alternative — writing a node on every list/preview
— would turn every read-only inspection into a mutation and make "how many
model-profile versions exist" depend on how many times someone happened to
list them, rather than on deliberate sync calls.

## Risk Assessment

- **Blast Radius**: `models/knowledge_graph.py` (`RegistryNodeType` enum),
  `mcp/tools/ontology_tools.py` (`ontology_model_profile` tool),
  `gateway/ontology_api.py` (`/ontology/model-profiles*` routes).
- **Backward Compatible**: Yes — `list` behavior (no write) is unchanged;
  `sync` is an additive explicit action.
- **Known weak point**: nothing calls `sync` automatically when the model
  registry's configuration changes — the graph's model-profile nodes can drift
  out of date relative to the actually-configured registry until something
  explicitly re-syncs.
