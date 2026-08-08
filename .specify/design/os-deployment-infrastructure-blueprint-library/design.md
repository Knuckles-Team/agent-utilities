# Design Document: Model operator-deployed company infrastructure (software + reusable deployment blueprints) as first-class typed KG nodes, instead of tracking it only as unstructured compose files / inventory sheets outside the graph

CONCEPT:AU-OS.deployment.infrastructure-blueprint-library

> `agent_utilities/models/company.py:315-338` (`DeploymentBlueprint(RegistryNode)`);
> `agent_utilities/models/knowledge_graph.py:390-391` (registry node-type
> constants); OWL mapping `agent_utilities/knowledge_graph/ontology_company_infra.ttl`
> (`:DeploymentBlueprint` class, lines 41-329); round-trip test
> `tests/unit/knowledge_graph/test_company_models.py:303-310`.

## Decision — `DeploymentBlueprint` (and its companion `CompanySoftware`) is a `RegistryNode` subclass with a typed schema (blueprint name/path, deployment mode, resource requirements, the `CompanySoftware` node it deploys), mapped to an OWL class in `ontology_company_infra.ttl`, so reusable Docker Swarm/Compose blueprints for company software are queryable/graph-native objects rather than plain files

The rest of "Company Operations" — goals, KPIs, licenses, governance docs,
regulatory filings — is already modeled as KG-native `RegistryNode` types under
`AU-KG.domains.company-operations`, giving the platform one consistent way to
query, relate, and reason across every facet of how the operator's company
actually runs. `DeploymentBlueprint` extends that same pattern to infrastructure:
a blueprint records what it deploys (`deploys_software_id`, linking to a
`CompanySoftware` node), how (`deployment_mode`: compose/swarm/kubernetes/native),
and its resource footprint (memory/CPU/GPU minimums) — as a typed, ontology-backed
object the rest of the KG (blast-radius queries, `MANAGED_BY_AGENT` edges, the
synergy engine's cross-pillar bridges) can reference the same way it references
any other registry node, rather than as free-text in a compose file the graph has
no visibility into.

## Rejected alternative — keep infrastructure/deployment metadata OUT of the KG entirely, in the workspace's existing unstructured `infrastructure/`/`inventory/` files

The workspace already has a place for this information that is NOT the KG:
Docker Compose manifests under `infrastructure/` and the asset catalog under
`inventory/` (see the workspace root map). The alternative to modeling
`DeploymentBlueprint` at all was leaving deployment blueprints there permanently —
files a human reads, not graph-queryable objects. That keeps the ontology smaller
and avoids a schema that has to be kept in sync with what is actually deployed, at
the cost of making "which blueprint deploys this software, and what does it need"
unanswerable by any KG query — the operator (or an agent) has to go read compose
files directly instead of asking the graph. Given the platform's stated design
philosophy of treating "everything is a node in the graph," extending that same
`RegistryNode` shape to infrastructure blueprints was chosen over carving out a
permanent KG-blind spot for company deployment metadata specifically.

## Risk Assessment

- **Blast Radius**: `agent_utilities/models/company.py`,
  `agent_utilities/models/knowledge_graph.py` (registry node/edge type
  constants), `agent_utilities/knowledge_graph/ontology_company_infra.ttl`.
- **Backward Compatible**: Yes — an additive schema/ontology extension; no
  existing node types change shape.
- **Known weak point**: as of this writing the model is schema-and-ontology-only —
  grep across the tree finds no code path that actually constructs a
  `DeploymentBlueprint` from real provisioning activity (the compose/swarm
  manifests under `infrastructure/`), only the Pydantic model, its OWL mapping,
  and one unit test that round-trips the model's own fields. The decision to model
  this domain is real and deliberate; wiring it to the actual deployment pipeline
  so blueprints are populated from live infrastructure (rather than constructed by
  hand) is not yet done.
