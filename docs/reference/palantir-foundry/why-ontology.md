# Why Ontology

> Source: <https://www.palantir.com/docs/foundry/ontology/why-ontology/>
> Captured during the ontology-parity effort. Concrete feature taxonomy only — marketing language elided.

## Decision-centric model

Foundry frames the Ontology around four integrated dimensions of an operational decision:

- **Data** — information sources plus the *decision data* an operation itself generates (context, options evaluated, implications).
- **Logic** — reasoning, algorithms, ML models, optimization/simulation, and deterministic functions complementing non-deterministic LLM reasoning.
- **Action** — execution/orchestration of the chosen decision (the "verbs").
- **Security** — policy enforcement and access control applied consistently across all of the above.

## Data architecture

- Integrates enterprise sources (ERP, MES, WMS, IoT, edge), unstructured repositories, and geospatial datastores into real-time object / property / link representations.
- **Data capture**: user-generated edits during workflows; decision lineage (when, against which data version, through which application); embedded Ontology for edge-device decisions.

## Logic binding

- "Flexible logic binding paradigm" — heterogeneous logic assets (on-prem, cloud, SaaS, platform-native) exposed behind consistent interfaces.
- Assets: transactional business logic, ML models, optimization/simulation algorithms, deterministic functions.
- Logic becomes agent-accessible **tools** with a controlled invocation scope.

## Action model

- Actions are the **semantic representation of enterprise operations** (the verbs).
- Scenario-based staging for safe exploration; governed execution with granular access control; writeback to transactional systems, edge devices, and custom apps.
- Batch-staged operations with human review; conditional automation with graduated agent autonomy; change/release-management integration.

## Security architecture

- Marking-, purpose-, and role-based policies; row- and column-level restrictions.
- Dynamic runtime computation combining security markings with user attributes.
- Tool-usage enforcement is dependent on the underlying data access (no privilege escalation via a tool).
- Markings applied consistently to logs and memory artifacts; logging accessibility controlled per project/workflow.

## Ontology lifecycle components

- **Definition / modeling**: object types, link types, properties, interfaces, structs, value types.
- **Evolution / branching**: proposal-review workflows, branching framework for safe modification, schema migration, shared ontology across teams.
- **Consumption**: Object Explorer (search/analyze), Object Views (standard/full/panel/legacy), Vertex (graph viz + scenarios), Workshop / Map / Rules apps.
