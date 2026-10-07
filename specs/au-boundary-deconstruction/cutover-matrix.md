# Deliverable matrix

Each row is a separately reviewable deliverable. The owner named after the arrow must supply the replacement before AU deletes its copy. Every row needs a public caller test, refusal test and exact merged-head evidence; a grouped PR does not merge their statuses.

| Requirement | AU source cut and target behavior | Required acceptance focus |
|---|---|---|
| AU-BOUNDARY-R001 | Remove `gateway/**`; graph-os serves gateway routes/widgets, including unique AU behavior | Route parity, artifact tenancy, schema drift, no AU gateway import |
| AU-BOUNDARY-R002 | Remove `deployment/**` and duplicate scripts; graph-os owns doctor, skill certification and deployment | CLI ownership, dry-run install, no colliding entry points |
| AU-BOUNDARY-R003 | Remove `mcp/kg_server.py`, `mcp/tools/**` and old action manifest; graph-os serves approved intent operations | Every old action mapped or approved drop, authorization parity |
| AU-BOUNDARY-R004 | Remove AU multiplexer family; graph-os owns fleet and OAuth host modules | Per-server attribution, failover, no duplicate fleet state |
| AU-BOUNDARY-R005 | Migrate connector packages A–E to SDK and generated EG client | Import census and real read/write receipt for each package |
| AU-BOUNDARY-R006 | Migrate connector packages F–J | Same per-package transport and scope proof |
| AU-BOUNDARY-R007 | Migrate connector packages K–O | Same per-package transport and scope proof |
| AU-BOUNDARY-R008 | Migrate connector packages P–S | Same proof, including repository manager consumers |
| AU-BOUNDARY-R009 | Migrate connector packages T–Z | Same proof and no last-package AU import |
| AU-BOUNDARY-R010 | Delete AU connector toolkit and HTTP/credential helpers now replaced in SDK | Zero connector imports of AU toolkit, no compatibility alias |
| AU-BOUNDARY-R011 | Delete AU source lifecycle and certification code; SDK owns it | Cursor, conflict, mapping and certification conformance |
| AU-BOUNDARY-R012 | Delete second EG projection and control-plane shim | Generated EG client is sole projection and contract digest |
| AU-BOUNDARY-R013 | Make graph-os import AU solely via `agent_utilities.api` | AST import census and served agent operation |
| AU-BOUNDARY-R014 | Move request identity, error and other server boundary modules to graph-os | Principal/tenant/context preservation and deny tests |
| AU-BOUNDARY-R015 | Move AU `server/**` and A2A/ACP/AG-UI hosting to graph-os | Endpoint and cancellation parity, no AU listener |
| AU-BOUNDARY-R016 | Split `core/config.py`: agent/model policy stays AU; connector settings to SDK; hosting settings to graph-os | Config fixture matrix and no duplicate environment reader |
| AU-BOUNDARY-R017 | Split messaging: graph-os hosts channel adapters, EG owns durable bus/log, AU receives typed events | Delivery, replay, ordering, authorization and outage proof |
| AU-BOUNDARY-R018 | Replace AU graph engine facade with generated EG client | No Python graph authority or fallback path |
| AU-BOUNDARY-R019 | Replace graph-compute and session facades with typed EG calls | Session scope, compute budget, typed error parity |
| AU-BOUNDARY-R020 | Move durable work and queues to EG | Claim fencing, restart, idempotency and tenant isolation |
| AU-BOUNDARY-R021 | Move graph tenancy, topology and admission authority to EG | Cross-tenant denial and authoritative graph selection |
| AU-BOUNDARY-R022 | Retire AU reasoning/analytics copies when EG native coverage is proven | Method-by-method golden and proof parity |
| AU-BOUNDARY-R023 | Split ingestion: EG commits/maps/derives; AU keeps only model-backed candidate claims | Atomic cursor/receipt, failure rollback, no AU writer |
| AU-BOUNDARY-R024 | Move enterprise source synchronization to SDK | Full/delta/reconcile and withdrawal receipt parity |
| AU-BOUNDARY-R025 | Move document, session and feed source adapters to SDK | Typed source envelopes and recovery after an interrupted source read |
| AU-BOUNDARY-R026 | Move vendor enrichment extractors and sinks to SDK packs | Vendor transport, schema mapping and conflict proof |
| AU-BOUNDARY-R027 | Move deterministic enrichment/derivation to EG | Golden output and no model-dependent graph writer |
| AU-BOUNDARY-R028 | Move remaining graph standardization, security and compute modules to EG | Public method coverage and AU import deletion |
| AU-BOUNDARY-R029 | Move ontology object model and graph parsing to EG | Exact source identity, graph isomorphism and no AU RDF parser |
| AU-BOUNDARY-R030 | Move hand-written RDF/OWL/SHACL emitters behind EG pack compilation | Typed declaration in AU, generated artifact and conflict tests |
| AU-BOUNDARY-R031 | Replace AU graph-schema DTOs with EG-generated types | Exact digest, unknown field rejection, no copied model |
| AU-BOUNDARY-R032 | Delete legacy Fuseki/Stardog SPARQL backend and setup paths | No runtime or CLI caller; typed unavailable response |
| AU-BOUNDARY-R033 | Move other external graph/database federation to EG | Scoped foreign-source query and fail-closed outage |
| AU-BOUNDARY-R034 | Move retrieval engines to EG; AU retains context compilation | Same-snapshot tenant/source proof, cited ranking parity |
| AU-BOUNDARY-R035 | Place drift gate in SDK sync and schema candidate/activation in EG | Quarantine without cursor advance, approved activation |
| AU-BOUNDARY-R036 | Move durable usage facts to EG; AU emits events only | Restart, tenant read and duplicate event proof |
| AU-BOUNDARY-R037 | Move development-governance tooling (merge queue, concept reservation) to repository-manager | Interrupted merge-queue recovery and zero AU governance import |
| AU-CONTEXT-R007 | Move deterministic finance math to EG and feeds/effects to SDK; retain AU roles | Golden calculations, account scope, paper/live separation |
| AU-BOUNDARY-R038 | Generate cross-owner component registry and AU path/script guard | New wrong-owner module/script fails a hermetic PR test |
| AU-BOUNDARY-R039 | Move frontends off AU internals to AU API, graph-os and EG client | Import census and end-to-end user route |
| AU-BOUNDARY-R040 | Move durable agent memory, learning and media store to EG | Trusted migration/quarantine, scoped read and restart |
| AU-BOUNDARY-R041 | Remove the finance modules that produced fabricated data | Import census: no listed module and no caller remains |

Every requirement ID in this table is defined in [requirements.md](requirements.md) and its delivery state is recorded in [status.json](status.json). A row is delivered only by a commit merged to the default branch; an unmerged implementation branch does not prove landing or acceptance.
