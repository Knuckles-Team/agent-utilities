# Deliverable matrix

Each row is a separately reviewable deliverable. The owner named after the arrow must supply the replacement before AU deletes its copy. Every row needs a public caller test, refusal test and exact merged-head evidence; a grouped PR does not merge their statuses.

| Item | AU source cut and target behavior | Required acceptance focus |
|---|---|---|
| EH-476 | Remove `gateway/**`; graph-os serves gateway routes/widgets, including unique AU behavior | Route parity, artifact tenancy, schema drift, no AU gateway import |
| EH-477 | Remove `deployment/**` and duplicate scripts; graph-os owns doctor, skill certification and deployment | CLI ownership, dry-run install, no colliding entry points |
| EH-478 | Remove `mcp/kg_server.py`, `mcp/tools/**` and old action manifest; graph-os serves approved intent operations | Every old action mapped or approved drop, authorization parity |
| EH-479 | Remove AU multiplexer family; graph-os owns fleet and OAuth host modules | Per-server attribution, failover, no duplicate fleet state |
| EH-480 | Migrate connector packages A–E to SDK and generated EG client | Import census and real read/write receipt for each package |
| EH-481 | Migrate connector packages F–J | Same per-package transport and scope proof |
| EH-482 | Migrate connector packages K–O | Same per-package transport and scope proof |
| EH-483 | Migrate connector packages P–S | Same proof, including repository manager consumers |
| EH-484 | Migrate connector packages T–Z | Same proof and no last-package AU import |
| EH-485 | Delete AU connector toolkit and HTTP/credential helpers now replaced in SDK | Zero connector imports of AU toolkit, no compatibility alias |
| EH-486 | Delete AU source lifecycle and certification code; SDK owns it | Cursor, conflict, mapping and certification conformance |
| EH-487 | Delete second EG projection and control-plane shim | Generated EG client is sole projection and contract digest |
| EH-488 | Make graph-os import AU solely via `agent_utilities.api` | AST import census and served agent operation |
| EH-489 | Move request identity, error and other server boundary modules to graph-os | Principal/tenant/context preservation and deny tests |
| EH-490 | Move AU `server/**` and A2A/ACP/AG-UI hosting to graph-os | Endpoint and cancellation parity, no AU listener |
| EH-491 | Split `core/config.py`: agent/model policy stays AU; connector settings to SDK; hosting settings to graph-os | Config fixture matrix and no duplicate environment reader |
| EH-492 | Split messaging: graph-os hosts channel adapters, EG owns durable bus/log, AU receives typed events | Delivery, replay, ordering, authorization and outage proof |
| EH-493 | Replace AU graph engine facade with generated EG client | No Python graph authority or fallback path |
| EH-494 | Replace graph-compute and session facades with typed EG calls | Session scope, compute budget, typed error parity |
| EH-495 | Move durable work and queues to EG | Claim fencing, restart, idempotency and tenant isolation |
| EH-496 | Move graph tenancy, topology and admission authority to EG | Cross-tenant denial and authoritative graph selection |
| EH-497 | Retire AU reasoning/analytics copies when EG native coverage is proven | Method-by-method golden and proof parity |
| EH-498 | Split ingestion: EG commits/maps/derives; AU keeps only model-backed candidate claims | Atomic cursor/receipt, failure rollback, no AU writer |
| EH-499 | Move enterprise source synchronization to SDK | Full/delta/reconcile and withdrawal receipt parity |
| EH-500 | Move document, session and feed source adapters to SDK | Typed source envelopes and source checkpoint recovery |
| EH-501 | Move vendor enrichment extractors and sinks to SDK packs | Vendor transport, schema mapping and conflict proof |
| EH-502 | Move deterministic enrichment/derivation to EG | Golden output and no model-dependent graph writer |
| EH-503 | Move remaining graph standardization, security and compute modules to EG | Public method coverage and AU import deletion |
| EH-504 | Move ontology object model and graph parsing to EG | Exact source identity, graph isomorphism and no AU RDF parser |
| EH-505 | Move hand-written RDF/OWL/SHACL emitters behind EG pack compilation | Typed declaration in AU, generated artifact and conflict tests |
| EH-506 | Replace AU graph-schema DTOs with EG-generated types | Exact digest, unknown field rejection, no copied model |
| EH-507 | Delete legacy Fuseki/Stardog SPARQL backend and setup paths | No runtime or CLI caller; typed unavailable response |
| EH-508 | Move other external graph/database federation to EG | Scoped foreign-source query and fail-closed outage |
| EH-509 | Move retrieval engines to EG; AU retains context compilation | Same-snapshot tenant/source proof, cited ranking parity |
| EH-510 | Place drift gate in SDK sync and schema candidate/activation in EG | Quarantine without cursor advance, approved activation |
| EH-511 | Move durable usage facts to EG; AU emits events only | Restart, tenant read and duplicate event proof |
| EH-512 | Move repository/lane governance to repository-manager | Worktree/merge queue recovery and zero AU governance import |
| EH-513 | Move deterministic finance math to EG and feeds/effects to SDK; retain AU roles | Golden calculations, account scope, paper/live separation |
| EH-514 | Generate cross-owner component registry and AU path/script guard | New wrong-owner module/script fails a hermetic PR test |
| EH-515 | Move frontends off AU internals to AU API, graph-os and EG client | Import census and end-to-end user route |
| EH-516 | Move durable agent memory, learning and media store to EG | Trusted migration/quarantine, scoped read and restart |

The status of every row starts at TODO pending an exact merged-head audit. Prior implementation branches may contain useful source, but their existence does not prove landing or acceptance.
