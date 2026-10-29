# Deliverable matrix

Each row is a separately reviewable deliverable. The owner named after the arrow must supply the replacement before AU deletes its copy. Every row needs a public caller test, refusal test and exact merged-head evidence. A grouped PR does not merge their statuses.

| Requirement | AU source cut and target behavior | Required acceptance focus |
|---|---|---|
| MERISTEM-BOUNDARY-R002 | Remove `deployment/**` and duplicate scripts; arbora owns doctor, skill certification and deployment | CLI ownership, dry-run install, no colliding entry points |
| MERISTEM-BOUNDARY-R005 | Migrate connector packages A–E to SDK and generated EG client | Import census and real read/write receipt for each package |
| MERISTEM-BOUNDARY-R006 | Migrate connector packages F–J | Same per-package transport and scope proof |
| MERISTEM-BOUNDARY-R007 | Migrate connector packages K–O | Same per-package transport and scope proof |
| MERISTEM-BOUNDARY-R008 | Migrate connector packages P–S | Same proof, including repository manager consumers |
| MERISTEM-BOUNDARY-R009 | Migrate connector packages T–Z | Same proof and no last-package AU import |
| MERISTEM-BOUNDARY-R010 | Delete AU connector toolkit and HTTP/credential helpers now replaced in SDK | Zero connector imports of AU toolkit, no compatibility alias |
| MERISTEM-BOUNDARY-R011 | Delete AU source lifecycle and certification code; SDK owns it | Cursor, conflict, mapping and certification conformance |
| MERISTEM-BOUNDARY-R012 | Delete second EG projection and control-plane shim | Generated EG client is sole projection and contract digest |
| MERISTEM-BOUNDARY-R014 | Move request identity, error and other server boundary modules to arbora | Principal/tenant/context preservation and deny tests |
| MERISTEM-BOUNDARY-R015 | Move AU `server/**` and A2A/ACP/AG-UI hosting to arbora | Endpoint and cancellation parity, no AU listener |
| MERISTEM-BOUNDARY-R016 | Split `core/config.py`: agent/model policy stays AU. Connector settings to SDK. Hosting settings to arbora | Config fixture matrix and no duplicate environment reader |
| MERISTEM-BOUNDARY-R017 | Split messaging: arbora hosts channel adapters, EG owns durable bus/log, AU receives typed events | Delivery, replay, ordering, authorization and outage proof |
| MERISTEM-BOUNDARY-R018 | Replace AU graph engine facade with generated EG client | No Python graph authority or fallback path |
| MERISTEM-BOUNDARY-R019 | Replace graph-compute and session facades with typed EG calls | Session scope, compute budget, typed error parity |
| MERISTEM-BOUNDARY-R020 | Move durable work and queues to EG | Claim fencing, restart, idempotency and tenant isolation |
| MERISTEM-BOUNDARY-R021 | Move graph tenancy, topology and admission authority to EG | Cross-tenant denial and authoritative graph selection |
| MERISTEM-BOUNDARY-R022 | Retire AU reasoning/analytics copies when EG native coverage is proven | Method-by-method golden and proof parity |
| MERISTEM-BOUNDARY-R023 | Split ingestion: EG commits/maps/derives. AU keeps only model-backed candidate claims | Atomic cursor/receipt, failure rollback, no AU writer |
| MERISTEM-BOUNDARY-R024 | Move enterprise source synchronization to SDK | Full/delta/reconcile and withdrawal receipt parity |
| MERISTEM-BOUNDARY-R025 | Move document, session and feed source adapters to SDK | Typed source envelopes and recovery after an interrupted source read |
| MERISTEM-BOUNDARY-R026 | Move vendor enrichment extractors and sinks to SDK packs | Vendor transport, schema mapping and conflict proof |
| MERISTEM-BOUNDARY-R027 | Move deterministic enrichment/derivation to EG | Golden output and no model-dependent graph writer |
| MERISTEM-BOUNDARY-R028 | Move remaining graph standardization, security and compute modules to EG | Public method coverage and AU import deletion |
| MERISTEM-BOUNDARY-R029 | Move ontology object model and graph parsing to EG | Exact source identity, graph isomorphism and no AU RDF parser |
| MERISTEM-BOUNDARY-R030 | Move hand-written RDF/OWL/SHACL emitters behind EG pack compilation | Typed declaration in AU, generated artifact and conflict tests |
| MERISTEM-BOUNDARY-R031 | Replace AU graph-schema DTOs with EG-generated types | Exact digest, unknown field rejection, no copied model |
| MERISTEM-BOUNDARY-R032 | Delete legacy external SPARQL-store backend and setup paths | No runtime or CLI caller; typed unavailable response |
| MERISTEM-BOUNDARY-R033 | Move other external graph/database federation to EG | Scoped foreign-source query and fail-closed outage |
| MERISTEM-BOUNDARY-R034 | Move retrieval engines to EG; AU retains context compilation | Same-snapshot tenant/source proof, cited ranking parity |
| MERISTEM-BOUNDARY-R035 | Place drift gate in SDK sync and schema candidate/activation in EG | Quarantine without cursor advance, approved activation |
| MERISTEM-BOUNDARY-R036 | Move durable usage facts to EG; AU emits events only | Restart, tenant read and duplicate event proof |
| MERISTEM-BOUNDARY-R037 | Move development-governance tooling (merge queue, concept reservation) to repository-manager | Interrupted merge-queue recovery and zero AU governance import |
| MERISTEM-CONTEXT-R007 | Move deterministic finance math to EG and feeds/effects to SDK. Retain AU roles | Golden calculations, account scope, paper/live separation |
| MERISTEM-BOUNDARY-R038 | Generate cross-owner component registry and AU path/script guard | New wrong-owner module/script fails a hermetic PR test |
| MERISTEM-BOUNDARY-R040 | Move durable agent memory, learning and media store to EG | Trusted migration/quarantine, scoped read and restart |
| MERISTEM-BOUNDARY-R041 | Remove the finance modules that produced fabricated data | Import census: no listed module and no caller remains |
| MERISTEM-BOUNDARY-R049 | Run every AU write-back sink through the SDK governed write-back contract | Canonical change set, audit reservation, ledger receipt and SDK refusal before the sink |

Every requirement ID in this table is defined in [requirements.md](requirements.md) and its delivery state is recorded in [status.json](status.json). A row is delivered only by a commit merged to the default branch. An unmerged implementation branch does not prove landing or acceptance.
