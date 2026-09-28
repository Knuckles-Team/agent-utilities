# EH-338 — Test contract

| ID | Requirements | Setup and action | Expected result |
|---|---|---|---|
| E338-1 | FR-1 | Enumerate tracked AU graph modules and imported symbols against a reviewed owner map. | Every module has one disposition, replacement or retention reason and owner; unknown fails. |
| E338-2 | FR-2 | Invoke one read and one write from a synthetic-tenant AU agent through generated EG client and disposable engine. | Scoped result/citation and durable receipt match documented old behavior; only EG mutates. |
| E338-3 | FR-2 | Run golden graphs through each retired OWL/SPARQL/compute API and native EG counterpart. | Results, proofs, ordering and typed errors match or explicit versioned differences are reviewed. |
| E338-4 | FR-3 | Inspect imports, scripts and dependencies after cutover; invoke real GraphOS→AU agent route. | No duplicate AU owner or dead caller; live path reaches EG once. |
| E338-5 | FR-4 | Mismatched digest, spoofed tenant, denied actor, timeout and unavailable EG. | Fail before side effect or fallback; retryability/receipt is explicit. |

Capture exact merged revision, fixture digest and command output for every row. Full AU tests and CCCC, differential jscpd and Dupehound must pass under the repository's pinned configuration; source-only parity is not served acceptance.
