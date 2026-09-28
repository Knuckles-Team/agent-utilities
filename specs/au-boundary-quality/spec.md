# Reproducible AU boundary and quality gates — AU-QUAL-01

**Owner:** agent-utilities. **Delivery:** partial; **acceptance:** pending. Work item IDs: EH-176, EH-380, EH-385, EH-386, EH-391. This is a buildable contract for making the Python control plane green from a fresh clone and in a cloud contribution session.

## Actors and outcome

An external contributor can run a focused test and a deterministic PR gate without access to a private network, a preinstalled sibling checkout, a running control plane, or an operator's credentials. A maintainer receives failures that correspond to changed behavior and can distinguish a reproducible source defect from a fixture or provisioning fault. Hosted CI is the authoritative merge check; live deployment certification is a separately named release proof.

## User stories and acceptance

1. **P0 — portable development.** From a fresh clone, the contributor installs declared dependencies and runs unit, contract, and gateway tests. Any required service is created as a disposable fixture by the test; no test silently reaches a real environment.
2. **P0 — honest green.** The `tiny` profile gateway returns correct authorization status and is covered by both success and forbidden action tests. Bootstrap isolation tests patch stable public seams, not deleted functions. Connector packages required by widget/ingest tests are declared and provisioned.
3. **P0 — right boundary.** Connector validation consumes the engine's composed `GraphSchema` and returns typed errors. AU never treats its own shapes pack as the source of schema authority.
4. **P1 — complete type coverage.** All tests enter mypy coverage in reviewable directory batches. Errors are fixed with meaningful types; `Any`, blanket ignores, or fresh exclusions are not a substitute for a resolved contract.
5. **P1 — useful gates.** Time based liveness obligations are checked predictably in hosted CI or a scheduled job; ordinary local commits are not blocked solely because time elapsed. Documentation is published through GitHub Pages; a stale `/docs` directory or a documentation deployment gate cannot block unrelated code. Privacy, security, functional, and duplication checks remain actionable and runnable in cloud sessions.

## Functional requirements

| ID | Requirement | Source | Acceptance evidence |
|---|---|---|---|
| QUAL-01 | Provide declared, reproducible Python/tool install and disposable service fixtures for every required PR test. | EH-380, EH-386 | fresh clone CI matrix |
| QUAL-02 | Fix `tiny` profile RBAC error propagation and bootstrap isolation coverage at real gateway entry points. | EH-380 | positive/negative served tests |
| QUAL-03 | Use composed engine `GraphSchema` for connector validation; remove AU shape authority. | EH-385 | contract and schema drift tests |
| QUAL-04 | Repair placement mining, trace miner, claim flywheel, ingestion, observability, widget and assimilation benchmark failures with causal tests rather than skips. | EH-386 | exact test list and run |
| QUAL-05 | Include `tests/` in mypy and resolve its errors in small owned batches without new ignores, casts to `Any`, or exclusions. | EH-391 | mypy output and config diff |
| QUAL-06 | Give liveness deferrals a visible owner and review date; expired entries fail a deterministic hosted/scheduled check with a useful message. | EH-176, EH-380 | time frozen unit tests + CI run |
| QUAL-07 | Keep PR gates hermetic and proportional: provisionable fixtures, source/contract quality, privacy and duplication. Require live stack proof only for release or a spec that explicitly needs it. | EH-386 | CI workflow + fresh fork run |

An unrelated documentation deployment or ambient environment check cannot be a merge blocker. Removing such a blocker does not waive a test that detects behavior, authorization, privacy or source duplication. Required tests must identify the prerequisite they provision and clean it up afterward.

## Edge cases and limits

An absent optional test tool reports its setup problem clearly; it does not claim success. A service fixture binds only loopback and random ports, isolates tenants, and leaves no credentials. Expired liveness entries remain visible even when no code changes. Mypy activation must not suppress preexisting source errors. Benchmark randomness must be seeded at its owning engine/client seam, not by relaxing expected results. A GitHub Pages build may fail its own publishing job, but cannot silently rewrite functional test status.
