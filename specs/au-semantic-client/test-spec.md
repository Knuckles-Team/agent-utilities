# Verification matrix

| Behavior | Positive | Refusal |
|---|---|---|
| Source cut | Same RDF dataset meaning and source identity after EG publication; AU asks by IRI | Unknown source, duplicate protected term and stale composed digest fail atomically |
| Generated client | `OntologyInspect`, schema list and validation call the served method with exact types | Wrong contract generation and hand-crafted unknown field fail before write |
| Claims | Agent candidate claim retains provenance and RunSpec digest; EG durable receipt is observed | Claim with forged tenant, missing source or invalid shape is rejected |
| Legacy removal | Public agent execution still reaches current EG method after old import deletion | Import census fails on `rdflib`, `pyshacl`, `owlrl`, `owlready2`, local graph writer or duplicate DTO |
| Memory and retrieval | Scoped current data survives restart through EG and honors invalidation | Cross-tenant read, untrusted historical row, stale cache and owner outage fail closed |

Start from a fresh public checkout: run `python3 scripts/uv_workspace.py doctor`, then targeted Pytest via `python3 scripts/uv_workspace.py run --all-extras pytest <path> -q`. A served test starts its own disposable EG fixture and pins the generated contract version. Run the full Pytest suite, `python3 scripts/check_current_only_contract.py`, `python3 scripts/check_tracked_privacy.py`, `python3 scripts/check_version_consistency.py`, and `agent-utilities lane lease --resource precommit-all-files --operation gate -- python3 scripts/safe_precommit_all_files.py`. If a live external service cannot be provisioned in CI, use a deterministic generated-contract fixture as the PR gate and keep live-path proof as a separately reported release gate.

Record CCCC complexity delta, jscpd and dupehound differential duplicate findings, KISS review, Ruff/mypy, exact commit and hosted CI URL in `evidence.md`. An unrun test has no pass status.
