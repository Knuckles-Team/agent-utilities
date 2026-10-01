# AU-FREEZE-001 — Freeze authoritative AU release evidence

**Owner:** agent-utilities for its own release input and evidence gate. **Status:** SPECIFIED; acceptance NOT_AUDITED. See [requirements.md](requirements.md) for the definition of every requirement ID and [status.json](status.json) for its delivery state and evidence. Epistemic-graph must independently pin and attest its own source, artifacts, schema and runtime evidence. A shared program view can aggregate signed receipts but does not replace either repository's authority.

## User outcome

A reviewer can identify the exact AU source, dependency lock, generated client contract, scanner results, release artifact and observed runtime corresponding to one candidate, and reject a mixture of revisions or stale observations. A public contributor can produce source-only evidence on a disposable runner; live production qualification is an explicit later profile.

## Requirements

| ID | Observable requirement | Acceptance |
|---|---|---|
| FR-1 | Generate a canonical, machine-readable freeze manifest from an exact Git commit with repository URL, tree hash, version, tracked source digest, lock/manifest hashes, generated EG client digest, build toolchain and timestamp. | Regeneration at the same commit is byte-stable except a separately declared observation timestamp; dirty tree or unknown input fails. |
| FR-2 | Bind build artifact digest and each mandatory scanner/test result to that source manifest and immutable CI run URL. | Hash/signature verification and exact SHA agreement pass; missing, stale, failed or mismatched result blocks qualification. |
| FR-3 | Record runtime/schema/data/consumer observations as separately signed or attested snapshots with observed_at, environment class, object identity and expiry. | A source-only PR can be reviewed without ambient live access; release qualification refuses absent/expired live evidence. |
| FR-4 | Make the freeze fail closed if AU and EG contract digests or published artifact metadata disagree. | Compatibility check proves both independently pinned authorities describe the same contract; no manually copied digest or fallback. |

This spec owns the AU freeze operation, not deployment authorization. No row becomes LANDED from a local manifest; acceptance requires exact merged commit and checked-in or durable public CI receipts.
