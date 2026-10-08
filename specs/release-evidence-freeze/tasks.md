# AU-FREEZE-001 — Tasks

- [ ] Inventory current freeze schemas, manifest/lock/version authorities and release gate callers.
- [ ] Make AU source input deterministic and bind artifact plus mandatory test/scanner receipts to its digest.
- [ ] Compare pinned AU generated client contract with independent EG contract receipt.
- [ ] Add separate, expiring runtime/schema/data/consumer observation envelopes and attestation verification.
- [ ] Test dirty, stale, mismatched, missing and revoked evidence plus a clean public-runner profile.
- [ ] Keep an unremediated, unmerged scanner finding quarantined and separate from the frozen evidence set so it cannot count as a passing mandatory scanner result; add a qualification test confirming the freeze blocks while the finding stays open. Closes AU-FREEZE-R002.
- [ ] Run focused/full tests and CCCC, jscpd, Dupehound, KISS review; publish exact merged-head and release receipts before status change.
- [ ] Install the engine in `gates` from the `engine-main` release and keep the PyPI check advisory. Record the first run with the release wheel and the failing-set delta against main. Extend the same install to `numeric-runtime-gate` once Windows wheels publish. Closes AU-FREEZE-R003.
