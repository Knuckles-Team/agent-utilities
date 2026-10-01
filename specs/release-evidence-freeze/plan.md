# AU-FREEZE-001 — Design

## Existing wiring and source of truth

Reuse `scripts/source_freeze_gate.py`, `scripts/release/generate_component_evidence.py`, `scripts/release/check_compatibility.py`, `scripts/release/assemble_manifest.py` and the checked-in `deploy/release/source-freeze-*.schema.json` plus gate declaration. `agent_utilities/_version.py` and the lock are existing version authorities. Do not introduce a second release manifest, scanner runner or handwritten EG contract hash.

## Manifest and binding

The AU source-freeze input is canonical JSON (sorted keys, UTF-8, no nondeterministic fields) with `schema_version`, repository URL, commit/tree SHA, version, source digest, lock digest, generated-client contract digest, toolchain versions and referenced CI run. A separate observation envelope adds `observed_at`, kind, actor, environment class, artifact digest, evidence URI and attestation identity. CI builds from a clean checkout, computes artifact SHA-256, then signs or attests a relation from source manifest digest to artifact digest and test/scanner run IDs. Validators recompute all local hashes and reject duplicate JSON keys, path traversal, unknown required fields, wrong repo, stale run or changing source after gate execution.

## Cross-repository and live evidence

EG's independently produced freeze receipt supplies the matching generated schema/API digest and its own artifact/engine evidence. The AU compatibility gate compares contract digests and pinned API version; AU never copies an EG source tree or assumes a sibling checkout. Runtime observations reference exact deployed artifact digest, schema/data identity, principal/tenant scope and capture time. A missing live environment produces `NOT_RUN` for release qualification, not a fabricated pass; pure source/contract CI remains runnable on a fresh public runner.

## Failure, security and quality

Only trusted CI/maintainer attestation identities can qualify release evidence. Source payloads cannot mint an attestation. Revoked/expired signatures, changed source, failed scanner, absent consumer, mismatched artifact and stale runtime observation fail closed with a specific reason. Reuse current freeze gate and schema validators for KISS. Apply CCCC (`python3 scripts/check_complexity_staged.py`, no new cyclomatic >10 or cognitive >15 functions), Dupehound (`python3 scripts/check_dupehound.py`), and differential jscpd (`python3 scripts/check_duplication.py diff`) under pinned `pyproject.toml` settings. Run `python3 scripts/check_version_consistency.py`, focused release checks and full AU tests. Exact merged SHA, artifact digest and public CI URL are required before acceptance.
