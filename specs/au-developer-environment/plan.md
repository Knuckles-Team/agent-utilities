# Plan

1. Inventory current AU and ecosystem development skills and contributor entry points by source path and public repository URL.
2. Specify a minimal unit profile and a composed disposable fixture. Replace private/ambient prerequisites with declared public packages and synthetic fixtures.
3. Extend the existing universal installer and skill manifest for deterministic dry run, refresh and stale detection; wire the AU development skill to it.
4. Document exact commands in the contribution guide and run them in a fresh-fork CI matrix.
5. Keep release-only live certification separately labelled and record exact merged-head acceptance evidence.

## Design

### Profiles

| Profile | Inputs | Outputs |
|---|---|---|
| Spec and unit | Public AU checkout, lockfile, declared toolchain | Installed spec skills, deterministic unit/contract results; no service needed |
| Composed fixture | Public EG/SDK/GraphOS/AU revisions plus disposable containers | Generated client digest, synthetic identity, served route receipts, teardown log |
| Release certification | Explicit operator-provisioned services and credentials | Separate live-path receipts; never implicit in ordinary PR CI |

The checked-in skill manifest records provider repository URL, source path, version/digest, installed ID and target. The refresh command compares hashes, reports add/update/remove and writes atomically only after validation. Use the existing universal-skills installer rather than a second AU skill installer. The development skill calls existing AU `scripts/uv_workspace.py doctor` and test wrappers; it does not construct its own package resolver. A composite fixture uses the generated EG client and GraphOS public surface; AU receives typed verified context through its API.

Failure is explicit: missing toolchain lists install step; absent optional live profile says not run; stale skill refuses a false current claim; digest mismatch blocks served execution; fixture cleanup always runs. No gate silently skips a required behavioral test.

Quality: CCCC for installer/control-flow complexity, jscpd and dupehound for duplicated bootstrap scripts, KISS for one toolchain manifest and one installer, plus Ruff/mypy/Pytest and normal hooks. A docs publishing workflow may report its own failure but does not gate unrelated code merges.
