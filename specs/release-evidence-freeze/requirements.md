# AU-FREEZE-001 requirements

| ID | Requirement | Verification |
|---|---|---|
| `AU-FREEZE-R001` | **Canonical AU release-evidence freeze manifest.** AU generates a canonical, machine-readable freeze manifest from one exact Git commit, covering repository URL, tree hash, version, tracked source digest, lock and manifest hashes, the generated EG client digest, build toolchain and timestamp, and binds it to the build artifact digest and each mandatory scanner and test result together with its immutable CI run URL. | Regenerating the manifest at the same commit produces byte-identical output except for a separately declared observation timestamp, and a dirty working tree or an unknown input fails generation. |
| `AU-FREEZE-R002` | **Quarantine unresolved scanner findings from the freeze.** A scanner finding that has not been remediated and merged is kept quarantined, separate from the frozen evidence set, so an unresolved finding cannot be counted as a passing mandatory scanner result. | A qualification check confirms the freeze blocks when an in-scope candidate still has a quarantined, unresolved scanner finding. |
