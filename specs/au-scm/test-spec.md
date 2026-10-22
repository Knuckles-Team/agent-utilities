# AU-SCM — Test specification

Governing spec: [spec.md](spec.md).

| Test ID | Requirement | Level | Setup and input | Expected observation | Evidence |
|---|---|---|---|---|---|
| T-001 | `AU-SCM-R001` | Served/integration | One served coding run. One edit to one file. | `source.commit` lands one changeset. `source.land` applies the landing policy. The run reports the changeset id. |  |
| T-002 | `AU-SCM-R001` | Integration | One clean submit. One overlapping second writer. | The first submit lands. The second returns `SOURCE_CONFLICT{paths}`. The loop rebases and retries. The retry lands. |  |
| T-003 | `AU-SCM-R001` | Unit | A conflict that keeps overlapping past the bound of 3 rebases. | The loop refuses with `conflict_retry_exhausted`. It submits no further changeset. |  |
| T-004 | `AU-SCM-R001` | Unit | No registered `ChangesetSubmitter` provider. | The port fails closed. The loop refuses before the first edit. |  |

## Negative and boundary cases

A failed fast gate refuses the submit with zero effects. A
`SOURCE_CONFLICT{paths}` refusal lists every conflicting path. The loop
re-applies the edits only on the re-read head. The bound of 3 rebases
exhausts to a typed refusal. A clean submit writes no second changeset
for the same edits.

## Quality and release proof

Focused: `uv run --frozen python -m pytest
tests/harness/test_changeset_loop.py -q`.
Language-native: ruff, mypy and the repository scanners through
`.config/pre-commit.yaml` and `release.yml`. A source test is not served
acceptance.

## Symbol-level edits in the run loop

| Row | Proving test |
|---|---|
| `AU-SCM-R002` | `tests/harness/test_changeset_loop.py::test_symbol_edit_retries_once_on_stale_base` |
