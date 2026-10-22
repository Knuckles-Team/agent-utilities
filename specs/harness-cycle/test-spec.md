# AU-HARNESS-CYCLE — Test specification

Governing spec: [spec.md](spec.md).

| Test ID | Requirement | Level | Setup and input | Expected observation | Evidence |
|---|---|---|---|---|---|
| T-001 | `AU-HARNESS-CYCLE-R001.9` | Served/integration | One served task, two candidates, k = 2 | The graph shows the adopted version, the contrast and the lesson. The next propose returns the lesson. |  |
| T-002 | `AU-HARNESS-CYCLE-R001.1` | Unit | A candidate and an outcome fixture | A dataclass round trip keeps every field |  |
| T-003 | `AU-HARNESS-CYCLE-R001.2` | Unit | A library answer of two versions | One candidate per version, and no write |  |
| T-004 | `AU-HARNESS-CYCLE-R001.3` | Unit | Two candidates, one shared seed and budget | The same conditions digest, one trace per candidate |  |
| T-005 | `AU-HARNESS-CYCLE-R001.4` | Unit | Probe traces and the contrast | One `Decide` call per tick, and no ranking code |  |
| T-006 | `AU-HARNESS-CYCLE-R001.5` | Unit | An adopted candidate | Three envelopes and one `apply_edits` call |  |
| T-007 | `AU-HARNESS-CYCLE-R001.6` | Unit | An adopted outcome and a neutral outcome | One lesson on adoption, and none on abstention |  |
| T-008 | `AU-HARNESS-CYCLE-R001.7` | Unit | Each refusal input | The typed code, and the current harness kept |  |
| T-009 | `AU-HARNESS-CYCLE-R001.8` | Unit | No registered provider | The port raises the closed failure |  |
| T-011 | `AU-HARNESS-CYCLE-R001.10` | Served | An adopted candidate on the released EG client | Three envelopes that a graph read returns |  |
| T-012 | `AU-HARNESS-CYCLE-R001.11` | Served | A host-registered provider | One tick and one run event |  |

## Negative and boundary cases

An empty library answer refuses with `no_candidates`. A lost probe
trace refuses with `probe_trace_missing`. An engine abstention refuses
with `compose_abstention` and persists no adoption. An allocation under
two candidate slots refuses with `budget_below_contrast`. A rerun of
the same tick writes no second lesson for the same contrast.

## Quality and release proof

Focused: `uv run --frozen python -m pytest tests/harness/test_harness_cycle.py -q`.
Served: `uv run --frozen python -m pytest tests/harness/test_harness_cycle_served.py -q`.
Language-native: ruff, mypy and the repository scanners through
`.config/pre-commit.yaml` and `release.yml`. A source test is not served
acceptance.
