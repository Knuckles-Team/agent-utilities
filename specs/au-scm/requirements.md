# AU-SCM — Requirements

Governing spec: [spec.md](spec.md).

Each row is one requirement. A child row `.n` is one five-minute unit of work. A row names the file it touches and the test that proves it. The generator sets each row's state in `status.json`.

| ID | Requirement | Acceptance |
|---|---|---|
| `AU-SCM-R001` | **Development loop on changesets.** The coding harness edits in a virtual workspace. It runs fast gates in its sandbox. It submits through the repository-manager connector tools (`source.commit`/`source.land`). It reacts to `SOURCE_CONFLICT{paths}` (EG-SCM-R003) by rebasing on the new head and retrying. Local git is not required. File it touches: `agent_utilities/harness/changeset_loop.py`. Planned test: `tests/harness/test_changeset_loop.py::test_changeset_loop_rebases_on_conflict`. | One edit cycle lands one changeset through `source.commit`. A `SOURCE_CONFLICT{paths}` refusal triggers one rebase and one retry. The loop holds no store and no local git state. |
| `AU-SCM-R002` | **Symbol-level edits in the run loop.** The coding harness must submit a code change as a symbol edit through `source.edit_symbol` (ARBORA-OPS-R848.4, EG-SCM-R013) when the file has a parser. It names the symbol, the anchor span digest and the new text. It applies the engine answer and computes no projection. On `EDIT_BASE_STALE` it reads the symbol at the new head and retries once. On `EDIT_ROUNDTRIP_MISMATCH` it returns the refusal to the model as a step observation. File it touches: `agent_utilities/harness/changeset_loop.py`. | `tests/harness/test_changeset_loop.py::test_symbol_edit_retries_once_on_stale_base` lands one edit after one stale refusal. The loop holds no projection code. |
