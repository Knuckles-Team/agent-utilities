# AU-SCM — Requirements

Governing spec: [spec.md](spec.md).

Each row is one requirement. A child row `.n` is one five-minute unit of work. A row names the file it touches and the test that proves it. The generator sets each row's state in `status.json`.

| ID | Requirement | Acceptance |
|---|---|---|
| `AU-SCM-R001` | **Development loop on changesets.** The coding harness edits in a virtual workspace. It runs fast gates in its sandbox. It submits through the repository-manager connector tools (`source.commit`/`source.land`). It reacts to `SOURCE_CONFLICT{paths}` (EG-SCM-R003) by rebasing on the new head and retrying. Local git is not required. File it touches: `agent_utilities/harness/changeset_loop.py`. Planned test: `tests/harness/test_changeset_loop.py::test_changeset_loop_rebases_on_conflict`. | One edit cycle lands one changeset through `source.commit`. A `SOURCE_CONFLICT{paths}` refusal triggers one rebase and one retry. The loop holds no store and no local git state. |
