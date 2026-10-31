---
name: meristem-development
description: Develop and release the meristem harness repository. Use when bumping the version, tagging or publishing a release.
---

# meristem development

Canonical rules live in `AGENTS.md`; this skill repeats the release procedure exactly.

## Spec-driven flow

Run this before any code. Full table and gates: the Spec-driven development section of `AGENTS.md` and the [spec standard](https://github.com/Knuckles-Team/pipelines/blob/main/reference/spec-standard.md).

1. Read `.specify/memory/constitution.md`; create it from pipelines `specs/_template/constitution.md` if absent (`spec-generator`).
2. Specify the feature: `spec-intake-wizard` gathers the inputs, `spec-generator` writes `specs/<stable-id>/spec.md` and `requirements.md`.
3. Clarify: resolve every `[NEEDS CLARIFICATION: ...]` marker (`spec-generator`).
4. Plan: `plan.md` from pipelines `specs/_template/plan.md`, including Build contract, Production caller chain and Cross-repo seams (`spec-generator`).
5. Tasks: `tasks.md` lines `- [ ] T001 [P] [US1] <ROW-ID> ...` (`task-planner`).
6. Analyze: fix the gaps and fill `checklists/requirements.md` (`spec-verifier`).
7. Implement and land with the delivery contract in this skill (`feature-development-pipeline`); release through pipelines `releases/*.yml` bundles.

Never write code for a row that is not SPECIFIED. Feature directories use the stable spec ID, never a numbered prefix. The state is computed from rows, `Spec:` trailers and test bindings; never write `status.json` or a `Status:` line.

## Release

Do not push, tag or publish without explicit scope. Follow in order.

1. Bump first. Version lives in `meristem/_version.py` (pyproject reads it dynamically), `README.md` (`*Version: X.Y.Z*`) and `CHANGELOG.md`; `.bumpversion.cfg` `current_version` must match. Run `bump-my-version`/`bumpversion` with `--no-commit --no-tag --new-version X.Y.Z` (or edit all four by hand), then `grep -rn 'OLD_VERSION' meristem/_version.py README.md CHANGELOG.md .bumpversion.cfg` to confirm none still show the old value. The editable root entry in `uv.lock` carries no version, so `uv lock --check` must pass; refresh it only if it fails. This repo has no `scripts/check_version_consistency.py` and no recorded fixture embeds the version.
2. Merge the bump PR to `main` (rebase merge). Never tag a branch commit.
3. Verify the merged commit `<sha>`: `git show <sha>:meristem/_version.py` shows `__version__ = "X.Y.Z"`, `git show <sha>:.bumpversion.cfg` shows `current_version = X.Y.Z`, and `curl -s -o /dev/null -w '%{http_code}' https://pypi.org/pypi/meristem/X.Y.Z/json` prints `404`.
4. Only then push the tag on that commit: `git tag vX.Y.Z <sha> && git push origin vX.Y.Z`. `release.yml` publishes only from an exact `vX.Y.Z` tag and fails if the tag differs from `_version.py`, but PyPI refuses a rebuilt old version, so verify before tagging.
5. Confirm PyPI has both files: `curl -s https://pypi.org/pypi/meristem/X.Y.Z/json` lists a `bdist_wheel` and an `sdist` in `urls`. `release.yml` runs `uv publish dist/*` after `uv build`, which uploads both; report it if either is missing.
6. Wrong-commit tag: if nothing was published, delete and re-tag (`git push origin :refs/tags/vX.Y.Z`, `git tag -d vX.Y.Z`, then step 4). If anything was published, never reuse the version; cut the next patch from step 1.
