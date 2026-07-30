# Development workspace bootstrap

Given a Git URL and target directory, bootstrap a reproducible multi-repository
workspace without assuming this operator’s paths or private origins.

## Procedure

1. Clone the supplied bootstrap repository at an explicit revision.
2. Read its README, AGENTS instructions, and release/provenance metadata.
3. Locate the canonical root `workspace.yml`; do not substitute a packaged seed when
   the root manifest exists.
4. Validate manifest syntax and resolve environment-backed origins without logging
   their values.
5. Install `repository-manager` from the pinned source or approved release.
6. Preview the dependency-closed, filtered repository set and require that every
   selected private repository has a resolvable origin reference.
7. Run the parallel clone path:

   ```bash
   repository-manager \
     --file <workspace-root>/workspace.yml \
     --workspace <workspace-root> \
     --threads <bounded-parallelism> \
     --clone
   ```

   Use `--repositories` for a named subset. Choose thread count from disk and network
   measurements; more cloning can reduce throughput on slow storage.
8. Mirror the unchanged manifest into the configured Graph-OS XDG location and the
   repository-manager packaged seed only when the ecosystem contract requires it.
   Verify that all copies have the same digest.
9. Install selected repositories in dependency order, using editable installs only
   for development. Keep each repository’s own lockfile and validation command.
10. Invoke `agent-utilities-deployment` with `deployment_profile: development` and
    the resolved component set.
11. After Graph-OS health passes, delta-ingest the checked-out sources and verify
    repository/skill/tool/prompt/ontology discovery.

## Selection

The ideal manifest records repository visibility, profiles, capabilities,
dependencies, optionality, install/build commands, and validation commands. Until
those selectors are present, use an explicit allow-list and report unresolved
private origins rather than guessing or silently dropping repositories.

## Checkpointing

Checkpoint after manifest resolution, clone fan-out, dependency install, service
startup, and ingestion. Record repository, revision, target path reference, status,
duration, and failure reason. On resume, hash-check completed clones and retry only
failed or changed work.

## Exit gates

- selected set is dependency-closed;
- manifest digests match;
- every clone is at its declared revision;
- no credentials appear in remotes, logs, or generated files;
- editable installs point to the intended checkouts;
- repository validations pass;
- the delegated runtime reaches the same backend as production entrypoints;
- a second bootstrap is a no-op except for declared updates.
