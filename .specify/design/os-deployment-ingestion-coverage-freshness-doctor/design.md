# Design Document: `doctor` gets a symmetric coverage+freshness check per ingestion domain (agent-packages repos, external connectors), comparing against `DeltaManifest` watermarks, so a dark or stale source is a visible WARN instead of a silent gap

CONCEPT:AU-OS.deployment.connector-coverage-check (covers the cluster: `AU-OS.deployment.flagging-repos` is the pointer for the repo-ingestion half of the SAME pattern)

> `agent_utilities/deployment/doctor.py:3489-3510` (`_check_ingestion_coverage`,
> the repo-freshness leg) and `:3576-3641` (`_check_connector_coverage`, "the
> connector analogue of `ingestion_coverage`" — `doctor.py:3582`);
> `agent_utilities/knowledge_graph/ingestion/coverage.py`,
> `agent_utilities/knowledge_graph/ingestion/connector_coverage.py`,
> `agent_utilities/knowledge_graph/ingestion/manifest.py` (`DeltaManifest`).

## Decision — every ingestion domain gets its own `doctor` check that compares its expected-source set against `DeltaManifest` freshness watermarks and reports `warn` (missing/stale counts) rather than staying silent, applied first to agent-packages repos (`ingestion_coverage`) and then, as an explicit analogue, extended to external connectors (`connector_coverage`)

Native codebase-context-via-KG (and, symmetrically, connector-backed world-model
queries) is only as good as what actually got ingested. If a repo has no `:Code`
symbols, or a connector's last delta sync is stale, a KG query against that domain
returns nothing and the agent silently falls back to grep/direct-source-hit — a
regression nobody notices until it is diagnosed by hand. `_check_ingestion_coverage`
(`doctor.py:3489`) closes that gap for repos: it enumerates the agent-packages
subtree from `workspace.yml`, gets live symbol counts per repo, and assesses
coverage/freshness, surfacing "coverage gaps... rather than silent" (`doctor.py:3495`).
`_check_connector_coverage` (`doctor.py:3581`) is explicitly the SAME pattern
generalized to a second domain — "the connector analogue of `ingestion_coverage`: a
dark or stale connector means the world-model for that domain... is silently wrong"
(`doctor.py:3582-3583`) — reusing the identical `DeltaManifest` freshness-watermark
comparison against a different expected-source enumerator
(`enumerate_expected_connectors` vs. `enumerate_agent_packages_repos`). Both checks
redact identities from the doctor result (aggregate counts only), keeping the
report safe to share without leaking which specific repos/connectors are affected.

## Rejected alternative — trust that ingestion happened, or bespoke ad hoc checks per domain with no shared freshness model

The alternative this replaced is simply not having a coverage check at all: an
operator (or the agent itself) discovers a gap only when a KG query against a
specific domain unexpectedly returns nothing, then has to manually work out
whether the repo/connector was never ingested, failed to ingest, or went stale.
That is rejected as a silent failure mode inconsistent with the doctor's role
elsewhere in the deployment surface. The narrower alternative — writing a
DOMAIN-SPECIFIC freshness check for connectors from scratch, independent of the
repo-coverage check's `DeltaManifest`-based model — was rejected once the pattern
proved out on repos first: reusing the exact same freshness-watermark comparison
and doctor-result shape (`_result(...)` with `missing_count`/`stale_count`/
`coverage_pct`) for connectors means one mental model and one remediation path
("`source_sync source=all mode=delta`") covers both domains, and a THIRD ingestion
domain added later inherits the same pattern instead of inventing a fourth
freshness convention.

## Risk Assessment

- **Blast Radius**: `agent_utilities/deployment/doctor.py`,
  `agent_utilities/knowledge_graph/ingestion/coverage.py`,
  `agent_utilities/knowledge_graph/ingestion/connector_coverage.py`,
  `agent_utilities/knowledge_graph/ingestion/manifest.py`.
- **Backward Compatible**: Yes — additive doctor checks; `skip` (not `fail`) when
  the workspace/connector context does not apply (e.g. not a workspace checkout,
  no connectors configured).
- **Known weak point**: both checks report `warn`, never `fail` — a completely dark
  ingestion domain degrades the doctor's overall health summary but never blocks a
  deployment from being reported healthy; the check's value depends on an operator
  actually reading `doctor`'s warn-level output rather than only its pass/fail
  headline.
