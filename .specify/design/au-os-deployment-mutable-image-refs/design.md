# Design Document: Compose image references are digest-pinned, not mutable tags

CONCEPT:AU-OS.deployment.mutable-image-refs

> Realised by `agent_utilities/mcp/check_env_var_drift.py`'s compose scanner,
> which reads the `image:` key (not only `environment:`) so an
> operator-supplied digest reference such as
> `image: ${MEDIA_DOWNLOADER_MCP_IMAGE:?set-...-to-image@sha256-digest}` is
> seen as a live env-var read rather than reported DEAD.

## KG Analysis

### Nearest Existing Concepts

| Concept ID | Name | Similarity | Pillar |
|---|---|---|---|
| AU-OS.deployment.vault-seed-service | Deployment-time secret seeding | partial — deployment-time integrity, different subject | OS |
| AU-OS.config.env-var-single-source-of-truth | Code is authoritative for env vars | partial — this is the scanner that enforces it | OS |

### Extension Analysis

- **Primary Extension Point**: the existing `deployment` domain of the OS pillar.
- **Why not a new domain**: this was originally tagged `AU-OS.supply-chain.*`,
  but `supply-chain` is not in the OS pillar's closed vocabulary
  (`agent_utilities/governance/domain_vocab.yaml`: identity, governance,
  deployment, observability, state, safety, config, host, context, scaling,
  audit), so `check_domain_vocab.py` fails the build on it. Per
  **Extend-Before-Invent** the id moves onto the already-registered
  `deployment` domain rather than registering a twelfth. Behaviour is
  unchanged; only the identifier moves. Recorded under `renamed:` in
  `agent_utilities/governance/concept_lineage.yaml`.

## Decision

A container image reference in a tracked compose file is **pinned to an
immutable digest** supplied by the operator through an environment variable,
never to a mutable tag such as `:latest`. Two consequences the tooling must
respect:

1. The env var carrying the digest is a **live, required** read — every
   `docker compose up` genuinely needs it — so the drift checker scans
   `image:` alongside `environment:` and must not report it DEAD.
2. Because the reference lives in a version-bearing file, the file must be
   registered in `.bumpversion.cfg` so the pin is advanced deliberately by
   `bump-my-version` rather than drifting by hand.

## Scope

Deliberately limited to the four compose keys the scanner reads. Not extended
to `labels:`, `volumes:`, `ports:`, or resource scalars — those carry no image
reference and widening the scan would produce false live-reads.
