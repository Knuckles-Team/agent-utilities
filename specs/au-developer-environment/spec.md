# Public AU development environment and skill lifecycle

**ID:** AU-DEV-001 · **Owner:** agent-utilities · **Delivery:** SPECIFIED; acceptance NOT AUDITED.
**Items:** AU-DEV-R001 (AU skill portion), AU-DEV-R002 (AU source move), AU-DEV-R003; supports every other AU spec. See [requirements.md](requirements.md) for the definition of every requirement ID and [status.json](status.json) for its delivery state and evidence.

## Outcome

A contributor starting with public GitHub repositories can provision the AU development environment, discover the same spec workflow and development skill used by maintainers, run PR gates without private services, and select a separately provisioned integration profile when a feature needs a served graph. Installed skills match checked-in source and can be refreshed deterministically.

## Requirements

1. `graphos-development` or its public successor describes a fresh-clone graph-os ecosystem bootstrap: supported Python/Rust/Node versions, package install, generated EG client, disposable EG fixture, GraphOS/AU/SDK wiring, identity fixture, test and cleanup commands. It must not assume sibling checkout layout, private registry, host inventory, VPN, credentials or pre-running service.
2. The AU-specific development skill binds that workflow to `scripts/uv_workspace.py` and this repository's `AGENTS.md` quality contract. A contributor can select a lightweight unit/contract profile and a composed integration profile; each lists its actual dependencies and produces reproducible receipts.
3. Skills used for spec generation, verification, task planning and implementation are linked from public `CONTRIBUTING.md` with install/refresh commands and version or commit pins. Their spec output must include architecture, existing wiring reuse, design, positive/negative tests, quality gates, tasks, status and exact evidence.
4. A source-to-installed inventory detects moved/renamed provider skills and stale installed copies. Refresh is idempotent, prints its plan before mutation and verifies resulting hashes/IDs. Missing optional integrations are reported without blocking the pure spec or unit profile.
5. CI jobs required for a PR provision declared disposable dependencies or use deterministic contract fixtures. A live external identity provider, model vendor, broker, GPU or private network is a release qualification profile, not an ambient commit prerequisite.
6. Secret references are injected at runtime only. Fixtures use synthetic identities and loopback endpoints; no credential, inventory, local absolute path or private endpoint enters tracked docs or artifacts.

## Acceptance

A fresh fork and disposable CI runner execute skill install/refresh, AU unit/contract tests and a composed public fixture without hidden state. An intentionally stale skill is detected and repaired, and normal quality gates pass at the same merged head.
