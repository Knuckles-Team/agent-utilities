# Design Document: Log location/endpoint/caller-identifier redaction

> Every feature begins with a design document. This gates creation through
> the Knowledge Graph to enforce the **Extend-Before-Invent** principle.

## KG Analysis (Required)

### Nearest Existing Concepts

| Concept ID | Name | Similarity | Pillar |
|---|---|---|---|
| AU-OS.security.persistence-privacy-guard | location-field redaction on persisted records | high | AU-OS |
| AU-OS.identity.authenticated-identity-enforcement | actor/identity handling boundary | med | AU-OS |
| AU-OS.observability.otlp-secret-headers | observability-path secret handling | med | AU-OS |

### Extension Analysis

- **Primary Extension Point**: none of the above cover the *log-emission* boundary specifically — the persistence-privacy guard redacts fields before they are written to the graph, but nothing previously stopped a raw filesystem path, network endpoint, or caller identifier from reaching a `logger.info`/`warning`/`error` call directly.
- **Extension Strategy**: new, narrowly-scoped primitive (specialize the existing "don't leak location/identity data" family for the log-record boundary rather than the persistence boundary).
- **New Concept Required?**: Yes — one small, reusable helper plus a static AST gate.

## Problem

A raw filesystem path, network endpoint, or caller identifier passed directly into a log call leaks host/tenant topology to anyone with log-read access (a log-aggregation operator, a compromised sidecar, a support bundle). No mechanism previously stopped this at the log call site itself — `tests/unit/security/test_log_location_privacy_static_gate.py` statically walks every `agent_utilities` module for a log call passing a variable/attribute named like one of these sensitive kinds (`path`, `endpoint`, `host`, `file_path`, ...) and fails the build if it finds one, but until this change there was no single, correct helper to fix a flagged call site with.

## Design

- **`CONCEPT:AU-OS.observability.log-location-privacy`** — `agent_utilities/security/log_redaction.py::redact_for_log(value)` is the one shared, deterministic redaction every log call site uses instead of the raw value: `None`/empty redacts to `"<empty>"`; anything else renders as `str(value)` reduced to a 12-hex-character SHA-256 prefix (`<redacted:xxxxxxxxxxxx>`). The same input always produces the same short tag within a process, so operators can still correlate repeated log lines about the same path/endpoint/caller across a session without the log ever carrying the literal value. Non-reversible (one-way digest, not encoding), so it is safe even if the log stream itself is later exposed.
- Call sites fixed under this concept in the initial landing: `agent_utilities/knowledge_graph/core/graph_compute.py` (placement-routed endpoint warnings). The remaining ~76 raw-exception/~15 raw path-or-host log sites across `mcp/`, `gateway/`, `server/` are tracked as separate follow-up work (out of scope for this landing) that will call this same helper.

## Wire-First

Verified by `tests/unit/security/test_log_location_privacy_static_gate.py` (the static AST gate itself, green) plus direct exercise of `redact_for_log` at the `graph_compute.py` call site it fixed. No new external dependency; pure stdlib `hashlib`.
