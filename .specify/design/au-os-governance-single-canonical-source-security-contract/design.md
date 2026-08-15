# Design Document: One canonical fleet-wide security-contract module

CONCEPT:AU-OS.governance.single-canonical-source-security-contract ·
CONCEPT:AU-OS.host.windows-job-object-process-tree-bound

> `scripts/security_contract.py`, reached fleet-wide through
> `scripts/run_agent_utilities_gate.py --script scripts/security_contract.py -- ...`.

## KG Analysis (Required)

### Nearest Existing Concepts

| Concept ID | Name | Similarity | Pillar |
|---|---|---|---|
| `scripts/run_agent_utilities_gate.py`'s locate-au-and-run pattern (already used for `check_no_legacy_markers.py`, `mermaid_linter.py`, `check_stubs.py`) | the existing reuse mechanism this module adopts rather than reinventing | high | OS |

### Extension Analysis

- **Primary Extension Point**: the existing locate-au-and-run forwarder
  pattern.
- **Extension Strategy**: augment — route a fourth script through the same
  pattern; the module's own internal Windows-portability fix is a
  specialization inside it.
- **New Concept Required?**: Yes — both the consolidation and the
  process-tree-kill portability fix are real decisions with rejected
  alternatives.

## Decision 1 — one canonical copy, reached by every consumer repo, not 74 pasted copies

CONCEPT:AU-OS.governance.single-canonical-source-security-contract

### Problem

`security_contract.py` (execute bounded repository security hooks and
validate their evidence) previously existed as an identical or
near-identical file copy-pasted into ~74 `agent-packages/*` repos. One defect
(an unguarded `import resource`, Unix-only) replicated 74-fold across those
copies and drifted into several inconsistent variants over time — the classic
copy-paste-fork failure mode, at fleet scale.

### Decision

This module is the single canonical copy for the whole fleet. Every consumer
repository reaches it through
`scripts/run_agent_utilities_gate.py --script scripts/security_contract.py -- ...`
— the same locate-au-and-run pattern already used for
`check_no_legacy_markers.py`, `mermaid_linter.py`, and `check_stubs.py` —
instead of keeping a local copy. A fix lands once, here, and every consumer
repo picks it up on its next run; there is no second file to remember to
patch.

The reusable security workflow calls this module directly from the checked-
out repository. A repository contract names argv arrays rather than shell
strings, so untrusted configuration cannot introduce an extra
shell-evaluation layer. All paths are relative regular files below the
repository root, outputs are bounded, hook environments exclude
credential-like variables, and missing or malformed evidence fails closed.

## Decision 2 — atomic process-tree kill on Windows via Job Objects

CONCEPT:AU-OS.host.windows-job-object-process-tree-bound

### Problem

A bounded security hook that spawns child processes needs to be killable as
one unit — the hook AND every descendant it spawned — or a runaway/hung
descendant survives past the hook's own bounded lifetime. `os.killpg(...,
SIGKILL)` gives this guarantee on POSIX via process groups; Windows has no
process-group equivalent.

### Decision

On Windows (`os.name == "nt"`), assign the hook process to a Job Object
created with `KILL_ON_JOB_CLOSE`. Terminating/closing the job kills the hook
and every descendant it spawned atomically — the Windows analogue of a POSIX
process group, giving the same all-or-nothing kill guarantee cross-platform
rather than leaving Windows with a weaker, best-effort kill.

## Wire-First

Both decisions live in the one module, exercised by the same reusable
security workflow every consumer repo's pre-push gate runs through
`run_agent_utilities_gate.py`.
