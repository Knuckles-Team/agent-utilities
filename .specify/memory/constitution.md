# Meristem constitution

This repository inherits the [organization constitution](https://github.com/Knuckles-Team/pipelines/blob/main/.specify/memory/constitution.md) version 1.0.0. Every article of that constitution applies here. This file adds only the rules and limits of this repository.

**Inherits**: organization constitution v1.0.0 | **Version**: 1.0.0 | **Ratified**: 2026-10-31 | **Last amended**: 2026-10-31

## Organization standards

- [Spec standard](https://github.com/Knuckles-Team/pipelines/blob/main/reference/spec-standard.md): one owning spec per capability, computed states, `Spec:` trailers.
- [Ecosystem boundaries](https://github.com/Knuckles-Team/pipelines/blob/main/reference/ecosystem-boundaries.md): which repository owns which requirement.
- [Wording standard](https://github.com/Knuckles-Team/pipelines/blob/main/reference/ste.md): sentences of 20 words or fewer, with `must`, `should`, or `may`.

## Repository articles

### R-I. The harness applies decisions and never makes them

Meristem runs one agent run. It applies a decision it receives and enforces a limit it receives. It has no store, planner, selector, router, retriever or policy, and no MCP, server, API or A2A.

**Enforced by:** Pre-commit hooks `boundary-owners`, `boundary-coverage`, `boundary-file-owners` and `boundary-history` (`scripts/check_boundary_*.py`); [ecosystem boundaries](https://github.com/Knuckles-Team/pipelines/blob/main/reference/ecosystem-boundaries.md).

**Plan check:** Does every threshold in the change arrive as a received value with a default?

### R-II. Dependencies stay minimal

The harness depends on the connector SDK and the model client library only. It imports no MCP, web server or storage package.

**Enforced by:** Forbidden-import matrix of the [ecosystem boundaries](https://github.com/Knuckles-Team/pipelines/blob/main/reference/ecosystem-boundaries.md).

**Plan check:** Does the change add no import of a server, MCP or storage package?

### R-III. A boundary owner exists for every directory

Each directory and file of the package has a named boundary owner. A file without an owner fails the gate.

**Enforced by:** The four `boundary-*` hooks above.

**Plan check:** Does every new file sit under a registered boundary owner?

## Repository limits

This repository sets no numeric limit of its own. It follows the organization limits table.

Amend this file in a reviewed change. State the impact in the pull request. When the organization constitution changes its version, update the **Inherits** line in the same change.
