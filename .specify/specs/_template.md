# Spec: [Feature Title]

## Pre-Flight Checklist (Mandatory — DSTDD)

> Every spec MUST have a corresponding design document that has passed validation.

- [ ] **KG search completed** — `.specify/design/[feature]/design.md` exists
- [ ] **Extension point identified** — or New Concept Proposal approved
- [ ] **C4 diagram created** — showing integration into pillar topology
- [ ] **No new CONCEPT: tag** without pillar reference
- [ ] **`code-enhancer` audit** run against proposed changes
- [ ] **Design validation passes** — `SDDManager.validate_design(feature_id)` returns no violations

## Design Reference

→ [Link to `.specify/design/[feature]/design.md`]

## User Stories

### US-1: [Story Title]

**As a** [role], **I want** [capability], **so that** [benefit].

**Acceptance Criteria:**
- [ ] [Criterion 1]
- [ ] [Criterion 2]
- [ ] [Criterion 3]

## Non-Functional Requirements

- [ ] All existing tests continue to pass (zero regression)
- [ ] Pre-commit hooks pass cleanly
- [ ] Documentation updated in `docs/pillars/` if pillar topology changes
- [ ] New functionality wired into ServiceRegistry for discovery
