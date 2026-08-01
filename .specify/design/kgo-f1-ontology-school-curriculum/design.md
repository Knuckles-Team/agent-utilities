# Design Document: The Ontology School curriculum is a small, versioned, human-editable manifest — deliberately duplicated into the frontend rather than served from a backend

CONCEPT:AU-KG.ontology.learning-curriculum-manifest

> `docs/learn/manifest.yaml`.

## Decision — a schema-versioned courses→lessons index, consumed by TWO independent readers because `docs/` is not shipped in the installed package

`manifest.yaml:1-20` states the shape and the reason for its consumption
pattern directly: "A small, versioned, human-editable curriculum index:
courses -> lessons (title + a markdown body path + an optional quiz). This
ships the FRAMEWORK plus two real starter lessons authored from existing
platform material — see `docs/learn/index.md` for the honest scope note (not
the full curriculum)." It is "consumed by two independent readers,
DELIBERATELY (this is documentation CONTENT, not a runtime capability —
`docs/` is excluded from the installed Python package, see `pyproject.toml`
`exclude`, so nothing here is reachable from a deployed agent-utilities
process)": the mkdocs docs site (linking straight to the lesson `.md` files),
and agent-webui's `LearnView`, which "carries its own copy of this manifest +
lessons (`agent-webui/src/content/learn/`) so the in-app course list/lesson
reader/presentation mode/quiz work with no backend round-trip."

The machine triage tool flagged this id "review" because it found "no sites
in the shipped tree" — true in the narrow sense that `docs/` is excluded from
the installed package, but the marker is a real, deliberate, and DOCUMENTED
decision, not a retirement candidate; the file explains its own scope and
consumption model rather than being an orphaned fragment.

**The rejected alternative is serving the curriculum from a backend API** —
the obvious design for an in-app "Learn" view, and the one that would let
agent-webui fetch the manifest live rather than carry a duplicate. That's
foreclosed by the `docs/` package-exclude decision (a separate, upstream
choice this file works within, not around): since nothing in `docs/` ships
with a deployed process, `LearnView` cannot round-trip to a backend for this
content at all — so it carries its own copy instead. The cost accepted is
explicit staleness risk: the docs-site copy and the agent-webui copy can
drift, since nothing here synchronizes them automatically; the file's own
honesty about "not the full curriculum" is the same discipline applied to
scope as to duplication.

## Risk Assessment

- **Blast Radius**: `docs/learn/manifest.yaml`, `docs/learn/lessons/*.md`,
  `agent-webui/src/content/learn/` (the duplicated copy).
- **Backward Compatible**: Yes — documentation content; no runtime path
  depends on it.
- **Known weak point**: the two copies (docs-site manifest, agent-webui's own
  copy) are maintained independently — nothing detects or prevents them
  drifting apart after an edit to one and not the other.
