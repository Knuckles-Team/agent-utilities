# Design Document: Every registered graph-os tool must be claimed by exactly one skill's machine-readable `agents/graph-os.yaml` sidecar, with no naming-convention fallback

CONCEPT:AU-ECO.mcp.kg-skill-verb-coverage

> `agent_utilities/mcp/skill_coverage.py:1-60` (module docstring,
> `INTENTIONALLY_UNSKILLED`).

## Decision — coverage is computed by discovering each skill's `agents/graph-os.yaml` sidecar and diffing its explicit claims against the immutable canonical `ToolSpec` universe, rather than inferring coverage from the skill's slug/name

The bundled skill suite is intentionally small — one workflow skill (e.g.
`graph-runtime-and-governance`) owns MANY related graph-os verbs. That means
"coverage" cannot be inferred by pattern-matching skill names against tool names
(`skill_coverage.py:6-7`: *"Coverage therefore cannot be inferred from a skill
slug"*). Instead every participating skill declares its contract explicitly in a
sidecar file, `agents/graph-os.yaml`, separate from the portable `SKILL.md`
frontmatter every agent client reads. `skill_coverage.py` discovers those sidecars
across every installed skill provider, validates their schema, and compares the
union of their claims against `TOOL_SPECS_BY_NAME`/`canonical_tool_names()` — the
same immutable tool universe the surface-parity gate uses. There are "no naming
fallbacks, frontmatter fallbacks, or intentionally-unskilled waivers" beyond an
explicit, justified, hand-maintained exception list (`INTENTIONALLY_UNSKILLED`,
`skill_coverage.py:31`) — every other required or feature-qualified tool MUST be
claimed by exactly one valid sidecar or the gate fails.

## Rejected alternative — infer a skill's tool coverage from its slug/description, or accept an unclaimed tool silently

The most tempting shortcut, given the skill suite is small by design, is exactly
what the docstring rules out first: assume a skill named e.g. `graph-runtime-and-governance`
covers "the runtime/governance tools" by name-matching its slug or description
text against tool names. That breaks down precisely because one skill owns MANY
unrelated-sounding verbs — a slug can never be a reliable proxy for a skill's real
claimed surface. The second rejected shape is looser still: let any tool with no
matching sidecar claim pass silently (an implicit waiver). That was rejected in
favour of a hard gate with an explicit, reviewed exception list — each entry in
`INTENTIONALLY_UNSKILLED` carries a written reason (e.g. the `engine_rbac`/`engine_admin`
entries at `skill_coverage.py:44-56` explain exactly why those two are waived and
what the intended fix is) — "every entry weakens the gate" is stated directly in
the source (`skill_coverage.py:29`), so an unreviewed silent pass was rejected as
the same gate-erosion risk a blank waiver list would create.

## Risk Assessment

- **Blast Radius**: `agent_utilities/mcp/skill_coverage.py`,
  `agent_utilities/mcp/tool_specs.py`, every skill's `agents/graph-os.yaml`.
- **Backward Compatible**: Yes — a coverage gate; does not change tool registration
  itself.
- **Known weak point**: the `INTENTIONALLY_UNSKILLED` waiver list is
  hand-maintained free text; nothing enforces that a waiver's stated follow-up
  (e.g. "extend that skill's `wraps:` in the epistemic-graph repo") actually
  happens — a permanent waiver looks identical to a forgotten one.
