# Design Document: Map the existing fleet onto the draft Agentic Resource Discovery (ARD) spec's envelope, from ONE shared publish-side module both surfaces call, instead of building a bespoke discovery format

CONCEPT:AU-ECO.mcp.eco-serves-two-ard

> `agent_utilities/ecosystem/ard_registry.py:1-60` (module docstring,
> `build_ai_catalog`, `ard_search`); consumed by
> `agent_utilities/server/routers/ard.py:3` (gateway REST) and
> `agent_utilities/mcp/kg_server.py:3843` (`@mcp.custom_route` mirror).

## Decision — implement ARD's two publish-side artifacts (the static `ai-catalog.json` manifest and the dynamic `POST /search` registry API) in ONE core module the gateway REST router and the graph-os MCP custom route both call, mapping the EXISTING fleet catalog/skill data onto ARD's resource envelope rather than inventing a new discovery protocol

ARD is a draft open spec (Hugging Face, Microsoft, Google, GoDaddy and others) for
separating capability *discovery* from *execution* — instead of an agent
hardcoding an MCP URL, it discovers tools/skills/agents at runtime via a static
manifest plus a ranked-search API. Rather than inventing agent-utilities' own
discovery format on top of what the fleet already exposes,
`ard_registry.py`'s explicit design choice is "we map our existing fleet onto
ARD's envelope rather than building anything new" (`ard_registry.py:16`): every
fleet MCP server (probed via the multiplexer catalog) becomes an
`application/mcp-server+json` resource (tags/example queries from
`derive_capability_synonyms`), every KG `:Skill` node becomes an
`application/ai-skill` resource, and `/search` ranking reuses
`MCPMultiplexer.discover_tools` — the SAME token-overlap-blended-with-KG-semantic-search
engine `find_tools` already rides. This is the ONE core both serving surfaces call
(`ard_registry.py:14-16`) — the gateway REST router and the graph-os
`@mcp.custom_route` mirror — "keeping them in lockstep per the surface-parity
rule," and entries are Ed25519-signed (`security/ard_signing`) so a consuming
agent can verify them against the manifest's `publisherKey`.

## Rejected alternative — build agent-utilities' own bespoke capability-discovery format instead of adopting the draft ARD spec

The fleet already had internal capability discovery (`find_tools`, the multiplexer
catalog, KG skill nodes) before ARD support existed — the alternative to adopting
ARD was simply not implementing it, and continuing to rely on those internal-only
mechanisms plus each MCP client's own `mcp_config.json`. That was rejected in
favour of adopting a spec multiple external vendors are converging on, specifically
BECAUSE the fleet's data was already shaped closely enough to map onto it "rather
than building anything new" — i.e. the marginal cost was an envelope/mapping layer,
not a new discovery mechanism, making external interoperability (any ARD-aware
agent, not just this fleet's own clients, can discover these capabilities) nearly
free. The module also explicitly rejects hard-coding the exact ARD JSON shape
throughout the codebase: it is "intentionally quarantined to this module (and the
consume parser in `connectors/ard.py`) so a draft-spec field rename is a one-file
edit" (`ard_registry.py:27-29`) — anticipating that a DRAFT spec will change, and
isolating that blast radius up front rather than accepting spec churn spread across
every caller.

## Risk Assessment

- **Blast Radius**: `agent_utilities/ecosystem/ard_registry.py`,
  `agent_utilities/server/routers/ard.py`, `agent_utilities/mcp/kg_server.py`
  (custom route mirror), `agent_utilities/connectors/ard.py` (consume side).
- **Backward Compatible**: Yes — additive discovery surface; existing MCP/REST
  tool dispatch is unaffected.
- **Known weak point**: ARD is explicitly a DRAFT spec — `ardSpecVersion` is
  stamped per the module's own admission of "the assumed spec revision"
  (`ard_registry.py:29`), so a future breaking spec change is a known, accepted
  risk this module chose to quarantine rather than avoid.
