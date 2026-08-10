# Adjudication packet — AU-ECO.mcp

33 live concepts. The deterministic pass already decided 11 pointer(s) and 1 retirement(s) from module locality, git archaeology and id shape alone. Confirm or correct the items below, then write the decisions into .specify/triage/AU-ECO.mcp.yaml.

## Clusters — confirm ONE parent each; the members inherit it

### intent-surface-condensed-collapse  (10 concepts)
    agent_utilities/mcp/verbose_tools.py:70 | #: Tag stamped ONLY when ``MCP_TOOL_MODE=intent`` (CONCEPT·AU-ECO.mcp.intent-surface-condensed-collapse):
    agent_utilities/mcp/multiplexer.py:4659 | # Split out the host server's OWN gated tools (CONCEPT·AU-ECO.mcp.intent-surface-condensed-collapse) —
    members: fleet-meta-tools-always-on, gateway-dispatch-isolation, intent-surface-condensed-collapse, intent-surface-cpd-ranking, intent-surface-delegation-shape, intent-surface-outcome-learning, intent-surface-resolution-cache, intent-surface-tool-lifecycle, knowledge-graph-exposure, two-surfaces-mcp-rest

### client-side-chat-session  (2 concepts)
    agent_utilities/knowledge_graph/retrieval/capabilities-power.json:23967 | "one_line": "Ingest AI agent chat/session history into the usage store + KG (CONCEPT·AU-ECO.mcp.client-side-chat-session). 'collect' auto-detects installed agents on THIS host and 
    agent_utilities/cli/__init__.py:155 | # CONCEPT·AU-ECO.mcp.client-side-chat-session — client-side chat/session ingestion for Claude + Antigravity
    members: client-side-chat-session, usage-cost-observability-surface

### cross-process-skill-harvest  (2 concepts)
    agent_utilities/governance/concept_lineage.yaml:25 | fleet_skill_harvest's CONCEPT·AU-ECO.mcp.cross-process-skill-harvest" -- same D-2.2-2.3-1 rationale
    agent_utilities/knowledge_graph/ingestion/fleet_skill_harvest.py:3 | CONCEPT·AU-ECO.mcp.cross-process-skill-harvest — Cross-Process Skill Harvest.
    members: cross-process-skill-harvest, skills-over-mcp-provider

### fleet-wide-verbose-auto  (2 concepts)
    agent_utilities/mcp/verbose_tools.py:621 | CONCEPT·AU-ECO.mcp.fleet-wide-verbose-auto — fleet-wide verbose auto-wire from condensed action enums
    agent_utilities/mcp/verbose_tools.py:794 | whose explicit targets already cover its whole surface). CONCEPT·AU-ECO.mcp.fleet-wide-verbose-auto.
    members: fleet-wide-verbose-auto, verbose-auto-wire

## Proposed RETIRE — the id names nothing (confirm or rescue) (1)

### fleet-meta-tools-always-on
    why: the id reads as a slugified prose fragment (fleet-meta-tools-always-on)
    agent_utilities/mcp/kg_server.py:4489 | # CONCEPT·AU-ECO.mcp.fleet-meta-tools-always-on
    .specify/design/fleet-meta-tools-always-on/design.md:25 | - **Proposed ID**: `CONCEPT·AU-ECO.mcp.fleet-meta-tools-always-on`

## UNDECIDED — the cheap signals ran out here (2)

### eco-serves-two-ard
    why: the marker text is truncated by the grammar ('# ARD registry surface (CONCEPT·AU-ECO.mcp.eco-serves-two-ard/ECO-4.97) — the graph-os twi') — the id itself reads like a real name, so the marker text needs cleaning either way; decide whether the concept survives that cleanup
    agent_utilities/server/routers/ard.py:3 | CONCEPT·AU-ECO.mcp.eco-serves-two-ard / ECO-4.97. Serves the two ARD artifacts at the bare domain root so
    agent_utilities/ecosystem/ard_registry.py:225 | CONCEPT·AU-ECO.mcp.eco-serves-two-ard. Assembles fleet MCP servers + KG skills into ARD resources, signs

### toolkit-live-discovery
    why: the marker exists only in prose/doc files — nothing in the shipped tree realises it, which is usually a retirement but occasionally a real decision recorded only in prose
    docs/journey.md:157 | To handle massive asynchronous volumes, the ecosystem utilizes the **Native Messaging Backend Abstraction** (`CONCEPT·AU-ECO.toolkit.journey-map-milestones`), managing high-through
    docs/guides/dynamic-tool-selection.md:101 | ### Concept Overview (CONCEPT·AU-ECO.mcp.toolkit-live-discovery)

## Proposed OWN DOCUMENT — is this really a decision? (19)

### client-side-chat-session
    why: the head of a 2-concept cluster — the siblings are proposed as pointers at this one, so this is the decision that has to be written down
    agent_utilities/knowledge_graph/retrieval/capabilities-power.json:23967 | "one_line": "Ingest AI agent chat/session history into the usage store + KG (CONCEPT·AU-ECO.mcp.client-side-chat-session). 'collect' auto-detects installed agents on THIS host and 
    agent_utilities/cli/__init__.py:155 | # CONCEPT·AU-ECO.mcp.client-side-chat-session — client-side chat/session ingestion for Claude + Antigravity

### cross-process-skill-harvest
    why: the head of a 2-concept cluster — the siblings are proposed as pointers at this one, so this is the decision that has to be written down
    agent_utilities/governance/concept_lineage.yaml:25 | fleet_skill_harvest's CONCEPT·AU-ECO.mcp.cross-process-skill-harvest" -- same D-2.2-2.3-1 rationale
    agent_utilities/knowledge_graph/ingestion/fleet_skill_harvest.py:3 | CONCEPT·AU-ECO.mcp.cross-process-skill-harvest — Cross-Process Skill Harvest.

### fastmcp-middleware
    why: a singleton: no sibling shares its source footprint or introducing commit (1 source file(s), 4 marker site(s))
    agent_utilities/mcp/middlewares.py:230 | CONCEPT·AU-ECO.mcp.fastmcp-middleware - Assimilated from FastMCP 'cross-cutting concern interception layer'.
    docs/journey.md:151 | As the parallel specialists run, they communicate over the **A2A (Agent-to-Agent) Network & Consensus engine** (`CONCEPT·AU-ECO.mcp.fastmcp-middleware`). Rather than executing in i

### fleet-wide-verbose-auto
    why: the head of a 2-concept cluster — the siblings are proposed as pointers at this one, so this is the decision that has to be written down
    agent_utilities/mcp/verbose_tools.py:621 | CONCEPT·AU-ECO.mcp.fleet-wide-verbose-auto — fleet-wide verbose auto-wire from condensed action enums
    agent_utilities/mcp/verbose_tools.py:794 | whose explicit targets already cover its whole surface). CONCEPT·AU-ECO.mcp.fleet-wide-verbose-auto.

### full-api-mcp-surface
    why: a singleton: no sibling shares its source footprint or introducing commit (3 source file(s), 7 marker site(s))
    agent_utilities/mcp/tools/engine_tools.py:45 | CONCEPT·AU-ECO.mcp.full-api-mcp-surface — Full engine API + MCP surface (REST + MCP in lockstep)
    agent_utilities/mcp/kg_server.py:4028 | # The low-level engine_<domain> tools (CONCEPT·AU-ECO.mcp.full-api-mcp-surface) are generic

### graph-reach-mcp-tool
    why: a singleton: no sibling shares its source footprint or introducing commit (1 source file(s), 4 marker site(s))
    agent_utilities/mcp/tools/reach_tools.py:1 | """graph_reach MCP tool — outbound messaging + last-active channel routing (CONCEPT·AU-ECO.mcp.graph-reach-mcp-tool).
    agent_utilities/mcp/tools/reach_tools.py:8 | CONCEPT·AU-ECO.mcp.graph-reach-mcp-tool — graph_reach MCP tool and REST twin for outbound user messaging

### harness-exposure
    why: a singleton: no sibling shares its source footprint or introducing commit (1 source file(s), 1 marker site(s))
    agent_utilities/mcp/harness_server.py:4 | CONCEPT·AU-ECO.mcp.harness-exposure — Harness MCP Exposure
    .specify/design/eco-mcp-harness-server-separation/design.md:3 | CONCEPT·AU-ECO.mcp.harness-exposure

### intent-surface-condensed-collapse
    why: the head of a 10-concept cluster — the siblings are proposed as pointers at this one, so this is the decision that has to be written down
    agent_utilities/mcp/verbose_tools.py:70 | #: Tag stamped ONLY when ``MCP_TOOL_MODE=intent`` (CONCEPT·AU-ECO.mcp.intent-surface-condensed-collapse):
    agent_utilities/mcp/multiplexer.py:4659 | # Split out the host server's OWN gated tools (CONCEPT·AU-ECO.mcp.intent-surface-condensed-collapse) —

### intent-surface-selection-accuracy
    why: a singleton: no sibling shares its source footprint or introducing commit (2 source file(s), 6 marker site(s))
    agent_utilities/knowledge_graph/retrieval/intent_selection_accuracy.py:4 | """Intent-surface selection-accuracy harness — CONCEPT·AU-ECO.mcp.intent-surface-selection-accuracy.
    agent_utilities/knowledge_graph/retrieval/intent_selection_accuracy.py:51 | #: CONCEPT·AU-ECO.mcp.intent-surface-selection-accuracy — hand-labelled corpus, one case per

### kg-skill-verb-coverage
    why: a singleton: no sibling shares its source footprint or introducing commit (2 source file(s), 4 marker site(s))
    agent_utilities/mcp/skill_coverage.py:2 | """CONCEPT·AU-ECO.mcp.kg-skill-verb-coverage — Graph-OS domain-skill coverage.
    .pre-commit-config.yaml:355 | # CONCEPT·AU-ECO.mcp.kg-skill-verb-coverage — third parity leg: every live

### live-server-metadata-cache
    why: a singleton: no sibling shares its source footprint or introducing commit (1 source file(s), 3 marker site(s))
    agent_utilities/knowledge_graph/core/engine_mcp_discovery.py:222 | CONCEPT·AU-ECO.mcp.live-server-metadata-cache — Live MCP server connection for tool metadata caching.
    agent_utilities/knowledge_graph/core/engine_mcp_discovery.py:3 | CONCEPT·AU-ECO.mcp.live-server-metadata-cache — MCP Server Live Tool Discovery

### profile-differences-from-client
    why: a singleton: no sibling shares its source footprint or introducing commit (6 source file(s), 20 marker site(s))
    agent_utilities/mcp/multiplexer.py:4300 | """Register the always-present fleet-health meta-tool (CONCEPT·AU-ECO.mcp.profile-differences-from-client)."""
    agent_utilities/observability/gateway_metrics.py:282 | # MCP multiplexer per-child resilience (CONCEPT·AU-ECO.mcp.profile-differences-from-client): one series per

### protocol-compat-bridge
    why: a singleton: no sibling shares its source footprint or introducing commit (4 source file(s), 17 marker site(s))
    pyproject.toml:166 | # fastmcp 4 is the default (CONCEPT·AU-ECO.mcp.protocol-compat-bridge). Empirically
    pyproject.toml:683 | # CONCEPT·AU-ECO.mcp.protocol-compat-bridge — fastmcp 4 is the default (see the

### standardized-interfaces
    why: a singleton: no sibling shares its source footprint or introducing commit (8 source file(s), 9 marker site(s))
    agent_utilities/mcp/__init__.py:3 | CONCEPT·AU-ECO.mcp.standardized-interfaces — MCP Standardized Interfaces
    agent_utilities/mcp/concurrency.py:53 | on the shared dispatch path. CONCEPT·AU-ECO.mcp.standardized-interfaces

### tasks-workitem-bridge
    why: a singleton: no sibling shares its source footprint or introducing commit (4 source file(s), 7 marker site(s))
    agent_utilities/mcp/tools/mcp_apps.py:16 | exact ``graph_jobs`` tool the Tasks↔WorkItem bridge (CONCEPT·AU-ECO.mcp.tasks-workitem-bridge)
    agent_utilities/mcp/server_factory.py:1631 | # CONCEPT·AU-ECO.mcp.tasks-workitem-bridge -- mount the native WorkItem-backed

### tool-mode-standardization
    why: a singleton: no sibling shares its source footprint or introducing commit (5 source file(s), 11 marker site(s))
    agent_utilities/mcp/readme_tools.py:1 | """Auto-generate the MCP tools table in an agent package's README (CONCEPT·AU-ECO.mcp.tool-mode-standardization).
    scripts/gen_graphos_manifest.py:375 | "The graph-os verbose 1:1 tool surface (CONCEPT·AU-ECO.mcp.tool-mode-standardization): one entry per CRUD\n"

### unified-mcp-skill-a2a-ingest
    why: a singleton: no sibling shares its source footprint or introducing commit (1 source file(s), 2 marker site(s))
    agent_utilities/knowledge_graph/core/engine_ingestion.py:785 | CONCEPT·AU-ECO.mcp.unified-mcp-skill-a2a-ingest — Unified MCP/Skill/A2A ingestion pipeline with live
    agent_utilities/knowledge_graph/core/engine_ingestion.py:775 | # CONCEPT·AU-ECO.mcp.unified-mcp-skill-a2a-ingest — Unified Agent Toolkit Ingestion

### v2-gateway-otel-tracing
    why: a singleton: no sibling shares its source footprint or introducing commit (2 source file(s), 6 marker site(s))
    .specify/design/v2-gateway-otel-tracing/design.md:33 | - **Proposed ID**: `CONCEPT·AU-ECO.mcp.v2-gateway-otel-tracing`
    .specify/design/v2-gateway-otel-tracing/design.md:3 | CONCEPT·AU-ECO.mcp.v2-gateway-otel-tracing

### webui-governed-mcp-delegation
    why: a singleton: no sibling shares its source footprint or introducing commit (2 source file(s), 5 marker site(s))
    agent_utilities/server/app.py:852 | # CONCEPT·AU-ECO.mcp.webui-governed-mcp-delegation
    agent_utilities/server/webui_mcp_delegation.py:3 | CONCEPT·AU-ECO.mcp.webui-governed-mcp-delegation
