# Implementation plan

1. Freeze the public owner map and an old-operation inventory; classify unique gateway, MCP and deployment behavior.
2. Ship EG and SDK generated contracts first, then graph-os served routes, then AU API adapters. Pin exact compatible versions before cutting callers.
3. Cut served shell, connector lifecycle, graph/semantic state, and cross-cutting state in dependency order. Each PR identifies the requirement IDs it closes and proves a complete vertical path.
4. Remove old AU imports, modules, scripts, dependencies and stale tests in the same change that activates replacement wiring.
5. Add a generated owner-manifest rule plus import/script census to prevent reversal. Run full gates and record exact merged-head evidence.

The published specification is independently implementable. Cross-repository owners may keep their own specs, but all AU requirements, decisions and acceptance criteria needed for this cut are stated here.
