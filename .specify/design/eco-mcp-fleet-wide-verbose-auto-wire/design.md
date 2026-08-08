# Design Document: Derive verbose (condensed-action) tools automatically from a connector's declared action surface, instead of hand-writing a verbose tool per action

CONCEPT:AU-ECO.mcp.fleet-wide-verbose-auto

> Covers the cluster. `AU-ECO.mcp.intent-surface-condensed-collapse` is a
> SEPARATE, already-documented decision (the 6-verb intent surface — see
> `docs/architecture/intent-surface.md`); this document covers the orthogonal
> "condensed action tool → many verbose `tool__action` tools" auto-wiring below
> it. `AU-ECO.mcp.verbose-auto-wire` is the pointer for its dynamic-action half.

> `agent_utilities/mcp/verbose_tools.py:590-660` (`autowire_verbose_from_condensed`).

## Decision — auto-derive one verbose tool per action from a condensed tool's static enum or a registered runtime action-provider, using FastMCP's `Tool.from_tool` transform, rather than requiring every connector to hand-author its verbose tools

Every fleet connector registers ONE condensed, action-routed tool
(`<service>_<domain>(action, params_json)`) so the tool count stays bounded as the
fleet grows to dozens of connectors. But some callers (smaller models, or an
IDE's static tool picker) do better with one **verbose** tool per action
(`<tool>__<action>`) with `action` preset and hidden. Before this,
`autowire_verbose_from_condensed` (`verbose_tools.py:590`) only handled tools that
declared `action` as a `Literal` enum (staticly known); connectors whose actions are
determined at **runtime** (`atlassian`, `arr`, and most other action-routed
connectors, whose valid action names depend on installed capabilities/config) got
**zero** verbose tools — the gap the function's own docstring names `ECO-4.90`. The
fix generalises the source of the action list to two ordered options — static enum
first, else a registered **action provider** (`register_action_provider`: a list, a
callable, or credential-free client-class introspection) called at verbose-tool-build
time — so every action-routed connector, static or dynamic, gets full verbose
coverage from the SAME derivation path, using FastMCP's native `Tool.from_tool`
transform so the derived tool still routes through the original handler (preserving
`Depends` client binding and `Context` injection) with zero re-implementation.

## Rejected alternative — require each connector module to hand-write its own `tool__action` wrapper functions

This is what the static-enum-only path amounts to in practice for any connector
whose actions cannot be expressed as a compile-time `Literal` — someone would have to
write, by hand, one thin wrapper function per action, for every dynamic-action
connector in the fleet (atlassian, arr, and "most action-routed connectors" per the
docstring). That was rejected on maintenance grounds visible directly in the
docstring's framing of the alternative it closes: "Before this, such tools yielded
zero verbose tools — the gap ECO-4.90 closes" — i.e. the fleet had already tried
"connectors that need this write it themselves" and the dynamic-action set simply
never got written, because it does not scale per-connector. Generalizing the
derivation once, fleet-wide, converts an ongoing per-connector authoring burden into
a one-time mechanism every current and future dynamic-action connector inherits for
free by registering an action provider.

## Risk Assessment

- **Blast Radius**: `agent_utilities/mcp/verbose_tools.py`; every connector that
  registers a dynamic action-routed tool (calls `register_action_provider`).
- **Backward Compatible**: Yes — additive tool derivation; existing condensed tools
  and any hand-written verbose tools are unaffected (idempotent re-run skips
  already-derived `tool__action` names via the `"verbose"` tag check).
- **Known weak point**: tools with **neither** a static enum nor a registered
  dynamic-action provider still get no verbose tools (`verbose_tools.py:614-617`) —
  a newly added dynamic connector that forgets to call
  `register_action_provider` silently reproduces the exact gap this closed for
  the earlier ones.
