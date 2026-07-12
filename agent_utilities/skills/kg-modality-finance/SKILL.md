---
name: kg-modality-finance
skill_type: skill
description: >-
  Run quantitative-finance operations in the epistemic-graph engine — portfolio
  optimization, risk, regime detection, trading signals, HFT primitives, and
  derivatives pricing. Use when you need in-engine quant compute ("optimize a
  portfolio", "compute risk/VaR", "detect market regime", "price a derivative",
  "generate trading signals").
license: MIT
tags: [graph-os, engine, modality, finance, quant, risk]
tier: modality
wraps: [engine_finance]
metadata:
  author: Genius
  version: '0.1.0'
---

# KG Modality — Finance (quantitative finance)

> **Condensed intent-surface note (Seam 8).** Under the default intent surface (`MCP_TOOL_MODE=intent`), `engine_finance` is held back from the default tool list (nothing removed — REST + `_execute_tool` still reach it exactly as documented below). Two ways to use this skill unchanged: (1) `load_tools(tools=["engine_finance"])` once per session (as below), then proceed exactly as documented; or (2) call the `ask` intent verb with the same natural-language request — the resolver routes to `engine_finance` for you and returns the result plus a routing justification. Set `MCP_TOOL_MODE=condensed`/`verbose`/`both` to expose the granular tools eagerly instead.


Fronts the epistemic-graph engine's **`finance`** domain: native quantitative
finance operations — portfolio optimization, risk metrics, regime detection,
signal generation, high-frequency-trading primitives, and derivatives pricing —
computed in-engine over the same substrate that holds the market/entity graph
and time-series data.

This is the **modality** tier — a thin wrapper over the low-level engine surface
(`engine_finance`), action-routed 1:1 over the `epistemic_graph` client's
`FinanceClient`. The action set is discovered from the client (optimize, risk,
regime, signals, hft, derivatives, …); call with an empty `action` to list the
live set.

> Note: the separate `quant` (emerald-exchange) tool is intentionally
> unskilled — this skill fronts the generic engine `finance` domain.

## How to reach it

**Via the multiplexer:**
1. `load_tools(tools=["engine_finance"])`.
2. `engine_finance(action="", params_json="{}")` — list actions.
3. `engine_finance(action="optimize", params_json="{...}", graph="")` — invoke.
4. `unload_tools(...)` when done.

**Direct MCP on graph-os:** `engine_finance` is a registered tool; per-method
verbose tools appear under `MCP_TOOL_MODE=verbose|both`.

**REST twin:** `POST /engine/finance` with body
`{"action": "<method>", "params_json": "{...}", "graph": ""}`.

## Example

```jsonc
// discover the finance actions the live engine supports
engine_finance(action="", params_json="{}")

// optimize a portfolio (exact args come from the action list)
engine_finance(action="optimize",
  params_json="{\"assets\": [\"AAPL\", \"MSFT\"], \"objective\": \"max_sharpe\"}")
```

Pair with `kg-modality-timeseries` for the price history and
`kg-modality-analytics` for cross-modal statistics.
