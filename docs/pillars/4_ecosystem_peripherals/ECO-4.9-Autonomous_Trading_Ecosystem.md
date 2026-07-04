# Autonomous Trading Ecosystem (CONCEPT:AU-ECO.bus.pluggable-event-queue)

## Overview
The Autonomous Trading Ecosystem provides a full suite of infrastructure, configurations, and AI-native patterns for executing quantitative and algorithmic trading via the `emerald-exchange` MCP server and the `agent-utilities` ontology framework.

## 1. Unified Configuration Schema
The trading ecosystem is configured centrally in the XDG-compliant `~/.config/agent-utilities/config.json` file under the `"trading"` root key.
It supports:
- **Exchange Backends**: Abstractions for `paper`, `alpaca`, `ccxt` (Binance, Coinbase, Kraken), and `freqtrade`.
- **Risk Limits**: Automated OS-5.1 pre-trade controls (`max_position_pct`, `max_portfolio_drawdown_pct`, `regime_shift_halt`).
- **Signal Fusion**: Configurations for bayesian fusion and minimum confidence scoring.
- **Monitoring Intervals**: Background cron scheduling frequencies for P&L tracking and snapshotting.

## 2. Infrastructure Blueprints
Automated provisioning of trading nodes is handled via the `skill-graphs/infrastructure-blueprints/trading/` YAMLs:
- **`emerald-exchange.yaml`**: The primary MCP server handling order execution and pre-trade risk validation.
- **`freqtrade-node.yaml`**: A pre-configured Freqtrade container for crypto-native algorithmic execution.
- **`trading-grafana.yaml`**: Observability stack deploying Grafana + Redis TimeSeries for portfolio visualization.
- **`gpu-compute-node.yaml`**: Dedicated GPU node routing to `data-science-mcp` for QLib backtesting and model training.

## 3. Skill-Graph Decomposition (CONCEPT:AU-AHE.assimilation.trading-ecosystem-spec)
To enable multi-agent specialization, trading knowledge is broken down into four discrete `skill-graphs` subdirectories:
- **`strategies/`**: Documentation of momentum, mean reversion, multi-factor, and ML trading archetypes.
- **`data-sources/`**: Market data ingestion patterns (AKShare, Yahoo Finance, CCXT) and on-chain analytics.
- **`execution/`**: Backend protocol implementation details and the order lifecycle.
- **`risk-management/`**: Implementation of Kelly criterion sizing, KS-Test regime detection, and Kill Switches.

## 4. MCP Domains
The `emerald-exchange` MCP server exposes 9 discrete tool domains:
1. **`market_data`**: Historical OHLCV and real-time quotes.
2. **`orders`**: Submit, cancel, status, halt, resume.
3. **`portfolio`**: Account equity, positions.
4. **`risk`**: Pre-trade validation, daily loss checks, Kelly sizing.
5. **`signals`**: Alpha generation, signal fusion, regime detection.
6. **`strategy`**: Lifecycle management (draft → paper → live).
7. **`crypto`**: Funding rates, on-chain whale alerts, cross-exchange arb scanning.
8. **`debate`**: Multi-agent hypothesis debates with risk compliance veto power.
9. **`prediction_markets`**: Prediction-market tooling with risk-guard integration.

## Implementation Details
- **Source Code**: `agent-packages/agents/emerald-exchange`
- **Pillar**: Ecosystem Peripherals
- **Bridge**: CONCEPT:AU-AHE.assimilation.autonomous-trading-ecosystem through EE-020
