# Comparative Analysis — Quant Agentic Frameworks vs `agent-utilities`

Comparison of three open-source quant agentic frameworks against the
`agent-utilities` finance domain (`agent_utilities/domains/finance/*`,
`CONCEPT:KG-2.6`), with three concrete gaps closed:

- **`CONCEPT:KG-2.26`** — Trade-journal bias auditor + shadow account
- **`CONCEPT:KG-2.27`** — Agent calibration / reputation tracking
- **`CONCEPT:KG-2.28`** — Persona decision-heuristic enrichment

The thesis: each borrowed capability is made **stronger** by our OWL/KG /
graph-native / self-evolution substrate — the borrowed feature stops being a
one-shot report and becomes a **reasoned-over, provenance-bearing, calibrated
fact** the rest of the system can cite.

---

## 1. Per-repo capability summary

### Vibe-Trading — research / strategy-generation studio
A skill-rich research agent: **74 skills** (ML-strategy, SMC, Elliott-wave,
sector-rotation, perp-funding-basis, on-chain analysis, liquidation-heatmap,
options/**SABR** vol-surface, …) and **29 swarm presets**. Its standout, novel
capability is the **shadow account / trade-journal audit**
(`agent/src/shadow_account/*`, `agent/src/tools/trade_journal_tool.py`): it parses
a broker export, FIFO-matches roundtrips, builds a trading **profile** (win rate,
holding days, PnL ratio, max drawdown), and runs **4 behavioural-bias
diagnostics** (disposition effect, overtrading, momentum-chasing, anchoring),
then distils profitable roundtrips into if-then **shadow rules** and backtests
the counterfactual "shadow" portfolio against the user's real trades for delta-PnL
attribution. The audit is a **flat report** — not persisted as a queryable fact.

### AutoHedge — thin autonomous crypto hedge-fund pipeline
A deliberately **thin** autonomous loop (`autohedge/`): a fixed
`Director → Quant → Risk → Execution` agent chain (plus sentiment), structured
JSON output, risk-first sizing, on-chain (Solana/Jupiter) execution. Strength:
clean risk-first pipeline ergonomics. Weakness: no memory of which agent was
*right* before, no behavioural feedback, single-venue, no graph substrate. Its
role taxonomy already maps onto our `SwarmRole`.

### FinceptTerminal — Bloomberg-class terminal
A desktop terminal (`fincept-qt/`) wrapping **100+ data sources** (yfinance,
OECD/BEA/BoE macro, akshare, financial-datasets, …), **16 broker execution**
integrations, and the richest **persona registry** of the three: **37 persona
agents** across `TraderInvestorsAgent`, `hedgeFundAgents` (Renaissance-style
archetypes), `GeopoliticsAgents`, and `EconomicAgents`. Critically, each persona
config (`TraderInvestorsAgent/configs/agent_definitions.json`) carries an
**explicit, scored decision framework** — Buffett: ROE≥15% / D/E<0.5 /
owner-earnings yield; Graham-style value gates; named `scoring_weights` and
required `line_items`. That structure is the seed for `KG-2.28`. Weakness: no KG,
no calibration/reputation, frameworks live in prose + weights but aren't
executable, queryable graph facts.

---

## 2. Gap table

| Capability | Vibe-Trading | AutoHedge | FinceptTerminal | agent-utilities (before) | agent-utilities (after) |
|---|:---:|:---:|:---:|:---:|:---:|
| Bull/bear + persona-voice debate w/ risk veto | partial | basic | persona prose | ✅ `debate_engine` + `investor_debate` | ✅ |
| Role-weighted swarm consensus | preset swarms | fixed chain | — | ✅ `trading_swarm` | ✅ |
| Forensic earnings screen (M/Z/F/Sloan) | — | — | data only | ✅ `forensic_screener` (engine) | ✅ |
| Portfolio opt / VaR / regime / alpha factors | partial | basic | data | ✅ KG-2.6 suite | ✅ |
| **Trade-journal behavioural-bias audit** | ✅ (flat report) | — | — | ❌ | ✅ **KG-2.26** (KG-persisted) |
| **Shadow account (trader profile as signal)** | ✅ (report) | — | — | ❌ | ✅ **KG-2.26** |
| **Agent calibration / reputation feedback** | — | — | — | ❌ | ✅ **KG-2.27** (Brier → swarm weights) |
| **Executable, queryable persona heuristics** | — | — | prose + weights | partial (voice only) | ✅ **KG-2.28** (OWL + evaluator) |
| 100+ data sources / many brokers | some | 1 venue | ✅ | partial | (out of scope) |
| Graph-native provenance / OWL reasoning | — | — | — | ✅ epistemic-graph + OWL | ✅ |

The three gaps closed here are the ones that are **both** absent from us **and**
multiplied by our substrate — not the breadth gaps (data-source/broker count),
which are integration surface rather than capability.

---

## 3. Hidden value-adds — where our substrate makes a borrowed capability stronger

The three frameworks treat these features as terminal outputs. Because we have an
**epistemic graph + OWL ontology + calibration-feedback loop**, the same feature
becomes an input to further reasoning:

- **Trade-journal audit → learning signal (KG-2.26).** Vibe-Trading produces a
  PDF-shaped report. We persist the profile + each bias as `:TraderProfile` and
  `:BehavioralBias` nodes (`EXHIBITED_BY`), so a *future* Bull/Bear debate or the
  risk officer can **cite** them: "this account exhibits a HIGH disposition
  effect → weight the bear's stop-loss discipline up." The audit becomes a fact
  reasoned over, with provenance, not a one-shot artifact.
- **Calibration feedback (KG-2.27).** Palantir AIP and Fincept have personas but
  no memory of who was *right*. We record each persona's directional calls vs
  outcomes, score them with the engine's **`brier_score`** kernel, and feed the
  calibration back into the **weighted `SwarmConsensus`** so historically-accurate
  voices outvote the rest. Each score is an `:AgentCalibration` node
  (`CALIBRATION_OF` the agent) — a queryable reputation, e.g. "which persona has
  the best Brier on tech shorts?"
- **Persona heuristics as OWL facts (KG-2.28).** Fincept's frameworks live in
  prose + weights. We make them `:DecisionHeuristic` OWL individuals
  (`HEURISTIC_OF` a persona `:Agent`) with a deterministic evaluator, so the
  graph can answer "which personas' value criteria does ACME pass?" and a Buffett
  bull cites the **exact** passing/failing rule. The engine's forensic kernels
  feed Burry's short triggers directly — borrowed structure, grounded in our
  reasoned-over numbers.

In every case the differentiator is the same: **reasoned-over facts + KG
provenance + calibration feedback** turn a static borrowed feature into a
self-improving one.

---

## 4. Implemented gaps — API + wiring

### a. Trade-journal bias auditor + shadow account — `CONCEPT:KG-2.26`
`agent_utilities/domains/finance/trade_journal.py`

```python
from agent_utilities.domains.finance import TradeJournalAuditor, Roundtrip
auditor = TradeJournalAuditor()
profile = auditor.audit("acct_42", roundtrips)   # win rate, PnL ratio, max DD, 4 biases
profile.bias("disposition_effect").severity      # low | medium | high
auditor.persist(profile, backend)                # -> :TraderProfile + :BehavioralBias KG nodes
```

- **Real maths:** FIFO-matched roundtrips in; win rate, avg holding period, PnL
  ratio (avg-win / avg-|loss|), cumulative-equity max drawdown; disposition
  (loser/winner hold ratio), overtrading (busy-vs-quiet-day PnL gap),
  momentum-chasing (buys >3% above own prior buy), anchoring (<5% entry-price CV).
- **Wiring → KG:** `to_batch()` emits the standard `ExtractionBatch` (GraphNode /
  EnrichmentEdge) and `persist()` writes it through the **same `write_batch` →
  `GraphBackend`** path every enrichment source uses. `None` backend degrades to a
  no-op so the audit runs fully offline.

### b. Agent calibration / reputation tracking — `CONCEPT:KG-2.27`
`agent_utilities/domains/finance/calibration_tracker.py`

```python
from agent_utilities.domains.finance import (
    CalibrationTracker, apply_calibration_to_swarm,
)
t = CalibrationTracker()
t.record_call("quant_01", direction=+1, confidence=0.85, subject="ACME")
t.record_outcome("quant_01", realized_direction=+1, subject="ACME")
t.score("quant_01")                       # accuracy + Brier + calibration in [0,1]
apply_calibration_to_swarm(swarm, t)      # LIVE wire: mutates swarm.config.role_weights
```

- **Brier:** engine `client.finance.brier_score` when reachable, vetted local
  fallback `mean((f-o)^2)` offline; calibration = `clamp(1 - 2·brier, 0, 1)`.
- **Wiring → SwarmConsensus:** `calibrated_role_weights()` scales each role's base
  weight by the average calibration of its agents (floored, never zeroed);
  `apply_calibration_to_swarm()` writes them back into the live swarm's
  `config.role_weights`, which the **existing** `TradingSwarm.analyze` weighted
  aggregation already consumes — so the next `analyze()` lets a high-calibration
  quant outvote a miscalibrated sentiment agent. A live-path test proves a swarm's
  decision flips HOLD → BUY after calibration.
- **Wiring → KG:** `persist()` writes `:AgentCalibration` nodes (`CALIBRATION_OF`).

### c. Persona decision-heuristic enrichment — `CONCEPT:KG-2.28`
`agent_utilities/domains/finance/persona_heuristics.py`

```python
from agent_utilities.domains.finance import evaluate_persona, evaluate_all
ev = evaluate_persona("graham_investor", {"pe": 11, "pb": 1.1, "margin_of_safety": 0.4})
ev.verdict        # bullish | neutral | bearish | insufficient_data
ev.citation()     # names the exact passing/failing rules
```

- **Structured heuristics:** `PERSONA_HEURISTICS` attaches typed `Heuristic`
  rules to Graham (P/E<15, P/B<1.5, MoS≥30%), Buffett (ROE/ROIC/owner-earnings/
  D/E), Burry (forensic short triggers on Sloan accruals / Beneish-M / Altman-Z),
  Damodaran (DCF), Druckenmiller (regime), and a Lynch PEG lens. The Burry lens is
  **inverted**: satisfying the triggers ⇒ *bearish*. Missing metrics yield
  `unknown` (never a silent pass).
- **Wiring → debate/screen:** `DebateContext` gained a `metrics` field;
  `DebateEngine.persona_heuristic_evidence()` / `_heuristic_block()` fold each
  bound persona's verdict into its bull/bear prompt **by default when metrics are
  present** (generic path untouched otherwise). A live-path test asserts the
  Graham verdict + `KG-2.28` marker appear in the prompt block.
- **Wiring → KG/OWL:** `persona_heuristics_batch()` / `seed_persona_heuristics()`
  emit `:DecisionHeuristic` nodes (`HEURISTIC_OF`); matching OWL classes
  (`:DecisionHeuristic`, `:AgentCalibration`, `:TraderProfile`, `:BehavioralBias`)
  and object properties (`:heuristicOf`, `:calibrationOf`, `:exhibitedBy`) were
  added to `agent_utilities/knowledge_graph/ontology_quant.ttl`.

---

## 5. Concept IDs

| ID | Capability | Borrowed from | Module |
|---|---|---|---|
| `KG-2.26` | Trade-journal bias auditor + shadow account | Vibe-Trading | `domains/finance/trade_journal.py` |
| `KG-2.27` | Agent calibration / reputation tracking | (novel; gap vs Palantir/Fincept) | `domains/finance/calibration_tracker.py` |
| `KG-2.28` | Persona decision-heuristic enrichment | FinceptTerminal | `domains/finance/persona_heuristics.py` |

All three degrade cleanly offline (lazy engine + optional KG backend), export via
`domains/finance/__init__.py`, and are covered by tests under
`tests/unit/finance/`.
