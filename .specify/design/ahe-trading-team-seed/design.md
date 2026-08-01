# Design Document: The trading team is hand-curated and pre-seeded, not left to organically emerge from usage

CONCEPT:AU-AHE.harness.trading-team-seed

> `agent_utilities/graph/trading_team_seed.py`.

## Decision — a blessed, high-`success_rate` `TeamConfig` is seeded upfront for trading, with execution capability concentrated at a single auditable node

The module docstring (`trading_team_seed.py:1-9`) states the shape: a
curated multi-specialist team mapped onto the Finance Pipeline topology
(router → alpha → risk → execution → attribution), seeded with a high
`success_rate` specifically so `compose_team` reuses it as a proven team
"rather than re-synthesising the roster on every trading query." Two more
decisions ride along: the `risk-manager` carries the `critic` capability
(an adversarial sizing veto), and the `execution-specialist` is the SOLE
holder of order/derivative/crypto/prediction tools, "so paper-first
execution gating concentrates at one node."

**The rejected alternative is letting `proven-team-reuse`'s organic
mechanism handle trading the same way it handles every other domain**: start
cold, let ad hoc coalitions form, and only promote a `TeamConfig` once one
proves itself through repeated real usage. For most domains that's the
right default — but for a domain where a bad team composition means real
(even if paper-first) financial actions, waiting for the system to
*discover* a good roster through trial and error is the wrong risk profile.
Seeding a blessed roster upfront skips that discovery period entirely. The
tool-concentration decision has its own rejected alternative: distributing
order/derivative/crypto/prediction tools across multiple specialists, which
would be more flexible but means execution-gating logic (the paper-first
safety boundary) would need to be enforced consistently across every
tool-holding node instead of audited at one.

## Risk Assessment

- **Blast Radius**: `agent_utilities/graph/trading_team_seed.py`,
  `agent_utilities/core/registry/kg_adapter.py` (`compose_team`, proven-team
  matching), the emerald-exchange tool surface.
- **Backward Compatible**: Yes — a seed is additive graph state; removing it
  would fall back to organic team composition for trading queries.
- **Known weak point**: the seed's `success_rate` is a fixed, hand-set value
  representing confidence at authoring time, not a measured outcome from
  real runs — it competes with organically-promoted `TeamConfig`s on the
  same `success_rate`-sort ranking (`proven-team-reuse`) as if it were
  equally earned evidence.
