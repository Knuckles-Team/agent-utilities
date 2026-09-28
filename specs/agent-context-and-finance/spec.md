# Agent context, retrieval feedback and finance roles

**ID:** AU-CONTEXT-001 · **Owner:** agent-utilities · **Delivery:** SPECIFIED; no acceptance claim.
**Items:** EH-394, EH-397–EH-399, EH-419, EH-423, EH-513, EH-704. EH-400 supplies EG invalidation events; AU cache consumption is specified in [AU-SEC-01](../agent-security-policy/spec.md). Other EG, SDK and graph-os owners implement their own storage, transport and UI obligations.

## User outcomes

An agent receives current, cited, budgeted context selected through durable EG retrieval and Decide records. Finance agents can explain a position or signal with reproducible math and evidence, support paper workflows, and abstain when evidence is inadequate. Informational recommendations never become orders merely because a model produced them.

## Requirements

1. Retrieve candidate paths and evidence from EG using verified tenant/graph scope. AU compiles a context package from exact tokenizer weights, model window, cost and latency budgets. It must preserve cited range/source IDs and abstain if no admissible package fits.
2. A RetrievalPath template is keyed by task class and composed-schema digest. AU submits an intent/RunSpec; EG owns ranking, certified optimization, feedback records and durable hard negatives. AU cannot self-promote a template from its own unverified outcome.
3. A corpus re-embed request is a governed AU orchestration workflow under a capacity lease. It asks EG to construct a shadow ANN generation, checks population-stability and retrieval-quality receipts, then proposes activation. EG performs atomic swap and rollback. The model training adapter may be external; no heavy training dependency enters AU base install.
4. AU proposes changes to embedding admission classes from never-retrieved, independently observed feedback; EG or a reviewer accepts the table. AU does not directly rewrite durable admission policy.
5. Finance roles in AU are agent execution and explanation only: `trading_swarm`, `debate_engine`, `investor_debate`, `research_autopilot`, and `persona_heuristics`. Deterministic finance math and durable datasets belong in EG; exchange/account feeds and write-back transport belong in the SDK; graph-os hosts scheduling and user surfaces. Delete duplicated AU math and connector modules only after parity.
6. A finance explanation states strategy version, evidence IDs, calculation inputs/results, risk assumptions and uncertainty. It distinguishes observation, proof and model claim. Per-holding/watchlist recommendations are calibrated, may abstain, and are informational `AnalysisSnapshot` records.
7. Paper actions use an isolated simulation account. Live orders require the SDK governed write-back contract, verified human approval lease, account capability and exact idempotency key. No AU model text or UI payload can grant authority.

## Acceptance

Real AU entry points call the legal owners; context citations and budgets are exact; finance role output is reproducible; denied and uncertain cases abstain; code/dependency deletions and full quality gates pass at exact merged heads. A local source branch, paper simulation, or passing unit test alone does not establish a live-order capability.
