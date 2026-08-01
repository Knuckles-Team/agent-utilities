# Design Document: A real four-rule promotion validator replaces an injectable slot that production never actually filled

CONCEPT:AU-AHE.harness.promotion-governance-validator

> `agent_utilities/knowledge_graph/research/promotion_governance.py`.

## Decision — `GovernedAutoMerger` gets a real validator as its production default, checked against four independently-observable rules

The module docstring (`promotion_governance.py:4-11`) states the prior gap
directly: `GovernedAutoMerger` always accepted an injected
`governance_validator`, but until this module, only test mocks were ever
injected — so a "governed" auto-merge in production either held every
proposal (validator required but absent) or validated nothing at all. This
module is the real validator and the merger now constructs it as the
DEFAULT whenever an engine is available. A candidate must clear four rules:
`MergePolicy` quality/completeness thresholds, SHACL governance-shape
conformance, a recorded regression-gate `pass` (deferring to the live check
when no record exists), and no match against active `forbid`-kind
constitution rules.

**The rejected alternative is exactly the prior state: an unimplemented
injection point that functioned as either fail-open (validate nothing) or
fail-closed (hold everything)**, neither of which is actual governance — it's
governance-shaped plumbing with no governor. A second decision inside the
same module: the validator is "deliberately conservative where it CAN
observe... and non-blocking where governance data simply does not exist."
**The rejected alternative there is failing closed on missing data
everywhere** — treating "no regression-gate record" the same as "regression
gate failed." Instead only an actual negative signal (a matching forbid
rule, a recorded hold, a SHACL violation) blocks; absence of data defers to
the merger's own live check. The master switch `KG_GOLDEN_AUTO_MERGE` stays
`False` by default regardless of what the validator would decide.

## Risk Assessment

- **Blast Radius**: `agent_utilities/knowledge_graph/research/promotion_governance.py`,
  `agent_utilities/knowledge_graph/research/auto_merge.py`
  (`GovernedAutoMerger`), `agent_utilities/knowledge_graph/adaptation/failure_analyzer.py`
  (regression-gate records this validator reads).
- **Backward Compatible**: Yes — existing test-mock injection still works;
  this only changes what gets constructed by default in production.
- **Known weak point**: "non-blocking where governance data simply does not
  exist" means a proposal in a domain with no SHACL shape, no recorded
  regression-gate verdict, and no matching constitution rule sails through
  on rules 1 and nothing else — governance coverage is only as complete as
  the shapes/rules that have actually been authored for that domain.
