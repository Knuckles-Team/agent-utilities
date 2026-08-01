# Design Document: A forecast is committed before the run and can never be edited to match the result afterward

CONCEPT:AU-AHE.harness.forecasting-discipline

> `agent_utilities/harness/forecasting.py`.

## Decision — predict-before-run, then resolve; an unresolved forecast is immutable-in-spirit once written

The module docstring (`forecasting.py:4-19`) operationalizes a specific
research-craft habit — Hamming/Karpathy's predict-before-run + calibration
discipline: before an experiment runs, the researcher writes down a
prediction *and* a confidence; only after the run does the forecast resolve
against the observed result, over many forecasts yielding a Brier score,
hit-rate, and calibration curve. `Forecast` (`forecasting.py:41-58`) encodes
this structurally: a forecast is created with `resolved=False` the moment a
prediction is committed, and is resolved later by filling in `actual` and
flipping `resolved` — the ordering is load-bearing.

**The rejected alternative is scoring after the fact — "staring at the
outputs" retrospectively and rationalizing what was expected, without a
pre-committed, timestamped prediction.** That's exactly the unmeasured
intuition this module exists to replace with a structured, repeatable
feedback loop: without a forecast committed *before* the result is known,
there's no way to distinguish genuine predictive skill from hindsight bias.
The module is explicitly the **longitudinal register** — distinct from
`reliability_scorers.BrierSkillScorer` (AHE-3.1), which scores one
probabilistic forecast against one realized outcome — accumulating many
predict-before-run forecasts across experiments into a calibration
scoreboard over the whole set.

## Risk Assessment

- **Blast Radius**: `agent_utilities/harness/forecasting.py` only — a
  standalone register, not wired into any gate.
- **Backward Compatible**: Yes — additive instrumentation.
- **Known weak point**: nothing in `Forecast`/`ForecastBoard` mechanically
  prevents a caller from calling the resolve step with a fabricated
  `predicted` value logged just before resolution (i.e. the "predict before
  run" discipline is a calling convention, not an enforced invariant — the
  dataclass only guarantees a forecast can't be *edited* after resolution,
  not that it was genuinely written down beforehand).
