# Delivery plan — AU-QUAL-01

1. Capture a fresh clone failing inventory by test name, fixture, package, and root cause. Separate source failures from missing declared dependencies.
2. Repair the `tiny` gateway result mapping and obsolete bootstrap test seams. Assert both allowed and denied responses at the served boundary.
3. Replace connector shape validation with engine-composed schema input and pin its typed contract across AU, SDK, and engine.
4. Repair each remaining AU failure group with deterministic fixtures. Seed benchmark randomness at the relevant owner seam; do not lower assertions to hide failure.
5. Fix test typing by directory, then remove the mypy test exclusion and run the complete suite. Keep a reviewable before/after error count.
6. Provision missing CI tools and local services in workflow jobs. Move time-triggered liveness to an always visible hosted/scheduled check; remove the obsolete documentation deployment blocker and ensure Pages publishing is independently reported.
7. Run exact revision focused tests, full repository gates, fresh fork CI and release-only live certification as applicable. Record results by requirement.

The gate migration is complete only when a contributor who has no sibling repository or private deployment can obtain the same PR result. Existing security and privacy assertions stay active.
