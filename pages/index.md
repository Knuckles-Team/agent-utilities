# Meristem

Meristem is the in-run agent harness. It owns the run loop, dependency
execution, context assembly, and bounded host ports. It applies the decisions
it receives. It holds no store and serves no MCP, API or A2A. Arbora serves
those, and Medulla holds the facts.

## Specification delivery

The generated [spec delivery dashboard](spec-delivery/) computes spec status
from this repository at read time. Status files are not committed. It distinguishes
documented, source-landed, and accepted work; a page snapshot does not claim a
release or deployment.

Browse the [public specification sources](https://github.com/Knuckles-Team/meristem/tree/main/specs)
for the contracts and evidence behind the dashboard.
