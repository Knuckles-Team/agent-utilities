# AgentSpecs Catalog (CONCEPT:AHE-3.4)

The **AgentSpec Catalog Generator** bridges the gap between dynamic agent generation and reproducible, structured deployments.

## Overview
While the Knowledge Graph natively stores agent templates (`TeamConfig`, `Specialist`), enterprise deployments often require portable, JSON-based artifacts. The `AgentSpecGenerator` compiles OWL-driven topologies into structured JSON files known as **AgentSpecs**.

### Key Benefits
- **Shareable Architectures**: Trading bots and research agents can be exported as JSON blueprints and shared across teams.
- **Semantic Consistency**: Every generated AgentSpec is tied to an `ontology_class` mapping back to our OWL ontologies, ensuring strict typing and validation.
- **Reproducibility**: AgentSpecs guarantee that the same tools, capabilities, and system prompts are assembled identically in every environment.

## Implementation Details
Located in `agent_utilities.core.agentspec_catalog`, the generator parses existing nodes in the Knowledge Graph and structures them into the v1.0 AgentSpec JSON schema.

### Example
```python
from agent_utilities import AgentSpecGenerator

spec = AgentSpecGenerator.generate_spec(
    name="QuantBotAlpha",
    description="Mean-reversion trading bot for equities.",
    tools=["fetch_prices", "execute_trade"]
)

AgentSpecGenerator.export_catalog([spec], "catalog.json")
```
