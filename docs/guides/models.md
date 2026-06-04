# Multi-Model Registry & Configuration

The `agent-utilities` ecosystem supports a dynamic **multi-model registry**. Instead of hardcoding a single LLM, you can configure a suite of models with different routing tiers (`light`, `medium`, `heavy`, `reasoning`) and capability tags (`vision`, `code`, `fast`).

When the graph orchestrator dispatches tasks to specialist agents, it uses `pick_for_task()` to autonomously select the right model based on the task's required complexity and capabilities.

## The `config.json` File

Define your models directly in the unified `config.json` file located in the XDG-compliant path: `~/.config/agent-utilities/config.json`.

### Example `config.json`

```json
{
  "chat_models": [
    {
      "id": "gpt-4o-mini",
      "provider": "openai",
      "intelligence_level": "normal",
      "supports_json": true,
      "vision": true,
      "reasoning": false,
      "tools_enabled": true,
      "parallel_instances": 1,
      "can_route": true,
      "can_kg": true
    },
    {
      "id": "claude-3-opus",
      "provider": "anthropic",
      "intelligence_level": "super",
      "supports_json": true,
      "vision": true,
      "reasoning": true,
      "tools_enabled": true,
      "parallel_instances": 1,
      "can_route": true,
      "can_kg": false
    }
  ],
  "embedding_models": [
    {
      "id": "text-embedding-3-small",
      "provider": "openai",
      "parallel_instances": 100,
      "chunk_size": 1536
    }
  ]
}
```

## Schema Breakdown (`ChatModelConfig` & `EmbeddingModelConfig`)

- `id`: Stable identifier used by the system (e.g., `gpt-4o-mini`).
- `provider`: Provider string (`openai`, `anthropic`, `google`, `ollama`).
- `intelligence_level`: The routing tier (`light`, `normal`, `super`).
- `supports_json`: Boolean indicating if the model supports native JSON output.
- `vision`: Boolean indicating if the model supports multimodal vision inputs.
- `reasoning`: Boolean indicating if the model supports chain-of-thought reasoning constraints.
- `tools_enabled`: Boolean indicating if the model supports function calling.
- `parallel_instances`: Integer indicating concurrency limits for this model.
- `context_window`: Optional integer specifying max context tokens.
- `can_route`: Boolean indicating if this model is eligible to act as a routing coordinator.
- `can_kg`: Boolean indicating if this model is eligible for Knowledge Graph extraction tasks.

## Autonomous Routing

In your code, you don't need to specify model strings. Two complementary
selection surfaces exist:

**1. `config` registry helpers** — pick a model by `intelligence_level`
(`light` / `normal` / `super`):

```python
from agent_utilities.core.config import config

# Primary (intelligence_level='normal', fallback to first)
model = config.default_chat_model

# Lightweight (intelligence_level='light')
model = config.lite_chat_model

# Super/heavy (intelligence_level='super')
model = config.super_chat_model
```

**2. `ModelRegistry.pick_for_task()`** — the graph orchestrator resolves the
best fit dynamically by routing tier (`light` / `medium` / `heavy` /
`reasoning`) plus capability tags:

```python
from agent_utilities.models.model_registry import ModelRegistry

registry = ModelRegistry(...)
model = registry.pick_for_task(complexity="heavy", required_tags=["vision"])
```

### Selection Algorithm (`pick_for_task`)

1. **Tag Filtering**: Only models carrying every `required_tag` are considered (AND semantics).
2. **Tier Matching**: An exact match on the requested tier is preferred.
3. **Graceful Fallback**: If an exact tier isn't available, a tier-specific fallback order applies — heavier tiers fall back to reasoning, lighter tiers fall back through medium/heavy.
4. **Safety Net**: If tag filtering eliminates every candidate, it re-tries without tags; as a last resort it returns the registry default.
