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

In your code, you don't need to specify model strings. The graph orchestrator resolves the best fit dynamically:

```python
from agent_utilities.core.config import config

# Retrieves a fast, JSON-capable model
model = config.get_chat_model(intelligence_level="light", require_json=True)

# Retrieves a heavy model with vision support
model = config.get_chat_model(intelligence_level="super", require_vision=True)
```

### Selection Algorithm

1. **Feature Filtering**: Only models that meet required features (`supports_json`, `vision`, `tools_enabled`) are considered.
2. **Tier Matching**: An exact match on `intelligence_level` is preferred.
3. **Graceful Fallback**: If an exact tier isn't available, heavier tiers fall back to reasoning, while lighter tiers fall back through normal/super.
4. **Safety Net**: Defaults to the first matching configuration if multiple are found.
