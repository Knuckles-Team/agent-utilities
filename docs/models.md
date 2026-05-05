# Multi-Model Registry & Configuration

The `agent-utilities` ecosystem supports a dynamic **multi-model registry**. Instead of hardcoding a single LLM, you can configure a suite of models with different routing tiers (`light`, `medium`, `heavy`, `reasoning`) and capability tags (`vision`, `code`, `fast`).

When the graph orchestrator dispatches tasks to specialist agents, it uses `pick_for_task()` to autonomously select the right model based on the task's required complexity and capabilities.

## The `MODELS_CONFIG` File

Define your models in a JSON or YAML file and set the `MODELS_CONFIG` environment variable to its path:

```bash
export MODELS_CONFIG=/path/to/models.json
```

### Example `models.json`

```json
{
  "models": [
    {
      "id": "fast-local",
      "name": "Local LM Studio",
      "provider": "openai",
      "model_id": "llama-3.2-3b-instruct",
      "base_url": "http://localhost:1234/v1",
      "tier": "light",
      "is_default": true,
      "tags": ["fast"],
      "cost": {
        "input": 0.0,
        "output": 0.0
      }
    },
    {
      "id": "gpt-mini",
      "name": "GPT-4o Mini",
      "provider": "openai",
      "model_id": "gpt-4o-mini",
      "api_key_env": "OPENAI_API_KEY",
      "tier": "medium",
      "tags": ["code", "tools"],
      "cost": {
        "input": 0.15,
        "output": 0.60
      }
    },
    {
      "id": "claude-opus",
      "name": "Claude 3 Opus",
      "provider": "anthropic",
      "model_id": "claude-3-opus-20240229",
      "api_key_env": "ANTHROPIC_API_KEY",
      "tier": "heavy",
      "tags": ["reasoning", "complex"],
      "cost": {
        "input": 15.0,
        "output": 75.0
      }
    }
  ]
}
```

## Schema Breakdown (`ModelDefinition`)

- `id`: Stable identifier used by the system.
- `name`: Human-readable name displayed in UIs.
- `provider`: Provider string (`openai`, `anthropic`, `google-gla`, `ollama`).
- `model_id`: The actual model string sent to the provider API.
- `base_url`: Override for the API endpoint (useful for local models).
- `api_key_env`: Environment variable name holding the API key. Null means no auth needed.
- `tier`: The routing tier (`light`, `medium`, `heavy`, `reasoning`).
- `tags`: List of string capabilities (e.g. `vision`, `tools`).
- `cost`: Per-1M-token cost (`input`, `output`). Use zero for local models.
- `is_default`: Boolean indicating the default fallback model.

## Autonomous Routing

In your code, you don't need to specify model strings. The graph orchestrator resolves the best fit dynamically:

```python
# Selects a fast model for a simple summarization task
model = registry.pick_for_task(complexity="light")

# Selects a heavy model that explicitly supports code generation
model = registry.pick_for_task(complexity="heavy", required_tags=["code"])
```

### Selection Algorithm

1. **Tag Filtering**: Only models with all `required_tags` are considered.
2. **Tier Matching**: An exact match on `complexity` tier is preferred.
3. **Graceful Fallback**: If an exact tier isn't available, heavier tiers fall back to reasoning, while lighter tiers fall back through medium/heavy.
4. **Safety Net**: If tag filtering eliminates all candidates, it retries without tags before defaulting to the `is_default` model.
