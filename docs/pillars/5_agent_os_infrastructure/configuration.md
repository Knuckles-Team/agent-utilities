# Agent Configuration

The `agent-utilities` ecosystem uses a standardized XDG-compliant JSON configuration for its language models (LLMs), embeddings, and system properties. This architecture ensures a single source of truth across all tools and agent sessions.

## Unified Configuration (`config.json`)

The primary configuration file is located at `~/.config/agent-utilities/config.json`. This file is dynamically hot-reloadable.

### Example `config.json`

```json
{
  "chat_models": [
    {
      "id": "gpt-4o",
      "provider": "openai",
      "intelligence_level": "normal",
      "supports_json": true,
      "vision": true
    },
    {
      "id": "gpt-4o-mini",
      "provider": "openai",
      "intelligence_level": "light",
      "supports_json": true,
      "vision": true
    },
    {
      "id": "claude-3-5-sonnet-latest",
      "provider": "anthropic",
      "intelligence_level": "super",
      "supports_json": true,
      "vision": true
    }
  ],
  "embedding_models": [
    {
      "id": "text-embedding-nomic-embed-text-v2-moe",
      "provider": "openai",
      "base_url": "http://10.0.0.18:1234/v1"
    }
  ]
}
```

## Model Properties

*   `id`: The specific model string identifier to pass to the API.
*   `provider`: The API provider (e.g., `openai`, `anthropic`, `ollama`).
*   `intelligence_level`: Categorizes the model's capability (`light`, `normal`, `super`). Replaces legacy `LITE_LLM`, `SUPER_LLM` tier routing.
*   `supports_json`: Boolean indicating if the model natively supports JSON mode.
*   `vision`: Boolean indicating if the model supports multimodal inputs (images).

## Environment Variables (`.env`)

Environment variables are now strictly reserved for sensitive credentials. They are decoupled from routing flags.

```env
# Sensitive Credentials Only
LLM_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
```
