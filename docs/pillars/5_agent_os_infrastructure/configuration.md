# Agent Configuration

The `agent-utilities` ecosystem uses a standardized set of environment variables for configuring Language Models (LLMs) and Embeddings. This tiered architecture ensures flexibility across different execution environments and task complexities.

## Standardized LLM Environment Variables

These variables should be defined in your `.env` file at the root of your workspace or project.

### Core LLM Settings (Tier 0)
Fallback configurations for all agents.
*   `LLM_PROVIDER`: The primary LLM provider (e.g., `openai`, `anthropic`, `ollama`).
*   `LLM_MODEL_ID`: The primary model identifier (e.g., `gpt-4o`, `claude-3-5-sonnet-latest`).
*   `LLM_BASE_URL`: (Optional) Base URL for the LLM API endpoint.
*   `LLM_API_KEY`: The API key for the primary LLM provider.

### Lite LLM Settings (Tier 1)
Optimized for standard agent tasks, high speed, and lower cost. Defaults to Core settings if not provided.
*   `LITE_LLM_PROVIDER`: The provider for lite tasks.
*   `LITE_LLM_MODEL_ID`: The model identifier for lite tasks (e.g., `gpt-4o-mini`, `claude-3-haiku`).
*   `LITE_LLM_BASE_URL`: (Optional) Base URL for the lite LLM API.
*   `LITE_LLM_API_KEY`: API key for the lite provider.

### Super LLM Settings (Tier 2)
Reserved for expert graph agents and orchestration reasoning. Defaults to Core settings if not provided.
*   `SUPER_LLM_PROVIDER`: The provider for complex reasoning tasks.
*   `SUPER_LLM_MODEL_ID`: The model identifier for complex reasoning.
*   `SUPER_LLM_BASE_URL`: (Optional) Base URL for the super LLM API.
*   `SUPER_LLM_API_KEY`: API key for the super provider.

### Role-Specific Overrides
*   `ROUTER_MODEL`: Model used specifically by the graph router (defaults to `LITE_LLM_MODEL_ID`).
*   `KG_MODEL_ID`: Model used specifically for Knowledge Graph inferences (defaults to `LITE_LLM_MODEL_ID`).

### Embedding Settings
Standardized settings for vector embeddings (e.g., RAG, semantic search).
*   `EMBEDDING_PROVIDER`: The provider for embeddings. Defaults to `LLM_PROVIDER`.
*   `EMBEDDING_MODEL_ID`: The model identifier for embeddings (e.g., `text-embedding-nomic-embed-text-v2-moe`).
*   `EMBEDDING_BASE_URL`: (Optional) Base URL for the embeddings API. Defaults to `LLM_BASE_URL`.
*   `EMBEDDING_API_KEY`: API key for embeddings. Defaults to `LLM_API_KEY`.

## Example `.env`

```env
# Core settings
LLM_PROVIDER=openai
LLM_MODEL_ID=gpt-4o
LLM_BASE_URL=http://10.0.0.18:1234/v1
LLM_API_KEY=your_api_key

# Lite settings
LITE_LLM_PROVIDER=openai
LITE_LLM_MODEL_ID=gpt-4o-mini

# Super settings
SUPER_LLM_PROVIDER=anthropic
SUPER_LLM_MODEL_ID=claude-3-5-sonnet-latest

# Embeddings
EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL_ID=text-embedding-nomic-embed-text-v2-moe
```
