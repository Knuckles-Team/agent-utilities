import logging
from typing import Any

from ..types import (
    PhaseResult,
    PipelineContext,
    PipelinePhase,
)

logger = logging.getLogger(__name__)

# Configuration is now read dynamically from agent_utilities.core.config


def _generate_embedding_batch(texts: list[str]) -> list[list[float]] | None:
    """Generate embeddings via LM Studio's OpenAI-compatible endpoint.

    CONCEPT:KG-2.3

        Uses the same pattern as vector-mcp's create_embedding_model() and
        maintenance.py's generate_embedding(), connecting to the local LM Studio
        server at LLM_BASE_URL/embeddings.
    """
    from agent_utilities.core.config import (
        DEFAULT_EMBEDDING_BASE_URL,
        DEFAULT_EMBEDDING_MODEL_ID,
    )

    url = DEFAULT_EMBEDDING_BASE_URL or "http://localhost:1234/v1"
    model = DEFAULT_EMBEDDING_MODEL_ID

    try:
        import requests

        payload = {"model": model, "input": texts}
        response = requests.post(f"{url}/embeddings", json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        if "data" in data:
            # Sort by index to maintain order
            sorted_data = sorted(data["data"], key=lambda x: x.get("index", 0))
            return [item["embedding"] for item in sorted_data]
    except ImportError:
        logger.warning("requests package not available for embedding generation")
    except requests.exceptions.RequestException as e:
        logger.warning(f"Embedding generation failed: {e}")
        raise ConnectionError(f"Embedding server unreachable: {e}") from e
    except Exception as e:
        logger.warning(f"Embedding generation failed unexpectedly: {e}")
    return None


def _generate_embedding_llamaindex(texts: list[str]) -> list[list[float]] | None:
    """Generate embeddings via LlamaIndex create_embedding_model (vector-mcp pattern).

    Falls back to this method if the direct HTTP approach is preferred to use
    the same embedding model factory as vector-mcp.
    """
    try:
        from agent_utilities.core.embedding_utilities import create_embedding_model

        embed_model = create_embedding_model()
        embeddings = []
        for text in texts:
            emb = embed_model.get_text_embedding(text)
            embeddings.append(emb)
        return embeddings
    except ImportError:
        logger.debug("LlamaIndex embedding not available, using direct HTTP")
    except Exception as e:
        logger.warning(f"LlamaIndex embedding failed: {e}")
    return None


async def execute_embedding(
    ctx: PipelineContext, deps: dict[str, PhaseResult]
) -> dict[str, Any]:
    """Generate semantic embeddings for graph nodes using LM Studio.

    Uses the local LM Studio OpenAI-compatible endpoint (same pattern as
    vector-mcp project). Embeds node descriptions/content for hybrid
    graph+vector retrieval in LadybugDB.
    """
    if not ctx.config.enable_embeddings:
        return {"status": "skipped", "reason": "embeddings disabled"}

    graph = ctx.graph
    embeddings_generated = 0
    errors = 0

    # Collect nodes that have text content worth embedding
    nodes_to_embed = []

    import hashlib

    for node_id, data in graph.nodes(data=True):
        # Skip if already has embedding locally
        if data.get("embedding"):
            continue

        # Combine name + description for rich embedding
        text_parts = []
        if data.get("name"):
            text_parts.append(str(data["name"]))
        if data.get("description"):
            text_parts.append(str(data["description"]))
        if data.get("content"):
            text_parts.append(str(data["content"]))

        text = " ".join(text_parts).strip()
        if not text or len(text) <= 10:  # Skip very short texts
            continue

        content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        data["content_hash"] = content_hash

        # Check if backend already has this exact embedding
        if ctx.backend:
            try:
                result = ctx.backend.execute(
                    "MATCH (n) WHERE n.id = $id RETURN n", {"id": node_id}
                )
                if result and len(result) > 0:
                    existing_node = result[0].get("n")
                    if (
                        existing_node
                        and existing_node.get("content_hash") == content_hash
                        and existing_node.get("embedding")
                    ):
                        data["embedding"] = existing_node["embedding"]
                        continue
            except Exception as e:
                logger.debug(f"Failed to check backend cache for {node_id}: {e}")

        nodes_to_embed.append((node_id, text))

    if not nodes_to_embed:
        return {
            "status": "completed",
            "embeddings_generated": 0,
            "reason": "no nodes to embed",
        }

    # Process in batches of 32 (LM Studio handles batch well)
    batch_size = 32
    for i in range(0, len(nodes_to_embed), batch_size):
        batch = nodes_to_embed[i : i + batch_size]
        texts = [text for _, text in batch]

        # Try direct HTTP first (faster, fewer deps), then LlamaIndex fallback
        try:
            embeddings = _generate_embedding_batch(texts)
        except ConnectionError as e:
            if i == 0:
                logger.error(f"Aborting embedding phase gracefully: {e}")
                return {
                    "status": "skipped",
                    "embeddings_generated": 0,
                    "reason": f"Embedding server unreachable on first batch: {e}",
                }
            embeddings = None

        if embeddings is None:
            embeddings = _generate_embedding_llamaindex(texts)

        if embeddings and len(embeddings) == len(batch):
            for (node_id, _), embedding in zip(batch, embeddings, strict=False):
                graph.nodes[node_id]["embedding"] = embedding
                embeddings_generated += 1
        else:
            errors += len(batch)
            logger.warning(f"Failed to embed batch {i // batch_size + 1}")

    return {
        "status": "completed",
        "embeddings_generated": embeddings_generated,
        "errors": errors,
        "total_candidates": len(nodes_to_embed),
    }


embedding_phase = PipelinePhase(
    name="embedding", deps=["parse"], execute_fn=execute_embedding
)
