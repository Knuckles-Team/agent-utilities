import logging
import os
from typing import Any, Dict, List, Optional
from ..types import (
    PipelinePhase,
    PipelineContext,
    PhaseResult,
)

logger = logging.getLogger(__name__)

# LM Studio defaults (matching vector-mcp and maintenance.py patterns)
LM_STUDIO_URL = os.environ.get("LLM_BASE_URL", "http://localhost:1234/v1")
EMBEDDING_MODEL = os.environ.get(
    "EMBEDDING_MODEL", "text-embedding-nomic-embed-text-v2-moe"
)


def _generate_embedding_batch(texts: List[str]) -> Optional[List[List[float]]]:
    """Generate embeddings via LM Studio's OpenAI-compatible endpoint.

    Uses the same pattern as vector-mcp's create_embedding_model() and
    maintenance.py's generate_embedding(), connecting to the local LM Studio
    server at LLM_BASE_URL/embeddings.
    """
    try:
        import requests

        payload = {"model": EMBEDDING_MODEL, "input": texts}
        response = requests.post(
            f"{LM_STUDIO_URL}/embeddings", json=payload, timeout=30
        )
        response.raise_for_status()
        data = response.json()
        if "data" in data:
            # Sort by index to maintain order
            sorted_data = sorted(data["data"], key=lambda x: x.get("index", 0))
            return [item["embedding"] for item in sorted_data]
    except ImportError:
        logger.warning("requests package not available for embedding generation")
    except Exception as e:
        logger.warning(f"Embedding generation failed: {e}")
    return None


def _generate_embedding_llamaindex(texts: List[str]) -> Optional[List[List[float]]]:
    """Generate embeddings via LlamaIndex create_embedding_model (vector-mcp pattern).

    Falls back to this method if the direct HTTP approach is preferred to use
    the same embedding model factory as vector-mcp.
    """
    try:
        from agent_utilities.embedding_utilities import create_embedding_model

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
    ctx: PipelineContext, deps: Dict[str, PhaseResult]
) -> Dict[str, Any]:
    """Generate semantic embeddings for graph nodes using LM Studio.

    Uses the local LM Studio OpenAI-compatible endpoint (same pattern as
    vector-mcp project). Embeds node descriptions/content for hybrid
    graph+vector retrieval in LadybugDB.
    """
    if not ctx.config.enable_embeddings:
        return {"status": "skipped", "reason": "embeddings disabled"}

    graph = ctx.nx_graph
    embeddings_generated = 0
    errors = 0

    # Collect nodes that have text content worth embedding
    nodes_to_embed = []
    for node_id, data in graph.nodes(data=True):
        # Skip if already has embedding
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
        if text and len(text) > 10:  # Skip very short texts
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
        embeddings = _generate_embedding_batch(texts)
        if embeddings is None:
            embeddings = _generate_embedding_llamaindex(texts)

        if embeddings and len(embeddings) == len(batch):
            for (node_id, _), embedding in zip(batch, embeddings):
                graph.nodes[node_id]["embedding"] = embedding
                embeddings_generated += 1
        else:
            errors += len(batch)
            logger.warning(f"Failed to embed batch {i//batch_size + 1}")

    return {
        "status": "completed",
        "embeddings_generated": embeddings_generated,
        "errors": errors,
        "total_candidates": len(nodes_to_embed),
    }


embedding_phase = PipelinePhase(
    name="embedding", deps=["parse"], execute_fn=execute_embedding
)
