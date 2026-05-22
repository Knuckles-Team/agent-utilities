import asyncio
import logging

from agent_utilities.core.config import config
from agent_utilities.mcp.kg_server import _get_engine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def ingest_config():
    """Ingests the unified AgentConfig (models + system) into the Knowledge Graph."""
    try:
        engine = _get_engine()
        logger.info("Knowledge Graph Engine initialized.")

        # Reload the config from disk to ensure we have the latest
        config.reload()

        # 1. Ingest Language Models
        for chat_model in config.chat_models:
            model_props = chat_model.model_dump()
            model_id = model_props.pop("id")

            # The schema doesn't have base_url or api_key, these shouldn't go to KG
            model_props.pop("base_url", None)
            model_props.pop("api_key", None)

            engine.add_node(
                node_id=model_id, node_type="LanguageModel", properties=model_props
            )
            logger.info(f"Ingested LanguageModel: {model_id}")

        # 2. Ingest Embedding Models
        for embed_model in config.embedding_models:
            embed_props = embed_model.model_dump()
            embed_id = embed_props.pop("id")

            # The schema doesn't have base_url or api_key
            embed_props.pop("base_url", None)
            embed_props.pop("api_key", None)

            engine.add_node(
                node_id=embed_id, node_type="EmbeddingModel", properties=embed_props
            )
            logger.info(f"Ingested EmbeddingModel: {embed_id}")

        # 3. Ingest SystemConfig
        system_props = {
            "routing_strategy": config.routing_strategy,
            "graph_router_timeout": config.graph_router_timeout,
            "kg_llm_concurrency": config.kg_llm_concurrency,
            "enable_otel": config.enable_otel,
            "a2a_broker": config.a2a_broker,
            "a2a_storage": config.a2a_storage,
        }
        engine.add_node(
            node_id="agent_system_config",
            node_type="SystemConfig",
            properties=system_props,
        )
        logger.info("Ingested SystemConfig: agent_system_config")

        # Validation query
        logger.info("Ingestion completed successfully.")

    except Exception as e:
        logger.error(f"Failed to ingest config to Knowledge Graph: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(ingest_config())
