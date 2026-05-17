from agent_utilities.core.embedding_utilities import create_embedding_model
from agent_utilities.knowledge_graph.backends.ladybug_backend import LadybugBackend

backend = LadybugBackend()
emb = create_embedding_model()
q_emb = emb.get_text_embedding("multi-agent orchestration")

try:
    res = backend.execute(
        "CALL QUERY_VECTOR_INDEX('Article', 'idx_article_embedding', $emb, 5) YIELD node, score RETURN node, score",
        {"emb": q_emb},
    )
    print("Vector Search Result:", res)
except Exception as e:
    print("Error:", e)
