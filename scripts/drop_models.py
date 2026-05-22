from agent_utilities.mcp.kg_server import _get_engine

engine = _get_engine()
try:
    engine.backend.conn.execute("DROP TABLE LanguageModel;")
    print("Dropped LanguageModel")
except Exception as e:
    print(f"Error dropping LanguageModel: {e}")
try:
    engine.backend.conn.execute("DROP TABLE EmbeddingModel;")
    print("Dropped EmbeddingModel")
except Exception as e:
    print(f"Error dropping EmbeddingModel: {e}")
