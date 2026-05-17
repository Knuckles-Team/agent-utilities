from agent_utilities.knowledge_graph.backends.ladybug_backend import LadybugBackend

backend = LadybugBackend()
print(backend.db_path)
res = backend.execute("CALL SHOW_TABLES() YIELD name RETURN name")
print(res)
