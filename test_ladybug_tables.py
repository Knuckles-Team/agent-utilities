from agent_utilities.knowledge_graph.backends.ladybug_backend import LadybugBackend

backend = LadybugBackend()
res = backend.conn.execute("CALL SHOW_TABLES() YIELD name RETURN name")
while res.has_next():
    print(res.get_next())
