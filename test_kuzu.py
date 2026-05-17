import kuzu

db = kuzu.Database("/home/apps/workspace/agent-packages/agent-utilities/tests/test_db")
conn = kuzu.Connection(db)
res = conn.execute("RETURN array_cosine_similarity([1.0, 2.0], [1.0, 2.0]) AS sim")
while res.has_next():
    print(res.get_next())
