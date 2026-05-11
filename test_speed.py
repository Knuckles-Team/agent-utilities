import time

import ladybug

start = time.time()
for _ in range(10):
    db = ladybug.Database("test_db")
    conn = ladybug.Connection(db)
    conn.execute("MATCH (n) RETURN COUNT(n)")
    del conn
    del db
end = time.time()
print(f"Time: {end - start:.2f}s")
