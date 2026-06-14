# Graph Database Deployment & Multi-Backend Guide

> **CONCEPT:KG-2.0** — High-Scale Graph Database Backends

This guide provides a comprehensive, production-ready reference for deploying, configuring, and maintaining scale-out persistent graph backends for the `agent-utilities` Knowledge Graph engine.

The **Rust-native EpistemicGraph** (`GRAPH_BACKEND=memory`/`file`) is the zero-config, zero-dependency embedded default, and **PostgreSQL + pgvector (pg-age)** is the primary durable/production tier. The demoted **contrib** backends — **LadybugDB**, **FalkorDB**, and **Neo4j** — remain fully supported as opt-in imports (`backends/contrib/`) with Cypher CRUD, HNSW vector embeddings, and relationship indexing. Select any of them via `create_backend()` or the `GRAPH_BACKEND` environment variable.

---

## 📊 Database Comparison Matrix

Select the right database backend based on your scale, infrastructure constraints, and architectural patterns:

> [!NOTE]
> The zero-config embedded default is the **Rust-native EpistemicGraph** (`GRAPH_BACKEND=memory`/`file`), not listed below. **PostgreSQL + pg-age** is the primary durable/production tier. LadybugDB, FalkorDB, and Neo4j are demoted **contrib** backends (`backends/contrib/`) — fully supported but opt-in (require their driver packages).

| Metric / Feature | 🐞 LadybugDB (contrib) | 🦅 FalkorDB (contrib) | 🟢 Neo4j (contrib) | 🐘 PostgreSQL + pg-age |
|---|---|---|---|---|
| **Primary Use Case** | Embedded / Local Dev | Low Latency / High Throughput | Enterprise Property Graph | Relational + Graph Unity |
| **ACID Compliance** | Single-writer WAL | In-Memory (Optional AOF) | Full ACID (Clustered) | Full ACID (Standard Relational) |
| **Vector Search (HNSW)**| ✅ Embedded | ✅ Redis-native | ✅ Neo4j GenAI Index | ✅ `pgvector` HNSW |
| **Full-Text BM25 Search**| — | — | — | ✅ ParadeDB `pg_search` |
| **Scale Limits** | Single Node / Local | Sharded Redis Cluster | Multi-region clusters | High-scale Postgres instances |
| **Zero Configuration** | ✅ Yes | ❌ Requires Docker/Server | ❌ Requires Docker/Server | ❌ Requires Docker/Server |
| **Traversal Engine** | Native SQLite Cypher | Redis RESP-native RESP | Native Bolt JVM | pg-age CSR Traversal |

---

## 🛠️ Docker Deployment and Configuration

Below are the complete, production-grade Docker Compose configurations and environment parameters for deploying each database backend.

> [!NOTE]
> For development environments running on the same host, ensure database ports do not conflict. The compose configurations below use standard testing/staging ports, but can be customized as needed.

### 1. FalkorDB (Low-Latency Graph Workloads)

FalkorDB is a low-latency, high-throughput graph database built on Redis, utilizing the fast RESP protocol and sparse matrix multiplication to achieve near-instantaneous Cypher evaluations.

#### 🐳 Docker Compose (`docker/falkordb.compose.yml`)
```yaml
version: "3.8"

services:
  falkordb-db:
    image: falkordb/falkordb:latest
    container_name: agent-falkordb
    ports:
      - "6380:6379"
    volumes:
      - falkordb-data:/data
    restart: unless-stopped

volumes:
  falkordb-data:
```

#### ⚙️ Configuration Setup
To point your agents to FalkorDB, set the following variables:
```bash
export GRAPH_BACKEND=falkordb
export GRAPH_DB_HOST=localhost
export GRAPH_DB_PORT=6380
export GRAPH_DB_NAME=agent_graph
```

#### 🔌 Python Connection Example
```python
from agent_utilities.knowledge_graph.backends import create_backend

backend = create_backend(
    backend_type="falkordb",
    host="localhost",
    port=6380,
    db_name="agent_graph"
)
# The backend auto-initializes the schema on connection
```

---

### 2. Neo4j (Enterprise Property Graphs)

Neo4j is the industry standard for ACID property graphs, supporting multi-region clustering, fine-grained role permissions, and the APOC (Awesome Procedures on Cypher) plugin suite.

#### 🐳 Docker Compose (`docker/neo4j.compose.yml`)
```yaml
version: "3.8"

services:
  neo4j-db:
    image: neo4j:latest
    container_name: agent-neo4j
    environment:
      - NEO4J_AUTH=neo4j/password
      - NEO4J_PLUGINS=["apoc"]
      - NEO4J_dbms_security_procedures_unrestricted=apoc.*
    ports:
      - "7474:7474"   # HTTP Admin Dashboard
      - "7687:7687"   # Bolt Query Protocol
    volumes:
      - neo4j-data:/data
    restart: unless-stopped

volumes:
  neo4j-data:
```

#### ⚙️ Configuration Setup
Configure your environment to point to Neo4j via the Bolt protocol:
```bash
export GRAPH_BACKEND=neo4j
export GRAPH_DB_URI=bolt://localhost:7687
export GRAPH_DB_USER=neo4j
export GRAPH_DB_PASSWORD=password
```

#### 🔌 Python Connection Example
```python
from agent_utilities.knowledge_graph.backends import create_backend

backend = create_backend(
    backend_type="neo4j",
    uri="bolt://localhost:7687",
    user="neo4j",
    password="password"
)
```

---

### 3. PostgreSQL + pg-age (Relational & Graph Unity)

The PostgreSQL backend leverages `pgvector` for HNSW cosine search, ParadeDB (`pg_search`) for BM25 lexical search, and the `pg-age` extension for highly optimized CSR (Compressed Sparse Row) graph traversals. The `CypherTranspiler` translates Cypher queries directly into PostgreSQL SQL parameters.

#### 🐳 Docker Compose (`docker/pg-age.compose.yml`)
```yaml
version: "3.8"

services:
  pg-age-db:
    image: paradedb/paradedb:latest
    container_name: agent-pg-age
    environment:
      POSTGRES_USER: agent
      POSTGRES_PASSWORD: agent
      POSTGRES_DB: agent_kg
    ports:
      - "5433:5432"   # Mapped to 5433 to avoid host conflicts
    volumes:
      - pg-age-data:/var/lib/postgresql
      - ./pg-age-init:/docker-entrypoint-initdb.d
    restart: unless-stopped

volumes:
  pg-age-data:
```

#### 📜 Initialization Script (`docker/pg-age-init/01-extensions.sql`)
Place the following inside the initialization directory to enable the required extensions automatically on container boot:
```sql
-- pg-age + pgvector initialization script
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- ParadeDB extensions (included in paradedb image)
CREATE EXTENSION IF NOT EXISTS pg_search;

-- pg-age extension (optional, uncomment if installed)
-- CREATE EXTENSION IF NOT EXISTS pg-age;
```

#### ⚙️ Configuration Setup
Point your agents to the Postgres graph backend:
```bash
export GRAPH_BACKEND=postgresql
export GRAPH_DB_URI=postgresql://agent:agent@localhost:5433/agent_kg
export GRAPH_POOL_MIN=2
export GRAPH_POOL_MAX=10
export GRAPH_PGGRAPH_SCHEMA=public
```

#### 🔌 Python Connection Example
```python
from agent_utilities.knowledge_graph.backends import create_backend

backend = create_backend(
    backend_type="postgresql",
    uri="postgresql://agent:agent@localhost:5433/agent_kg",
    db_name="agent_graph"
)
```

---

## 🔄 Verification & Operations

### Health Checks
Before starting the agent workflow or running pipelines, verify that the databases are online and accessible. You can do this programmatically or via CLI.

#### Dynamic Connectivity Verifier (`verify_connection.py`)
```python
import socket
import time

def wait_for_db(host: str, port: int, timeout: int = 30):
    start_time = time.time()
    while True:
        try:
            with socket.create_connection((host, port), timeout=1):
                print(f"✅ Connection successful to {host}:{port}!")
                return True
        except (socket.timeout, ConnectionRefusedError):
            if time.time() - start_time > timeout:
                raise TimeoutError(f"Database at {host}:{port} failed to start in {timeout}s.")
            print(f"Waiting for database at {host}:{port}...")
            time.sleep(1)

# Example: Check pg-age port
wait_for_db("localhost", 5433)
```

---

## 🧪 Sequential Integration Testing Harness

To prevent resource starvation or port collisions on localized development environments, the integration test suite runs sequentially. The harness manages starting the container, waiting for it to accept connections, executing deep CRUD/vector operations, stress-testing with the ingestion pipeline, and tearing down the container cleanly.

```bash
# Run the sequential multi-backend integration tests
PYTHONPATH=. pytest tests/integration/knowledge_graph/test_multibackend_integration.py -v -s
```

### Flow Diagram of the Testing Lifecycle

```mermaid
graph TD
    A[Start integration run] --> B[Initialize Docker Container]
    B --> C[Check Connectivity Socket]
    C -->|Waiting...| C
    C -->|Connected| D[Init DB Backend Schema]
    D --> E["CRUD & Node Writes"]
    E --> F["Vector Search & HNSW Indexing"]
    F --> G[Run Ingestion Pipeline]
    G --> H[Stop and Prune Container]
    H --> I[Execute Next Database Backend]
```

---

## ⚠️ Troubleshooting & Best Practices

### 1. Vector Dimension Mismatch
Ensure your embedding chunk sizes match what the database indexes were configured with. The standard configuration uses `768` dimensions (for nomic-embed).
- **Symptom**: `ERROR: vector dimension mismatch` on Postgres or `Vector index dimension size mismatch` on Neo4j.
- **Fix**: Verify `embedding_models` in `config.json` is set to the same model provider and chunk size dimensions.

### 2. Neo4j Authentication Failures
If you get `Neo.ClientError.Security.Unauthorized`, double check `NEO4J_AUTH` in docker-compose matches `GRAPH_DB_PASSWORD`.
- **Note**: Neo4j requires the password to be at least 8 characters. Do not use extremely simple passwords like `123` as Neo4j will reject them during container boot.

### 3. Transaction Deadlocks on Scale-Out
Under heavy multi-agent concurrency, database locks might trigger transaction retries.
- **SQLite (LadybugDB)**: File locking can occur. Use a unique `GRAPH_DB_PATH` per agent or switch to **Neo4j** or **PostgreSQL** which natively support row-level locks and high-concurrency connections.
- **Postgres Connection Pool**: Set `GRAPH_POOL_MAX` appropriately based on the number of active agents. Each agent should get a dedicated connection from the pool.
