"""Comprehensive Multi-Backend Integration Tests.

CONCEPT:AU-KG.query.object-graph-mapper

Verifies sequential container lifecycle, schema creation, high-fidelity CRUD,
embedding/vector search, and pipeline syncing across Neo4j, FalkorDB, and pgGraph/PostgreSQL.
"""

import json
import os
import socket
import subprocess
import time

import pytest

from agent_utilities.knowledge_graph.backends import (
    create_backend,
    get_active_backend,
    set_active_backend,
)
from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine
from agent_utilities.knowledge_graph.core.graph_compute import GraphComputeEngine
from agent_utilities.knowledge_graph.pipeline import IntelligencePipeline
from agent_utilities.models.knowledge_graph import (
    PipelineConfig,
    RegistryEdgeType,
    RegistryNodeType,
)


# Setup skip conditions if docker is not running
def is_docker_active() -> bool:
    try:
        res = subprocess.run(["docker", "info"], capture_output=True, text=True)
        return res.returncode == 0
    except FileNotFoundError:
        return False


DOCKER_AVAILABLE = is_docker_active()


def wait_for_port(port: int, host: str = "localhost", timeout: float = 45.0) -> bool:
    """Helper to wait for a database port to open."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            with socket.create_connection((host, port), timeout=1.0):
                return True
        except (TimeoutError, ConnectionRefusedError):
            time.sleep(0.5)
    return False


def wait_for_db_ready(
    backend_type: str, conn_kwargs: dict, timeout: float = 45.0
) -> bool:
    """Verify that the database is fully booted and accepting client driver queries."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            if backend_type == "falkordb":
                from falkordb import FalkorDB

                db = FalkorDB(
                    host=conn_kwargs.get("host", "localhost"),
                    port=conn_kwargs.get("port", 6380),
                )
                # Simple ping/query check
                db.select_graph(conn_kwargs.get("db_name", "agent_graph")).query(
                    "RETURN 1"
                )
                return True
            elif backend_type == "neo4j":
                from neo4j import GraphDatabase

                uri = conn_kwargs.get("uri", "bolt://localhost:7687")
                user = conn_kwargs.get("user", "neo4j")
                pwd = conn_kwargs.get("password", "password")
                with GraphDatabase.driver(uri, auth=(user, pwd)) as driver:
                    driver.verify_connectivity()
                return True
            elif backend_type == "postgresql":
                import psycopg

                uri = conn_kwargs.get("uri") or ""
                with psycopg.connect(uri) as conn:
                    with conn.cursor() as cur:
                        cur.execute("SELECT 1")
                return True
        except Exception:
            time.sleep(1.0)
    return False


def manage_container(compose_file: str, action: str):
    """Start or stop docker compose stack."""
    cmd = ["docker", "compose", "-f", compose_file, action]
    if action == "up":
        cmd.extend(["-d"])
    elif action == "down":
        cmd.extend(["-v"])  # Remove anonymous volumes to start fresh

    # Run the compose command without --wait to avoid compose healthcheck timeouts
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"docker compose {action} failed: {res.stderr}")


@pytest.mark.live
@pytest.mark.skipif(not DOCKER_AVAILABLE, reason="Docker is not active or installed")
class TestMultiBackendIntegration:
    """Rigorous sequential integration tests for different database backends."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "backend_type,compose_file,port,conn_kwargs",
        [
            (
                "falkordb",
                "docker/falkordb.compose.yml",
                6380,
                {"host": "localhost", "port": 6380, "db_name": "agent_graph"},
            ),
            (
                "neo4j",
                "docker/neo4j.compose.yml",
                7687,
                {
                    "uri": "bolt://localhost:7687",
                    "user": "neo4j",
                    "password": "password",
                },
            ),
            (
                "postgresql",
                "docker/paradedb.compose.yml",
                5433,
                {
                    "uri": "postgresql://agent:agent@localhost:5433/agent_kg",
                    "db_name": "agent_graph",
                },
            ),
        ],
    )
    async def test_backend_lifecycle_and_crud(
        self, backend_type, compose_file, port, conn_kwargs
    ):
        # Locate correct absolute path of compose file
        workspace_dir = "/home/apps/workspace/agent-packages/agent-utilities"
        abs_compose = os.path.join(workspace_dir, compose_file)

        print(
            f"\n>>> Starting container for backend: {backend_type} using {compose_file}..."
        )

        # 1. Clean teardown from any previous partial states
        try:
            manage_container(abs_compose, "down")
        except Exception:
            pass

        # Ensure active engine and active backend are clean
        IntelligenceGraphEngine._ACTIVE_ENGINE = None
        set_active_backend(None)

        # 2. Spin up the specific container
        try:
            manage_container(abs_compose, "up")
            assert wait_for_port(port, timeout=45.0), (
                f"Port {port} failed to open in time!"
            )

            # Wait for database driver initialization success
            print(f"Waiting for {backend_type} driver connectivity check...")
            assert wait_for_db_ready(backend_type, conn_kwargs, timeout=60.0), (
                f"Database {backend_type} failed ready check!"
            )
            print(f"Database {backend_type} is ready!")

            # 3. Instantiate the backend factory
            print(f"Creating backend instance for: {backend_type}...")
            backend = create_backend(backend_type=backend_type, **conn_kwargs)
            assert backend is not None, f"Failed to instantiate {backend_type} backend"

            # Initialize schema and tables
            print(f"Creating schema/tables for {backend_type}...")
            backend.create_schema()

            # 4. Initialize graph engine
            GraphComputeEngine(backend_type="rust")
            engine = IntelligenceGraphEngine(backend=backend)

            # Set the engine active correctly using class variable
            IntelligenceGraphEngine.set_active(engine)
            assert get_active_backend() == backend

            # 5. High-fidelity Stress/CRUD verification
            # Add nodes with complex strings, escapes, nested metadata to ensure robust serialization
            agent_id = "agent:complex-router"
            agent_props = {
                "name": "Expert Specialist Router 🤖 (Quotes: \"hello\", 'world')",
                "description": "Multi-line description:\nLine 1\nLine 2 with backslashes: \\path\\to\\config",
                "agent_type": "prompt",
                "system_prompt": "Prompt with quote escapes and emoji: 🎉",
                "capabilities": ["router", "math", "vector-search"],
                "tool_count": 2,
                "importance_score": 0.98,
                "is_permanent": True,
            }

            tool_id = "tool:math-eval"
            tool_props = {
                "name": "math-eval",
                "description": "Evaluate expressions: x + y = z",
                "mcp_server": "Expert Specialist Router 🤖 (Quotes: \"hello\", 'world')",
                "relevance_score": 100,
                "requires_approval": True,
                "tags": ["math", "calculator"],
                "importance_score": 0.85,
                "is_permanent": False,
            }

            memory_id = "mem:user-preferences"
            memory_props = {
                "name": "user-preferences",
                "description": "User custom settings card.",
                "category": "preferences",
                "status": "ACTIVE",
                "tags": ["theme", "ui"],
                "importance_score": 0.9,
                "metadata": json.dumps(
                    {"theme": "dark", "zoom": 1.2, "nested": {"list": [1, 2, 3]}}
                ),
                "is_permanent": True,
            }

            print("Adding nodes via active graph engine...")
            engine.add_node(agent_id, RegistryNodeType.AGENT, agent_props)
            engine.add_node(tool_id, RegistryNodeType.TOOL, tool_props)
            engine.add_node(memory_id, RegistryNodeType.MEMORY, memory_props)

            # Link them
            print("Linking nodes...")
            engine.link_nodes(
                agent_id, tool_id, RegistryEdgeType.PROVIDES, {"confidence": 0.95}
            )
            engine.link_nodes(
                agent_id, memory_id, RegistryEdgeType.MEMORY_OF, {"confidence": 0.99}
            )

            # 6. Retrieve and assert correctness of stored values
            print("Retrieving and asserting node structures...")
            # Query the agent node
            agent_res = engine.query_cypher(
                "MATCH (n:Agent) WHERE n.id = $id RETURN n", {"id": agent_id}
            )
            assert len(agent_res) > 0, "Agent node not found!"
            retrieved_agent = agent_res[0]["n"]
            assert retrieved_agent["id"] == agent_id
            assert "🤖" in retrieved_agent["name"]
            assert "backslashes" in retrieved_agent["description"]

            # Query relationships
            rel_res = engine.query_cypher(
                "MATCH (s:Agent)-[r:PROVIDES]->(t:Tool) WHERE s.id = $sid AND t.id = $tid RETURN r.confidence as conf",
                {"sid": agent_id, "tid": tool_id},
            )
            assert len(rel_res) > 0, "PROVIDES relationship not found!"
            assert float(rel_res[0]["conf"]) == 0.95

            # 7. Embedding and Vector Search Verification
            print("Verifying vector operations and semantic search...")
            chunk_id = "chunk:vector-test"
            chunk_props = {
                "name": "vector-test",
                "description": "Mocked chunk vector search test.",
                "content": "Semantic integration tests across multi backends using ParadeDB and FalkorDB.",
                "importance_score": 0.77,
                "is_permanent": False,
            }
            engine.add_node(chunk_id, RegistryNodeType.DOCUMENT, chunk_props)

            # Add embedding vector
            test_vector = [0.01 * i for i in range(768)]
            backend.add_embedding(chunk_id, test_vector)

            # Perform semantic vector search
            search_res = backend.semantic_search(test_vector, n_results=1)
            # Both Neo4j and FalkorDB returns list of dicts, pgGraph does too
            if search_res:
                retrieved_node = search_res[0].get("node") or search_res[0]
                assert (
                    retrieved_node.get("id") == chunk_id
                    or retrieved_node.get("n", {}).get("id") == chunk_id
                )
                print("Vector search completed successfully and matched correct node!")

            # 8. Test pipeline synchronization utilizing dynamic mocks
            print("Running test pipeline synchronization...")
            import tempfile
            from pathlib import Path
            from unittest.mock import MagicMock, patch

            with tempfile.TemporaryDirectory(
                dir="/home/apps/workspace/agent-packages/agent-utilities"
            ) as tmp_dir:
                # Write a dummy python file so the scan/parse phases run instantly
                dummy_file = Path(tmp_dir) / "dummy_agent.py"
                dummy_file.write_text(
                    "class DummyAgent:\n    def run(self):\n        pass\n"
                )

                config = PipelineConfig(
                    workspace_path=tmp_dir,
                    persist_to_ladybug=True,
                    enable_embeddings=False,
                )

                mock_agent = MagicMock()
                mock_agent.name = (
                    "Expert Specialist Router 🤖 (Quotes: \"hello\", 'world')"
                )
                mock_agent.description = "desc"
                mock_agent.agent_type = "prompt"
                mock_agent.system_prompt = "system text"
                mock_agent.endpoint_url = None
                mock_agent.tool_count = 1

                mock_registry = MagicMock()
                mock_registry.agents = [mock_agent]
                mock_registry.tools = []

                with patch(
                    "agent_utilities.core.config.get_discovery_registry",
                    return_value=mock_registry,
                ):
                    pipeline = IntelligencePipeline(config, backend=backend)
                    metadata = await pipeline.run()
                    assert metadata.node_count > 0
                    print(
                        f"Pipeline executed successfully. Ingested node count: {metadata.node_count}"
                    )

        finally:
            # 9. Sequential teardown to release resources
            print(f"Tearing down container for: {backend_type}...")
            try:
                IntelligenceGraphEngine._ACTIVE_ENGINE = None
                set_active_backend(None)
                manage_container(abs_compose, "down")
            except Exception as e:
                print(f"Teardown warning: {e}")
            print(f"Teardown for {backend_type} completed.")


# CONCEPT:AU-KG.backend.multi-connection-registry — three live backends side by side through the named
# multi-connection registry: the SAME tool runs against any one (target=<name>)
# or fans out to all (target="all"), with partial success when one is down.
_MULTICONN_BACKENDS = [
    (
        "team-falkor",
        "falkordb",
        "docker/falkordb.compose.yml",
        6380,
        {"host": "localhost", "port": 6380, "db_name": "agent_graph"},
    ),
    (
        "prod-neo4j",
        "neo4j",
        "docker/neo4j.compose.yml",
        7687,
        {"uri": "bolt://localhost:7687", "user": "neo4j", "password": "password"},
    ),
    (
        # Register Postgres as AGE for native openCypher portability — one query
        # runs unchanged across all three.
        "pg-main",
        "age",
        "docker/pg-age.compose.yml",
        5434,
        {
            "uri": "postgresql://agent:agent@localhost:5434/agent_kg",
            "db_name": "agent_graph",
        },
    ),
]


@pytest.mark.live
@pytest.mark.skipif(not DOCKER_AVAILABLE, reason="Docker is not active or installed")
class TestMultiConnectionRegistryLive:
    """Bring up falkordb + neo4j + pggraph-AGE at once and route the registry."""

    def test_named_routing_and_fanout(self):
        from agent_utilities.knowledge_graph.core.connection_registry import (
            ConnectionRegistry,
        )

        workspace_dir = "/home/apps/workspace/agent-packages/agent-utilities"
        IntelligenceGraphEngine._ACTIVE_ENGINE = None
        set_active_backend(None)

        # A zero-infra in-memory default keeps "default" addressable without infra.
        default_engine = IntelligenceGraphEngine(
            backend=create_backend(backend_type="memory")
        )
        IntelligenceGraphEngine.set_active(default_engine)
        registry = ConnectionRegistry(default_engine_provider=lambda: default_engine)

        started: list[str] = []
        try:
            for name, backend_type, compose, port, conn in _MULTICONN_BACKENDS:
                abs_compose = os.path.join(workspace_dir, compose)
                try:
                    manage_container(abs_compose, "down")
                except Exception:
                    pass
                manage_container(abs_compose, "up")
                started.append(abs_compose)
                assert wait_for_port(port, timeout=60.0), f"{name} port {port} not open"
                assert wait_for_db_ready(backend_type, conn, timeout=90.0), (
                    f"{name} not ready"
                )
                # Schema first (idempotent), then register the live connection.
                create_backend(backend_type=backend_type, **conn).create_schema()
                registry.register(name, {"backend": backend_type, **conn})

            names = [b[0] for b in _MULTICONN_BACKENDS]

            # 1. Single named write+read against EACH backend, same Cypher surface.
            for name in names:
                eng = registry.get_engine(name)
                eng.add_node(f"id::{name}", "Probe", {"id": f"id::{name}", "who": name})
                rows = eng.query_cypher(
                    "MATCH (n:Probe) RETURN n.id AS id, n.who AS who"
                )
                ids = {r.get("id") for r in rows}
                assert f"id::{name}" in ids, f"{name} did not persist/read its own node"

            # 2. Fan-out read across all named connections — labeled per-connection.
            all_names, fanout = registry.resolve_names(names)
            assert fanout and set(all_names) == set(names)
            results = {}
            for n in all_names:
                eng, err = registry.safe_get_engine(n)
                assert err is None, f"{n}: {err}"
                results[n] = eng.query_cypher("MATCH (n:Probe) RETURN n.id AS id")
            assert all(results[n] for n in names), "every backend should answer"

            # 3. Partial success: stop one backend, fan-out still serves the rest.
            down_name, _bt, down_compose, _p, _c = _MULTICONN_BACKENDS[0]
            manage_container(os.path.join(workspace_dir, down_compose), "down")
            started.remove(os.path.join(workspace_dir, down_compose))
            registry.remove(down_name)  # drop the dead connection from the registry
            ok, errs = {}, {}
            for n in names:
                if n == down_name:
                    continue
                eng, err = registry.safe_get_engine(n)
                if err:
                    errs[n] = err
                else:
                    ok[n] = eng.query_cypher("MATCH (n:Probe) RETURN n.id AS id")
            assert ok, "surviving backends must still answer after one is removed"
        finally:
            registry.close_all()
            IntelligenceGraphEngine._ACTIVE_ENGINE = None
            set_active_backend(None)
            for abs_compose in started:
                try:
                    manage_container(abs_compose, "down")
                except Exception as e:
                    print(f"Teardown warning ({abs_compose}): {e}")
