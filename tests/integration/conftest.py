"""Shared ephemeral-backend fixtures for cross-backend parity + profile tests.

CONCEPT:KG-2.0 / KG-2.7 — Vendor-agnostic Graph Backend parity.

These fixtures stand up a **throwaway** instance of every supported external
service so both the backend conformance matrix (``backends/``) and the enterprise
deployment-profile e2e (``profiles/``) can run against real instances. They live
here (one level above both packages) so both subtrees can request them.

Why testcontainers (not raw ``docker compose`` on fixed ports): the homelab runs
long-lived Neo4j/pg-age/Kafka on the canonical ports (7687/5433/9092). A suite
that bound those would collide with — or mutate — live services. testcontainers
allocates a **random free host port** per container and tears it down
deterministically, so a run is hermetic and intentionally disposable.

Import discipline: ``testcontainers`` and the per-backend drivers are imported
**lazily inside each fixture**, never at module top — so collection still works on
a minimal install and each container fixture ``skip``s cleanly when Docker or its
optional dep is absent.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

# Repo root: tests/integration/conftest.py -> parents[2].
_REPO_ROOT = Path(__file__).resolve().parents[2]
_PARADEDB_INIT_DIR = _REPO_ROOT / "docker" / "paradedb-init"


def _skip_without_docker() -> None:
    """Skip the calling live fixture unless testcontainers + a Docker daemon exist."""
    try:
        import testcontainers  # noqa: F401
    except ImportError:
        pytest.skip(
            "testcontainers not installed — `pip install agent-utilities[test-backends]`"
        )
    import shutil
    import subprocess

    if shutil.which("docker") is None:
        pytest.skip("docker CLI not found")
    probe = subprocess.run(  # noqa: S603,S607 — fixed argv, test-only readiness probe
        ["docker", "info"], capture_output=True, text=True
    )
    if probe.returncode != 0:
        pytest.skip("Docker daemon not reachable")

    # Give slow-booting containers headroom: the testcontainers default readiness
    # wait is 120s (max_tries=120 × sleep_time=1s), but a heavyweight image on a
    # cold/loaded CI runner — notably neo4j:latest (a JVM server that can take
    # >120s to become bolt-ready) and a first-pull Fuseki/ParadeDB — exceeds that
    # and the fixture errors at setup. Raise the ceiling so a slow start is just
    # slow, not a spurious parity failure.
    from testcontainers.core.config import testcontainers_config as _tc_cfg

    # ``timeout`` is a read-only property derived from ``max_tries`` × ``sleep_time``;
    # raising max_tries lifts the effective readiness wait to ~300s.
    if _tc_cfg.max_tries < 300:
        _tc_cfg.max_tries = 300


# ───────────────────────────── container fixtures ──────────────────────────────
# All session-scoped: containers are expensive to boot, and the conformance body
# recreates schema per test, so cross-test isolation needs no fresh container.


@pytest.fixture(scope="session")
def ephemeral_pg_age() -> Iterator[dict[str, Any]]:
    """A throwaway ParadeDB (Postgres + pgvector + pg_search) on a random port."""
    _skip_without_docker()
    from testcontainers.postgres import PostgresContainer

    container = PostgresContainer(
        "paradedb/paradedb:latest",
        username="agent",
        password="agent",  # noqa: S106 — ephemeral throwaway container
        dbname="agent_kg",
    )
    # Mirror docker/paradedb.compose.yml: run the extension bootstrap on first init.
    if _PARADEDB_INIT_DIR.is_dir():
        container.with_volume_mapping(
            str(_PARADEDB_INIT_DIR), "/docker-entrypoint-initdb.d", mode="ro"
        )
    container.start()
    try:
        host = container.get_container_host_ip()
        port = container.get_exposed_port(5432)
        yield {
            "uri": f"postgresql://agent:agent@{host}:{port}/agent_kg",
            "db_name": "agent_graph",
        }
    finally:
        container.stop()


@pytest.fixture(scope="session")
def ephemeral_neo4j() -> Iterator[dict[str, Any]]:
    """A throwaway Neo4j (with APOC) on a random bolt port."""
    _skip_without_docker()
    from testcontainers.neo4j import Neo4jContainer

    container = Neo4jContainer("neo4j:latest", password="password")  # noqa: S106
    container.start()
    try:
        yield {
            "uri": container.get_connection_url(),  # bolt://host:port
            "user": "neo4j",
            "password": "password",
        }
    finally:
        container.stop()


@pytest.fixture(scope="session")
def ephemeral_falkordb() -> Iterator[dict[str, Any]]:
    """A throwaway FalkorDB (Redis-protocol graph) on a random port."""
    _skip_without_docker()
    from testcontainers.core.container import DockerContainer
    from testcontainers.core.waiting_utils import wait_for_logs

    container = DockerContainer("falkordb/falkordb:latest").with_exposed_ports(6379)
    container.start()
    try:
        wait_for_logs(container, "Ready to accept connections", timeout=60)
        yield {
            "host": container.get_container_host_ip(),
            "port": int(container.get_exposed_port(6379)),
            "db_name": "agent_graph",
        }
    finally:
        container.stop()


@pytest.fixture(scope="session")
def ephemeral_fuseki() -> Iterator[dict[str, Any]]:
    """A throwaway Apache Jena Fuseki SPARQL server (in-memory dataset)."""
    _skip_without_docker()
    from testcontainers.core.container import DockerContainer
    from testcontainers.core.waiting_utils import wait_for_logs

    dataset = "agent_kg"
    container = (
        DockerContainer("stain/jena-fuseki:latest")
        .with_exposed_ports(3030)
        .with_env("ADMIN_PASSWORD", "admin")
        # The stain entrypoint forwards args to fuseki-server: create an in-memory
        # dataset so the JenaFusekiBackend has a target without manual provisioning.
        .with_command(f"--mem /{dataset}")
    )
    container.start()
    try:
        # Jena-Fuseki's first-run image pull + JVM boot can exceed 90s on a loaded
        # or cold CI runner; give it headroom so the SPARQL parity tests don't error
        # at setup on a slow start.
        wait_for_logs(container, "Started", timeout=180)
        host = container.get_container_host_ip()
        port = container.get_exposed_port(3030)
        yield {"url": f"http://{host}:{port}", "dataset": dataset}
    finally:
        container.stop()


@pytest.fixture(scope="session")
def ephemeral_kafka() -> Iterator[dict[str, Any]]:
    """A throwaway single-broker Kafka (KRaft) with a random bootstrap port."""
    _skip_without_docker()
    from testcontainers.kafka import KafkaContainer

    container = KafkaContainer()
    container.start()
    try:
        yield {"bootstrap_servers": container.get_bootstrap_server()}
    finally:
        container.stop()


@pytest.fixture()
def ephemeral_ladybug(tmp_path: Path) -> Iterator[dict[str, Any]]:
    """An embedded LadybugDB (Kuzu) backed by a throwaway file under ``tmp_path``.

    LadybugDB is single-writer/embedded — it needs no container, just an isolated
    path. ``AGENT_UTILITIES_TESTING=true`` (set in the root conftest) puts it in
    transient-connection mode so it never wedges the file lock between tests.
    """
    yield {"db_path": str(tmp_path / "parity_knowledge_graph.db")}
