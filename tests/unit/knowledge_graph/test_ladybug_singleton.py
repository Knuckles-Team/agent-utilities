#!/usr/bin/python
"""Test the process-wide ladybug database and lock singletons.

CONCEPT:KG-2.0
"""

import threading
from unittest.mock import MagicMock, patch

import pytest

from agent_utilities.knowledge_graph.backends.ladybug_backend import (
    _ACTIVE_DATABASES,
    _ACTIVE_LOCKS,
    LadybugBackend,
)


@pytest.fixture(autouse=True)
def clear_registries():
    """Clear process-wide registries before and after each test."""
    _ACTIVE_DATABASES.clear()
    _ACTIVE_LOCKS.clear()
    yield
    _ACTIVE_DATABASES.clear()
    _ACTIVE_LOCKS.clear()


def test_ladybug_backend_singleton_sharing():
    """Verify that multiple LadybugBackend instances for the same path share the same ladybug.Database."""
    db_path = "test_shared.db"

    mock_conn_instance = MagicMock()

    with (
        patch(
            "ladybug.Database", side_effect=lambda *args, **kwargs: MagicMock()
        ) as mock_db_class,
        patch("ladybug.Connection", return_value=mock_conn_instance),
    ):
        # Instantiate first backend
        backend1 = LadybugBackend(db_path)
        backend1._ensure_connection()

        # Instantiate second backend pointing to the exact same path
        backend2 = LadybugBackend(db_path)
        backend2._ensure_connection()

        # Instantiate third backend pointing to a different path
        backend_diff = LadybugBackend("different.db")
        backend_diff._ensure_connection()

        # Assertions
        assert backend1.db is backend2.db
        assert backend_diff.db is not backend1.db

        # The class should be instantiated exactly twice (once for test_shared.db, once for different.db)
        assert mock_db_class.call_count == 2


def test_ladybug_backend_thread_lock_sharing():
    """Verify that multiple LadybugBackend instances for the exact same path share the same threading.Lock."""
    db_path = "test_shared_lock.db"

    with patch("ladybug.Database"), patch("ladybug.Connection"):
        backend1 = LadybugBackend(db_path)
        backend2 = LadybugBackend(db_path)
        backend_diff = LadybugBackend("different_lock.db")

        assert backend1._thread_lock is backend2._thread_lock
        assert backend_diff._thread_lock is not backend1._thread_lock


def test_ladybug_backend_concurrent_initialization():
    """Verify concurrent instantiation and connection from multiple threads."""
    db_path = "concurrent.db"
    backends = []
    errors = []

    mock_db_instance = MagicMock()
    mock_conn_instance = MagicMock()

    with (
        patch("ladybug.Database", return_value=mock_db_instance) as mock_db_class,
        patch("ladybug.Connection", return_value=mock_conn_instance),
    ):

        def worker():
            try:
                backend = LadybugBackend(db_path)
                backend._ensure_connection()
                backends.append(backend)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(backends) == 10

        # All of them must share the same Database object
        first_db = backends[0].db
        for b in backends:
            assert b.db is first_db

        # The ladybug.Database class constructor must have been called exactly once
        assert mock_db_class.call_count == 1


def test_ladybug_backend_self_healing_recreates_database():
    """Verify that _recover_connection pops the database from registry to allow recreation."""
    db_path = "recoverable.db"

    mock_db_instance_1 = MagicMock()
    mock_db_instance_2 = MagicMock()
    mock_conn_instance = MagicMock()

    with (
        patch(
            "ladybug.Database", side_effect=[mock_db_instance_1, mock_db_instance_2]
        ) as mock_db_class,
        patch("ladybug.Connection", return_value=mock_conn_instance),
    ):
        backend = LadybugBackend(db_path)
        backend._ensure_connection()
        assert backend.db is mock_db_instance_1

        # Self-healing recovery should pop and recreate database
        backend._recover_connection()
        assert backend.db is mock_db_instance_2
        assert mock_db_class.call_count == 2
