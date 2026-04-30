#!/usr/bin/python
"""Pluggable Secrets Manager.

CONCEPT:AU-011 — Secrets & Authentication

Provides encrypted secrets storage with three pluggable backends:

- **InMemoryBackend** (default): Fernet-encrypted dict, zero-config, lost on restart.
- **SQLiteBackend**: Persistent encrypted storage using standard sqlite3 + Fernet.
- **VaultBackend**: HashiCorp Vault integration via ``hvac`` (optional dependency).

Usage::

    from agent_utilities.security.secrets_client import create_secrets_client

    client = create_secrets_client()
    client.set("gitlab/token", "glpat-xxx")
    token = client.get_or_env("gitlab/token", "GITLAB_TOKEN")

URI reference resolution::

    client.resolve_ref("vault://agents/mcp/gitlab/token")
    client.resolve_ref("env://GITLAB_TOKEN")
    client.resolve_ref("sqlite://gitlab/token")
"""

from __future__ import annotations

import abc
import json
import logging
import os
import sqlite3
from pathlib import Path
from typing import Any

from cryptography.fernet import Fernet, InvalidToken
from pydantic import BaseModel, Field, SecretStr

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class SecretValue(BaseModel):
    """Typed wrapper for a secret value with optional metadata.

    CONCEPT:AU-011 — Secrets & Authentication
    """

    value: SecretStr
    metadata: dict[str, Any] = Field(default_factory=dict)


class SecretsConfig(BaseModel):
    """Configuration for the secrets client factory.

    CONCEPT:AU-011 — Secrets & Authentication
    """

    backend: str = Field(
        default="inmemory",
        description="Backend type: 'inmemory', 'sqlite', or 'vault'.",
    )
    sqlite_path: str | None = Field(
        default=None,
        description="Path to SQLite secrets database (used with 'sqlite' backend).",
    )
    vault_url: str | None = Field(
        default=None,
        description="HashiCorp Vault URL (used with 'vault' backend).",
    )
    vault_mount: str = Field(
        default="secret",
        description="Vault KV v2 mount point.",
    )
    master_key: str | None = Field(
        default=None,
        description="Master encryption key (base64-encoded Fernet key). Auto-generated if omitted.",
    )


# ---------------------------------------------------------------------------
# Abstract Backend
# ---------------------------------------------------------------------------


class SecretsBackend(abc.ABC):
    """Abstract base class for secrets storage backends.

    CONCEPT:AU-011 — Secrets & Authentication
    """

    @abc.abstractmethod
    def get(self, key: str) -> str | None:
        """Retrieve a secret by key. Returns ``None`` if not found."""

    @abc.abstractmethod
    def set(self, key: str, value: str, **metadata: Any) -> None:
        """Store a secret with optional metadata."""

    @abc.abstractmethod
    def delete(self, key: str) -> bool:
        """Delete a secret. Returns ``True`` if it existed."""

    @abc.abstractmethod
    def list_keys(self) -> list[str]:
        """List all stored secret keys."""


# ---------------------------------------------------------------------------
# InMemory Backend (default)
# ---------------------------------------------------------------------------


class InMemoryBackend(SecretsBackend):
    """Fernet-encrypted in-memory backend.

    Secrets are encrypted at rest in a Python dict and lost on process exit.
    Suitable for development, testing, and short-lived agent sessions.

    CONCEPT:AU-011 — Secrets & Authentication
    """

    def __init__(self, master_key: bytes | None = None) -> None:
        if master_key is None:
            master_key = Fernet.generate_key()
        self._fernet = Fernet(master_key)
        self._store: dict[str, bytes] = {}

    def get(self, key: str) -> str | None:
        cipher = self._store.get(key)
        if cipher is None:
            return None
        try:
            return self._fernet.decrypt(cipher).decode()
        except InvalidToken:
            logger.warning("Failed to decrypt secret '%s' — corrupt or wrong key.", key)
            return None

    def set(self, key: str, value: str, **metadata: Any) -> None:
        self._store[key] = self._fernet.encrypt(value.encode())
        logger.info("Secret '%s' stored (in-memory).", key)

    def delete(self, key: str) -> bool:
        existed = key in self._store
        self._store.pop(key, None)
        if existed:
            logger.info("Secret '%s' deleted (in-memory).", key)
        return existed

    def list_keys(self) -> list[str]:
        return sorted(self._store.keys())


# ---------------------------------------------------------------------------
# SQLite Backend (persistent)
# ---------------------------------------------------------------------------


class SQLiteBackend(SecretsBackend):
    """Persistent SQLite backend with Fernet field-level encryption.

    Secret *values* are encrypted; key names and metadata are stored in
    plaintext for queryability.  The DB file itself should live in a
    user-private directory (e.g. ``~/.agent-utilities/secrets.db``).

    CONCEPT:AU-011 — Secrets & Authentication
    """

    def __init__(
        self,
        db_path: str | Path = "~/.agent-utilities/secrets.db",
        master_key: bytes | None = None,
    ) -> None:
        self._db_path = Path(db_path).expanduser()
        self._db_path.parent.mkdir(parents=True, exist_ok=True)

        if master_key is None:
            # Derive from env or generate and persist alongside the DB.
            key_file = self._db_path.with_suffix(".key")
            if key_file.exists():
                master_key = key_file.read_bytes().strip()
            else:
                master_key = Fernet.generate_key()
                key_file.write_bytes(master_key)
                # Best-effort restrictive permissions.
                try:
                    key_file.chmod(0o600)
                except OSError:
                    pass

        self._fernet = Fernet(master_key)
        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS secrets ("
            "  key TEXT PRIMARY KEY,"
            "  value BLOB NOT NULL,"
            "  metadata TEXT DEFAULT '{}'"
            ")"
        )
        self._conn.commit()
        logger.info("SQLite secrets backend initialised at %s", self._db_path)

    def get(self, key: str) -> str | None:
        row = self._conn.execute(
            "SELECT value FROM secrets WHERE key = ?", (key,)
        ).fetchone()
        if row is None:
            return None
        try:
            return self._fernet.decrypt(row[0]).decode()
        except InvalidToken:
            logger.warning(
                "Failed to decrypt secret '%s' from SQLite — corrupt or wrong key.", key
            )
            return None

    def set(self, key: str, value: str, **metadata: Any) -> None:
        encrypted = self._fernet.encrypt(value.encode())
        meta_json = json.dumps(metadata) if metadata else "{}"
        self._conn.execute(
            "INSERT OR REPLACE INTO secrets (key, value, metadata) VALUES (?, ?, ?)",
            (key, encrypted, meta_json),
        )
        self._conn.commit()
        logger.info("Secret '%s' stored (SQLite).", key)

    def delete(self, key: str) -> bool:
        cursor = self._conn.execute("DELETE FROM secrets WHERE key = ?", (key,))
        self._conn.commit()
        deleted = cursor.rowcount > 0
        if deleted:
            logger.info("Secret '%s' deleted (SQLite).", key)
        return deleted

    def list_keys(self) -> list[str]:
        rows = self._conn.execute("SELECT key FROM secrets ORDER BY key").fetchall()
        return [r[0] for r in rows]


# ---------------------------------------------------------------------------
# Vault Backend (enterprise, optional)
# ---------------------------------------------------------------------------


class VaultBackend(SecretsBackend):
    """HashiCorp Vault KV v2 backend.

    Requires the ``hvac`` package (``pip install agent-utilities[vault]``).

    CONCEPT:AU-011 — Secrets & Authentication
    """

    def __init__(
        self,
        url: str = "http://127.0.0.1:8200",
        token: str | None = None,
        mount_point: str = "secret",
    ) -> None:
        try:
            import hvac  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "The 'hvac' package is required for the Vault backend. "
                "Install it with: pip install agent-utilities[vault]"
            ) from exc

        self._client = hvac.Client(url=url, token=token or os.getenv("VAULT_TOKEN"))
        self._mount = mount_point
        logger.info("Vault backend initialised at %s (mount: %s)", url, mount_point)

    def get(self, key: str) -> str | None:
        try:
            resp = self._client.secrets.kv.v2.read_secret_version(
                path=key, mount_point=self._mount
            )
            data = resp.get("data", {}).get("data", {})
            return data.get("value")
        except Exception:
            logger.debug("Vault get('%s') failed.", key, exc_info=True)
            return None

    def set(self, key: str, value: str, **metadata: Any) -> None:
        secret_data = {"value": value}
        if metadata:
            secret_data.update(metadata)
        self._client.secrets.kv.v2.create_or_update_secret(
            path=key, secret=secret_data, mount_point=self._mount
        )
        logger.info("Secret '%s' stored (Vault).", key)

    def delete(self, key: str) -> bool:
        try:
            self._client.secrets.kv.v2.delete_metadata_and_all_versions(
                path=key, mount_point=self._mount
            )
            logger.info("Secret '%s' deleted (Vault).", key)
            return True
        except Exception:
            return False

    def list_keys(self) -> list[str]:
        try:
            resp = self._client.secrets.kv.v2.list_secrets(
                path="", mount_point=self._mount
            )
            return sorted(resp.get("data", {}).get("keys", []))
        except Exception:
            return []


# ---------------------------------------------------------------------------
# High-level Client
# ---------------------------------------------------------------------------


class SecretsClient:
    """High-level secrets client with URI resolution and env-var fallback.

    CONCEPT:AU-011 — Secrets & Authentication

    Wraps any ``SecretsBackend`` and adds:

    - ``get_or_env(key, env_var)`` — falls back to ``os.environ`` if the
      key is not in the backend.
    - ``resolve_ref(uri)`` — resolves ``vault://``, ``env://``, and plain
      key references.
    - Typed ``get_secret()`` returning a ``SecretValue`` Pydantic model.
    """

    def __init__(self, backend: SecretsBackend | None = None) -> None:
        self._backend = backend or InMemoryBackend()

    @property
    def backend(self) -> SecretsBackend:
        """The underlying storage backend."""
        return self._backend

    # -- Core operations ---------------------------------------------------

    def get(self, key: str) -> str | None:
        """Retrieve a secret by key."""
        return self._backend.get(key)

    def set(self, key: str, value: str, **metadata: Any) -> None:
        """Store a secret."""
        self._backend.set(key, value, **metadata)

    def delete(self, key: str) -> bool:
        """Delete a secret."""
        return self._backend.delete(key)

    def list_keys(self) -> list[str]:
        """List all stored keys."""
        return self._backend.list_keys()

    # -- Extended operations -----------------------------------------------

    def get_or_env(self, key: str, env_var: str | None = None) -> str | None:
        """Get a secret, falling back to an environment variable.

        Args:
            key: Secret key in the backend.
            env_var: Environment variable name to check if the key is missing.

        Returns:
            The secret value, or the env var value, or ``None``.
        """
        val = self._backend.get(key)
        if val is not None:
            return val
        if env_var:
            return os.environ.get(env_var)
        return None

    def get_secret(self, key: str) -> SecretValue | None:
        """Retrieve a secret as a typed ``SecretValue``."""
        val = self._backend.get(key)
        if val is None:
            return None
        return SecretValue(value=SecretStr(val))

    def resolve_ref(self, ref: str) -> str | None:
        """Resolve a URI-style secret reference.

        Supported schemes:

        - ``vault://path/to/secret`` → backend lookup
        - ``env://VAR_NAME`` → ``os.environ.get(VAR_NAME)``
        - ``sqlite://key`` → backend lookup
        - Plain string → backend lookup

        Args:
            ref: Secret reference string.

        Returns:
            The resolved secret value, or ``None``.
        """
        if ref.startswith("env://"):
            var_name = ref[len("env://") :]
            return os.environ.get(var_name)
        if ref.startswith("vault://") or ref.startswith("secret://"):
            key = ref.split("://", 1)[1]
            return self._backend.get(key)
        if ref.startswith("sqlite://"):
            key = ref[len("sqlite://") :]
            return self._backend.get(key)
        # Plain key
        return self._backend.get(ref)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_secrets_client(config: SecretsConfig | None = None) -> SecretsClient:
    """Create a ``SecretsClient`` from configuration.

    CONCEPT:AU-011 — Secrets & Authentication

    The backend is selected by ``config.backend``:

    - ``"inmemory"`` (default): Encrypted in-memory dict.
    - ``"sqlite"``: Persistent encrypted SQLite.
    - ``"vault"``: HashiCorp Vault KV v2.

    Args:
        config: Secrets configuration. If ``None``, reads from environment
            variables (``SECRETS_BACKEND``, ``SECRETS_SQLITE_PATH``,
            ``SECRETS_VAULT_URL``, ``SECRETS_VAULT_MOUNT``).

    Returns:
        A configured ``SecretsClient`` instance.
    """
    if config is None:
        config = SecretsConfig(
            backend=os.getenv("SECRETS_BACKEND", "inmemory"),
            sqlite_path=os.getenv("SECRETS_SQLITE_PATH"),
            vault_url=os.getenv("SECRETS_VAULT_URL"),
            vault_mount=os.getenv("SECRETS_VAULT_MOUNT", "secret"),
            master_key=os.getenv("AGENT_SECRETS_MASTER_KEY"),
        )

    master_key_bytes: bytes | None = None
    if config.master_key:
        master_key_bytes = config.master_key.encode()

    if config.backend == "sqlite":
        path = config.sqlite_path or "~/.agent-utilities/secrets.db"
        backend: SecretsBackend = SQLiteBackend(
            db_path=path, master_key=master_key_bytes
        )
    elif config.backend == "vault":
        url = config.vault_url or "http://127.0.0.1:8200"
        backend = VaultBackend(url=url, mount_point=config.vault_mount)
    else:
        backend = InMemoryBackend(master_key=master_key_bytes)

    logger.info("SecretsClient initialised with '%s' backend.", config.backend)
    return SecretsClient(backend=backend)
