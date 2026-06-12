#!/usr/bin/python
from __future__ import annotations

"""Pluggable Secrets Manager.

CONCEPT:OS-5.1 — Secrets & Authentication

Provides encrypted secrets storage with three pluggable backends:

- **InEpistemicGraphBackend** (default): Fernet-encrypted dict, zero-config, lost on restart.
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


import abc
import json
import logging
import sqlite3
from pathlib import Path
from typing import Any

from cryptography.fernet import Fernet, InvalidToken
from pydantic import BaseModel, Field, SecretStr

from agent_utilities.core.config import setting

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class SecretValue(BaseModel):
    """Typed wrapper for a secret value with optional metadata.

    CONCEPT:OS-5.1 — Secrets & Authentication
    """

    value: SecretStr
    metadata: dict[str, Any] = Field(default_factory=dict)


class SecretsConfig(BaseModel):
    """Configuration for the secrets client factory.

    CONCEPT:OS-5.1 — Secrets & Authentication
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
    vault_auth_method: str = Field(
        default="auto",
        description=(
            "Vault authentication method: 'oidc', 'approle', 'token', "
            "'kubernetes', or 'auto' (auto-detect)."
        ),
    )
    vault_auth_mount: str = Field(
        default="jwt",
        description=(
            "Mount path of the Vault auth method.  Supports custom mounts "
            "(e.g. 'oidc', 'jwt', 'my-okta-auth').  Default: 'jwt'."
        ),
    )
    vault_role: str | None = Field(
        default=None,
        description="Vault role name for OIDC/JWT or Kubernetes login.",
    )
    vault_path_prefix: str | None = Field(
        default=None,
        description=(
            "Path prefix within the KV v2 mount.  E.g. 'agents/mcp/' scopes "
            "all secret reads/writes under 'secret/data/agents/mcp/'."
        ),
    )
    vault_role_id: str | None = Field(
        default=None,
        description="AppRole role_id for Vault authentication.",
    )
    vault_secret_id: str | None = Field(
        default=None,
        description="AppRole secret_id for Vault authentication.",
    )
    vault_k8s_sa_token_path: str = Field(
        default="/var/run/secrets/kubernetes.io/serviceaccount/token",
        description="Path to the Kubernetes service account token file.",
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

    CONCEPT:OS-5.1 — Secrets & Authentication
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


class InEpistemicGraphBackend(SecretsBackend):
    """Fernet-encrypted in-memory backend.

    Secrets are encrypted at rest in a Python dict and lost on process exit.
    Suitable for development, testing, and short-lived agent sessions.

    CONCEPT:OS-5.1 — Secrets & Authentication
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

    CONCEPT:OS-5.1 — Secrets & Authentication
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
    """HashiCorp Vault KV v2 backend with multi-auth support.

    Requires the ``hvac`` package (``pip install agent-utilities[vault]``).

    Supports four authentication strategies (in priority order):

    1. **OIDC/JWT** — Exchanges the SSO user token (from
       ``UserTokenMiddleware``) for a user-scoped Vault token via
       Vault's JWT/OIDC auth method.
    2. **AppRole** — Machine-to-machine auth via ``role_id`` +
       ``secret_id`` (ideal for CI/CD pipelines).
    3. **Static Token** — Classic ``VAULT_TOKEN`` env var (backward
       compatible).
    4. **Kubernetes** — Auto-detects pod-mounted service-account JWT
       (useful for K8s-native deployments).

    The ``auth_mount`` parameter supports custom mount paths, so the
    auth method does not need to be at the default ``/auth/jwt`` —
    any path (e.g. ``/auth/my-okta-oidc``) works.

    Path prefixes scope secret reads/writes within the KV v2 mount::

        VaultBackend(path_prefix="agents/mcp/")
        backend.get("gitlab/token")
        # reads: secret/data/agents/mcp/gitlab/token

    CONCEPT:OS-5.1 — Secrets & Authentication
    """

    def __init__(
        self,
        url: str = "http://127.0.0.1:8200",
        token: str | None = None,
        mount_point: str = "secret",
        auth_method: str = "auto",
        auth_mount: str = "jwt",
        role: str | None = None,
        path_prefix: str | None = None,
        role_id: str | None = None,
        secret_id: str | None = None,
        k8s_sa_token_path: str | None = None,
    ) -> None:
        try:
            import hvac  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "The 'hvac' package is required for the Vault backend. "
                "Install it with: pip install agent-utilities[vault]"
            ) from exc

        self._mount = mount_point
        self._path_prefix = path_prefix.rstrip("/") if path_prefix else None
        self._auth_method = auth_method
        self._auth_mount = auth_mount
        self._role = role or setting("VAULT_ROLE", "default")
        self._role_id = role_id or setting("VAULT_ROLE_ID")
        self._secret_id = secret_id or setting("VAULT_SECRET_ID")
        self._k8s_sa_token_path = (
            k8s_sa_token_path or "/var/run/secrets/kubernetes.io/serviceaccount/token"
        )
        self._token_lease_duration: float = 0.0
        self._token_auth_time: float = 0.0

        # Initialise hvac client — may not have a token yet
        static_token = token or setting("VAULT_TOKEN")
        self._client = hvac.Client(url=url, token=static_token)

        # Authenticate using the configured method
        self._authenticate(static_token)

        logger.info(
            "Vault backend initialised at %s (mount: %s, auth: %s, prefix: %s)",
            url,
            mount_point,
            self._auth_method,
            self._path_prefix or "<root>",
        )

    # -- Authentication strategies -----------------------------------------

    def _authenticate(self, static_token: str | None = None) -> None:
        """Authenticate to Vault using the best available method.

        When ``auth_method='auto'``, tries in order:
        OIDC/JWT → AppRole → static token → Kubernetes.
        """
        import time as _time

        method = self._auth_method

        if method == "auto":
            if self._try_oidc():
                self._auth_method = "oidc"
                return
            if self._try_approle():
                self._auth_method = "approle"
                return
            if static_token and self._client.is_authenticated():
                self._auth_method = "token"
                logger.info("Vault: Authenticated via static token.")
                return
            if self._try_kubernetes():
                self._auth_method = "kubernetes"
                return
            # Fallback — hope the token is valid
            logger.warning(
                "Vault: No auth method succeeded; using unauthenticated client."
            )
            return

        if method == "oidc":
            if not self._try_oidc():
                raise RuntimeError(
                    "Vault OIDC auth failed. Ensure the MCP server has an active "
                    "SSO session and VAULT_ROLE is set."
                )
        elif method == "approle":
            if not self._try_approle():
                raise RuntimeError(
                    "Vault AppRole auth failed. Check VAULT_ROLE_ID and VAULT_SECRET_ID."
                )
        elif method == "kubernetes":
            if not self._try_kubernetes():
                raise RuntimeError(
                    "Vault Kubernetes auth failed. Ensure a service account token "
                    "is mounted and VAULT_ROLE is set."
                )
        elif method == "token":
            if not self._client.is_authenticated():
                raise RuntimeError(
                    "Vault token auth failed. Set VAULT_TOKEN or provide a token."
                )
            logger.info("Vault: Authenticated via static token.")
        else:
            raise ValueError(f"Unsupported vault_auth_method: {method!r}")

        self._token_auth_time = _time.monotonic()

    def _try_oidc(self) -> bool:
        """Authenticate using the SSO user token from ``UserTokenMiddleware``.

        Uses Vault's JWT/OIDC auth method at the configured ``auth_mount``
        path (default: ``jwt``).  Works with any custom mount path.
        """
        try:
            from agent_utilities.mcp.delegated_auth import get_user_token

            user_jwt = get_user_token()
            if not user_jwt:
                return False

            resp = self._client.auth.jwt.jwt_login(
                role=self._role,
                jwt=user_jwt,
                path=self._auth_mount,
            )
            self._client.token = resp["auth"]["client_token"]
            self._token_lease_duration = float(resp["auth"].get("lease_duration", 3600))
            import time as _time

            self._token_auth_time = _time.monotonic()
            logger.info(
                "Vault: OIDC/JWT auth successful (mount: %s, role: %s, ttl: %ss)",
                self._auth_mount,
                self._role,
                self._token_lease_duration,
            )
            return True
        except Exception as e:
            logger.debug("Vault OIDC auth failed: %s", e, exc_info=True)
            return False

    def _try_approle(self) -> bool:
        """Authenticate using AppRole (role_id + secret_id)."""
        if not self._role_id or not self._secret_id:
            return False
        try:
            resp = self._client.auth.approle.login(
                role_id=self._role_id,
                secret_id=self._secret_id,
            )
            self._client.token = resp["auth"]["client_token"]
            self._token_lease_duration = float(resp["auth"].get("lease_duration", 3600))
            import time as _time

            self._token_auth_time = _time.monotonic()
            logger.info("Vault: AppRole auth successful.")
            return True
        except Exception as e:
            logger.debug("Vault AppRole auth failed: %s", e, exc_info=True)
            return False

    def _try_kubernetes(self) -> bool:
        """Authenticate using Kubernetes service account JWT."""
        sa_path = Path(self._k8s_sa_token_path)
        if not sa_path.exists():
            return False
        try:
            sa_jwt = sa_path.read_text().strip()
            resp = self._client.auth.kubernetes.login(
                role=self._role,
                jwt=sa_jwt,
            )
            self._client.token = resp["auth"]["client_token"]
            self._token_lease_duration = float(resp["auth"].get("lease_duration", 3600))
            import time as _time

            self._token_auth_time = _time.monotonic()
            logger.info("Vault: Kubernetes auth successful.")
            return True
        except Exception as e:
            logger.debug("Vault Kubernetes auth failed: %s", e, exc_info=True)
            return False

    def _ensure_authenticated(self) -> None:
        """Re-authenticate if the current Vault token is near expiry."""
        import time as _time

        if self._token_lease_duration <= 0:
            return  # Static token or unknown TTL — skip

        elapsed = _time.monotonic() - self._token_auth_time
        # Renew when 80% of TTL has elapsed
        if elapsed >= (self._token_lease_duration * 0.8):
            logger.info("Vault: Token nearing expiry, re-authenticating...")
            self._authenticate()

    # -- Path prefix helper ------------------------------------------------

    def _full_path(self, key: str) -> str:
        """Prepend the configured path prefix to a secret key.

        Example::

            prefix = "agents/mcp"
            _full_path("gitlab/token") → "agents/mcp/gitlab/token"
        """
        if self._path_prefix:
            return f"{self._path_prefix}/{key}"
        return key

    # -- SecretsBackend interface -------------------------------------------

    def get(self, key: str) -> str | None:
        self._ensure_authenticated()
        full_key = self._full_path(key)
        try:
            resp = self._client.secrets.kv.v2.read_secret_version(
                path=full_key, mount_point=self._mount
            )
            data = resp.get("data", {}).get("data", {})
            return data.get("value")
        except Exception:
            logger.debug("Vault get('%s') failed.", full_key, exc_info=True)
            return None

    def set(self, key: str, value: str, **metadata: Any) -> None:
        self._ensure_authenticated()
        full_key = self._full_path(key)
        secret_data = {"value": value}
        if metadata:
            secret_data.update(metadata)
        self._client.secrets.kv.v2.create_or_update_secret(
            path=full_key, secret=secret_data, mount_point=self._mount
        )
        logger.info("Secret '%s' stored (Vault, path: %s).", key, full_key)

    def delete(self, key: str) -> bool:
        self._ensure_authenticated()
        full_key = self._full_path(key)
        try:
            self._client.secrets.kv.v2.delete_metadata_and_all_versions(
                path=full_key, mount_point=self._mount
            )
            logger.info("Secret '%s' deleted (Vault, path: %s).", key, full_key)
            return True
        except Exception:
            return False

    def list_keys(self) -> list[str]:
        self._ensure_authenticated()
        prefix = self._full_path("")
        try:
            resp = self._client.secrets.kv.v2.list_secrets(
                path=prefix, mount_point=self._mount
            )
            return sorted(resp.get("data", {}).get("keys", []))
        except Exception:
            return []


# ---------------------------------------------------------------------------
# High-level Client
# ---------------------------------------------------------------------------


class SecretsClient:
    """High-level secrets client with URI resolution and env-var fallback.

    CONCEPT:OS-5.1 — Secrets & Authentication

    Wraps any ``SecretsBackend`` and adds:

    - ``get_or_env(key, env_var)`` — falls back to ``os.environ`` if the
      key is not in the backend.
    - ``resolve_ref(uri)`` — resolves ``vault://``, ``env://``, and plain
      key references.
    - Typed ``get_secret()`` returning a ``SecretValue`` Pydantic model.
    """

    def __init__(self, backend: SecretsBackend | None = None) -> None:
        self._backend = backend or InEpistemicGraphBackend()

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
            return setting(env_var)
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
        - ``env://VAR_NAME`` → ``setting(VAR_NAME)``
        - ``sqlite://key`` → backend lookup
        - Plain string → backend lookup

        Args:
            ref: Secret reference string.

        Returns:
            The resolved secret value, or ``None``.
        """
        if ref.startswith("env://"):
            var_name = ref[len("env://") :]
            return setting(var_name)
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

    CONCEPT:OS-5.1 — Secrets & Authentication

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
            backend=setting("SECRETS_BACKEND", "inmemory"),
            sqlite_path=setting("SECRETS_SQLITE_PATH"),
            vault_url=setting("SECRETS_VAULT_URL"),
            vault_mount=setting("SECRETS_VAULT_MOUNT", "secret"),
            vault_auth_method=setting("VAULT_AUTH_METHOD", "auto"),
            vault_auth_mount=setting("VAULT_AUTH_MOUNT", "jwt"),
            vault_role=setting("VAULT_ROLE"),
            vault_path_prefix=setting("VAULT_PATH_PREFIX"),
            vault_role_id=setting("VAULT_ROLE_ID"),
            vault_secret_id=setting("VAULT_SECRET_ID"),
            vault_k8s_sa_token_path=setting(
                "VAULT_K8S_SA_TOKEN_PATH",
                "/var/run/secrets/kubernetes.io/serviceaccount/token",
            ),
            master_key=setting("AGENT_SECRETS_MASTER_KEY"),
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
        backend = VaultBackend(
            url=url,
            mount_point=config.vault_mount,
            auth_method=config.vault_auth_method,
            auth_mount=config.vault_auth_mount,
            role=config.vault_role,
            path_prefix=config.vault_path_prefix,
            role_id=config.vault_role_id,
            secret_id=config.vault_secret_id,
            k8s_sa_token_path=config.vault_k8s_sa_token_path,
        )
    else:
        backend = InEpistemicGraphBackend(master_key=master_key_bytes)

    logger.info("SecretsClient initialised with '%s' backend.", config.backend)
    return SecretsClient(backend=backend)


# ---------------------------------------------------------------------------
# Post-Quantum Cryptography (ML-KEM / ML-DSA)
# ---------------------------------------------------------------------------


def generate_pq_kem_keypair() -> Any:
    """Generate a Post-Quantum ML-KEM (Kyber) keypair.

    Requires cryptography>=48.0.0.

    CONCEPT:OS-5.1 — Post-Quantum Secrecy
    """
    import importlib

    ml_kem = importlib.import_module("cryptography.hazmat.primitives.asymmetric.ml_kem")

    private_key = ml_kem.MLKEM768PrivateKey.generate()
    public_key = private_key.public_key()
    return private_key, public_key


def generate_pq_dsa_keypair() -> Any:
    """Generate a Post-Quantum ML-DSA (Dilithium) keypair.

    Requires cryptography>=48.0.0.

    CONCEPT:OS-5.1 — Post-Quantum Signatures
    """
    import importlib

    ml_dsa = importlib.import_module("cryptography.hazmat.primitives.asymmetric.ml_dsa")

    private_key = ml_dsa.MLDSA65PrivateKey.generate()
    public_key = private_key.public_key()
    return private_key, public_key
