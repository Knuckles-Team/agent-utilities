#!/usr/bin/python
from __future__ import annotations

"""Pluggable Secrets Manager.

CONCEPT:AU-OS.config.secrets-authentication — Secrets & Authentication
CONCEPT:AU-OS.identity.encrypted-secret-store — Engine-backed encrypted secret store

Provides encrypted secrets storage with two backends:

- **InEpistemicGraphBackend** (default everywhere): a *durable*, engine-backed
  store. Secrets live as ``:Secret`` nodes in a dedicated ``__secrets__``
  epistemic-graph graph; the secret VALUE is held as an **encrypted node
  property** (sealed by the engine's encryption-at-rest, CONCEPT:EG-KG.sharding.row-level-security
  ChaCha20-Poly1305 over redb value blobs, keyed through the
  ``EPISTEMIC_GRAPH_ENCRYPTION_KEY_REF`` launcher boundary + KMS seam). The key NAME and metadata stay
  queryable plaintext. There is **no local-disk / RAM fallback**: even the
  zero-infra ``tiny`` profile gets a real engine, because
  ``GraphComputeEngine`` auto-starts the full engine artifact installed by the
  hard-base ``epistemic-graph[full]>=2.23.1,<3.0.0`` dependency on demand (the
  OS-5.63 resolver); that artifact includes the encrypted store.
- **VaultBackend**: HashiCorp Vault / OpenBao integration via ``hvac`` — the
  enterprise path (UNTOUCHED by OS-5.66).

Usage::

    from agent_utilities.security.secrets_client import create_secrets_client

    client = create_secrets_client()
    client.set("gitlab/token", "glpat-xxx")
    token = client.get_or_env("gitlab/token", "GITLAB_TOKEN")

URI reference resolution::

    client.resolve_ref("vault://agents/mcp/gitlab/token")
    client.resolve_ref("env://GITLAB_TOKEN")
"""


import abc
import json
import logging
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, SecretStr

from agent_utilities.core.config import setting

logger = logging.getLogger(__name__)

#: The dedicated engine graph that holds secret nodes, isolated from all other
#: content/control-plane writes (mirrors the ``__control__`` isolation pattern,
#: CONCEPT:AU-KG.backend.schedule-on-control-graph). Its value blobs are sealed by the engine's
#: encryption-at-rest when the launcher resolves its external data-key reference.
#: (Named ``__secrets__`` — a system ``__…__`` graph like ``__control__`` /
#: ``__commons__`` — kept as a single constant so the dedicated-graph name is the
#: one place to change it.)
SECRETS_GRAPH = "__secrets__"

#: Node label for a stored secret in the ``__secrets__`` graph.
SECRET_LABEL = "Secret"


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class SecretValue(BaseModel):
    """Typed wrapper for a secret value with optional metadata.

    CONCEPT:AU-OS.config.secrets-authentication — Secrets & Authentication
    """

    value: SecretStr
    metadata: dict[str, Any] = Field(default_factory=dict)


class SecretsConfig(BaseModel):
    """Configuration for the secrets client factory.

    CONCEPT:AU-OS.config.secrets-authentication — Secrets & Authentication
    """

    model_config = ConfigDict(extra="forbid")

    backend: Literal["engine", "vault"] = Field(
        default="engine",
        description="Backend type: encrypted engine storage or 'vault'.",
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


# ---------------------------------------------------------------------------
# Abstract Backend
# ---------------------------------------------------------------------------


class SecretsBackend(abc.ABC):
    """Abstract base class for secrets storage backends.

    CONCEPT:AU-OS.config.secrets-authentication — Secrets & Authentication
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
# Engine-backed Backend (the durable default)
# ---------------------------------------------------------------------------


def _node_id(key: str) -> str:
    """Deterministic engine node id for a secret key (namespaced + escaped)."""
    return f"secret:{key}"


class InEpistemicGraphBackend(SecretsBackend):
    """Durable, engine-backed secrets store — the name is finally true.

    CONCEPT:AU-OS.identity.encrypted-secret-store — Engine-backed encrypted secret store
    CONCEPT:AU-OS.config.secrets-authentication — Secrets & Authentication

    Secrets are stored as ``:Secret`` nodes in a dedicated ``__secrets__``
    epistemic-graph graph (isolated from all other content/control-plane
    writes — cf. the ``__control__`` pattern, CONCEPT:AU-KG.backend.schedule-on-control-graph). The split
    separates queryable metadata from the encrypted value:

    - the secret **value** is held as the encrypted ``value`` node property —
      sealed on disk by the engine's encryption-at-rest (CONCEPT:EG-KG.sharding.row-level-security,
      ChaCha20-Poly1305 over redb value blobs, keyed through the
      ``EPISTEMIC_GRAPH_ENCRYPTION_KEY_REF`` launcher boundary + KMS seam);
    - the key **name** and **metadata** stay queryable plaintext properties
      (``key``, ``metadata``, ``label``) so ``list``/lookup work over the
      engine's labeled-fetch without decrypting anything.

    AgentConfig retains only the external reference. The launcher resolves and
    validates it immediately before spawning a local Rust child; the raw value
    is neither an AgentConfig field nor inherited ambient process state.

    This is the secret store in *every* profile. There is no local-disk / RAM
    fallback: ``GraphComputeEngine`` auto-starts the mandatory full engine
    artifact on demand (the OS-5.63 resolver), so an engine — and therefore the
    encrypted ``__secrets__`` graph — is always available, including on supported
    constrained hosts.
    """

    def __init__(self, graph: Any | None = None) -> None:
        if graph is None:
            from agent_utilities.knowledge_graph.core.graph_compute import (
                GraphComputeEngine,
            )

            graph = GraphComputeEngine.get_or_create(
                graph_name=SECRETS_GRAPH, backend_type="rust"
            )
        self._graph = graph

    def get(self, key: str) -> str | None:
        props = self._graph._get_node_properties(_node_id(key))
        if not props:
            return None
        val = props.get("value")
        return val if isinstance(val, str) else None

    def set(self, key: str, value: str, **metadata: Any) -> None:
        self._graph.add_node(
            _node_id(key),
            {
                "label": SECRET_LABEL,
                "node_type": SECRET_LABEL,
                "key": key,
                "value": value,
                "metadata": json.dumps(metadata) if metadata else "{}",
            },
        )
        logger.info("Secret stored in the engine backend")

    def delete(self, key: str) -> bool:
        nid = _node_id(key)
        existed = self._graph.has_node(nid)
        if existed:
            self._graph.remove_node(nid)
            logger.info("Secret deleted from the engine backend")
        return existed

    def list_keys(self) -> list[str]:
        rows = self._graph.get_nodes_by_label(SECRET_LABEL)
        keys: list[str] = []
        for _nid, props in rows:
            if isinstance(props, dict) and isinstance(props.get("key"), str):
                keys.append(props["key"])
        return sorted(keys)


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

    CONCEPT:AU-OS.config.secrets-authentication — Secrets & Authentication
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

        logger.info("Vault backend initialised auth_method=%s", self._auth_method)

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
            logger.info("Vault OIDC/JWT authentication succeeded")
            return True
        except Exception as e:
            logger.debug(
                "Vault OIDC authentication failed (exception_type=%s)",
                type(e).__name__,
            )
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
            logger.debug(
                "Vault AppRole authentication failed (exception_type=%s)",
                type(e).__name__,
            )
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
            logger.debug(
                "Vault Kubernetes authentication failed (exception_type=%s)",
                type(e).__name__,
            )
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
        path, separator, field = key.partition("#")
        full_key = self._full_path(path)
        try:
            resp = self._client.secrets.kv.v2.read_secret_version(
                path=full_key, mount_point=self._mount
            )
            data = resp.get("data", {}).get("data", {})
            return data.get(field if separator else "value")
        except Exception as exc:
            logger.debug(
                "Vault secret lookup failed (exception_type=%s)",
                type(exc).__name__,
            )
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
        logger.info("Secret stored in the configured vault backend")

    def delete(self, key: str) -> bool:
        self._ensure_authenticated()
        full_key = self._full_path(key)
        try:
            self._client.secrets.kv.v2.delete_metadata_and_all_versions(
                path=full_key, mount_point=self._mount
            )
            logger.info("Secret deleted from the configured vault backend")
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

    CONCEPT:AU-OS.config.secrets-authentication — Secrets & Authentication

    Wraps any ``SecretsBackend`` and adds:

    - ``get_or_env(key, env_var)`` — falls back to ``os.environ`` if the
      key is not in the backend.
    - ``resolve_ref(uri)`` — resolves bounded ``vault://``, ``secret://``, and
      ``env://`` runtime references.
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
        Args:
            ref: Secret reference string.

        Returns:
            The resolved secret value, or ``None``.
        """
        scheme, separator, target = str(ref or "").strip().partition("://")
        if (
            not separator
            or scheme not in {"env", "vault", "secret"}
            or not target
            or len(ref.encode("utf-8")) > 1_024
            or any(character.isspace() or ord(character) < 32 for character in ref)
            or ".." in target.split("/")
        ):
            raise ValueError("runtime secret reference is invalid")
        if scheme == "env":
            if not target.replace("_", "a").isalnum() or target[0].isdigit():
                raise ValueError("runtime secret reference is invalid")
            var_name = target
            return setting(var_name)
        if not all(character.isalnum() or character in "_./#-" for character in target):
            raise ValueError("runtime secret reference is invalid")
        return self._backend.get(target)

    def vault_sync(
        self,
        service: str,
        env_keys: list[str],
        values: dict[str, str] | None = None,
        *,
        overwrite: bool = False,
    ) -> dict[str, Any]:
        """Reconcile a service's secrets with the store (read-existing + seed).

        CONCEPT:AU-OS.deployment.vault-first-routine-genesis — the vault-first routine genesis/deployment uses so an
        operator (or Claude) never re-supplies a secret that already exists. For
        each env var name a service needs, it:

        1. **Reads** the existing value from the store at ``<service>/<KEY>`` (the
           standardized ``apps/<service>`` layout when the backend is mounted at
           ``apps``) — already-present keys are kept and reported, never re-prompted.
        2. **Writes** any value supplied in ``values`` for a key that is missing
           (or for every supplied key when ``overwrite=True``).
        3. Emits the ``vault://<service>/<KEY>`` reference for every key so the
           caller can drop resolvable refs straight into ``config.json`` (they
           round-trip through :meth:`resolve_ref`).

        Backend-agnostic: works against Vault or encrypted engine storage via the same
        ``get``/``set`` contract — ``vault://`` is just the canonical ref scheme.

        Args:
            service: Logical service name (the ``apps/<service>`` path segment).
            env_keys: The env var names the service consumes.
            values: Optional ``{KEY: value}`` to seed for missing (or all) keys.
            overwrite: When True, write every supplied value even if one exists.

        Returns:
            ``{service, refs: {KEY: "vault://<service>/<KEY>"}, present, written,
            missing}`` — ``present`` already existed, ``written`` were just stored,
            ``missing`` have neither a stored value nor a supplied one.
        """
        values = values or {}
        refs: dict[str, str] = {}
        present: list[str] = []
        written: list[str] = []
        missing: list[str] = []
        for key in env_keys:
            store_key = f"{service}/{key}"
            refs[key] = f"vault://{store_key}"
            existing = self._backend.get(store_key)
            if existing is not None and not overwrite:
                present.append(key)
                continue
            supplied = values.get(key)
            if supplied is not None and supplied != "":
                self._backend.set(store_key, supplied, service=service)
                written.append(key)
            elif existing is not None:
                # overwrite requested but no new value — keep the existing one.
                present.append(key)
            else:
                missing.append(key)
        return {
            "service": service,
            "refs": refs,
            "present": present,
            "written": written,
            "missing": missing,
        }


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_secrets_client(config: SecretsConfig | None = None) -> SecretsClient:
    """Create a ``SecretsClient`` from configuration.

    CONCEPT:AU-OS.config.secrets-authentication — Secrets & Authentication
    CONCEPT:AU-OS.identity.encrypted-secret-store — Engine-backed encrypted secret store

    The backend is selected by ``config.backend``:

    - ``"engine"`` (default everywhere): the durable engine-backed
      ``__secrets__`` store (the OS-5.63 resolver auto-starts the mandatory full
      engine artifact when nothing is running, so this works in every profile). No
      local-disk / RAM fallback.
    - ``"vault"``: HashiCorp Vault / OpenBao KV v2 (enterprise, UNTOUCHED).

    Args:
        config: Secrets configuration. If ``None``, reads from environment
            variables (``SECRETS_BACKEND``, ``SECRETS_VAULT_URL``,
            ``SECRETS_VAULT_MOUNT``).

    Returns:
        A configured ``SecretsClient`` instance.
    """
    if config is None:
        config = SecretsConfig(
            backend=setting("SECRETS_BACKEND", "engine"),
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
        )

    if config.backend == "vault":
        url = config.vault_url or "http://127.0.0.1:8200"
        backend: SecretsBackend = VaultBackend(
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
        logger.info("SecretsClient initialised with 'vault' backend.")
        return SecretsClient(backend=backend)

    # Default everywhere: the durable engine-backed encrypted ``__secrets__``
    # store. The OS-5.63 resolver auto-starts the mandatory full engine artifact
    # when nothing is running, so this is the store in every profile —
    # there is no local-disk / RAM fallback (CONCEPT:AU-OS.identity.encrypted-secret-store).
    engine_backend = InEpistemicGraphBackend()
    logger.info("SecretsClient initialised with engine-backed backend.")
    return SecretsClient(backend=engine_backend)


# ---------------------------------------------------------------------------
# Post-Quantum Cryptography (ML-KEM / ML-DSA)
# ---------------------------------------------------------------------------


def generate_pq_kem_keypair() -> Any:
    """Generate a Post-Quantum ML-KEM (Kyber) keypair.

    Requires cryptography>=48.0.0.

    CONCEPT:AU-OS.config.secrets-authentication — Post-Quantum Secrecy
    """
    import importlib

    ml_kem = importlib.import_module("cryptography.hazmat.primitives.asymmetric.ml_kem")

    private_key = ml_kem.MLKEM768PrivateKey.generate()
    public_key = private_key.public_key()
    return private_key, public_key


def generate_pq_dsa_keypair() -> Any:
    """Generate a Post-Quantum ML-DSA (Dilithium) keypair.

    Requires cryptography>=48.0.0.

    CONCEPT:AU-OS.config.secrets-authentication — Post-Quantum Signatures
    """
    import importlib

    ml_dsa = importlib.import_module("cryptography.hazmat.primitives.asymmetric.ml_dsa")

    private_key = ml_dsa.MLDSA65PrivateKey.generate()
    public_key = private_key.public_key()
    return private_key, public_key
