"""LMCache / vLLM remote-backend connector for the epistemic-graph KV-cache.

CONCEPT:AU-KG.backend.remote-kvcache-contract — the Python half of the EG-187 remote KV-cache contract.

The engine exposes a shared, content-addressed KV-cache backend
(``eg_kvcache::SharedKvIndex``, CONCEPT:EG-KG.enrichment.content-address-separation) over a small HTTP surface
(CONCEPT:EG-KG.backend.is-configured-so-co). This module drives that surface from Python so that parallel
vLLM / LMCache workers pool their KV-cache blocks by token-hash: an identical KV
page produced by two workers is stored **once** (dedup) and a cold worker can
fetch a page a warm worker already computed.

Wire contract (see ``epistemic-graph/docs/architecture/kvcache-remote-backend.md``):

===============  ==========================================================
``GET  /kv/<hash>``       block bytes (``200``) or ``404`` if absent
``PUT  /kv/<hash>``       store binary body; ``201`` new / ``200`` dedup hit
``HEAD /kv/<hash>``       ``200`` present / ``404`` absent (existence probe)
``GET  /kv/<hash>/exists``  ``200`` JSON ``{"hash":…,"exists":bool}``
``GET  /kv/stats``        ``200`` JSON occupancy + dedup stats
===============  ==========================================================

``<hash>`` is the **caller's opaque token-hash key** (what LMCache computes over
the token ids of a block); the engine stores the body verbatim under it and does
NOT re-hash. Auth is an optional ``Authorization: Bearer <token>`` guard.

Graceful degradation is a hard requirement: this sits on the inference hot path,
so every network / protocol error is swallowed and mapped to a cache **miss**
(``get`` → ``None``, ``contains`` → ``False``, ``put`` → ``False``,
``stats`` → empty). The connector must never crash token generation.

Registering with vLLM / LMCache
-------------------------------
LMCache drives a *remote backend* through a ``get(key) -> bytes | None`` /
``put(key, bytes)`` / ``contains(key) -> bool`` shape (behind a local
``TieredCache`` / ``SharedKvIndex`` L1 that falls through to the network only on a
local miss). :class:`EpistemicGraphKVBackend` implements exactly that shape and
is drop-in wherever a custom LMCache/vLLM remote-backend object is accepted::

    from agent_utilities.kvcache import EpistemicGraphKVBackend

    backend = EpistemicGraphKVBackend.from_env()   # reads EG-187 env
    # hand `backend` to your LMCache RemoteBackend registration, or wrap it in
    # the lmcache StorageBackendInterface adapter if that package is installed.

If the real ``lmcache`` package is importable its abstract base is detected (see
:data:`_LMCACHE_BASE`); today it is treated as an *optional* mixin so this
connector stays usable — and unit-testable — with lmcache absent.
"""

from __future__ import annotations

import json
import logging
from types import TracebackType
from typing import TYPE_CHECKING, Any
from urllib.parse import quote

import httpx
from pydantic import BaseModel

from agent_utilities.core.http_client import create_http_client
from agent_utilities.kvcache.config import KvCacheConfig

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from agent_utilities.mcp.client_credentials import ClientCredentialsAuth

logger = logging.getLogger(__name__)


# --- Optional lmcache abstract-base detection (CONCEPT:AU-KG.backend.remote-kvcache-contract) --------------
# If the real lmcache package is installed we expose its abstract backend base so
# callers can adapt; absent (the common case in this repo's test env) we fall
# back to `object` and remain a standalone, fully-usable connector.
try:  # pragma: no cover - exercised only where lmcache is installed
    from lmcache.storage_backend.abstract_backend import (  # type: ignore[import-not-found]
        LMCBackendInterface as _lmcache_base,
    )

    _LMCACHE_BASE: type | None = _lmcache_base
except Exception:  # noqa: BLE001 - any import failure ⇒ standalone mode
    _LMCACHE_BASE = None

LMCACHE_AVAILABLE = _LMCACHE_BASE is not None


class KvCacheStats(BaseModel):
    """Parsed ``GET /kv/stats`` response (CONCEPT:AU-KG.backend.remote-kvcache-contract).

    Fields mirror the engine's occupancy + dedup counters. Unknown extra keys
    are ignored so a newer engine can add counters without breaking the client.
    """

    unique_blocks: int = 0
    total_refs: int = 0
    resident_bytes: int = 0
    logical_bytes: int = 0
    dedup_savings_bytes: int = 0
    dedup_hits: int = 0
    get_hits: int = 0
    get_misses: int = 0


class EpistemicGraphKVBackend:
    """LMCache/vLLM remote backend backed by the engine's ``/kv`` HTTP surface.

    CONCEPT:AU-KG.backend.remote-kvcache-contract. Implements the LMCache remote-backend shape
    (``get`` / ``put`` / ``contains`` / ``stats``) against EG-187 with a pooled,
    keep-alive :class:`httpx.Client` (connection reuse), a short per-request
    timeout, an optional bearer token, and total graceful degradation — every
    error is a cache miss, never a raised exception on the inference path.

    Args:
        config: Endpoint / auth / timeout settings. Defaults to
            :class:`KvCacheConfig` defaults; use :meth:`from_env` to source from
            the engine's EG-187 environment.
        client: Optional pre-built :class:`httpx.Client` (dependency injection —
            unit tests pass one wired to :class:`httpx.MockTransport`). When
            supplied the connector does not own its lifecycle and will not close
            it.
    """

    def __init__(
        self,
        config: KvCacheConfig | None = None,
        *,
        client: httpx.Client | None = None,
    ) -> None:
        self.config = config or KvCacheConfig()
        self._owns_client = client is None
        self._client = client if client is not None else self._build_client()

    # -- construction ---------------------------------------------------------
    @classmethod
    def from_env(cls) -> EpistemicGraphKVBackend:
        """Build a connector from the engine's EG-187 environment (KG-2.306)."""
        return cls(KvCacheConfig.from_env())

    def _build_client(self) -> httpx.Client:
        # Auth precedence (paired with the platform's overall auth, not a separate
        # mechanism): JWT FIRST — the same Keycloak client-credentials bearer graph-os
        # mints for the fleet, via the shared self-refreshing provider
        # (``ClientCredentialsAuth``: per-request token, auto-refresh before expiry,
        # one-shot re-mint + retry on 401 — so token rotation and cold restarts are
        # handled automatically). Falls back to a static ``EPISTEMIC_GRAPH_KVCACHE_TOKEN``
        # bearer (the documented OpenBao-sourced option) when OIDC isn't configured
        # (e.g. a standalone vLLM/LMCache worker), and to anonymous otherwise.
        # Lazy + guarded: a standalone vLLM/LMCache worker must not need the mcp
        # layer to import this connector — if it's unavailable, degrade to the
        # static-token / anonymous path.
        try:
            from agent_utilities.mcp.client_credentials import bearer_auth
        except Exception:  # pragma: no cover - mcp layer optional on inference hosts

            def bearer_auth(existing: dict | None) -> ClientCredentialsAuth | None:
                return None

        headers: dict[str, str] = {}
        auth = bearer_auth(headers)  # ClientCredentialsAuth | None (None ⇒ OIDC off)
        if auth is None and self.config.token:
            headers["Authorization"] = f"Bearer {self.config.token}"
        return create_http_client(
            timeout=self.config.timeout_s,
            verify=self.config.verify_tls,
            headers=headers,
            base_url=self.config.base_url,
            limits=httpx.Limits(max_connections=self.config.max_connections),
            auth=auth,
        )

    # -- key handling ---------------------------------------------------------
    @staticmethod
    def _path(key: str) -> str:
        """Path for a block key. The key is opaque; percent-encode for URL safety.

        The engine stores the body verbatim under the decoded key, so encoding
        here is purely transport hygiene (keys may contain ``/`` or other
        reserved characters).
        """
        return f"/kv/{quote(str(key), safe='')}"

    # -- LMCache remote-backend contract (CONCEPT:AU-KG.backend.remote-kvcache-contract) -------------------
    def get(self, key: str) -> bytes | None:
        """Fetch KV-block bytes for ``key`` (LMCache load).

        Returns the block bytes on a ``200`` hit, ``None`` on a ``404`` miss, and
        ``None`` (a miss) on any transport/protocol error — never raises.
        """
        try:
            resp = self._client.get(self._path(key))
        except httpx.HTTPError as exc:
            logger.warning("kvcache get(%s) failed, treating as miss: %s", key, exc)
            return None
        if resp.status_code == 200:
            return resp.content
        if resp.status_code != 404:
            logger.warning(
                "kvcache get(%s) unexpected status %s, treating as miss",
                key,
                resp.status_code,
            )
        return None

    def put(self, key: str, value: bytes) -> bool:
        """Store KV-block ``value`` under ``key`` (LMCache offload).

        Returns ``True`` when the engine accepted the block (``200`` dedup hit or
        ``201`` newly created) and ``False`` on any error — a failed offload is
        non-fatal (the block simply is not pooled).
        """
        try:
            resp = self._client.put(
                self._path(key),
                content=value,
                headers={"Content-Type": "application/octet-stream"},
            )
        except httpx.HTTPError as exc:
            logger.warning("kvcache put(%s) failed, dropping offload: %s", key, exc)
            return False
        if resp.status_code in (200, 201):
            return True
        logger.warning("kvcache put(%s) rejected with status %s", key, resp.status_code)
        return False

    def contains(self, key: str) -> bool:
        """Existence probe for ``key`` via ``HEAD /kv/<hash>`` (KG-2.306).

        ``True`` on ``200``, ``False`` on ``404`` or any error. Used to skip an
        upload the cluster already has.
        """
        try:
            resp = self._client.head(self._path(key))
        except httpx.HTTPError as exc:
            logger.warning("kvcache contains(%s) failed, assuming absent: %s", key, exc)
            return False
        return resp.status_code == 200

    def exists(self, key: str) -> bool:
        """Method-agnostic existence probe via ``GET /kv/<hash>/exists`` (KG-2.306).

        Some LMCache call sites cannot issue ``HEAD``; this uses the JSON
        ``{"hash":…,"exists":bool}`` endpoint instead. Errors ⇒ ``False``.
        """
        try:
            resp = self._client.get(f"{self._path(key)}/exists")
        except httpx.HTTPError as exc:
            logger.warning("kvcache exists(%s) failed, assuming absent: %s", key, exc)
            return False
        if resp.status_code != 200:
            return False
        try:
            body: Mapping[str, Any] = resp.json()
        except (json.JSONDecodeError, ValueError):
            return False
        return bool(body.get("exists", False))

    def stats(self) -> KvCacheStats:
        """Fetch occupancy + dedup stats via ``GET /kv/stats`` (KG-2.306).

        Returns a parsed :class:`KvCacheStats`; on any error returns an
        all-zero instance rather than raising.
        """
        try:
            resp = self._client.get("/kv/stats")
            if resp.status_code != 200:
                return KvCacheStats()
            return KvCacheStats.model_validate(resp.json())
        except (httpx.HTTPError, json.JSONDecodeError, ValueError) as exc:
            logger.warning("kvcache stats() failed: %s", exc)
            return KvCacheStats()

    # -- zero-copy snapshot → fork (CONCEPT:EG-KG.memory.zero-copy-snapshot-fork) ----------
    # The engine's ``eg-kvcache`` crate exposes a "snapshot → branch" primitive on
    # its content-addressed shared KV store: ``snapshot(keys)`` pins a set of
    # already-resident pages, ``fork(snapshot)`` fans out N branches that all read
    # those SAME physical pages by ``Arc`` (one copy regardless of N), and a
    # ``branch_put`` is copy-on-write per branch. Surfaced over the same ``/kv``
    # HTTP surface as ``POST /kv/snapshot`` + ``POST /kv/snapshot/<id>/fork`` +
    # ``GET|PUT /kv/branch/<bid>/<key>`` (``GET /kv/fork/stats`` proves resident
    # bytes stay flat vs branch count). Same graceful-degradation posture as the
    # remote-backend contract above: every error is a safe default, never a raise.
    #
    # HONEST SCOPE: this drives the SHARING/plumbing of pages that ALREADY EXIST in
    # the store. It does NOT produce KV pages — the vLLM/LMCache model-side mapping
    # of live attention KV onto this store is external (see ``put``). Snapshotting a
    # key the store has never seen simply pins nothing for it.
    @staticmethod
    def _branch_path(branch_id: int, key: str) -> str:
        """URL path for a branch-local key (the opaque key is percent-encoded)."""
        return f"/kv/branch/{int(branch_id)}/{quote(str(key), safe='')}"

    def snapshot(self, keys: Sequence[str]) -> int | None:
        """Pin ``keys`` into an immutable snapshot via ``POST /kv/snapshot``.

        Returns the integer snapshot id on success, or ``None`` on any transport /
        protocol error or a non-200 status — never raises (the caller falls back to
        the per-branch copy path when snapshotting is unavailable).
        """
        try:
            resp = self._client.post(
                "/kv/snapshot", json={"keys": [str(k) for k in keys]}
            )
        except httpx.HTTPError as exc:
            logger.warning("kvcache snapshot() failed: %s", exc)
            return None
        if resp.status_code != 200:
            logger.warning("kvcache snapshot() unexpected status %s", resp.status_code)
            return None
        try:
            return int(resp.json()["snapshot"])
        except (json.JSONDecodeError, ValueError, KeyError, TypeError) as exc:
            logger.warning("kvcache snapshot() malformed response: %s", exc)
            return None

    def fork(self, snapshot_id: int) -> int | None:
        """Fork a copy-on-write branch off ``snapshot_id`` via ``POST /kv/snapshot/<id>/fork``.

        All branches forked off one snapshot share its pages zero-copy (``Arc``);
        this is the rung that makes fanning out N branches O(1) in copies. Returns
        the integer branch id, or ``None`` on any error — never raises.
        """
        try:
            resp = self._client.post(f"/kv/snapshot/{int(snapshot_id)}/fork")
        except httpx.HTTPError as exc:
            logger.warning("kvcache fork(%s) failed: %s", snapshot_id, exc)
            return None
        if resp.status_code != 200:
            logger.warning(
                "kvcache fork(%s) unexpected status %s", snapshot_id, resp.status_code
            )
            return None
        try:
            return int(resp.json()["branch"])
        except (json.JSONDecodeError, ValueError, KeyError, TypeError) as exc:
            logger.warning("kvcache fork(%s) malformed response: %s", snapshot_id, exc)
            return None

    def branch_get(self, branch_id: int, key: str) -> bytes | None:
        """Read a branch-local page via ``GET /kv/branch/<bid>/<key>``.

        Reads through to the shared snapshot page unless this branch has written its
        own copy-on-write override. Returns the bytes on a ``200`` hit, ``None`` on a
        ``404`` miss, and ``None`` on any error — never raises.
        """
        try:
            resp = self._client.get(self._branch_path(branch_id, key))
        except httpx.HTTPError as exc:
            logger.warning("kvcache branch_get(%s,%s) failed: %s", branch_id, key, exc)
            return None
        if resp.status_code == 200:
            return resp.content
        if resp.status_code != 404:
            logger.warning(
                "kvcache branch_get(%s,%s) unexpected status %s, treating as miss",
                branch_id,
                key,
                resp.status_code,
            )
        return None

    def branch_put(self, branch_id: int, key: str, value: bytes) -> bool:
        """Write a copy-on-write branch-local page via ``PUT /kv/branch/<bid>/<key>``.

        The write is private to this branch (siblings keep reading the shared page),
        which is what makes ``max_concurrency>1`` fan-out safe. Returns ``True`` on a
        ``200``/``201`` accept and ``False`` on any error — never raises.
        """
        try:
            resp = self._client.put(
                self._branch_path(branch_id, key),
                content=value,
                headers={"Content-Type": "application/octet-stream"},
            )
        except httpx.HTTPError as exc:
            logger.warning("kvcache branch_put(%s,%s) failed: %s", branch_id, key, exc)
            return False
        if resp.status_code in (200, 201):
            return True
        logger.warning(
            "kvcache branch_put(%s,%s) rejected with status %s",
            branch_id,
            key,
            resp.status_code,
        )
        return False

    def fork_stats(self) -> dict[str, Any]:
        """Fetch snapshot/branch occupancy via ``GET /kv/fork/stats``.

        Returns the raw JSON dict (``branches``, ``shared_bytes``, ``shared_pages``,
        ``overlay_bytes``, ``overlay_pages``, ``resident_fork_bytes``, ``snapshots``)
        — ``shared_*`` staying flat as ``branches`` grows is the zero-copy proof.
        Returns an empty ``{}`` on any error rather than raising.
        """
        try:
            resp = self._client.get("/kv/fork/stats")
            if resp.status_code != 200:
                return {}
            body = resp.json()
        except (httpx.HTTPError, json.JSONDecodeError, ValueError) as exc:
            logger.warning("kvcache fork_stats() failed: %s", exc)
            return {}
        return dict(body) if isinstance(body, dict) else {}

    # -- lifecycle ------------------------------------------------------------
    def close(self) -> None:
        """Close the pooled client if this connector owns it (KG-2.306)."""
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> EpistemicGraphKVBackend:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()
