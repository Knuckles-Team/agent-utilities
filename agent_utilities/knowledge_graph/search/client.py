"""Thin ``opensearch-py`` wrapper — the au OpenSearch client.

(CONCEPT:AU-KG.retrieval.opensearch-cdc-indexer, CA-24, DEC-CA-09)

Mirrors the fleet convention of wrapping a real client library rather than
hand-rolling REST calls (cf. ``backends/sparql/jena_fuseki_backend.py``'s
``execute``/``upload_graph`` shape, and the "inject a client for tests"
pattern :mod:`knowledge_graph.streams.kafka_adapter` already uses for
``aiokafka``). ``opensearch-py`` is an optional dependency (the
``[opensearch]`` extra) — importing this module never requires it installed;
constructing a live connection does.

No deployment hostname is embedded here (the fleet-wide "GitHub public vs
GitLab internal standards" rule — this package ships publicly). The default
endpoint is OpenSearch's own generic ``localhost:9200``; every real
deployment (e.g. CA-50's ``https://example-opensearch.arpa`` behind the homelab CA,
security-plugin basic auth) is supplied via ``OPENSEARCH_URL`` (routed
through ``agent_utilities.core._env.setting``, never a bare ``os.environ``
read — the codebase-wide ``check_no_env_sprawl.py`` rule) or by
constructing :class:`OpenSearchClientConfig` directly.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, TypedDict

logger = logging.getLogger(__name__)

__all__ = [
    "OpenSearchClientConfig",
    "OpenSearchClient",
    "OpenSearchNotFoundError",
    "BulkResult",
]

_DEFAULT_ENDPOINT = "http://localhost:9200"
# OpenSearch's own generic default port -- no internal deployment
# hostname is baked into this shipped-publicly package.


class BulkResult(TypedDict):
    """This module's own shape for :meth:`OpenSearchClient.bulk` -- unlike
    ``index_document``/``search`` (raw external-library passthroughs typed
    ``Any``), this dict IS constructed here, so it gets a real contract."""

    success: int
    errors: list[Any]


class OpenSearchNotFoundError(LookupError):
    """Raised by :meth:`OpenSearchClient.get_document` when the document is
    absent — distinct from a connection/auth failure, which raises through.
    """


@dataclass
class OpenSearchClientConfig:
    """Static, schema-worthy connection config (the ``AgentConfig`` shape
    the module docstring of ``core._env`` says static infra config should
    take) — read once, not a live per-call env poll.
    """

    endpoint: str = _DEFAULT_ENDPOINT
    username: str | None = None
    password: str | None = None
    verify_certs: bool = True
    ca_certs: str | None = None
    timeout: int = 30

    @classmethod
    def from_env(cls) -> OpenSearchClientConfig:
        from agent_utilities.core._env import setting

        return cls(
            endpoint=setting("OPENSEARCH_URL", default=_DEFAULT_ENDPOINT),
            username=setting("OPENSEARCH_USER", default=None),
            password=setting("OPENSEARCH_PASSWORD", default=None),
            # No explicit cast=bool here — a REAL bug this lane's own live
            # proof caught (2026-08-26): `cast=bool` calls Python's bare
            # `bool(...)` constructor on the raw string, and `bool("false")`
            # is `True` (any non-empty string is truthy) — so
            # OPENSEARCH_VERIFY_CERTS=false would NOT have disabled cert
            # verification, in either direction. Omitting `cast` lets
            # `setting()` auto-infer `to_boolean` from `default=True`'s type
            # (see `core._env.setting`'s own docstring), which correctly
            # parses {"false","0","no"} etc.
            verify_certs=setting("OPENSEARCH_VERIFY_CERTS", default=True),
            ca_certs=setting("OPENSEARCH_CA_CERTS", default=None),
            timeout=setting("OPENSEARCH_TIMEOUT_S", default=30, cast=int),
        )


class OpenSearchClient:
    """Index/get/search/bulk/delete operations over one OpenSearch cluster.

    Accepts an injected ``client`` (any object exposing the ``opensearch-py``
    ``OpenSearch`` surface used here: ``index``/``get``/``delete``/``search``/
    ``count``/``bulk``/``indices``/``delete_by_query``) so every caller —
    the indexer, ``rebuild.py``, and every unit test — can run against an
    in-process test double without a live cluster, exactly the seam
    ``KafkaStreamAdapter``/``DebeziumKafkaConsumer`` already use for
    ``aiokafka``.
    """

    def __init__(
        self,
        config: OpenSearchClientConfig | None = None,
        *,
        client: Any = None,
    ) -> None:
        self.config = config or OpenSearchClientConfig()
        self._client = client
        self._owns_client = client is None

    # -- connection -----------------------------------------------------

    def connect(self) -> None:
        if self._client is not None:
            return
        try:
            from opensearchpy import OpenSearch
        except ImportError as exc:  # pragma: no cover - optional dep
            raise RuntimeError(
                "OpenSearch client requires 'opensearch-py'. "
                "Install agent-utilities[opensearch]."
            ) from exc
        http_auth = None
        if self.config.username:
            http_auth = (self.config.username, self.config.password or "")
        self._client = OpenSearch(
            hosts=[self.config.endpoint],
            http_auth=http_auth,
            use_ssl=self.config.endpoint.startswith("https://"),
            verify_certs=self.config.verify_certs,
            ca_certs=self.config.ca_certs,
            timeout=self.config.timeout,
        )
        logger.info("OpenSearch client connected to %s", self.config.endpoint)

    @property
    def raw(self) -> Any:
        """The underlying ``opensearch-py`` client (or injected double)."""
        if self._client is None:
            self.connect()
        return self._client

    # -- index lifecycle --------------------------------------------------

    def index_exists(self, index: str) -> bool:
        return bool(self.raw.indices.exists(index=index))

    def ensure_index(
        self,
        index: str,
        *,
        mappings: dict[str, Any] | None = None,
        settings: dict[str, Any] | None = None,
    ) -> bool:
        """Create ``index`` if absent. Returns True if it was created."""
        if self.index_exists(index):
            return False
        body: dict[str, Any] = {}
        if mappings is not None:
            body["mappings"] = mappings
        if settings is not None:
            body["settings"] = settings
        self.raw.indices.create(index=index, body=body or None)
        return True

    def delete_index(self, index: str) -> None:
        if self.index_exists(index):
            self.raw.indices.delete(index=index)

    # -- documents ----------------------------------------------------------

    def index_document(
        self,
        index: str,
        doc_id: str,
        body: dict[str, Any],
        *,
        refresh: bool | str = False,
    ) -> Any:
        """Returns the raw ``opensearch-py`` response verbatim -- an
        external library's response shape, not this module's contract to
        model or constrain."""
        return self.raw.index(index=index, id=doc_id, body=body, refresh=refresh)

    def get_document(self, index: str, doc_id: str) -> dict[str, Any] | None:
        """Return the document's ``_source`` dict, or ``None`` if absent.

        Distinguishes "not found" from every other failure: a genuine
        connection/auth/server error raises through rather than being
        collapsed into "treat as absent" — the caller's ordering check must
        never mistake an unreachable cluster for a fresh node.
        """
        try:
            from opensearchpy.exceptions import NotFoundError
        except ImportError:  # pragma: no cover - optional dep
            NotFoundError = OpenSearchNotFoundError  # type: ignore[assignment]
        try:
            result = self.raw.get(index=index, id=doc_id)
        except NotFoundError:
            return None
        except Exception as exc:  # noqa: BLE001 - re-raise as our own 404 only when it truly is one
            if _looks_like_not_found(exc):
                return None
            raise
        return result.get("_source")

    def delete_document(self, index: str, doc_id: str) -> bool:
        """Delete one document. Returns True if it existed and was deleted,
        False if it was already absent (idempotent — never raises on a
        redelivered tombstone for an already-deleted node)."""
        try:
            from opensearchpy.exceptions import NotFoundError
        except ImportError:  # pragma: no cover - optional dep
            NotFoundError = OpenSearchNotFoundError  # type: ignore[assignment]
        try:
            self.raw.delete(index=index, id=doc_id)
            return True
        except NotFoundError:
            return False
        except Exception as exc:  # noqa: BLE001
            if _looks_like_not_found(exc):
                return False
            raise

    def delete_by_query(self, index_pattern: str, query: dict[str, Any]) -> int:
        """Delete every doc in ``index_pattern`` matching ``query``. Returns
        the deleted count. A no-match is not an error (0 deleted)."""
        result = self.raw.delete_by_query(
            index=index_pattern, body={"query": query}, ignore_unavailable=True
        )
        return int(result.get("deleted", 0))

    def search(
        self,
        index: str,
        body: dict[str, Any],
        *,
        client: Any = None,
    ) -> Any:
        """Run a search. Returns the raw ``opensearch-py`` response verbatim
        (an external library's response shape). ``client`` overrides
        ``self.raw`` for exactly one call — the seam DLS negative-control
        tests use to search as a DIFFERENT authenticated role (a distinct
        ``OpenSearch`` instance bound to a restricted user's credentials)
        without constructing a whole second :class:`OpenSearchClient`."""
        target = client if client is not None else self.raw
        return target.search(index=index, body=body)

    def count(self, index: str, body: dict[str, Any] | None = None) -> int:
        result = self.raw.count(index=index, body=body)
        return int(result.get("count", 0))

    def bulk(self, actions: list[dict[str, Any]]) -> BulkResult:
        from opensearchpy.helpers import bulk as _bulk

        success, errors = _bulk(self.raw, actions, raise_on_error=False)
        return BulkResult(success=success, errors=errors)


def _looks_like_not_found(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None)
    return status == 404
