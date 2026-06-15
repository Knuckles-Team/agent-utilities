"""Enterprise-architecture tool clients for the KG federation (CONCEPT:KG-2.9).

The single place the knowledge graph resolves *ready* clients for the EA
systems-of-record it mirrors and writes back to — LeanIX (and, best-effort,
Archi). Both ``get_*_client`` helpers return ``None`` when the tool is not
configured, so every consumer (ingest hydration, capability/relationship
write-back) degrades to a clean no-op rather than raising.

Design notes:

* **Self-contained transport.** ``LeanixEAClient`` talks to LeanIX directly over
  ``httpx`` — MTM OAuth token exchange then the Pathfinder GraphQL/REST APIs —
  the same proven flow as ``egeria_mcp.harvest.leanix``. No MCP-multiplexer
  round-trip, no extra infra; mockable in tests by patching ``httpx``.
* **Config-driven, no bare env reads.** URL/token resolve through
  :func:`agent_utilities.core.config.setting` (config.json-injected), per the
  repo's configuration discipline.
* **Tolerant.** Network/transport failures degrade to ``[]`` / ``{}`` with a
  debug log, never an exception that aborts an ingest or write-back batch.

The client exposes exactly the surface the KG needs:

* :meth:`LeanixEAClient.meta_model` — the live data model (fact sheet types,
  fields, relations) that drives the metamodel→OWL compiler (Piece 1).
* :meth:`LeanixEAClient.factsheets` — paginated mirror read with embedded typed
  relations (the duck-typed surface the LeanIX extractor consumes), with an
  optional ``since`` watermark and ``ids`` narrowing for delta sync (Piece 3).
* write methods (``create_fact_sheet`` / ``update_fact_sheet`` /
  ``create_fact_sheet_relation`` / ``add_tag``) for backfeed (Piece 4).
"""

from __future__ import annotations

import json
import logging
from typing import Any

from agent_utilities.core.config import setting

logger = logging.getLogger(__name__)

try:
    import httpx

    _HTTPX = True
except Exception:  # pragma: no cover - httpx is a core dep, guard anyway
    _HTTPX = False


# ── LeanIX ──────────────────────────────────────────────────────────────────


class LeanixEAClient:
    """Raw-httpx facade over the LeanIX Pathfinder API (read + write).

    Auth: the API token is exchanged for a short-lived bearer via the MTM
    OAuth endpoint; the bearer is cached for the client's lifetime and
    re-minted on demand.
    """

    def __init__(
        self,
        base_url: str,
        api_token: str,
        *,
        verify_ssl: bool = False,
        timeout: float = 30.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._api_token = api_token
        self._verify_ssl = verify_ssl
        self._timeout = timeout
        self._bearer: str | None = None
        self._data_model: dict[str, Any] | None = None

    # -- transport -----------------------------------------------------------

    def _get_bearer(self) -> str | None:
        if self._bearer:
            return self._bearer
        if not _HTTPX:
            return None
        try:
            with httpx.Client(verify=self._verify_ssl, timeout=self._timeout) as c:
                r = c.post(
                    f"{self.base_url}/services/mtm/v1/oauth2/token",
                    data={"grant_type": "client_credentials"},
                    auth=("apitoken", self._api_token),
                )
            if r.status_code == 200:
                self._bearer = r.json().get("access_token")
            else:
                logger.debug("LeanIX token exchange failed: HTTP %s", r.status_code)
        except Exception as exc:  # noqa: BLE001 - tolerant transport
            logger.debug("LeanIX token exchange error: %s", exc)
        return self._bearer

    def _headers(self) -> dict[str, str] | None:
        bearer = self._get_bearer()
        if not bearer:
            return None
        return {"Authorization": f"Bearer {bearer}", "Content-Type": "application/json"}

    def _gql(self, query: str, variables: dict[str, Any] | None = None) -> dict:
        """Execute a Pathfinder GraphQL operation; returns the ``data`` map ({} on error)."""
        headers = self._headers()
        if not headers or not _HTTPX:
            return {}
        try:
            with httpx.Client(verify=self._verify_ssl, timeout=self._timeout) as c:
                r = c.post(
                    f"{self.base_url}/services/pathfinder/v1/graphql",
                    headers=headers,
                    json={"query": query, "variables": variables or {}},
                )
            if r.status_code != 200:
                logger.debug("LeanIX GraphQL HTTP %s", r.status_code)
                return {}
            body = r.json() or {}
            if body.get("errors"):
                logger.debug("LeanIX GraphQL errors: %s", body["errors"])
            return body.get("data") or {}
        except Exception as exc:  # noqa: BLE001 - tolerant transport
            logger.debug("LeanIX GraphQL error: %s", exc)
            return {}

    def _rest(self, method: str, path: str) -> Any:
        headers = self._headers()
        if not headers or not _HTTPX:
            return None
        try:
            with httpx.Client(verify=self._verify_ssl, timeout=self._timeout) as c:
                r = c.request(
                    method,
                    f"{self.base_url}/services/pathfinder/v1{path}",
                    headers=headers,
                )
            if r.status_code != 200:
                logger.debug("LeanIX REST %s %s -> HTTP %s", method, path, r.status_code)
                return None
            body = r.json() or {}
            # Pathfinder wraps payloads under {"data": ...}; unwrap when present.
            return body.get("data", body) if isinstance(body, dict) else body
        except Exception as exc:  # noqa: BLE001 - tolerant transport
            logger.debug("LeanIX REST error %s %s: %s", method, path, exc)
            return None

    # -- metamodel -----------------------------------------------------------

    def meta_model(self, *, refresh: bool = False) -> dict[str, Any]:
        """Return the live LeanIX data model (fact sheet types, fields, relations).

        Prefers ``/models/dataModel`` (the rich type/field/relation schema);
        falls back to ``/metaModel``. Cached for the client's lifetime.
        """
        if self._data_model is not None and not refresh:
            return self._data_model
        model = self._rest("GET", "/models/dataModel")
        if not isinstance(model, dict) or not model:
            model = self._rest("GET", "/metaModel")
        self._data_model = model if isinstance(model, dict) else {}
        return self._data_model

    def _fact_sheet_types(self) -> dict[str, Any]:
        """Map of ``{factSheetType: typeDefinition}`` from the data model (tolerant)."""
        model = self.meta_model()
        fs = model.get("factSheets")
        if isinstance(fs, dict):
            return fs
        if isinstance(fs, list):
            return {t.get("type") or t.get("name"): t for t in fs if isinstance(t, dict)}
        return {}

    def _relation_fields(self, fs_type: str) -> list[str]:
        """Relation field names (``rel*``) declared for ``fs_type`` in the data model."""
        defn = self._fact_sheet_types().get(fs_type) or {}
        rels = defn.get("relations")
        names: list[str] = []
        if isinstance(rels, dict):
            names = list(rels.keys())
        elif isinstance(rels, list):
            names = [r.get("name") for r in rels if isinstance(r, dict) and r.get("name")]
        # Tolerant fallback: relation fields conventionally start with "rel".
        return [n for n in names if isinstance(n, str) and n.startswith("rel")]

    # -- mirror read ---------------------------------------------------------

    def factsheets(
        self,
        type: str | None = None,  # noqa: A002 - matches the extractor's duck-typed call
        *,
        since: str | None = None,
        ids: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Fetch fact sheets (optionally one type), with embedded typed relations.

        Each returned dict carries ``id``, ``name``, ``type``, ``updatedAt``,
        ``tags`` and one key per relation field (``rel*``) shaped as LeanIX's
        ``{edges:[{node:{factSheet:{id,type}}}]}`` — exactly the shape the
        extractor's tolerant relation parser handles.

        ``since`` filters client-side on ``updatedAt`` (LeanIX has no reliable
        server-side modified-time filter); ``ids`` narrows to specific fact
        sheets (webhook-driven delta).
        """
        if type is None:
            out: list[dict[str, Any]] = []
            for fs_type in self._fact_sheet_types():
                out.extend(self.factsheets(fs_type, since=since, ids=ids))
            return out

        rel_fields = self._relation_fields(type)
        rel_fragment = "".join(
            " " + rf + "{edges{node{factSheet{id type}}}}" for rf in rel_fields
        )
        # Inline fragment carries the type-specific relation fields.
        query = (
            "query($first:Int!,$after:String,$filter:FilterInput){"
            "allFactSheets(first:$first,after:$after,filter:$filter){"
            "totalCount pageInfo{hasNextPage endCursor} "
            "edges{node{id name type updatedAt tags{name} "
            "...on " + type + "{" + rel_fragment + " } } } }"
        )
        filt: dict[str, Any] = {"facetFilters": [
            {"facetKey": "FactSheetTypes", "keys": [type]}
        ]}
        results: list[dict[str, Any]] = []
        cursor: str | None = None
        id_set = set(ids) if ids else None
        while True:
            data = self._gql(
                query, {"first": 500, "after": cursor, "filter": filt}
            )
            conn = (data or {}).get("allFactSheets") or {}
            for edge in conn.get("edges") or []:
                node = edge.get("node") if isinstance(edge, dict) else None
                if not isinstance(node, dict) or not node.get("id"):
                    continue
                if id_set is not None and node["id"] not in id_set:
                    continue
                if since and str(node.get("updatedAt") or "") <= since:
                    continue
                results.append(node)
            page = conn.get("pageInfo") or {}
            if not page.get("hasNextPage") or not page.get("endCursor"):
                break
            cursor = page["endCursor"]
        return results

    def fact_sheet_ids(self) -> set[str]:
        """All live fact sheet ids — for reconcile/tombstoning (Piece 3)."""
        data = self._gql(
            "query($first:Int!,$after:String){allFactSheets(first:$first,after:$after)"
            "{pageInfo{hasNextPage endCursor} edges{node{id}}}}",
            {"first": 1000, "after": None},
        )
        out: set[str] = set()
        conn = (data or {}).get("allFactSheets") or {}
        while True:
            for edge in conn.get("edges") or []:
                node = edge.get("node") if isinstance(edge, dict) else None
                if isinstance(node, dict) and node.get("id"):
                    out.add(node["id"])
            page = conn.get("pageInfo") or {}
            if not page.get("hasNextPage") or not page.get("endCursor"):
                break
            conn = (
                self._gql(
                    "query($first:Int!,$after:String){allFactSheets(first:$first,after:$after)"
                    "{pageInfo{hasNextPage endCursor} edges{node{id}}}}",
                    {"first": 1000, "after": page["endCursor"]},
                ).get("allFactSheets")
                or {}
            )
        return out

    # -- write-back (Piece 4) ------------------------------------------------

    def create_fact_sheet(self, fs_type: str, name: str) -> dict[str, Any]:
        """Create a fact sheet; returns ``{"id": ...}`` or ``{}`` on failure."""
        data = self._gql(
            "mutation($name:String!,$type:FactSheetType!){"
            "createFactSheet(input:{name:$name,type:$type}){factSheet{id}}}",
            {"name": name, "type": fs_type},
        )
        fs = ((data or {}).get("createFactSheet") or {}).get("factSheet") or {}
        return {"id": fs.get("id")} if fs.get("id") else {}

    def update_fact_sheet(
        self, fs_id: str, patches: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Apply JSON-patch-style ``patches`` (``{op,path,value}``) to a fact sheet."""
        data = self._gql(
            "mutation($id:ID!,$patches:[Patch]!){"
            "updateFactSheet(id:$id,patches:$patches){factSheet{id}}}",
            {"id": fs_id, "patches": patches},
        )
        fs = ((data or {}).get("updateFactSheet") or {}).get("factSheet") or {}
        return {"id": fs.get("id")} if fs.get("id") else {}

    def create_fact_sheet_relation(
        self, fs_id: str, rel_field: str, target_id: str
    ) -> dict[str, Any]:
        """Add a relation (``rel_field``) from ``fs_id`` to ``target_id`` (idempotent upsert)."""
        patch = {
            "op": "add",
            "path": f"/{rel_field}/new_1",
            "value": json.dumps({"factSheetId": target_id}),
        }
        return self.update_fact_sheet(fs_id, [patch])

    def add_tag(self, fs_id: str, tag_id: str) -> dict[str, Any]:
        """Attach an existing tag (by id) to a fact sheet."""
        patch = {
            "op": "add",
            "path": "/tags",
            "value": json.dumps([{"tagId": tag_id}]),
        }
        return self.update_fact_sheet(fs_id, [patch])


def get_leanix_client() -> LeanixEAClient | None:
    """Resolve a configured LeanIX client, or ``None`` when unconfigured.

    Reads ``LEANIX_URL`` and ``LEANIX_TOKEN`` (``LEANIX_API_TOKEN`` accepted as
    an alias) through ``config.setting`` — never a bare env read.
    """
    base_url = setting("LEANIX_URL", "")
    token = setting("LEANIX_TOKEN", "") or setting("LEANIX_API_TOKEN", "")
    if not base_url or not token:
        return None
    verify = bool(setting("LEANIX_VERIFY_SSL", False, cast=bool))
    return LeanixEAClient(base_url, token, verify_ssl=verify)


# ── Archi (best-effort) ─────────────────────────────────────────────────────


def get_archi_client() -> Any | None:
    """Best-effort Archi model client (``add_element`` surface), or ``None``.

    Only resolves when ``archimate_mcp`` is importable and ``ARCHI_MODEL_PATH``
    is configured; otherwise returns ``None`` so write-back simply skips Archi.
    """
    model_path = setting("ARCHI_MODEL_PATH", "")
    if not model_path:
        return None
    try:  # archimate-mcp is a sibling package, optional at runtime
        from archimate_mcp.api.api_client_archi import ArchiApi  # type: ignore

        return ArchiApi(model_path=model_path)
    except Exception as exc:  # noqa: BLE001 - optional dependency / config
        logger.debug("Archi client unavailable: %s", exc)
        return None
