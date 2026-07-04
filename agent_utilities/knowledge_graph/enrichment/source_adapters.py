"""In-process source-client adapters (CONCEPT:AU-KG.ingest.enterprise-source-extractor).

Bridge a connector package's raw API to the duck-typed surface each extractor
consumes, so ServiceNow and ERPNext flow through the same materialize/source_sync
path as Camunda/ARIS/Egeria. Tolerant: any transport failure degrades to ``[]``.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import inspect
import logging
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)


def run_sync(factory: Callable[[], Any]) -> Any:
    """Run an async ``factory()`` coroutine to completion from sync code.

    The sync-over-async bridge for connectors whose client is async (e.g.
    Microsoft Graph) but whose materialize/extractor path is synchronous. Works
    whether or not an event loop is already running: with no running loop it uses
    ``asyncio.run``; inside one (the MCP handler may itself be async) it runs the
    coroutine on a dedicated thread with its own loop so we never re-enter the
    current loop. The coroutine is *created inside* the executing thread so it is
    bound to the loop that runs it.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(factory())
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        return ex.submit(lambda: asyncio.run(factory())).result()


def _rows(resp: Any) -> list[dict]:
    """Normalize a ServiceNow/Frappe response to a list of dict rows."""
    body = resp
    if hasattr(resp, "json") and callable(resp.json):  # httpx.Response
        try:
            body = resp.json()
        except Exception:  # noqa: BLE001
            return []
    if isinstance(body, dict):
        payload = body.get("result")
        if payload is None:
            payload = body.get("data")
        if payload is None:
            payload = body.get("message")
        body = payload
    if isinstance(body, dict):  # single record
        return [body]
    if isinstance(body, list):
        return [r for r in body if isinstance(r, dict)]
    return []


class ServiceNowSourceClient:
    """Adapts the ``servicenow-api`` ``Api`` to the ServiceNow extractor surface.

    Exposes ``incidents()`` / ``changes()`` / ``cmdb_cis()`` plus the Technology
    Reference Model reads ``cmdb_models()`` / ``assets()`` (Phase 3), each a tolerant
    ``list[dict]``. CMDB classes are a configurable set (probe-and-skip), not hard
    requirements — instances vary by plugin.
    """

    CI_CLASSES = (
        "cmdb_ci_appl",
        "cmdb_ci_server",
        "cmdb_ci_database",
        "cmdb_ci_service",
    )
    MODEL_CLASSES = ("cmdb_model",)
    ASSET_CLASSES = ("alm_hardware", "alm_asset")

    def __init__(
        self,
        api: Any,
        *,
        ci_classes: tuple[str, ...] | None = None,
        model_classes: tuple[str, ...] | None = None,
        asset_classes: tuple[str, ...] | None = None,
    ) -> None:
        self._api = api
        self._ci_classes = ci_classes or self.CI_CLASSES
        self._model_classes = model_classes or self.MODEL_CLASSES
        self._asset_classes = asset_classes or self.ASSET_CLASSES

    def _call(self, name: str, **kwargs: Any) -> list[dict]:
        method = getattr(self._api, name, None)
        if not callable(method):
            return []
        try:
            return _rows(method(**kwargs))
        except Exception:  # noqa: BLE001 - tolerant transport
            logger.debug("servicenow %s failed", name, exc_info=True)
            return []

    def _by_classes(self, classes: tuple[str, ...], ci_class_key: str) -> list[dict]:
        out: list[dict] = []
        for cls in classes:
            for rec in self._call("get_cmdb_instances", className=cls):
                rec.setdefault(ci_class_key, cls)
                out.append(rec)
        return out

    def incidents(self) -> list[dict]:
        return self._call("get_incidents")

    def changes(self) -> list[dict]:
        return self._call("get_change_requests")

    def cmdb_cis(self) -> list[dict]:
        return self._by_classes(self._ci_classes, "ci_class")

    def cmdb_models(self) -> list[dict]:
        return self._by_classes(self._model_classes, "ci_class")

    def assets(self) -> list[dict]:
        return self._by_classes(self._asset_classes, "ci_class")


class ErpNextSourceClient:
    """Adapts the ``erpnext-agent`` resource ``Api`` to the ERPNext extractor surface.

    Exposes ``get_list(doctype)`` over the connector's ``list_documents`` (which
    returns ``{"data": [...]}``). Tolerant: missing doctype / transport → ``[]``.
    """

    def __init__(self, api: Any) -> None:
        self._api = api

    def get_list(self, doctype: str, fields: Any = None) -> list[dict]:
        lister = getattr(self._api, "list_documents", None) or getattr(
            self._api, "get_list", None
        )
        if not callable(lister):
            return []
        try:
            return _rows(lister(doctype))
        except Exception:  # noqa: BLE001
            logger.debug("erpnext list_documents(%s) failed", doctype, exc_info=True)
            return []

    def get_document(self, doctype: str, name: str) -> dict:
        getter = getattr(self._api, "get_document", None)
        if not callable(getter):
            return {}
        try:
            rows = _rows(getter(doctype, name))
            return rows[0] if rows else {}
        except Exception:  # noqa: BLE001
            return {}


def _graph_rows(resp: Any) -> list[dict]:
    """Normalize a Microsoft Graph response (``{"value": [...]}``) to dict rows."""
    body = resp
    if isinstance(body, dict):
        payload = body.get("value")
        if payload is None:
            payload = body.get("data", body.get("result"))
        if payload is not None:
            body = payload
        else:
            return [body]
    if isinstance(body, list):
        return [r for r in body if isinstance(r, dict)]
    return []


class MicrosoftGraphSourceClient:
    """Adapts the async ``microsoft-agent`` ``MicrosoftGraphApi`` to the M365
    extractor surface (CONCEPT:AU-KG.ingest.enterprise-source-extractor).

    The Graph client is fully ``async``; this wrapper bridges each call to sync
    via :func:`run_sync`, so M365 flows through the same synchronous
    materialize/source_sync path as every other connector. Exposes
    ``calendar_events()`` and ``users()`` as tolerant ``list[dict]`` (Graph's
    ``{"value": [...]}`` envelope unwrapped). Accepts a sync fake too (tests).
    """

    def __init__(self, api: Any) -> None:
        self._api = api

    def _call(self, name: str, *args: Any, **kwargs: Any) -> list[dict]:
        method = getattr(self._api, name, None)
        if not callable(method):
            return []
        try:
            if inspect.iscoroutinefunction(method):
                resp = run_sync(lambda: method(*args, **kwargs))
            else:
                resp = method(*args, **kwargs)
        except Exception:  # noqa: BLE001 - tolerant transport
            logger.debug("microsoft graph %s failed", name, exc_info=True)
            return []
        return _graph_rows(resp)

    def calendar_events(self) -> list[dict]:
        return self._call("list_calendar_events")

    def users(self) -> list[dict]:
        return self._call("list_users")
