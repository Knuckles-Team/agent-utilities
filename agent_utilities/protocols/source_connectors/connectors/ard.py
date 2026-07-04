from __future__ import annotations

"""Agentic Resource Discovery (ARD) registry connector — consume side.

CONCEPT:AU-ECO.connector.ingest-external-ard-registry — ingest an external ARD registry (Hugging Face and others) as
discoverable resources. ARD is a public discovery *protocol* (a static signed
``ai-catalog.json`` at a well-known URL plus a ``POST /search`` API), not an MCP-wrapped
system — so it is a native protocol connector in the same family as ``rss``/``web``/
``reader`` (zero-infra, no service to deploy), not an ``mcp_tool`` preset.

The connector fetches a registry's static catalog and yields one
:class:`SourceDocument` per resource, preserving the raw ARD entry in
``metadata["record"]`` so the ``_sync_ard`` handler (KG-2.188) can materialize each as a
typed ``:MCPServer`` / ``:A2AAgentCard`` / ``:ServiceCapability`` node linked to its
``:ResourceRegistry``. Ed25519 publisher verification (OS-5.60) is enforced on inbound:
an entry whose signature fails is dropped; an unsigned entry is dropped only when
``ARD_REQUIRE_SIGNATURE`` is set (the draft spec means many registries won't sign yet).
The incremental ``poll`` uses a seen-id belt + optional ``updatedAt`` watermark, exactly
like the ``rss`` connector.
"""

import json
from collections.abc import Callable, Iterator
from typing import Any
from urllib.parse import urlparse

from ....core.config import setting
from ....security import ard_signing
from ..base import (
    CheckpointedBatch,
    ConnectorCheckpoint,
    ExternalAccess,
    LoadConnector,
    PollConnector,
    SourceDocument,
)
from ..registry import register_source

FetchFn = Callable[[str], str]
_SEEN_CAP = 5000
_FETCH_TIMEOUT_S = 20.0

#: Built-in registry presets (server + endpoints), mirroring ``MCP_TOOL_PRESETS``.
#: A new registry is a preset, not new transport code.
ARD_PRESETS: dict[str, dict[str, Any]] = {
    "huggingface": {
        "registry_name": "huggingface",
        "catalog_url": "https://huggingface.co",
        "search_url": "https://huggingface-hf-discover.hf.space/search",
        "media_types": ["application/ai-skill", "application/mcp-server+json"],
    },
}


def _default_fetch(url: str) -> str:
    """Fetch a URL with lazy ``httpx`` (clear error if unavailable)."""
    try:
        import httpx
    except ImportError as exc:  # pragma: no cover - environment without httpx
        raise RuntimeError(
            "ArdRegistryConnector needs 'httpx' to fetch a catalog. "
            "Install it, or pass a fetch_fn for offline use."
        ) from exc
    resp = httpx.get(
        url,
        timeout=_FETCH_TIMEOUT_S,
        follow_redirects=True,
        headers={
            "User-Agent": "agent-utilities-ard/1.0",
            "Accept": "application/json",
        },
    )
    resp.raise_for_status()
    return resp.text


def _catalog_url(base: str) -> str:
    """Resolve the well-known manifest URL from a base (or pass a full ``.json`` through)."""
    base = (base or "").strip().rstrip("/")
    if base.endswith(".json"):
        return base
    return f"{base}/.well-known/ai-catalog.json"


@register_source("ard")
class ArdRegistryConnector(LoadConnector, PollConnector):
    """Fetch + parse an external ARD registry into resource documents (CONCEPT:AU-ECO.connector.ingest-external-ard-registry).

    Config:
        preset: A key in :data:`ARD_PRESETS` (e.g. ``huggingface``) seeding the rest.
        catalog_url: Registry base or full ``ai-catalog.json`` URL (required if no preset).
        registry_name: Provenance label for the registry (default: the catalog host).
        media_types: Optional allow-list of ARD media types to ingest.
        verify: Verify Ed25519 signatures (default True; failures are dropped).
        fetch_fn: Optional ``(url) -> json_text`` injectable for offline tests.
    """

    provider = "ARD Registry"

    def configure(
        self,
        *,
        preset: str | None = None,
        catalog_url: str = "",
        registry_name: str = "",
        media_types: list[str] | None = None,
        verify: bool = True,
        fetch_fn: FetchFn | None = None,
        **_: object,
    ) -> None:
        cfg = dict(ARD_PRESETS.get(preset, {})) if preset else {}
        catalog_url = catalog_url or str(cfg.get("catalog_url", ""))
        if not catalog_url:
            raise ValueError(
                "ArdRegistryConnector requires 'catalog_url' or a known 'preset'"
            )
        self.catalog_url = _catalog_url(catalog_url)
        host = urlparse(self.catalog_url).hostname or ""
        self.registry_name = (
            registry_name or str(cfg.get("registry_name", "")) or host or "ard"
        )
        self.publisher_host = host
        self.media_types = media_types or cfg.get("media_types") or None
        self.verify = bool(verify)
        self.require_signature = bool(setting("ARD_REQUIRE_SIGNATURE", default=False))
        self._fetch: FetchFn = fetch_fn or _default_fetch
        #: Set by :meth:`_entries` so the sync handler can surface verification drops.
        self.verify_failures = 0

    def health_check(self) -> bool:
        return bool(self.catalog_url)

    # -- fetch + parse -----------------------------------------------------

    def _fetch_manifest(self) -> dict[str, Any]:
        """Fetch + parse the registry's ``ai-catalog.json`` (a dead registry → ``{}``)."""
        try:
            raw = self._fetch(self.catalog_url)
            data = json.loads(raw)
            return data if isinstance(data, dict) else {}
        except Exception:  # noqa: BLE001 — a dead/invalid registry must not abort a sweep
            return {}

    def _accept(self, entry: dict[str, Any], publisher_key: str) -> bool:
        """Verification gate: domain-anchoring + Ed25519 signature (fail-closed)."""
        if not self.verify:
            return True
        # Domain-anchored identity: the entry's publisher domain must match the host
        # we fetched the catalog from (an entry claiming another domain is rejected).
        domain = str((entry.get("publisher") or {}).get("domain", "")).strip()
        if domain and self.publisher_host and domain != self.publisher_host:
            # Allow subdomain/host matches but reject a mismatched apex domain.
            if not (
                self.publisher_host.endswith(domain)
                or domain.endswith(self.publisher_host)
            ):
                self.verify_failures += 1
                return False
        signature = entry.get("signature")
        if not signature:
            # Unsigned: allowed unless a strict posture is configured.
            if self.require_signature:
                self.verify_failures += 1
                return False
            return True
        if not publisher_key:
            self.verify_failures += 1
            return False
        # Sign/verify is over the entry sans its own signature field.
        unsigned = {k: v for k, v in entry.items() if k != "signature"}
        if ard_signing.verify_datapoint(unsigned, str(signature), publisher_key):
            return True
        self.verify_failures += 1
        return False

    def _entries(self) -> list[SourceDocument]:
        manifest = self._fetch_manifest()
        resources = manifest.get("resources")
        if not isinstance(resources, list):
            return []
        publisher_key = str(manifest.get("publisherKey") or "")
        allow = set(self.media_types) if self.media_types else None
        out: list[SourceDocument] = []
        for entry in resources:
            if not isinstance(entry, dict):
                continue
            eid = entry.get("id")
            media = entry.get("type") or ""
            if not eid:
                continue
            if allow is not None and media not in allow:
                continue
            if not self._accept(entry, publisher_key):
                continue
            out.append(
                SourceDocument(
                    id=str(eid),
                    source_uri=self.catalog_url,
                    title=str(entry.get("name") or eid)[:300],
                    text=str(entry.get("description") or ""),
                    doc_type="ard_resource",
                    updated_at=entry.get("updatedAt"),
                    metadata={
                        "record": entry,
                        "ard_media_type": media,
                        "registry": self.registry_name,
                        "publisher": manifest.get("publisher") or {},
                        "verified": bool(entry.get("signature")) and self.verify,
                    },
                    external_access=ExternalAccess.public(),
                )
            )
        return out

    # -- LoadConnector -----------------------------------------------------

    def load(self) -> Iterator[SourceDocument]:
        yield from self._entries()

    # -- PollConnector -----------------------------------------------------

    def poll(self, checkpoint: ConnectorCheckpoint | None = None) -> CheckpointedBatch:
        """Emit entries not already seen (seen-id belt + optional updatedAt watermark)."""
        prior_ids = set(checkpoint.seen_ids) if checkpoint else set()
        wm = checkpoint.watermark if checkpoint else None
        all_docs = self._entries()
        fresh = [
            d
            for d in all_docs
            if d.id not in prior_ids
            and (wm is None or not d.updated_at or d.updated_at >= wm)
        ]
        dates = [d.updated_at for d in all_docs if d.updated_at]
        if wm:
            dates.append(wm)
        new_wm = max(dates) if dates else wm
        new_ids = sorted(prior_ids | {d.id for d in fresh})[-_SEEN_CAP:]
        cp = ConnectorCheckpoint(has_more=False, watermark=new_wm, seen_ids=new_ids)
        return CheckpointedBatch(documents=fresh, checkpoint=cp)
