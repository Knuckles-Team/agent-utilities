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
    LoadConnector,
    PollConnector,
    SourceDocument,
    default_external_access,
)
from ..http_safety import require_safe_source_url, safe_get_text
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

    def _resolve_registry_identity(
        self,
        preset: str | None,
        catalog_url: str,
        registry_name: str,
        media_types: list[str] | None,
        private_hosts: list[str],
    ) -> tuple[str, str, str, list[str] | None]:
        """Resolve + validate ``(catalog_url, registry_name, publisher_host, media_types)``."""
        cfg = dict(ARD_PRESETS.get(preset, {})) if preset else {}
        catalog_url = catalog_url or str(cfg.get("catalog_url", ""))
        if not catalog_url:
            raise ValueError(
                "ArdRegistryConnector requires 'catalog_url' or a known 'preset'"
            )
        catalog_url = _catalog_url(catalog_url)
        require_safe_source_url(
            catalog_url,
            allowed_private_hosts=private_hosts,
            resolve_dns=False,
        )
        host = urlparse(catalog_url).hostname or ""
        registry_name = (
            registry_name or str(cfg.get("registry_name", "")) or host or "ard"
        )
        resolved_media_types = media_types or cfg.get("media_types") or None
        return catalog_url, registry_name, host, resolved_media_types

    def _make_safe_fetch(
        self,
        private_hosts: list[str],
        redirect_hosts: list[str],
        max_response_bytes: int,
    ) -> FetchFn:
        """Build the default fetch function, closed over its safety limits."""

        def _safe_fetch(url: str) -> str:
            return safe_get_text(
                url,
                timeout=_FETCH_TIMEOUT_S,
                headers={
                    "User-Agent": "agent-utilities-ard/1.0",
                    "Accept": "application/json",
                },
                max_bytes=max_response_bytes,
                allowed_private_hosts=private_hosts,
                allowed_redirect_hosts=redirect_hosts,
            )

        return _safe_fetch

    def configure(
        self,
        *,
        preset: str | None = None,
        catalog_url: str = "",
        registry_name: str = "",
        media_types: list[str] | None = None,
        verify: bool = True,
        fetch_fn: FetchFn | None = None,
        allowed_private_hosts: list[str] | None = None,
        allowed_redirect_hosts: list[str] | None = None,
        max_response_bytes: int = 10 * 1024 * 1024,
        **_: object,
    ) -> None:
        private_hosts = list(allowed_private_hosts or [])
        self.catalog_url, self.registry_name, self.publisher_host, self.media_types = (
            self._resolve_registry_identity(
                preset, catalog_url, registry_name, media_types, private_hosts
            )
        )
        self.verify = bool(verify)
        self.require_signature = bool(setting("ARD_REQUIRE_SIGNATURE", default=False))
        self.external_access = default_external_access()
        if fetch_fn is not None:
            self._fetch = fetch_fn
        else:
            self._fetch = self._make_safe_fetch(
                private_hosts, list(allowed_redirect_hosts or []), max_response_bytes
            )
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

    def _domain_verified(self, entry: dict[str, Any]) -> bool:
        """Domain-anchored identity: the entry's publisher domain must match the
        host we fetched the catalog from (an entry claiming another domain is
        rejected), allowing a true DNS subdomain/parent relationship. Plain
        suffix matching would accept an attacker-controlled lookalike domain.
        """
        domain = (
            str((entry.get("publisher") or {}).get("domain", ""))
            .strip()
            .lower()
            .rstrip(".")
        )
        publisher_host = self.publisher_host.lower().rstrip(".")
        if not domain or not publisher_host or domain == publisher_host:
            return True
        return publisher_host.endswith(f".{domain}") or domain.endswith(
            f".{publisher_host}"
        )

    def _signature_verified(self, entry: dict[str, Any], publisher_key: str) -> bool:
        """Ed25519 signature check, or the unsigned-allowed policy when unsigned."""
        signature = entry.get("signature")
        if not signature:
            # Unsigned: allowed unless a strict posture is configured.
            return not self.require_signature
        if not publisher_key:
            return False
        # Sign/verify is over the entry sans its own signature field.
        unsigned = {k: v for k, v in entry.items() if k != "signature"}
        return ard_signing.verify_datapoint(unsigned, str(signature), publisher_key)

    def _accept(self, entry: dict[str, Any], publisher_key: str) -> bool:
        """Verification gate: domain-anchoring + Ed25519 signature (fail-closed)."""
        if not self.verify:
            return True
        if not self._domain_verified(entry):
            self.verify_failures += 1
            return False
        if not self._signature_verified(entry, publisher_key):
            self.verify_failures += 1
            return False
        return True

    def _entry_document(
        self, entry: dict[str, Any], eid: Any, media: str, manifest_publisher: Any
    ) -> SourceDocument:
        """Build the SourceDocument for an entry that already passed filtering."""
        return SourceDocument(
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
                "publisher": manifest_publisher or {},
                "verified": bool(entry.get("signature")) and self.verify,
            },
            external_access=self.external_access.model_copy(deep=True),
        )

    def _entry_to_document(
        self,
        entry: Any,
        publisher_key: str,
        allow: set[str] | None,
        manifest_publisher: Any,
    ) -> SourceDocument | None:
        """Build a SourceDocument from one raw ARD entry, or None to skip it."""
        if not isinstance(entry, dict):
            return None
        eid = entry.get("id")
        media = entry.get("type") or ""
        if not eid:
            return None
        if allow is not None and media not in allow:
            return None
        if not self._accept(entry, publisher_key):
            return None
        return self._entry_document(entry, eid, media, manifest_publisher)

    def _entries(self) -> list[SourceDocument]:
        manifest = self._fetch_manifest()
        resources = manifest.get("resources")
        if not isinstance(resources, list):
            return []
        publisher_key = str(manifest.get("publisherKey") or "")
        allow = set(self.media_types) if self.media_types else None
        manifest_publisher = manifest.get("publisher")
        out: list[SourceDocument] = []
        for entry in resources:
            doc = self._entry_to_document(
                entry, publisher_key, allow, manifest_publisher
            )
            if doc is not None:
                out.append(doc)
        return out

    # -- LoadConnector -----------------------------------------------------

    def load(self) -> Iterator[SourceDocument]:
        yield from self._entries()

    # -- PollConnector -----------------------------------------------------

    @staticmethod
    def _fresh_entries(
        all_docs: list[SourceDocument], prior_ids: set[str], wm: str | None
    ) -> list[SourceDocument]:
        """Docs not already seen, and not older than the watermark."""
        return [
            d
            for d in all_docs
            if d.id not in prior_ids
            and (wm is None or not d.updated_at or d.updated_at >= wm)
        ]

    @staticmethod
    def _next_watermark(all_docs: list[SourceDocument], wm: str | None) -> str | None:
        """The new high-watermark across this batch's dated docs and the prior one."""
        dates = [d.updated_at for d in all_docs if d.updated_at]
        if wm:
            dates.append(wm)
        return max(dates) if dates else wm

    def poll(self, checkpoint: ConnectorCheckpoint | None = None) -> CheckpointedBatch:
        """Emit entries not already seen (seen-id belt + optional updatedAt watermark)."""
        prior_ids = set(checkpoint.seen_ids) if checkpoint else set()
        wm = checkpoint.watermark if checkpoint else None
        all_docs = self._entries()
        fresh = self._fresh_entries(all_docs, prior_ids, wm)
        new_wm = self._next_watermark(all_docs, wm)
        new_ids = sorted(prior_ids | {d.id for d in fresh})[-_SEEN_CAP:]
        cp = ConnectorCheckpoint(has_more=False, watermark=new_wm, seen_ids=new_ids)
        return CheckpointedBatch(documents=fresh, checkpoint=cp)
