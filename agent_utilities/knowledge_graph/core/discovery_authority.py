"""Transport-neutral authority values for private fleet discovery snapshots.

CONCEPT:AU-KG.ingest.fleet-catalog-relational-tables

The remote OAuth broker mints these non-secret values after resolving a live
grant. Lower catalog writers consume the exact type without importing the MCP
adapter that minted it. Provenance remains enforced by the multiplexer's
identity-bound side channel; this module is a value contract, not a registry or
an authority factory.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

__all__ = ["OAuthGrantBinding"]


@dataclass(frozen=True)
class OAuthGrantBinding:
    """Non-secret identity of one broker-resolved OAuth grant."""

    tenant_id: str
    principal_id: str
    provider_id: str
    resource_url: str
    audience: str
    granted_scopes: tuple[str, ...]
    key_version: int
    grant_revision: str

    @property
    def fingerprint(self) -> str:
        material = {
            "schema": "au.oauth-grant-binding.v1",
            "tenant": self.tenant_id,
            "principal": self.principal_id,
            "provider": self.provider_id,
            "resource": self.resource_url,
            "audience": self.audience,
            "scopes": list(self.granted_scopes),
            "key_version": self.key_version,
            "grant_revision": self.grant_revision,
        }
        encoded = json.dumps(
            material, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()
