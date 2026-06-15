"""Identity write-back sinks — Okta & Keycloak (CONCEPT:KG-2.9).

Backfeeds access-governance decisions: provision users, assign apps (inferred
access), and deprovision (retire) — fail-closed (``OKTA_ENABLE_WRITE`` /
``KEYCLOAK_ENABLE_WRITE``), dry-run-first. Resolves KG nodes to provider ids via
the ``externalToolId``/``domain`` federation key.
"""

from __future__ import annotations

import logging
from typing import Any

from ..core import WritebackContext, WritebackResult, register_sink

logger = logging.getLogger(__name__)


def _resolve_client(ops: dict[str, Any], module: str) -> Any | None:
    client = ops.get("client")
    if client is not None:
        return client
    try:
        mod = __import__(f"{module}.auth", fromlist=["get_client"])
        return mod.get_client()
    except Exception:  # noqa: BLE001 - connector absent / unconfigured
        logger.debug("%s write client unavailable", module, exc_info=True)
        return None


class OktaSink:
    """Write-back sink for Okta (provision / assign-app / deprovision)."""

    domain = "okta"
    enable_flag = "OKTA_ENABLE_WRITE"

    def run(
        self, ctx: WritebackContext, ops: dict[str, Any], *, dry_run: bool
    ) -> WritebackResult:
        result = WritebackResult(target=self.domain)
        client = _resolve_client(ops, "okta_agent")
        if client is None and not dry_run:
            result.skipped += 1
            return result
        resolve = ctx.resolver("okta")

        for c in ops.get("creations") or []:
            name = c.get("name")
            if not name:
                continue
            if dry_run:
                result.proposals.append({"op": "create_user", "name": name})
                continue
            try:
                client.create_user({"profile": {"login": name, "email": name}})  # type: ignore[union-attr]  # client None-checked above
                result.created += 1
            except Exception:  # noqa: BLE001
                logger.debug("okta create_user failed", exc_info=True)
                result.errors += 1

        # inferred access: user -ASSIGNED_APP-> app
        for edge in ops.get("inferences") or []:
            user = resolve(edge.get("source"))
            app = resolve(edge.get("target"))
            if not (user and app):
                result.skipped += 1
                continue
            if dry_run:
                result.proposals.append(
                    {"op": "assign_user_to_app", "user": user, "app": app}
                )
                continue
            try:
                client.assign_user_to_app(app, user)  # type: ignore[union-attr]  # client None-checked above
                result.relations_written += 1
            except Exception:  # noqa: BLE001
                logger.debug("okta assign_user_to_app failed", exc_info=True)
                result.errors += 1

        for item in ops.get("retirements") or []:
            uid = resolve(item.get("node"))
            if not uid:
                result.skipped += 1
                continue
            if dry_run:
                result.proposals.append({"op": "deactivate_user", "user": uid})
                continue
            try:
                client.deactivate_user(uid)  # type: ignore[union-attr]  # client None-checked above
                result.retired += 1
            except Exception:  # noqa: BLE001
                logger.debug("okta deactivate_user failed", exc_info=True)
                result.errors += 1

        return result


class KeycloakSink:
    """Write-back sink for Keycloak (provision users/clients in a realm)."""

    domain = "keycloak"
    enable_flag = "KEYCLOAK_ENABLE_WRITE"

    def run(
        self, ctx: WritebackContext, ops: dict[str, Any], *, dry_run: bool
    ) -> WritebackResult:
        result = WritebackResult(target=self.domain)
        client = _resolve_client(ops, "keycloak_agent")
        if client is None and not dry_run:
            result.skipped += 1
            return result
        realm = ops.get("realm", "master")

        for c in ops.get("creations") or []:
            name = c.get("name")
            ntype = (c.get("type") or "").lower()
            if not name:
                continue
            if dry_run:
                result.proposals.append(
                    {
                        "op": "create",
                        "type": ntype or "user",
                        "name": name,
                        "realm": realm,
                    }
                )
                continue
            try:
                if ntype == "application":
                    client.create_client(realm, {"clientId": name})  # type: ignore[union-attr]  # client None-checked above
                else:
                    client.create_user(realm, {"username": name, "enabled": True})  # type: ignore[union-attr]  # client None-checked above
                result.created += 1
            except Exception:  # noqa: BLE001
                logger.debug("keycloak create failed", exc_info=True)
                result.errors += 1

        return result


register_sink(OktaSink())
register_sink(KeycloakSink())
