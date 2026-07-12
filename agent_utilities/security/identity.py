#!/usr/bin/python
from __future__ import annotations

"""IdP-agnostic identity normalization (CONCEPT:AU-OS.identity.idp-agnostic-role-inheritance).

The platform authenticates callers against ANY OIDC issuer (Okta, Keycloak,
Auth0, Entra ID). Each provider spells "the caller's roles/groups" differently:
Keycloak puts realm roles under ``realm_access.roles`` and client roles under
``resource_access.<client>.roles``; Okta emits a first-class ``groups`` claim.
Left un-normalized, "inherit the caller's role as the agent's base authorization"
would work for one provider and silently not the other.

This module is the ONE place that reads a validated JWT's claims and produces a
provider-neutral :class:`NormalizedIdentity` — **Okta groups and Keycloak roles
are first-class and interchangeable**: both land in the same normalized
``roles``/``groups`` sets and derive the same base *capability* set. Every
consumer (the KG gateway's :func:`~agent_utilities.security.request_identity.actor_from_claims`,
graph-os delegation scope inheritance, and container-manager's k8s
impersonation) builds on this single normalizer rather than re-parsing claims.

Design rules:

* **Interchangeable, not branched.** We do not switch behavior on "which
  provider" — we read every standard claim location and union them. ``provider``
  is informational (audit/logging) only, never an access decision.
* **Order-preserving, de-duplicated.** Roles/groups keep first-seen order so
  downstream tuples are deterministic.
* **Dependency-light.** Pure stdlib so ``agents/*`` packages (container-manager)
  can import it without pulling the KG serving plane.
* **The base is a CEILING.** :func:`base_capabilities` yields the *maximum* an
  inheriting agent may hold; downstream policy (per-agent allow-lists, Eunomia,
  k8s RBAC on the impersonated identity) can only INTERSECT it, never escalate.
"""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

# Claim keys, grouped by the concept they carry. Read ALL of them (union) so a
# token from any provider resolves — never branch on provider.
_SUBJECT_KEYS = ("sub", "client_id", "azp")
_ROLE_KEYS = ("roles", "role")  # generic / Okta custom "roles" claim
_SCOPE_KEYS = ("scope", "scp")  # OAuth2 scopes → capabilities
_GROUP_KEYS = ("groups", "group")  # Okta default + Keycloak group-membership mapper
_TENANT_KEYS = ("tenant_id", "tenant", "org_id", "tid", "org")


def _as_str_list(value: Any) -> list[str]:
    """Coerce a claim value into a list of non-empty strings.

    Accepts a list/tuple, a space- or comma-separated string, or a scalar.
    Group entries that arrive as dicts (some IdPs emit ``{"name": ...}``) are
    reduced to their name/display value.
    """
    if value is None:
        return []
    if isinstance(value, str):
        # OAuth scope is space-separated; role/group CSVs are comma-separated.
        parts = value.replace(",", " ").split()
        return [p.strip() for p in parts if p.strip()]
    if isinstance(value, Mapping):
        name = value.get("name") or value.get("displayName") or value.get("id")
        return [str(name)] if name else []
    if isinstance(value, Iterable):
        out: list[str] = []
        for item in value:
            out.extend(_as_str_list(item))
        return out
    text = str(value).strip()
    return [text] if text else []


def _dedup(items: Iterable[str]) -> tuple[str, ...]:
    """Order-preserving de-duplication (first occurrence wins)."""
    return tuple(dict.fromkeys(i for i in items if i))


def _normalize_group_name(name: str) -> str:
    """Strip a Keycloak-style leading ``/`` group path to its leaf-agnostic form.

    Keycloak group mappers emit ``/engineering/kg-admin``; Okta emits
    ``kg-admin``. We keep the full path minus the leading slash so both a flat
    Okta group and a Keycloak group resolve to a stable, comparable token, and a
    caller can map either spelling in :data:`base_capabilities`'s group map.
    """
    return name.lstrip("/") if name.startswith("/") else name


def detect_provider(claims: Mapping[str, Any]) -> str:
    """Best-effort provider label (informational only — never an access gate)."""
    iss = str(claims.get("iss") or "").lower()
    if "okta" in iss:
        return "okta"
    if "/realms/" in iss or "realm_access" in claims:
        return "keycloak"
    if "login.microsoftonline" in iss or "sts.windows.net" in iss:
        return "entra"
    if "auth0" in iss:
        return "auth0"
    return "oidc"


@dataclass(frozen=True)
class NormalizedIdentity:
    """Provider-neutral view of a validated caller identity.

    ``roles`` and ``groups`` are kept distinct because they serve two consumers:
    graph-os capability scoping unions both into the base capability set, while
    k8s impersonation needs the raw ``groups`` (and ``subject``) to set
    ``Impersonate-User``/``Impersonate-Group``. ``provider`` is audit metadata.
    """

    subject: str = "jwt"
    roles: tuple[str, ...] = ()
    groups: tuple[str, ...] = ()
    tenant: str = ""
    email: str | None = None
    provider: str = "oidc"
    scopes: tuple[str, ...] = field(default_factory=tuple)


def normalize_identity(claims: Mapping[str, Any]) -> NormalizedIdentity:
    """Map a validated JWT's ``claims`` to a :class:`NormalizedIdentity`.

    Reads every standard role/group/scope/tenant location across Okta,
    Keycloak, and generic OIDC and unions them (interchangeable, not branched):

    * **roles** ← ``roles``/``role`` (generic/Okta) ∪ ``realm_access.roles``
      (Keycloak realm) ∪ every ``resource_access.<client>.roles`` (Keycloak
      client roles).
    * **scopes** ← ``scope``/``scp``.
    * **groups** ← ``groups``/``group`` (Okta default groups claim; Keycloak
      group-membership mapper), path-normalized.
    * **tenant** ← ``tenant_id``/``tenant``/``org_id``/``tid``/``org``.
    """
    subject = next(
        (str(claims[k]) for k in _SUBJECT_KEYS if claims.get(k)), "jwt"
    )

    roles: list[str] = []
    for key in _ROLE_KEYS:
        roles.extend(_as_str_list(claims.get(key)))
    realm = claims.get("realm_access")
    if isinstance(realm, Mapping):
        roles.extend(_as_str_list(realm.get("roles")))
    resource = claims.get("resource_access")
    if isinstance(resource, Mapping):
        for client_block in resource.values():
            if isinstance(client_block, Mapping):
                roles.extend(_as_str_list(client_block.get("roles")))

    scopes: list[str] = []
    for key in _SCOPE_KEYS:
        scopes.extend(_as_str_list(claims.get(key)))

    groups: list[str] = []
    for key in _GROUP_KEYS:
        groups.extend(_normalize_group_name(g) for g in _as_str_list(claims.get(key)))

    tenant = next((str(claims[k]) for k in _TENANT_KEYS if claims.get(k)), "")
    email = claims.get("email")

    return NormalizedIdentity(
        subject=subject,
        roles=_dedup(roles),
        groups=_dedup(groups),
        tenant=str(tenant),
        email=str(email) if email else None,
        provider=detect_provider(claims),
        scopes=_dedup(scopes),
    )


def base_capabilities(
    identity: NormalizedIdentity,
    group_map: Mapping[str, Iterable[str]] | None = None,
) -> tuple[str, ...]:
    """Derive the caller's **base** capability set — the inheritance ceiling.

    The base is ``roles ∪ scopes ∪ capabilities(groups)``. By default a group
    name IS a capability (identity mapping), so an Okta group ``kg-admin`` and a
    Keycloak role ``kg-admin`` yield the same capability with zero config. When
    ``group_map`` is supplied (deployment config, e.g. Okta's opaque group ids →
    capability names), a mapped group expands to its listed capabilities; an
    unmapped group falls back to its own name so nothing is silently dropped.

    This set is a CEILING: downstream policy (:mod:`apply_tool_scope`, Eunomia,
    the impersonated identity's own k8s RBAC) may only intersect it.
    """
    caps: list[str] = []
    caps.extend(identity.roles)
    caps.extend(identity.scopes)
    for group in identity.groups:
        if group_map and group in group_map:
            caps.extend(str(c) for c in group_map[group])
        else:
            caps.append(group)
    return _dedup(caps)


__all__ = [
    "NormalizedIdentity",
    "base_capabilities",
    "detect_provider",
    "normalize_identity",
]
