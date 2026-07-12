#!/usr/bin/python
from __future__ import annotations

"""Identity-scoped resource auto-load (CONCEPT:AU-OS.identity.identity-scoped-resource-autoload).

Generalizes IdP role inheritance from "what may I do" to "what may I *reach*":
a caller's normalized capabilities (from
:func:`agent_utilities.security.identity.base_capabilities` — Okta groups and
Keycloak roles, interchangeable) determine WHICH backend resources an MCP server
auto-loads and connects to by default. So an operator connects to exactly the
kube contexts / SSH hosts / repositories / database connections their Okta or
Keycloak groups grant — no per-server, per-environment manual configuration.

One shared resolver lives here; every ``agents/*`` server inherits it and only
supplies (a) its resource *namespace* (``"k8s"``, ``"ssh"``, ``"gitlab"``…) and
(b) the list of resources it could offer. The resolver returns the entitled
subset to auto-load.

**Capability grammar** (all interchangeable, provider-neutral):

* ``"<namespace>:<resource>"`` — entitles that one resource (e.g. ``"k8s:prod"``).
* ``"<namespace>:*"`` / ``"<namespace>:admin"`` — entitles ALL resources in the
  namespace.
* ``"admin"`` / ``"system"`` (or a configured super-capability) — entitles all.
* a bare capability equal to a resource name — the **zero-config** case: an Okta
  group or Keycloak role literally named after the context (``"prod"``) entitles
  the ``"prod"`` resource with no namespacing needed.

**Fail-closed:** with no matching capability the entitled set is empty. A server
decides what an empty set means (deny, or fall back to a public/default
resource) — the resolver never invents access.
"""

from collections.abc import Iterable

# Capabilities that grant every resource in every namespace (the identity ceiling
# already caps this; these are just the "see everything I'm allowed to" tokens).
DEFAULT_SUPER_CAPS: frozenset[str] = frozenset({"admin", "system"})

# Wildcard suffixes that, when namespaced, grant every resource in that namespace.
_NAMESPACE_WILDCARDS: frozenset[str] = frozenset({"*", "admin", "all"})


def _split_cap(cap: str) -> tuple[str, str] | None:
    """Split ``"ns:resource"`` → ``(ns, resource)``; ``None`` if not namespaced."""
    if ":" not in cap:
        return None
    ns, _, resource = cap.partition(":")
    return ns.strip(), resource.strip()


def grants_all_in_namespace(
    capabilities: Iterable[str],
    namespace: str,
    *,
    super_caps: Iterable[str] = DEFAULT_SUPER_CAPS,
) -> bool:
    """True if ``capabilities`` grant EVERY resource in ``namespace``.

    Triggered by a super-capability (``admin``/``system``) or a namespaced
    wildcard (``"<namespace>:*"``/``":admin"``/``":all"``).
    """
    supers = set(super_caps)
    for cap in capabilities:
        if cap in supers:
            return True
        parts = _split_cap(cap)
        if parts and parts[0] == namespace and parts[1] in _NAMESPACE_WILDCARDS:
            return True
    return False


def entitled_resources(
    capabilities: Iterable[str],
    namespace: str,
    available: Iterable[str] | None = None,
    *,
    super_caps: Iterable[str] = DEFAULT_SUPER_CAPS,
) -> tuple[str, ...]:
    """Resolve which resources in ``namespace`` the caller may auto-load.

    Args:
        capabilities: the caller's base capabilities (from
            :func:`~agent_utilities.security.identity.base_capabilities`).
        namespace: the server's resource namespace, e.g. ``"k8s"``/``"ssh"``.
        available: the resources the server could offer (contexts/hosts/…). When
            given, the result is the entitled subset that actually exists (order
            preserved from ``available``); a wildcard/super grant expands to ALL
            of ``available``. When ``None``, the result is exactly the resources
            named by the caller's namespaced/bare capabilities (the server has no
            catalog to intersect against yet).
        super_caps: capabilities that grant everything (default admin/system).

    Returns:
        An order-stable tuple of entitled resource identifiers. Empty when
        nothing matches (fail-closed).
    """
    caps = list(capabilities)

    if available is not None:
        available_list = list(dict.fromkeys(available))
        if grants_all_in_namespace(caps, namespace, super_caps=super_caps):
            return tuple(available_list)
        # A resource is entitled if the caller holds either the namespaced
        # capability ``ns:<resource>`` or a bare capability equal to its name.
        cap_set = set(caps)
        namespaced = {f"{namespace}:{r}" for r in available_list}
        return tuple(
            r
            for r in available_list
            if f"{namespace}:{r}" in cap_set or r in cap_set
        )

    # No catalog: derive the named resources directly from the capabilities.
    named: list[str] = []
    for cap in caps:
        parts = _split_cap(cap)
        if parts and parts[0] == namespace and parts[1] not in _NAMESPACE_WILDCARDS:
            named.append(parts[1])
    return tuple(dict.fromkeys(named))


def is_entitled(
    capabilities: Iterable[str],
    namespace: str,
    resource: str,
    *,
    super_caps: Iterable[str] = DEFAULT_SUPER_CAPS,
) -> bool:
    """True if the caller may reach ``resource`` in ``namespace``."""
    caps = list(capabilities)
    if grants_all_in_namespace(caps, namespace, super_caps=super_caps):
        return True
    cap_set = set(caps)
    return f"{namespace}:{resource}" in cap_set or resource in cap_set


def identity_scoped_resources(
    namespace: str,
    available: Iterable[str],
    *,
    actor: object | None = None,
    super_caps: Iterable[str] = DEFAULT_SUPER_CAPS,
) -> tuple[str, ...]:
    """The one call an ``agents/*`` MCP server makes to auto-load by identity.

    Resolves the ambient authenticated caller (``current_actor()`` unless an
    explicit ``actor`` is passed) to the subset of ``available`` resources their
    Okta/Keycloak groups entitle. ``ActorContext.roles`` is already the base
    capability set (roles ∪ scopes ∪ group-derived capabilities), so this is a
    thin wrapper over :func:`entitled_resources`.

    Back-compat / fail-safe: the ambient ``SYSTEM_ACTOR`` (unauthenticated local
    callers) holds the ``admin``/``system`` super-capability, so it is entitled
    to ALL ``available`` resources — a server behaves exactly as today until a
    real authenticated identity with specific groups scopes it down. This is the
    native, default-on path: no flag, no opt-in, and every server that lists its
    resources through this call inherits identity-scoped auto-loading for free.
    """
    from .brain_context import current_actor

    ctx = actor if actor is not None else current_actor()
    capabilities = getattr(ctx, "roles", ()) or ()
    return entitled_resources(
        capabilities, namespace, available, super_caps=super_caps
    )


__all__ = [
    "DEFAULT_SUPER_CAPS",
    "entitled_resources",
    "grants_all_in_namespace",
    "identity_scoped_resources",
    "is_entitled",
]
