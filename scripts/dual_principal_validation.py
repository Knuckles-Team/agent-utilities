#!/usr/bin/env python3
"""Proves — or disproves — that a signed-in HUMAN principal and the graph-os
SERVICE credential see the SAME data across every REST/MCP surface.

WHY THIS EXISTS
----------------
The user's complaint: "the api authenticated with service credential is able
to see all the KG data and mcp tools and servers, but my agent-webui user
grant did not see that -- I only saw a small fraction of the data." Every
prior probe in this repo (``scripts/full_validation_harness.py``,
``scripts/delegation_probe.py``) authenticates with a SERVICE token only.
A service-token probe cannot detect a browser-path outage or a
human-vs-service authorization gap: a 401 without a token proves a guard
exists, nothing about what a signed-in human actually sees. This script
closes that gap by driving the REAL human sign-in flow -- the same
``OIDCBrowserSessionMiddleware`` a browser uses (``/auth/login`` ->
Keycloak -> ``/auth/callback``, sealed session cookie) -- and calling every
route as BOTH principals, diffing status + a structural fingerprint.

GROUND TRUTH THIS SCRIPT RELIES ON (verified live 2026-08-25)
---------------------------------------------------------------
* namespace ``platform``, deployment ``graph-os``: MCP on :8000
  (ingress ``graph-os.arpa``), agent-webui SPA + full ``/api/*`` REST
  gateway on :8080 (ingress ``au.arpa``).
* ``GET /health`` (bare, no ``/api`` prefix) falls through the SPA
  catch-all and returns 200 + index.html -- NEVER a health signal
  (BUG-PE-004). This script never probes it; ``/api/health`` is the real
  health check.
* ``/api/mining/*``, ``/api/graphlearn/*``, ``/api/mining/deep/*`` are dead
  routes that never register (BUG-PE-005) -- expected 404 for both
  principals, not in this script's route table, and MUST NOT be reported
  as a discrepancy if ever added.
* Keycloak realm ``homelab``. The ``agent-webui`` OIDC client (the one
  ``WEBUI_OIDC_CLIENT_ID`` names) is CONFIDENTIAL with
  ``directAccessGrantsEnabled=False`` -- Resource Owner Password
  Credentials against it returns ``400 unauthorized_client`` (confirmed
  live). This script tries ROPC first (cheap, and correct if a future
  client config ever enables it) and transparently falls back to the real
  authorization-code + PKCE dance through ``/auth/login`` with a cookie
  jar, exactly as the task brief anticipates.
* The internal TLS trust anchor is a cert-manager-issued CA
  (``cert-manager/homelab-arpa-ca-secret``), not present in a bare dev
  host's system trust store. This script fetches that CA via ``kubectl``
  and verifies TLS against it -- it never disables certificate
  verification.
* Keycloak admin credentials: k8s Secret ``keycloak-creds`` (namespace
  ``platform``), key ``KEYCLOAK_ADMIN_PASSWORD``, username ``admin``,
  against the ``master`` realm's ``admin-cli`` client (public, direct
  grants enabled -- the realm's own bootstrap admin, unrelated to the
  ``homelab`` realm's users).
* Service credential: k8s Secret ``graph-os-secrets`` holds
  ``OIDC_CLIENT_ID``/``OIDC_TOKEN_URL``/``OIDC_AUDIENCE`` (plain) and
  ``OIDC_CLIENT_SECRET`` (plain value) / ``OIDC_CLIENT_SECRET_REF``
  (``env://OIDC_CLIENT_SECRET``). This script acquires the service token
  via the REAL production path,
  ``agent_utilities.security.request_identity.acquire_process_identity_token``
  -- never hand-rolls a second token-acquisition code path.

WHAT "DISCREPANCY" MEANS -- AND WHAT IT DOES NOT (READ BEFORE TRUSTING GREEN)
------------------------------------------------------------------------------
Per explicit user directive: two principals failing IDENTICALLY on a route
(both 401, both 404, both 503, ...) is NOT parity and is NOT reported as a
match. It means the route was never actually exercised with a working
credential on either side, so nothing was proven about it. Every route
lands in exactly one bucket:

  MATCH        both principals got a 2xx with the same response shape and
               (when a count signal exists) counts within ``--tolerance``.
               This is the only bucket that is actual evidence of parity.
  DISCREPANCY  the principals disagree -- different status, different
               response shape, or counts differing by more than the
               tolerance. THIS is the human-vs-service gap the user
               reported; the whole point of this script is to surface it.
  UNTESTED     both principals got the SAME non-2xx status (a guard, a
               missing feature, a 503) -- reported separately, sorted with
               DISCREPANCY (worst-first), and called out by name in the
               summary so it is never silently read as "passing".

COST POSTURE (explicit user directive: this must be cheap enough for CI)
---------------------------------------------------------------------------
Every probed route uses a small ``limit``/``top_k`` where the endpoint
supports one, and status+shape+count is compared -- never a full result
set. HTTP calls run through a bounded thread pool (default 4 concurrent
requests, see ``--concurrency``) so ~30 routes x 2 principals is a few
seconds, not minutes, and the live engine is never hit with more than a
single-digit number of concurrent requests.

DESIGN (explicit user directive: one readable script, no plugin architecture)
---------------------------------------------------------------------------------
``ROUTES`` is a flat table of ``(method, path, body)``. One function calls
one route for one principal and returns a ``RouteResult``. One function
diffs two ``RouteResult``s into a verdict. The driver is a loop over the
table submitted to a bounded thread pool, then a sort, then a print. No
per-route strategy classes, no config DSL.

Token acquisition is reused, not reimplemented: the service leg calls
``agent_utilities.security.request_identity.acquire_process_identity_token``
verbatim (the same function ``scripts/_harness_browser_api.py``'s
``stage_api_gateway`` and ``scripts/full_validation_harness.py`` call), and
the MCP handshake for BOTH principals reuses
``scripts._harness_mcp.stage_mcp_handshake`` verbatim (the same
SSE-frame-matching handshake ``scripts/full_validation_harness.py`` stage 1
uses) rather than a second hand-rolled JSON-RPC client.

USAGE
-----
    # from a host with `kubectl` pointed at the cluster and network
    # reachability to au.arpa / keycloak.arpa / graph-os.arpa (verified live
    # 2026-08-25; this is simpler and just as correct as `kubectl exec`
    # because the human OIDC flow's redirect_uri is registered against the
    # `au.arpa` ingress hostname specifically -- an in-cluster short DNS name
    # would not byte-match it and Keycloak would reject the callback):
    python3 scripts/dual_principal_validation.py --json /tmp/dual-principal.json

    # fallback when this pod/host cannot reach the Keycloak admin API or
    # kubectl is unavailable: supply an already-minted human bearer token
    # (must carry an `email` claim to be classified HUMAN) and skip
    # provisioning + the browser dance entirely:
    python3 scripts/dual_principal_validation.py --user-token "$HUMAN_JWT"

EXIT-CODE CONTRACT
-------------------
0  every stage completed and every probed route that got a working
   credential on both sides agreed (MATCH or UNTESTED only).
1  a hard stage failure (provisioning, human login, or service token
   acquisition all failed with no usable fallback).
2  every stage completed but at least one route is a DISCREPANCY.
"""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import dataclasses
import http.cookiejar
import json
import os
import re
import secrets
import ssl
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import zlib
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# stdin-safe sibling-module import (same fallback chain as
# scripts/full_validation_harness.py's _import_harness_modules -- this file
# is designed to also be runnable via `python3 - < dual_principal_validation.py`)
# ---------------------------------------------------------------------------
_DEFAULT_MODULES_DIR = "/au"


def _import_stage_mcp_handshake(modules_dir: str | None):
    try:
        from scripts._harness_mcp import McpHandshakeError, stage_mcp_handshake

        return stage_mcp_handshake, McpHandshakeError
    except ModuleNotFoundError:
        pass
    # When invoked as a real file (`python3 scripts/dual_principal_validation.py`,
    # not piped via stdin), sys.path[0] is this file's own directory
    # (scripts/), not the repo root, so the plain import above fails even
    # though the repo root is right there -- derive it before falling back
    # to --modules-dir / the production /au default.
    inferred = None
    if __file__ != "<stdin>":
        try:
            inferred = str(Path(__file__).resolve().parents[1])
        except (OSError, IndexError):
            inferred = None
    for candidate in (c for c in (modules_dir, inferred, _DEFAULT_MODULES_DIR) if c):
        root = Path(candidate)
        if not (root / "scripts" / "_harness_mcp.py").is_file():
            continue
        if candidate not in sys.path:
            sys.path.insert(0, candidate)
        for name in ("scripts", "scripts._harness_mcp"):
            sys.modules.pop(name, None)
        try:
            from scripts._harness_mcp import McpHandshakeError, stage_mcp_handshake

            return stage_mcp_handshake, McpHandshakeError
        except ModuleNotFoundError:
            continue
    raise RuntimeError(
        "could not import scripts._harness_mcp -- pass --modules-dir "
        "<repo-root containing scripts/> (production default: "
        f"{_DEFAULT_MODULES_DIR!r})"
    )


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NAMESPACE = "platform"
REALM = "homelab"
PROBE_USERNAME = "kg-validation-probe"
PROBE_EMAIL = "kg-validation-probe@kg-validation.internal"
REAL_USER_EMAIL = "knucklessg1@gmail.com"

DEFAULT_BASE_URL = "https://au.arpa"
DEFAULT_MCP_URL = "https://graph-os.arpa/mcp"
DEFAULT_KEYCLOAK_URL = "https://keycloak.arpa"

DEFAULT_TIMEOUT_S = 15.0
DEFAULT_CONCURRENCY = 4
MAX_RESPONSE_BYTES = 200_000


def _chain(exc: BaseException) -> str:
    """Full causal chain -- same convention as full_validation_harness.py's
    ``_chain``. Several functions this script calls into
    (``acquire_process_identity_token``, the OAuth2 client-credentials
    provider's ``_mint``) deliberately do ``raise ... from None``, discarding
    the real cause -- see the BUGS FOUND note in this lane's final report.
    This still unwraps whatever chain IS available."""
    out: list[str] = []
    seen: set[int] = set()
    cur: BaseException | None = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        out.append(f"{type(cur).__name__}: {cur}")
        cur = cur.__cause__ or cur.__context__
    return "\n      caused by ".join(out)


def _emit(stage: str, ok: bool, detail: str = "") -> None:
    mark = "PASS" if ok else "FAIL"
    print(f"  {mark:4s} {stage:20s} {detail}"[:2000], flush=True)


# ---------------------------------------------------------------------------
# Secrets -- k8s Secret access via `kubectl`. Values are returned to the
# caller and NEVER printed/logged anywhere in this file; only key NAMES and
# derived non-secret facts (client ids, role names, counts) are printed.
# ---------------------------------------------------------------------------
def kube_secret(name: str, key: str, *, namespace: str = NAMESPACE) -> str:
    # kubectl's jsonpath treats an unescaped "." as a field separator, so a
    # literal dot in the key itself (e.g. "ca.crt") must be backslash-escaped
    # or it silently resolves to nothing -- kubectl still exits 0 with empty
    # stdout, which looks exactly like "the key doesn't exist" instead of a
    # syntax problem. Escape every dot in the key, not just known ones.
    escaped_key = key.replace(".", r"\.")
    out = subprocess.run(
        [
            "kubectl",
            "-n",
            namespace,
            "get",
            "secret",
            name,
            "-o",
            f"jsonpath={{.data.{escaped_key}}}",
        ],
        capture_output=True,
        text=True,
        timeout=20,
    )
    if out.returncode != 0 or not out.stdout.strip():
        raise RuntimeError(
            f"could not read k8s Secret {namespace}/{name} key {key!r} "
            f"(kubectl exit {out.returncode}: {out.stderr.strip()[:300]})"
        )
    return base64.b64decode(out.stdout).decode("utf-8")


def fetch_ca_bundle(dest: Path) -> Path:
    """Fetch the homelab's internal CA (cert-manager/homelab-arpa-ca-secret)
    and write it to ``dest``. Every TLS connection this script makes is
    verified against this file -- certificate verification is never
    disabled. Raises if the CA cannot be fetched (see ``--insecure-tls`` for
    the documented, explicit escape hatch)."""
    ca_pem = kube_secret("homelab-arpa-ca-secret", "ca.crt", namespace="cert-manager")
    dest.write_text(ca_pem)
    return dest


# ---------------------------------------------------------------------------
# HTTP plumbing (stdlib-only, mirrors scripts/_harness_browser_api.py's
# conventions: a redirect-rejecting opener for API calls so the real status
# is seen, capped response reads, status=0 for a connection failure).
# ---------------------------------------------------------------------------
class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001
        return None


def tls_context(cafile: str | None, insecure: bool) -> ssl.SSLContext:
    if insecure:
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        return ctx
    return ssl.create_default_context(cafile=cafile)


def build_opener(
    ctx: ssl.SSLContext,
    *,
    cookiejar: http.cookiejar.CookieJar | None = None,
    follow_redirects: bool = False,
) -> urllib.request.OpenerDirector:
    """``follow_redirects=False`` (the default) surfaces a 30x AS a 30x, the
    same convention ``scripts/_harness_browser_api.py`` uses for API-status
    grading. The real browser OIDC dance (``browser_login``) needs the
    opposite -- it must actually traverse GET /auth/login -> Keycloak ->
    login page -- and passes ``follow_redirects=True`` explicitly."""
    handlers: list[Any] = [urllib.request.HTTPSHandler(context=ctx)]
    if not follow_redirects:
        handlers.append(_NoRedirect())
    if cookiejar is not None:
        handlers.append(urllib.request.HTTPCookieProcessor(cookiejar))
    return urllib.request.build_opener(*handlers)


@dataclasses.dataclass(frozen=True)
class RouteResult:
    status: int  # 0 = connection/transport failure
    elapsed_s: float
    fingerprint: dict[str, Any]
    truncated: bool
    note: str = ""


def call_route(
    opener: urllib.request.OpenerDirector,
    base_url: str,
    method: str,
    path: str,
    body: dict[str, Any] | None,
    *,
    headers: dict[str, str],
    timeout: float,
) -> RouteResult:
    url = base_url.rstrip("/") + path
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req_headers = {
        "Accept": "application/json",
        "User-Agent": "dual-principal-validation/1",
    }
    req_headers.update(headers)
    if data is not None:
        req_headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, method=method, headers=req_headers)
    t0 = time.monotonic()
    try:
        resp = opener.open(req, timeout=timeout)
        status = int(getattr(resp, "status", None) or resp.getcode() or 0)
        content_type = resp.headers.get("Content-Type", "") or ""
        raw = resp.read(MAX_RESPONSE_BYTES + 1)
        resp.close()
    except urllib.error.HTTPError as exc:
        status = int(exc.code)
        content_type = exc.headers.get("Content-Type", "") if exc.headers else ""
        raw = exc.read(MAX_RESPONSE_BYTES + 1)
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        elapsed = time.monotonic() - t0
        return RouteResult(
            0, elapsed, {"kind": "transport-error"}, False, note=str(exc)[:200]
        )
    elapsed = time.monotonic() - t0
    truncated = len(raw) > MAX_RESPONSE_BYTES
    if truncated:
        raw = raw[:MAX_RESPONSE_BYTES]
    return RouteResult(
        status, elapsed, response_fingerprint(content_type, raw, truncated), truncated
    )


def response_fingerprint(
    content_type: str, raw: bytes, truncated: bool
) -> dict[str, Any]:
    """A stable, cheap structural summary of a response body: never the full
    body, just enough to detect "same shape, same rough size" vs not."""
    if not raw:
        return {"kind": "empty"}
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return {"kind": "non-json", "len": len(raw), "content_type": content_type}
    if isinstance(data, list):
        count = None if truncated else len(data)
        sample = data[0] if data and isinstance(data[0], dict) else {}
        return {"kind": "list", "count": count, "keys": sorted(sample.keys())[:15]}
    if isinstance(data, dict):
        count = None
        if not truncated:
            for key in ("count", "total", "total_count"):
                value = data.get(key)
                if isinstance(value, int) and not isinstance(value, bool):
                    count = value
                    break
            if count is None:
                for key in (
                    "items",
                    "results",
                    "tools",
                    "skills",
                    "nodes",
                    "edges",
                    "relationships",
                    "servers",
                    "sets",
                    "bindings",
                    "mcp_tools",
                    "builtin_tools",
                ):
                    value = data.get(key)
                    if isinstance(value, list):
                        count = len(value)
                        break
        return {"kind": "dict", "keys": sorted(data.keys())[:20], "count": count}
    return {"kind": type(data).__name__}


# ---------------------------------------------------------------------------
# Keycloak admin client -- thin, self-refreshing (the admin-cli token is
# short-lived, ~60s, so every call refreshes if within a safety margin).
# ---------------------------------------------------------------------------
class KeycloakAdmin:
    def __init__(self, base_url: str, ctx: ssl.SSLContext) -> None:
        self._base_url = base_url.rstrip("/")
        self._opener = build_opener(ctx)
        self._token: str | None = None
        self._expires_at = 0.0

    def _admin_token(self) -> str:
        if self._token and time.monotonic() < self._expires_at - 10:
            return self._token
        password = kube_secret("keycloak-creds", "KEYCLOAK_ADMIN_PASSWORD")
        data = urllib.parse.urlencode(
            {
                "client_id": "admin-cli",
                "username": "admin",
                "password": password,
                "grant_type": "password",
            }
        ).encode()
        req = urllib.request.Request(
            f"{self._base_url}/realms/master/protocol/openid-connect/token",
            data=data,
            method="POST",
        )
        resp = self._opener.open(req, timeout=DEFAULT_TIMEOUT_S)
        payload = json.loads(resp.read())
        self._token = payload["access_token"]
        self._expires_at = time.monotonic() + float(payload.get("expires_in", 60))
        return self._token

    def _call(self, method: str, path: str, body: Any = None) -> tuple[int, Any]:
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(
            f"{self._base_url}/admin/realms/{REALM}{path}",
            data=data,
            method=method,
            headers={
                "Authorization": f"Bearer {self._admin_token()}",
                "Content-Type": "application/json",
            },
        )
        try:
            resp = self._opener.open(req, timeout=DEFAULT_TIMEOUT_S)
            raw = resp.read()
            return resp.status, (json.loads(raw) if raw else None)
        except urllib.error.HTTPError as exc:
            raw = exc.read()
            return exc.code, (json.loads(raw) if raw else None)

    def get(self, path: str) -> Any:
        status, body = self._call("GET", path)
        if status >= 400:
            raise RuntimeError(f"Keycloak admin GET {path} -> {status}: {body}")
        return body

    def post(self, path: str, body: Any = None) -> tuple[int, Any]:
        return self._call("POST", path, body)

    def put(self, path: str, body: Any = None) -> tuple[int, Any]:
        return self._call("PUT", path, body)


def discover_real_user_grants(
    kc: KeycloakAdmin, email: str
) -> tuple[list[str], list[str]]:
    """Find the real human user by email and return their DIRECT (not
    composite-expanded) realm role names and group paths -- the exact set to
    replicate onto the synthetic probe user."""
    users = kc.get(
        "/users?" + urllib.parse.urlencode({"email": email, "exact": "true"})
    )
    if not users:
        raise RuntimeError(
            f"no Keycloak user found in realm {REALM!r} with email {email!r}"
        )
    uid = users[0]["id"]
    roles = kc.get(f"/users/{uid}/role-mappings/realm")
    groups = kc.get(f"/users/{uid}/groups")
    role_names = [r["name"] for r in roles]
    group_paths = [g["path"] for g in groups]
    return role_names, group_paths


def provision_probe_user(
    kc: KeycloakAdmin, role_names: list[str], group_paths: list[str]
) -> tuple[str, str]:
    """Idempotently create/update ``kg-validation-probe`` with the given
    direct realm roles and group memberships, and a freshly generated
    password (reset every run so the credential is always known to THIS
    run, never persisted to disk). Returns (username, password)."""
    users = kc.get(
        "/users?"
        + urllib.parse.urlencode({"username": PROBE_USERNAME, "exact": "true"})
    )
    if users:
        uid = users[0]["id"]
    else:
        status, _body = kc.post(
            "/users",
            {
                "username": PROBE_USERNAME,
                "email": PROBE_EMAIL,
                "emailVerified": True,
                "enabled": True,
                "firstName": "KG",
                "lastName": "ValidationProbe",
            },
        )
        if status not in (200, 201):
            raise RuntimeError(f"could not create probe user (status {status})")
        users = kc.get(
            "/users?"
            + urllib.parse.urlencode({"username": PROBE_USERNAME, "exact": "true"})
        )
        uid = users[0]["id"]

    password = secrets.token_urlsafe(24)
    status, _ = kc.put(
        f"/users/{uid}/reset-password",
        {"type": "password", "value": password, "temporary": False},
    )
    if status not in (200, 204):
        raise RuntimeError(f"could not set probe user password (status {status})")

    for role_name in role_names:
        role_repr = kc.get(f"/roles/{urllib.parse.quote(role_name, safe='')}")
        status, _ = kc.post(f"/users/{uid}/role-mappings/realm", [role_repr])
        if status not in (200, 204):
            raise RuntimeError(f"could not assign role {role_name!r} (status {status})")

    all_groups = kc.get("/groups")
    by_path = {g["path"]: g["id"] for g in all_groups}
    for group_path in group_paths:
        gid = by_path.get(group_path)
        if gid is None:
            raise RuntimeError(f"real user's group {group_path!r} not found in realm")
        status, _ = kc.put(f"/users/{uid}/groups/{gid}")
        if status not in (200, 204):
            raise RuntimeError(f"could not join group {group_path!r} (status {status})")

    return PROBE_USERNAME, password


# ---------------------------------------------------------------------------
# Human OIDC acquisition: ROPC first (cheap; correct if ever enabled),
# transparently falling back to the real authorization-code + PKCE dance
# through /auth/login -> Keycloak -> /auth/callback with a cookie jar. This
# is the leg that genuinely exercises the browser path, per the module
# docstring's core requirement.
# ---------------------------------------------------------------------------
def attempt_ropc(
    keycloak_url: str,
    client_id: str,
    client_secret: str,
    username: str,
    password: str,
    ctx: ssl.SSLContext,
) -> str | None:
    opener = build_opener(ctx)
    data = urllib.parse.urlencode(
        {
            "grant_type": "password",
            "client_id": client_id,
            "client_secret": client_secret,
            "username": username,
            "password": password,
            "scope": "openid profile email",
        }
    ).encode()
    req = urllib.request.Request(
        f"{keycloak_url}/realms/{REALM}/protocol/openid-connect/token",
        data=data,
        method="POST",
    )
    try:
        resp = opener.open(req, timeout=DEFAULT_TIMEOUT_S)
        return json.loads(resp.read())["access_token"]
    except urllib.error.HTTPError:
        return None
    except (urllib.error.URLError, OSError, TimeoutError):
        return None


class BrowserLoginError(RuntimeError):
    pass


def browser_login(
    base_url: str, username: str, password: str, ctx: ssl.SSLContext
) -> http.cookiejar.CookieJar:
    """Drive the REAL authorization-code + PKCE flow a human's browser takes:
    GET /auth/login (redirects through Keycloak to its login page), POST the
    credentials to the login form's own action URL, and land back on
    /auth/callback which seals the session into au_session* cookies. Returns
    the populated cookiejar -- every subsequent request made through an
    opener built with this cookiejar IS the human's authenticated browser
    session, exercising OIDCBrowserSessionMiddleware exactly as a real
    sign-in does."""
    cj = http.cookiejar.CookieJar()
    opener = build_opener(ctx, cookiejar=cj, follow_redirects=True)

    req = urllib.request.Request(
        base_url.rstrip("/") + "/auth/login?next=/",
        headers={"User-Agent": "dual-principal-validation/1"},
    )
    try:
        resp = opener.open(req, timeout=DEFAULT_TIMEOUT_S)
        html = resp.read().decode("utf-8", "replace")
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        raise BrowserLoginError(f"GET /auth/login failed: {exc}") from exc

    match = re.search(r'<form[^>]+id="kc-form-login"[^>]+action="([^"]+)"', html)
    if not match:
        raise BrowserLoginError(
            "Keycloak login form not found in the /auth/login redirect target -- "
            "the OIDC flow, the client's standardFlowEnabled, or the realm's login "
            "theme may be misconfigured. Response head: " + html[:300]
        )
    action = match.group(1).replace("&amp;", "&")

    form_data = urllib.parse.urlencode(
        {"username": username, "password": password}
    ).encode()
    req2 = urllib.request.Request(
        action,
        data=form_data,
        method="POST",
        headers={
            "Content-Type": "application/x-www-form-urlencoded",
            "User-Agent": "dual-principal-validation/1",
        },
    )
    try:
        resp2 = opener.open(req2, timeout=DEFAULT_TIMEOUT_S)
        final_url = resp2.geturl()
        resp2.read()
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        raise BrowserLoginError(f"login form POST failed: {exc}") from exc

    if not any(cookie.name == "au_session0" for cookie in cj):
        raise BrowserLoginError(
            "browser flow completed (landed on "
            f"{final_url!r}) but no au_session* cookie was set -- sign-in did "
            "not actually complete; check the probe user's credentials/roles "
            "and the WEBUI_OIDC_* configuration"
        )
    return cj


def unseal_session(
    cj: http.cookiejar.CookieJar, base_url: str, session_key: str
) -> dict[str, Any]:
    """Decrypt the au_session* cookie chunks with WEBUI_SESSION_KEY (fetched
    from the graph-os-webui-oidc Secret, never printed) to recover the raw
    access_token -- needed for the MCP leg, which has no cookie/browser
    concept and is bearer-only."""
    from cryptography.fernet import Fernet

    host = urllib.parse.urlsplit(base_url).hostname or ""
    chunks: list[str] = []
    index = 0
    while True:
        found = None
        for cookie in cj:
            if (
                cookie.name == f"au_session{index}"
                and (cookie.domain or "").lstrip(".") == host
            ):
                found = cookie.value
                break
        if found is None:
            break
        chunks.append(found)
        index += 1
    if not chunks:
        raise BrowserLoginError("no au_session* cookies present to unseal")
    fernet = Fernet(session_key.encode("ascii"))
    opened = fernet.decrypt("".join(chunks).encode("ascii"))
    return json.loads(zlib.decompress(opened))


def decode_jwt_claims(token: str) -> dict[str, Any]:
    """Decode a JWT's payload WITHOUT verifying its signature -- this script
    already only ever uses tokens it just received directly from Keycloak
    over a verified TLS connection, so this is purely for inspecting claims
    (confirming `email` is present), never a trust decision."""
    parts = token.split(".")
    if len(parts) != 3:
        raise ValueError("not a JWT")
    padded = parts[1] + "=" * (-len(parts[1]) % 4)
    return json.loads(base64.urlsafe_b64decode(padded))


def acquire_human_token(
    base_url: str,
    keycloak_url: str,
    username: str,
    password: str,
    ctx: ssl.SSLContext,
    cafile: str | None,
) -> tuple[str, dict[str, Any], http.cookiejar.CookieJar | None]:
    """Returns (access_token, claims, cookiejar). ``cookiejar`` is the real
    browser session (None only if ROPC happened to succeed, in which case
    there is no browser session to reuse and the REST leg falls back to
    Authorization: Bearer for the human too -- logged loudly, since it means
    the browser path itself was NOT exercised)."""
    client_id = kube_secret("graph-os-webui-oidc", "WEBUI_OIDC_CLIENT_ID")
    client_secret = kube_secret("graph-os-webui-oidc", "WEBUI_OIDC_CLIENT_SECRET")
    session_key = kube_secret("graph-os-webui-oidc", "WEBUI_SESSION_KEY")

    token = attempt_ropc(
        keycloak_url, client_id, client_secret, username, password, ctx
    )
    if token is not None:
        print(
            "  NOTE  ROPC succeeded directly against the webui OIDC client -- "
            "the browser cookie/session path was NOT exercised for the REST "
            "leg; MCP still uses this real bearer either way.",
            flush=True,
        )
        return token, decode_jwt_claims(token), None

    cj = browser_login(base_url, username, password, ctx)
    session = unseal_session(cj, base_url, session_key)
    access_token = str(session.get("access_token") or "")
    if not access_token:
        raise BrowserLoginError("sealed session cookie carried no access_token")
    return access_token, decode_jwt_claims(access_token), cj


def acquire_service_token(cafile: str | None) -> str:
    """Acquire the graph process (service) identity via the REAL production
    path -- reused verbatim, not reimplemented. Sets OIDC_CLIENT_SECRET from
    the graph-os-secrets Secret (never printed) and SSL_CERT_FILE so the
    fleet's own TLS trust-profile machinery verifies against the fetched
    homelab CA instead of failing closed on this host's default trust
    store. ``cafile`` is None only under ``--insecure-tls``, in which case
    this leaves the ambient trust store untouched (this inner client's TLS
    verification is not otherwise disableable from here)."""
    if cafile:
        os.environ.setdefault("SSL_CERT_FILE", cafile)
        os.environ.setdefault("REQUESTS_CA_BUNDLE", cafile)
    os.environ["OIDC_CLIENT_SECRET"] = kube_secret(
        "graph-os-secrets", "OIDC_CLIENT_SECRET"
    )

    from agent_utilities.core.config import AgentConfig
    from agent_utilities.security.request_identity import acquire_process_identity_token

    cfg = AgentConfig(
        **{
            "KG_IDENTITY_OAUTH2": {
                "token_url": kube_secret("graph-os-secrets", "OIDC_TOKEN_URL"),
                "client_id": kube_secret("graph-os-secrets", "OIDC_CLIENT_ID"),
                "client_secret": "env://OIDC_CLIENT_SECRET",
                "audience": kube_secret("graph-os-secrets", "OIDC_AUDIENCE"),
            }
        }
    )
    return acquire_process_identity_token(cfg)


# ---------------------------------------------------------------------------
# The route table -- flat, per the explicit "no plugin architecture"
# directive. (method, path, json body-or-None). Paths use small limits
# wherever the endpoint accepts one. Excludes /api/mining|/api/graphlearn
# (BUG-PE-005 -- dead routes, expected 404 for everyone, not a discrepancy).
# ---------------------------------------------------------------------------
ROUTES: list[tuple[str, str, dict[str, Any] | None]] = [
    # -- core --
    ("GET", "/api/health", None),
    ("GET", "/api/tools", None),
    ("GET", "/api/sessions", None),
    ("GET", "/api/goals", None),
    # -- enhanced graph --
    ("GET", "/api/enhanced/graph/stats", None),
    ("GET", "/api/enhanced/graph/nodes", None),
    ("GET", "/api/enhanced/graph/relationships", None),
    ("GET", "/api/enhanced/graph/search?query=agent&top_k=5", None),
    ("GET", "/api/enhanced/graph/impact/agent_utilities", None),
    (
        "POST",
        "/api/enhanced/graph/query",
        {"query": "MATCH (n) RETURN count(n) AS c LIMIT 1"},
    ),
    # -- enhanced ontology family --
    ("GET", "/api/enhanced/ontology/object-types", None),
    ("GET", "/api/enhanced/ontology/property-types", None),
    ("GET", "/api/enhanced/ontology/interfaces", None),
    ("GET", "/api/enhanced/ontology/actions", None),
    ("GET", "/api/enhanced/ontology/object-set/list", None),
    # -- enhanced tools/skills/mcp --
    ("GET", "/api/enhanced/tools", None),
    ("GET", "/api/enhanced/skills", None),
    ("GET", "/api/enhanced/mcp/server-schema", None),
    # -- durable registry catalog --
    ("GET", "/api/registry/servers?limit=5", None),
    ("GET", "/api/registry/discoveries?limit=5", None),
    ("GET", "/api/registry/tools?limit=5", None),
    ("GET", "/api/registry/prompts?limit=5", None),
    ("GET", "/api/registry/resources?limit=5", None),
    ("GET", "/api/registry/skills?limit=5", None),
    # -- gateway-native ontology + objects --
    ("GET", "/api/ontology/value-types", None),
    ("GET", "/api/ontology/property-types", None),
    ("GET", "/api/ontology/interfaces", None),
    ("GET", "/api/ontology/schema-summary", None),
    ("GET", "/api/ontology/catalogue", None),
    ("GET", "/api/objects/kg-validation-probe-nonexistent", None),
    # -- fleet --
    ("GET", "/api/fleet/health", None),
    ("GET", "/api/fleet/topology", None),
    # -- sparql --
    (
        "GET",
        "/api/sparql?"
        + urllib.parse.urlencode({"query": "SELECT * WHERE { ?s ?p ?o } LIMIT 1"}),
        None,
    ),
]


# ---------------------------------------------------------------------------
# Sweep + diff
# ---------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class PrincipalCtx:
    name: str
    opener: urllib.request.OpenerDirector
    headers: dict[str, str]


VERDICT_DISCREPANCY = "DISCREPANCY"
VERDICT_UNTESTED = "UNTESTED"
VERDICT_MATCH = "MATCH"


@dataclasses.dataclass(frozen=True)
class RouteVerdict:
    method: str
    path: str
    service: RouteResult
    human: RouteResult
    verdict: str
    reason: str


def classify(
    service: RouteResult, human: RouteResult, tolerance: int
) -> tuple[str, str]:
    if service.status == 0 or human.status == 0:
        if service.status == human.status == 0:
            return VERDICT_UNTESTED, "both sides had a transport failure"
        return (
            VERDICT_DISCREPANCY,
            "one side had a transport failure, the other did not",
        )
    if service.status != human.status:
        return (
            VERDICT_DISCREPANCY,
            f"status mismatch: service={service.status} human={human.status}",
        )
    if not (200 <= service.status < 300):
        return (
            VERDICT_UNTESTED,
            f"both sides returned {service.status} -- not proof of parity",
        )
    if service.fingerprint.get("kind") != human.fingerprint.get("kind"):
        return (
            VERDICT_DISCREPANCY,
            f"response shape mismatch: service={service.fingerprint.get('kind')} "
            f"human={human.fingerprint.get('kind')}",
        )
    s_count, h_count = service.fingerprint.get("count"), human.fingerprint.get("count")
    if (
        isinstance(s_count, int)
        and isinstance(h_count, int)
        and abs(s_count - h_count) > tolerance
    ):
        return (
            VERDICT_DISCREPANCY,
            f"item count mismatch: service={s_count} human={h_count}",
        )
    return VERDICT_MATCH, "agree"


def run_route_sweep(
    routes: list[tuple[str, str, dict[str, Any] | None]],
    principals: list[PrincipalCtx],
    base_url: str,
    *,
    concurrency: int,
    timeout: float,
    tolerance: int,
) -> list[RouteVerdict]:
    tasks = [
        (method, path, body, principal)
        for method, path, body in routes
        for principal in principals
    ]
    results: dict[tuple[str, str, str], RouteResult] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {
            pool.submit(
                call_route,
                principal.opener,
                base_url,
                method,
                path,
                body,
                headers=principal.headers,
                timeout=timeout,
            ): (method, path, principal.name)
            for method, path, body, principal in tasks
        }
        for future in concurrent.futures.as_completed(futures):
            key = futures[future]
            results[key] = future.result()

    verdicts: list[RouteVerdict] = []
    by_principal = {p.name: p for p in principals}
    assert "service" in by_principal and "human" in by_principal
    for method, path, _body in routes:
        service_result = results[(method, path, "service")]
        human_result = results[(method, path, "human")]
        verdict, reason = classify(service_result, human_result, tolerance)
        verdicts.append(
            RouteVerdict(method, path, service_result, human_result, verdict, reason)
        )

    order = {VERDICT_DISCREPANCY: 0, VERDICT_UNTESTED: 1, VERDICT_MATCH: 2}
    verdicts.sort(key=lambda v: (order[v.verdict], v.path))
    return verdicts


def render_table(verdicts: list[RouteVerdict]) -> str:
    lines = [
        f"  {'VERDICT':11s} {'METHOD':6s} {'PATH':55s} {'SERVICE':8s} {'HUMAN':8s}  REASON"
    ]
    for v in verdicts:
        lines.append(
            f"  {v.verdict:11s} {v.method:6s} {v.path[:55]:55s} "
            f"{v.service.status:<8d} {v.human.status:<8d}  {v.reason}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# MCP sweep -- reuses scripts._harness_mcp.stage_mcp_handshake verbatim.
# ---------------------------------------------------------------------------
def mcp_sweep(
    stage_mcp_handshake: Any,
    mcp_handshake_error: type[Exception],
    mcp_url: str,
    tokens: dict[str, str],
    timeout: float,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, token in tokens.items():
        try:
            result = stage_mcp_handshake(mcp_url, timeout=timeout, auth_token=token)
            out[name] = {"ok": True, "tool_count": result.tool_count, "error": None}
        except mcp_handshake_error as exc:
            out[name] = {"ok": False, "tool_count": 0, "error": str(exc)}
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--base-url", default=DEFAULT_BASE_URL, help="agent-webui REST/browser base URL"
    )
    p.add_argument(
        "--mcp-url",
        default=DEFAULT_MCP_URL,
        help="graph-os MCP streamable-http endpoint",
    )
    p.add_argument("--keycloak-url", default=DEFAULT_KEYCLOAK_URL)
    p.add_argument(
        "--user-token",
        default="",
        help="skip provisioning + the browser OIDC dance entirely; use this "
        "already-minted human bearer token instead (loudly logged as a "
        "fallback -- must carry an 'email' claim to be classified HUMAN)",
    )
    p.add_argument(
        "--tolerance",
        type=int,
        default=0,
        help="allowed item-count delta before it's a discrepancy",
    )
    p.add_argument(
        "--concurrency",
        type=int,
        default=DEFAULT_CONCURRENCY,
        help="bounded HTTP thread-pool size",
    )
    p.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S)
    p.add_argument(
        "--insecure-tls",
        action="store_true",
        help="skip TLS verification instead of fetching the homelab CA (NOT the default -- verification is never silently dropped)",
    )
    p.add_argument(
        "--modules-dir",
        default="",
        help="repo-root containing scripts/ (stdin-piped fallback)",
    )
    p.add_argument(
        "--json", default="", help="write machine-readable results to this path"
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    stage_mcp_handshake, McpHandshakeError = _import_stage_mcp_handshake(
        args.modules_dir or None
    )

    reasons: list[str] = []
    json_out: dict[str, Any] = {}

    print(
        f"\ndual-principal validation -- base_url={args.base_url!r} mcp_url={args.mcp_url!r}\n"
    )

    # --- CA bundle ---
    scratch_dir = (
        Path(os.environ.get("XDG_RUNTIME_DIR", "/tmp")) / "dual-principal-validation"
    )
    scratch_dir.mkdir(parents=True, exist_ok=True)
    ca_path = scratch_dir / "homelab-ca.crt"
    if args.insecure_tls:
        ctx = tls_context(None, insecure=True)
        _emit("tls_trust", True, "--insecure-tls: certificate verification DISABLED")
    else:
        try:
            fetch_ca_bundle(ca_path)
            ctx = tls_context(str(ca_path), insecure=False)
            _emit(
                "tls_trust", True, f"verifying against fetched homelab CA ({ca_path})"
            )
        except Exception as exc:
            _emit("tls_trust", False, _chain(exc))
            reasons.append(f"tls_trust: {_chain(exc)}")
            print(f"\n  SUMMARY: NOT GREEN -- {len(reasons)} reason(s): {reasons}\n")
            return 1

    # --- service token (real production path) ---
    try:
        service_token = acquire_service_token(
            None if args.insecure_tls else str(ca_path)
        )
        service_claims = decode_jwt_claims(service_token)
        _emit(
            "service_token",
            True,
            f"actor={service_claims.get('azp') or service_claims.get('client_id')!r} "
            f"email_present={'email' in service_claims} "
            f"roles={service_claims.get('realm_access', {}).get('roles')}",
        )
    except Exception as exc:
        _emit("service_token", False, _chain(exc))
        reasons.append(f"service_token: {_chain(exc)}")
        print(f"\n  SUMMARY: NOT GREEN -- {len(reasons)} reason(s): {reasons}\n")
        return 1

    # --- human principal: provision + real OIDC login, or --user-token fallback ---
    human_cookiejar: http.cookiejar.CookieJar | None = None
    if args.user_token:
        print(
            "  NOTE  --user-token supplied: provisioning + the browser OIDC "
            "dance are SKIPPED. The REST leg uses this bearer directly, which "
            "does NOT exercise OIDCBrowserSessionMiddleware/the cookie path "
            "-- only JWT-claim classification is proven this way.",
            flush=True,
        )
        human_token = args.user_token
        try:
            human_claims = decode_jwt_claims(human_token)
        except Exception as exc:
            _emit("human_token", False, f"--user-token does not decode as a JWT: {exc}")
            reasons.append("human_token: --user-token invalid")
            print(f"\n  SUMMARY: NOT GREEN -- {len(reasons)} reason(s): {reasons}\n")
            return 1
        _emit(
            "human_token",
            True,
            f"email_present={'email' in human_claims} (from --user-token)",
        )
    else:
        try:
            kc = KeycloakAdmin(args.keycloak_url, ctx)
            role_names, group_paths = discover_real_user_grants(kc, REAL_USER_EMAIL)
            print(
                f"  NOTE  real user's direct realm roles copied: {role_names}",
                flush=True,
            )
            print(
                f"  NOTE  real user's group memberships copied: {group_paths}",
                flush=True,
            )
            username, password = provision_probe_user(kc, role_names, group_paths)
            _emit(
                "provisioning",
                True,
                f"user={username!r} roles={role_names} groups={group_paths}",
            )
            json_out["copied_roles"] = role_names
            json_out["copied_groups"] = group_paths
        except Exception as exc:
            _emit("provisioning", False, _chain(exc))
            reasons.append(f"provisioning: {_chain(exc)}")
            print(
                "\n  Could not provision a synthetic human principal. Re-run with "
                "--user-token <already-minted human JWT> to proceed without "
                "provisioning.\n"
            )
            print(f"\n  SUMMARY: NOT GREEN -- {len(reasons)} reason(s): {reasons}\n")
            return 1

        try:
            human_token, human_claims, human_cookiejar = acquire_human_token(
                args.base_url, args.keycloak_url, username, password, ctx, str(ca_path)
            )
            _emit(
                "human_login",
                True,
                f"email_present={'email' in human_claims} "
                f"(HUMAN classification: {'email' in human_claims}) "
                f"roles={human_claims.get('realm_access', {}).get('roles')} "
                f"browser_session={'yes' if human_cookiejar is not None else 'no (ROPC path)'}",
            )
            if "email" not in human_claims:
                reasons.append(
                    "human_login: token has no 'email' claim -- would classify as AUTOMATED_SERVICE, not HUMAN"
                )
        except Exception as exc:
            _emit("human_login", False, _chain(exc))
            reasons.append(f"human_login: {_chain(exc)}")
            print(f"\n  SUMMARY: NOT GREEN -- {len(reasons)} reason(s): {reasons}\n")
            return 1

    # --- REST route sweep ---
    service_ctx = PrincipalCtx(
        "service", build_opener(ctx), {"Authorization": f"Bearer {service_token}"}
    )
    if human_cookiejar is not None:
        human_ctx = PrincipalCtx(
            "human", build_opener(ctx, cookiejar=human_cookiejar), {}
        )
    else:
        human_ctx = PrincipalCtx(
            "human", build_opener(ctx), {"Authorization": f"Bearer {human_token}"}
        )

    verdicts = run_route_sweep(
        ROUTES,
        [service_ctx, human_ctx],
        args.base_url,
        concurrency=args.concurrency,
        timeout=args.timeout,
        tolerance=args.tolerance,
    )
    discrepancies = [v for v in verdicts if v.verdict == VERDICT_DISCREPANCY]
    untested = [v for v in verdicts if v.verdict == VERDICT_UNTESTED]
    matches = [v for v in verdicts if v.verdict == VERDICT_MATCH]
    print("\n" + render_table(verdicts) + "\n")
    _emit(
        "route_sweep",
        not discrepancies,
        f"{len(matches)} match, {len(untested)} untested (both sides failed identically -- "
        f"NOT proof of parity), {len(discrepancies)} discrepancy",
    )
    if discrepancies:
        reasons.append(
            f"route_sweep: {len(discrepancies)} discrepant route(s) (see table above)"
        )
    json_out["route_verdicts"] = [
        {
            "method": v.method,
            "path": v.path,
            "verdict": v.verdict,
            "reason": v.reason,
            "service_status": v.service.status,
            "human_status": v.human.status,
            "service_fingerprint": v.service.fingerprint,
            "human_fingerprint": v.human.fingerprint,
        }
        for v in verdicts
    ]

    # --- MCP sweep ---
    mcp_results = mcp_sweep(
        stage_mcp_handshake,
        McpHandshakeError,
        args.mcp_url,
        {"service": service_token, "human": human_token},
        args.timeout,
    )
    mcp_ok = all(r["ok"] and r["tool_count"] > 0 for r in mcp_results.values())
    mcp_discrepancy = mcp_results["service"]["tool_count"] != mcp_results["human"][
        "tool_count"
    ] or (mcp_results["service"]["ok"] != mcp_results["human"]["ok"])
    _emit(
        "mcp_sweep",
        mcp_ok and not mcp_discrepancy,
        f"service={mcp_results['service']} human={mcp_results['human']}",
    )
    if not mcp_ok:
        reasons.append(
            f"mcp_sweep: a principal's handshake failed or reported zero tools: {mcp_results}"
        )
    elif mcp_discrepancy:
        reasons.append(
            f"mcp_sweep: tool count/status mismatch between principals: {mcp_results}"
        )
    json_out["mcp_results"] = mcp_results

    if args.json:
        Path(args.json).write_text(json.dumps(json_out, indent=2, default=str))
        print(f"\n  JSON results written to {args.json}")

    if reasons:
        print(f"\n  SUMMARY: NOT GREEN -- {len(reasons)} reason(s):")
        for reason in reasons:
            print(f"    - {reason}")
        print()
        return 2
    print(
        f"\n  SUMMARY: all clear -- {len(matches)} match, {len(untested)} untested, 0 discrepancy\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
