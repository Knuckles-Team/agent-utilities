#!/usr/bin/env python3
"""LANE H-http — the browser-path and API-gateway stage module for the committed
validation harness (``scripts/full_validation_harness.py``, owned by a parallel lane).

WHY THIS MODULE EXISTS
-----------------------
A prior, uncommitted probe reported the deployment "green" twice while every real
human sign-in was 503-ing. It proved that only by presenting a service token: a
``401`` returned to a request that carries *no* credential proves nothing except
that some guard exists somewhere — it says nothing about whether the app is even
serving traffic, and it says nothing about the path an actual signed-in user takes.
That probe was never committed and now survives only in a dead scratchpad.

This module is the committed replacement for its HTTP legs. It provides three
independent stage callables:

  * :func:`stage_browser`     — the unauthenticated, user-visible surface. Checked
    FIRST because it is the authoritative user-visible signal: if the app itself is
    down, nothing downstream matters.
  * :func:`stage_api_gateway` — the service-token API surface, probed WITH a real
    bearer acquired the same way graph-os itself acquires one.
  * :func:`stage_admission`   — the tenant-admission RPC a signed-in principal
    actually triggers on every authenticated request. This is a **surrogate**, not
    proof of human sign-in — see its docstring's HONESTY REQUIREMENT below and in
    the emitted :class:`StageReport`.

CONVENTIONS
-----------
Follows ``scripts/delegation_probe.py`` house style: each stage is a plain
callable that returns a structured result and raises on failure — never
``sys.exit`` itself. The failing exception (a :class:`HarnessStageError`
subclass) always carries the full :class:`StageReport` as ``.report``, so a
caller that catches it loses no detail versus the success path. The driver
(``scripts/full_validation_harness.py``) owns process exit codes and the
"stop at the first failing stage" contract; this module only reports.

stdlib-only: the harness runs inside the graph-os pod via
``kubectl exec -i ... -- python3 -`` with no ability to install anything, so
HTTP calls use :mod:`urllib`, never ``requests``/``httpx``.
"""

from __future__ import annotations

import dataclasses
import time
import urllib.error
import urllib.request
from typing import Any

__all__ = [
    "DEFAULT_BASE_URL",
    "DEFAULT_BROWSER_TIMEOUT_S",
    "DEFAULT_API_TIMEOUT_S",
    "EndpointCheck",
    "StageReport",
    "HarnessStageError",
    "BrowserStageError",
    "ApiGatewayStageError",
    "AdmissionStageError",
    "stage_browser",
    "stage_api_gateway",
    "stage_admission",
]

#: The pod serves the webui on this address; never hardcode it into a stage —
#: always accept it as a parameter.
DEFAULT_BASE_URL = "http://127.0.0.1:8080"

DEFAULT_BROWSER_TIMEOUT_S = 10.0
#: /api/enhanced/tools has historically taken ~13s versus its
#: /api/registry/tools counterpart's ~2s (the Workstream D regression signal
#: this stage exists to keep measuring) — the timeout has to clear that
#: comfortably or a real regression reads as a false connection failure.
DEFAULT_API_TIMEOUT_S = 30.0

_BROWSER_PATHS: tuple[str, ...] = ("/", "/auth/login", "/graph", "/skills")

#: 5xx or a connection failure (status 0) is always FAIL. 200/302/303/401 are
#: PASS — the app is up and is either serving the page or correctly failing
#: closed. Anything else (e.g. an unexpected 404/403/3xx) is deliberately
#: treated as FAIL too: this stage only asserts what it was told is safe, it
#: never guesses that an unlisted status is fine.
_BROWSER_PASS_STATUSES = frozenset({200, 302, 303, 401})

_REGISTRY_ENDPOINTS: tuple[str, ...] = ("tools", "skills", "prompts", "servers")
_ENHANCED_ENDPOINTS: tuple[str, ...] = (
    "tools",
    "skills",
    "graph/nodes",
    "graph/stats",
    "llm/models",
    "llm/model-schema",
)

#: Path used for the /api/registry/tools vs /api/enhanced/tools latency
#: comparison called out in the lane brief.
_LATENCY_SIGNAL_REGISTRY_PATH = "/api/registry/tools?limit=5"
_LATENCY_SIGNAL_ENHANCED_PATH = "/api/enhanced/tools"


# ---------------------------------------------------------------------------
# Structured results
# ---------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class EndpointCheck:
    """One probed HTTP path and what actually happened.

    ``status`` is ``0`` for a connection failure (refused, DNS, timeout) —
    never a code an HTTP server could return, so it is unambiguous.
    """

    method: str
    path: str
    status: int
    elapsed_s: float
    ok: bool
    note: str = ""


@dataclasses.dataclass(frozen=True)
class StageReport:
    """The structured result of one stage — returned on success, and also
    attached (as ``.report``) to the exception raised on failure, so no
    detail is lost on either path."""

    stage: str
    ok: bool
    elapsed_s: float
    checks: tuple[EndpointCheck, ...] = ()
    detail: str = ""
    #: True only for stage_admission — see its HONESTY REQUIREMENT docstring.
    surrogate: bool = False
    surrogate_note: str = ""

    def render(self) -> str:
        mark = "PASS" if self.ok else "FAIL"
        lines = [f"[{mark}] {self.stage} [{self.elapsed_s:6.2f}s] {self.detail}"]
        for check in self.checks:
            cmark = "PASS" if check.ok else "FAIL"
            lines.append(
                f"    {cmark:4s} {check.method:4s} {check.path:36s} "
                f"status={check.status:<4d} {check.elapsed_s:6.2f}s {check.note}"
            )
        if self.surrogate:
            lines.append(f"    SURROGATE: {self.surrogate_note}")
        return "\n".join(lines)


class HarnessStageError(RuntimeError):
    """Raised by a stage on failure. Carries the full :class:`StageReport`
    (``.report``) so a caller never has to re-derive what happened from the
    message string alone. Stage functions raise this instead of calling
    ``sys.exit`` — the driver decides what a failure means for the process
    exit code."""

    def __init__(self, report: StageReport) -> None:
        super().__init__(report.detail or f"{report.stage} stage failed")
        self.report = report


class BrowserStageError(HarnessStageError):
    """The unauthenticated browser-path stage failed."""


class ApiGatewayStageError(HarnessStageError):
    """The service-token API-gateway stage failed."""


class AdmissionStageError(HarnessStageError):
    """The surrogate tenant-admission stage failed."""


# ---------------------------------------------------------------------------
# HTTP plumbing (stdlib-only)
# ---------------------------------------------------------------------------
class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Report a 30x AS a 30x. Left to its default, ``urllib`` transparently
    follows redirects and the caller never sees the 302/303 a browser-path
    check needs to grade."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _no_redirect_opener() -> urllib.request.OpenerDirector:
    return urllib.request.build_opener(_NoRedirect)


def _check_endpoint(
    opener: urllib.request.OpenerDirector,
    method: str,
    base_url: str,
    path: str,
    *,
    timeout: float,
    pass_statuses: frozenset[int],
    headers: dict[str, str] | None = None,
) -> EndpointCheck:
    """Probe one path and grade it against ``pass_statuses``.

    Never raises: a connection failure is recorded as ``status=0`` and
    ``ok=False`` (unless 0 is itself a pass status, which it never is here)
    so a whole stage can finish and report every path instead of aborting on
    the first bad one.
    """
    url = base_url.rstrip("/") + path
    req = urllib.request.Request(url, method=method, headers=dict(headers or {}))
    t0 = time.monotonic()
    try:
        resp = opener.open(req, timeout=timeout)
        try:
            status = int(getattr(resp, "status", None) or resp.getcode() or 0)
        finally:
            resp.close()
        note = ""
    except urllib.error.HTTPError as exc:
        status = int(exc.code)
        note = str(exc.reason or "")
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        status = 0
        reason = getattr(exc, "reason", exc)
        note = f"{type(exc).__name__}: {reason}"
    elapsed = time.monotonic() - t0
    return EndpointCheck(
        method=method,
        path=path,
        status=status,
        elapsed_s=elapsed,
        ok=status in pass_statuses,
        note=note,
    )


# ---------------------------------------------------------------------------
# Stage: browser
# ---------------------------------------------------------------------------
def stage_browser(
    base_url: str = DEFAULT_BASE_URL, *, timeout: float = DEFAULT_BROWSER_TIMEOUT_S
) -> StageReport:
    """Probe the unauthenticated, user-visible path — checked FIRST because
    it is the authoritative user-visible signal.

    ``GET /``, ``/auth/login``, ``/graph``, ``/skills`` with NO credential.
    ``5xx`` or a connection failure (``status=0``) is FAIL; ``200``/``302``/
    ``303``/``401`` is PASS (the app is up and either serving or correctly
    failing closed). Every path's actual status is recorded in the returned
    report (and in the raised error's ``.report``) so a human can see
    exactly what happened, not just pass/fail.

    Raises :class:`BrowserStageError` if any path fails.
    """
    opener = _no_redirect_opener()
    t0 = time.monotonic()
    checks = tuple(
        _check_endpoint(
            opener,
            "GET",
            base_url,
            path,
            timeout=timeout,
            pass_statuses=_BROWSER_PASS_STATUSES,
        )
        for path in _BROWSER_PATHS
    )
    elapsed = time.monotonic() - t0
    ok = all(check.ok for check in checks)
    detail = "; ".join(f"{check.path}={check.status}" for check in checks)
    report = StageReport(stage="browser", ok=ok, elapsed_s=elapsed, checks=checks, detail=detail)
    if not ok:
        failing = "; ".join(
            f"{check.path} status={check.status}" + (f" ({check.note})" if check.note else "")
            for check in checks
            if not check.ok
        )
        raise BrowserStageError(
            dataclasses.replace(
                report,
                detail=(
                    "browser path FAILED (unauthenticated user-visible signal — "
                    f"the app itself is down, not merely an auth guard): {failing}"
                ),
            )
        )
    return report


# ---------------------------------------------------------------------------
# Stage: api_gateway
# ---------------------------------------------------------------------------
def _api_gateway_paths() -> tuple[str, ...]:
    registry = tuple(f"/api/registry/{name}?limit=5" for name in _REGISTRY_ENDPOINTS)
    enhanced = tuple(f"/api/enhanced/{name}" for name in _ENHANCED_ENDPOINTS)
    return registry + enhanced


def _latency_signal(checks: tuple[EndpointCheck, ...]) -> str:
    by_path = {check.path: check for check in checks}
    registry = by_path.get(_LATENCY_SIGNAL_REGISTRY_PATH)
    enhanced = by_path.get(_LATENCY_SIGNAL_ENHANCED_PATH)
    if registry is None or enhanced is None:
        return ""
    delta = enhanced.elapsed_s - registry.elapsed_s
    return (
        f" latency_signal registry/tools={registry.elapsed_s:.2f}s "
        f"enhanced/tools={enhanced.elapsed_s:.2f}s delta={delta:+.2f}s"
    )


def stage_api_gateway(
    base_url: str = DEFAULT_BASE_URL,
    *,
    timeout: float = DEFAULT_API_TIMEOUT_S,
    config: Any = None,
) -> StageReport:
    """Probe the service-token API surface with a real bearer.

    Acquires a bearer via ``acquire_process_identity_token(config)`` from
    ``agent_utilities.security.request_identity`` (``config`` defaults to a
    fresh ``AgentConfig()``; a test may pass any object through unused by
    injecting a fake ``acquire_process_identity_token``). That call XORs
    ``KG_AUTH_TOKEN_REF`` vs ``KG_IDENTITY_OAUTH2`` and raises if both or
    neither is configured — see
    ``agent_utilities/security/request_identity.py:574-612``.

    Probes, expecting HTTP 200 on each: ``/api/registry/tools|skills|
    prompts|servers?limit=5``, ``/api/enhanced/tools``, ``/api/enhanced/
    skills``, ``/api/enhanced/graph/nodes``, ``/api/enhanced/graph/stats``,
    ``/api/enhanced/llm/models``, ``/api/enhanced/llm/model-schema``.

    Records per-endpoint latency in every :class:`EndpointCheck` and surfaces
    the ``/api/enhanced/tools`` (historically ~13s) vs ``/api/registry/tools``
    (historically ~2s) comparison explicitly in ``detail`` — the Workstream D
    regression signal.

    Raises :class:`ApiGatewayStageError` if any endpoint fails to return 200,
    or if the bearer cannot be acquired at all.
    """
    from agent_utilities.security.request_identity import acquire_process_identity_token

    if config is None:
        from agent_utilities.core.config import AgentConfig

        config = AgentConfig()

    t0 = time.monotonic()
    try:
        token = acquire_process_identity_token(config)
    except Exception as exc:
        elapsed = time.monotonic() - t0
        report = StageReport(
            stage="api_gateway",
            ok=False,
            elapsed_s=elapsed,
            checks=(),
            detail=f"could not acquire a process identity bearer: {type(exc).__name__}: {exc}",
        )
        raise ApiGatewayStageError(report) from exc

    headers = {"Authorization": f"Bearer {token}"}
    opener = _no_redirect_opener()
    checks = tuple(
        _check_endpoint(
            opener,
            "GET",
            base_url,
            path,
            timeout=timeout,
            pass_statuses=frozenset({200}),
            headers=headers,
        )
        for path in _api_gateway_paths()
    )
    elapsed = time.monotonic() - t0
    ok = all(check.ok for check in checks)
    latency_note = _latency_signal(checks)
    detail = (
        "; ".join(f"{check.path}={check.status}({check.elapsed_s:.2f}s)" for check in checks)
        + latency_note
    )
    report = StageReport(stage="api_gateway", ok=ok, elapsed_s=elapsed, checks=checks, detail=detail)
    if not ok:
        failing = "; ".join(
            f"{check.path} status={check.status}" + (f" ({check.note})" if check.note else "")
            for check in checks
            if not check.ok
        )
        raise ApiGatewayStageError(
            dataclasses.replace(
                report,
                detail=f"api_gateway stage FAILED: {failing}{latency_note}",
            )
        )
    return report


# ---------------------------------------------------------------------------
# Stage: admission (SURROGATE)
# ---------------------------------------------------------------------------
_ADMISSION_SURROGATE_NOTE = (
    "SURROGATE ONLY — this is NOT proof a human can sign in. It mints this "
    "pod's own graph process identity and runs the exact "
    "run_tenant_admission()/provision_tenant_access() call that agent-webui's "
    "ensure_tenant_admission() makes on every authenticated request "
    "(agent-webui/agent/agent_webui/server.py HTTP leg ~1418-1460, WS leg "
    "~1367-1395). agent-webui's REAL login is OIDC authorization-code + PKCE "
    "via Keycloak, and there is no resource-owner-password grant anywhere in "
    "this codebase, so no service credential can perform an actual browser "
    "login. A PASS here proves the admission RPC itself is reachable and "
    "grants successfully under a verified process identity; it does NOT "
    "prove a human can sign in, and must never be read as if it does."
)


def stage_admission(
    *, tenant_slug: str = "homelab", config: Any = None
) -> StageReport:
    """Invoke the tenant-admission RPC a signed-in principal actually
    triggers — a SURROGATE for human sign-in, not proof of it.

    Mints an actor/session using this pod's own graph process identity (the
    same ``acquire_process_identity_token`` -> ``mint_actor_from_token_sync``
    -> ``mint_graph_session`` chain ``agent-webui``'s service-authority
    broker uses — see ``agent_webui/graph_admission.py``'s
    ``_service_authority``), binds it as the ambient actor/session, and calls
    ``run_tenant_admission(tenant_slug, [TenantPrincipal(agent_id=...)],
    apply=True)`` from ``agent_utilities.security.tenant_admission_cli`` —
    exactly the call ``ensure_tenant_admission`` makes on every authenticated
    request.

    HONESTY REQUIREMENT: this stage's output — both the docstring here and
    every :class:`StageReport` it returns or raises — is labeled explicitly
    as a surrogate (``report.surrogate is True``, with the caveat in
    ``report.surrogate_note``). agent-webui's real login is OIDC
    authorization-code + PKCE and there is no resource-owner-password grant
    anywhere in this codebase, so no service account can perform a real
    browser login. Never read a PASS here as "a human can sign in".

    Raises :class:`AdmissionStageError` if the identity cannot be minted or
    the admission call fails.
    """
    from agent_utilities.knowledge_graph.core.session import use_session
    from agent_utilities.security.brain_context import use_actor
    from agent_utilities.security.request_identity import (
        acquire_process_identity_token,
        mint_actor_from_token_sync,
        mint_graph_session,
    )
    from agent_utilities.security.tenant_admission_cli import run_tenant_admission
    from agent_utilities.security.tenant_rbac_admission import TenantPrincipal

    if config is None:
        from agent_utilities.core.config import AgentConfig

        config = AgentConfig()

    t0 = time.monotonic()
    try:
        token = acquire_process_identity_token(config)
        actor = mint_actor_from_token_sync(token)
        session = mint_graph_session(actor)
        session.engine_verified_context()
        with use_actor(session.actor), use_session(session):
            result = run_tenant_admission(
                tenant_slug,
                [TenantPrincipal(agent_id=session.actor.actor_id)],
                apply=True,
            )
    except Exception as exc:
        elapsed = time.monotonic() - t0
        report = StageReport(
            stage="admission",
            ok=False,
            elapsed_s=elapsed,
            checks=(),
            detail=f"admission SURROGATE call failed: {type(exc).__name__}: {exc}",
            surrogate=True,
            surrogate_note=_ADMISSION_SURROGATE_NOTE,
        )
        raise AdmissionStageError(report) from exc

    elapsed = time.monotonic() - t0
    ok = bool(getattr(result, "all_admitted", False))
    detail = (
        f"tenant={getattr(result, 'tenant_slug', tenant_slug)!r} "
        f"role={getattr(result, 'role', '?')!r} all_admitted={ok} "
        f"agent_id={session.actor.actor_id!r} [SURROGATE, not a human sign-in proof]"
    )
    report = StageReport(
        stage="admission",
        ok=ok,
        elapsed_s=elapsed,
        checks=(),
        detail=detail,
        surrogate=True,
        surrogate_note=_ADMISSION_SURROGATE_NOTE,
    )
    if not ok:
        raise AdmissionStageError(report)
    return report


# ---------------------------------------------------------------------------
# Standalone smoke-test entrypoint — the driver lane does NOT use this; it
# imports the three stage_* callables directly. Kept only for manually
# exercising this module in isolation inside the pod.
# ---------------------------------------------------------------------------
def _main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--tenant-slug", default="homelab")
    parser.add_argument(
        "--skip-admission",
        action="store_true",
        help="skip the surrogate admission stage (e.g. no live engine handy)",
    )
    args = parser.parse_args()

    stages: list[tuple[str, Any]] = [
        ("browser", lambda: stage_browser(args.base_url)),
        ("api_gateway", lambda: stage_api_gateway(args.base_url)),
    ]
    if not args.skip_admission:
        stages.append(("admission", lambda: stage_admission(tenant_slug=args.tenant_slug)))

    for n, (name, fn) in enumerate(stages, 1):
        try:
            report = fn()
        except HarnessStageError as exc:
            print(exc.report.render())
            return n
        print(report.render())
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
