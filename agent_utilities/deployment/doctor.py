#!/usr/bin/python
"""``agent-utilities doctor`` — one holistic health sweep of a deployment.

Like ``brew doctor`` / ``flutter doctor``: runs a battery of independent checks
across every subsystem, each reporting ok / warn / fail / skip with a concrete
**remediation** — and, where one exists, the **skill or command that fixes it** so
the operator (or Claude) can act or auto-fix. The doctor is a thin *aggregator*: it
composes the diagnostics that already exist (config_doctor, shard topology probe,
backend health_check, the hook doctor, the MCP-config validator, secrets resolution)
rather than re-implementing them.

Each check is defensive — a missing optional dependency or an unreachable service
yields ``skip``/``warn``/``fail`` with guidance, never a crash. ``run_doctor`` returns
a structured report; ``fix=True`` runs the conservative, idempotent auto-remediations
(only checks marked ``auto_fixable``).
"""

from __future__ import annotations

import importlib
import logging
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)

# Status precedence (worst wins for the overall verdict).
_RANK = {"ok": 0, "skip": 0, "warn": 1, "fail": 2, "error": 2}


def _result(
    name: str,
    status: str,
    detail: str,
    *,
    remediation: str | None = None,
    skill: str | None = None,
    auto_fixable: bool = False,
    data: Any = None,
) -> dict[str, Any]:
    return {
        "name": name,
        "status": status,
        "detail": detail,
        "remediation": remediation,
        "skill": skill,
        "auto_fixable": auto_fixable,
        "data": data,
    }


# ── individual checks (each returns one _result; never raises) ──────────────
def _check_python_env() -> dict[str, Any]:
    import platform
    import sys

    try:
        import agent_utilities

        ver = getattr(agent_utilities, "__version__", "unknown")
    except Exception as exc:  # noqa: BLE001
        return _result(
            "python_env",
            "fail",
            f"agent_utilities not importable: {exc}",
            remediation="pip install agent-utilities[all]",
        )
    optional = {}
    for mod, label in (
        ("rdflib", "owl/sparql"),
        ("psycopg", "postgres"),
        ("stardog", "stardog"),
    ):
        optional[label] = importlib.util.find_spec(mod) is not None
    py_ok = sys.version_info >= (3, 10)
    missing = [k for k, v in optional.items() if not v]
    status = "ok" if py_ok else "warn"
    detail = (
        f"Python {platform.python_version()}, agent-utilities {ver}; "
        f"optional extras present: {[k for k, v in optional.items() if v] or 'none'}"
    )
    if missing:
        detail += f"; absent (install if needed): {missing}"
    return _result(
        "python_env",
        status,
        detail,
        remediation=None if py_ok else "agent-utilities needs Python 3.10+",
        data=optional,
    )


def _check_config() -> dict[str, Any]:
    try:
        from agent_utilities.deployment.config_generator import config_doctor

        rep = config_doctor()
    except Exception as exc:  # noqa: BLE001
        return _result("config", "error", f"config_doctor failed: {exc}")
    healthy = rep.get("healthy")
    profile = rep.get("profile", "?")
    if healthy:
        return _result(
            "config", "ok", f"config healthy for profile {profile!r}", data=rep
        )
    # Tiny's durability findings are advisory.
    status = "warn" if profile == "tiny" else "fail"
    return _result(
        "config",
        status,
        f"config needs attention (profile {profile!r}) — see checks",
        remediation="`setup-config doctor` for detail; `setup-config generate --profile <p>` to (re)seed",
        skill="agent-utilities-deployment",
        data=rep,
    )


def _check_engine() -> dict[str, Any]:
    try:
        from agent_utilities.core.config import AgentConfig
        from agent_utilities.knowledge_graph.core.engine_resolver import resolve_engine
        from agent_utilities.knowledge_graph.core.shard_topology import (
            default_graph_name,
            shard_topology_status,
        )

        cfg = AgentConfig()
        st = shard_topology_status(cfg, probe=True, timeout=0.5)
        resolved = resolve_engine(cfg, default_graph_name(cfg))
    except Exception as exc:  # noqa: BLE001
        return _result("engine", "error", f"shard topology probe failed: {exc}")
    st["resolved_mode"] = resolved.mode
    endpoints = st.get("endpoints", [])
    reachable = [e for e in endpoints if e.get("reachable")]

    # CONCEPT:OS-5.63 — report the RESOLVED mode (how this process reaches the
    # engine), not just transport reachability.
    if resolved.mode == "remote":
        if reachable:
            return _result(
                "engine",
                "ok",
                f"engine reachable at remote endpoint {resolved.endpoint!r} "
                f"({len(reachable)}/{len(endpoints)} endpoint(s)) — resolved mode=remote (deployed elsewhere)",
                data=st,
            )
        return _result(
            "engine",
            "fail",
            f"configured remote engine {resolved.endpoint!r} is unreachable — "
            "remote mode never autostarts a local stand-in (fail-loud)",
            remediation="start the remote engine (Docker/host) or fix ENGINE_ENDPOINT / GRAPH_SERVICE_ENDPOINTS",
            skill="agent-utilities-deployment",
            data=st,
        )

    if reachable:
        return _result(
            "engine",
            "ok",
            f"engine reachable at {resolved.endpoint!r} — resolved mode=shared "
            "(reusing the already-running local engine)",
            data=st,
        )

    # Nothing up locally — describe the autostart behaviour the resolver WILL
    # take on first use, including the idle-shutdown lifecycle.
    if resolved.autostart_allowed:
        if resolved.idle_shutdown_secs > 0:
            life = (
                f"reference-counted (auto-stops {resolved.idle_shutdown_secs}s "
                "after the last client disconnects)"
            )
        else:
            life = "persistent (never auto-stops — runs like a local service)"
        return _result(
            "engine",
            "warn",
            f"no engine running yet at {resolved.endpoint!r} — resolved mode=autostart: "
            f"a detached, supervised engine will be spawned on first use, {life}",
            remediation="no action needed (auto-provisions on demand); start eagerly with `graph-os-daemon` if preferred",
            skill="agent-utilities-deployment",
            data=st,
        )
    return _result(
        "engine",
        "fail",
        f"no epistemic-graph engine endpoint reachable ({len(endpoints)} configured) and autostart disabled",
        remediation="start the engine/gateway: `graph-os-daemon` (or `cargo run -p epistemic-graph`), or set engine_mode=embedded",
        skill="agent-utilities-deployment",
        data=st,
    )


def _check_graph_backend() -> dict[str, Any]:
    # Only health-check an ALREADY-active backend — a doctor must be cheap and
    # non-mutating; constructing one here can lock the L2 store / retry-storm.
    try:
        from agent_utilities.knowledge_graph.backends import get_active_backend

        backend = get_active_backend()
        if backend is None:
            return _result(
                "graph_backend",
                "skip",
                "no graph backend active in this process (start the gateway to evaluate)",
            )
        inner = getattr(backend, "inner", backend)
        name = type(inner).__name__
        hc = getattr(inner, "health_check", None)
        ok = hc() if callable(hc) else True
    except Exception as exc:  # noqa: BLE001
        return _result(
            "graph_backend",
            "warn",
            f"backend not evaluable: {exc}",
            remediation="check GRAPH_BACKEND / GRAPH_DB_URI",
            skill="database-environment-setup",
        )
    if ok:
        return _result("graph_backend", "ok", f"{name} reachable")
    return _result(
        "graph_backend",
        "fail",
        f"{name} health_check failed",
        remediation="verify the durable DSN / that Postgres is up",
        skill="database-environment-setup",
    )


def _check_secrets() -> dict[str, Any]:
    try:
        from agent_utilities.core.config import AgentConfig
        from agent_utilities.deployment.config_generator import _unresolved_secret_refs

        unresolved = _unresolved_secret_refs(AgentConfig())
    except Exception as exc:  # noqa: BLE001
        return _result("secrets", "skip", f"secrets backend not evaluated: {exc}")
    if not unresolved:
        return _result("secrets", "ok", "no unresolved secret references")
    return _result(
        "secrets",
        "fail",
        f"unresolved secret refs: {unresolved}",
        remediation="seed the values in your secrets backend",
        skill="secret-vault-manager",
        data=unresolved,
    )


def _check_auth() -> dict[str, Any]:
    from agent_utilities.core.config import setting

    required = setting("KG_AUTH_REQUIRED", False, cast=bool)
    jwks = setting("AUTH_JWT_JWKS_URI", "")
    if not required:
        return _result("auth", "ok", "KG auth not required (open mode)")
    if jwks:
        # IdP-agnostic: any OIDC issuer's JWKS works. Name it for the report so
        # an operator on Okta isn't told they need Keycloak (CONCEPT:OS-5.43 genesis
        # IdP choice — keycloak deploy-if-absent OR an existing okta/other-oidc org).
        low = jwks.lower()
        idp = "Okta" if "okta" in low else ("Keycloak" if "keycloak" in low else "OIDC")
        return _result("auth", "ok", f"KG auth required and JWKS configured ({idp})")
    return _result(
        "auth",
        "fail",
        "KG_AUTH_REQUIRED is set but AUTH_JWT_JWKS_URI is missing — fail-closed",
        remediation=(
            "set AUTH_JWT_JWKS_URI to your IdP's JWKS/certs endpoint — your existing "
            "Okta org (…/oauth2/<server>/v1/keys) or deploy Keycloak"
        ),
        skill="keycloak-client-onboarder",
    )


def _check_mcp_fleet(live: bool = False) -> dict[str, Any]:
    try:
        import json
        from pathlib import Path

        from agent_utilities.core.workspace import get_mcp_config_path

        path = get_mcp_config_path()
        if not path or not Path(path).exists():
            return _result(
                "mcp_fleet",
                "skip",
                "no mcp_config.json found in workspace",
                remediation="`setup-config mcp` to generate the minimal config (graph-os + mcp-multiplexer)",
            )
        import importlib.util as _u

        spec = _u.find_spec("scripts.validate_mcp_config")
        if spec is None:
            return _result("mcp_fleet", "skip", "validate_mcp_config not importable")
        cfg = json.loads(Path(path).read_text())
        mod = importlib.import_module("scripts.validate_mcp_config")
        rep = mod.validate(
            cfg, mod.caddy_hosts() if hasattr(mod, "caddy_hosts") else set(), live=live
        )
    except Exception as exc:  # noqa: BLE001
        return _result("mcp_fleet", "skip", f"fleet check skipped: {exc}")
    if rep.get("passed"):
        return _result(
            "mcp_fleet", "ok", f"{len(rep.get('ok', []))} MCP server(s) valid", data=rep
        )
    bad = {**rep.get("invalid", {}), **rep.get("unreachable", {})}
    return _result(
        "mcp_fleet",
        "warn",
        f"{len(bad)} MCP server(s) need attention",
        remediation="`python scripts/validate_mcp_config.py --live` for detail",
        data=rep,
    )


def _check_hooks() -> dict[str, Any]:
    try:
        from agent_utilities.ecosystem.hook_installer import HookInstaller

        rep = HookInstaller().doctor()
    except Exception as exc:  # noqa: BLE001
        return _result("hooks", "skip", f"hook doctor unavailable: {exc}")
    installed = [k for k, v in rep.items() if v.get("status") == "healthy"]
    stale = [k for k, v in rep.items() if v.get("status") == "stale"]
    if stale:
        return _result(
            "hooks",
            "warn",
            f"{len(stale)} stale hook(s): {stale}",
            remediation="re-install hooks (`graph_configure action=install_hooks`)",
            auto_fixable=True,
            data=rep,
        )
    return _result("hooks", "ok", f"{len(installed)} agent hook(s) healthy", data=rep)


def _check_observability() -> dict[str, Any]:
    from agent_utilities.core.config import setting
    from agent_utilities.core.profile_guard import is_production_profile

    if not is_production_profile():
        return _result("observability", "skip", "not a production profile")
    otel = setting("OTEL_EXPORTER_OTLP_ENDPOINT", "")
    metrics = setting("GATEWAY_METRICS", False, cast=bool)
    if otel and metrics:
        return _result("observability", "ok", "metrics + OTEL export configured")
    return _result(
        "observability",
        "warn",
        f"production profile but observability partial (metrics={metrics}, otel={'set' if otel else 'unset'})",
        remediation="enable GATEWAY_METRICS and set OTEL_EXPORTER_OTLP_ENDPOINT",
        skill="service-observability-provisioner",
    )


def _check_graph_connections() -> dict[str, Any]:
    """Health-check the named graph-connection registry (CONCEPT:KG-2.63/2.89).

    Reports the registered external connections + their roles, and flags any
    stalled fan-out mirror (the actionable replication-health signal)."""
    try:
        from agent_utilities.mcp.kg_server import get_connection_registry

        status = get_connection_registry().status()
        conns = [c for c in status.get("connections", []) if c.get("name") != "default"]
    except Exception as exc:  # noqa: BLE001
        return _result("graph_connections", "skip", f"registry unavailable: {exc}")

    stalled: list[str] = []
    try:
        from agent_utilities.knowledge_graph.backends import get_active_backend
        from agent_utilities.knowledge_graph.backends.fanout_backend import (
            FanOutBackend,
        )

        backend = get_active_backend()
        cand = getattr(backend, "inner", backend)
        fan = cand if isinstance(cand, FanOutBackend) else getattr(cand, "l3", None)
        if isinstance(fan, FanOutBackend):
            mirrors = fan.durability_stats().get("mirrors") or {}
            stalled = [m for m, s in mirrors.items() if s.get("stalled")]
    except Exception:  # noqa: BLE001 — mirror stats are best-effort
        pass

    if stalled:
        return _result(
            "graph_connections",
            "warn",
            f"{len(conns)} connection(s); STALLED mirror(s): {', '.join(stalled)}",
            remediation="`graph_configure action=reconcile` and check the mirror backend",
            skill="database-environment-setup",
            data={"connections": conns, "stalled_mirrors": stalled},
        )
    by_role: dict[str, int] = {}
    for c in conns:
        r = str(c.get("role") or "read")
        by_role[r] = by_role.get(r, 0) + 1
    detail = (
        f"{len(conns)} external connection(s) ("
        + ", ".join(f"{k}={v}" for k, v in sorted(by_role.items()))
        + ")"
        if conns
        else "no external connections registered"
    )
    return _result("graph_connections", "ok", detail, data={"connections": conns})


def _check_ingestion_coverage() -> dict[str, Any]:
    """Assert the agent-packages repos are ingested + fresh (CONCEPT:OS-5.47).

    Native codebase-context-via-KG requires the index to be reliably populated:
    if a repo has no ``:Code`` symbols (or its last delta sync is stale) a KG code
    query returns nothing and the agent silently falls back to grep. This compares
    ``workspace.yml``'s agent-packages subtree against the live KG + DeltaManifest
    freshness, so coverage gaps are visible rather than silent (GAP 1)."""
    try:
        from agent_utilities.knowledge_graph.ingestion.coverage import (
            assess_coverage,
            enumerate_agent_packages_repos,
            find_workspace_manifest,
            repo_symbol_counts,
        )

        manifest = find_workspace_manifest()
        if manifest is None:
            return _result(
                "ingestion_coverage",
                "skip",
                "workspace.yml not found (not a workspace checkout)",
            )
        repos = enumerate_agent_packages_repos(manifest)
        if not repos:
            return _result(
                "ingestion_coverage", "skip", "no agent-packages repos in workspace.yml"
            )
        from agent_utilities.knowledge_graph.backends import get_active_backend

        backend = get_active_backend()
        counts = repo_symbol_counts(backend, repos)
    except Exception as exc:  # noqa: BLE001
        return _result(
            "ingestion_coverage", "skip", f"coverage probe unavailable: {exc}"
        )

    freshness: dict[str, str] = {}
    try:
        from agent_utilities.knowledge_graph.ingestion.manifest import DeltaManifest

        dm = DeltaManifest(backend=backend)
        for cat in ("codebase", "codebase_file"):
            freshness.update(dm.freshness("agent_graph", cat))
    except Exception:  # noqa: BLE001 — freshness is best-effort
        freshness = {}

    rep = assess_coverage(repos, counts, freshness)
    detail = (
        f"{rep['covered']}/{rep['total']} agent-packages repos ingested "
        f"({rep['coverage_pct']}%), {rep['total_symbols']} symbols"
    )
    if rep["stale"]:
        detail += f", {len(rep['stale'])} stale (>{rep['sla_days']}d)"
    if rep["missing"] or rep["stale"]:
        status = "fail" if rep["coverage_pct"] < 75 else "warn"
        miss = ", ".join(rep["missing"][:8]) + ("…" if len(rep["missing"]) > 8 else "")
        return _result(
            "ingestion_coverage",
            status,
            detail + (f"; missing: {miss}" if rep["missing"] else ""),
            remediation=(
                "`source_sync source=all mode=delta` (or `graph_ingest "
                "action=ingest target_path=<repo>`) to ingest/refresh the gaps"
            ),
            skill="kg-ingest",
            data=rep,
        )
    return _result("ingestion_coverage", "ok", detail, data=rep)


def _check_connector_coverage() -> dict[str, Any]:
    """Assert every configured connector is ingesting + fresh (CONCEPT:OS-5.48).

    The connector analogue of ``ingestion_coverage``: a dark or stale connector
    means the world-model for that domain (tickets, deploys, processes…) is silently
    wrong and the agent falls back to hitting the source system. Compares the
    expected connector set against their ``DeltaManifest`` watermarks."""
    try:
        from agent_utilities.knowledge_graph.backends import get_active_backend
        from agent_utilities.knowledge_graph.ingestion.connector_coverage import (
            CONNECTOR_CATEGORY,
            assess_connector_coverage,
            enumerate_expected_connectors,
        )
        from agent_utilities.knowledge_graph.ingestion.manifest import DeltaManifest

        expected = enumerate_expected_connectors()
        if not expected:
            return _result("connector_coverage", "skip", "no connectors configured")
        backend = get_active_backend()
        dm = DeltaManifest(backend=backend)
        freshness: dict[str, str] = {}
        for graph in ("agent_graph", "__commons__"):
            freshness.update(dm.freshness(graph, CONNECTOR_CATEGORY))
    except Exception as exc:  # noqa: BLE001
        return _result(
            "connector_coverage", "skip", f"connector probe unavailable: {exc}"
        )

    rep = assess_connector_coverage(expected, freshness)
    detail = (
        f"{rep['covered']}/{rep['total']} connectors ingesting ({rep['coverage_pct']}%)"
    )
    if rep["stale"]:
        detail += f", {len(rep['stale'])} stale (>{rep['sla_days']}d)"
    if rep["missing"] or rep["stale"]:
        miss = ", ".join(rep["missing"][:8]) + ("…" if len(rep["missing"]) > 8 else "")
        return _result(
            "connector_coverage",
            "warn",
            detail + (f"; dark: {miss}" if rep["missing"] else ""),
            remediation="`source_sync source=all mode=delta` to refresh; check the connector's creds/preset",
            skill="kg-ingest",
            data=rep,
        )
    return _result("connector_coverage", "ok", detail, data=rep)


def _check_workspace_config() -> dict[str, Any]:
    """Validate the ``workspace.yml`` repository manifest (CONCEPT:OS-5.67).

    ``workspace.yml`` is the canonical map of the ecosystem's repositories: the
    bootstrap (``clone_missing_projects``), the read-only project enumeration that
    self-configures KG ingestion breadth (``workspace_project_roots``, KG-2.7), and
    genesis all parse it. A malformed manifest, a repository entry with no ``url``,
    or an incoherent ``subdirectories`` shape silently shrinks what the platform
    clones/ingests — so we validate it through the SAME loader (no re-parse) and
    surface gaps as a doctor finding rather than a silent miss."""
    try:
        from agent_utilities.core.workspace_config import validate_workspace_yml

        rep = validate_workspace_yml()
    except Exception as exc:  # noqa: BLE001
        return _result(
            "workspace_config", "skip", f"workspace.yml validator unavailable: {exc}"
        )

    if not rep["found"]:
        return _result(
            "workspace_config",
            "skip",
            "no workspace.yml found (not a workspace checkout)",
            remediation=(
                "copy docs/examples/workspace.yml to the workspace root (or the "
                "agent-utilities XDG config dir) and edit it for your repos"
            ),
        )

    where = rep.get("path", "?")
    if rep["errors"]:
        head = "; ".join(rep["errors"][:5]) + ("…" if len(rep["errors"]) > 5 else "")
        return _result(
            "workspace_config",
            "fail",
            f"workspace.yml at {where} has {len(rep['errors'])} error(s): {head}",
            remediation=(
                "fix the listed entries; see docs/guides/workspace-config.md for the "
                "schema + an annotated template (docs/examples/workspace.yml)"
            ),
            skill="agent-utilities-deployment",
            data=rep,
        )
    detail = f"workspace.yml valid at {where} — {rep['repo_count']} repositories"
    if rep["warnings"]:
        nwarn = len(rep["warnings"])
        return _result(
            "workspace_config",
            "warn",
            detail
            + f", {nwarn} advisory warning(s): {rep['warnings'][0]}"
            + ("…" if nwarn > 1 else ""),
            remediation="see docs/guides/workspace-config.md for the full schema",
            data=rep,
        )
    return _result("workspace_config", "ok", detail, data=rep)


def _check_bus() -> dict[str, Any]:
    """Report agent-bus health: participants, online count, stale agents, mailbox backlog.

    CONCEPT:ECO-4.87 — the operator view of the AgentBus (ECO-4.84). A large un-acked
    backlog or zero online participants on a hub that should be busy is the signal that
    sessions aren't draining their mailboxes or aren't heart-beating.
    """
    try:
        from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine
        from agent_utilities.messaging.bus import AgentBus

        engine = IntelligenceGraphEngine.get_active()
        if engine is None:
            return _result("bus", "skip", "no active engine")
        bus = AgentBus.instance(engine)
        st = bus.status()
        backlog = bus._query(
            "MATCH (m:BusMessage {status: 'sent'}) RETURN count(m) as n", {}
        )
        pending = int(backlog[0].get("n", 0)) if backlog else 0
    except Exception as exc:  # noqa: BLE001
        return _result("bus", "skip", f"bus probe unavailable: {exc}")

    detail = (
        f"{st['online']}/{st['agents']} participants online, "
        f"{len(st['topics'])} topics, {pending} un-acked message(s)"
    )
    data = {**st, "pending_messages": pending}
    if pending > 1000:
        return _result(
            "bus",
            "warn",
            detail + " — large backlog; are readers draining their mailboxes?",
            remediation="check that registered sessions call graph_bus action=receive/ack",
            data=data,
        )
    return _result("bus", "ok", detail, data=data)


def _check_skills() -> dict[str, Any]:
    """Report whether the agent-utilities skill toolkit is installed in the XDG dir.

    CONCEPT:OS-5.52 — the agent factory auto-loads every ``SKILL.md`` under
    ``core.paths.skills_dir()``; the ``agent-utilities`` skill-graph + AU skills are
    what unlock how to use the platform. If they're absent, point at the one command
    that installs them.
    """
    try:
        from agent_utilities.core.paths import skills_dir

        sdir = skills_dir()
        if not sdir.exists():
            installed = 0
        else:
            installed = sum(1 for _ in sdir.rglob("SKILL.md"))
    except Exception as exc:  # noqa: BLE001
        return _result("skills", "skip", f"skills dir probe unavailable: {exc}")

    if installed == 0:
        return _result(
            "skills",
            "warn",
            "no skills installed in the agent-utilities skills dir — "
            "the platform skill-graph + AU skills are not loaded",
            remediation="`agent-utilities install-skills` (installs the toolkit incl. the agent-utilities skill-graph)",
            skill="agent-utilities",
            data={"skills_dir": str(skills_dir()), "installed": 0},
        )
    return _result(
        "skills",
        "ok",
        f"{installed} skill file(s) installed in the agent-utilities skills dir",
        data={"installed": installed},
    )


def _check_unified_install() -> dict[str, Any]:
    """Assert the unified XDG tree exists and matches installed providers (CONCEPT:OS-5.79).

    ``agent-utilities install`` materializes every provider contribution (skills +
    prompts + ontologies, incl. the hub's OWN under ``agent-utilities``) into one XDG
    data tree the runtime reads from. This flags providers that are installed but NOT
    materialized (a stale/absent tree the ``install`` command fixes).
    """
    try:
        from agent_utilities.core.paths import ontology_dir, skills_dir
        from agent_utilities.core.providers import (
            ONTOLOGY_PROVIDER_GROUP,
            PROMPT_PROVIDER_GROUP,
            SKILL_PROVIDER_GROUP,
            iter_provider_dirs,
        )
        from agent_utilities.core.unified_install import (
            OWN_PROVIDER,
            unified_prompts_dir,
        )
    except Exception as exc:  # noqa: BLE001
        return _result(
            "unified_install", "skip", f"unified-install probe unavailable: {exc}"
        )

    legs = {
        "skills": (SKILL_PROVIDER_GROUP, skills_dir()),
        "prompts": (PROMPT_PROVIDER_GROUP, unified_prompts_dir()),
        "ontologies": (ONTOLOGY_PROVIDER_GROUP, ontology_dir()),
    }
    # Expected providers per leg = live entry-point providers + the hub's own mirror.
    expected: dict[str, set[str]] = {}
    missing: list[str] = []
    materialized = 0
    for leg, (group, root) in legs.items():
        names = {name for name, _src in iter_provider_dirs(group)}
        names.add(OWN_PROVIDER)
        expected[leg] = names
        for name in sorted(names):
            if (root / name).is_dir():
                materialized += 1
            else:
                missing.append(f"{leg}:{name}")

    data = {
        "roots": {leg: str(root) for leg, (_g, root) in legs.items()},
        "expected": {leg: sorted(n) for leg, n in expected.items()},
        "missing": missing,
        "materialized": materialized,
    }
    if missing:
        return _result(
            "unified_install",
            "warn",
            f"{len(missing)} provider contribution(s) installed but not materialized "
            f"in the unified XDG tree: {', '.join(missing)}",
            remediation="`agent-utilities install` (materializes skills+prompts+ontologies into the XDG tree)",
            skill="agent-utilities",
            data=data,
        )
    return _result(
        "unified_install",
        "ok",
        f"unified XDG tree complete — {materialized} provider contribution(s) materialized",
        data=data,
    )


def _check_warm_fork() -> dict[str, Any]:
    """Report which warm-fork sandbox rungs are available on this host (CONCEPT:OS-5.59).

    The forkserver rung (os.fork, zero infra) should be up on any Unix host incl. ARM; wasm
    needs wasmtime + a payload; docker needs a daemon; firecracker needs KVM + forkd. Also
    reports how many warm parents are currently pooled.
    """
    rungs: dict[str, dict[str, Any]] = {}
    try:
        from agent_utilities.rlm.sandboxes.registry import default_sandboxes

        for b in default_sandboxes():
            caps = b.capabilities
            try:
                available = bool(b.is_available())
            except Exception:  # noqa: BLE001 - a probe must never crash the doctor
                available = False
            rungs[b.name] = {
                "available": available,
                "isolated": caps.isolated,
                "warm_fork": caps.warm_fork,
                "rank": caps.preference_rank,
            }
    except Exception as exc:  # noqa: BLE001
        return _result(
            "warm_fork", "error", f"could not enumerate sandbox rungs: {exc}"
        )

    try:
        from agent_utilities.runtime.warm_registry import WarmParentRegistry

        pool = WarmParentRegistry.get().stats()
    except Exception:  # noqa: BLE001
        pool = {}

    warm_rungs = sorted(
        n for n, r in rungs.items() if r["warm_fork"] and r["available"]
    )
    data = {"rungs": rungs, "warm_rungs": warm_rungs, "pool": pool}
    if warm_rungs:
        return _result(
            "warm_fork",
            "ok",
            f"native warm-fork available via: {', '.join(warm_rungs)}",
            data=data,
        )
    return _result(
        "warm_fork",
        "warn",
        "no warm-fork rung available — sandboxes will cold-start every run",
        remediation=(
            "forkserver needs a Unix host (os.fork); install the 'sandbox' extra for wasm "
            "(wasmtime), or run a docker/podman daemon, to enable a warm-fork tier."
        ),
        data=data,
    )


# Registry: name -> callable. Order is the report order.
CHECKS: dict[str, Callable[..., dict[str, Any]]] = {
    "python_env": _check_python_env,
    "config": _check_config,
    "workspace_config": _check_workspace_config,
    "engine": _check_engine,
    "graph_backend": _check_graph_backend,
    "graph_connections": _check_graph_connections,
    "ingestion_coverage": _check_ingestion_coverage,
    "connector_coverage": _check_connector_coverage,
    "secrets": _check_secrets,
    "auth": _check_auth,
    "mcp_fleet": _check_mcp_fleet,
    "hooks": _check_hooks,
    "observability": _check_observability,
    "bus": _check_bus,
    "skills": _check_skills,
    "unified_install": _check_unified_install,
    "warm_fork": _check_warm_fork,
}


def _auto_fix(name: str) -> dict[str, Any]:
    """Run a conservative, idempotent remediation for an auto-fixable check."""
    if name == "hooks":
        try:
            from agent_utilities.ecosystem.hook_installer import HookInstaller

            inst = HookInstaller()
            inst.install()
            return {
                "fixed": name,
                "result": "re-installed hooks",
                "errors": inst.errors,
            }
        except Exception as exc:  # noqa: BLE001
            return {"fixed": name, "error": str(exc)}
    return {"fixed": name, "result": "no auto-fix available"}


def run_doctor(
    only: list[str] | None = None, *, fix: bool = False, live: bool = False
) -> dict[str, Any]:
    """Run the health sweep and return a structured report.

    Args:
        only: restrict to these check names (default: all).
        fix: run conservative auto-remediations for ``auto_fixable`` checks, then
            re-run those checks.
        live: let network-touching checks (MCP fleet) probe endpoints.
    """
    names = only or list(CHECKS)
    results: list[dict[str, Any]] = []
    for name in names:
        fn = CHECKS.get(name)
        if fn is None:
            continue
        try:
            res = fn(live=live) if name == "mcp_fleet" else fn()
        except Exception as exc:  # noqa: BLE001 — a check must never crash the doctor
            res = _result(name, "error", f"check raised: {exc}")
        results.append(res)

    fixes: list[dict[str, Any]] = []
    if fix:
        for res in results:
            if res["status"] in ("warn", "fail") and res.get("auto_fixable"):
                fixes.append(_auto_fix(res["name"]))
                try:
                    res.update(CHECKS[res["name"]]())  # re-run after fix
                except Exception:  # noqa: BLE001
                    pass

    worst = max((_RANK[r["status"]] for r in results), default=0)
    overall = {0: "healthy", 1: "warnings", 2: "unhealthy"}[worst]
    counts: dict[str, int] = {}
    for r in results:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    return {
        "status": overall,
        "counts": counts,
        "checks": results,
        "fixes": fixes,
        "summary": _summarize(overall, results),
    }


def _summarize(overall: str, results: list[dict[str, Any]]) -> str:
    bad = [r["name"] for r in results if r["status"] in ("warn", "fail", "error")]
    if overall == "healthy":
        return "All checks passed."
    return f"{overall}: attend to {bad}. Each failing check lists a remediation/skill."


def main(argv: list[str] | None = None) -> int:
    """``agent-utilities-doctor`` console entry."""
    import argparse
    import json

    parser = argparse.ArgumentParser(
        prog="agent-utilities-doctor",
        description="Holistic health sweep of an agent-utilities deployment.",
    )
    parser.add_argument("--only", nargs="*", choices=list(CHECKS), default=None)
    parser.add_argument(
        "--fix", action="store_true", help="Run safe auto-remediations."
    )
    parser.add_argument(
        "--live", action="store_true", help="Probe network endpoints (MCP fleet)."
    )
    parser.add_argument(
        "--json", action="store_true", help="Emit JSON instead of text."
    )
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Run the host DEPENDENCY preflight (runtimes/tools) instead of the deployment sweep.",
    )
    parser.add_argument(
        "--profile",
        default="tiny",
        help="Deployment profile for --preflight (tiny | single-node-prod | enterprise).",
    )
    parser.add_argument(
        "--component",
        dest="components",
        action="append",
        default=None,
        help="UI component to preflight (repeatable): agent-webui | geniusbot | agent-terminal-ui.",
    )
    args = parser.parse_args(argv)

    if args.preflight:
        from .preflight import run_preflight

        report = run_preflight(args.profile, args.components)
        if args.json:
            print(json.dumps(report, indent=2, default=str))
        else:
            _print_human(report, title="agent-utilities preflight")
        return 0 if report["status"] != "blocked" else 1

    report = run_doctor(args.only, fix=args.fix, live=args.live)
    if args.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        _print_human(report)
    return 0 if report["status"] != "unhealthy" else 1


def _print_human(report: dict[str, Any], title: str = "agent-utilities doctor") -> None:
    glyph = {"ok": "✓", "warn": "!", "fail": "✗", "error": "✗", "skip": "·"}
    print(f"{title} — {report['status'].upper()}\n")
    for r in report["checks"]:
        line = f"  {glyph.get(r['status'], '?')} {r['name']:<14} {r['detail']}"
        print(line)
        if r["status"] in ("warn", "fail", "error"):
            if r.get("remediation"):
                print(f"      → fix: {r['remediation']}")
            if r.get("skill"):
                print(f"      → skill: {r['skill']}")
    print(f"\n{report['summary']}")


if __name__ == "__main__":  # pragma: no cover
    import sys

    sys.exit(main())
