#!/usr/bin/python
from __future__ import annotations

"""Signed hydration manifest — absent-vs-hidden resolution.

CONCEPT:AU-KG.audit.hydration-manifest-signed

BACKGROUND
----------
A live authenticated query against a running graph-os can return ``Server=0``
for a class that the declared universe (``deploy/mcp-fleet.registry.yml``) says
should have 62 entries. The engine enforces "RLS default-deny: rows require
explicit public visibility or an owner grant" (its own startup banner — see
:mod:`agent_utilities.knowledge_graph.maintenance.graph_ownership`). So a bare
``0`` is ambiguous: it means EITHER "never ingested" OR "ingested but invisible
to this principal/tenant" — and nobody can tell which from the count alone.
Every hydration-status claim built on that count is unfalsifiable until this
ambiguity is resolved.

WHAT THIS MODULE DOES
----------------------
For every registered source class it fuses, per :class:`SourceClassCoverage`:

* **expected vs actual** — ``expected`` from the declared universe where one
  genuinely exists (``deploy/mcp-fleet.registry.yml`` for ``Server``, the
  deduped skill-provider resolution for ``Skill``, ``workspace.yml`` for the
  ``Code`` repo-coverage rollup); ``actual`` from the live graph, read TWICE
  under two distinct identities (see below). Classes with no trustworthy
  declared universe report ``expected=None`` with an honest reason rather than
  a fabricated number.
* **absent vs hidden — the core distinction.** ``actual_serving`` is counted
  through a :class:`GraphReader` bound to the SAME identity a real MCP/REST
  caller gets (the per-request ``GraphSession``/RLS grant) — this is the
  number a live authenticated query returns today. ``actual_service`` is
  counted through a *second*, independent :class:`GraphReader` bound to
  graph-os's OWN trusted server-process engine handle
  (:func:`resolve_service_authority_reader` —
  ``IntelligenceGraphEngine.get_active()``, the SAME internal handle boot
  hydration and ``DeltaManifest`` already use for maintenance work). This
  module never synthesizes or widens authority to get that second reading —
  when no server-owned engine handle is active in-process (e.g. building the
  manifest from a standalone script with no live engine), ``actual_service`` is
  honestly ``None`` and the divergence check degrades to "cannot rule out a
  visibility gap" rather than a false "hidden" or a false "absent".
  A **divergence** — ``actual_serving == 0`` while ``actual_service > 0`` —
  proves an RLS visibility issue, never a hydration failure, and is reported
  as ``blocked``, not ``not_started``.
* **digests** — a content digest of whatever declared-universe input produced
  ``expected`` (so the manifest is tamper-evident against that input too, not
  only against itself), plus the au package version and the ambient
  ``GraphSession.catalog_epoch``/``policy_version`` when one is bound.
* **errors** — per-class failure, classified by WHICH stage failed (expected
  resolution / serving read / service read) and the exception type name, never
  collapsed into one opaque total.
* **freshness** — the boot hydration plan's own ``HydrationPlanStep`` record
  (CONCEPT:AU-KG.audit.hydration-manifest-signed builds ON
  ``agent_utilities.mcp.kg_server._run_boot_hydration_plan``, never replaces
  it) or the ``DeltaManifest`` watermark for classes it tracks.
* **completeness verdict** — ``complete`` / ``partial`` / ``not_started`` /
  ``blocked`` with a human-readable reason (:func:`_fuse_verdict`).

SIGNING
-------
The manifest reuses the repo's EXISTING Ed25519 release-signing authority
(:mod:`agent_utilities.knowledge_graph.ontology.ontology_integrity` —
``ReleaseSigner``/``verify_release_signature``, the same mechanism
``scripts/generate_native_connector_manifest.py`` already uses to sign provider
manifests off ``env://ONTOLOGY_RELEASE_SIGNING_PRIVATE_KEY``). No second
signing scheme is invented here; :func:`sign_hydration_manifest` /
:func:`verify_hydration_manifest` are thin call-throughs.
"""

import hashlib
import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal

from agent_utilities.security.identifiers import validate_identifier

logger = logging.getLogger(__name__)

__all__ = [
    "GraphReader",
    "GraphReaderError",
    "SourceClassCoverage",
    "HydrationManifest",
    "SOURCE_CLASSES",
    "resolve_serving_reader",
    "resolve_service_authority_reader",
    "build_hydration_manifest",
    "sign_hydration_manifest",
    "verify_hydration_manifest",
    "persist_hydration_manifest",
]

Verdict = Literal["complete", "partial", "not_started", "blocked"]
QueryFn = Callable[[str], Any]


class GraphReaderError(RuntimeError):
    """A read against the graph could not be completed."""


@dataclass(frozen=True)
class GraphReader:
    """A ``query_cypher``-shaped read bound to ONE identity.

    Two instances bound to two DIFFERENT identities are what let
    :func:`build_hydration_manifest` distinguish absent from hidden — see the
    module docstring. ``authority`` is a label for provenance only; it carries
    no privilege of its own.
    """

    query: QueryFn
    authority: Literal["serving", "service"]

    def count_label(self, node_label: str) -> int:
        safe = validate_identifier(node_label, kind="label")
        try:
            rows = self.query(f"MATCH (n:{safe}) RETURN count(n) AS c")
        except Exception as exc:
            raise GraphReaderError(
                f"count query failed for label {node_label!r}"
            ) from exc
        return _first_int(rows, node_label)

    def hydration_step(self, step_name: str) -> dict[str, Any] | None:
        """Read the boot hydration plan's own record for ``step_name``, if any.

        Mirrors ``agent_utilities.mcp.kg_server._record_boot_hydration_step``'s
        stable node id (``boot-hydration:<name>``) exactly — this module reads
        that plan's output, it does not create a second one.
        """
        node_id = f"boot-hydration:{step_name}"
        try:
            rows = self.query(
                "MATCH (n:HydrationPlanStep {id: "
                + json.dumps(node_id)
                + "}) RETURN n.status AS status, n.updated_at AS updated_at"
            )
        except Exception as exc:
            raise GraphReaderError(
                f"hydration-step query failed for {step_name!r}"
            ) from exc
        if not rows:
            return None
        row = rows[0]
        if not isinstance(row, dict):
            return None
        return {"status": row.get("status"), "updated_at": row.get("updated_at")}


def _first_int(rows: Any, label: str) -> int:
    if not rows:
        return 0
    row = rows[0]
    if isinstance(row, dict):
        for value in row.values():
            try:
                return int(value)
            except (TypeError, ValueError):
                continue
        # No int-convertible value in the row. This is the SAME unexpected shape
        # the tuple/list branch below raises on, and this module must never
        # fabricate a count: returning 0 here would make "the query came back
        # malformed" indistinguishable from "this label genuinely has 0 nodes" —
        # exactly the absent-vs-hidden ambiguity the manifest exists to resolve.
        raise GraphReaderError(
            f"count query returned a row with no integer value for label {label!r}"
        )
    try:
        return int(row[0])
    except (TypeError, ValueError, IndexError, KeyError) as exc:
        raise GraphReaderError(
            f"count query returned an unexpected row shape for label {label!r}"
        ) from exc


# ---------------------------------------------------------------------------
# The two readers: serving principal vs. server-owned service authority
# ---------------------------------------------------------------------------


def resolve_serving_reader(engine: Any) -> GraphReader | None:
    """The reader bound to whatever identity ``engine`` is already scoped to.

    ``engine`` is the SAME live engine handle a served MCP/REST call already
    has (e.g. the one bound to the caller's ``GraphSession``) — this function
    adds no authority of its own, it only wraps ``engine.query_cypher``.
    """
    query_cypher = getattr(engine, "query_cypher", None)
    if not callable(query_cypher):
        return None
    return GraphReader(query=query_cypher, authority="serving")


def resolve_service_authority_reader() -> GraphReader | None:
    """The graph-os SERVER's own background authority, if one is captured.

    IDENTITY COMES FROM THE AMBIENT ``GraphSession``, NOT FROM THE ENGINE
    OBJECT. Every engine operation resolves its principal/tenant/RLS grant from
    the task-local :func:`~..core.session.current_session` immediately before
    the native client signs the frame (see ``GraphComputeClient._send``, which
    raises ``SessionRequiredError`` when none is bound). So wrapping a second
    engine handle changes NOTHING about who the read runs as — a second reader
    is only genuinely a second authority if it BINDS a different session for
    the duration of its call.

    This therefore binds the process's own verified background session
    (``_background_worker_session``, captured by
    :meth:`~..core.engine_tasks.TaskManagerMixin.start_background_daemons` — the
    same authority boot hydration and the ingestion workers already run under),
    and returns ``None`` — never a fabricated elevated read — when:

    * no server-owned engine is active in this process (a standalone script, or
      before boot activated one); or
    * that engine captured no background session (daemons never started); or
    * the ambient serving session IS that background session, in which case a
      "second" read would be the same identity and reporting it as an
      independent confirmation would be a lie.

    When this returns ``None`` the manifest honestly records
    ``service_authority_attempted=False`` and :func:`_fuse_verdict` degrades to
    "an RLS visibility gap cannot be ruled out" instead of claiming a
    divergence check that never happened.
    """
    try:
        from agent_utilities.knowledge_graph.core.engine import (
            IntelligenceGraphEngine,
        )
        from agent_utilities.knowledge_graph.core.session import (
            GraphSession,
            current_session,
            use_session,
        )

        engine = IntelligenceGraphEngine.get_active()
    except Exception as exc:  # noqa: BLE001 - best-effort; never fabricate authority
        logger.debug("service-authority engine unavailable: %s", exc)
        return None
    if engine is None:
        return None
    query_cypher = getattr(engine, "query_cypher", None)
    if not callable(query_cypher):
        return None

    service_session = getattr(engine, "_background_worker_session", None)
    if not isinstance(service_session, GraphSession):
        logger.debug(
            "service-authority reader unavailable: the active engine has no "
            "captured background session, so a second read would run under the "
            "SAME ambient identity as the serving read"
        )
        return None
    if current_session() is service_session:
        logger.debug(
            "service-authority reader unavailable: the ambient serving session "
            "IS the server's background session, so the two reads cannot differ"
        )
        return None

    def _query(cypher: str) -> Any:
        with use_session(service_session):
            return query_cypher(cypher)

    return GraphReader(query=_query, authority="service")


# ---------------------------------------------------------------------------
# Declared-universe "expected" resolution — best-effort, never raises
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExpectedResult:
    count: int | None
    basis: str | None
    digest: str | None
    error: str | None = None


def _expected_servers() -> ExpectedResult:
    """``Server`` universe: ``deploy/mcp-fleet.registry.yml`` ``services:``."""
    try:
        import yaml

        from agent_utilities.orchestration.fleet_reconciler import (
            resolve_registry_path,
        )

        path = resolve_registry_path()
        if path is None:
            return ExpectedResult(None, None, None, "mcp-fleet.registry.yml not found")
        raw = path.read_bytes()
        data = yaml.safe_load(raw) or {}
        services = [
            s
            for s in (data.get("services") or [])
            if isinstance(s, dict) and s.get("name")
        ]
        digest = hashlib.sha256(raw).hexdigest()
        return ExpectedResult(len(services), f"{path} services[]", digest)
    except Exception as exc:  # noqa: BLE001 - expected-count resolution is best-effort
        return ExpectedResult(None, None, None, f"{type(exc).__name__}: {exc}")


def _expected_skills() -> ExpectedResult:
    """``Skill`` universe: the deduped skill-provider resolution.

    ``resolve_skill_provider_dirs`` already returns exactly one verified root
    per globally unique current skill (dedup + collision handling live there),
    so its length IS the expected count — no re-derivation needed.
    """
    try:
        from agent_utilities.core.providers import (
            SKILL_PROVIDER_GROUP,
            provider_registry_digest,
            resolve_skill_provider_dirs,
        )

        resolved = resolve_skill_provider_dirs()
        digest = provider_registry_digest(SKILL_PROVIDER_GROUP)
        return ExpectedResult(
            len(resolved), "resolve_skill_provider_dirs() deduped skill count", digest
        )
    except Exception as exc:  # noqa: BLE001 - expected-count resolution is best-effort
        return ExpectedResult(None, None, None, f"{type(exc).__name__}: {exc}")


def _expected_code_repos() -> ExpectedResult:
    """``Code`` universe: the ``agent-packages`` subtree of ``workspace.yml``.

    Reuses :mod:`agent_utilities.knowledge_graph.ingestion.coverage` (the
    existing ``ingestion_coverage`` doctor check) rather than re-deriving a
    repo list. This class's "expected"/"actual" are a repo-coverage rollup
    (how many declared repos have >=1 ``:Code`` symbol), NOT a raw symbol
    count — see :func:`_actual_code_coverage`.
    """
    try:
        from agent_utilities.knowledge_graph.ingestion.coverage import (
            enumerate_agent_packages_repos,
            find_workspace_manifest,
        )

        manifest = find_workspace_manifest()
        if manifest is None:
            return ExpectedResult(None, None, None, "workspace.yml not found")
        repos = enumerate_agent_packages_repos(manifest)
        digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
        return ExpectedResult(
            len(repos), f"{manifest} agent-packages repositories[]", digest
        )
    except Exception as exc:  # noqa: BLE001 - expected-count resolution is best-effort
        return ExpectedResult(None, None, None, f"{type(exc).__name__}: {exc}")


def _actual_code_coverage(
    serving: GraphReader | None, service: GraphReader | None
) -> tuple[int | None, str | None, int | None, str | None]:
    """Repo-coverage rollup for ``Code``: count of declared repos with >=1 symbol.

    Returns ``(actual_serving, serving_error, actual_service, service_error)``.

    D-28 fix: ``coverage.repo_symbol_counts`` now reports a per-repo query
    failure separately from a genuine zero count (it returns
    ``(counts, errors)`` rather than folding a failure into ``0``). When any
    repo's query fails under a reader, that reader's rollup is honestly
    reported as ``None`` with a ``*_error`` message — never as a lower-than-
    real count — so :func:`_fuse_verdict` correctly classifies it ``blocked``
    ("the serving-principal read failed; completeness cannot be assessed")
    instead of silently under-reporting it as ``not_started``.
    """
    try:
        from agent_utilities.knowledge_graph.ingestion.coverage import (
            enumerate_agent_packages_repos,
            find_workspace_manifest,
            repo_symbol_counts,
        )

        manifest = find_workspace_manifest()
        if manifest is None:
            return None, "workspace.yml not found", None, None
        repos = enumerate_agent_packages_repos(manifest)
    except Exception as exc:  # noqa: BLE001 - honestly surfaced as a serving-side error
        return None, f"{type(exc).__name__}: {exc}", None, None

    def _covered(reader: GraphReader | None) -> tuple[int | None, str | None]:
        if reader is None:
            return None, None
        try:
            counts, count_errors = repo_symbol_counts(_BackendAdapter(reader), repos)
        except Exception as exc:  # noqa: BLE001 - one reader's failure isolated
            return None, f"{type(exc).__name__}: {exc}"
        if count_errors:
            # A per-repo query failure must never masquerade as a lower (but
            # real) count — report the reader's rollup as unavailable so
            # _fuse_verdict classifies it "blocked", not "not_started" (D-28).
            sample = next(iter(count_errors.values()))
            return (
                None,
                f"{len(count_errors)}/{len(repos)} repo symbol-count "
                f"quer{'y' if len(count_errors) == 1 else 'ies'} failed ({sample})",
            )
        return sum(1 for n in counts.values() if n > 0), None

    serving_count, serving_error = _covered(serving)
    service_count, service_error = _covered(service)
    return serving_count, serving_error, service_count, service_error


class _BackendAdapter:
    """Adapts a :class:`GraphReader` to the ``backend.execute(cypher, params)``
    shape :func:`repo_symbol_counts` expects, without giving that module a
    second query mechanism to maintain."""

    def __init__(self, reader: GraphReader) -> None:
        self._reader = reader

    def execute(self, cypher: str, params: dict[str, Any]) -> list[dict[str, Any]]:
        needle = params.get("needle", "")
        rendered = cypher.replace("$needle", json.dumps(needle))
        rows = self._reader.query(rendered)
        return rows if isinstance(rows, list) else []


# ---------------------------------------------------------------------------
# Freshness — reuse the boot hydration plan + DeltaManifest, never re-derive
# ---------------------------------------------------------------------------

#: Maps a source class to the boot-hydration-plan step name that last hydrated
#: it (CONCEPT:AU-KG.audit.hydration-manifest-signed builds on
#: ``kg_server._run_boot_hydration_plan``'s existing 4-priority steps).
_CLASS_HYDRATION_STEP: dict[str, str] = {
    "Server": "fleet_tool_schemas",
    "CallableResource": "graphos_tool_surface",
    "Skill": "capabilities",
    "Prompt": "prompts",
    "Ontology": "ontologies",
    "Code": "code_and_connectors",
}


def _resolve_freshness(
    source_class: str, serving: GraphReader | None
) -> tuple[str | None, float | None, str | None]:
    """Return ``(last_synced_at, staleness_days, basis)`` for ``source_class``."""
    step = _CLASS_HYDRATION_STEP.get(source_class)
    if step is None or serving is None:
        return None, None, None
    try:
        record = serving.hydration_step(step)
    except GraphReaderError as exc:
        logger.debug("freshness read failed for %s: %s", source_class, exc)
        return None, None, None
    if not record or not record.get("updated_at"):
        return None, None, None
    updated_at = str(record["updated_at"])
    staleness = _age_days(updated_at)
    basis = f"boot-hydration:{step} (status={record.get('status', 'unknown')})"
    return updated_at, staleness, basis


def _age_days(updated_at: str) -> float | None:
    try:
        ts = datetime.fromisoformat(updated_at.replace("Z", "+00:00"))
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=UTC)
        return round((datetime.now(UTC) - ts).total_seconds() / 86400.0, 2)
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Source-class registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SourceClassSpec:
    name: str
    label: str
    expected_fn: Callable[[], ExpectedResult] | None = None
    #: Classes whose "actual" isn't a plain label count (currently only
    #: ``Code``, a repo-coverage rollup) set this instead of relying on the
    #: generic ``GraphReader.count_label`` path.
    actual_override: (
        Callable[
            [GraphReader | None, GraphReader | None],
            tuple[int | None, str | None, int | None, str | None],
        ]
        | None
    ) = None


#: The registered source classes this manifest reports on. Adding a class here
#: is the ONE place a new class needs to be declared; classes with no
#: ``expected_fn`` are exactly the ones the honest "could not compute a
#: trustworthy expected count" list (see the module report) names.
SOURCE_CLASSES: tuple[SourceClassSpec, ...] = (
    SourceClassSpec("Server", "Server", _expected_servers),
    SourceClassSpec("Skill", "Skill", _expected_skills),
    SourceClassSpec(
        "Code", "Code", _expected_code_repos, actual_override=_actual_code_coverage
    ),
    SourceClassSpec("CallableResource", "CallableResource"),
    SourceClassSpec("Prompt", "Prompt"),
    SourceClassSpec("Concept", "Concept"),
    SourceClassSpec("RunTrace", "RunTrace"),
    SourceClassSpec("ToolCall", "ToolCall"),
    SourceClassSpec("WorkItem", "WorkItem"),
    SourceClassSpec("HydrationPlanStep", "HydrationPlanStep"),
    SourceClassSpec("Ontology", "Ontology"),
    SourceClassSpec("IngestManifest", "IngestManifest"),
)


# ---------------------------------------------------------------------------
# Per-class coverage + verdict fusion
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SourceClassCoverage:
    source_class: str
    expected: int | None
    expected_basis: str | None
    expected_digest: str | None
    actual_serving: int | None
    actual_service: int | None
    service_authority_attempted: bool
    last_synced_at: str | None
    staleness_days: float | None
    freshness_basis: str | None
    errors: dict[str, str]
    verdict: Verdict
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_class": self.source_class,
            "expected": self.expected,
            "expected_basis": self.expected_basis,
            "expected_digest": self.expected_digest,
            "actual_serving": self.actual_serving,
            "actual_service": self.actual_service,
            "service_authority_attempted": self.service_authority_attempted,
            "last_synced_at": self.last_synced_at,
            "staleness_days": self.staleness_days,
            "freshness_basis": self.freshness_basis,
            "errors": dict(self.errors),
            "verdict": self.verdict,
            "reason": self.reason,
        }


def _fuse_verdict(
    expected: int | None,
    actual_serving: int | None,
    actual_service: int | None,
    service_attempted: bool,
) -> tuple[Verdict, str]:
    """The absent-vs-hidden decision rule (CONCEPT:AU-KG.audit.hydration-absent-vs-hidden).

    A divergence — 0 under the serving principal, >0 under service authority
    — is ALWAYS ``blocked`` (an RLS visibility issue), never ``not_started``,
    regardless of whether ``expected`` is known.
    """
    if actual_serving is None:
        return (
            "blocked",
            "the serving-principal read failed; completeness cannot be assessed",
        )

    if (
        service_attempted
        and actual_service is not None
        and actual_serving == 0
        and actual_service > 0
    ):
        return (
            "blocked",
            f"0 under the serving principal but {actual_service} under service "
            "authority — an RLS visibility gap, not a hydration failure",
        )

    if expected is None:
        if actual_serving > 0:
            return (
                "complete",
                "no declared-universe count is available for this class; nodes "
                "are present under the serving principal",
            )
        if service_attempted and actual_service == 0:
            return (
                "not_started",
                "0 under both the serving principal and service authority; "
                "genuinely never ingested",
            )
        return (
            "not_started",
            "0 under the serving principal; no declared universe exists to size "
            "an expected count, and service-authority confirmation was "
            + ("unavailable" if not service_attempted else "inconclusive"),
        )

    if expected == 0:
        return "complete", "the declared universe for this class is empty"

    if actual_serving >= expected:
        return (
            "complete",
            f"{actual_serving}/{expected} present under the serving principal",
        )

    if actual_serving == 0:
        if service_attempted and actual_service == 0:
            return (
                "not_started",
                f"expected {expected}, 0 under both the serving principal and "
                "service authority; never ingested",
            )
        return (
            "blocked",
            f"expected {expected}, 0 under the serving principal; "
            + (
                "service-authority read failed too"
                if service_attempted
                else "no service-authority read was available"
            )
            + " — an RLS visibility gap cannot be ruled out",
        )

    return "partial", f"{actual_serving}/{expected} present under the serving principal"


def _build_class_coverage(
    spec: SourceClassSpec,
    serving: GraphReader | None,
    service: GraphReader | None,
) -> SourceClassCoverage:
    errors: dict[str, str] = {}

    expected_result = (
        spec.expected_fn()
        if spec.expected_fn is not None
        else ExpectedResult(None, None, None)
    )
    if expected_result.error:
        errors["expected_resolution"] = expected_result.error

    if spec.actual_override is not None:
        actual_serving, serving_error, actual_service, service_error = (
            spec.actual_override(serving, service)
        )
    else:
        actual_serving, serving_error = _safe_count(serving, spec.label)
        actual_service, service_error = _safe_count(service, spec.label)
    if serving_error:
        errors["serving_read"] = serving_error
    if service_error:
        errors["service_read"] = service_error

    last_synced_at, staleness_days, freshness_basis = _resolve_freshness(
        spec.name, serving
    )

    verdict, reason = _fuse_verdict(
        expected_result.count, actual_serving, actual_service, service is not None
    )

    return SourceClassCoverage(
        source_class=spec.name,
        expected=expected_result.count,
        expected_basis=expected_result.basis,
        expected_digest=expected_result.digest,
        actual_serving=actual_serving,
        actual_service=actual_service,
        service_authority_attempted=service is not None,
        last_synced_at=last_synced_at,
        staleness_days=staleness_days,
        freshness_basis=freshness_basis,
        errors=errors,
        verdict=verdict,
        reason=reason,
    )


def _safe_count(
    reader: GraphReader | None, label: str
) -> tuple[int | None, str | None]:
    if reader is None:
        return None, None
    try:
        return reader.count_label(label), None
    except GraphReaderError as exc:
        return None, f"{type(exc.__cause__ or exc).__name__}: {exc}"


# ---------------------------------------------------------------------------
# The manifest itself
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HydrationManifest:
    generated_at: str
    tenant_id: str | None
    principal: str | None
    actor_type: str | None
    scopes: tuple[str, ...]
    graph: str | None
    policy_version: str | None
    catalog_epoch: int | None
    rls_posture: str
    service_authority_available: bool
    au_version: str | None
    generator: str
    coverage: tuple[SourceClassCoverage, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "tenant_id": self.tenant_id,
            "principal": self.principal,
            "actor_type": self.actor_type,
            "scopes": sorted(self.scopes),
            "graph": self.graph,
            "policy_version": self.policy_version,
            "catalog_epoch": self.catalog_epoch,
            "rls_posture": self.rls_posture,
            "service_authority_available": self.service_authority_available,
            "au_version": self.au_version,
            "generator": self.generator,
            "coverage": [c.to_dict() for c in self.coverage],
        }

    @property
    def error_summary(self) -> dict[str, int]:
        """Roll up every per-class error by ``(stage, exception_type)`` — a
        classification, not a single opaque total (see the module docstring)."""
        summary: dict[str, int] = {}
        for c in self.coverage:
            for stage, message in c.errors.items():
                exc_type = message.split(":", 1)[0].strip() or "Unknown"
                key = f"{stage}:{exc_type}"
                summary[key] = summary.get(key, 0) + 1
        return summary


_RLS_POSTURE = (
    "default-deny: rows require explicit public visibility or an owner grant "
    "(engine startup banner; see graph_ownership.py)"
)


def build_hydration_manifest(
    *,
    serving: GraphReader | None = None,
    service: GraphReader | None = None,
) -> HydrationManifest:
    """Build the full manifest: one :class:`SourceClassCoverage` per registered
    source class, fused from the declared universe + both readers.

    ``serving``/``service`` default to :func:`resolve_serving_reader` bound to
    the currently ambient engine and :func:`resolve_service_authority_reader`
    respectively — pass explicit fixtures in tests.
    """
    if serving is None:
        try:
            from agent_utilities.knowledge_graph.core.engine import (
                IntelligenceGraphEngine,
            )

            engine = IntelligenceGraphEngine.get_active()
        except Exception:  # noqa: BLE001 - best-effort ambient resolution
            engine = None
        serving = resolve_serving_reader(engine) if engine is not None else None
    if service is None:
        service = resolve_service_authority_reader()

    tenant_id = principal = actor_type = graph_name = None
    scopes: tuple[str, ...] = ()
    policy_version: str | None = None
    catalog_epoch: int | None = None
    try:
        from agent_utilities.knowledge_graph.core.session import current_session

        session = current_session()
        if session is not None:
            tenant_id = session.tenant or None
            principal = getattr(session.actor, "actor_id", None)
            actor_type = getattr(
                getattr(session.actor, "actor_type", None), "name", None
            )
            scopes = tuple(sorted(session.scopes))
            graph_name = session.graph or None
            policy_version = str(session.policy_version) or None
            catalog_epoch = session.catalog_epoch
    except Exception as exc:  # noqa: BLE001 - identity capture is best-effort
        logger.debug("hydration manifest: no ambient session (%s)", exc)

    au_version: str | None
    try:
        from agent_utilities._version import __version__ as au_version_value

        au_version = au_version_value
    except Exception:  # noqa: BLE001 - au_version is provenance metadata only; an
        # unimportable _version must leave the field honestly absent (None), never
        # block manifest construction and never invent a version string.
        au_version = None

    coverage = tuple(
        _build_class_coverage(spec, serving, service) for spec in SOURCE_CLASSES
    )

    return HydrationManifest(
        generated_at=datetime.now(UTC).isoformat(),
        tenant_id=tenant_id,
        principal=principal,
        actor_type=actor_type,
        scopes=scopes,
        graph=graph_name,
        policy_version=policy_version,
        catalog_epoch=catalog_epoch,
        rls_posture=_RLS_POSTURE,
        service_authority_available=service is not None,
        au_version=au_version,
        generator="agent_utilities.knowledge_graph.ingestion.hydration_manifest",
        coverage=coverage,
    )


# ---------------------------------------------------------------------------
# Signing — reuses the repo's existing Ed25519 release-signing authority
# ---------------------------------------------------------------------------


def sign_hydration_manifest(
    manifest: HydrationManifest,
    *,
    signer: Any | None = None,
) -> dict[str, Any]:
    """Sign ``manifest`` with the SAME Ed25519 release authority provider
    manifests already use (``ontology_integrity.ReleaseSigner``). Returns the
    full JSON-serializable, signed document; never mutates a durable key or
    prints/persists key material."""
    from agent_utilities.knowledge_graph.ontology import ontology_integrity

    document = manifest.to_dict()
    document["signature"] = None
    document["signer"] = None
    document["signature_algorithm"] = None
    document["signing_public_key"] = None
    digest = ontology_integrity.canonical_signed_document_hash(document)

    active_signer = signer or ontology_integrity.ReleaseSigner.from_runtime()
    document["digest"] = digest
    document["signer"] = active_signer.signer_id
    document["signature_algorithm"] = active_signer.algorithm
    document["signing_public_key"] = active_signer.public_key
    document["signature"] = active_signer.sign(digest)
    return document


def verify_hydration_manifest(
    document: dict[str, Any], *, trusted_public_keys: tuple[str, ...] | None = None
) -> bool:
    """Verify a signed manifest document produced by
    :func:`sign_hydration_manifest`. Thin call-through to
    ``ontology_integrity.verify_release_signature`` — no second verification
    mechanism."""
    from agent_utilities.knowledge_graph.ontology import ontology_integrity

    unsigned = dict(document)
    signature = unsigned.pop("signature", None)
    signer = unsigned.pop("signer", None)
    algorithm = unsigned.pop("signature_algorithm", None)
    public_key = unsigned.pop("signing_public_key", None)
    digest = unsigned.pop("digest", None)
    unsigned["signature"] = None
    unsigned["signer"] = None
    unsigned["signature_algorithm"] = None
    unsigned["signing_public_key"] = None
    recomputed = ontology_integrity.canonical_signed_document_hash(unsigned)
    if not digest or recomputed != digest:
        return False
    return ontology_integrity.verify_release_signature(
        digest,
        signature,
        signer_id=signer,
        algorithm=algorithm,
        public_key=public_key,
        trusted_public_keys=trusted_public_keys,
    )


# ---------------------------------------------------------------------------
# Persistence — durable, queryable, mirrors the existing HydrationPlanStep shape
# ---------------------------------------------------------------------------


def persist_hydration_manifest(engine: Any, signed_document: dict[str, Any]) -> None:
    """Persist a signed manifest as a durable, queryable ``:HydrationManifest``
    node when the active engine accepts nodes — best-effort, mirrors
    ``kg_server._record_boot_hydration_step``'s pattern exactly (a stable id
    per generation, keyed by ``generated_at``, not an unbounded node per call).
    """
    add_node = getattr(engine, "add_node", None)
    if not callable(add_node):
        return
    generated_at = str(signed_document.get("generated_at") or "")
    try:
        add_node(
            f"hydration-manifest:{generated_at}",
            "HydrationManifest",
            {
                "generated_at": generated_at,
                "tenant_id": signed_document.get("tenant_id"),
                "service_authority_available": signed_document.get(
                    "service_authority_available"
                ),
                "signer": signed_document.get("signer"),
                "signature_algorithm": signed_document.get("signature_algorithm"),
                "digest": signed_document.get("digest"),
                "document": json.dumps(signed_document, sort_keys=True),
            },
        )
    except Exception:  # noqa: BLE001 - observability must not stop hydration
        logger.debug("hydration manifest persistence failed", exc_info=True)
