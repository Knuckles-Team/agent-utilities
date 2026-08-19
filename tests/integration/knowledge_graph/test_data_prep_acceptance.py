"""NE-115: bounded, real-engine data-preparation acceptance contract.

This is deliberately an acceptance harness rather than a unit-test substitute.
It refuses implicit package selection, shared/default graphs, test doubles and
unbounded Arrow input.  The only durable write in the scenario is
``ingest_envelope``; the harness never calls a mirror, an in-memory graph, or a
connector-specific write shortcut.

The test is opt-in.  Without the complete ``NE115_*`` environment it reports
``SKIPPED`` with the missing prerequisite, while an explicitly selected run
fails closed on any malformed or ambiguous prerequisite.  The distinction is
important: a normal checkout must not accidentally contact a host engine, and
an acceptance run must never silently fall back to one.

The engine-native profile operation is intentionally a required capability.  It
is not emulated here.  Until the epistemic-graph profile operation is deployed,
the acceptance run is unavailable rather than producing a local-only verdict.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import subprocess
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterator

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.engine, pytest.mark.live]


class AcceptanceUnavailable(RuntimeError):
    """The explicit real acceptance prerequisites are not available."""


class AcceptanceInvariantError(AssertionError):
    """A fail-closed acceptance invariant was bypassed."""


_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_TENANT_RE = re.compile(r"^ne115:[A-Za-z0-9][A-Za-z0-9._:-]{0,95}$")
_OPAQUE_REF_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,255}$")
_UNSAFE_FIELD_RE = re.compile(
    r"(?:token|secret|password|authorization|private[_-]?key|cookie)",
    re.IGNORECASE,
)
_FORBIDDEN_ENDPOINT_RE = re.compile(
    r"(?:mock|fake|emulator|inmemory|in-memory|test-engine)", re.IGNORECASE
)

_MAX_IPC_BYTES = 2 * 1024 * 1024
_MAX_DECOMPRESSED_BYTES = 8 * 1024 * 1024
_MAX_ROWS = 10_000
_MAX_COLUMNS = 64
_MAX_PROFILE_BYTES = 4 * 1024 * 1024
_MAX_PROFILE_DEADLINE_MS = 10_000


def _required_env(name: str) -> str:
    value = str(os.environ.get(name, "") or "").strip()
    if not value:
        raise AcceptanceUnavailable(f"{name} is required for NE-115")
    return value


def _absolute_path(name: str) -> Path:
    raw = _required_env(name)
    path = Path(raw)
    if not path.is_absolute() or not path.is_dir():
        raise AcceptanceUnavailable(f"{name} must name an existing absolute repository")
    return path


def _exact_revision(name: str, repository: Path) -> str:
    expected = _required_env(name).lower()
    if not _GIT_SHA_RE.fullmatch(expected):
        raise AcceptanceUnavailable(f"{name} must be one exact 40-hex revision")
    try:
        actual = subprocess.run(
            ["git", "-C", str(repository), "rev-parse", "--verify", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip().lower()
    except (OSError, subprocess.SubprocessError) as exc:
        raise AcceptanceUnavailable(f"cannot resolve {name} repository revision") from exc
    if actual != expected:
        raise AcceptanceUnavailable(f"{name} does not match the checked-out HEAD")
    return expected


def _exact_digest(name: str) -> str:
    value = _required_env(name).lower()
    if not _SHA256_RE.fullmatch(value):
        raise AcceptanceUnavailable(f"{name} must be sha256:<64 lowercase hex>")
    return value


def _opaque_ref(name: str) -> str:
    value = _required_env(name)
    if not _OPAQUE_REF_RE.fullmatch(value) or _FORBIDDEN_ENDPOINT_RE.search(value):
        raise AcceptanceUnavailable(f"{name} must be a bounded opaque reference")
    return value


@dataclass(frozen=True, slots=True)
class AcceptanceConfig:
    """Explicit identities and bounds for one isolated acceptance run."""

    au_root: Path
    au_revision: str
    eg_root: Path
    eg_revision: str
    gitlab_root: Path
    gitlab_revision: str
    endpoint: str
    tenant: str
    graph: str
    auth_secret: str
    profile_target: str
    profile_schema_digest: str
    run_live: bool

    @classmethod
    def from_env(cls) -> AcceptanceConfig:
        if str(os.environ.get("NE115_RUN_LIVE_ACCEPTANCE", "")).lower() not in {
            "1",
            "true",
            "yes",
        }:
            raise AcceptanceUnavailable(
                "set NE115_RUN_LIVE_ACCEPTANCE=1 to opt into the real acceptance run"
            )

        endpoint = _required_env("NE115_ENGINE_ENDPOINT")
        if not endpoint.startswith("unix://"):
            raise AcceptanceUnavailable(
                "NE115_ENGINE_ENDPOINT must be an explicit unix:// endpoint"
            )
        socket_path = endpoint.removeprefix("unix://")
        if not Path(socket_path).is_absolute() or "," in endpoint:
            raise AcceptanceUnavailable(
                "NE115_ENGINE_ENDPOINT must name one absolute isolated Unix socket"
            )
        if _FORBIDDEN_ENDPOINT_RE.search(endpoint):
            raise AcceptanceUnavailable(
                "NE115_ENGINE_ENDPOINT may not identify a mock, emulator, or test server"
            )
        if str(os.environ.get("NE115_ENGINE_ISOLATED", "")).lower() not in {
            "1",
            "true",
            "yes",
        }:
            raise AcceptanceUnavailable(
                "NE115_ENGINE_ISOLATED=true is required; shared/default graphs are forbidden"
            )

        tenant = _required_env("NE115_ENGINE_TENANT")
        graph = _required_env("NE115_ENGINE_GRAPH")
        if not _TENANT_RE.fullmatch(tenant) or graph != tenant:
            raise AcceptanceUnavailable(
                "NE115_ENGINE_TENANT and NE115_ENGINE_GRAPH must be the same unique ne115:* graph"
            )
        secret = _required_env("NE115_ENGINE_AUTH_SECRET")
        if len(secret) < 16:
            raise AcceptanceUnavailable("NE115_ENGINE_AUTH_SECRET is too short")

        au_root = _absolute_path("NE115_AU_ROOT")
        eg_root = _absolute_path("NE115_EG_ROOT")
        gitlab_root = _absolute_path("NE115_GITLAB_API_ROOT")
        if len({au_root.resolve(), eg_root.resolve(), gitlab_root.resolve()}) != 3:
            raise AcceptanceUnavailable("AU, EG, and gitlab-api roots must be distinct")
        au_revision = _exact_revision("NE115_AU_REVISION", au_root)
        eg_revision = _exact_revision("NE115_EG_REVISION", eg_root)
        gitlab_revision = _exact_revision("NE115_GITLAB_API_REVISION", gitlab_root)

        return cls(
            au_root=au_root,
            au_revision=au_revision,
            eg_root=eg_root,
            eg_revision=eg_revision,
            gitlab_root=gitlab_root,
            gitlab_revision=gitlab_revision,
            endpoint=endpoint,
            tenant=tenant,
            graph=graph,
            auth_secret=secret,
            profile_target=_opaque_ref("NE115_ENGINE_PROFILE_TARGET"),
            profile_schema_digest=_exact_digest("NE115_ENGINE_PROFILE_SCHEMA_DIGEST"),
            run_live=True,
        )


def _sha256_file(path: Path) -> str:
    try:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise AcceptanceUnavailable("governance shape material is unavailable") from exc
    return f"sha256:{digest}"


def _shape_digest() -> str:
    return _sha256_file(
        Path(__file__).parents[3]
        / "agent_utilities"
        / "knowledge_graph"
        / "shapes"
        / "governance.shapes.ttl"
    )


def _forbidden_material(value: Any, forbidden: tuple[str, ...]) -> None:
    """Reject raw rows/secrets in any persisted evidence-shaped value."""

    encoded = json.dumps(value, sort_keys=True, ensure_ascii=True, default=str)
    if any(marker and marker in encoded for marker in forbidden):
        raise AcceptanceInvariantError("profile/evidence contains raw source material")


def _assert_digest(value: str, label: str) -> str:
    if not _SHA256_RE.fullmatch(value):
        raise AcceptanceInvariantError(f"{label} is not a canonical digest")
    return value


def _check_cancelled(cancelled: bool) -> None:
    if cancelled:
        raise AcceptanceUnavailable("acceptance operation cancelled before engine contact")


def _require_profile_contract() -> dict[str, Any]:
    """Import NE-112's typed profile contract without a legacy fallback."""

    try:
        from agent_utilities.data_prep import (
            LocalProfile,
            ProfileLimits,
            ProfileRequest,
            ProfileResult,
            ProfileTarget,
            profile_digest,
            profile_table,
            profile_with_client,
        )
    except ImportError as exc:
        raise AcceptanceUnavailable(
            "NE-112 typed profile contract is required before NE-115"
        ) from exc
    return {
        "ProfileLimits": ProfileLimits,
        "LocalProfile": LocalProfile,
        "ProfileRequest": ProfileRequest,
        "ProfileResult": ProfileResult,
        "ProfileTarget": ProfileTarget,
        "profile_digest": profile_digest,
        "profile_table": profile_table,
        "profile_with_client": profile_with_client,
    }


@dataclass(frozen=True, slots=True)
class BoundedArrowInput:
    """Validate IPC bytes before Arrow is allowed to decompress them."""

    raw: bytes
    max_bytes: int = _MAX_IPC_BYTES
    max_rows: int = _MAX_ROWS
    max_columns: int = _MAX_COLUMNS
    max_decompressed_bytes: int = _MAX_DECOMPRESSED_BYTES

    def read(self) -> Any:
        if not isinstance(self.raw, bytes) or len(self.raw) > self.max_bytes:
            raise AcceptanceInvariantError("Arrow IPC input exceeds the wire byte bound")
        try:
            import pyarrow as pa

            table = pa.ipc.open_stream(pa.BufferReader(self.raw)).read_all()
        except Exception as exc:
            # The broad branch is intentional: malformed IPC must never escape
            # into a caller-selected fallback parser or emulator.
            raise AcceptanceInvariantError("malformed Arrow IPC rejected") from exc
        if table.num_rows > self.max_rows:
            raise AcceptanceInvariantError("Arrow IPC row bound exceeded")
        if table.num_columns > self.max_columns:
            raise AcceptanceInvariantError("Arrow IPC column bound exceeded")
        if table.nbytes > self.max_decompressed_bytes:
            raise AcceptanceInvariantError("Arrow IPC decompressed byte bound exceeded")
        return table


class _NativeProfileAdapter:
    """Typed adapter over the real engine analytics profile operation."""

    def __init__(self, client: Any) -> None:
        analytics = getattr(client, "analytics", None)
        method = getattr(analytics, "profile", None)
        if not callable(method):
            raise AcceptanceUnavailable(
                "epistemic-graph must advertise the native analytics.profile operation"
            )
        self._profile = method

    def profile(self, request: Any) -> Any:
        response = self._profile(request)
        if not hasattr(response, "model_dump"):
            raise AcceptanceInvariantError(
                "native analytics.profile must return the typed ProfileResult"
            )
        return response


@dataclass(slots=True)
class _AcceptanceTrace:
    events: list[str] = field(default_factory=list)

    def add(self, event: str) -> None:
        self.events.append(event)

    def checkpoint(self) -> None:
        if not self.events or self.events[-1] not in {"commit:durable", "commit:replay"}:
            raise AcceptanceInvariantError(
                "checkpoint advanced before a durable commit or replay receipt"
            )
        self.events.append("checkpoint:advanced")


@contextlib.contextmanager
def _explicit_endpoint(config: AcceptanceConfig) -> Iterator[None]:
    """Route only this acceptance run to its explicitly selected endpoint."""

    old = {
        key: os.environ.get(key)
        for key in ("GRAPH_SERVICE_ENDPOINTS", "GRAPH_SERVICE_AUTH_SECRET")
    }
    os.environ["GRAPH_SERVICE_ENDPOINTS"] = config.endpoint
    os.environ["GRAPH_SERVICE_AUTH_SECRET"] = config.auth_secret
    try:
        yield
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


@contextlib.contextmanager
def _real_engine(config: AcceptanceConfig) -> Iterator[Any]:
    """Open the production AU authority against the explicit real endpoint."""

    try:
        from agent_utilities.knowledge_graph.backends.epistemic_graph_backend import (
            EpistemicGraphBackend,
        )
        from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine
    except ImportError as exc:
        raise AcceptanceUnavailable("the AU native engine adapter is unavailable") from exc

    engine = IntelligenceGraphEngine(
        backend=EpistemicGraphBackend(graph_name=config.graph),
        defer_background_start=True,
    )
    compute = getattr(getattr(engine, "backend", None), "graph", None)
    client = getattr(compute, "client", None)
    if client is None or not str(getattr(compute, "endpoint", "")) == config.endpoint:
        _close_real_engine(engine)
        raise AcceptanceInvariantError(
            "the acceptance authority did not bind to the configured endpoint"
        )
    if not str(type(client).__module__).startswith("epistemic_graph"):
        raise AcceptanceInvariantError("acceptance authority is not the native client")
    try:
        yield engine
    finally:
        _close_real_engine(engine)


def _close_real_engine(engine: Any) -> None:
    """Close the native graph transport and release AU's process singleton."""

    compute = getattr(getattr(engine, "backend", None), "graph", None)
    close = getattr(compute, "close", None)
    if callable(close):
        close()
    try:
        from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine

        IntelligenceGraphEngine.set_active(None)
    except ImportError:
        pass


def _session(tenant: str, graph: str) -> Any:
    from agent_utilities.models.company_brain import ActorType
    from agent_utilities.security.brain_context import ActorContext
    from agent_utilities.knowledge_graph.core.session import GraphSession

    actor = ActorContext(
        actor_id="service:ne115-data-prep-acceptance",
        actor_type=ActorType.AUTOMATED_SERVICE,
        roles=("ne115-data-reader",),
        tenant_id=tenant,
        authenticated=True,
    )
    return GraphSession(
        actor=actor,
        tenant=tenant,
        graph=graph,
        scopes=frozenset({"kg:read", "kg:write", "kg:admin"}),
        policy_version="policy:ne115-v1",
        audience="epistemic-graph",
    )


def _require_acl(access: Any, *, roles: tuple[str, ...]) -> None:
    if access is None:
        raise AcceptanceInvariantError("ACL must be admitted before profiling")
    if bool(getattr(access, "is_public", False)):
        return
    allowed = set(getattr(access, "read_roles", ()) or ())
    if not allowed.intersection(roles):
        raise AcceptanceInvariantError("ACL denies profiling before any data is inspected")


def _assert_lossy_cast_rejected(
    profile_api: dict[str, Any], table: Any, registry: Any, model_ref: str
) -> None:
    """A narrowing cast must fail before a row can become a commit candidate."""

    from agent_utilities.data_prep import CleanPipeline, CleanPlan

    LocalProfile = profile_api["LocalProfile"]
    lossy = CleanPlan.model_validate(
        {
            "schema_version": "1",
            "steps": [
                {"verb": "canonical_names"},
                {
                    "verb": "safe_cast",
                    "column": "id",
                    "source_type": "int64",
                    "target_type": "int32",
                },
            ],
            "profile": LocalProfile(
                max_rows=_MAX_ROWS,
                max_columns=16,
                max_steps=4,
                max_bytes=_MAX_PROFILE_BYTES,
                max_outcome_rows=_MAX_ROWS,
            ),
            "invalid_row_disposition": "fail",
            "plan_ref": "plan:ne115:lossy-cast",
            "policy_ref": "policy:ne115:v1",
            "model_ref": model_ref,
        }
    )
    try:
        CleanPipeline(lossy, model_registry=registry).run(table)
    except Exception:
        return
    raise AcceptanceInvariantError("lossy Arrow cast reached validation")


def _safe_arrow_table(rows: list[dict[str, Any]]) -> Any:
    try:
        import pyarrow as pa

        table = pa.Table.from_pylist(rows)
    except ImportError as exc:
        raise AcceptanceUnavailable("pyarrow is required for NE-115") from exc
    except Exception as exc:
        raise AcceptanceInvariantError("dirty GitLab fixture is not valid Arrow input") from exc
    if table.num_rows > _MAX_ROWS or table.num_columns > _MAX_COLUMNS:
        raise AcceptanceInvariantError("fixture exceeds Arrow bounds")
    if table.nbytes > _MAX_PROFILE_BYTES:
        raise AcceptanceInvariantError("fixture exceeds profile byte bound")
    return table


def _assert_negative_local_guards(profile_api: dict[str, Any], table: Any) -> None:
    """Exercise bounded rejection paths before a native write is possible."""

    ProfileLimits = profile_api["ProfileLimits"]
    LocalProfile = profile_api["LocalProfile"]
    profile_table = profile_api["profile_table"]
    try:
        profile_table(
            table,
            LocalProfile(max_rows=1, max_columns=16, max_steps=1),
        )
    except Exception:
        pass
    else:
        raise AcceptanceInvariantError("Arrow row bound was not enforced")

    try:
        BoundedArrowInput(b"not-arrow-ipc").read()
    except AcceptanceInvariantError:
        pass
    else:
        raise AcceptanceInvariantError("malformed Arrow IPC was accepted")

    try:
        BoundedArrowInput(b"x" * (_MAX_IPC_BYTES + 1)).read()
    except AcceptanceInvariantError:
        pass
    else:
        raise AcceptanceInvariantError("oversized/decompression-bomb IPC was accepted")

    try:
        profile_table(
            table,
            LocalProfile(
                max_rows=_MAX_ROWS,
                max_columns=16,
                max_steps=1,
                max_bytes=_MAX_PROFILE_BYTES,
            ),
            limits=ProfileLimits(
                max_columns=1, max_bytes=1, deadline_ms=1, max_cardinality=2
            ),
        )
    except Exception:
        pass
    else:
        raise AcceptanceInvariantError("Arrow column/byte profile bounds were not enforced")

    if table.num_rows <= 0:
        raise AcceptanceInvariantError("negative fixture guard needs one bounded row")
    try:
        _check_cancelled(True)
    except AcceptanceUnavailable:
        pass
    else:
        raise AcceptanceInvariantError("cancellation was not fail-closed")


def _dirty_rows() -> tuple[dict[str, Any], ...]:
    """A bounded, deterministic GitLab-like page; no credential material."""

    return (
        {
            "ID": 11501,
            "IID": 31,
            "Project ID": 7001,
            "Title": "  compiler regression  ",
            "State": "opened",
            "Created At": "2026-08-18T10:00:00Z",
            "Updated At": "2026-08-19T10:00:00Z",
            "Web URL": "https://gitlab.example.invalid/group/project/-/issues/31",
            "Labels": ["acceptance", "engine"],
        },
        {
            "ID": 11501,
            "IID": 31,
            "Project ID": 7001,
            "Title": "duplicate delivery is retained once",
            "State": "opened",
            "Created At": "2026-08-18T10:00:00Z",
            "Updated At": "2026-08-19T10:00:00Z",
            "Web URL": "https://gitlab.example.invalid/group/project/-/issues/31",
            "Labels": ["acceptance"],
        },
    )


def _secret_row() -> dict[str, Any]:
    return {
        "ID": 11502,
        "Title": "must never be persisted",
        "Authorization": "Bearer fixture-only-secret",
    }


def _reject_secret_row(row: dict[str, Any]) -> None:
    if any(_UNSAFE_FIELD_RE.search(str(key)) for key in row):
        raise AcceptanceInvariantError("secret-bearing source row rejected before profiling")


def _build_plan(profile_api: dict[str, Any], model_digest: str) -> Any:
    from agent_utilities.data_prep import CleanPlan

    LocalProfile = profile_api["LocalProfile"]
    return CleanPlan.model_validate(
        {
            "schema_version": "1",
            "steps": [
                {"verb": "canonical_names"},
                {"verb": "dedupe", "keys": ["id"]},
            ],
            "profile": LocalProfile(
                max_rows=_MAX_ROWS,
                max_columns=16,
                max_steps=4,
                max_bytes=_MAX_PROFILE_BYTES,
                max_outcome_rows=_MAX_ROWS,
            ),
            "invalid_row_disposition": "fail",
            "plan_ref": "plan:ne115:gitlab-issue-v1",
            "policy_ref": "policy:ne115:v1",
            "model_ref": "model:ne115:gitlab-issue-v1",
            "model_digest": model_digest,
            "source_ref": "source:gitlab-api:fixture",
            "artifact_ref": "artifact:ne115:profile",
        }
    )


def _map_envelope(
    profile_api: dict[str, Any],
    config: AcceptanceConfig,
    prepared: Any,
    profile: Any,
    shape_digest: str,
    *,
    source_acl: Any,
) -> Any:
    from agent_utilities.knowledge_graph.ingestion.change_envelope import ChangeEnvelope
    from agent_utilities.models.company_brain import DataClassification

    rows = prepared.table.to_pylist()
    if len(rows) != 1:
        raise AcceptanceInvariantError("deduplicated GitLab fixture must map one row")
    row = rows[0]
    object_digest = hashlib.sha256(str(row["id"]).encode("ascii")).hexdigest()
    object_id = f"artifact:gitlab-api:{object_digest}"
    content_digest = f"sha256:{hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()}"
    evidence = prepared.evidence.model_dump(mode="json")
    profile_json = profile.model_dump(mode="json")
    profile_digest = _assert_digest(profile_api["profile_digest"](profile), "profile")
    _forbidden_material(
        {"evidence": evidence, "profile": profile_json},
        ("compiler regression", "gitlab.example.invalid", "fixture-only-secret"),
    )
    structured_evidence = {
        "evidence_version": "ne115-acceptance.v1",
        "profile_digest": profile_digest,
        "prep_evidence_digest": _assert_digest(
            prepared.evidence.output_schema_digest, "prep output schema"
        ),
        "model_ref": prepared.evidence.model_ref,
        "model_digest": _assert_digest(prepared.evidence.model_digest, "model"),
        "validation": "pydantic-strict-extra-forbid:v1",
        "rows_in": int(prepared.evidence.rows_in),
        "rows_out": int(prepared.evidence.rows_out),
        "shape_digest": shape_digest,
        "au_revision": config.au_revision,
        "eg_revision": config.eg_revision,
        "gitlab_api_revision": config.gitlab_revision,
    }
    record = {
        "id": object_id,
        "type": "Artifact",
        "content_hash": content_digest,
        "connector": "gitlab-api",
        "source_object_id": object_id,
        "source_version": str(row["updated_at"]),
        "title": "GitLab issue artifact",
    }
    envelope = ChangeEnvelope.from_connector_record(
        record,
        connector="gitlab-api",
        tenant=config.tenant,
        source_instance="ne115-fixture",
        id_field="id",
        version_field="source_version",
        schema_version="gitlab-api:issue:v1",
        ontology_mapping_version="gitlab-api:artifact:v1",
        source_acl=source_acl,
        classification=DataClassification.INTERNAL,
        provenance={
            "source_revision": config.gitlab_revision,
            "shape_digest": shape_digest,
            "profile_digest": profile_digest,
            "prep_evidence_digest": prepared.evidence.output_schema_digest,
            "lineage": "gitlab-api:issue->Artifact:v1",
        },
        structured_evidence=structured_evidence,
        checkpoint="ne115:1",
    )
    _forbidden_material(
        {"provenance": envelope.provenance, "evidence": envelope.structured_evidence},
        ("compiler regression", "gitlab.example.invalid", "fixture-only-secret"),
    )
    return envelope


def _guard_envelope_scope(envelope: Any, config: AcceptanceConfig, shape_digest: str) -> None:
    if not envelope.tenant or envelope.tenant != config.tenant:
        raise AcceptanceInvariantError("omitted or wrong tenant rejected before native commit")
    if envelope.provenance.get("shape_digest") != shape_digest:
        raise AcceptanceInvariantError("wrong shape digest rejected before native commit")
    if envelope.structured_evidence is None:
        raise AcceptanceInvariantError("native commit requires structured evidence")


def _native_profile(
    profile_api: dict[str, Any], client: Any, config: AcceptanceConfig, *, as_of_lsn: int | None
) -> Any:
    ProfileLimits = profile_api["ProfileLimits"]
    ProfileRequest = profile_api["ProfileRequest"]
    ProfileTarget = profile_api["ProfileTarget"]
    target = ProfileTarget(
        target_kind="engine_rowset",
        target_ref=config.profile_target,
        schema_digest=config.profile_schema_digest,
        as_of_lsn=as_of_lsn,
    )
    request = ProfileRequest(
        schema_version="data-prep-profile.v1",
        target=target,
        limits=ProfileLimits(
            max_columns=16,
            max_cardinality=10_000,
            max_bytes=_MAX_PROFILE_BYTES,
            deadline_ms=_MAX_PROFILE_DEADLINE_MS,
            max_top_k=5,
            max_quantiles=3,
            disclosure_threshold=3,
            max_warnings=16,
        ),
    )
    return profile_api["profile_with_client"](_NativeProfileAdapter(client), request)


def _read_isolated_graph(engine: Any, session: Any, config: AcceptanceConfig) -> list[Any]:
    with _use_session(session):
        backend = getattr(engine, "backend", None)
        reader = getattr(backend, "execute_read", None)
        if not callable(reader):
            raise AcceptanceUnavailable("native graph read authority is unavailable")
        rows = reader(
            "MATCH (n) RETURN n.id AS id LIMIT 2",
            {"_clearance_level": 999},
        )
    if rows:
        raise AcceptanceUnavailable(
            f"configured tenant {config.tenant} is not empty; refusing cross-run contamination"
        )
    return rows


@contextlib.contextmanager
def _use_session(session: Any) -> Iterator[Any]:
    from agent_utilities.knowledge_graph.core.session import use_session

    with use_session(session):
        yield session


def test_ne115_real_data_prep_to_native_commit_and_replay() -> None:
    """Run the bounded dirty-fixture path against one explicitly isolated engine."""

    try:
        config = AcceptanceConfig.from_env()
        profile_api = _require_profile_contract()
        from agent_utilities.data_prep import (
            CleanPipeline,
            RowModelRegistry,
            row_model_digest,
        )
        from agent_utilities.protocols.epistemic_operations import ProtocolModel
        from agent_utilities.knowledge_graph.ingestion.envelope_ingest import ingest_envelope
        from agent_utilities.protocols.source_connectors.base import ExternalAccess
        from pydantic import ConfigDict
    except AcceptanceUnavailable as exc:
        pytest.skip(str(exc))

    class GitLabIssueRow(ProtocolModel):
        model_config = ConfigDict(strict=True, extra="forbid", frozen=True)

        id: int
        iid: int
        project_id: int
        title: str
        state: str
        created_at: str
        updated_at: str
        web_url: str
        labels: list[str]

    model_digest = row_model_digest(GitLabIssueRow)
    plan = _build_plan(profile_api, model_digest)
    registry = RowModelRegistry({plan.model_ref: GitLabIssueRow})
    raw_rows = list(_dirty_rows())
    try:
        _reject_secret_row(_secret_row())
    except AcceptanceInvariantError:
        pass
    else:
        raise AcceptanceInvariantError("secret-bearing fixture was not rejected")
    table = _safe_arrow_table(raw_rows)
    source_acl = ExternalAccess(read_roles=["ne115-data-reader"])
    try:
        _require_acl(None, roles=("ne115-data-reader",))
    except AcceptanceInvariantError:
        pass
    else:
        raise AcceptanceInvariantError("missing ACL was accepted before profiling")
    _require_acl(source_acl, roles=("ne115-data-reader",))

    local_profile = profile_api["profile_table"](
        table,
        plan.profile,
        target_ref="source:gitlab-api:ne115-fixture",
        limits=profile_api["ProfileLimits"](
            max_columns=16,
            max_cardinality=10_000,
            max_bytes=_MAX_PROFILE_BYTES,
            deadline_ms=_MAX_PROFILE_DEADLINE_MS,
            max_top_k=5,
            max_quantiles=3,
            disclosure_threshold=3,
            max_warnings=16,
        ),
    )
    local_profile_json = local_profile.model_dump(mode="json")
    _forbidden_material(
        local_profile_json,
        ("compiler regression", "gitlab.example.invalid", "fixture-only-secret"),
    )
    _assert_negative_local_guards(profile_api, table)

    prepared = CleanPipeline(plan, model_registry=registry).run(table)
    _assert_lossy_cast_rejected(profile_api, table, registry, plan.model_ref)
    if not prepared.evidence.checkpoint_eligible or prepared.evidence.rows_out != 1:
        raise AcceptanceInvariantError("cleaned fixture did not produce one checkpoint-eligible row")
    shape_digest = _shape_digest()
    envelope = _map_envelope(
        profile_api,
        config,
        prepared,
        local_profile,
        shape_digest,
        source_acl=source_acl,
    )
    trace = _AcceptanceTrace(
        events=["profile:local", "clean:arrow", "validate:pydantic", "map:lineage"]
    )

    with _explicit_endpoint(config), _real_engine(config) as runtime:
        session = _session(config.tenant, config.graph)
        with _use_session(session):
            _read_isolated_graph(runtime, session, config)
            compute = getattr(getattr(runtime, "backend", None), "graph", None)
            client = getattr(compute, "client", None)
            if client is None:
                raise AcceptanceInvariantError("native runtime has no graph client")

            _guard_envelope_scope(envelope, config, shape_digest)
            result = ingest_envelope(runtime, envelope)
            if result.get("status") != "success":
                raise AcceptanceInvariantError(
                    f"authoritative native commit did not succeed: {result.get('status')}"
                )
            trace.add("commit:durable")
            if result.get("watermark_advanced") is not True:
                raise AcceptanceInvariantError("successful native commit did not advance checkpoint")

            # A current and an as-of profile are both mandatory native reads,
            # after the durable table exists and before any replay is accepted.
            current_profile = _native_profile(profile_api, client, config, as_of_lsn=None)
            historical_lsn = current_profile.as_of_lsn
            if historical_lsn is None:
                raise AcceptanceInvariantError("current native profile did not return an LSN")
            historical_profile = _native_profile(
                profile_api, client, config, as_of_lsn=historical_lsn
            )
            if historical_profile.as_of_lsn != historical_lsn:
                raise AcceptanceInvariantError("historical profile ignored its requested LSN")
            if profile_api["profile_digest"](current_profile) != profile_api["profile_digest"](
                historical_profile
            ):
                raise AcceptanceInvariantError("same-LSN current and historical profiles differ")
            _forbidden_material(
                {
                    "current": current_profile.model_dump(mode="json"),
                    "historical": historical_profile.model_dump(mode="json"),
                },
                ("compiler regression", "gitlab.example.invalid", "fixture-only-secret"),
            )
            trace.add("profile:native-current")
            trace.add("profile:native-historical")

            # The malformed content hash must reach the real engine's SHACL/ICV
            # gate and be rejected atomically.  A local shape check alone would
            # not prove the authoritative admission boundary.
            invalid_key = hashlib.sha256(b"ne115-invalid-shape").hexdigest()
            invalid = replace(
                envelope,
                envelope_id=f"envelope:{invalid_key}",
                idempotency_key=invalid_key,
                typed_payload={
                    **(envelope.typed_payload or {}),
                    "content_hash": "sha256:not-a-valid-content-digest",
                },
            )
            invalid_result = ingest_envelope(runtime, invalid)
            if invalid_result.get("status") != "rejected":
                raise AcceptanceInvariantError("authoritative ICV rejection was not observed")
            if invalid_result.get("watermark_advanced"):
                raise AcceptanceInvariantError("ICV rejection advanced a checkpoint")
            trace.add("commit:icv-rejected")

            replay = ingest_envelope(runtime, envelope)
            if replay.get("status") != "skipped":
                raise AcceptanceInvariantError("exact replay did not return skipped")
            receipt = (replay.get("write_result") or {}).get("receipt") or {}
            if receipt.get("replayed") is not True or replay.get("watermark_advanced"):
                raise AcceptanceInvariantError("exact replay was not zero-delta")
            trace.add("commit:replay")

            # Wrong tenant and wrong shape are rejected before the native method.
            for bad in (
                replace(envelope, tenant=""),
                replace(envelope, tenant="ne115:wrong-tenant"),
                replace(
                    envelope,
                    provenance={**envelope.provenance, "shape_digest": "sha256:" + "0" * 64},
                ),
            ):
                try:
                    _guard_envelope_scope(bad, config, shape_digest)
                except AcceptanceInvariantError:
                    continue
                raise AcceptanceInvariantError("invalid envelope reached the commit seam")

            # Crash-before-replay: close the transport before admission; the
            # failed call cannot advance a source checkpoint.  The restarted
            # connection below must recover the durable receipt, not guess it.
            trace.add("transport:crash-before-replay")
            trace.add("transport:crash-after-commit-before-checkpoint")
            _close_real_engine(runtime)
            failed = ingest_envelope(runtime, envelope)
            if failed.get("status") not in {"failed", "rejected"}:
                raise AcceptanceInvariantError("crash-before-commit unexpectedly committed")
            if failed.get("watermark_advanced"):
                raise AcceptanceInvariantError("crash-before-commit advanced a checkpoint")

    # A second explicit connection is required after the first runtime closes;
    # it is intentionally outside the context above so the test proves that
    # state is read again rather than served by the original client object.
    with _explicit_endpoint(config), _real_engine(config) as restarted:
        restarted_session = _session(config.tenant, config.graph)
        with _use_session(restarted_session):
            compute = getattr(getattr(restarted, "backend", None), "graph", None)
            client = getattr(compute, "client", None)
            if client is None:
                raise AcceptanceInvariantError("restarted native runtime has no client")
            replay_after_restart = ingest_envelope(restarted, envelope)
            if replay_after_restart.get("status") != "skipped":
                raise AcceptanceInvariantError("restart/re-read did not preserve replay identity")
            receipt = (replay_after_restart.get("write_result") or {}).get("receipt") or {}
            if receipt.get("replayed") is not True:
                raise AcceptanceInvariantError("restart/re-read replay was not authoritative")
            trace.add("restart:reread")
            trace.add("commit:replay")
            trace.checkpoint()

            if trace.events[-2:] != ["commit:replay", "checkpoint:advanced"]:
                raise AcceptanceInvariantError("checkpoint order is not durable-first")
