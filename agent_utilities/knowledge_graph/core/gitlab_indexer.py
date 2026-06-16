"""Ingest whole GitLab instances as a resolved code-symbol graph (CONCEPT:KG-2.9g).

This is the piece that assimilates *all of GitLab* — personal instance and
enterprise — into the one ontology-driven KG, surpassing GitLab's own
single-repo `gitlab-org/rust/knowledge-graph` (gkg): we reuse the
``epistemic-graph`` engine's cross-file resolver (``index_repository``,
CONCEPT:KG-2.8r) to turn each project's source tree into ``:Code`` symbols with
RESOLVED ``calls``/``depends_on`` edges, then write them under a per-instance
``source_system`` so the code graph shares an ontology with the issues/MRs/
pipelines (API objects) and the EA feeds (Egeria/LeanIX/ArchiMate/ERPNext).

The orchestration here is dependency-injected (``GitLabSource`` + an ``index_fn``
+ an ``ingest`` callable) so the mapping/batching/namespacing logic is unit
tested without a live GitLab or engine; the thin live adapter is built in
``source_sync._sync_gitlab``.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any, Protocol

logger = logging.getLogger(__name__)

# File extensions the engine's tree-sitter parser understands — kept in sync with
# eg-compute `SUPPORTED_EXTENSIONS`. Non-code blobs are skipped before fetch so we
# never pull (or parse) a 200MB binary.
CODE_EXTENSIONS: frozenset[str] = frozenset(
    {
        "py",
        "pyi",
        "js",
        "jsx",
        "mjs",
        "cjs",
        "ts",
        "mts",
        "cts",
        "tsx",
        "go",
        "rs",
        "java",
        "c",
        "h",
        "cpp",
        "cc",
        "cxx",
        "hpp",
        "hxx",
        "hh",
        "cs",
    }
)

# Default per-file size ceiling (bytes). Generated/vendored bundles blow past this
# and add noise, not signal; skip them.
DEFAULT_MAX_FILE_BYTES = 1_000_000


@dataclass
class GitLabProject:
    """The minimal project facts the indexer needs."""

    id: str
    path_with_namespace: str
    default_branch: str = "main"
    web_url: str = ""
    last_activity_at: str | None = None


class GitLabSource(Protocol):
    """What the indexer needs from a GitLab instance (injectable for tests)."""

    def list_projects(self) -> Iterable[GitLabProject]:
        """Every project visible on the instance (already group-recursed)."""
        ...

    def list_files(self, project: GitLabProject) -> Iterable[str]:
        """Repository file paths (blobs) on the project's default branch."""
        ...

    def get_file(self, project: GitLabProject, path: str) -> bytes | None:
        """Raw bytes of one file on the project's default branch (``None`` if unreadable)."""
        ...


@dataclass
class IndexSummary:
    """Roll-up of one instance sync (returned to the source-sync layer)."""

    instance: str
    projects_indexed: int = 0
    projects_skipped: int = 0
    files_indexed: int = 0
    symbols: int = 0
    calls_resolved: int = 0
    inherits_resolved: int = 0
    realizes_resolved: int = 0
    similar_resolved: int = 0
    imports_resolved: int = 0
    nodes_written: int = 0
    edges_written: int = 0
    errors: list[str] = field(default_factory=list)
    watermark: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "instance": self.instance,
            "projects_indexed": self.projects_indexed,
            "projects_skipped": self.projects_skipped,
            "files_indexed": self.files_indexed,
            "symbols": self.symbols,
            "calls_resolved": self.calls_resolved,
            "inherits_resolved": self.inherits_resolved,
            "realizes_resolved": self.realizes_resolved,
            "similar_resolved": self.similar_resolved,
            "imports_resolved": self.imports_resolved,
            "nodes_written": self.nodes_written,
            "edges_written": self.edges_written,
            "errors": self.errors,
            "watermark": self.watermark,
        }


def _is_code_file(path: str) -> bool:
    ext = path.rsplit(".", 1)[-1].lower() if "." in path else ""
    return ext in CODE_EXTENSIONS


def index_instance(
    *,
    instance: str,
    source: GitLabSource,
    index_fn: Callable[[list[tuple[str, bytes]]], dict[str, Any]],
    ingest: Callable[[str, list[dict[str, Any]], list[dict[str, Any]]], Any],
    project_ids: set[str] | None = None,
    since: str | None = None,
    max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
) -> IndexSummary:
    """Index one GitLab instance's code into the KG, one project at a time.

    Each project is its own resolution scope: its code files are shipped to
    ``index_fn`` (the engine's ``index_repository``) together so intra-project
    calls/imports resolve, then the resolved graph is namespaced, augmented with
    a Repository node + File nodes, and written via ``ingest`` under
    ``source_system = "gitlab:<instance>"``.

    - ``project_ids`` narrows to specific projects (webhook delta).
    - ``since`` (a watermark) skips projects whose ``last_activity_at`` is not newer
      (delta sync); ``None`` indexes all (full sync).
    """
    summary = IndexSummary(instance=instance)
    domain = f"gitlab:{instance}"
    watermark = since

    for project in source.list_projects():
        pid = str(project.id)
        if project_ids is not None and pid not in project_ids:
            continue
        # Delta: skip untouched projects (watermark is the max last_activity_at seen).
        if since and project.last_activity_at and project.last_activity_at <= since:
            summary.projects_skipped += 1
            continue

        try:
            files = _collect_code_files(source, project, max_file_bytes)
        except Exception as exc:  # noqa: BLE001 - one bad project must not abort the sweep
            summary.errors.append(
                f"{project.path_with_namespace}: list/fetch failed: {exc}"
            )
            continue
        if not files:
            summary.projects_skipped += 1
            continue

        try:
            result = index_fn(files)
        except Exception as exc:  # noqa: BLE001
            summary.errors.append(
                f"{project.path_with_namespace}: index_repository failed: {exc}"
            )
            continue

        entities, relationships = map_index_result(
            result, project=project, instance=instance
        )
        try:
            ingest(domain, entities, relationships)
        except Exception as exc:  # noqa: BLE001
            summary.errors.append(
                f"{project.path_with_namespace}: ingest failed: {exc}"
            )
            continue

        summary.projects_indexed += 1
        summary.files_indexed += len(files)
        summary.symbols += int(result.get("symbols_extracted", 0) or 0)
        summary.calls_resolved += int(result.get("calls_resolved", 0) or 0)
        summary.inherits_resolved += int(result.get("inherits_edges", 0) or 0)
        summary.realizes_resolved += int(result.get("realizes_edges", 0) or 0)
        summary.similar_resolved += int(result.get("similar_edges", 0) or 0)
        summary.imports_resolved += int(result.get("imports_resolved", 0) or 0)
        summary.nodes_written += len(entities)
        summary.edges_written += len(relationships)
        if project.last_activity_at and (
            watermark is None or project.last_activity_at > watermark
        ):
            watermark = project.last_activity_at

    summary.watermark = watermark
    return summary


def _collect_code_files(
    source: GitLabSource, project: GitLabProject, max_file_bytes: int
) -> list[tuple[str, bytes]]:
    """Fetch the project's code files as ``(path, bytes)``, skipping oversize blobs."""
    files: list[tuple[str, bytes]] = []
    for path in source.list_files(project):
        if not _is_code_file(path):
            continue
        data = source.get_file(project, path)
        if data is None or len(data) > max_file_bytes:
            continue
        files.append((path, data))
    return files


def map_index_result(
    result: dict[str, Any], *, project: GitLabProject, instance: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Map an engine ``IndexResult`` into KG entities + relationships.

    Node ids are namespaced ``gitlab:<instance>:<project_id>:<engine_id>`` so the
    engine's content-hash symbol ids never collide across projects/instances. A
    ``Repository`` node + ``File`` nodes are synthesized so ``IMPLEMENTS`` /
    ``depends_on`` edges (which reference ``file:<path>``) are not dangling, and the
    code graph ties back to the GitLab project structure.
    """
    ns = f"gitlab:{instance}:{project.id}:"

    def nid(engine_id: str) -> str:
        return f"{ns}{engine_id}"

    repo_id = nid("repository")
    entities: list[dict[str, Any]] = [
        {
            "id": repo_id,
            "type": "Repository",
            "name": project.path_with_namespace,
            "project_id": str(project.id),
            "instance": instance,
            "default_branch": project.default_branch,
            "web_url": project.web_url,
        }
    ]
    relationships: list[dict[str, Any]] = []

    file_ids: set[str] = set()

    def ensure_file(engine_file_id: str) -> str:
        """Materialize a File node for an engine ``file:<path>`` id (once)."""
        fid = nid(engine_file_id)
        if fid not in file_ids:
            file_ids.add(fid)
            path = (
                engine_file_id[len("file:") :]
                if engine_file_id.startswith("file:")
                else engine_file_id
            )
            entities.append(
                {
                    "id": fid,
                    "type": "File",
                    "name": path,
                    "file_path": path,
                    "project_id": str(project.id),
                    "instance": instance,
                }
            )
            relationships.append({"source": repo_id, "target": fid, "type": "CONTAINS"})
        return fid

    # Symbol nodes (functions/classes/methods, with native test-quality metrics).
    for node in result.get("nodes", []) or []:
        if node.get("node_type") != "SYMBOL":
            continue
        props = dict(node.get("properties", {}) or {})
        entities.append(
            {
                "id": nid(node["node_id"]),
                "type": "Code",
                "project_id": str(project.id),
                "instance": instance,
                **props,
            }
        )

    # Edges: IMPLEMENTS (file→symbol), resolved calls (symbol→symbol), resolved
    # depends_on (file→file). File endpoints are materialized on demand.
    for edge in result.get("edges", []) or []:
        etype = edge.get("edge_type", "")
        src = str(edge.get("source", ""))
        tgt = str(edge.get("target", ""))
        if not src or not tgt:
            continue
        if etype == "IMPLEMENTS":
            relationships.append(
                {"source": ensure_file(src), "target": nid(tgt), "type": "IMPLEMENTS"}
            )
        elif etype == "calls":
            relationships.append(
                {
                    "source": nid(src),
                    "target": nid(tgt),
                    "type": "calls",
                    **_edge_props(edge),
                }
            )
        elif etype in ("inherits", "realizes", "similar_to"):
            # Class→class structural (KG-2.100) + model-free similarity (KG-2.101)
            # edges; both endpoints are SYMBOL ids, namespaced like calls.
            relationships.append(
                {
                    "source": nid(src),
                    "target": nid(tgt),
                    "type": etype,
                    **_edge_props(edge),
                }
            )
        elif etype == "depends_on":
            relationships.append(
                {
                    "source": ensure_file(src),
                    "target": ensure_file(tgt),
                    "type": "depends_on",
                    **_edge_props(edge),
                }
            )

    return entities, relationships


def _edge_props(edge: dict[str, Any]) -> dict[str, Any]:
    """Carry through an edge's extra properties (e.g. callee ``name``/``module``)."""
    return {k: v for k, v in (edge.get("properties") or {}).items()}


# ── Production adapter: GitLab REST → GitLabSource ────────────────────────────


@dataclass
class GitLabInstanceConfig:
    """Connection facts for one GitLab instance (personal or enterprise)."""

    name: str  # short, stable id used in `source_system = gitlab:<name>`
    url: str  # base URL, e.g. https://gitlab.com or https://gitlab.acme.internal
    token: str = ""
    verify_ssl: bool = True


class GitLabRestSource:
    """A :class:`GitLabSource` backed by the GitLab REST API.

    Self-contained (no dependency on the ``gitlab_api`` client, which has no tree/
    raw-file methods and is single-host) so it works against any number of
    instances. The tree + raw-file endpoints are the two gkg-equivalent reads;
    everything else (resolution, ontology, reasoning) is ours.
    """

    def __init__(
        self,
        config: GitLabInstanceConfig,
        *,
        timeout: float = 30.0,
        per_page: int = 100,
    ):
        self.config = config
        self.timeout = timeout
        self.per_page = per_page

    @property
    def _base(self) -> str:
        return f"{self.config.url.rstrip('/')}/api/v4"

    def _session(self) -> Any:
        import requests  # lazy: keep the pure logic importable without requests

        s = requests.Session()
        if self.config.token:
            s.headers["PRIVATE-TOKEN"] = self.config.token
        s.verify = self.config.verify_ssl
        return s

    def _paginate(
        self, session: Any, path: str, params: dict[str, Any]
    ) -> Iterable[dict[str, Any]]:
        page = 1
        params = {**params, "per_page": self.per_page}
        while page and page <= 10_000:  # hard safety cap
            resp = session.get(
                f"{self._base}{path}",
                params={**params, "page": page},
                timeout=self.timeout,
            )
            resp.raise_for_status()
            rows = resp.json()
            if not rows:
                return
            yield from rows
            nxt = resp.headers.get("X-Next-Page", "")
            page = int(nxt) if nxt.isdigit() else 0

    def list_projects(self) -> Iterable[GitLabProject]:
        session = self._session()
        for row in self._paginate(
            session,
            "/projects",
            {"membership": "true", "simple": "false", "archived": "false"},
        ):
            yield GitLabProject(
                id=str(row.get("id")),
                path_with_namespace=row.get("path_with_namespace", str(row.get("id"))),
                default_branch=row.get("default_branch") or "main",
                web_url=row.get("web_url", ""),
                last_activity_at=row.get("last_activity_at"),
            )

    def list_files(self, project: GitLabProject) -> Iterable[str]:
        session = self._session()
        for row in self._paginate(
            session,
            f"/projects/{project.id}/repository/tree",
            {"recursive": "true", "ref": project.default_branch},
        ):
            if row.get("type") == "blob" and row.get("path"):
                yield row["path"]

    def get_file(self, project: GitLabProject, path: str) -> bytes | None:
        from urllib.parse import quote

        session = self._session()
        encoded = quote(path, safe="")
        resp = session.get(
            f"{self._base}/projects/{project.id}/repository/files/{encoded}/raw",
            params={"ref": project.default_branch},
            timeout=self.timeout,
        )
        if resp.status_code != 200:
            return None
        return resp.content


def instance_config_from_dict(item: dict[str, Any]) -> GitLabInstanceConfig | None:
    """Map one ``{name,url,token,verify_ssl}`` entry to a config, or ``None`` if
    it has no url. The single parsing rule shared by the indexer and the
    ``gitlab-api`` instance registry, so the XDG ``gitlab_instances`` schema means
    the same thing on both sides."""
    url = str(item.get("url", "")).strip()
    if not url:
        return None
    return GitLabInstanceConfig(
        name=str(item.get("name") or _host_slug(url)),
        url=url,
        token=str(item.get("token", "")),
        verify_ssl=bool(item.get("verify_ssl", True)),
    )


def instances_from_config(config: Any = None) -> list[GitLabInstanceConfig]:
    """Resolve the configured GitLab instances from the agent-utilities XDG config.

    The structured ``gitlab_instances`` list in
    ``~/.config/agent-utilities/config.json`` (a typed ``AgentConfig`` field) is
    the multi-tenant source of truth; it falls back to the single-host
    ``GITLAB_URL``/``GITLAB_TOKEN`` settings when no instances are configured.
    ``config`` defaults to the live ``AgentConfig`` singleton (injectable for tests).
    """
    if config is None:
        from agent_utilities.core.config import config as config_singleton

        config = config_singleton

    out: list[GitLabInstanceConfig] = []
    for item in getattr(config, "gitlab_instances", None) or []:
        if (
            isinstance(item, dict)
            and (cfg := instance_config_from_dict(item)) is not None
        ):
            out.append(cfg)

    if not out:
        url = (
            config.setting("GITLAB_URL", default="https://gitlab.com")
            or "https://gitlab.com"
        ).strip()
        token = (config.setting("GITLAB_TOKEN", default="") or "").strip()
        if token:
            out.append(GitLabInstanceConfig(name=_host_slug(url), url=url, token=token))
    return out


def _host_slug(url: str) -> str:
    """A short instance id from a URL host (``https://gitlab.acme.io`` → ``gitlab.acme.io``)."""
    from urllib.parse import urlparse

    host = urlparse(url).netloc or url
    return host.lower()


# ── Incremental: GitLab webhook → scoped re-index (CONCEPT:KG-2.9g) ────────────


@dataclass
class WebhookEvent:
    """A parsed GitLab webhook reduced to what drives an incremental re-index."""

    project_id: str
    kind: str  # push | tag_push | merge_request | …
    changed_code_files: list[str] = field(default_factory=list)


def parse_gitlab_webhook(payload: dict[str, Any]) -> WebhookEvent | None:
    """Reduce a GitLab webhook payload to a :class:`WebhookEvent`, or ``None``.

    Handles the events that change code: ``push``/``tag_push`` (commits carry
    added/modified/removed paths) and ``merge_request``. The project id is taken
    from ``project.id`` (falling back to ``project_id``). Only code-extension
    paths are surfaced in ``changed_code_files`` — the file-level delta hint a
    future engine ast_hash diff can use; today it scopes the re-index to the
    project.
    """
    if not isinstance(payload, dict):
        return None
    kind = str(payload.get("object_kind") or payload.get("event_type") or "").strip()
    project = payload.get("project") or {}
    pid = project.get("id") if isinstance(project, dict) else None
    pid = pid if pid is not None else payload.get("project_id")
    if pid is None:
        return None

    changed: set[str] = set()
    for commit in payload.get("commits") or []:
        if not isinstance(commit, dict):
            continue
        for key in ("added", "modified", "removed"):
            for path in commit.get(key) or []:
                if isinstance(path, str) and _is_code_file(path):
                    changed.add(path)

    return WebhookEvent(
        project_id=str(pid),
        kind=kind or "unknown",
        changed_code_files=sorted(changed),
    )


def handle_gitlab_webhook(
    engine: Any,
    payload: dict[str, Any],
    *,
    sync: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Drive an incremental re-index from a GitLab webhook (near-real-time).

    Parses the payload and triggers a **project-scoped delta** sync — only the
    project that changed is re-enumerated (the push advances its
    ``last_activity_at`` past the watermark, so the delta naturally re-includes
    it). ``sync`` is injectable for tests; in production it defaults to the shared
    :func:`source_sync.sync_source` entrypoint, so the webhook reuses the exact
    same resolve-and-write path as a full sync.
    """
    event = parse_gitlab_webhook(payload)
    if event is None:
        return {"status": "ignored", "reason": "unparseable or non-project webhook"}
    if event.kind not in {"push", "tag_push", "merge_request"}:
        return {
            "status": "ignored",
            "reason": f"event '{event.kind}' does not change code",
        }

    runner = sync
    if runner is None:
        from .source_sync import sync_source

        runner = sync_source

    result = runner(engine, "gitlab", mode="delta", ids=[event.project_id])
    return {
        "status": "ok",
        "kind": event.kind,
        "project_id": event.project_id,
        "changed_code_files": event.changed_code_files,
        "sync": result,
    }
