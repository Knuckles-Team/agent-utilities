"""Regression guard: every ``/api/enhanced/*`` route has at least one consumer.

Scans ``agent_webui.api_extensions`` for ``@router`` declarations and asserts
that every route is referenced by either the React frontend
(``agent-webui/src/``) or the terminal UI
(``agent-terminal-ui/agent_terminal_ui/``). Agent-internal routes that are
intentionally not UI-wired live in ``ADMIN_ONLY_ENDPOINTS`` with a short
justification each.

Pure path-grep -- no HTTP, no cross-package imports, no Node tooling -- so
the scan finishes in a few seconds even on a cold WSL filesystem. Fails
loudly when a new route is added without either a consumer or an allowlist
entry, and when the repo layout drifts so the UI source trees cannot be
located.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration


ENHANCED_PREFIX = "/api/enhanced"

_AGENT_PACKAGES_ROOT = Path(__file__).resolve().parents[3]
_API_EXTENSIONS = (
    _AGENT_PACKAGES_ROOT / "agent-webui" / "agent" / "agent_webui" / "api_extensions.py"
)
_WEBUI_SRC = _AGENT_PACKAGES_ROOT / "agent-webui" / "src"
_TERMINAL_UI_SRC = _AGENT_PACKAGES_ROOT / "agent-terminal-ui" / "agent_terminal_ui"

_ROUTER_DECORATOR = re.compile(
    r"""@router\.(get|post|put|delete|patch|head|options)\s*"""
    r"""\(\s*[`'"]([^`'"]+)[`'"]""",
    re.IGNORECASE,
)

_ROUTE_PLACEHOLDER = re.compile(r"\{[^}]+\}")


ADMIN_ONLY_ENDPOINTS: dict[str, str] = {
    "/agents": (
        "Knowledge-Graph agent registry; consumed by the graph bootstrap "
        "layer, not a user-facing view."
    ),
    "/agent-icon": (
        "Static avatar FileResponse; served via markup/<img src>, not a "
        "fetch call in either UI tree."
    ),
    "/graph/link": (
        "Creates a relationship between two existing nodes. Only reached "
        "via the ``link_knowledge_nodes`` tool, never the UI."
    ),
    "/graph/query": (
        "Arbitrary Cypher. Server-side denylist gates dangerous verbs; "
        "intentionally hidden from both UIs."
    ),
    "/graph/magma": (
        "MAGMA orthogonal retrieval; agent-internal via "
        "``retrieve_orthogonal_context``."
    ),
    "/kb/update": (
        "Incremental KB re-ingestion; scheduled maintenance, no end-user entry point."
    ),
    "/upload": (
        "Workspace file upload. Current FilesView.tsx is a 124-LOC "
        "scaffold with no drop-target; allowlisted until upload is wired "
        "or the endpoint is retired."
    ),
    "/chats/{chat_id}": (
        "Parity gap: web-UI builds ``chats${id}`` without a slash "
        "(Chat.tsx, app-sidebar.tsx); terminal-UI hits the core Pydantic "
        "AI ``/chats/{chat_id}`` instead of this enhanced mirror. "
        "Tracked for P3 follow-up."
    ),
    "/chats/{chat_id}/title": (
        "Parity gap: no rename/delete chat flow in either UI. Backend "
        "also has a latent verb bug (``delete_chat`` on ``@router.get``). "
        "Allowlisted pending a coordinated fix."
    ),
}


def _load_api_extensions_text() -> str:
    """Return ``api_extensions.py`` contents for regex parsing."""
    assert _API_EXTENSIONS.is_file(), (
        f"Could not locate api_extensions.py at {_API_EXTENSIONS}. "
        "Has the repo layout changed?"
    )
    return _API_EXTENSIONS.read_text(encoding="utf-8")


def _extract_routes(source: str) -> list[tuple[str, str]]:
    """Return every ``(method, path)`` pair declared via ``@router``."""
    return [(m.upper(), p) for m, p in _ROUTER_DECORATOR.findall(source)]


def _route_paths(source: str) -> set[str]:
    """Return the unique path templates declared on the router."""
    return {p for _, p in _extract_routes(source)}


_WILDCARD_SEGMENT = r"""(?:\{[^}]+\}|\$\{[^}]+\}|[^/\s"'`?#]+)"""


def _build_path_regex(path: str) -> re.Pattern[str]:
    """Compile a regex for ``path`` that accepts any path-parameter form.

    FastAPI ``{name}`` captures, frontend ``${expr}`` template literals,
    and literal concrete values all satisfy the wildcard. Each segment is
    escaped independently so separators stay literal.
    """
    full = f"{ENHANCED_PREFIX}{path}"
    parts = [
        _WILDCARD_SEGMENT if _ROUTE_PLACEHOLDER.fullmatch(seg) else re.escape(seg)
        for seg in full.split("/")
    ]
    return re.compile("/".join(parts))


def _collect_tree_text(tree: Path, patterns: list[str]) -> str:
    """Concatenate every matching source file in ``tree``.

    Reading once and grepping many times keeps the per-route scan constant
    in file-count rather than linear per call.
    """
    if not tree.is_dir():
        return ""

    chunks: list[str] = []
    for ext in patterns:
        for file in tree.rglob(ext):
            if "node_modules" in file.parts or "__tests__" in file.parts:
                continue
            try:
                chunks.append(file.read_text(encoding="utf-8", errors="ignore"))
            except OSError:
                continue
    return "\n".join(chunks)


_WEBUI_SOURCE_CACHE: str | None = None
_TERMINAL_UI_SOURCE_CACHE: str | None = None


def _webui_source() -> str:
    """Cache the concatenated web-UI source blob."""
    global _WEBUI_SOURCE_CACHE
    if _WEBUI_SOURCE_CACHE is None:
        _WEBUI_SOURCE_CACHE = _collect_tree_text(_WEBUI_SRC, ["*.tsx", "*.ts"])
    return _WEBUI_SOURCE_CACHE


def _terminal_source() -> str:
    """Cache the concatenated terminal-UI source blob."""
    global _TERMINAL_UI_SOURCE_CACHE
    if _TERMINAL_UI_SOURCE_CACHE is None:
        _TERMINAL_UI_SOURCE_CACHE = _collect_tree_text(_TERMINAL_UI_SRC, ["*.py"])
    return _TERMINAL_UI_SOURCE_CACHE


def _has_web_consumer(path: str) -> bool:
    """Return True if the web-UI source references ``path``."""
    return bool(_build_path_regex(path).search(_webui_source()))


def _has_terminal_consumer(path: str) -> bool:
    """Return True if the terminal-UI source references ``path``."""
    return bool(_build_path_regex(path).search(_terminal_source()))


@pytest.fixture(scope="module")
def api_source() -> str:
    """Load ``api_extensions.py`` once for all tests in this module."""
    return _load_api_extensions_text()


@pytest.fixture(scope="module")
def route_paths(api_source: str) -> set[str]:
    """Cache the declared route path set across tests."""
    return _route_paths(api_source)


class TestEndpointConsumerCoverage:
    """Every enhanced route must have a UI consumer or be allowlisted."""

    def test_every_route_has_consumer_or_allowlist_entry(
        self, route_paths: set[str]
    ) -> None:
        """Assert UI parity for every ``/api/enhanced/*`` route."""
        orphans: list[str] = []
        for path in sorted(route_paths):
            if path in ADMIN_ONLY_ENDPOINTS:
                continue
            if _has_web_consumer(path):
                continue
            if _has_terminal_consumer(path):
                continue
            orphans.append(path)

        if orphans:
            rendered = "\n".join(f"  - {ENHANCED_PREFIX}{p}" for p in orphans)
            pytest.fail(
                "The following enhanced routes have no UI consumer and are "
                f"not in ADMIN_ONLY_ENDPOINTS:\n{rendered}\n\n"
                "Either wire a consumer in agent-webui or agent-terminal-ui, "
                "or add the path to ADMIN_ONLY_ENDPOINTS with a short "
                "justification."
            )

    def test_allowlist_entries_are_real_routes(self, route_paths: set[str]) -> None:
        """``ADMIN_ONLY_ENDPOINTS`` must not drift away from the router."""
        stale = sorted(set(ADMIN_ONLY_ENDPOINTS) - route_paths)
        if stale:
            rendered = "\n".join(f"  - {p}" for p in stale)
            pytest.fail(
                "ADMIN_ONLY_ENDPOINTS contains paths no longer declared "
                f"on the router:\n{rendered}"
            )

    def test_allowlist_entries_have_justification(self) -> None:
        """Every allowlist entry must document why it has no UI consumer."""
        bare = [
            path
            for path, reason in ADMIN_ONLY_ENDPOINTS.items()
            if not reason or len(reason.strip()) < 20
        ]
        if bare:
            pytest.fail(
                "ADMIN_ONLY_ENDPOINTS entries must include a >=20-char "
                f"justification: {', '.join(bare)}"
            )


class TestRouteExtraction:
    """Smoke checks on the regex that powers the consumer scan."""

    def test_router_regex_finds_routes(self, api_source: str) -> None:
        """The regex must extract a realistic number of routes."""
        routes = _extract_routes(api_source)
        assert len(routes) >= 30, (
            f"Expected >=30 @router declarations, found {len(routes)}"
        )

    def test_router_regex_captures_all_verbs(self, api_source: str) -> None:
        """GET, POST, PUT, and DELETE must all be represented."""
        verbs = {method for method, _ in _extract_routes(api_source)}
        for required in ("GET", "POST", "PUT", "DELETE"):
            assert required in verbs, f"{required} missing from route set"

    def test_known_core_routes_are_detected(self, route_paths: set[str]) -> None:
        """Sanity check: well-known paths must appear in the extracted set."""
        for known in ("/graph/stats", "/sdd/specs", "/kb/list", "/chats"):
            assert known in route_paths, f"Expected {known} in route paths"


class TestConsumerDetection:
    """Smoke checks on the consumer scan against hand-picked routes."""

    def test_graph_stats_has_consumer(self) -> None:
        """``/graph/stats`` is consumed by both UIs."""
        assert _has_web_consumer("/graph/stats")
        assert _has_terminal_consumer("/graph/stats")

    def test_graph_memory_param_route_has_consumer(self) -> None:
        """Parameterised ``/graph/memory/{memory_id}`` path matches."""
        assert _has_web_consumer("/graph/memory/{memory_id}")
        assert _has_terminal_consumer("/graph/memory/{memory_id}")

    def test_unknown_path_has_no_consumer(self) -> None:
        """A synthetic unknown path must report no consumer."""
        assert not _has_web_consumer("/unlikely/to/exist/42")
        assert not _has_terminal_consumer("/unlikely/to/exist/42")
