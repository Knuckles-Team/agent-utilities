"""GOC-83-W06 (R-27) — spy-transport proof that an ID-scoped GitLab sync
never issues an unscoped ``GET /projects`` (``list_projects``) HTTP call.

``tests/unit/knowledge_graph/core/test_gitlab_indexer.py`` already proves R-27
at the ``GitLabSource`` Protocol layer (a fake whose ``list_projects()``
raises if called at all) — real, solid evidence that ``index_instance``'s
OWN branching never calls it. But that proof is one layer removed from the
actual HTTP surface: it says nothing about whether the CONCRETE production
adapter, ``GitLabRestSource`` (the class that turns an id-scoped
:func:`~agent_utilities.knowledge_graph.core.gitlab_indexer.index_instance`
call into real requests), could ever itself internally enumerate
``/projects`` and filter client-side while still satisfying that Protocol-
level test (e.g. a ``get_project`` implementation that quietly called
``list_projects`` under the hood and searched the result would pass the
Protocol-level test byte-for-byte, since that test only observes
``GitLabSource`` method call counts, not what HTTP each method issues).

This asserts on the TRANSPORT directly: a fake ``requests.Session``-shaped
double records every ``(url, params)`` pair a live
``GitLabRestSource.get_project`` call issues, and the assertion is that
``/projects`` with ``membership=true`` (the broad enumeration endpoint) is
never among them — only ``/projects/<id>`` (the direct-by-id endpoint).
"""

from __future__ import annotations

from agent_utilities.knowledge_graph.core.gitlab_indexer import (
    GitLabInstanceConfig,
    GitLabRestSource,
)


class _FakeResponse:
    def __init__(self, status_code: int, payload):
        self.status_code = status_code
        self._payload = payload
        self.headers: dict[str, str] = {}

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class _SpySession:
    """Records every ``.get(url, params=...)`` call — the transport itself,
    not any higher-level method call count."""

    def __init__(self, responses: dict[str, _FakeResponse]):
        self.calls: list[tuple[str, dict]] = []
        self._responses = responses
        self.verify = True
        self.headers: dict[str, str] = {}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def close(self):
        pass

    def get(self, url, params=None, timeout=None):
        self.calls.append((url, dict(params or {})))
        for key, resp in self._responses.items():
            if key in url:
                return resp
        return _FakeResponse(404, None)


class _NoOpTrustProfile:
    def configure_requests_session(self, _session):
        pass

    def cleanup(self):
        pass


def _wire_fake_transport(monkeypatch, spy: _SpySession) -> None:
    """Patch the two seams ``GitLabRestSource._session`` opens: the requests
    session factory and the TLS trust resolver — both looked up via a LOCAL
    import inside ``_session()``, so patching the home module's attribute
    (not a re-exported alias) is what actually takes effect on the next call.
    """
    import agent_utilities.core.http_client as http_client_module
    import agent_utilities.core.transport_security as transport_security_module

    monkeypatch.setattr(
        http_client_module, "create_requests_session", lambda **_kw: spy
    )
    monkeypatch.setattr(
        transport_security_module,
        "resolve_configured_tls_profile",
        lambda *_a, **_kw: _NoOpTrustProfile(),
    )


def test_get_project_never_issues_a_membership_projects_listing_call(monkeypatch):
    """The exact R-27 invariant, at the transport: a single ``get_project``
    call (the direct-by-id lookup ``index_instance`` uses for an ID-scoped
    sync — see ``gitlab_indexer.py``'s ``index_instance``, lines ~192-207)
    must issue ONLY a ``GET /projects/<id>`` request, never
    ``GET /projects`` with ``membership=true``.
    """
    spy = _SpySession(
        {
            "/projects/42": _FakeResponse(
                200,
                {
                    "id": 42,
                    "path_with_namespace": "group/repo",
                    "default_branch": "main",
                    "web_url": "https://gitlab.example.test/group/repo",
                    "last_activity_at": "2026-08-01T00:00:00Z",
                },
            ),
        }
    )
    _wire_fake_transport(monkeypatch, spy)

    source = GitLabRestSource(
        GitLabInstanceConfig(name="test", url="https://gitlab.example.test")
    )

    project = source.get_project("42")

    assert project is not None
    assert project.id == "42"
    assert project.path_with_namespace == "group/repo"

    # The transport-level assertion: exactly one request, to the direct-by-id
    # endpoint, and NEVER to the broad membership-listing endpoint.
    assert len(spy.calls) == 1
    (url, params) = spy.calls[0]
    assert url.endswith("/projects/42")
    assert "membership" not in params

    for called_url, called_params in spy.calls:
        assert not (
            called_url.endswith("/projects")
            and called_params.get("membership") == "true"
        ), (
            "get_project() issued an unscoped GET /projects?membership=true "
            f"call — R-27 violated. All calls: {spy.calls!r}"
        )


def test_get_project_unauthorized_or_missing_still_never_lists(monkeypatch):
    """A 404 (or 403 — both fail-closed to ``None``, deliberately
    undistinguished) must not trigger any fallback broad-listing call
    either."""
    spy = _SpySession({})  # no id matches → every GET falls through to 404
    _wire_fake_transport(monkeypatch, spy)

    source = GitLabRestSource(
        GitLabInstanceConfig(name="test", url="https://gitlab.example.test")
    )

    project = source.get_project("99")

    assert project is None
    assert len(spy.calls) == 1
    (url, params) = spy.calls[0]
    assert url.endswith("/projects/99")
    assert "membership" not in params


def test_list_projects_itself_does_call_the_membership_endpoint(monkeypatch):
    """Control case: proves the spy actually WOULD catch a membership-listing
    call if one were made — ``list_projects()`` (the intentional unscoped
    full/delta-sync path) legitimately calls
    ``GET /projects?membership=true``. Without this, the two tests above
    would pass vacuously if the spy never matched a real membership call
    shape at all."""
    spy = _SpySession(
        {"/projects": _FakeResponse(200, [{"id": 1, "path_with_namespace": "a/b"}])}
    )
    _wire_fake_transport(monkeypatch, spy)

    source = GitLabRestSource(
        GitLabInstanceConfig(name="test", url="https://gitlab.example.test")
    )

    list(source.list_projects())

    assert any(
        url.endswith("/projects") and params.get("membership") == "true"
        for url, params in spy.calls
    ), f"expected a membership=true /projects call, got {spy.calls!r}"
