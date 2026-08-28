"""Fail-closed regression tests for `agent_utilities/mcp/server_factory.py`.

Both defects covered here are the same family -- **a degraded or unrecognised
input granting permission instead of refusing** -- and both violate this repo's
binding rule (AGENTS.md, "Fail closed -- a degraded read must never grant
permission").

BUG-CX-022 -- ``_component_passes`` returned ``True`` for any component
carrying neither ``name`` nor ``uri``, so such a component skipped **all**
visibility filtering, tag restrictions included, and was handed to the caller.

BUG-CX-023 -- ``_configure_auth`` fell off the end of its dispatch chain and
returned ``None`` ("no auth") for an unrecognised ``auth_type``. ``--auth-type``
carries ``choices=``, but argparse does **not** validate a *default*, and the
default is ``setting("AUTH_TYPE", "none")`` -- so a typo'd ``AUTH_TYPE``
environment variable reached the dispatch chain unchecked and produced a fully
UNAUTHENTICATED server.

Each test below fails on the pre-fix file and passes after it.
"""

from __future__ import annotations

import pytest

from agent_utilities.mcp import server_factory
from agent_utilities.mcp.server_factory import (
    _component_passes,
    _configure_auth,
    create_mcp_parser,
    create_mcp_server,
)


def _unsupported_auth_type_error() -> type[Exception]:
    """The typed refusal BUG-CX-023 adds.

    Resolved lazily (not imported at module scope) so that the BUG-CX-022 tests
    in this file still RUN -- and fail on their own assertion -- against the
    unfixed file, instead of the whole module erroring at collection.
    """
    error = getattr(server_factory, "UnsupportedAuthTypeError", None)
    if error is None:
        pytest.fail(
            "BUG-CX-023: server_factory defines no UnsupportedAuthTypeError -- "
            "an unrecognised auth_type still falls through to 'no auth'"
        )
    return error


class _Component:
    """Minimal stand-in for a FastMCP Tool/Resource/Prompt."""

    def __init__(self, name=None, uri=None, tags=None):
        self.name = name
        self.uri = uri
        self.tags = tags

    def __repr__(self):
        return f"_Component(name={self.name!r}, uri={self.uri!r}, tags={self.tags!r})"


@pytest.fixture
def clean_env(monkeypatch):
    """No env-based filters, no HTTP request context, no CLI overrides."""
    for key in (
        "MCP_ENABLED_TAGS",
        "MCP_DISABLED_TAGS",
        "MCP_ENABLED_TOOLS",
        "MCP_DISABLED_TOOLS",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(
        "agent_utilities.mcp.server_factory._optional_http_request", lambda: None
    )
    return monkeypatch


def _build_transform(command_args=None):
    """Build a real server and return its live DynamicVisibilityTransform."""
    _args, mcp, _mw = create_mcp_server(
        name="wd2-bug-sec-test-server", command_args=command_args or []
    )
    transforms = [
        t for t in mcp.transforms if type(t).__name__ == "DynamicVisibilityTransform"
    ]
    assert len(transforms) == 1, "expected exactly one DynamicVisibilityTransform"
    return transforms[0]


# ---------------------------------------------------------------------------
# BUG-CX-022 -- an unidentifiable component must be EXCLUDED, not exempted.
# ---------------------------------------------------------------------------
def test_unidentifiable_component_does_not_bypass_disabled_tags():
    """A component with a DISABLED tag but no name/uri must not be returned."""
    anon = _Component(name=None, uri=None, tags={"danger"})
    assert (
        _component_passes(
            anon,
            enabled_names=None,
            disabled_names=None,
            enabled_tags=None,
            disabled_tags={"danger"},
        )
        is False
    )


def test_unidentifiable_component_does_not_bypass_enabled_names():
    """An enabled-tools whitelist is a positive allowlist: something that
    cannot even be identified can never be ON it, so it must be excluded."""
    anon = _Component(name=None, uri=None, tags=None)
    assert (
        _component_passes(
            anon,
            enabled_names={"only-this-one"},
            disabled_names=None,
            enabled_tags=None,
            disabled_tags=None,
        )
        is False
    )


def test_unidentifiable_component_excluded_with_no_filters_at_all():
    """Even with every filter off, an object that is not an identifiable
    component is not something this server can vouch for exposing."""
    anon = _Component(name=None, uri=None, tags={"ok"})
    assert (
        _component_passes(
            anon,
            enabled_names=None,
            disabled_names=None,
            enabled_tags=None,
            disabled_tags=None,
        )
        is False
    )


def test_filter_components_hides_unidentifiable_component_under_tag_restriction(
    clean_env,
):
    """End-to-end through the live transform: the exact BUG-CX-022 scenario.

    A tag restriction is active; a component carrying neither name nor uri and
    a tag OUTSIDE the enabled set must NOT come back to the caller.
    """
    clean_env.setenv("MCP_ENABLED_TOOLS", "only-this-one")
    clean_env.setenv("MCP_ENABLED_TAGS", "some-tag")
    transform = _build_transform()

    anon = _Component(name=None, uri=None, tags={"unrelated"})
    named = _Component(name="only-this-one", tags={"some-tag"})
    excluded = _Component(name="excluded", tags={"some-tag"})

    out = transform._filter_components([anon, named, excluded])

    assert anon not in out, "BUG-CX-022: unidentifiable component bypassed filtering"
    assert named in out
    assert excluded not in out


def test_named_and_uri_only_components_still_pass():
    """The fix must not narrow anything else: a name-only and a uri-only
    component with no active filter both still pass."""
    for component in (_Component(name="tool-a"), _Component(uri="res://a")):
        assert (
            _component_passes(
                component,
                enabled_names=None,
                disabled_names=None,
                enabled_tags=None,
                disabled_tags=None,
            )
            is True
        )


# ---------------------------------------------------------------------------
# BUG-CX-023 -- an unrecognised auth_type must REFUSE, not fall through to
# "no auth".
# ---------------------------------------------------------------------------
def _base_args(**overrides):
    parser = create_mcp_parser()
    args, _ = parser.parse_known_args([])
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def test_unknown_auth_type_refuses():
    args = _base_args(auth_type="totally-bogus")
    with pytest.raises(_unsupported_auth_type_error()) as excinfo:
        _configure_auth(args)
    # The error must NAME the unsupported value so an operator can fix the typo.
    assert "totally-bogus" in str(excinfo.value)


def test_unsupported_auth_type_error_is_a_value_error():
    """Typed, but still catchable by the module's existing ValueError idiom."""
    assert issubclass(_unsupported_auth_type_error(), ValueError)


def test_unknown_auth_type_never_yields_an_unauthenticated_provider():
    """The point of the bug: the old code returned ``None`` (== no auth), which
    FastMCP accepts as an unauthenticated server. Assert we never get there."""
    args = _base_args(auth_type="jwt-typo")
    result = None
    try:
        result = _configure_auth(args)
    except _unsupported_auth_type_error():
        return
    pytest.fail(
        f"_configure_auth returned {result!r} for an unrecognised auth_type "
        "instead of refusing -- an UNAUTHENTICATED server"
    )


def test_env_auth_type_typo_refuses_to_build_a_server(monkeypatch):
    """argparse's ``choices=`` does NOT validate the *default*, and the default
    is ``setting("AUTH_TYPE", "none")``. This is the real BUG-CX-023 exposure
    path: a typo'd env var silently produced an unauthenticated server."""
    monkeypatch.setenv("AUTH_TYPE", "totally-bogus")
    args, _ = create_mcp_parser().parse_known_args([])
    assert args.auth_type == "totally-bogus", (
        "precondition: argparse must not validate the env-supplied default"
    )
    with pytest.raises(_unsupported_auth_type_error()):
        create_mcp_server(name="wd2-bug-sec-auth-typo", command_args=[])


@pytest.mark.parametrize("auth_type", ["none", ""])
def test_recognised_no_auth_values_are_unaffected(auth_type):
    """The fix must refuse only *unrecognised* values -- explicitly disabling
    auth remains a supported, unchanged configuration."""
    assert _configure_auth(_base_args(auth_type=auth_type)) is None
