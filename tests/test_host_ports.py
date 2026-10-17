"""Host ports: fail closed without a host, delegate once one is registered."""

from __future__ import annotations

import contextlib

import pytest

from agent_utilities import host_ports as hp


@pytest.fixture(autouse=True)
def _no_host():
    def clear():
        hp.register_native_toolset_provider(None)
        hp.register_fleet_catalog_provider(None)
        hp.register_authority_keepalive(None)
        hp.register_remote_toolset_provider(None)
        hp.register_tool_client_factory(None)
        hp.register_outbound_credential_provider(None)
        hp.register_delegated_identity_provider(None)

    clear()
    yield
    clear()


@pytest.mark.parametrize(
    "call",
    [
        lambda: hp.build_native_toolset(["a"], toolset_id="t"),
        lambda: hp.build_url_toolset("http://x"),
        lambda: hp.load_config_toolsets("cfg.json"),
        lambda: hp.open_tool_client("http://x"),
        hp.refresh_outbound_credentials,
    ],
)
def test_binding_ports_fail_closed(call):
    with pytest.raises(RuntimeError, match="graph-os"):
        call()


async def test_read_ports_report_empty_without_host():
    assert hp.native_tool_names() == frozenset()
    assert hp.fleet_catalog() == {}
    assert hp.outbound_auth() is None and hp.outbound_auth_header() == {}
    assert hp.outbound_auth_status()["ready"] is False
    assert hp.current_user_token() is None and hp.exchange_token_for_agent("x") is None
    async with hp.authority_keepalive_scope():
        pass


async def test_registered_host_is_used():
    entered: list[str] = []

    @contextlib.asynccontextmanager
    async def keepalive():
        entered.append("in")
        yield

    hp.register_native_toolset_provider(
        hp.NativeToolsetProvider(
            available_tools=lambda: ["go_a"],
            build_toolset=lambda names, tid: (names, tid),
        )
    )
    hp.register_fleet_catalog_provider(
        lambda: {" s ": hp.FleetServer(prefix="s"), "bad": object()}
    )
    hp.register_authority_keepalive(keepalive)
    hp.register_delegated_identity_provider(
        hp.DelegatedIdentityProvider(
            user_token=lambda: "u", exchange_for_agent=lambda label: f"t-{label}"
        )
    )
    assert hp.native_tool_names() == frozenset({"go_a"})
    assert hp.build_native_toolset(["go_a"], toolset_id="t") == (["go_a"], "t")
    assert list(hp.fleet_catalog()) == ["s"]
    assert (hp.current_user_token(), hp.exchange_token_for_agent("x")) == ("u", "t-x")
    async with hp.authority_keepalive_scope():
        pass
    assert entered == ["in"]


@pytest.mark.parametrize(
    ("prefix", "server", "tool", "expected"),
    [
        ("gl", "gitlab-api", "gitlab_api_get_project", "gl__get_project"),
        ("cm", "container-manager-mcp-r820", "container_manager_list", "cm__list"),
        ("x", "srv", "srv_" + "a" * 60, "x__" + "a" * 41),
    ],
)
def test_clean_tool_name(prefix, server, tool, expected):
    assert hp.clean_tool_name(prefix, server, tool) == expected
