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
    with pytest.raises(RuntimeError, match="arbora"):
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


@pytest.mark.spec("AU-BOUNDARY-R043.5.2")
async def test_scaffold_publisher_is_inactive_before_host_close():
    from contextlib import asynccontextmanager

    from agent_utilities.harness.plan import PlanState

    retained = []
    published = []

    class Publisher:
        async def publish(self, revision, items):
            published.append(revision)

    @asynccontextmanager
    async def factory(request):
        yield Publisher()
        with pytest.raises(RuntimeError, match="scope is closed"):
            await retained[0].publish(2, ())

    hp.register_scaffold_publication_factory(factory)
    async with hp.scaffold_publication_scope(object(), PlanState()) as publisher:
        retained.append(publisher)
        await publisher.publish(1, ())
    with pytest.raises(RuntimeError, match="scope is closed"):
        await retained[0].publish(3, ())
    assert published == [0, 1]


@pytest.mark.spec("AU-BOUNDARY-R043.5.2")
@pytest.mark.parametrize("bound,enabled", [(False, True), (True, False)])
async def test_absent_or_disabled_scaffold_has_no_publication_binding(bound, enabled):
    from agent_utilities.harness.plan import PlanState

    def factory(request):
        pytest.fail("disabled scaffold must not bind a publisher")

    hp.register_scaffold_publication_factory(factory if bound else None)
    async with hp.scaffold_publication_scope(
        object(), PlanState() if enabled else None
    ) as publisher:
        assert publisher is None


@pytest.mark.spec("AU-BOUNDARY-R043.5.2")
async def test_cancellation_during_publication_closes_host_scope():
    import asyncio
    from contextlib import asynccontextmanager

    from agent_utilities.harness.plan import PlanState

    entered = asyncio.Event()
    closed = []

    class Publisher:
        async def publish(self, revision, items):
            entered.set()
            await asyncio.Future()

    @asynccontextmanager
    async def factory(request):
        try:
            yield Publisher()
        finally:
            closed.append(True)

    async def publish():
        async with hp.scaffold_publication_scope(object(), PlanState()):
            pytest.fail("initial publication cannot complete")

    hp.register_scaffold_publication_factory(factory)
    task = asyncio.create_task(publish())
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert closed == [True]


@pytest.mark.spec("AU-BOUNDARY-R043.5.2")
@pytest.mark.parametrize("cancelled", [False, True])
async def test_host_close_failure_preserves_body_failure(cancelled):
    import asyncio
    from contextlib import asynccontextmanager

    from agent_utilities.harness.plan import PlanState

    class Publisher:
        async def publish(self, revision, items):
            pass

    @asynccontextmanager
    async def factory(request):
        try:
            yield Publisher()
        finally:
            raise ValueError("private-close-details")

    hp.register_scaffold_publication_factory(factory)
    failure = asyncio.CancelledError() if cancelled else LookupError("original")
    with pytest.raises(type(failure)) as caught:
        async with hp.scaffold_publication_scope(object(), PlanState()):
            raise failure
    assert caught.value is failure


@pytest.mark.spec("AU-BOUNDARY-R043.5.2")
async def test_failed_publication_keeps_accepted_revision():
    from contextlib import asynccontextmanager

    from agent_utilities.harness.plan import PlanItem, PlanState

    class Publisher:
        async def publish(self, revision, items):
            if revision:
                raise ValueError("uncertain external outcome")

    @asynccontextmanager
    async def factory(request):
        yield Publisher()

    hp.register_scaffold_publication_factory(factory)
    state = PlanState()
    accepted = (PlanItem(text="accepted", status="done"),)
    with pytest.raises(RuntimeError, match="scaffold publication failed"):
        async with hp.scaffold_publication_scope(object(), state) as publisher:
            revision, items = state.replace_versioned(accepted)
            await publisher.publish(revision, items)
    assert state.read_versioned() == (1, accepted)
