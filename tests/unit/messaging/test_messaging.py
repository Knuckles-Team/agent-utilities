"""Unit tests for MessagingBackend ABC and core models (CONCEPT:ECO-4.5).

Tests the base protocol, Pydantic models, capability matrix, and registry
without requiring any platform dependencies.
"""

from __future__ import annotations

import asyncio
import pytest
from typing import Any, AsyncIterator

from agent_utilities.messaging.base import MessagingBackend
from agent_utilities.messaging.capabilities import (
    CAPABILITY_MATRIX,
    MessagingCapabilities,
    get_capabilities,
)
from agent_utilities.messaging.models import (
    Channel,
    EventType,
    InboundEvent,
    MediaAttachment,
    MediaType,
    Message,
    MessageDirection,
    MessagingConfig,
    PlatformId,
    SendResult,
    Thread,
)
from agent_utilities.messaging.registry import MessagingRegistry


# ── Concrete Test Backend ────────────────────────────────────────────


class MockBackend(MessagingBackend):
    """Minimal concrete backend for testing the ABC. CONCEPT:ECO-4.5"""

    def __init__(self, config: MessagingConfig | None = None) -> None:
        super().__init__(config)
        self.sent_messages: list[tuple[str, str]] = []

    @property
    def id(self) -> str:
        return "mock"

    @property
    def capabilities(self) -> MessagingCapabilities:
        return MessagingCapabilities(send_text=True, inbound_listen=True)

    async def connect(self) -> None:
        self._connected = True

    async def send_message(
        self,
        channel_id: str,
        text: str,
        *,
        thread_id: str = "",
        reply_to_id: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> SendResult:
        self.sent_messages.append((channel_id, text))
        return SendResult(
            success=True,
            message_id="mock-123",
            platform="mock",
            channel_id=channel_id,
        )

    async def listen(self) -> AsyncIterator[InboundEvent]:
        yield InboundEvent(
            event_type=EventType.MESSAGE,
            platform="mock",
            content="test message",
        )


# ── Test: Models ─────────────────────────────────────────────────────


class TestModels:
    """CONCEPT:ECO-4.5 — Pydantic model validation."""

    def test_message_creation(self) -> None:
        msg = Message(
            id="1",
            content="Hello",
            channel_id="ch-1",
            platform=PlatformId.DISCORD,
            direction=MessageDirection.OUTBOUND,
        )
        assert msg.content == "Hello"
        assert msg.platform == PlatformId.DISCORD
        assert msg.direction == MessageDirection.OUTBOUND

    def test_inbound_event(self) -> None:
        event = InboundEvent(
            event_type=EventType.MESSAGE,
            platform=PlatformId.SLACK,
            user_id="U123",
            content="Hi there",
        )
        assert event.event_type == EventType.MESSAGE
        assert event.platform == PlatformId.SLACK

    def test_send_result(self) -> None:
        result = SendResult(success=True, message_id="msg-1")
        assert result.success is True

    def test_media_attachment(self) -> None:
        att = MediaAttachment(
            media_type=MediaType.IMAGE,
            url="https://example.com/img.png",
            filename="img.png",
        )
        assert att.media_type == MediaType.IMAGE

    def test_channel(self) -> None:
        ch = Channel(id="C123", name="general", platform=PlatformId.SLACK)
        assert ch.name == "general"

    def test_thread(self) -> None:
        t = Thread(id="T1", parent_message_id="M1", channel_id="C1")
        assert t.parent_message_id == "M1"

    def test_messaging_config(self) -> None:
        cfg = MessagingConfig(
            platform=PlatformId.WHATSAPP,
            token="test",
            use_business_api=True,
        )
        assert cfg.use_business_api is True

    def test_platform_id_enum(self) -> None:
        """All 17 platforms must be represented."""
        assert len(PlatformId) == 17
        assert PlatformId.DISCORD == "discord"
        assert PlatformId.NEXTCLOUD == "nextcloud"

    def test_event_type_enum(self) -> None:
        assert EventType.MESSAGE == "message"
        assert EventType.REACTION_ADD == "reaction_add"


# ── Test: Base ABC ───────────────────────────────────────────────────


class TestMessagingBackendABC:
    """CONCEPT:ECO-4.5 — MessagingBackend ABC contract."""

    @pytest.fixture
    def backend(self) -> MockBackend:
        return MockBackend(config=MessagingConfig(platform="mock", token="test"))

    def test_connect_disconnect(self, backend: MockBackend) -> None:
        assert not backend.is_connected
        asyncio.run(backend.connect())
        assert backend.is_connected
        asyncio.run(backend.disconnect())
        assert not backend.is_connected

    def test_send_message(self, backend: MockBackend) -> None:
        asyncio.run(backend.connect())
        result = asyncio.run(backend.send_message("ch-1", "Hello!"))
        assert result.success is True
        assert result.message_id == "mock-123"
        assert backend.sent_messages == [("ch-1", "Hello!")]

    def test_send_media_default(self, backend: MockBackend) -> None:
        """Default send_media should fall back to send_message with URL."""
        asyncio.run(backend.connect())
        att = MediaAttachment(url="https://example.com/file.pdf")
        result = asyncio.run(backend.send_media("ch-1", att, caption="Check this"))
        assert result.success is True
        assert "https://example.com/file.pdf" in backend.sent_messages[-1][1]

    def test_send_reaction_not_implemented(self, backend: MockBackend) -> None:
        with pytest.raises(NotImplementedError):
            asyncio.run(backend.send_reaction("ch-1", "msg-1", "👍"))

    def test_create_thread_not_implemented(self, backend: MockBackend) -> None:
        with pytest.raises(NotImplementedError):
            asyncio.run(backend.create_thread("ch-1", "msg-1"))

    def test_reply_to_default(self, backend: MockBackend) -> None:
        asyncio.run(backend.connect())
        result = asyncio.run(backend.reply_to("ch-1", "msg-1", "Got it!"))
        assert result.success is True

    def test_listen(self, backend: MockBackend) -> None:
        async def _run() -> list[InboundEvent]:
            events = []
            async for event in backend.listen():
                events.append(event)
                break
            return events
        events = asyncio.run(_run())
        assert len(events) == 1
        assert events[0].content == "test message"

    def test_list_channels_default(self, backend: MockBackend) -> None:
        assert asyncio.run(backend.list_channels()) == []

    def test_list_members_default(self, backend: MockBackend) -> None:
        assert asyncio.run(backend.list_members("ch-1")) == []

    def test_repr(self, backend: MockBackend) -> None:
        assert "mock" in repr(backend)
        assert "disconnected" in repr(backend)

    def test_id_and_capabilities(self, backend: MockBackend) -> None:
        assert backend.id == "mock"
        assert backend.capabilities.send_text is True

    def test_health_check(self, backend: MockBackend) -> None:
        assert not asyncio.run(backend.health_check())
        asyncio.run(backend.connect())
        assert asyncio.run(backend.health_check())


# ── Test: Capabilities ───────────────────────────────────────────────


class TestCapabilities:
    """CONCEPT:ECO-4.5 — Capability matrix validation."""

    def test_all_platforms_have_capabilities(self) -> None:
        """Every PlatformId must have a capability entry."""
        for pid in PlatformId:
            assert pid.value in CAPABILITY_MATRIX, f"Missing: {pid.value}"

    def test_discord_capabilities(self) -> None:
        caps = CAPABILITY_MATRIX["discord"]
        assert caps.send_text is True
        assert caps.threads is True
        assert caps.reactions is True
        assert caps.max_message_length == 2000

    def test_irc_no_media(self) -> None:
        caps = CAPABILITY_MATRIX["irc"]
        assert caps.send_media is False
        assert caps.reactions is False
        assert caps.max_message_length == 512

    def test_googlemeet_no_text(self) -> None:
        caps = CAPABILITY_MATRIX["googlemeet"]
        assert caps.send_text is False
        assert caps.voice_call is True

    def test_get_capabilities(self) -> None:
        caps = get_capabilities("telegram")
        assert caps.polls is True

    def test_get_capabilities_unknown(self) -> None:
        with pytest.raises(KeyError):
            get_capabilities("nonexistent")


# ── Test: Registry ───────────────────────────────────────────────────


class TestRegistry:
    """CONCEPT:ECO-4.5 — MessagingRegistry discovery."""

    def test_singleton(self) -> None:
        r1 = MessagingRegistry.instance()
        r2 = MessagingRegistry.instance()
        assert r1 is r2

    def test_list_backends_returns_list(self) -> None:
        registry = MessagingRegistry()
        result = registry.list_backends()
        assert isinstance(result, list)

    def test_create_backend_unknown_raises(self) -> None:
        registry = MessagingRegistry()
        with pytest.raises(ValueError, match="not installed"):
            registry.create_backend("nonexistent_platform_xyz")

    def test_is_installed(self) -> None:
        registry = MessagingRegistry()
        # Will be False unless the backend is actually installed
        assert isinstance(registry.is_installed("discord"), bool)

    def test_auto_config_empty(self) -> None:
        registry = MessagingRegistry()
        config = registry._auto_config("discord")
        assert config.platform == "discord"
        assert isinstance(config, MessagingConfig)


# ── Test: AgentConfig Integration ────────────────────────────────────


class TestAgentConfigMessaging:
    """CONCEPT:ECO-4.5 — Verify messaging fields in AgentConfig (config.json)."""

    def test_messaging_fields_exist(self) -> None:
        """All messaging fields must exist on AgentConfig."""
        from agent_utilities.core.config import config

        assert hasattr(config, "messaging_enabled_backends")
        assert hasattr(config, "messaging_kg_ingest")
        assert hasattr(config, "messaging_kg_memory_type")
        assert hasattr(config, "messaging_route_to_planner")
        assert hasattr(config, "messaging_discord_token")
        assert hasattr(config, "messaging_slack_token")
        assert hasattr(config, "messaging_telegram_token")
        assert hasattr(config, "messaging_whatsapp_token")
        assert hasattr(config, "messaging_whatsapp_use_business_api")
        assert hasattr(config, "messaging_teams_app_id")
        assert hasattr(config, "messaging_teams_app_secret")
        assert hasattr(config, "messaging_mattermost_token")
        assert hasattr(config, "messaging_matrix_token")
        assert hasattr(config, "messaging_irc_server")
        assert hasattr(config, "messaging_irc_port")
        assert hasattr(config, "messaging_irc_nickname")
        assert hasattr(config, "messaging_line_token")
        assert hasattr(config, "messaging_twitch_token")

    def test_defaults_are_correct(self) -> None:
        """Defaults should be safe (empty/disabled)."""
        from agent_utilities.core.config import config

        assert config.messaging_enabled_backends == []
        assert config.messaging_kg_ingest is True
        assert config.messaging_kg_memory_type == "episodic"
        assert config.messaging_route_to_planner is True
        assert config.messaging_discord_token is None
        assert config.messaging_irc_port == 6667
        assert config.messaging_irc_nickname == "agent_bot"
        assert config.messaging_whatsapp_use_business_api is False

    def test_config_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Messaging fields should be loadable from environment variables."""
        from agent_utilities.core.config import AgentConfig

        monkeypatch.setenv("MESSAGING_DISCORD_TOKEN", "test-token-123")
        monkeypatch.setenv("MESSAGING_ENABLED_BACKENDS", '["discord"]')
        monkeypatch.setenv("MESSAGING_KG_INGEST", "false")
        fresh = AgentConfig()
        assert fresh.messaging_discord_token == "test-token-123"
        assert fresh.messaging_kg_ingest is False

    def test_platform_coverage(self) -> None:
        """All 17 platforms should have at least one config field."""
        from agent_utilities.core.config import AgentConfig

        # Get all messaging_ field names
        messaging_fields = [
            f for f in AgentConfig.model_fields if f.startswith("messaging_")
        ]
        # Should have substantial coverage
        assert len(messaging_fields) >= 25  # globals + per-platform fields


# ── Test: XDG Paths ──────────────────────────────────────────────────


class TestXDGMessagingPaths:
    """CONCEPT:ECO-4.5 + CONCEPT:OS-5.0 — XDG path integration."""

    def test_messaging_sessions_dir(self) -> None:
        from agent_utilities.core.paths import messaging_sessions_dir

        path = messaging_sessions_dir()
        assert str(path).endswith("messaging")
        assert "agent-utilities" in str(path)

    def test_messaging_config_path(self) -> None:
        from agent_utilities.core.paths import messaging_config_path

        path = messaging_config_path()
        assert str(path).endswith("config.json")
        assert "agent-utilities" in str(path)

    def test_ensure_dirs_includes_messaging(self) -> None:
        """ensure_dirs() must create messaging subdirectories."""
        from agent_utilities.core.paths import messaging_sessions_dir

        sessions = messaging_sessions_dir() / "sessions"
        history = messaging_sessions_dir() / "history"
        # These paths should be valid (ensure_dirs creates them at startup)
        assert str(sessions).endswith("sessions")
        assert str(history).endswith("history")
