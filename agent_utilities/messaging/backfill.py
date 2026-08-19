from __future__ import annotations

"""Messaging conversation-history backfill (CONCEPT:AU-ECO.messaging.conversational-history-backfill, BUG-041).

**The problem this closes.** ``InboundMessage``/``Thread`` KG nodes (``messaging/inbox.py``,
``messaging/router.py``, ``messaging/enrichment.py``) are written ONCE at live intake. Before
this module there was no reingest path for ANY of the 17 messaging backends, and the
existing prose ("Telegram's ``getUpdates`` means acknowledged updates are gone upstream")
was written as if it applied to every platform — it does not. Some platform APIs genuinely
retain and re-serve message history; some genuinely do not. This module recovers the
former and refuses the latter loudly rather than pretending either way.

**Per-platform recoverability** (BUG-041 audit; see the module constants below and
``docs/guides/knowledge-graph.md`` "Low-Signal Pruning" for the retention-policy
consequence of this split):

===========  ============  ================================================================
Platform     Recoverable?  Why
===========  ============  ================================================================
discord      yes, wired    ``GET /channels/{id}/messages`` — full history, bot token
slack        yes, wired    ``conversations.history`` — full history, bot token
matrix       yes, wired    ``/rooms/{id}/messages`` (homeserver retains room history)
nextcloud    yes, wired    Talk chat API retains + serves history, basic auth
voicecall    yes, wired    Twilio retains Message/Call resources, basic auth (SID/token);
                            single-page (most-recent ``PageSize``) here — see the preset's
                            comment for a real ``next_url_field``/httpx query-merge bug that
                            blocks full pagination, tracked as a follow-up
mattermost   yes, NOT wired  full REST history exists, but the payload is a dict-of-posts
                            (not a JSON array) so it doesn't fit :class:`RestJsonConnector`'s
                            shape, AND the fleet already ships a dedicated ``mattermost-mcp``
                            server — the correct reuse path is an ``mcp_tool`` preset against
                            THAT server (CONCEPT:AU-KG.ingest.mcp-tool-connector), not a
                            bespoke REST shape here.
googlechat   yes, NOT wired  ``spaces.messages.list`` retains + serves history, but auth is a
                            service-account OAuth2 token exchange, not a static bearer
                            secret — doesn't fit this module's credential shape.
teams        yes, NOT wired  Microsoft Graph ``chatMessage`` list retains + serves history,
                            but requires a Graph API app registration/consent distinct from
                            the Bot Framework password this repo's ``teams`` backend uses.
imessage     yes, NOT wired  full history persists in the LOCAL macOS Messages.app
                            ``chat.db`` — recoverable, but it is a local sqlite read, not an
                            upstream REST re-fetch; a different connector shape entirely.
telegram     NO            Bot API ``getUpdates``/webhook: an acknowledged update is gone
                            upstream — Telegram will not re-serve it, in either transport
                            mode. No history endpoint exists for bots.
whatsapp     NO            Business Cloud API is webhook-push only; no message-history
                            endpoint for previously-received messages.
signal       NO            End-to-end relay; the server retains nothing after delivery.
irc          NO            Ephemeral protocol; no native history (assumes no bouncer).
line         NO            Messaging API is webhook-push only; no history-retrieval
                            endpoint.
twitch       NO            Regular chat is a live IRC stream; no chat-message history API.
synology     NO            Integrated here via incoming/outgoing webhook only; no history
                            API reached by this integration shape.
googlemeet   NO            In-call text chat is not a retrievable REST resource.
===========  ============  ================================================================

**Mechanism.** Reuses :class:`RestJsonConnector`
(``protocols/source_connectors/connectors/rest.py``) for pagination, dotted-field
extraction, and the fail-closed SSRF/egress boundary (``http_safety.py``) instead of
reimplementing any of that — "one reuse path" (AGENTS.md). Recovered messages are written
back through :func:`agent_utilities.messaging.inbox.record_inbound` with
``status="backfilled"`` and the ORIGINAL platform timestamp, using the exact same
content-addressed id scheme (:func:`agent_utilities.messaging.inbox._inbox_id`) live intake
uses — so backfilling a message that was never deleted is a no-op (idempotent upsert), and
backfilling one that WAS deleted reconstructs the identical node.
"""

import base64
import logging
import re
from typing import Any, TypedDict
from urllib.parse import quote

from agent_utilities.core.config import setting
from agent_utilities.messaging.inbox import record_inbound
from agent_utilities.protocols.source_connectors.connectors.rest import (
    RestJsonConnector,
)

logger = logging.getLogger(__name__)

__all__ = [
    "RECOVERABLE_PLATFORM_PRESETS",
    "UNRECOVERABLE_PLATFORMS",
    "NOT_YET_IMPLEMENTED_RECOVERABLE_PLATFORMS",
    "MessagingBackfillError",
    "PlatformNotRecoverableError",
    "BackfillResult",
    "backfill_platform_history",
]


class MessagingBackfillError(RuntimeError):
    """A configuration or transport failure while backfilling message history."""


class PlatformNotRecoverableError(MessagingBackfillError):
    """The requested platform has no server-side history a bot/service account can re-fetch
    (BUG-041 per-platform recoverability audit — see this module's docstring table)."""


#: Platforms confirmed to have NO way to re-fetch message history once
#: acknowledged/delivered. Deliberately an explicit refusal list (checked FIRST) rather
#: than "anything missing from the preset dict" — a preset that hasn't been added yet must
#: raise the same way a genuinely-unrecoverable platform does, and this set makes the
#: reason legible instead of an implicit KeyError.
UNRECOVERABLE_PLATFORMS: frozenset[str] = frozenset(
    {
        "telegram",
        "whatsapp",
        "signal",
        "irc",
        "line",
        "twitch",
        "synology",
        "googlemeet",
    }
)

#: Platforms whose upstream API DOES retain re-fetchable history but that are not wired
#: into :data:`RECOVERABLE_PLATFORM_PRESETS` in this change — see the module docstring
#: table for exactly why each one doesn't fit this module's REST/static-credential shape,
#: and what the correct follow-up connector is.
NOT_YET_IMPLEMENTED_RECOVERABLE_PLATFORMS: frozenset[str] = frozenset(
    {"mattermost", "googlechat", "teams", "imessage"}
)


class _Preset(TypedDict, total=False):
    url_template: str
    records_field: str
    id_field: str
    text_field: str
    updated_field: str
    cursor_field: str
    cursor_param: str
    next_url_field: str
    params: dict[str, str]
    auth: str  # "bot" | "bearer" | "basic"
    token_env: str  # the secret (bot token / bearer token / basic-auth password)
    url_env: str  # resolves {base_url} in url_template, when the template needs it
    user_env: (
        str  # resolves {account_sid} AND/OR the "user" half of basic auth, when needed
    )


#: Declarative per-platform history-fetch config (CONCEPT:AU-KG.ingest.mcp-tool-connector
#: spirit — data, not code, exactly like ``MCP_TOOL_PRESETS``). ``{channel_id}``/``{base_url}``
#: are filled in by :func:`backfill_platform_history`. Every credential is resolved through
#: the SAME env vars the live backend already uses (``messaging/registry.py``
#: ``_auto_config``'s ``native_token_vars``/``app_id_vars``) — no new configuration surface.
RECOVERABLE_PLATFORM_PRESETS: dict[str, _Preset] = {
    "discord": {
        "url_template": "https://discord.com/api/v10/channels/{channel_id}/messages",
        "records_field": "",  # the response body IS the array
        "id_field": "id",
        "text_field": "content",
        "updated_field": "timestamp",
        "cursor_field": "id",  # last record's id, echoed back as `before`
        "cursor_param": "before",
        "params": {"limit": "100"},
        "auth": "bot",
        "token_env": "DISCORD_BOT_TOKEN",
    },
    "slack": {
        "url_template": "https://slack.com/api/conversations.history",
        "records_field": "messages",
        "id_field": "ts",
        "text_field": "text",
        "updated_field": "ts",
        "cursor_field": "response_metadata.next_cursor",
        "cursor_param": "cursor",
        "params": {"limit": "200"},
        "auth": "bearer",
        "token_env": "SLACK_BOT_TOKEN",
    },
    "matrix": {
        # {base_url} is the homeserver (MATRIX_HOMESERVER); {channel_id} is the room id.
        "url_template": "{base_url}/_matrix/client/v3/rooms/{channel_id}/messages",
        "records_field": "chunk",
        "id_field": "event_id",
        "text_field": "content.body",
        "updated_field": "origin_server_ts",
        "cursor_field": "end",
        "cursor_param": "from",
        "params": {"dir": "b", "limit": "100"},
        "auth": "bearer",
        "token_env": "MATRIX_ACCESS_TOKEN",
        "url_env": "MATRIX_HOMESERVER",
    },
    "nextcloud": {
        # {base_url} is the Nextcloud instance (NEXTCLOUD_URL); {channel_id} is the Talk room token.
        "url_template": "{base_url}/ocs/v2.php/apps/spreed/api/v4/chat/{channel_id}",
        "records_field": "ocs.data",
        "id_field": "id",
        "text_field": "message",
        "updated_field": "timestamp",
        "params": {"lookIntoFuture": "0", "limit": "200"},
        "auth": "basic",
        "token_env": "NEXTCLOUD_TOKEN",
        "url_env": "NEXTCLOUD_URL",
        "user_env": "NEXTCLOUD_USER",
    },
    "voicecall": {
        # {account_sid} fills BOTH the URL path AND the basic-auth "user" half — Twilio
        # scopes history by account, not by "channel" (channel_id is accepted but unused).
        #
        # Deliberately single-page (most-recent ``PageSize`` messages), NOT paginated via
        # Twilio's ``next_page_uri``: that field is a full URL carrying its own query
        # string, and ``RestJsonConnector``'s ``next_url_field`` path hands it straight to
        # ``http_safety.safe_get_bytes``, which builds the logical request URL via
        # ``httpx.URL(current, params=current_params)`` — httpx replaces (never merges) an
        # existing query when ``params`` is passed at all, even ``{}``, so the embedded
        # ``?Page=...`` is silently discarded and every "next" page re-fetches page 1
        # forever. This is a genuine latent bug in the shared connector/http_safety layer,
        # not specific to Twilio; tracked as a follow-up rather than fixed here (out of
        # this BUG-041 lane's scope — it's shared, security-critical infra every native
        # connector depends on). ``cursor_field``-driven pagination (Discord/Matrix/Slack
        # above) is unaffected: it always re-issues against the ORIGINAL bare ``url``, never
        # a response-supplied URL that already carries a query string.
        "url_template": "https://api.twilio.com/2010-04-01/Accounts/{account_sid}/Messages.json",
        "records_field": "messages",
        "id_field": "sid",
        "text_field": "body",
        "updated_field": "date_sent",
        "params": {"PageSize": "50"},
        "auth": "basic",
        "token_env": "TWILIO_AUTH_TOKEN",
        "user_env": "TWILIO_ACCOUNT_SID",
    },
}


class BackfillResult(TypedDict):
    platform: str
    channel_id: str
    recovered: int
    errors: int


def _basic_auth_header(user: str, secret: str) -> str:
    raw = f"{user}:{secret}".encode()
    return "Basic " + base64.b64encode(raw).decode("ascii")


def _resolve_headers(preset: _Preset, *, token: str, user: str) -> dict[str, str]:
    auth = preset.get("auth", "")
    if auth == "bot":
        return {"Authorization": f"Bot {token}"}
    if auth == "bearer":
        return {"Authorization": f"Bearer {token}"}
    if auth == "basic":
        return {"Authorization": _basic_auth_header(user, token)}
    raise MessagingBackfillError(f"Unknown auth kind {auth!r} in preset")


class _Credentials(TypedDict):
    token: str
    url: str
    user: str


_MAX_CHANNEL_ID_BYTES = 256


def _validate_channel_id(platform: str, channel_id: str) -> str:
    """Validate one bounded platform identifier before URL construction.

    Path-bearing presets accept only their platform's identifier grammar; the
    connector still percent-encodes the accepted value as one path segment.
    Voicecall intentionally ignores ``channel_id`` (Twilio history is scoped
    by the configured account), but keeps the same bounded/control-character
    guard so it cannot become a future URL injection when the preset evolves.
    """
    if not isinstance(channel_id, str) or not channel_id:
        raise MessagingBackfillError(f"{platform}: channel_id must be non-empty")
    if len(channel_id.encode("utf-8")) > _MAX_CHANNEL_ID_BYTES:
        raise MessagingBackfillError(
            f"{platform}: channel_id exceeds {_MAX_CHANNEL_ID_BYTES} bytes"
        )
    if any(ord(char) < 0x20 or ord(char) == 0x7F for char in channel_id):
        raise MessagingBackfillError(
            f"{platform}: channel_id contains control characters"
        )
    if platform == "discord" and not re.fullmatch(r"[0-9]{1,32}", channel_id):
        raise MessagingBackfillError("discord: channel_id must be a numeric channel id")
    if platform == "slack" and not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", channel_id):
        raise MessagingBackfillError("slack: channel_id has invalid characters")
    if platform == "matrix" and not re.fullmatch(
        r"![^\s/?#\\]{1,127}:[^\s/?#\\]{1,127}", channel_id
    ):
        raise MessagingBackfillError("matrix: channel_id must be a room id")
    if platform == "nextcloud" and not re.fullmatch(
        r"[A-Za-z0-9_-]{1,128}", channel_id
    ):
        raise MessagingBackfillError("nextcloud: channel_id has invalid characters")
    return channel_id


def _resolve_credentials(platform: str, preset: _Preset) -> _Credentials:
    """Resolve every configured credential env var — the SAME ones the live backend reads
    (``messaging/registry.py`` ``_auto_config``)."""
    resolved: dict[str, str] = {"token": "", "url": "", "user": ""}
    for field, key in (
        ("token", "token_env"),
        ("url", "url_env"),
        ("user", "user_env"),
    ):
        env_name = str(preset.get(key, "") or "")  # type: ignore[literal-required]
        if not env_name:
            continue
        value = str(setting(env_name, "") or "").strip()
        if not value:
            raise MessagingBackfillError(
                f"{platform}: {env_name} is not configured — cannot backfill without the "
                "same credential the live backend uses."
            )
        resolved[field] = value
    return _Credentials(
        token=resolved["token"], url=resolved["url"], user=resolved["user"]
    )


def _build_connector(
    platform: str,
    preset: _Preset,
    *,
    channel_id: str,
    fetch_fn: Any = None,
    transport: Any = None,
    allowed_private_hosts: list[str] | None = None,
) -> RestJsonConnector:
    if fetch_fn is not None:
        # Not a credential: a caller-supplied fetch_fn replaces the built-in
        # credential-based fetch entirely (headers is set to None right below),
        # so this placeholder is never read for authentication.
        creds: _Credentials = _Credentials(token="", url="", user="")  # nosec B106
        headers = None
    else:
        creds = _resolve_credentials(platform, preset)
        headers = _resolve_headers(preset, token=creds["token"], user=creds["user"])
    # Twilio's account SID fills both the URL path AND (for its basic-auth preset) the
    # "user" half — resolved once above, applied to both here.
    account_sid = creds["user"] if platform == "voicecall" else ""
    params = dict(preset.get("params", {}))
    if platform == "slack":
        # Slack's endpoint is not channel-scoped by path; preserve the
        # requested channel in its required query parameter instead.
        params["channel"] = channel_id
    url = preset["url_template"].format(
        channel_id=quote(channel_id, safe=""),
        base_url=creds["url"].rstrip("/"),
        account_sid=account_sid,
    )
    return RestJsonConnector(
        url=url,
        records_field=preset.get("records_field", ""),
        id_field=preset.get("id_field", "id"),
        text_field=preset.get("text_field", "text"),
        updated_field=preset.get("updated_field", ""),
        cursor_field=preset.get("cursor_field", ""),
        cursor_param=preset.get("cursor_param", "cursor"),
        next_url_field=preset.get("next_url_field", ""),
        params=params,
        headers=headers,
        fetch_fn=fetch_fn,
        transport=transport,
        allowed_private_hosts=allowed_private_hosts,
    )


def backfill_platform_history(
    engine: Any,
    *,
    platform: str,
    channel_id: str,
    session: str = "",
    fetch_fn: Any = None,
    transport: Any = None,
    allowed_private_hosts: list[str] | None = None,
) -> BackfillResult:
    """Recover a channel/room's message history into durable ``InboundMessage`` nodes.

    CONCEPT:AU-ECO.messaging.conversational-history-backfill (BUG-041). Refuses outright —
    never a silent no-op — for a platform in :data:`UNRECOVERABLE_PLATFORMS` (raises
    :class:`PlatformNotRecoverableError`) or one with no preset yet
    (:class:`MessagingBackfillError`, naming :data:`NOT_YET_IMPLEMENTED_RECOVERABLE_PLATFORMS`
    when applicable). ``fetch_fn``/``transport`` are the same offline-test injection seam
    :class:`RestJsonConnector` already exposes — production code leaves both ``None`` and a
    real bounded, redirect-validated ``httpx`` call is made using the live backend's own
    configured credential.

    Returns a count of recovered/errored rows; never raises per-row (a single malformed
    history record is skipped and counted as an error, not a fatal abort of the whole
    backfill).
    """
    if platform in UNRECOVERABLE_PLATFORMS:
        raise PlatformNotRecoverableError(
            f"{platform!r} has no server-side message history a bot/service account can "
            "re-fetch (BUG-041) — see agent_utilities/messaging/backfill.py's module "
            "docstring for the per-platform recoverability table."
        )
    preset = RECOVERABLE_PLATFORM_PRESETS.get(platform)
    if preset is None:
        if platform in NOT_YET_IMPLEMENTED_RECOVERABLE_PLATFORMS:
            raise MessagingBackfillError(
                f"{platform!r} history IS recoverable upstream but has no preset wired "
                "here yet — see agent_utilities/messaging/backfill.py's module docstring "
                "for why and what the correct connector shape is."
            )
        raise MessagingBackfillError(
            f"Unknown platform {platform!r} — not in RECOVERABLE_PLATFORM_PRESETS, "
            "UNRECOVERABLE_PLATFORMS, or NOT_YET_IMPLEMENTED_RECOVERABLE_PLATFORMS."
        )

    channel_id = _validate_channel_id(platform, channel_id)

    connector = _build_connector(
        platform,
        preset,
        channel_id=channel_id,
        fetch_fn=fetch_fn,
        transport=transport,
        allowed_private_hosts=allowed_private_hosts,
    )

    recovered = 0
    errors = 0
    for doc in connector.load():
        try:
            iid = record_inbound(
                engine,
                platform=platform,
                channel_id=channel_id,
                message_id=doc.id,
                text=doc.text,
                session=session,
                status="backfilled",
                received_at=doc.updated_at,
            )
        except Exception as exc:  # noqa: BLE001 — one bad row must not abort the whole backfill
            logger.warning(
                "[CONCEPT:AU-ECO.messaging.conversational-history-backfill] "
                "%s: failed to record backfilled message %s: %s",
                platform,
                doc.id,
                exc,
            )
            errors += 1
            continue
        if iid:
            recovered += 1
        else:
            errors += 1

    logger.info(
        "[CONCEPT:AU-ECO.messaging.conversational-history-backfill] "
        "%s/%s backfill: recovered=%d errors=%d",
        platform,
        channel_id,
        recovered,
        errors,
    )
    return BackfillResult(
        platform=platform, channel_id=channel_id, recovered=recovered, errors=errors
    )
