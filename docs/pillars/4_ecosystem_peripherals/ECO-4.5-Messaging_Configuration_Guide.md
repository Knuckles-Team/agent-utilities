# AU-ECO.toolkit.journey-map-milestones — Messaging Configuration Guide

> **CONCEPT:AU-ECO.messaging.native-backend-abstraction** | Pillar 4: Ecosystem & Peripherals
>
> Complete reference for configuring all 17 messaging backends through the
> unified XDG `config.json` system. Every environment variable and config
> key documented exhaustively.

---

## Configuration Architecture

All messaging configuration lives in a single file:

```
~/.config/agent-utilities/config.json
```

This file is loaded at startup by `_load_xdg_json_config()` in `core/config.py`.
Every key in `config.json` is automatically promoted to an environment variable
(uppercased). This means:

```json
{
    "messaging_discord_token": "BOT_TOKEN_HERE"
}
```

...becomes `MESSAGING_DISCORD_TOKEN=BOT_TOKEN_HERE` in the process environment,
which the `MessagingRegistry._auto_config()` reads.

### Priority Chain

Configuration values are resolved in this order (first wins):

1. **Environment variable** — explicit `MESSAGING_DISCORD_TOKEN=...`
2. **XDG config.json** — `~/.config/agent-utilities/config.json`
3. **`.env` file** — project-local `.env` file
4. **Platform-native env vars** — `DISCORD_BOT_TOKEN` (fallback)

---

## Global Settings

These control the overall messaging subsystem behavior.

| config.json Key | Env Var | Type | Default | Description |
|---|---|---|---|---|
| `messaging_enabled_backends` | `MESSAGING_ENABLED_BACKENDS` | `list[str]` | `[]` | Backend IDs to auto-connect on startup. Example: `["discord", "slack"]` |
| `messaging_kg_ingest` | `MESSAGING_KG_INGEST` | `bool` | `true` | Auto-ingest all messages into the Knowledge Graph |
| `messaging_kg_memory_type` | `MESSAGING_KG_MEMORY_TYPE` | `str` | `"episodic"` | KG memory tier: `"episodic"`, `"semantic"`, or `"procedural"` |
| `messaging_route_to_planner` | `MESSAGING_ROUTE_TO_PLANNER` | `bool` | `true` | Route inbound events to Planner Graph Agent (CONCEPT:AU-ORCH.planning.recursion-nesting-depth) |

---

## Generic Pattern Variables

Every backend supports these generic env vars via the `MESSAGING_<PLATFORM>_` prefix.
The registry's `_auto_config()` reads them automatically:

| Pattern | Example (Discord) | Description |
|---|---|---|
| `MESSAGING_<PLATFORM>_TOKEN` | `MESSAGING_DISCORD_TOKEN` | Primary auth token |
| `MESSAGING_<PLATFORM>_APP_ID` | `MESSAGING_DISCORD_APP_ID` | Application/client ID |
| `MESSAGING_<PLATFORM>_APP_SECRET` | `MESSAGING_DISCORD_APP_SECRET` | Application secret |
| `MESSAGING_<PLATFORM>_WEBHOOK_URL` | `MESSAGING_DISCORD_WEBHOOK_URL` | Webhook URL for inbound events |
| `MESSAGING_<PLATFORM>_WEBHOOK_PORT` | `MESSAGING_DISCORD_WEBHOOK_PORT` | Local port for webhook server |

Valid `<PLATFORM>` values (uppercase): `DISCORD`, `SLACK`, `TELEGRAM`, `WHATSAPP`,
`TEAMS`, `GOOGLECHAT`, `GOOGLEMEET`, `MATTERMOST`, `MATRIX`, `IRC`, `SIGNAL`,
`IMESSAGE`, `LINE`, `TWITCH`, `SYNOLOGY`, `VOICECALL`, `NEXTCLOUD`.

---

## Per-Platform Configuration

### Discord

| config.json Key | Env Var | Fallback Env Var | Type | Default | Description |
|---|---|---|---|---|---|
| `messaging_discord_token` | `MESSAGING_DISCORD_TOKEN` | `DISCORD_BOT_TOKEN` | `str` | `null` | Bot token from Discord Developer Portal |

**Setup:**
1. Create a bot at [discord.com/developers](https://discord.com/developers)
2. Enable `MESSAGE_CONTENT`, `GUILD_MEMBERS`, and `REACTIONS` intents
3. Add token to `config.json`

---

### Slack

| config.json Key | Env Var | Fallback Env Var | Type | Default | Description |
|---|---|---|---|---|---|
| `messaging_slack_token` | `MESSAGING_SLACK_TOKEN` | `SLACK_BOT_TOKEN` | `str` | `null` | Bot OAuth token (`xoxb-...`) |
| `messaging_slack_app_token` | `MESSAGING_SLACK_APP_TOKEN` | `SLACK_APP_TOKEN` | `str` | `null` | App-level token (`xapp-...`) for Socket Mode |

**Setup:**
1. Create an app at [api.slack.com/apps](https://api.slack.com/apps)
2. Enable Socket Mode and subscribe to `message` events
3. Install to workspace and copy both tokens

---

### Telegram

| config.json Key | Env Var | Fallback Env Var | Type | Default | Description |
|---|---|---|---|---|---|
| `messaging_telegram_token` | `MESSAGING_TELEGRAM_TOKEN` | `TELEGRAM_BOT_TOKEN` | `str` | `null` | Bot token from @BotFather |

**Setup:**
1. Talk to [@BotFather](https://t.me/BotFather) on Telegram
2. Create a new bot and copy the token

---

### WhatsApp

| config.json Key | Env Var | Fallback Env Var | Type | Default | Description |
|---|---|---|---|---|---|
| `messaging_whatsapp_token` | `MESSAGING_WHATSAPP_TOKEN` | `WHATSAPP_TOKEN` | `str` | `null` | Business API access token or bridge token |
| `messaging_whatsapp_phone_number_id` | `MESSAGING_WHATSAPP_PHONE_NUMBER_ID` | `WHATSAPP_PHONE_NUMBER_ID` | `str` | `null` | Business API phone number ID |
| `messaging_whatsapp_use_business_api` | `MESSAGING_WHATSAPP_USE_BUSINESS_API` | — | `bool` | `false` | `true` for official API, `false` for neonize bridge |

**Dual Mode:**
- **Business API** (`use_business_api: true`): Official WhatsApp Cloud API via Meta
- **Bridge** (`use_business_api: false`): Local WhatsApp Web connection via `neonize` (QR code auth, stored in `~/.local/share/agent-utilities/messaging/sessions/`)

---

### Microsoft Teams

| config.json Key | Env Var | Fallback Env Var | Type | Default | Description |
|---|---|---|---|---|---|
| `messaging_teams_app_id` | `MESSAGING_TEAMS_APP_ID` | `MSTEAMS_APP_ID` | `str` | `null` | Bot Framework app ID |
| `messaging_teams_app_secret` | `MESSAGING_TEAMS_APP_SECRET` | `MSTEAMS_APP_PASSWORD` | `str` | `null` | Bot Framework app password |
| — | `MESSAGING_TEAMS_WEBHOOK_URL` | — | `str` | `""` | Incoming webhook URL for Teams messages |
| — | `MESSAGING_TEAMS_WEBHOOK_PORT` | — | `int` | `0` | Local port for webhook server |

---

### Google Chat

| config.json Key | Env Var | Fallback Env Var | Type | Default | Description |
|---|---|---|---|---|---|
| `messaging_googlechat_token` | `MESSAGING_GOOGLECHAT_TOKEN` | `GOOGLE_CHAT_SERVICE_ACCOUNT` | `str` | `null` | Path to service account JSON file |
| — | `MESSAGING_GOOGLECHAT_APP_ID` | `GOOGLE_CHAT_PROJECT_ID` | `str` | `""` | GCP project ID |

**Setup:**
1. Enable Chat API in Google Cloud Console
2. Create a service account and download JSON key
3. Set path in `config.json`

---

### Google Meet

| config.json Key | Env Var | Fallback Env Var | Type | Default | Description |
|---|---|---|---|---|---|
| `messaging_googlemeet_token` | `MESSAGING_GOOGLEMEET_TOKEN` | `GOOGLE_MEET_SERVICE_ACCOUNT` | `str` | `null` | Path to service account JSON file |

> **Note:** Google Meet is a voice/video platform. The `send_message()` method is not supported — use `create_meeting()` instead.

---

### Mattermost

| config.json Key | Env Var | Fallback Env Var | Type | Default | Description |
|---|---|---|---|---|---|
| `messaging_mattermost_token` | `MESSAGING_MATTERMOST_TOKEN` | `MATTERMOST_TOKEN` | `str` | `null` | Personal access token |
| `messaging_mattermost_url` | `MESSAGING_MATTERMOST_URL` | `MATTERMOST_URL` | `str` | `null` | Server URL (e.g. `https://mm.company.com`) |

---

### Matrix

| config.json Key | Env Var | Fallback Env Var | Type | Default | Description |
|---|---|---|---|---|---|
| `messaging_matrix_token` | `MESSAGING_MATRIX_TOKEN` | `MATRIX_ACCESS_TOKEN` | `str` | `null` | Access token |
| `messaging_matrix_homeserver` | `MESSAGING_MATRIX_HOMESERVER` | `MATRIX_HOMESERVER` | `str` | `null` | Homeserver URL (e.g. `https://matrix.org`) |
| `messaging_matrix_user_id` | `MESSAGING_MATRIX_USER_ID` | `MATRIX_USER_ID` | `str` | `null` | Full user ID (e.g. `@bot:matrix.org`) |

---

### IRC

| config.json Key | Env Var | Fallback Env Var | Type | Default | Description |
|---|---|---|---|---|---|
| `messaging_irc_server` | `MESSAGING_IRC_SERVER` | `IRC_SERVER` | `str` | `null` | IRC server hostname (e.g. `irc.libera.chat`) |
| `messaging_irc_port` | `MESSAGING_IRC_PORT` | `IRC_PORT` | `int` | `6667` | IRC server port |
| `messaging_irc_nickname` | `MESSAGING_IRC_NICKNAME` | `IRC_NICKNAME` | `str` | `"agent_bot"` | Bot nickname |
| `messaging_irc_channels` | `MESSAGING_IRC_CHANNELS` | — | `list[str]` | `[]` | Channels to auto-join (JSON array in env var) |

---

### Signal

| config.json Key | Env Var | Fallback Env Var | Type | Default | Description |
|---|---|---|---|---|---|
| `messaging_signal_token` | `MESSAGING_SIGNAL_TOKEN` | `SIGNAL_PHONE_NUMBER` | `str` | `null` | Registered phone number (e.g. `+1234567890`) |

> **Prerequisite:** `signal-cli` must be installed and registered with the phone number.

---

### iMessage

| config.json Key | Env Var | Fallback Env Var | Type | Default | Description |
|---|---|---|---|---|---|
| — | `MESSAGING_IMESSAGE_TOKEN` | — | `str` | `""` | Not required (macOS AppleScript bridge) |

> **Platform Constraint:** macOS only. Uses `osascript` to interface with Messages.app. No token required — the backend reads from the local `chat.db`.

---

### LINE

| config.json Key | Env Var | Fallback Env Var | Type | Default | Description |
|---|---|---|---|---|---|
| `messaging_line_token` | `MESSAGING_LINE_TOKEN` | `LINE_CHANNEL_ACCESS_TOKEN` | `str` | `null` | Channel access token |
| — | `MESSAGING_LINE_APP_ID` | `LINE_CHANNEL_ID` | `str` | `""` | Channel ID |
| — | `MESSAGING_LINE_WEBHOOK_URL` | — | `str` | `""` | Webhook URL for inbound messages |
| — | `MESSAGING_LINE_WEBHOOK_PORT` | — | `int` | `0` | Local port for webhook server |

---

### Twitch

| config.json Key | Env Var | Fallback Env Var | Type | Default | Description |
|---|---|---|---|---|---|
| `messaging_twitch_token` | `MESSAGING_TWITCH_TOKEN` | `TWITCH_OAUTH_TOKEN` | `str` | `null` | OAuth access token |
| `messaging_twitch_channels` | `MESSAGING_TWITCH_CHANNELS` | `TWITCH_CHANNELS` | `list[str]` | `[]` | Channels to join (comma-separated in native env var, JSON array in config.json) |

---

### Synology Chat

| config.json Key | Env Var | Fallback Env Var | Type | Default | Description |
|---|---|---|---|---|---|
| `messaging_synology_webhook_url` | `MESSAGING_SYNOLOGY_WEBHOOK_URL` | `SYNOLOGY_CHAT_WEBHOOK_URL` | `str` | `null` | Incoming webhook URL from Synology Chat integration settings |

---

### Voice Call (Twilio)

| config.json Key | Env Var | Fallback Env Var | Type | Default | Description |
|---|---|---|---|---|---|
| `messaging_voicecall_app_id` | `MESSAGING_VOICECALL_APP_ID` | `TWILIO_ACCOUNT_SID` | `str` | `null` | Twilio Account SID |
| `messaging_voicecall_token` | `MESSAGING_VOICECALL_TOKEN` | `TWILIO_AUTH_TOKEN` | `str` | `null` | Twilio Auth Token |
| `messaging_voicecall_from_number` | `MESSAGING_VOICECALL_FROM_NUMBER` | `TWILIO_FROM_NUMBER` | `str` | `null` | Twilio "from" phone number (e.g. `+1234567890`) |

---

### Nextcloud Talk

| config.json Key | Env Var | Fallback Env Var | Type | Default | Description |
|---|---|---|---|---|---|
| `messaging_nextcloud_url` | `MESSAGING_NEXTCLOUD_URL` | `NEXTCLOUD_URL` | `str` | `null` | Nextcloud server URL (e.g. `https://cloud.example.com`) |
| `messaging_nextcloud_token` | `MESSAGING_NEXTCLOUD_TOKEN` | `NEXTCLOUD_TOKEN` | `str` | `null` | App password or user token |
| `messaging_nextcloud_app_id` | `MESSAGING_NEXTCLOUD_APP_ID` | `NEXTCLOUD_USER` | `str` | `null` | Nextcloud username |

---

## Complete config.json Reference

Below is the **full** messaging section of `config.json` with every supported key:

```json
{
    "messaging_enabled_backends": ["discord", "slack", "telegram"],
    "messaging_kg_ingest": true,
    "messaging_kg_memory_type": "episodic",
    "messaging_route_to_planner": true,

    "messaging_discord_token": "",

    "messaging_slack_token": "",
    "messaging_slack_app_token": "",

    "messaging_telegram_token": "",

    "messaging_whatsapp_token": "",
    "messaging_whatsapp_phone_number_id": "",
    "messaging_whatsapp_use_business_api": false,

    "messaging_teams_app_id": "",
    "messaging_teams_app_secret": "",
    "messaging_teams_webhook_url": "",
    "messaging_teams_webhook_port": 0,

    "messaging_googlechat_token": "/path/to/service-account.json",

    "messaging_googlemeet_token": "/path/to/service-account.json",

    "messaging_mattermost_token": "",
    "messaging_mattermost_url": "https://mattermost.example.com",

    "messaging_matrix_token": "",
    "messaging_matrix_homeserver": "https://matrix.org",
    "messaging_matrix_user_id": "@bot:matrix.org",

    "messaging_irc_server": "irc.libera.chat",
    "messaging_irc_port": 6667,
    "messaging_irc_nickname": "agent_bot",
    "messaging_irc_channels": ["#general", "#dev"],

    "messaging_signal_token": "+1234567890",

    "messaging_line_token": "",
    "messaging_line_webhook_url": "",
    "messaging_line_webhook_port": 0,

    "messaging_twitch_token": "",
    "messaging_twitch_channels": ["channelname"],

    "messaging_synology_webhook_url": "https://nas.local/webapi/entry.cgi?...",

    "messaging_voicecall_app_id": "",
    "messaging_voicecall_token": "",
    "messaging_voicecall_from_number": "+1234567890",

    "messaging_nextcloud_url": "https://cloud.example.com",
    "messaging_nextcloud_token": "",
    "messaging_nextcloud_app_id": "admin"
}
```

---

## Complete Environment Variable Reference

### Global Variables

| Env Var | Description |
|---|---|
| `MESSAGING_ENABLED_BACKENDS` | JSON array of backend IDs |
| `MESSAGING_KG_INGEST` | Enable/disable KG ingestion (`true`/`false`) |
| `MESSAGING_KG_MEMORY_TYPE` | Memory tier: `episodic`, `semantic`, `procedural` |
| `MESSAGING_ROUTE_TO_PLANNER` | Route to Planner Graph Agent (`true`/`false`) |

### Discord

| Env Var | Description |
|---|---|
| `MESSAGING_DISCORD_TOKEN` | Bot token (primary) |
| `DISCORD_BOT_TOKEN` | Bot token (fallback) |

### Slack

| Env Var | Description |
|---|---|
| `MESSAGING_SLACK_TOKEN` | Bot OAuth token `xoxb-...` (primary) |
| `MESSAGING_SLACK_APP_TOKEN` | App-level token `xapp-...` for Socket Mode |
| `SLACK_BOT_TOKEN` | Bot token (fallback) |
| `SLACK_APP_TOKEN` | App token (fallback) |

### Telegram

| Env Var | Description |
|---|---|
| `MESSAGING_TELEGRAM_TOKEN` | Bot token (primary) |
| `TELEGRAM_BOT_TOKEN` | Bot token (fallback) |

### WhatsApp

| Env Var | Description |
|---|---|
| `MESSAGING_WHATSAPP_TOKEN` | API access token (primary) |
| `MESSAGING_WHATSAPP_PHONE_NUMBER_ID` | Phone number ID for Business API |
| `MESSAGING_WHATSAPP_USE_BUSINESS_API` | `true` for official API, `false` for neonize |
| `WHATSAPP_TOKEN` | API token (fallback) |
| `WHATSAPP_PHONE_NUMBER_ID` | Phone number ID (fallback) |

### Microsoft Teams

| Env Var | Description |
|---|---|
| `MESSAGING_TEAMS_APP_ID` | Bot Framework app ID (primary) |
| `MESSAGING_TEAMS_APP_SECRET` | Bot Framework app password (primary) |
| `MESSAGING_TEAMS_WEBHOOK_URL` | Incoming webhook URL |
| `MESSAGING_TEAMS_WEBHOOK_PORT` | Local webhook server port |
| `MSTEAMS_APP_ID` | App ID (fallback) |
| `MSTEAMS_APP_PASSWORD` | App password (fallback) |

### Google Chat

| Env Var | Description |
|---|---|
| `MESSAGING_GOOGLECHAT_TOKEN` | Service account JSON path (primary) |
| `MESSAGING_GOOGLECHAT_APP_ID` | GCP project ID |
| `GOOGLE_CHAT_SERVICE_ACCOUNT` | Service account path (fallback) |
| `GOOGLE_CHAT_PROJECT_ID` | Project ID (fallback) |

### Google Meet

| Env Var | Description |
|---|---|
| `MESSAGING_GOOGLEMEET_TOKEN` | Service account JSON path (primary) |
| `GOOGLE_MEET_SERVICE_ACCOUNT` | Service account path (fallback) |

### Mattermost

| Env Var | Description |
|---|---|
| `MESSAGING_MATTERMOST_TOKEN` | Personal access token (primary) |
| `MESSAGING_MATTERMOST_URL` | Server URL |
| `MATTERMOST_TOKEN` | Token (fallback) |
| `MATTERMOST_URL` | URL (fallback) |

### Matrix

| Env Var | Description |
|---|---|
| `MESSAGING_MATRIX_TOKEN` | Access token (primary) |
| `MESSAGING_MATRIX_HOMESERVER` | Homeserver URL |
| `MESSAGING_MATRIX_USER_ID` | Full user ID `@bot:matrix.org` |
| `MATRIX_ACCESS_TOKEN` | Token (fallback) |
| `MATRIX_HOMESERVER` | Homeserver (fallback) |
| `MATRIX_USER_ID` | User ID (fallback) |

### IRC

| Env Var | Description |
|---|---|
| `MESSAGING_IRC_TOKEN` | Not used (IRC uses server connection) |
| `MESSAGING_IRC_SERVER` | Server hostname |
| `MESSAGING_IRC_PORT` | Server port (default: `6667`) |
| `MESSAGING_IRC_NICKNAME` | Bot nickname (default: `agent_bot`) |
| `MESSAGING_IRC_CHANNELS` | JSON array of channels to auto-join |
| `IRC_SERVER` | Server (fallback) |
| `IRC_PORT` | Port (fallback) |
| `IRC_NICKNAME` | Nickname (fallback) |

### Signal

| Env Var | Description |
|---|---|
| `MESSAGING_SIGNAL_TOKEN` | Registered phone number (primary) |
| `SIGNAL_PHONE_NUMBER` | Phone number (fallback) |

### iMessage

| Env Var | Description |
|---|---|
| `MESSAGING_IMESSAGE_TOKEN` | Not required (macOS AppleScript bridge) |

### LINE

| Env Var | Description |
|---|---|
| `MESSAGING_LINE_TOKEN` | Channel access token (primary) |
| `MESSAGING_LINE_APP_ID` | Channel ID |
| `MESSAGING_LINE_WEBHOOK_URL` | Webhook URL |
| `MESSAGING_LINE_WEBHOOK_PORT` | Webhook server port |
| `LINE_CHANNEL_ACCESS_TOKEN` | Token (fallback) |
| `LINE_CHANNEL_ID` | Channel ID (fallback) |

### Twitch

| Env Var | Description |
|---|---|
| `MESSAGING_TWITCH_TOKEN` | OAuth access token (primary) |
| `MESSAGING_TWITCH_CHANNELS` | JSON array of channels |
| `TWITCH_OAUTH_TOKEN` | Token (fallback) |
| `TWITCH_CHANNELS` | Comma-separated channels (fallback) |

### Synology Chat

| Env Var | Description |
|---|---|
| `MESSAGING_SYNOLOGY_WEBHOOK_URL` | Incoming webhook URL (primary) |
| `SYNOLOGY_CHAT_WEBHOOK_URL` | Webhook URL (fallback) |

### Voice Call (Twilio)

| Env Var | Description |
|---|---|
| `MESSAGING_VOICECALL_APP_ID` | Twilio Account SID (primary) |
| `MESSAGING_VOICECALL_TOKEN` | Twilio Auth Token (primary) |
| `MESSAGING_VOICECALL_FROM_NUMBER` | Twilio "from" phone number |
| `MESSAGING_VOICECALL_WEBHOOK_URL` | Webhook URL for call events |
| `MESSAGING_VOICECALL_WEBHOOK_PORT` | Webhook server port |
| `TWILIO_ACCOUNT_SID` | Account SID (fallback) |
| `TWILIO_AUTH_TOKEN` | Auth Token (fallback) |
| `TWILIO_FROM_NUMBER` | From number (fallback) |

### Nextcloud Talk

| Env Var | Description |
|---|---|
| `MESSAGING_NEXTCLOUD_TOKEN` | App password or user token (primary) |
| `MESSAGING_NEXTCLOUD_APP_ID` | Nextcloud username |
| `MESSAGING_NEXTCLOUD_URL` | Server URL |
| `NEXTCLOUD_TOKEN` | Token (fallback) |
| `NEXTCLOUD_USER` | Username (fallback) |
| `NEXTCLOUD_URL` | URL (fallback) |

---

## XDG Directory Layout

```
~/.config/agent-utilities/
├── config.json                    ← All messaging config lives here
├── mcp_config.json
└── a2a_config.json

~/.local/share/agent-utilities/
├── kg/
│   └── knowledge_graph.db         ← Messages stored here as memory nodes
├── messaging/
│   ├── sessions/                  ← Backend-specific auth state (neonize QR, etc.)
│   └── history/                   ← Local message history cache
└── ...
```

## Programmatic Access

```python
from agent_utilities.core.config import config
from agent_utilities.core.paths import messaging_sessions_dir, messaging_config_path

# Read messaging config from AgentConfig (populated from config.json)
print(config.messaging_enabled_backends)    # ['discord', 'slack']
print(config.messaging_discord_token)       # 'BOT_TOKEN...'
print(config.messaging_kg_ingest)           # True

# XDG paths
print(messaging_sessions_dir())  # ~/.local/share/agent-utilities/messaging/
print(messaging_config_path())   # ~/.config/agent-utilities/config.json

# Reload config at runtime (hot-reload without restart)
config.reload()
```

---

## Cross-References

- [AU-ECO.toolkit.journey-map-milestones Architecture](ECO-4.5-Native_Messaging_Backend.md) — Full architecture and capability matrix
- [OS-5.0 XDG Paths](../5_agent_os/OS-5.0-Agent_OS_Infrastructure.md) — XDG path resolution
- [KG-2.1 Tiered Memory](../2_knowledge_graph/KG-2.1-Memory_Architecture.md) — Message memory persistence
- [ORCH-1.1 Planner Graph Agent](../1_orchestration/ORCH-1.1-Planner_Graph_Agent.md) — Inbound event routing
