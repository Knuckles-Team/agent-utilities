# Design Document: One `MessagingBackend` ABC that all 17 platform backends implement

CONCEPT:AU-ECO.messaging.native-backend-abstraction ·
CONCEPT:AU-ECO.messaging.telegram-webhook-receiver-started

> `agent_utilities/messaging/base.py` (the ABC); implementations under
> `agent_utilities/messaging/backends/*.py` (Telegram, Mattermost, Slack,
> Discord, Teams, WhatsApp, Signal, Matrix, IRC, LINE, Google Chat, Google
> Meet, iMessage, Nextcloud, Synology, Twitch, voicecall — 17 total per
> `base.py`'s own docstring).

## Decision — every platform backend implements the SAME `MessagingBackend` ABC

`base.py` defines `MessagingBackend` as an `ABC`: abstract methods define the
required contract (connect, listen, send_message, reply_to, react, …),
concrete methods provide sensible defaults, and a factory function
auto-detects the best backend from configured credentials. This is
explicitly modelled on two existing patterns in the codebase rather than
invented fresh: the Python equivalent of OpenClaw's `ChannelPlugin<T>`
interface, and the same ABC shape as the proven `TraceBackend` pattern in
`harness/trace_backend.py` (`base.py:1-19`).

**The rejected alternative** is a per-platform ad hoc client with its own
method names and its own event shape — the router, the coalescer, the
command dispatcher and the reach service would each need per-platform
branches. Instead every inbound event normalizes to the shared
`InboundEvent` model and every backend exposes the same send/reply/react
surface, so the ENTIRE rest of the messaging stack (router, coalescer,
commands, reach service) is written once against the ABC and works
identically across all 17 platforms — adding platform #18 means
implementing the ABC, not touching the router.

The concrete `MessagingBackend` used by a given deployment is chosen by
`configured_platforms()` auto-detecting which platform's credentials are
present (e.g. `TELEGRAM_BOT_TOKEN`) — an operator does not select a backend
in code, they set a token and the abstraction picks it up.

### Pointer — `CONCEPT:AU-ECO.messaging.telegram-webhook-receiver-started`

`backends/telegram.py:384`, `_start_intake`. One concrete backend's
lifecycle detail, not a second abstraction: Telegram's intake starts either
webhook push (via `python-telegram-bot`'s built-in `start_webhook`, which
validates Telegram's `secret_token` header and calls `setWebhook`) or falls
back to long-polling, chosen by whether `MESSAGING_WEBHOOK_BASE_URL` is set.
Webhook mode binds a LOCAL port (`MESSAGING_WEBHOOK_PORT`) that the
deployment's tunnel/edge (pangolin/Cloudflare/Caddy) forwards the public
`webhook_url` to, so nothing is exposed directly and only Telegram's signed
requests are accepted — this is Telegram-specific plumbing sitting behind
the SAME abstract `listen()` contract every other backend implements
differently (Mattermost uses a WebSocket consumer instead, see
`.specify/design/eco-messaging-mattermost-backend/design.md`).

## Risk Assessment

- **Blast Radius**: `agent_utilities/messaging/base.py` and every file under
  `agent_utilities/messaging/backends/`.
- **Backward Compatible**: Yes — describes the existing, shipped contract.
- **Known weak point**: the ABC's default/concrete methods encode assumptions
  (e.g. reaction support) that not every one of the 17 platforms genuinely
  has; backends without a capability degrade to a no-op rather than the ABC
  enforcing capability declarations at the type level.
