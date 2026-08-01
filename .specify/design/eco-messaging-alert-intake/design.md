# Design Document: HTTP alert-intake routes external monitoring through the one messaging stack

CONCEPT:AU-ECO.messaging.alert-intake

> `agent_utilities/messaging/alert_intake.py`

## Decision — external monitors POST to one webhook instead of each configuring its own notifier

Uptime-Kuma, Alertmanager, and any other webhook-capable monitor deliver
alerts by POSTing to `alert_intake.py`'s HTTP listener, which forwards the
text through the SAME connected messaging backend the daemon already runs
(Telegram/Mattermost/etc.), rather than each monitoring tool configuring its
own separate Telegram bot token or Mattermost webhook.

**The rejected alternative** — each tool ships its own notifier — is what
this replaces: N monitoring tools would each need their own bot credentials,
their own delivery-retry logic, and their own idea of "who to notify." Here
that is built once in messaging and every producer reuses it (the
"Universal-capability" pattern named in the module docstring): one place
handles delivery, timeouts and failure.

**Opt-in and blast-radius-bounded, by design, not incidentally:**

- Only started when `MESSAGING_ALERT_INTAKE_PORT` is set (`alert_intake.py:1`)
  — a daemon with no monitoring wired up never opens this listener.
- Runs as an independent `asyncio` task; a failure here never touches the
  inbound chat listeners (same file docstring).
- Bounded inputs: `_MAX_BODY_BYTES = 256 * 1024`, `_MAX_ALERT_CHARS = 4_000`,
  `_MAX_ALERTS = 100`, `_DELIVERY_TIMEOUT_SECONDS = 30`,
  `_MAX_CONCURRENT_DELIVERIES = 16` (`alert_intake.py:31-35`) — a
  misbehaving or malicious monitor cannot flood the messaging backend or
  hold the delivery pool open indefinitely.

Payload shapes are handled generically rather than per-vendor: a bare
string, `{"text"|"msg"|"message"|"content": ...}` (Uptime-Kuma sends `msg`),
and Alertmanager's `{"alerts": [...]}` array (`_extract_text`,
`alert_intake.py:37`) — one parser covers the two monitoring tools actually
in use in this deployment without hard-coding either vendor's schema as the
only accepted shape.

## Risk Assessment

- **Blast Radius**: `agent_utilities/messaging/alert_intake.py`,
  `agent_utilities/messaging/daemon.py:217` (where it is started).
- **Backward Compatible**: Yes — opt-in via env var.
- **Known weak point**: HMAC/auth on the webhook (the module imports `hmac`)
  is best-effort per-monitor-tool capability, not a uniform signature scheme
  across every possible sender — a monitor that can't sign its payload relies
  on network-level trust (loopback/tunnel) instead.
