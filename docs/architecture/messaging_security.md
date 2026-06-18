# Secure messaging ingress — instant push with zero open ports (ECO-4.66)

The messaging stack is designed so the user gets **instant communication** without ever
exposing a port to the internet, and so only **designated identities** (via Keycloak) and
sanctioned **automation** (Claude→MCP, the message bots) can reach anything.

## The core idea

Almost everything is **outbound-initiated** and therefore needs no inbound port:

- **Claude → MCP fleet** dials *out* through the multiplexer.
- **Message send + long-poll** dial *out* to Telegram/Slack APIs.

The only thing that wants to come *in* is a **webhook push**. We never open a port for it —
it rides an **outbound tunnel**.

```mermaid
flowchart LR
    TG([Telegram / Slack cloud]) -->|signed webhook over HTTPS| EDGE
    subgraph EDGE["Edge (no homelab ports opened)"]
        T[Tunnel terminator\npangolin / Cloudflare Tunnel]
        AUTH[Keycloak forward-auth\n(human surfaces only)]
        CS[CrowdSec WAF / rate-limit]
    end
    T -. outbound WireGuard/QUIC .-> GW
    subgraph HOME["Homelab (egress-only)"]
        GW[gateway daemon\n127.0.0.1:webhook_port]
        GW --> RT[InboundRouter → agent]
    end
    EDGE -.->|forward only /messaging/webhook/*| GW
    classDef e fill:#533483,stroke:#7b2cbf,color:#fff
    class TG,EDGE e
```

## The webhook modes (all first-class)

| Mode | Ingress | Open ports | Edge node? | When |
|---|---|---|---|---|
| **Cloudflare Tunnel (recommended)** | `cloudflared` on the homelab → Cloudflare edge | **none** (outbound only) | **none needed** | True push, zero exposure, no infra to own. |
| **pangolin / self-hosted tunnel** | WireGuard tunnel to your own edge | **none** (outbound only) | a small VPS edge | Fully self-hosted, no third party. |
| **Opt-in (default polling)** | none | none | none | Baseline — long-poll is already near-real-time; webhook path inactive until `MESSAGING_WEBHOOK_BASE_URL` is set. |
| **Public edge (Caddy)** | Caddy public HTTPS | inbound 443 (shared) | a reachable host | Only if Caddy is already public; harden per below. |

### Cloudflare Tunnel — no edge node required (recommended)

You do **not** need to own an edge-ingress node. `cloudflared` runs **on the homelab host
itself** and dials **outbound** to Cloudflare; Cloudflare *is* the public edge (TLS, DDoS,
and **Zero-Trust Access** for human gating). Nothing listens on a public IP at your site.

```
Telegram ──HTTPS──▶ Cloudflare edge ──(outbound tunnel)──▶ cloudflared (homelab)
                         │  Access (Zero Trust) gates HUMAN routes
                         └─ forwards ONLY /messaging/webhook/* ▶ 127.0.0.1:MESSAGING_WEBHOOK_PORT
```

Setup (high level): create a Cloudflare Tunnel, run `cloudflared` on the host, map a
hostname (e.g. `hooks.<your-domain>`) to `http://127.0.0.1:${MESSAGING_WEBHOOK_PORT}`, set
`MESSAGING_WEBHOOK_BASE_URL=https://hooks.<your-domain>`, and put **Cloudflare Access**
policies (or Keycloak as the OIDC IdP behind Access) on every route *except* the
signature-validated `/messaging/webhook/*` path. The bot's `secret_token` + Telegram's IP
ranges lock that one open path.

`MESSAGING_WEBHOOK_BASE_URL` set → the Telegram backend uses `python-telegram-bot`'s
`start_webhook` (binds the local port, validates Telegram's `secret_token` header, calls
`setWebhook`); empty → long-polling. Same code path for every mode.

## Defense in depth (built-in + recommended)

- **Webhook authenticity (in code):** Telegram `secret_token` header validated by the
  receiver (`MESSAGING_WEBHOOK_SECRET`); add Slack signing-secret HMAC for Slack. Only the
  platform's signed requests are accepted even if the URL leaks.
- **Keycloak forward-auth** (Caddy `forward_auth` / oauth2-proxy) on every *human* surface
  (WebUI, dashboards) — only your designated Keycloak group gets in. The webhook path is
  the sole unauthenticated route and is locked by the signature + IP allowlist.
- **CrowdSec** at the edge — ban/rate-limit abusive IPs (+ Telegram publishes its sender
  IP ranges for an allowlist).
- **Eunomia** — authorizes which automation may call which MCP tool.
- **OpenBao** — stores the bot token + `MESSAGING_WEBHOOK_SECRET`.
- **mTLS** on the tunnel↔gateway hop and service-to-service overlay.
- **Admin access** stays on WireGuard/Tailscale — never a public admin port.

## Services checklist

Already running: **Caddy, CrowdSec, Keycloak, OpenBao, Eunomia**. To complete the secure
push path you only add a tunnel — and with **Cloudflare Tunnel that's a single agent
(`cloudflared`) on an existing host, no new edge node**:

- **Recommended:** Cloudflare Tunnel (`cloudflared`) + **Cloudflare Access** (Zero Trust)
  for human routes. Keycloak can sit behind Access as the OIDC IdP so identity stays yours.
- **Fully self-hosted alternative:** pangolin (vendored in `open-source-libraries/pangolin`)
  on a small VPS edge.
- **Already-public alternative:** Caddy public HTTPS + Keycloak `forward_auth` (webhook
  path exempt, locked by signature + IP allowlist + CrowdSec).

Across all flavors the in-code protections are identical (webhook `secret_token`/HMAC, the
egress-only design, OpenBao-held secrets, Eunomia tool authz), so switching ingress is a
deployment choice — not a code change.
