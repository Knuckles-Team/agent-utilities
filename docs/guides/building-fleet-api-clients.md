# Building Fleet API Clients

**CONCEPT:AU-ECO.ui.fleet-http-client-library — Fleet HTTP Client Library** (`agent_utilities.http`)

The consolidation audit found a ~95%-identical `api_client_base.py` duplicated
across 10+ connector repos (kafka-mcp, portainer-agent, okta-agent,
dockerhub-api, ...) — none with retry, rate-limit capture, or log redaction.
`agent_utilities.http` is the single shared base those repos strangle their
local copies onto, built on the canonical `agent_utilities.core.http_client`
factory so every fleet client inherits the unified safety defaults (finite
timeout, TLS verification on, standard `User-Agent`, optional
`ResiliencePolicy` transport retry).

## Quickstart

```python
from agent_utilities.http import BaseApiClient, TokenAuth

client = BaseApiClient(
    "https://portainer.arpa/api",
    auth=TokenAuth(api_key, header="X-API-Key", prefix=None),
)
envelope = client.get("stacks")          # relative endpoints join base_url
envelope["status_code"]                  # 200
envelope["data"]                         # parsed JSON (text for non-JSON)
envelope["rate_limit"]                   # latest typed rate-limit snapshot
```

The envelope shape — `{"status_code", "data", "rate_limit"}` (plus
`"headers"` with `include_response_headers=True`) — follows dockerhub-api,
the newest fleet convention. `AsyncBaseApiClient` is the `async`/`await`
twin with the same constructor and methods.

## Auth strategies

| Strategy | Use for |
|---|---|
| `TokenAuth("tok")` | `Authorization: Bearer tok` (default) |
| `TokenAuth("tok", prefix="SSWS")` | Okta-style scheme prefixes |
| `TokenAuth(key, header="X-API-Key", prefix=None)` | bare header API keys (Portainer) |
| `TokenAuth(token_provider=mgr.get_token)` | OAuth/JWT manager delegation (dockerhub `TokenManager`, salesforce flows) — consulted per request |
| `BasicAuth(user, password)` | RFC 7617 basic credentials |
| `QueryApiKeyAuth("api_key", key)` | API key in the query string |

A strategy may define `invalidate()` to opt in to one transparent retry
after a 401 (token refresh), mirroring dockerhub-api's semantics.

## Rate limits and 429 backoff

Every response's `X-RateLimit-*` / `X-Rate-Limit-*` / `Retry-After` headers
are parsed into a typed `RateLimitSnapshot`, attached to every envelope and
kept on the client (`client.rate_limit`). HTTP 429 responses are retried
automatically with a **bounded** backoff — `Retry-After` wins, otherwise an
epoch `*-Reset` header (clamped to `retry_after_cap_s`, default 15s, so a
hostile server can never stall a caller) — up to `max_retries_429` times.

## Error mapping

HTTP >= 400 raises the canonical `agent_utilities.core.exceptions` types:
400/404 → `ParameterError`, 401 → `AuthError`, 403 → `UnauthorizedError`,
everything else → `ApiError`. Override per-API with the
`error_map={418: TeapotError}` constructor argument, or override
`_map_error()` for bespoke error envelopes (Okta's
`errorCode`/`errorSummary`, etc.). Pass `raise_for_status=False` to receive
the error envelope instead. Error messages are redacted before raising.

## Pagination

`client.paginate(...)` returns a lazy `PaginationIterator`
(`AsyncPaginationIterator` on the async client) over five dialects:

```python
for user in client.paginate(
    "/api/v1/users",
    mode="link",            # cursor | page | offset | link | since_id
    max_items=500,
):
    ...
```

- `cursor` — cursor param + dotted response path (`cursor_param`,
  `cursor_path`); semantics match the AU-KG.ingest.mcp-tool-connector `mcp_tool` connector so
  configurations translate 1:1 (the implementations stay parallel — see the
  `agent_utilities/http/pagination.py` module docstring for why).
- `page` — `page_param`/`page_size_param`, stops on a short page.
- `offset` — `offset_param`/`limit_param`, advances by items received.
- `link` — RFC 5988 `Link: <...>; rel="next"` (Okta, GitHub).
- `since_id` — keyset resume from the last record's id.

After iteration, `iterator.truncated`, `.next_cursor`, `.pages_fetched` and
`.items_yielded` describe the sweep; `iterator.collect()` drains into an
Okta-style `{data, count, truncated, next_cursor}` envelope.

## Log redaction

The `agent_utilities.http.client` logger carries a `LogRedactor` filter by
default, and clients register their literal secrets with it at
construction. `LogRedactor` / `redact_text` are importable for connector
loggers too — they promote the per-repo `scripts/security_sanitizer.py`
patterns: scheme credentials (`Bearer`/`SSWS`/`Basic`), DSN passwords
(`scheme://user:***@host`), well-known token shapes (GitHub/GitLab PATs,
AWS keys), and `token=...` assignments.

## Destructive gating

```python
client = BaseApiClient(url, auth=..., allow_destructive=False)  # default
client.guard_destructive("delete_stack")  # raises DestructiveOperationError
```

Gate every HTTP `DELETE`-class action behind `guard_destructive()` and let
operators enable it explicitly, as dockerhub-api and okta-agent do.

## Retry: ResiliencePolicy vs RetryManager (the rule)

Two retry mechanisms exist in agent-utilities. They are **not** redundant —
they retry different things:

| | `ResiliencePolicy` (AU-ORCH.execution.retry-predicate-raised-treating) | `RetryManager`/`RetryConfig` (ORCH-1.3) |
|---|---|---|
| Retries | one **in-process callable** | a whole **agent execution** |
| Verified by | the exception raised | shell `SuccessCheck` commands |
| Lives in | `agent_utilities.orchestration.resilience` | `agent_utilities.security.execution_stability_engine` |
| For HTTP | **yes** — pass `retry=http_retry_policy(...)` to retry transport failures (connect errors, resets, timeouts) | **no** — never wrap HTTP requests in it |

HTTP status handling is layered separately: 429 backoff is built into the
client (rate-limit aware), and other 4xx/5xx raise mapped exceptions for the
caller. `RetryManager`/`RetryConfig` are exported from
`agent_utilities.http` (and `agent_utilities.security`) because the audit
found them unexported — use them for orchestration-level run-until-green
loops with `on_failure` remediation hooks, not for network calls.

## Migrating from `requests.Session`

The fleet's older copies wrapped `requests.Session`; this base is
httpx-only for one-stack coherence with the core factory:

- `session.verify = False` → `verify=False` constructor argument (keep it
  `True` unless a site has an explicit, justified insecure flag);
- `session.headers.update({...})` → `headers={...}` or override
  `default_headers()`;
- `session.auth = (user, pass)` → `auth=BasicAuth(user, pass)`;
- header tokens → `auth=TokenAuth(...)`;
- `requests_mock` / `responses` in tests → `transport=httpx.MockTransport(handler)`;
- per-request `timeout=` keeps its meaning (and the client default is
  always finite).

Keep the connector's public `Api` class surface identical while swapping the
internals — a strangler behind the facade — so downstream MCP tools and
agents see no change.
