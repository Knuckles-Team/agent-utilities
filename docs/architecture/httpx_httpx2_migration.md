# Staged `httpx` → `httpx2` migration (GOC-87)

CONCEPT:AU-ECO.mcp.protocol-compat-bridge

## Why this is a strangler, not a substitution

**Both `httpx` and `httpx2` are concrete runtime contracts today.** As of the
2026-08-16 lock:

- `httpx==0.28.1` is a direct base dependency (`pyproject.toml`) and is
  required — transitively, at the resolved-lock level — by **17** other
  locked packages: `agent-terminal-ui`, `agent-utilities` itself, `anthropic`,
  `epistemic-graph`, `google-genai`, `groq`, `huggingface-hub`, `langfuse`,
  `llama-index-core`, `mistralai`, `neonize`, `ollama`, `openai`,
  `pydantic-ai-harness`, `pydantic-ai-slim`, `pydantic-graph`,
  `python-telegram-bot`.
- `httpx2==2.9.1` is a real, separately published PyPI package (same
  upstream author as `httpx`) pulled in transitively by **3** locked
  packages required for the `[mcp]` extra: `fastmcp-slim`, `mcp`, and
  `genai-prices`. FastMCP 4 / MCP SDK v2's client transports
  (`StreamableHttpTransport`, `SSETransport`, `mcp.client.sse.sse_client`,
  `mcp.client.streamable_http.streamable_http_client`) are typed against
  `httpx2`, not `httpx`.
- **48 files** under `agent_utilities/` import `httpx` directly (`import
  httpx` / `from httpx import ...`); **2** import `httpx2`
  (`agent_utilities/mcp/httpx_boundary.py`, and its test).

A process-wide alias (`httpx = httpx2` at import time) would appear to work
and would silently break every one of the 17 `httpx`-typed third-party SDKs
above the moment they did an `isinstance` check against the real class — see
`agent_utilities/mcp/httpx_boundary.py`'s docstring (D-MTT-1) for a
production incident of exactly this shape, already fixed once for the
`auth=` parameter crossing into fastmcp/mcp SDK v2 client code. **This lane
does not do that anywhere.** It is prohibited by the lane's own authority
section, and the design below never imports one package into the module
that adapts the other.

## The neutral seam

```
                    application call site
                            │
                            │  family="…"
                            ▼
     agent_utilities.httpsupport.transport_factory
        create_http_client / create_async_http_client
                            │
              ┌─────────────┴─────────────┐
              │ family in                 │ (default)
              │ MIGRATED_HTTPX2_FAMILIES  │
              ▼                            ▼
    httpsupport.httpx2_adapter   httpsupport.httpx_adapter
    Httpx2Adapter/AsyncHttpx2Adapter   HttpxAdapter/AsyncHttpxAdapter
              │                            │
       httpx2.Client/AsyncClient   core.http_client.create_*_http_client()
              │                            │   (unchanged — DNS pinning,
              ▼                            ▼    air-gap guard, retry, TLS)
        real httpx2 transport        real httpx transport
```

Both adapters implement the same structural protocol
(`agent_utilities.httpsupport.client_protocol.HttpClient` /
`AsyncHttpClient`): `request(method, url, **kwargs) -> HttpResponse`,
`close()` / `aclose()`. Neither adapter's *module* imports the other
package — proven statically in
`tests/unit/httpsupport/test_client_protocol.py` (`test_httpx_adapter_module_never_imports_httpx2`
/ `test_httpx2_adapter_module_never_imports_httpx`) — so it is not merely
convention that a concrete `httpx.Client`/`httpx2.Client` can't cross the
boundary; it's structurally impossible for either adapter to construct one.
Responses are normalized into a package-neutral `HttpResponse` dataclass
(`client_protocol.normalize_response`) immediately on return, so a real
`httpx.Response`/`httpx2.Response` is never handed back to the caller
either. Exceptions from both packages' distinct hierarchies (identically
named — `ConnectError`, `ConnectTimeout`, `TooManyRedirects`, ...) map by
class name onto one AU-owned taxonomy (`HttpTransportError` and its
`HttpConnectError` / `HttpTimeoutError` / `HttpTooManyRedirectsError` /
`HttpProtocolError` subclasses) in `client_protocol.map_transport_error`, so
calling code never branches on which package is behind the boundary.

`HttpxAdapter` / `AsyncHttpxAdapter` are a **pure wrapper** over the
existing, unmodified `agent_utilities.core.http_client.create_http_client` /
`create_async_http_client` factory — the one that already enforces a
mandatory finite timeout, mandatory TLS verification, DNS-pinned egress
(`pin_egress=True`), the air-gap guard, and
`ResiliencePolicy`-driven retry. Every call family NOT in
`MIGRATED_HTTPX2_FAMILIES` gets exactly this adapter, so **every existing
`httpx` consumer this lane did not touch sees zero behavior change.**

## Construction is centralized and linted

`scripts/check_http_egress_boundary.py` (already existed for `httpx`, run in
CI/pre-commit as `tests/gates/test_http_egress_boundary.py`) was extended in
this lane to also reject direct `httpx2.Client()` / `httpx2.AsyncClient()`
construction anywhere except `agent_utilities/httpsupport/httpx2_adapter.py`
(the sole exemption, mirroring the existing `core/http_client.py` exemption
for `httpx`). `tests/gates/test_http_egress_boundary.py::test_direct_httpx2_client_is_rejected`
proves this against a known-bad input (a synthetic file constructing
`httpx2.AsyncClient()` outside the adapter).

## What has been ported (W05) — and why it qualifies as low-risk

**`agent_utilities/gateway/widgets/ollama.py`** (`fetch_data`) — two
unauthenticated, unpinned, non-streaming, local-network GET calls
(`/api/tags`, `/api/ps`) that populate a dashboard widget. Family:
`"gateway-widget-diagnostics"`.

Before/after comparison (GOC-87 authority #5 — every invariant explicitly
compared, not assumed preserved):

| Property | Before (raw `httpx.get()`) | After (`Httpx2Adapter`) |
|---|---|---|
| Timeout | 5.0s, per-call | 5.0s, client-level (unchanged value) |
| TLS verification | httpx default (on for `https://`) | mandatory (`Httpx2Adapter` rejects `verify=False`) |
| DNS pinning | none | none (unchanged — this family doesn't need it; see `httpx2_adapter.py` docstring for why a family that DOES need it must stay on `HttpxAdapter`) |
| Air-gap guard | none | none (unchanged — not wired for this family; `core.http_client`'s guard only applies via `HttpxAdapter`) |
| Redirect policy | httpx default | httpx2 default (same upstream author, same default) |
| Retry | none | none (unchanged) |
| Standard headers | none | `User-Agent: agent-utilities/<version>` (added — was previously absent everywhere this call went through bare `httpx.get()`) |
| Streaming | n/a (not streamed) | n/a |
| Connection reuse | none (2 ephemeral one-shot clients) | 1 client reused for both requests (net-positive, not a regression) |
| Error surface | broad `except Exception` → `_error_data(e)` | unchanged — `HttpTransportError` subclasses are still `Exception` instances |

This consumer qualifies as low-risk *because* it started with none of the
protections `core.http_client` provides — porting it to `Httpx2Adapter`
cannot regress an invariant it never had. Proven end-to-end (a real loopback
HTTP server, not a mocked transport) in
`tests/unit/test_gateway_widget_ollama_httpx2.py`, including the
known-bad-input case (an unreachable service on a closed port still returns
`status="error"`, matching the pre-port contract exactly).

## What was deliberately left unported

- **The other 47 files** importing `httpx` directly — the majority are
  model-provider SDKs (`anthropic`, `openai`, `google-genai`, ... construct
  their own `httpx` clients internally; AU does not control that
  construction — third-party-owned), the OAuth/PKCE and MCP streaming
  boundary (`security/oauth_client_credentials.py`,
  `security/browser_auth.py`, `security/oidc_discovery.py`,
  `mcp/client_credentials.py`, `mcp/toolset_factory.py`,
  `mcp/httpx_boundary.py`), and `core/http_client.py` itself (the factory
  `HttpxAdapter` wraps — porting the factory's own DNS-pinning/air-gap/retry
  transports to httpx2 is the deferred W06 follow-on the `httpx2_adapter.py`
  docstring names explicitly; it requires re-deriving those
  `httpx.BaseTransport` subclasses against `httpx2.BaseTransport`, which is
  real, security-critical work this lane did not rush).
- **`MIGRATED_HTTPX2_FAMILIES` stays a one-entry set.** Adding a family here
  is a deliberate, reviewed, one-line change — never a bulk flip.

## Removal gate (W08 — not reached)

`httpx` is removed from `pyproject.toml`/the lock only once a **freshly
regenerated** lock and SBOM show zero runtime consumers. That is nowhere
close to true today (17 locked packages still require it) — this lane does
not attempt removal, and does not reduce `httpx`'s presence in the
dependency graph at all; it only proves the neutral seam and ports one
genuinely low-risk consumer through it.
