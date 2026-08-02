# Design Document: Air-gap mode blocks outbound requests by IP-literal/loopback/private classification only, never by DNS resolution, and is enforced as the outermost transport wrap on the ONE canonical HTTP client factory

CONCEPT:AU-OS.deployment.airgap-mode

> `agent_utilities/core/http_client.py:374-412` (`_airgap_enabled`,
> `is_local_host`, `_check_airgap`) and `:600-626` (transport-wrap ordering);
> `agent_utilities/core/model_factory.py:605-616` (`create_async_http_client`
> application to LLM traffic); `.env.example:95-99`.

## Decision — `AIRGAP_MODE=1` makes the canonical outbound HTTP factory (`core/http_client.py`) and the LLM client constructor (`core/model_factory.py`) refuse any request whose target host is not loopback/RFC1918-private/link-local, fail-closed with a typed `AirgapViolation` raised BEFORE the request is sent, classifying "local" from the literal host string alone — never via DNS resolution

A sovereign/self-hosted deployment needs a hard guarantee that model traffic (and
any other outbound HTTP through the canonical client) cannot silently phone home to
the public internet. `is_local_host` (`http_client.py:381`) makes that guarantee
checkable and deterministic: it accepts `localhost` or a literal IP address parsed
by `ipaddress.ip_address`, classifying loopback/private/link-local as local and
**everything else — including any DNS hostname — as non-local and blocked**. The
guard is wired as the OUTERMOST transport wrap (`http_client.py:619-626`, "outermost
so a blocked host is never handed to the retry transport above") so a blocked
request cannot leak through a retry or the DNS-pinned egress transport underneath
it, and it is a no-op with zero overhead when `AIRGAP_MODE` is off (the default).

## Rejected alternative — resolve the hostname via DNS and classify the resolved IP as local/non-local

The natural-looking alternative is to let a configured hostname (e.g. an on-LAN
vLLM/Ollama host reachable by a real DNS name) resolve normally and then check
whether the RESOLVED address is private. That is explicitly rejected: "Deliberately
does **not** perform a DNS lookup — resolving a hostname would itself be network
activity, and would make the air-gap check non-deterministic (test-hostile,
resolver-dependent)" (`http_client.py:382-385`). Under a genuine air-gap posture,
even the DNS lookup itself is an outbound network dependency the guard is supposed
to prevent — checking "is this safe to call" by first making a network call defeats
the property being enforced. The chosen alternative pushes the cost onto the
operator instead: "point an air-gapped deployment's local endpoints... at their
private IP literal, not a DNS name, to reach them under `AIRGAP_MODE=1`"
(`http_client.py:388-390`) — a small, explicit configuration requirement in
exchange for a check that is itself air-gap-safe and fully deterministic.

## Risk Assessment

- **Blast Radius**: `agent_utilities/core/http_client.py`,
  `agent_utilities/core/model_factory.py`, `agent_utilities/core/config.py`
  (`airgap_mode` setting).
- **Backward Compatible**: Yes — off by default; a deployment not setting
  `AIRGAP_MODE` sees no behaviour change.
- **Known weak point**: an operator who configures a local endpoint by DNS name
  (instead of IP literal) under `AIRGAP_MODE=1` gets a hard, correctly fail-closed
  `AirgapViolation` rather than a working connection — the guard cannot
  distinguish "a hostname that would resolve locally" from "a hostname that would
  phone out," so it blocks both, by design, at the cost of that operator having to
  know the IP-literal requirement up front.
