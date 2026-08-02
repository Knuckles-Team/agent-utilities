# Design Document: One `CredentialProvider` abstraction answers "give me credentials for source X" by name, backed by typed `SourceCredential`s resolved through the existing secrets store — instead of each connector reading its own env vars

CONCEPT:AU-OS.deployment.universal-outbound-credentialprovider

> `agent_utilities/security/credential_provider.py:1-35` (module docstring,
> `CredentialProvider`); companion typed registry
> `agent_utilities/security/source_credentials.py:1-27`
> (`AU-OS.config.source-credential-registry`, `SourceCredential`).

## Decision — `CredentialProvider` is the INVERSE of the secrets store: callers ask for a credential by SOURCE NAME (`"x"`, `"reddit"`, `"github"`, …), not by secret key, and get back a typed, ready-to-apply `SourceCredential` (an API key, cookie session, or auto-refreshing OAuth2 token) whose descriptor points at a `vault://`/`secret://`/`env://` URI resolved through the EXISTING `SecretsClient` — storage is not reimplemented, only the "which kind of credential, and how do I apply it to a request" layer is added on top

Every external-source connector needs credentials, and every one of them needs to
answer the same two questions: does a usable credential exist for this source, and
how do I attach it to an outbound HTTP request (header / query param / cookie jar /
basic auth), including handling ones that expire and self-refresh (OAuth2). This
module answers both generically: a declarative, `config.json`-driven descriptor map
(`SOURCE_CREDENTIALS`) or programmatic registration (`CredentialProvider.register`)
maps a source name to a `SourceCredential` type; `materialize()` returns the
`AuthMaterial` a caller merges onto its request, and `is_present()` reports whether
real secret material actually exists — the exact signal a connector's
backend-ladder selection needs to pick the highest-fidelity backend it can
actually authenticate to (`credential_provider.py:16-18`). Its first consumer is
PulseLink's open-web/social source server, whose backend ladders use exactly this
`available`-reporting contract.

## Rejected alternative — leave credential handling ad hoc and per-agent, each reading its own environment variables directly

The rejected alternative is named as the platform's own prior, shipped pattern:
"it unifies what was previously ad-hoc, per-agent credential plumbing (e.g.
`agents/github-agent/github_agent/auth.py` reading env vars directly) behind one
abstraction" (`credential_provider.py:13-15`). Per-agent direct env-var reads mean
every connector reinvents its own notion of "do I have a credential for this,"
bypasses the platform's existing `SecretsClient`/Vault-backed storage discipline
(reading raw `os.environ` instead of a resolvable `vault://` reference), and gives
each connector its own bespoke request-application logic (headers vs. cookies vs.
OAuth2 refresh) with no shared self-refresh pattern — `OAuth2Credential` instead
explicitly reuses the existing `ClientCredentialsTokenProvider` cache-and-skew
pattern (`source_credentials.py:22-24`) "rather than introducing a second
refresher." Centralizing the by-name lookup and the typed-application logic means a
new connector gets Vault-backed storage, self-refreshing OAuth2, and
availability-driven backend selection for free, instead of writing its own
`auth.py` from scratch.

## Risk Assessment

- **Blast Radius**: `agent_utilities/security/credential_provider.py`,
  `agent_utilities/security/source_credentials.py`,
  `agent_utilities/security/secrets_client.py` (the underlying resolver).
- **Backward Compatible**: Yes — additive abstraction; existing per-agent
  env-var-reading code is unaffected unless migrated onto it.
- **Known weak point**: adoption is opt-in per connector — `github-agent`'s
  `auth.py` (the module's own cited example of the pattern being replaced) is
  cited as motivation, not confirmed migrated; a connector that never switches
  onto `CredentialProvider` keeps its own ad hoc env-var reads indefinitely, with
  nothing in this module detecting or flagging the holdout.
