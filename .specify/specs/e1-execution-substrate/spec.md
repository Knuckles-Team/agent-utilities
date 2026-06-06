# Spec: EPIC 1 — Unified Execution Substrate (ORCH-1.33 + ORCH-1.34)

> Designs: `.specify/design/orch-1.33-multi-cli-adapter-registry/design.md`,
> `.specify/design/orch-1.34-provider-normalizing-proxy/design.md`. **Lighthouse (with EPIC 4).**

## Pre-Flight Checklist
- [x] Design docs exist with KG-nearest-concepts tables (ORCH-1.33 max 0.58, ORCH-1.34 max 0.61 — both <0.70).
- [x] Extension points identified (engine.py stub; model_factory; guardrails egress).
- [x] New CONCEPT:ORCH-1.33/1.34 justified as new (runtime-backend + BYOK-normalization axes).
- [x] Wire-First confirmed: ≤2 hops from `/api/proxy/<provider>/stream` and `graph_orchestrate`.
- [ ] Live `kg_search` confirms similarities <0.70 before committing markers.
- [ ] `code-enhancer` audit run against proposed changes.

## User Stories

### US-1 — Drive any installed CLI through a canonical interface
**As** the orchestrator, **I want** `engine.run(manifest)` to dispatch a step to an installed agent CLI
via a declarative `AdapterDefinition`, **so that** plans are backend-agnostic.
- **AC1**: An `AdapterDefinition` declares `bin, version_args, build_args(), stream_format, prompt_delivery, list_models, fallback_models`.
- **AC2**: `AdapterRegistry.detect()` probes PATH at startup (non-blocking), returns `{id, available, version, models, auth_status}`; a broken/missing CLI yields `available=False` without raising.
- **AC3**: A stub adapter dispatched through `engine.run` yields canonical events; adding a new def requires **no** change to `engine.py` (registry-only).
- **AC4**: `stream_format` dispatches to a handler factory; ≥2 formats (`plain`, `jsonl`) normalize to one event schema.

### US-2 — Proxy any BYOK provider with a canonical stream
**As** an operator, **I want** `POST /api/proxy/<provider>/stream` to normalize anthropic/openai/azure/google/ollama into `{type:start|delta|error|end}`, **so that** I can BYOK without a per-vendor SDK.
- **AC5**: Each provider route forwards `{baseUrl, apiKey?, model, messages, systemPrompt?, maxTokens?}` and emits canonical SSE.
- **AC6**: `model_factory.create_model(provider="custom")` routes through `ProviderProxy`.

### US-3 — Block SSRF at the edge
**As** a security owner, **I want** custom `baseUrl`s DNS-resolved and checked, **so that** internal IPs can't be reached.
- **AC7**: `validate_base_url_resolved()` rejects RFC1918/loopback*/link-local/CGNAT/metadata IPs **after DNS resolution**; *loopback is allowed* for local LLMs (configurable).
- **AC8**: An internal-IP `baseUrl` returns a 4xx **before** any upstream fetch (test asserts no outbound call).

### US-4 — Operator credential + model resolution
**As** an operator, **I want** `env > file > none` credential resolution and `model_override_env_var`, **so that** CI/Docker/packaged installs configure once.
- **AC9**: `CredentialResolver` prefers env, then config file, then None (unit test).
- **AC10**: `model_override_env_var`, when set, wins over the per-call default.

## Non-Functional Requirements
- Tests in `tests/unit/` + `tests/integration/` with `@pytest.mark.concept(id="ORCH-1.33")` / `("ORCH-1.34")`; no real network (mock upstreams); ≤60s.
- `pre-commit run --all-files` green (ruff/mypy/bandit + guardrails).
- `check_wiring.py --entry-points server/app.py,mcp/kg_server.py --max-hops 3` passes for new modules.
- Default behavior unchanged when no adapter/proxy is selected (in-process pydantic-ai path preserved).
- Post-modification artifacts: `docs/pillars/1_graph_orchestration/ORCH-1.33.md` + `ORCH-1.34.md`, `docs/concepts.yaml` regen, CHANGELOG, README feature line, AGENTS.md (new route/tool surface).
