# Tasks: EPIC 1 — Unified Execution Substrate

> TDD order (Red→Green→Refactor). Each task lands with its `@pytest.mark.concept` test.

## ORCH-1.33 — Multi-CLI Adapter Registry
- [ ] T1. `core/execution/adapters/base.py`: `AdapterDefinition` dataclass + `StreamFormat` enum + `prompt_delivery` enum. *(unit: schema)*
- [ ] T2. `core/execution/adapters/registry.py`: `AdapterRegistry` (load defs, `detect()` PATH probe non-blocking, cache w/ TTL). *(unit: detection with a fake PATH bin + a missing bin)*
- [ ] T3. `core/execution/stream_handlers.py`: `StreamHandlerFactory` for `plain` + `jsonl`; both → canonical `ExecEvent`. *(unit: both formats → same schema)*
- [ ] T4. `core/execution/engine.py`: fill `UnifiedExecutionEngine.run(manifest)` to resolve adapter, spawn, normalize, write `RunProvenance` to KG. *(integration: stub adapter end-to-end)*
- [ ] T5. `core/execution/adapters/defs/`: 2–3 real defs (claude-code, ollama-openai-compatible, generic-cmd) + `fallback_models`. *(unit: build_args)*
- [ ] T6. Wire `step.runtime` selection from the manifest; default = in-process. *(integration: default path unchanged)*

## ORCH-1.34 — Provider-Normalizing Proxy
- [ ] T7. `security/guardrails.py`: `validate_base_url` + `validate_base_url_resolved` (DNS-resolve, IP blocklist, loopback carve-out). *(unit: internal IP via public DNS rejected; loopback allowed)*
- [ ] T8. `core/model_factory.py`: `ProviderProxy` provider + `create_model(provider="custom")`. *(unit: returns a pydantic-ai-compatible model)*
- [ ] T9. `server/routers/proxy.py`: `/api/proxy/<provider>/stream` for anthropic/openai/azure/google/ollama → canonical SSE; mount in `server/app.py`. *(integration: mock upstream → canonical events; SSRF 4xx pre-fetch)*
- [ ] T10. `core/config.py`: `CredentialResolver` (env>file>none) + `model_override_env_var`. *(unit)*

## Cross-cutting
- [ ] T11. C4 + concept docs (`docs/pillars/1_graph_orchestration/ORCH-1.33.md`, `ORCH-1.34.md`); regen `docs/concepts.yaml`.
- [ ] T12. `check_wiring.py` ≤3 hops; CHANGELOG/README/AGENTS lines; full `pytest` green.
