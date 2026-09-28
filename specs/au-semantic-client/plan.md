# Delivery plan

1. Pin and test the generated EG contract, including schema inspect, validation, graph/session, work, retrieval and memory methods required by active AU callers.
2. Record a caller inventory by module and public entry point. Migrate ontology and shape sources to EG or certified packs, prove semantic parity, then delete AU semantic files and imports.
3. Replace each graph facade and durable writer with typed client calls, removing legacy fallback in the same PR. Keep model-backed extraction as candidate-claim generation only.
4. Migrate untrusted historical data through an attested tenant-bound path or quarantine it. Never infer tenant from a mutable payload field.
5. Run static, contract, served, privacy and whole-repo gates; register exact merged-head evidence and update each item independently.
