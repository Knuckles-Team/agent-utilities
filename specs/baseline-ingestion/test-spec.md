# AU-BASELINE-001 — Test specification

Status: BUILDING. Governing spec: [spec.md](spec.md).

| Test ID | Requirement | Level | Setup and input | Expected observation | Evidence |
|---|---|---|---|---|---|
| T-001 | AU-BASELINE-R001 | Unit | Patched thread factory; two starts on one engine | One thread starts, named `KG-Baseline-Ingest`; the second start returns `False` | PASS @ fd74ce90b |
| T-002 | AU-BASELINE-R001 | Unit | Switch off; thread factory raises; planner raises | Each case returns without raising | PASS @ fd74ce90b |
| T-003 | AU-BASELINE-R002 | Unit | Temporary workspace manifest with core, skills and agents repositories | Prompts and three skill providers first, then the checked-out core repositories | PASS @ fd74ce90b |
| T-004 | AU-BASELINE-R002 | Unit | Scopes `all`, a name list and `none`; cap of one; manifest read error | Each scope selects the expected repositories; the cap holds; prompts and skills survive the error | PASS @ fd74ce90b |
| T-005 | AU-BASELINE-R003 | Unit | Recording engine that rejects one target | The other items queue; the rejection names the leg, class and exception type | PASS @ fd74ce90b |
| T-006 | AU-BASELINE-R004 | Unit | Stage table and task types | S1 fast, S2 and S3 medium, S4 to S6 slow heavy; buckets 1, 2 and 3; stamped stage metadata | PASS @ fd74ce90b |
| T-007 | AU-BASELINE-R005 | Unit with MCP client | Two temporary ontology providers | `resources/list` returns two `ontology://` and two `shapes://` URIs; a read returns the Turtle body | PASS @ fd74ce90b |

## Negative and boundary cases

- A provider that raises during registration is skipped; the next provider still registers.
- A resolver failure degrades registration to zero resources.
- An unknown skill provider raises `LookupError` inside its own WorkItem only.
- A prompt ingest runs off the worker event loop and keeps the caller context.
- The maintenance allowlist contains `baseline_prompts`, and the engine exposes `_tick_baseline_prompts`.

## Quality and release proof

```text
pytest tests/unit/knowledge_graph/test_baseline_ingest.py \
       tests/unit/knowledge_graph/test_semantic_tiers.py \
       tests/unit/mcp/test_content_resources.py
python scripts/check_complexity_staged.py
python scripts/check_no_env_sprawl.py
python scripts/check_dupehound.py --base-ref origin/main
python scripts/check_duplication.py enforce --base-ref origin/main
```

Served acceptance needs a daemon-role deployment on an empty store. The proof records the WorkItem list and the sparse-index ratio.
