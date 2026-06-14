# Agent-Native Research Artifacts (ARA) vs agent-utilities — comparative analysis & adopt-and-surpass

**Date:** 2026-06-14 · **Paper:** arXiv:2604.24658 *"Agent-Native Research Artifacts"* ·
**Scope:** automated research in agent-utilities (the Loop engine + KG / OWL-RDF ontology) ·
**Concepts delivered:** KG-2.78 (Loop), KG-2.79 (OntologyReasoningDriver), KG-2.80 (ARA)

## 1. What the paper proposes

ARA recasts a research paper from a narrative PDF into an **agent-executable, 4-layer
artifact** plus three mechanisms:

| Element | Paper |
|---|---|
| **4-layer artifact** | `/logic` (claims), `/src` (code), `/trace` (exploration graph: decisions, **dead-ends**, **pivots**), `/evidence` (raw outputs), with forensic bindings between layers |
| **Live Research Manager** | continuously captures the *act* of research as typed events with provenance, crystallizing them into the artifact |
| **ARA Compiler** | converts legacy papers/repos into an ARA (semantic-deconstruct → cognitive-map → physical-ground → exploration-extract) |
| **ARA-Native Seal** | reviews at 3 escalating levels — L1 structural, L2 rigor, L3 execution-reproducibility — with `/evidence` **withheld** so reviewers cannot fabricate-to-pass |
| **Reported lift** | paper-QA 72.4 → 93.7 %; reproduction 57.4 → 64.4 % |
| **Acknowledged weakness** | collective/cross-paper inference only emerges *"at critical mass"* of a corpus |

## 2. Comparative analysis — what we already had vs the gap

| ARA capability | agent-utilities already had (reused) | Gap we built (ontology-native) |
|---|---|---|
| 4-layer artifact + forensic bindings | `ExtractedArticle`/`fact_extractor`, legacy `ResearchArtifact`, `DocumentProcessor` (KG-2.48), the Palantir-parity ontology object/interface system | **ARA as OWL classes + object-properties** — `research_artifact`/`claim`/`code_spec`/`evidence`/`exploration_node` + `contains`/`grounded_in`/`implemented_by`, registered as ontology interfaces (`to_owl`) + typed links + SHACL value-types (A1) |
| exploration graph (dead-end/pivot DAG) | `EvidenceGraph`/`SearchTask` (KG-2.70-72), reserved `trajectory`/`deliberation` types (no producer), failure clustering (AHE-3.18) | a **materialized RDF exploration DAG** producer: failures → `dead_end`, ConceptMatcher rejects → `pivot` (A2) |
| Live Research Manager | `ingest_sessions`/parsers/`collector`, `EditLedger` (KG-2.43), identity (OS-5.14) | typed research events → **RDF individuals with provenance** (user/ai-suggested/ai-executed/user-revised) + crystallize + flush (A3) |
| ARA Compiler | KB extraction, `DocumentProcessor`, codebase ingest, `ConceptMatcher` (KG-2.75) | a compiler that emits an **OWL-native** ARA and **physical-grounds every claim to the ecosystem** (A4) |
| ARA Seal (L1/L2/L3, evidence withheld) | governed auto-merge (AHE-3.14), promotion governance (AHE-3.20), reliability scorers (AHE-3.1), **SHACL** (KG-2.6), interface conformance (KG-2.38), markings (KG-2.46) | Seal **L1 = SHACL + interface conformance + OWL consistency**; L3 evidence-withheld via markings; signed `seal_certificate` (A5) |
| **collective inference "at critical mass"** *(their weakness)* | **whole-ecosystem OWL reasoning today** (KG-2.9 upper ontology, `OWLBridge.run_cycle`, `ecosystem_topology`, Egeria SoR) | **the keystone** — reasoning as the *engine*, extrapolating cross-domain links from artifact #1 (A0/KG-2.79) |

## 3. How we surpass it

1. **Reasoning is the engine, not a post-step.** Each Loop cycle promotes its new
   information, runs `OWLBridge.run_cycle` over the ecosystem, and **harvests the inferred
   edges/concepts back as new topics** (`OntologyReasoningDriver`, KG-2.79). The old
   pipeline ran one-shot enrichment and *never consumed* the inferences — that gap is
   closed; it is now a closed extrapolation loop.
2. **Ecosystem-spanning, not single-repo.** The paper's collective inference waits for a
   corpus. We map `agent-packages/agents/*` + `services/*` + enterprise + research into
   **one ontology**, so a paper's claim is inferred (subClassOf / transitive / inverse /
   property-chain) to relate to a deployed **service** or **agent capability** from the
   first artifact. `grounded_in` is transitive with a `supports` inverse, so claim →
   evidence → ecosystem-code chains materialize automatically.
3. **Workflow/skill loops reason too.** `reason` is a default-on cycle stage for research
   **and** develop/skill Loops, surfacing which services/capabilities an objective touches.
4. **Native provenance, governance, multi-tenant — and dual-surface.** The whole capability
   is exposed identically over the **MCP tool `research_artifact`** and **`POST
   /api/research/*`** through one shared `ARAService` (single source of truth), actor- and
   tenant-scoped (OS-5.14) — vs the paper's single-repo skill.

## 4. What was built (all merged locally; not pushed)

| Slice | Module(s) | Tests |
|---|---|---|
| **L1** Loop unit | `research/loops.py` (`submit_loop`/`active_loops`, kinds research/develop/skill) | `test_loops.py` |
| **A0** reasoning-as-engine (keystone) | `research/ara/reasoning_driver.py`; `reason` stage wired into the cycle | `test_ara_reasoning_driver.py` |
| **A1** OWL-native artifact | `research/ara/artifact.py`; node/edge types + promotable; interfaces `VerifiableClaim`/`ResearchArtifactShape`; typed links `grounds`/`artifact_contains_claim`/`implements_claim`; `ClaimConfidence` value-type; `grounded_in` transitive + `grounded_in`↔`supports` inverse | `test_ara_artifact.py` |
| **A2** exploration producer | `research/ara/exploration.py` (dead-ends ← failures, pivots ← matcher rejects) | `test_ara_exploration_lrm.py` |
| **A3** Live Research Manager | `research/ara/live_manager.py` (7 typed events + provenance + crystallize + flush) | `test_ara_exploration_lrm.py` |
| **A4** ARA Compiler | `research/ara/compiler.py` (deconstruct → lift → ecosystem-ground → materialize) | `test_ara_compiler_seal.py` |
| **A5** ARA Seal | `research/ara/seal.py` (L1 SHACL/interface/OWL-consistency, L2 rigor, L3 evidence-withheld; `seal_certificate`) | `test_ara_compiler_seal.py` |
| **A6** exposure | `research/ara/service.py` + `mcp/kg_server.py` (`research_artifact` tool + route) + `gateway/research_api.py` (granular router) | `test_ara_service_exposure.py` |
| **A7** docs + concepts + report | `owl_rdf_layer.md`, `vendor_neutral_enterprise_ontology.md`, `start-here.md`; `concepts.yaml` regen (247); this report | — |

**Test status:** 34 ARA/loop unit tests green; the broader ontology / owl-bridge /
interface / link / value-type suites pass. Three pre-existing, environment-only failures
are unrelated to this work: PyMuPDF is not installed in the test env, and the running
`epistemic-graph` engine binary predates the `GetTriples` RPC (the documented live-engine
gap).

## 5. Residual / live-verification notes

- **Live demo is environment-blocked**, not code-blocked: the embedder endpoint
  (`vllm-embed.arpa`) is returning 502 and the deployed engine binary lacks `GetTriples`,
  so the full reason→compile→seal round-trip against the live graph awaits an engine
  redeploy onto a current binary + embedder recovery. The unit path is fully exercised with
  injected fakes (fake OWL bridge / generator / ground-fn / judge-fn).
- **Reasoning defaults to lightweight** RDFS+ closures per cycle (transitive/symmetric/
  inverse over the raw graph); full-DL is on-demand and never blocks a Loop (best-effort,
  timeout-bounded, like the other stages).
- **Anti-fabrication** is the Seal-L3 evidence-withholding via markings (KG-2.46); the
  executable re-run gate reuses the Loop's existing regression/sandbox output gate
  (AHE-3.14/3.18), wired by the caller.

## 6. Bottom line

We adopted **all** of ARA and made it **native to the one ontology-driven ecosystem graph**.
The paper's headline weakness — cross-paper inference only "at critical mass" — is exactly
where we are strongest: OWL/RDF reasoning over the whole ecosystem makes *extrapolation the
engine of research*, so the system relates a brand-new paper to the real deployed estate
from the first artifact, and exposes the result over both MCP and REST through one service.
