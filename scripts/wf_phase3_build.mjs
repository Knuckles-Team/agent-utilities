export const meta = {
  name: 'phase3-universal-ingestion',
  description: 'Design the universal multi-modal OWL ingestion components (readers, structure router, ontology grounding, background closure) as complete code for serial integration',
  phases: [{ title: 'Design components' }],
}

const ROOT = '/home/apps/workspace/agent-packages/agent-utilities'

const CONV = `
You are building part of agent-utilities' universal multi-modal ingestion funnel:
  reader -> structure-router -> {open | schema} extraction -> ontology grounding
  -> background OWL-RL/SHACL closure.
The central seam already exists: IngestionEngine._enrich_text (concepts+facts,
chunk-bounded) in agent_utilities/knowledge_graph/ingestion/engine.py, and
read_document_text in .../enrichment/extractors/document.py (now PyMuPDF-fast).

Repo root: ${ROOT}. READ the files named in your task before designing. Follow the
existing code style (type hints, guarded optional imports, best-effort/never-raise
in ingest paths, snake_case, CONCEPT:KG-2.x markers in docstrings). Optional heavy
deps (faster-whisper, python-pptx, openpyxl, pytesseract) must be import-guarded
and auto-detected: run when importable, degrade to a clear no-op otherwise.

Return COMPLETE file contents (not diffs) for any new/replacement module, a concise
wiring_note telling the main loop exactly where/how to integrate (file + anchor +
what to change), and a pytest test file. Do NOT edit any files yourself.`

const SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['component', 'new_files', 'wiring_note', 'test_file', 'usable'],
  properties: {
    component: { type: 'string' },
    new_files: {
      type: 'array',
      items: {
        type: 'object',
        additionalProperties: false,
        required: ['path', 'content'],
        properties: { path: { type: 'string' }, content: { type: 'string' } },
      },
    },
    wiring_note: { type: 'string', description: 'Exact integration steps: file, anchor text, what to insert/replace.' },
    test_file: {
      type: 'object',
      additionalProperties: false,
      required: ['path', 'content'],
      properties: { path: { type: 'string' }, content: { type: 'string' } },
    },
    usable: { type: 'boolean' },
    notes: { type: 'string' },
  },
}

const COMPONENTS = [
  {
    key: 'reader-registry',
    task: `Build a self-registering READER REGISTRY (CONCEPT:KG-2.66 family) at
agent_utilities/knowledge_graph/extraction/readers.py: a @register_reader(*exts)
decorator + a registry dict + a discover() that pkgutil-imports a readers/ subpackage,
mirroring agent_utilities/protocols/source_connectors/registry.py (READ it). Provide
read_any(path) -> str that dispatches by extension/MIME through the registry and falls
back to the existing read_document_text for .pdf/.md/.txt. Register a handful of
built-in readers inline (txt/md already covered by fallback). The wiring_note must
explain how read_document_text / the document adaptor should call read_any so new
modalities flow through the SAME pipeline. Read .../enrichment/extractors/document.py
and .../protocols/source_connectors/registry.py first.`,
  },
  {
    key: 'readers-docs',
    task: `Build modality READERS for documents: email (.eml/.msg via stdlib email +
optional extract-msg), presentations (.pptx via python-pptx), spreadsheets
(.xlsx/.csv via openpyxl/csv). Put them in agent_utilities/knowledge_graph/extraction/
readers_office.py as functions decorated with @register_reader from the reader-registry
component (assume readers.py exists with register_reader). Each returns extracted text
(emails: headers + body; pptx: slide text; xlsx: sheet/row text). All optional deps
import-guarded. Read .../kb/parser.py (it already has docx/epub readers) for style.`,
  },
  {
    key: 'readers-media',
    task: `Build modality READERS for media: audio transcripts (.wav/.mp3/.m4a via
optional faster-whisper, auto-detected, model size from env KG_ASR_MODEL default
"base") and images/scanned PDFs (.png/.jpg/.tiff via optional pytesseract/rapidocr).
Put them in agent_utilities/knowledge_graph/extraction/readers_media.py as
@register_reader functions. Heavy deps strictly import-guarded + lazy (never import at
module load); return "" with a logged note if the dep/model is unavailable. Read
.../enrichment/extractors/document.py for style.`,
  },
  {
    key: 'structure-router',
    task: `Build a STRUCTURE ROUTER at agent_utilities/knowledge_graph/extraction/
structure_router.py: classify_text(text, doc_type="") -> "prose" | "structured" |
"mixed" using cheap heuristics (delimiter density, key:value lines, table/CSV shape,
JSON, known form/record markers like invoices/bills/Jira fields). Provide
route_for_extraction(text) -> {mode, prose_text, records} where structured content is
parsed into key/value records (list[dict]) for schema mapping and prose is returned for
open extraction; mixed splits headers(records) from body(prose). The wiring_note must
explain how IngestionEngine._enrich_text could consult this to pick open (fact_extractor)
vs schema mapping. Read .../extraction/fact_extractor.py and .../enrichment/extractors/
document.py (detect_doc_type) first. Keep it dependency-free (stdlib only).`,
  },
  {
    key: 'ontology-grounding',
    task: `Build an ONTOLOGY GROUNDING step at agent_utilities/knowledge_graph/
extraction/ontology_grounding.py: ground_fact(subject, predicate, obj) ->
{subject_type, object_type, predicate} that maps an extracted entity/predicate onto the
OWL ontology — resolve value types via .../ontology/value_types.py (coerce_value_type/
validate_value_type) and interfaces via .../ontology/interfaces.py (InterfaceRegistry.
resolve_target / Interface.conforms), so cross-modal entities (a "vendor" on a bill, a
"supplier" in CMDB, a "company" in a post) converge on the same OWL class. Provide
ground_facts(facts: list) returning the facts annotated with ontology types (best-effort,
never raise). The wiring_note explains how persist_facts / _extract_facts_into_graph
attaches these as node/edge type properties. READ .../ontology/value_types.py,
.../ontology/interfaces.py, and .../extraction/fact_extractor.py (ExtractedFact,
persist_facts) first.`,
  },
  {
    key: 'closure-job',
    task: `Build a BACKGROUND OWL-RL + SHACL CLOSURE job at agent_utilities/
knowledge_graph/maintenance/owl_closure.py: run_closure(engine, limit=2000) that
promotes recently-ingested nodes to RDF, runs the OWL-RL reasoner to materialize implied
edges back into the graph, and validates against shapes, returning a summary
{promoted, inferred_edges, conforms, violations}. REUSE agent_utilities/knowledge_graph/
core/owl_bridge.py and core/shacl_validator.py (READ both — use their real public APIs;
do not invent methods) + shapes/governance.shapes.ttl. Best-effort, never raise. The
wiring_note must specify (a) a new graph_analyze action "close" in
agent_utilities/mcp/kg_server.py that calls run_closure(_get_engine()) and returns JSON
(give the exact elif snippet, 12-space indent), and (b) a maintenance-tick hookpoint.`,
  },
]

phase('Design components')
const specs = await parallel(
  COMPONENTS.map((c) => () =>
    agent(`${CONV}\n\nCOMPONENT: ${c.key}\n${c.task}`, {
      label: `build:${c.key}`,
      phase: 'Design components',
      schema: SCHEMA,
    }),
  ),
)

return specs.filter(Boolean)
