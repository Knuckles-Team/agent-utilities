#!/usr/bin/env python3
import re
from pathlib import Path

DOCS_DIR = Path("docs")
CONCEPT_MAP = DOCS_DIR / "concept_map.md"


def load_canonical_concepts():
    """Extract all valid CONCEPT IDs and their names from concept_map.md."""
    valid_concepts = {}
    if not CONCEPT_MAP.exists():
        print(f"Error: {CONCEPT_MAP} not found!")
        return valid_concepts

    with open(CONCEPT_MAP) as f:
        content = f.read()

    # Matches | `ORCH-1.0` | Intelligence Graph Core |
    matches = re.findall(r"\|\s*`([A-Z]+-\d+\.\d+)`\s*\|\s*([^\|]+?)\s*\|", content)
    for m in matches:
        concept_id, name = m[0], m[1].strip()
        # Clean up markers like 🔬
        name = name.replace("🔬", "").strip()
        valid_concepts[concept_id] = name

    return valid_concepts


def get_injection_rules():
    """Defines substring-to-concept rules for automated injection."""
    return {
        "ORCH-1.0": [
            "graph core",
            "orchestration core",
            "joiner",
            "dispatcher",
            "intelligencepipeline",
            "ai (vercel sdk)",
            "vercel sdk",
            "query router",
            "intelligence graph engine",
            "intelligencegraphengine",
            "kgteamcomposer",
            "magma",
            "user query",
            "response composition",
            "user query + images",
            "synthesizer",
            "query: 'deploy",
            "user request + images",
            "query",
        ],
        "ORCH-1.1": [
            "planner",
            "htn",
            "hierarchical task network",
            "hierarchicalplanner",
            "branch{decision}",
            "decision",
            "task a",
            "task b",
            "task c",
            "task decomposer",
        ],
        "ORCH-1.2": [
            "router",
            "routing",
            "discovery",
            "teamconfig",
            "specialist selection",
            "graph router",
            "teamconfig match",
            "specialist selector",
            "contrarian",
            "outsider",
            "specialist",
            "devops specialist",
            "python specialist",
            "specialist selection",
            "council",
            "critic",
            "trader",
            "chairman",
            "architect",
            "proposal 1",
            "proposal 2",
            "proposal 3",
            "proposal 4",
            "councilverdict",
            "reviewer",
            "researcher",
            "skip llm planning",
            "direct dispatch",
            "find_matching_team_config",
            "programmer",
            "programmer output hidden",
            "expansionist",
            "observer",
            "reflector",
            "planning agent",
        ],
        "ORCH-1.3": [
            "safety",
            "state manager",
            "usage guard",
            "rate limiting",
            "verify",
            "statecheckpointer",
            "error_recovery",
            "global_elicitation_callback",
        ],
        "ORCH-1.4": [
            "wiring engine",
            "capability wiring",
            "registered capabilities",
            "abstractcapability",
            "capabilityhandlerprotocol",
            "capabilityorchestrator",
            "wiringengine",
            "topologyengine",
        ],
        "ORCH-1.6": [
            "dstdd",
            "sdd implementation plan",
            "acceptance.md",
            "active.md",
            "constraints.md",
            "observations.md",
            "profile.md",
            "reflections.md",
            "requirements.md",
            "spec",
            "implementation plan",
        ],
        "ORCH-1.20": [
            "graph factory",
            "factory",
            "agent resolution",
            "create_graph_agent",
            "agent spawned",
            "tool binding",
            "topological sort",
        ],
        "ORCH-1.21": [
            "agent runner",
            "runner",
            "graph_orchestrate",
            "config builder",
            "run_graph",
            "end result",
            "completed",
            "running",
            "preempt",
            "failure taxonomies",
            "quota exceeded?",
            "mixed dag",
            "parallel execution",
            "parallel groups",
            "sequential steps",
            "priority queue",
            "event queue",
            "mixed sources list",
            "execution plan",
            "executor",
            "group 1",
            "group 2",
            "startup context builder",
            "waiting",
            "recursive_orchestrator",
        ],
        "KG-2.0": [
            "active knowledge graph",
            "knowledge graph",
            "graph db",
            "ladybug",
            "neo4j",
            "query_cypher",
            "backend.execute",
            "add_node",
            "link_nodes",
            "pggraph csr",
            "active node",
            "status=active",
            "kg: status=active",
            "node tables",
            "kg_edges table",
            "sqlite",
            "permanently removed",
            "kg: permanently removed",
            "remove kg",
            "kg: taskmanager",
            "kg_diff",
            "diffentry",
            "kg: diffentry node",
            "kg_ingest",
            "kg: ingest engine",
            "ladybugdb",
            "transpiler",
            "falkordb",
            "bolt protocol",
            "redis protocol",
            "graphbackend",
            "graph execution",
            "checkpoint to kg",
            "assimilated_into",
            "graph build",
            "graphmaintainer",
            "grok build",
            "hydrate kg",
            "kg resolution",
            "kg semantic clustering",
            "match callableresource",
            "match server nodes",
            "memory crud",
            "load_subgraph",
            "backend type?",
            "pggraph extension",
            "entity view",
            "versioned files + kg nodes",
        ],
        "KG-2.1": ["tiered memory", "memory tier", "contextual embeddings"],
        "KG-2.2": [
            "ontology",
            "epistemics",
            "owl materialize",
            "kg: owl bridge",
            "kg_owl",
            "concept cross-reference matrix",
            "policyingestor",
            "constitution rules",
            "constitution compliance",
            "rules & policies",
            "concept map",
            "kg: concept map",
            "shacl",
            "concept:",
            "first principles",
            "hermit reasoner",
            "materializer",
            "owlbackend abc",
            "owlbridge",
            "owlready2backend",
        ],
        "KG-2.3": [
            "retrieval",
            "hybrid search",
            "paradedb",
            "bm25",
            "pgvector hnsw",
            "search_knowledge_graph",
            "semantic_search",
            "search_hybrid",
            "lexical ranking",
            "cosine similarity",
            "hybrid semantic search",
            "semantic view",
            "cosine search",
            "vectormcpbackend",
            "filter",
        ],
        "KG-2.4": ["inductive", "hypergraph"],
        "KG-2.5": [
            "topological analysis",
            "analysis engine",
            "get_code_impact",
            "causal",
        ],
        "KG-2.6": ["finance domain", "market analyzer", "kalman filter"],
        "KG-2.7": [
            "research intelligence",
            "ingest paper",
            "scholarx",
            "document ingestion",
            "ingestion pipeline",
            "deletion pipeline",
            "update pipeline",
            "cleanup manager",
            "layered evidence corpus",
        ],
        "KG-2.8": ["enterprise domain"],
        "KG-2.9": ["quant orchestration"],
        "KG-2.10": [
            "observational memory",
            "memento",
            "l1 summary",
            "l2 summary",
            "rlm summarizer",
            "message 1",
            "message 2",
            "message 3",
            "message 4",
            "message 5",
            "summaries & clusters",
        ],
        "KG-2.12": [
            "temporal weighted decay",
            "temporal decay",
            "tsgraph",
            "temporal view",
        ],
        "KG-2.13": [
            "external graph federation",
            "external graph",
            "external ontology",
            "ontologyreference",
            "federated query",
            "sparql",
            "leanix",
            "factsheet",
            "external system",
            "ontology_enterprise",
            "multi-tenancy",
            "stardog",
            "fuseki",
            "jenabackend",
            "ext graphs",
        ],
        "AHE-3.0": ["agentic harness", "harness core"],
        "AHE-3.1": [
            "evaluation engine",
            "continuous eval",
            "evalrunner",
            "evaluator",
            "ahe_eval",
            "ahe: evaluator",
            "continuous evaluation",
            "distill",
            "outcome evaluation",
            "outcome rewards",
            "verifier",
            "execution outcome",
            "record_team_outcome",
            "validate",
        ],
        "AHE-3.2": [
            "evolution engine",
            "ahe_evolve",
            "ahe: evolution",
            "skill evolver",
            "eco_skill",
            "eco: skill evolver",
            "prompt resolution",
            "skill evolution",
            "evolution",
        ],
        "AHE-3.3": [
            "team optimization",
            "immunity",
            "ahe_immune",
            "ahe: immunity",
            "self-improvement",
            "autonomous self-improvement",
            "self-model",
            "selfmodel",
            "coalition:",
            "promote_coalition_to_template",
            "teamcomposition",
        ],
        "AHE-3.5": ["heavy thinking", "background intelligence"],
        "AHE-3.6": ["backtest", "curriculum"],
        "ECO-4.0": [
            "mcp tool",
            "tool interface",
            "mcp factory",
            "scholarx paper search",
            "systems-manager",
            "container-manager",
            "tunnel-manager",
            "repository-manager",
            "searxng-mcp",
            "eco_tool",
            "eco: tool executor",
            "host:server",
            "mcp server",
            "fastmcp",
            "tailwindcss",
            "external consumer",
            "claude code",
            "codex",
            "windsurf",
            "devin",
            "opencode",
            "antigravity ide",
            "ag-ui",
            "sse stream",
            "agent-terminal-ui",
            "agent-webui",
            "agent-utilities server",
            "tool call",
        ],
        "ECO-4.1": [
            "a2a network",
            "a2a",
            "a2a adapter",
            "acp",
            "a2a card fetcher",
            "ws sync",
        ],
        "ECO-4.2": ["community telemetry", "ecosystem map"],
        "ECO-4.3": ["market data connectors"],
        "ECO-4.10": [
            "agent toolkit ingestor",
            "type detector",
            "agent registry",
            "agent-packages",
            "agent package",
            "available/",
            "installed/",
            "sources:",
            "auto-detect",
            "package json",
            "invalidate cache",
            "cache\ninvalidation",
            "registry cache",
            "model registry",
            "serviceregistry",
            "mcp config parser",
            "skill parser",
            "tool flag parser",
            "freshness check",
            "skip tool extraction",
            "open-source-libraries",
            "remove mcp config",
            "invalidate_registry_cache",
            "pipeline completion",
            "universal skills, skill graphs",
            "mcpagentregistrymodel",
            "mcpagentregistrymodel",
            "kg: server + callableresource",
            "merge mcp config",
            "remote json fetcher",
            "cache invalidation",
        ],
        "ECO-4.11": [
            "mcp live discovery",
            "live discovery",
            "post /mcp/reload",
            "hot reload /mcp/reload",
        ],
        "OS-5.0": ["os kernel", "xdg paths", "host os"],
        "OS-5.1": [
            "security",
            "auth",
            "threat scanner",
            "secret engine",
            "identity/policy",
            "os_policy",
            "os: policy engine",
            "hmac",
            "hmac sign",
            "promptinjectionscanner",
            "approvalmanager",
            "anonymize",
            "agentidentity",
            "check policy",
            "issue_identity",
            "unified id registry",
        ],
        "OS-5.2": [
            "guardrails",
            "cognitive scheduler",
            "os_guard",
            "os: guardrails",
            "doomloopdetector",
            "recursiondepthexceeded",
            "slot available?",
            "tool_guard: requires_approval",
        ],
        "OS-5.4": ["telemetry", "observability", "logfire", "langfuse", "runtrace"],
    }


def parse_line_nodes(line_str):
    """Find all node definitions on a single line.

    Returns a list of tuples: (node_id, label, raw_part).
    """
    if (
        not line_str
        or line_str.startswith("%%")
        or line_str.startswith("subgraph")
        or line_str.startswith("direction")
        or line_str.startswith("style")
        or line_str.startswith("end")
        or line_str.startswith("graph")
        or line_str.startswith("flowchart")
        or line_str.startswith("C4Context")
        or line_str.startswith("C4Container")
        or line_str.startswith("C4Component")
        or line_str.startswith("title")
    ):
        return []

    if any(
        line_str.startswith(x)
        for x in ["Person", "System", "System_Ext", "Container", "Component", "Rel"]
    ):
        return []

    # Split line by arrow connections or transitions
    parts = re.split(r"\s*(?:--+|-\.-+|==+)(?:>|<|>)?(?:\|[^|]+\|)?\s*", line_str)

    nodes = []
    for part in parts:
        part = part.strip()
        if not part:
            continue

        match = re.match(r'^([A-Za-z0-9_]+)\s*([\[\(\{>]+.*[\]\)\}"]+)$', part)
        if match:
            node_id = match.group(1)
            brackets_label = match.group(2)

            # Extract actual text within the outermost brackets/quotes
            label_match = re.search(r'[\[\(\{">]+(.*)[\]\)\}"]+', brackets_label)
            label = label_match.group(1) if label_match else brackets_label
            label = label.replace('"', "").strip()
            label = re.sub(r'^[\[\(\{">]+|[\]\)\}"]+$', "", label).strip()

            if any(
                x in label.lower()
                for x in [
                    "phase ",
                    "stage ",
                    "step ",
                    "background research",
                    "synthesis",
                    "feature recommendations",
                    "wiring audit",
                ]
            ):
                continue
            if any(
                y in label.lower()
                for y in [
                    "<b>",
                    "<br",
                    "pydantic",
                    "scripts/",
                    "git:",
                    "fastapi",
                    "vite",
                    "react",
                    "textual",
                    "rich",
                    "httpx",
                    "neo4j",
                    "networkx",
                    "database",
                    "sqlite",
                    "postgresql",
                ]
            ):
                continue
            if any(
                y == label.lower()
                for y in [
                    "nx",
                    "val",
                    "exp",
                    "evo",
                    "db",
                    "ui",
                    "api",
                    "cli",
                    "auth",
                    "mcp",
                    "htn",
                    "c4",
                ]
            ):
                continue

            nodes.append((node_id, label, part))

    return nodes


def inject_diagrams():
    valid_concepts = load_canonical_concepts()
    rules = get_injection_rules()

    concept_names_lower = {v.lower(): k for k, v in valid_concepts.items()}

    total_injected = 0
    files_modified = 0

    for md_file in DOCS_DIR.rglob("*.md"):
        with open(md_file) as f:
            content = f.read()

        new_content = []
        in_mermaid = False
        file_changed = False

        for line in content.split("\n"):
            if line.strip().startswith("```mermaid"):
                in_mermaid = True
                new_content.append(line)
                continue
            if in_mermaid and line.strip() == "```":
                in_mermaid = False
                new_content.append(line)
                continue

            if in_mermaid:
                line_str = line.strip()
                detected_nodes = parse_line_nodes(line_str)
                if detected_nodes:
                    new_line = line
                    for node_id, label, part in detected_nodes:
                        # Skip if it already has a Concept ID
                        if re.search(r"([A-Z]+-\d+\.\d+)", part):
                            continue

                        injected_id = None

                        # 0. Try ontological prefix/symbol check (KG-2.2 Ontology & Epistemics)
                        if (
                            label.strip().startswith(":")
                            or "⊂" in label
                            or "≡" in label
                            or any(
                                label.lower().strip().startswith(x)
                                for x in ["bfo:", "prov:", "skos:", "schema:"]
                            )
                        ):
                            injected_id = "KG-2.2"

                        # 1. Try exact concept name match
                        if not injected_id:
                            label_clean = (
                                label.lower()
                                .replace("\\n", " ")
                                .replace("\n", " ")
                                .strip()
                            )
                            for name_lower, c_id in concept_names_lower.items():
                                if name_lower in label_clean:
                                    injected_id = c_id
                                    break

                        # 2. Try rules
                        if not injected_id:
                            label_clean = (
                                label.lower()
                                .replace("\\n", " ")
                                .replace("\n", " ")
                                .strip()
                            )
                            for c_id, keywords in rules.items():
                                if any(kw in label_clean for kw in keywords):
                                    injected_id = c_id
                                    break

                        if injected_id:
                            # Let's locate the label in the part and replace it
                            idx = part.find(label)
                            if idx != -1:
                                new_part = (
                                    part[:idx]
                                    + f"{injected_id}: {label}"
                                    + part[idx + len(label) :]
                                )
                                new_line = new_line.replace(part, new_part, 1)
                                total_injected += 1
                                file_changed = True

                    new_content.append(new_line)
                    continue

            new_content.append(line)

        if file_changed:
            with open(md_file, "w") as f:
                f.write("\n".join(new_content))
            files_modified += 1
            print(f"Injected concept IDs into {md_file.relative_to(DOCS_DIR)}")

    print(f"Total nodes injected: {total_injected}")
    print(f"Files modified: {files_modified}")


if __name__ == "__main__":
    inject_diagrams()
