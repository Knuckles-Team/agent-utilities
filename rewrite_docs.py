import re


def rewrite_overview():
    with open("docs/overview.md") as f:
        content = f.read()

    # Remove ORCH-1.4
    content = re.sub(
        r'ORCH14\["<b>ORCH-1.4: Swarm Preset Template Engine</b>"\]\n\s*', "", content
    )
    content = re.sub(r"\| \*\*ORCH-1\.4\*\* \| \[.*?\n", "", content)

    # Rename OS-5.4, 5.11, 5.12 -> Threat Defense Engine
    content = re.sub(
        r'OS54\["<b>OS-5.4: Prompt Injection Scanner</b>"\]',
        'OS54["<b>OS-5.4: Threat Defense Engine (Injection)</b>"]',
        content,
    )
    content = re.sub(
        r'OS511\["<b>OS-5.11: Topological Vulnerability Scanner</b>"\]',
        'OS511["<b>OS-5.11: Threat Defense Engine (Topological)</b>"]',
        content,
    )
    content = re.sub(
        r'OS512\["<b>OS-5.12: Jailbreak Robustness Hardening</b>"\]',
        'OS512["<b>OS-5.12: Threat Defense Engine (Jailbreak)</b>"]',
        content,
    )

    content = content.replace(
        "Prompt Injection Scanner", "Threat Defense Engine (Injection)"
    )
    content = content.replace(
        "Topological Vulnerability Scanner", "Threat Defense Engine (Topological)"
    )
    content = content.replace(
        "Jailbreak Robustness Hardening", "Threat Defense Engine (Jailbreak)"
    )
    content = content.replace(
        "Enhanced Doom-Loop Detector", "Execution Stability Engine (Doom-Loop)"
    )

    # Context Compaction -> Adaptive Context Manager
    content = content.replace(
        "Token-Aware Context Compaction", "Adaptive Context Manager (Compaction)"
    )
    content = content.replace(
        "Elastic Context Operators", "Adaptive Context Manager (Operators)"
    )
    content = content.replace(
        "Multi-Timescale Memory Dynamics", "Adaptive Context Manager (Multi-Timescale)"
    )

    # Evaluation
    content = content.replace(
        "Evaluation & Distillation", "Continuous Evaluation Engine (Distillation)"
    )
    content = content.replace(
        "Multi-Strategy EvalRunner", "Continuous Evaluation Engine (EvalRunner)"
    )
    content = content.replace(
        "Backtest Evaluation Harness", "Continuous Evaluation Engine (Backtest)"
    )

    # Knowledge Retrieval Engine
    content = content.replace(
        "Hybrid Search Index", "Knowledge Retrieval Engine (Hybrid Index)"
    )
    content = content.replace(
        "RAG-KG Unification", "Knowledge Retrieval Engine (RAG-KG)"
    )
    content = content.replace(
        "Graph Distillation Migration",
        "Knowledge Retrieval Engine (Graph Distillation)",
    )

    # Ontological Reasoning Engine
    content = content.replace(
        "Ontology & Epistemics", "Ontological Reasoning Engine (Epistemics)"
    )
    content = content.replace(
        "OWL-Driven Semantic Subsumption", "Ontological Reasoning Engine (Subsumption)"
    )
    content = content.replace(
        "Structural Causal Reasoning Engine", "Ontological Reasoning Engine (Causal)"
    )

    # Execution Stability Engine
    content = content.replace(
        "Structured Retry Manager", "Execution Stability Engine (Retry)"
    )
    content = content.replace(
        "Tool Repetition Guard", "Execution Stability Engine (Repetition Guard)"
    )

    with open("docs/overview.md", "w") as f:
        f.write(content)


rewrite_overview()
