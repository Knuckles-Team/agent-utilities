def rewrite_1():
    path = "docs/pillars/1_graph_orchestration.md"
    with open(path) as f:
        content = f.read()

    # Remove ORCH-1.4 reference
    content = content.replace(
        "via `Swarm Preset` convergence and evolutionary aggregation.",
        "via dynamic subgraph convergence and evolutionary aggregation.",
    )

    # Update Concepts
    content = content.replace(
        "- **ORCH-1.0**: Unified Intelligence Graph",
        "- **ORCH-1.0**: Dynamic Subgraph Orchestrator",
    )
    content = content.replace(
        "- **ORCH-1.1**: Recursive HTN Planning & Execution Visibility",
        "- **ORCH-1.1**: Agentic Planning Engine (Planning)",
    )
    content = content.replace(
        "- **ORCH-1.2**: Specialist Routing & Registry Hot Cache",
        "- **ORCH-1.2**: Agentic Planning Engine (Routing)",
    )

    with open(path, "w") as f:
        f.write(content)


def rewrite_2():
    path = "docs/pillars/2_epistemic_knowledge_graph.md"
    with open(path) as f:
        content = f.read()

    content = content.replace(
        "Token-Aware Context Compaction", "Adaptive Context Manager"
    )
    content = content.replace(
        "Elastic Context Operators", "Adaptive Context Manager (Operators)"
    )
    content = content.replace(
        "Multi-Timescale Memory Dynamics", "Adaptive Context Manager (Timescale)"
    )

    with open(path, "w") as f:
        f.write(content)


def rewrite_3():
    path = "docs/pillars/3_agentic_harness_engineering.md"
    with open(path) as f:
        content = f.read()

    content = content.replace(
        "Evaluation & Distillation", "Continuous Evaluation Engine"
    )
    content = content.replace(
        "Multi-Strategy EvalRunner", "Continuous Evaluation Engine (EvalRunner)"
    )
    content = content.replace(
        "Backtest Evaluation Harness", "Continuous Evaluation Engine (Backtest)"
    )
    content = content.replace("Structured Retry Manager", "Execution Stability Engine")
    content = content.replace(
        "Temporal Drift & EWC Consolidation", "Continual Learning Engine"
    )

    with open(path, "w") as f:
        f.write(content)


def rewrite_4():
    path = "docs/pillars/4_ecosystem_peripherals.md"
    with open(path) as f:
        content = f.read()

    content = content.replace("MCP & Universal Skills", "Capability Registry Engine")
    content = content.replace(
        "Self-Describing Function Registry", "Capability Registry Engine (Functions)"
    )
    content = content.replace("Dynamic Skill Evolution", "Skill Evolution Engine")
    content = content.replace(
        "Dynamic Tool Assignment Orchestration", "Dynamic Tool Orchestrator"
    )

    with open(path, "w") as f:
        f.write(content)


def rewrite_5():
    path = "docs/pillars/5_agent_os_infrastructure.md"
    with open(path) as f:
        content = f.read()

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
        "Tool Repetition Guard", "Execution Stability Engine (Repetition Guard)"
    )
    content = content.replace(
        "Enhanced Doom-Loop Detector", "Execution Stability Engine (Doom-Loop)"
    )

    with open(path, "w") as f:
        f.write(content)


rewrite_1()
rewrite_2()
rewrite_3()
rewrite_4()
rewrite_5()
