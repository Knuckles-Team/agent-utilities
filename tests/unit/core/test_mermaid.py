from agent_utilities.mermaid import (
    MermaidBuilder,
    FlowchartBuilder,
    ClassDiagramBuilder,
    EntityRelationshipBuilder,
    MermaidTheme,
)
from agent_utilities.models.graph import GraphPlan, ExecutionStep
from agent_utilities.models.sdd import Tasks, Task, TaskStatus, ImplementationPlan
from agent_utilities.models.codemap import CodemapArtifact, CodemapNode, CodemapEdge
from agent_utilities.knowledge_graph.engine import IntelligenceGraphEngine, FocusedSubgraph
import networkx as nx


def test_flowchart_builder():
    builder = FlowchartBuilder(title="Test Flow", direction="LR", theme=MermaidTheme.FOREST)
    builder.add_node("node1", label="Node 1", shape="round", css_class="active")
    builder.add_node("node2", label="Node 2", shape="box")
    builder.add_edge("node1", "node2", label="links to", edge_type="==>")
    builder.add_subgraph("Group A", ["node1", "node2"])

    result = builder.render()
    assert "---" in result
    assert "title: Test Flow" in result
    assert "theme: forest" in result
    assert "flowchart LR" in result
    assert "node1(\"Node 1\")" in result
    assert "class node1 active" in result
    assert "node1 ==> |\"links to\"| node2" in result
    assert "subgraph Group A" in result


def test_class_diagram_builder():
    builder = ClassDiagramBuilder(title="Test Classes")
    builder.add_class("MyClass", attributes=["int x", "string y"], methods=["void run()"], annotation="Interface")
    builder.add_relationship("MyClass", "OtherClass", rel_type="<|--", label="inherits")

    result = builder.render()
    assert "classDiagram" in result
    assert "class MyClass {" in result
    assert "<<Interface>>" in result
    assert "int x" in result
    assert "void run()" in result
    assert "MyClass <|-- OtherClass : inherits" in result


def test_er_diagram_builder():
    builder = EntityRelationshipBuilder(title="Test ER")
    builder.add_entity("User", attributes=[("string", "name"), ("int", "age")])
    builder.add_relationship("User", "Post", rel_type="||--|{", label="writes")

    result = builder.render()
    assert "erDiagram" in result
    assert "User {" in result
    assert "string name" in result
    assert "User ||--|{ Post : \"writes\"" in result


def test_graph_plan_to_mermaid():
    plan = GraphPlan(
        steps=[
            ExecutionStep(node_id="step1", status="completed"),
            ExecutionStep(node_id="step2", depends_on=["step1"], status="in_progress"),
            ExecutionStep(node_id="step3", depends_on=["step1"], is_parallel=True),
        ]
    )
    mermaid = plan.to_mermaid()
    assert "flowchart TD" in mermaid
    assert "step1" in mermaid
    assert "step2" in mermaid
    assert "step1 --> step2" in mermaid
    assert "class step1 success" in mermaid
    assert "class step2 active" in mermaid


def test_sdd_tasks_to_mermaid():
    tasks = Tasks(
        tasks=[
            Task(id="T1", title="Task 1", description="Desc 1", status=TaskStatus.COMPLETED),
            Task(id="T2", title="Task 2", description="Desc 2", depends_on=["T1"], status=TaskStatus.IN_PROGRESS),
        ]
    )
    mermaid = tasks.to_mermaid()
    # Account for sanitization of brackets in label
    assert "T1[\"Task 1\n&#40;T1&#41;\"]" in mermaid
    assert "T1 --> T2" in mermaid
    assert "class T1 completed" in mermaid


def test_implementation_plan_to_mermaid():
    plan = ImplementationPlan(
        feature_id="F1",
        title="Test Feature",
        proposed_changes=["api.py: add endpoint", "models.py: add table"]
    )
    mermaid = plan.to_mermaid()
    assert "classDiagram" in mermaid
    assert "class ImplementationPlan" in mermaid
    assert "class api_py" in mermaid
    assert "ImplementationPlan --> api_py : modifies" in mermaid


def test_codemap_artifact_to_mermaid():
    artifact = CodemapArtifact(
        id="test-map",
        prompt="Visualize code",
        mode="smart",
        nodes=[
            CodemapNode(id="f1", label="main.py", type="file", file="main.py", line=1, end_line=10, description="test", importance=0.5),
            CodemapNode(id="c1", label="MyClass", type="class", file="main.py", line=1, end_line=10, description="test", importance=0.5),
        ],
        edges=[
            CodemapEdge(source="f1", target="c1", type="contains")
        ]
    )
    mermaid = artifact.to_mermaid()
    assert "Codemap: Visualize code" in mermaid
    # Account for sanitization
    assert "f1[(\"main.py\n&#40;file&#41;\")]" in mermaid
    assert "f1 --> |\"contains\"| c1" in mermaid


def test_kg_mermaid_generation():
    graph = nx.MultiDiGraph()
    graph.add_node("agent1", type="agent", name="Agent 1")
    graph.add_node("mem1", type="memory", description="Some memory")
    graph.add_edge("agent1", "mem1", type="CREATED")

    engine = IntelligenceGraphEngine(graph=graph)
    mermaid = engine.generate_mermaid_graph()

    # Account for sanitization
    assert "agent1((\"Agent 1\n&#40;agent&#41;\"))" in mermaid
    assert "mem1[(\"mem1\n&#40;memory&#41;\")]" in mermaid
    assert "agent1 --> |\"CREATED\"| mem1" in mermaid


def test_focused_subgraph_to_mermaid():
    subgraph = FocusedSubgraph(
        nodes=[
            {"id": "n1", "label": "Node 1", "type": "class"},
            {"id": "n2", "label": "Node 2", "type": "file"},
        ],
        edges=[
            {"source": "n1", "target": "n2", "type": "defined_in"}
        ],
        summary="Test subgraph",
        query="test"
    )
    assert True, 'Focused subgraph rendering completed'
from agent_utilities.graph.mermaid import get_graph_mermaid

def test_mermaid_builder_sanitization():
    builder = MermaidBuilder()
    text = "Link [text](url) with (brackets) and [others]"
    sanitized = builder._sanitize(text)
    assert "url" not in sanitized
    assert "&#40;brackets&#41;" in sanitized
    assert "&#91;others&#93;" in sanitized


def test_flowchart_builder_extended():
    builder = FlowchartBuilder(title="Extended Flow")
    builder.add_node("n1", shape="diamond")
    builder.add_node("n2", shape="circle")
    builder.add_node("n3", shape="cylinder")
    builder.add_node("n4", shape="unknown")
    builder.add_edge("n1", "n2", edge_type="---")
    builder.add_edge("n2", "n3", edge_type="-.->")
    builder.add_edge("n3", "n4", edge_type="==>")

    result = builder.render()
    assert "n1{\"n1\"}" in result
    assert "n2((\"n2\"))" in result
    assert "n3[(\"n3\")]" in result
    assert "n4(\"n4\")" in result
    assert "n1 --- n2" in result
    assert "n2 -.-> n3" in result


def test_class_diagram_builder_minimal():
    builder = ClassDiagramBuilder()
    builder.add_class("Simple")
    result = builder.render()
    assert "class Simple {" in result


def test_get_graph_mermaid():
    class MockGraph:
        def render(self):
            return "flowchart TD\n  router --> domain_execution"

    graph = MockGraph()
    config = {"router_model": "gpt-4"}

    # Test with title and routed_domain
    result = get_graph_mermaid(graph, config, title="Custom Title", routed_domain="test_domain")
    assert "title: Custom Title" in result
    assert "Router (gpt-4)" in result
    assert "Domain Node (test_domain)" in result
    assert "classDef router" in result

    # Test with existing title and no routed_domain
    class MockGraphWithCode:
        def mermaid_code(self):
            return "---\ntitle: Old Title\n---\nflowchart TD\n  router"

    result = get_graph_mermaid(MockGraphWithCode(), config)
    assert "title: Graph" in result
    assert "Router (gpt-4)" in result


def test_graph_plan_to_mermaid_failure():
    plan = GraphPlan(
        steps=[
            ExecutionStep(node_id="fail_node", status="failed"),
        ]
    )
    mermaid = plan.to_mermaid()
    assert "class fail_node error" in mermaid


def test_codemap_artifact_to_mermaid_extended():
    artifact = CodemapArtifact(
        id="test-map",
        prompt="Visualize code",
        mode="smart",
        nodes=[
            CodemapNode(id="c1", label="MyClass", type="class", file="main.py", line=1, end_line=10, description="test", importance=0.5),
            CodemapNode(id="f1", label="main.py", type="file", file="main.py", line=1, end_line=10, description="test", importance=0.5),
            CodemapNode(id="m1", label="mod", type="module", file="mod.py", line=1, end_line=10, description="test", importance=0.5),
        ],
        edges=[
            CodemapEdge(source="c1", target="c2", type="inherits"),
            CodemapEdge(source="f1", target="f2", type="imports"),
        ]
    )
    mermaid = artifact.to_mermaid()
    assert "c1(\"MyClass\n&#40;class&#41;\")" in mermaid
    assert "f1[(\"main.py\n&#40;file&#41;\")]" in mermaid
    assert "c1 ==> |\"inherits\"| c2" in mermaid
    assert "f1 -.-> |\"imports\"| f2" in mermaid


def test_graph_plan_to_acp_entries():
    plan = GraphPlan(
        steps=[
            ExecutionStep(node_id="step1", input_data="data1", status="completed"),
            ExecutionStep(node_id="step2", is_parallel=True, status="pending"),
        ]
    )
    entries = plan.to_acp_plan_entries()
    assert len(entries) == 2
    assert entries[0]["content"] == "step1: data1"
    assert entries[0]["status"] == "completed"
    assert entries[0]["priority"] == "high"
    assert entries[1]["priority"] == "medium"


def test_mermaid_builder_render_no_title():
    builder = MermaidBuilder(theme=MermaidTheme.NEUTRAL)
    builder.lines.append("test line")
    result = builder.render()
    assert "---" not in result
    assert "test line" in result


def test_flowchart_builder_no_label():
    builder = FlowchartBuilder()
    builder.add_node("node1")
    result = builder.render()
    assert "node1(\"node1\")" in result


def test_codemap_json_methods():
    artifact = CodemapArtifact(
        id="test-json",
        prompt="json test",
        mode="fast",
        nodes=[],
        edges=[]
    )
    json_str = artifact.to_json()
    assert "test-json" in json_str

    restored = CodemapArtifact.from_json(json_str)
    assert restored.id == "test-json"
