import glob
import json
import os

prompts_dir = (
    "/home/apps/workspace/agent-packages/agent-utilities/agent_utilities/prompts"
)
ontology_file = "/home/apps/workspace/agent-packages/agent-utilities/agent_utilities/knowledge_graph/ontology_company.ttl"

json_files = glob.glob(os.path.join(prompts_dir, "*.json"))


def assign_dept(role_name: str) -> str:
    role_lower = role_name.lower()
    if any(
        k in role_lower
        for k in ["programmer", "engineer", "architect", "coder", "devops"]
    ):
        return "Engineering"
    elif any(k in role_lower for k in ["qa", "tester", "auditor", "verifier"]):
        return "QA"
    elif any(
        k in role_lower for k in ["manager", "coordinator", "officer", "operations"]
    ):
        return "Operations"
    elif any(
        k in role_lower for k in ["designer", "ui", "ux", "content", "brand", "media"]
    ):
        return "Product"
    elif any(k in role_lower for k in ["research", "scientist", "quant", "analyst"]):
        return "Research"
    elif any(k in role_lower for k in ["finance", "trading", "arr"]):
        return "Finance"
    elif any(
        k in role_lower for k in ["compliance", "legal", "safety", "guard", "policy"]
    ):
        return "Compliance"
    elif "specialist" in role_lower:
        return "IT"
    return "Admin"


departments = {}
agent_roles = {}
tools_set = set()

for jf in json_files:
    try:
        with open(jf) as f:
            data = json.load(f)
    except json.JSONDecodeError:
        continue

    agent_id = data.get("task", "")
    if not agent_id:
        continue

    identity = data.get("identity", {})
    role = identity.get("role", agent_id.replace("_", " ").title())

    dept = assign_dept(role)
    if dept not in departments:
        departments[dept] = []

    tools = data.get("tools", [])
    for t in tools:
        tools_set.add(t)

    node_name = "Agent" + agent_id.replace("_", " ").title().replace(" ", "")
    agent_roles[agent_id] = {
        "node_name": node_name,
        "role_name": role,
        "dept": dept,
        "tools": tools,
        "id": agent_id,
    }
    departments[dept].append(agent_roles[agent_id])

ttl_lines = [
    "\n",
    "### AUTO-GENERATED ENTERPRISE ORG CHART ###\n",
    ':AgentRole a owl:Class ;\n    rdfs:label "Agent Role" ;\n    rdfs:subClassOf bfo:0000023 .\n',
    ':MCPServer a owl:Class ;\n    rdfs:label "MCP Server" ;\n    rdfs:subClassOf bfo:0000031 .\n',
    ":hasAgentRole a owl:ObjectProperty ;\n    rdfs:domain :Department ;\n    rdfs:range :AgentRole .\n",
    ":usesTool a owl:ObjectProperty ;\n    rdfs:domain :AgentRole ;\n    rdfs:range :MCPServer .\n",
    ":reportsTo a owl:ObjectProperty ;\n    rdfs:domain :AgentRole ;\n    rdfs:range :AgentRole .\n",
    "\n",
]

for dept in departments:
    dept_node = "Dept" + dept
    ttl_lines.append(f':{dept_node} a :Department ;\n    rdfs:label "{dept}" .\n')

for tool in tools_set:
    tool_node = "Tool" + tool.replace("-", "_").replace("_", " ").title().replace(
        " ", ""
    )
    ttl_lines.append(f':{tool_node} a :MCPServer ;\n    rdfs:label "{tool}" .\n')

for aid, rdata in agent_roles.items():
    node = rdata["node_name"]
    dept = rdata["dept"]
    dept_node = "Dept" + dept
    ttl_lines.append(f":{dept_node} :hasAgentRole :{node} .\n")

    ttl_lines.append(f":{node} a :AgentRole ;")
    ttl_lines.append(f"    rdfs:label \"{rdata['role_name']}\" ;")
    ttl_lines.append(f"    :id \"{rdata['id']}\" ;")
    ttl_lines.append(f"    :role \"{rdata['role_name']}\" .\n")

    for tool in rdata["tools"]:
        tool_node = "Tool" + tool.replace("-", "_").replace("_", " ").title().replace(
            " ", ""
        )
        ttl_lines.append(f":{node} :usesTool :{tool_node} .\n")

with open(ontology_file, "a") as f:
    f.writelines(ttl_lines)

print(
    f"Added {len(agent_roles)} roles across {len(departments)} departments to {ontology_file}."
)
