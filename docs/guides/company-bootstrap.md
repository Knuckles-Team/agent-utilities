# Company Bootstrap Deployment Guide

**CONCEPT:AU-KG.research.research-pipeline-runner — Company Operations Domain**
**CONCEPT:AU-ECO.ui.company-infrastructure-orchestration — Company Infrastructure Orchestration**

This guide walks through bootstrapping an autonomous AI-driven company
from scratch using the agent-utilities ecosystem.

## Prerequisites

- [ ] agent-utilities installed and configured (`~/.config/agent-utilities/config.json`)
- [ ] Docker Swarm or standalone Docker available on target hosts
- [ ] Portainer deployed and accessible via `portainer-mcp`
- [ ] AdGuard Home deployed and accessible via `adguard-home-mcp`
- [ ] KG engine running (graph-os MCP server operational)
- [ ] SSH access to target infrastructure via `tunnel-manager-mcp`

## Phase 1: Company Entity Registration

### 1.1 Create Company Profile in KG

```python
from agent_utilities.models.company import CompanyProfile

company = CompanyProfile(
    id="my_company",
    name="My Company LLC",
    legal_name="My Company LLC",
    dba_name="My Company",
    entity_type="llc",
    state_of_incorporation="WY",
    ein="XX-XXXXXXX",
)
```

Register via KG:
```
graph_write(action="add_node", node_id="my_company", node_type="company",
            properties='{"legal_name": "My Company LLC", "entity_type": "llc", ...}')
```

### 1.2 Create Agent Departments

For each department (legal, hr, finance, ops, strategy):
```python
from agent_utilities.models.company import AgentDepartment

legal_dept = AgentDepartment(
    id="dept_legal",
    name="Legal Department",
    department_type="legal",
    team_config_id="tc_legal",
    automation_level=0.7,  # 70% automated, 30% human oversight
    human_oversight_required=True,
    prompt_agent="legal_compliance_coordinator",
)
```

### 1.3 Register Strategic Goals

```python
from agent_utilities.models.company import StrategicGoal

revenue_goal = StrategicGoal(
    id="goal_revenue_2026",
    name="2026 Revenue Target",
    goal_type="profit",
    target_value=1000000.0,
    measurement_unit="USD",
    deadline="2026-12-31",
)
```

## Phase 2: Infrastructure Deployment

### 2.1 Deploy Company Software

Use the infrastructure-blueprints skill-graph to deploy required software:

```bash
# Deploy ERPNext for accounting
portainer_stack(action="create_standalone_stack",
    endpoint_id=1,
    stack_file_content=<erpnext.yaml>)

# Deploy Twenty CRM
portainer_stack(action="create_standalone_stack",
    endpoint_id=1,
    stack_file_content=<twenty-crm.yaml>)

# Deploy Docassemble for legal
portainer_stack(action="create_standalone_stack",
    endpoint_id=1,
    stack_file_content=<docassemble.yaml>)
```

### 2.2 Configure DNS Rewrites

```python
# Via adguard-home-mcp
adguard_home_rewrites(action="add_rewrite",
    params_json='{"domain": "erp.knuckles.team", "answer": "10.0.0.X"}')
adguard_home_rewrites(action="add_rewrite",
    params_json='{"domain": "crm.knuckles.team", "answer": "10.0.0.X"}')
adguard_home_rewrites(action="add_rewrite",
    params_json='{"domain": "legal.knuckles.team", "answer": "10.0.0.X"}')
```

### 2.3 Register in KG Topology

Create CompanySoftware nodes and wire to infrastructure:
```python
from agent_utilities.models.company import CompanySoftware

erp = CompanySoftware(
    id="sw_erpnext",
    name="ERPNext Instance",
    software_name="ERPNext",
    version="15.0",
    function_category="accounting",
    api_endpoint="https://erp.knuckles.team/api",
    dns_rewrite="erp.knuckles.team",
    host_id="host_prod_01",
)
```

## Phase 3: Workflow Registration

### 3.1 Register Company Operations Agents

Each company department maps to an operations coordinator agent. The
following coordinator prompts ship under `agent_utilities/prompts/`:

| Coordinator prompt | Responsibility |
|--------------------|----------------|
| `legal_compliance_coordinator` | Legal compliance review, contract analysis & risk |
| `hr_operations_coordinator` | Employee onboarding, OKR-driven performance review |
| `finance_operations_coordinator` | Payroll computation, financial reporting |

### 3.2 Compile Skills to KG

Company workflows are authored as standard `SKILL.md` skill directories and
compiled into `GraphPlan` objects via `SkillCompiler`, then registered in the
KG. Use `SkillCompiler.compile(skill_dir)` to build a plan and
`SkillCompiler.register_in_kg(engine, skill_dir)` to persist it:

```python
from pathlib import Path

from agent_utilities.workflows.skill_compiler import SkillCompiler

# Compile and register each company operations skill directory
for skill_dir in Path("skills/ops").iterdir():
    plan = SkillCompiler.compile(skill_dir)        # SKILL.md -> GraphPlan
    if plan is not None:
        SkillCompiler.register_in_kg(kg_engine, skill_dir)
```

Alternatively, register a skill directly through MCP with
`graph_orchestrate(action="compile_workflow", ...)`.

## Phase 4: KPI Dashboard Setup

### 4.1 Register Financial KPIs

```python
from agent_utilities.models.company import KPI

mrr_kpi = KPI(
    id="kpi_mrr",
    name="Monthly Recurring Revenue",
    kpi_type="revenue",
    current_value=0.0,
    target_value=100000.0,
    trend="improving",
    measurement_frequency="monthly",
    goal_id="goal_revenue_2026",
)
```

### 4.2 Schedule KPI Updates

Use the KG cron scheduler to periodically update KPIs:
```python
graph_orchestrate(action="trigger_cron_job",
    task="Update all financial KPIs from ERP data",
    agent_name="finance_operations_coordinator")
```

## Phase 5: Security Hardening

> [!CAUTION]
> Before going to production, complete a full OS-5.1 security assessment.

- [ ] Enable TLS for all services
- [ ] Configure SSO/OIDC via Authentik or Keycloak
- [ ] Set up audit logging for all financial operations
- [ ] Enable MFA for human operators
- [ ] Configure network segmentation
- [ ] Set up backup and disaster recovery
- [ ] Run `graph_analyze(action="security_scan")` on the full topology

## Verification Checklist

- [ ] All company nodes registered in KG
- [ ] All departments have TeamConfigNode assignments
- [ ] All software stacks deployed and healthy
- [ ] DNS rewrites resolving correctly
- [ ] Workflow compilation successful
- [ ] KPI dashboard populated
- [ ] Human oversight rules configured for legal, finance, and HR
- [ ] Backup strategy documented and tested
