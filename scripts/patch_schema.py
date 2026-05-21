import sys

with open("agent_utilities/models/schema_definition.py", "r") as f:
    content = f.read()

new_tables = [
    "GeopoliticalEntity",
    "EnergyEntity",
    "InfrastructureEntity",
    "Country",
    "Region",
    "Leader",
    "Alliance",
    "SanctionEvent",
    "RiskIndex",
    "Commodity",
    "Chokepoint",
    "ProductionFacility",
    "EnergyReserve",
    "SupplyRoute",
    "DisruptionEvent",
    "EconomicIndicator",
    "DemandForecast",
    "WeatherEvent",
]

tables_str = ""
for t in new_tables:
    tables_str += f"""        TableDefinition(
            name="{t}",
            columns={{
                "id": "STRING PRIMARY KEY",
                "type": "STRING",
                "name": "STRING",
                "description": "STRING",
                "importance_score": "FLOAT",
                "timestamp": "STRING",
                "metadata": "STRING",
                "is_permanent": "BOOLEAN",
            }},
        ),
"""

new_rels = [
    "HAS_RISK_ON",
    "IMPOSES_SANCTION_ON",
    "CONTROLS",
    "AFFECTS_PRICE_OF",
    "SUPPLIES_TO",
    "DISRUPTS_SUPPLY_TO",
    "INFLUENCES_VOLATILITY",
    "TRANSPORTS_VIA",
    "IMPACTS_DEMAND_FOR",
    "PREDICTED_IMPACT",
]
rels_str = ""
for r in new_rels:
    # A generic catch-all connection from DomainEntity to DomainEntity is fine, or we can use specific ones.
    # We will just do a generic connection since most of these apply broadly within the domain
    rels_str += f"""        RelDefinition(
            type="{r}",
            connections=[{{"from": "DomainEntity", "to": "DomainEntity"}}],
        ),
"""

# Insert tables
parts = content.split("    ],\n    edges=[\n")
if len(parts) == 2:
    # Insert at the end of tables
    new_content = parts[0] + tables_str + "    ],\n    edges=[\n" + parts[1]

    # Insert at the end of edges
    edges_parts = new_content.rsplit("    ],\n)", 1)
    if len(edges_parts) == 2:
        final_content = edges_parts[0] + rels_str + "    ],\n)\n"
        with open("agent_utilities/models/schema_definition.py", "w") as f:
            f.write(final_content)
        print("Schema successfully updated.")
    else:
        print("Failed to find end of edges.")
else:
    print("Failed to find start of edges.")
