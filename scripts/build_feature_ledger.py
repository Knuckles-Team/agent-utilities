import ast
import os
import yaml
from typing import Dict, List, Any

TARGET_DIRS = [
    "agent_utilities/graph",
    "agent_utilities/knowledge_graph/core",
    "agent_utilities/knowledge_graph/memory",
    "agent_utilities/knowledge_graph/pipeline/phases",
    "agent_utilities/orchestration",
    "agent_utilities/core",
    "agent_utilities/sdd",
    "agent_utilities/tools",
]

# Explicitly excluded files or directories
EXCLUDES = [
    "__init__.py",
    "agent_utilities/core/paths.py",
    "agent_utilities/core/exceptions.py",
]

def find_files() -> List[str]:
    files = []
    for d in TARGET_DIRS:
        for root, _, filenames in os.walk(d):
            for filename in filenames:
                if filename.endswith(".py"):
                    filepath = os.path.join(root, filename)
                    if not any(filepath.endswith(ex) for ex in EXCLUDES):
                        files.append(filepath)
    return sorted(files)

def parse_file(filepath: str) -> List[Dict[str, Any]]:
    entries = []
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                    if not node.name.startswith("_"):
                        docstring = ast.get_docstring(node)
                        
                        # Generate ID
                        fid = f"{os.path.basename(filepath).replace('.py', '')}_{node.name}"
                        
                        entries.append({
                            "id": fid,
                            "name": node.name,
                            "concept": "UNKNOWN",
                            "source": f"{filepath}:{node.lineno}",
                            "behavior": (docstring or "TODO: Describe behavior").split("\n")[0].strip(),
                            "status": "stubbed-intended", # default to require migration
                            "target": "TODO",
                            "characterization_test": "TODO"
                        })
    except Exception as e:
        print(f"Error parsing {filepath}: {e}")
    return entries

def main():
    if not os.path.exists("gap_plans"):
        os.makedirs("gap_plans")
        
    all_entries = []
    for filepath in find_files():
        all_entries.extend(parse_file(filepath))
        
    with open("gap_plans/_feature_ledger.yaml", "w", encoding="utf-8") as f:
        yaml.dump(all_entries, f, default_flow_style=False, sort_keys=False)
        
    print(f"Generated {len(all_entries)} entries in gap_plans/_feature_ledger.yaml")

if __name__ == "__main__":
    main()
