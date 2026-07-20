import ast
import json
import os


def find_all_python_files(root_dir):
    files = []
    for root, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(".py"):
                files.append(os.path.join(root, filename))
    return files


def extract_imports(filepath):
    imports = []
    try:
        with open(filepath, encoding="utf-8") as f:
            content = f.read()
            tree = ast.parse(content)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.append(node.module)
    except Exception:
        pass
    return imports


def main():
    root_dir = "agent_utilities"
    all_files = find_all_python_files(root_dir)

    callmap = {}
    for filepath in all_files:
        imports = extract_imports(filepath)
        for imp in imports:
            if imp.startswith("agent_utilities."):
                module = imp.replace("agent_utilities.", "").replace(".", "/") + ".py"
                if module not in callmap:
                    callmap[module] = []
                callmap[module].append(filepath)

    if not os.path.exists("gap_plans"):
        os.makedirs("gap_plans")

    with open("gap_plans/_callmap.json", "w", encoding="utf-8") as f:
        json.dump(callmap, f, indent=2)

    print(f"Generated callmap for {len(callmap)} modules in gap_plans/_callmap.json")


if __name__ == "__main__":
    main()
