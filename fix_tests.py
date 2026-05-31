import os
import re

TEST_DIR = "tests"

# Find python files containing IntelligenceGraphEngine
for root, _, files in os.walk(TEST_DIR):
    for f in files:
        if f.endswith(".py"):
            path = os.path.join(root, f)
            with open(path) as file:
                content = file.read()

            original_content = content

            # Replace common variations
            content = re.sub(
                r"IntelligenceGraphEngine\(\s*(?:graph\s*=\s*)?[A-Za-z0-9_]+\s*,\s*backend\s*=\s*None\s*\)",
                'IntelligenceGraphEngine(db_path=":memory:")',
                content,
            )
            content = re.sub(
                r"IntelligenceGraphEngine\(\s*(?:graph\s*=\s*)?[A-Za-z0-9_]+\s*,\s*backend\s*=\s*(.*?)\s*\)",
                r"IntelligenceGraphEngine(backend=\1)",
                content,
            )
            content = re.sub(
                r"IntelligenceGraphEngine\(\s*(?:graph\s*=\s*)?[A-Za-z0-9_]+\s*\)",
                'IntelligenceGraphEngine(db_path=":memory:")',
                content,
            )
            # Remove any trailing kwargs if it was just graph=X
            content = re.sub(
                r'IntelligenceGraphEngine\(\s*graph\s*=\s*GraphComputeEngine\([^)]*\)\s*,\s*db_path\s*=\s*":memory:"\s*\)',
                'IntelligenceGraphEngine(db_path=":memory:")',
                content,
            )

            if content != original_content:
                with open(path, "w") as file:
                    file.write(content)
                print(f"Fixed {path}")
