import sys
from pathlib import Path

# Add the scripts directory to sys.path so we can import mermaid_linter
scripts_dir = str(Path(__file__).resolve().parents[1] / "scripts")
if scripts_dir not in sys.path:
    sys.path.insert(0, scripts_dir)

import mermaid_linter


def test_validate_valid_mermaid_block():
    # A perfectly valid diagram with double quotes and no special characters in unquoted nodes
    block = [
        (1, "flowchart LR"),
        (2, "    A --> B"),
        (3, '    C["Label & Special Characters"] --> D'),
    ]
    findings = mermaid_linter.validate_mermaid_block("dummy.md", 1, block)
    assert len(findings) == 0


def test_validate_unquoted_special_chars():
    # Invalid unquoted special characters in node label
    block = [
        (1, "flowchart LR"),
        (2, "    A[Label & Special] --> B"),
    ]
    findings = mermaid_linter.validate_mermaid_block("dummy.md", 1, block)
    assert len(findings) == 1
    assert "Unquoted special character(s)" in findings[0]["message"]
    assert findings[0]["line"] == 2


def test_validate_mismatched_quotes():
    # Odd number of quotes on a line
    block = [
        (1, "flowchart LR"),
        (2, '    A["Label --> B'),
    ]
    findings = mermaid_linter.validate_mermaid_block("dummy.md", 1, block)
    assert len(findings) == 1
    assert "Mismatched double quotes" in findings[0]["message"]
    assert findings[0]["line"] == 2


def test_validate_sequence_diagram_unclosed_block():
    # Sequence diagram with missing end block
    block = [
        (1, "sequenceDiagram"),
        (2, "    alt Is Happy"),
        (3, "        Alice->>Bob: Hello"),
    ]
    findings = mermaid_linter.validate_mermaid_block("dummy.md", 1, block)
    assert len(findings) == 1
    assert "Unclosed sequence diagram block" in findings[0]["message"]
    assert findings[0]["line"] == 2
