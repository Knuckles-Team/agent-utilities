import logging
import shutil
import time
from pathlib import Path

logger = logging.getLogger(__name__)


def backup_file_pre_edit(filepath: str | Path) -> str | None:
    """
    Creates a pre-edit safety backup of the specified file in the XDG standard
    agent-utilities backups directory before any AI agent makes a destructive edit.

    CONCEPT:AU-OS.safety.tool-agnostic-file-safety: Tool-Agnostic File Safety Hooks
    """
    filepath = Path(filepath)
    if not filepath.exists() or not filepath.is_file():
        return None

    backup_dir = Path.home() / ".local" / "share" / "agent-utilities" / "backups"
    backup_dir.mkdir(parents=True, exist_ok=True)

    timestamp = int(time.time())
    backup_filename = f"{timestamp}_{filepath.name}.bak"
    backup_path = backup_dir / backup_filename

    try:
        shutil.copy2(filepath, backup_path)
        return str(backup_path)
    except Exception as e:
        logger.warning(f"Failed to create pre-edit backup for {filepath}: {e}")
        return None
