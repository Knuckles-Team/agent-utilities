"""Long-context benchmark tasks (CONCEPT:AHE-3.32).

Importing this package registers every task into the ``base`` registry, so
``base.get_task`` / ``base.list_tasks`` see them.
"""

from . import browsecomp_plus, longbench_codeqa, oolong, oolong_pairs, s_niah

__all__ = ["s_niah", "oolong", "oolong_pairs", "browsecomp_plus", "longbench_codeqa"]
