"""Typed control-plane domains.

Transport adapters and persistence implementations stay outside this package;
the control plane exposes small, policy-aware protocol seams instead.

Each domain (``foundation``, ``connectors``, ``agents``, ``policy``, ``runs``,
``sources``, ``retrieval``, ``economics``, ``webui``, ``workflows``,
``migrations``, ``projection``) is imported from its own submodule. This
package deliberately re-exports nothing: a star re-export here would make every
consumer's import surface depend on load order across twelve independently
owned domains, and would need blanket ``noqa`` suppressions to pass lint.
"""
