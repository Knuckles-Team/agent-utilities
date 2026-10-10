"""Public persistence-privacy reference for AU application integrations.

Exposes the AU-owned keyed persistence reference to application consumers
without making them import the internal security package.
"""

from agent_utilities.security.persistence_privacy import persistence_reference

__all__ = ["persistence_reference"]
