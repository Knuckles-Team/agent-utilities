"""``python -m agent_utilities``: print the harness version.

agent-utilities is the in-run agent harness. It serves nothing on its own;
arbora hosts it and serves its operations.
"""

from agent_utilities._version import __version__


def main() -> None:
    print(f"agent-utilities {__version__} (in-run harness; hosted by arbora)")


if __name__ == "__main__":
    main()
