import os


def test_env_var():
    print(f"\nAGENT_UTILITIES_TESTING={os.environ.get('AGENT_UTILITIES_TESTING')}")
    assert os.environ.get('AGENT_UTILITIES_TESTING') == 'true'
