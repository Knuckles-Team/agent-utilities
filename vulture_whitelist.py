# Vulture whitelist
def _vulture_whitelist():
    # asynccontextmanager requires yield even after raise
    yield  # noqa
