"""Shared fixtures. The FakeLLM fixture and the Postgres fixture land with Phase 0 / Phase 1."""
import os
import pathlib

import pytest


@pytest.fixture(scope="session")
def database_url() -> str:
    """Connection string for the integration tests, read from the Docker secret or the environment."""
    secret = pathlib.Path(os.environ.get("DATABASE_URL_FILE", "/run/secrets/pg_url"))
    if secret.exists():
        return secret.read_text().strip()
    url = os.environ.get("DATABASE_URL")
    if not url:
        pytest.skip("no DATABASE_URL or DATABASE_URL_FILE available")
    return url
