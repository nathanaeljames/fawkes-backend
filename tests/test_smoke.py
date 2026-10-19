"""Smoke tests so the unit and integration jobs collect at least one test each (pytest exits 5 on an empty suite)."""
import pytest


def test_suite_collects() -> None:
    assert True


@pytest.mark.integration
def test_database_reachable(database_url: str) -> None:
    psycopg = pytest.importorskip("psycopg")
    with psycopg.connect(database_url) as conn:
        assert conn.execute("SELECT 1").fetchone() == (1,)
