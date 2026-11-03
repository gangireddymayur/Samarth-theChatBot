import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

# Keep tests off the committed samarth.duckdb file, and make the relative
# dataset paths in backend/config.py resolve from the repo root.
os.environ["SAMARTH_DB"] = ":memory:"
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def client():
    from fastapi.testclient import TestClient

    from backend.main import app

    # Entering the client runs the startup event, which ingests both CSVs.
    with TestClient(app) as c:
        yield c


def ask(client, query):
    return client.post("/chat", json={"query": query})
