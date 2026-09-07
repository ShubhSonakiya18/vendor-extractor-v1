"""Shared fixtures.

`client` gives each test a FastAPI TestClient backed by a fresh in-memory
SQLite database, with `get_db` overridden and a registered+authenticated user
so the auth-gated endpoints are reachable.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool


@pytest.fixture
def db_session_factory():
    # One shared connection (StaticPool) so every session in a test sees the
    # same in-memory database.
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    from app.database.db import Base
    import app.models.model  # noqa: F401  -- register tables

    Base.metadata.create_all(bind=engine)
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    yield TestingSessionLocal
    Base.metadata.drop_all(bind=engine)
    engine.dispose()


@pytest.fixture
def client(db_session_factory):
    import main
    from app.database.db import get_db

    def _override_get_db():
        db = db_session_factory()
        try:
            yield db
        finally:
            db.close()

    main.app.dependency_overrides[get_db] = _override_get_db
    with TestClient(main.app) as c:
        yield c
    main.app.dependency_overrides.clear()


@pytest.fixture
def auth_client(client):
    """A `client` with an Authorization header for a freshly registered user."""
    client.post("/auth/register", json={"email": "tester@example.com", "password": "pw123456"})
    r = client.post("/auth/login", json={"email": "tester@example.com", "password": "pw123456"})
    token = r.json()["access_token"]
    client.headers.update({"Authorization": f"Bearer {token}"})
    return client
