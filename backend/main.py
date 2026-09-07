"""FastAPI application factory.

Run from this directory with:
    uvicorn main:app --reload

The `app` name below is rebound from the package to the FastAPI instance once
the imports above have resolved, so import names out of `app.*` rather than
importing the package itself.
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.config.settings import (
    APP_TITLE,
    ENV,
    STATIC_DIR,
    configure_logging,
    ensure_directories,
    settings,
)
from app.routers.extraction import router
from app.routers.onboarding import router as onboarding_router
from app.database.db import get_db, Base, engine
from app.routers.auth import router as auth_router
from app.routers.vendors import router as vendors_router
from app.routers.customers import router as customers_router
from app.routers.business_central import router as bc_router

# Import the models module so every table is registered on Base.metadata
# (the routers above import it transitively, but do not rely on that ordering).
import app.models.model  # noqa: F401

# Schema is managed by Alembic (`alembic upgrade head`). For local dev only,
# ENV=local still auto-creates missing tables so `uvicorn main:app` works
# without running migrations first. Test / Production must run migrations.
if ENV == "local":
    Base.metadata.create_all(bind=engine)

def create_app() -> FastAPI:
    configure_logging()
    ensure_directories()

    app = FastAPI(title=APP_TITLE)

    # CORS. Local dev is permissive (any origin -- the React dev server's port
    # varies and tunnels change hostnames). Test / Production allow ONLY the
    # exact portal origins listed in CORS_ALLOW_ORIGINS, so the Production API
    # cannot be called from the Test portal or vice versa.
    if ENV == "local":
        cors_kwargs = dict(allow_origins=[], allow_origin_regex=r"https?://.*")
    else:
        origins = settings.cors_origin_list
        if not origins:
            raise RuntimeError(
                f"ENV={ENV} requires CORS_ALLOW_ORIGINS to list the portal origin(s)."
            )
        cors_kwargs = dict(allow_origins=origins)

    app.add_middleware(
        CORSMiddleware,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        **cors_kwargs,
    )

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    app.include_router(router)
    app.include_router(onboarding_router)
    app.include_router(auth_router)
    app.include_router(vendors_router)
    app.include_router(customers_router)
    app.include_router(bc_router)
    return app


app = create_app()
