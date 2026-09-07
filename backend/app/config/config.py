"""Single source of runtime configuration for the backend.

Everything the app reads from the environment lives here, on one pydantic
``Settings`` object. Paths are anchored to this file's location so the app does
not depend on the directory uvicorn is started from.

`app.config.settings` is a thin backward-compatibility shim that re-exports the
module-level names below (`UPLOAD_DIR`, `TEMPLATE_DIR`, `DEFAULT_MAPPING`,
`configure_logging`, ...) for the modules that still import them that way.

Environments
------------
`ENV` selects which deployment this process is: `local` (default), `test`, or
`production`. It drives defaults for things that must differ between the Test
and Production portals -- see docs/PROMOTION.md. Point `ENV_FILE` at the env
file for the environment (`.env`, `.env.test`, `.env.production`); it defaults
to `backend/.env`.
"""

from __future__ import annotations

import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# --- Paths -----------------------------------------------------------------
_HERE = Path(__file__).resolve()
APP_DIR = _HERE.parents[1]          # backend/app
BACKEND_DIR = _HERE.parents[2]      # backend
PROJECT_ROOT = _HERE.parents[3]     # repository root
FRONTEND_DIR = PROJECT_ROOT / "frontend"

# Which env file to load. Overridable so a Test / Production deploy can point at
# its own file without renaming: `ENV_FILE=.env.production uvicorn main:app`.
_ENV_FILE = os.environ.get("ENV_FILE", str(BACKEND_DIR / ".env"))

VALID_ENVS = ("local", "test", "production")


class Settings(BaseSettings):
    # -- Environment ----------------------------------------------------
    ENV: str = "local"                     # local | test | production

    # -- App ---------------------------------------------------------------
    APP_TITLE: str = "Vendor Form Extractor"
    DEBUG: bool = False

    # -- Security (auth flow) --------------------------------------------
    SECRET_KEY: str
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 15
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7

    # -- Database -------------------------------------------------------
    # Local dev default is SQLite. Test / Production set this to their own
    # Postgres instance -- the two must never share a database.
    DATABASE_URL: str = "sqlite:///./app.db"

    # -- CORS ---------------------------------------------------------
    # Comma-separated list of allowed browser origins for the portal frontend.
    # Empty in local dev (the permissive regex below covers localhost); Test /
    # Production must list their exact portal origin and nothing else.
    CORS_ALLOW_ORIGINS: str = ""

    # -- Business Central --------------------------------------------
    # BC_ENABLED gates the push feature entirely. The push is currently a
    # manual flow: the portal builds the OData payload, an operator POSTs it
    # from a machine on the VPN (see scripts/push_to_bc.ps1). The base URL /
    # company are still used to build the payload's target URL for that script
    # and for display.
    BC_ENABLED: bool = False
    BC_ODATA_BASE: str = "http://ntz-srv-bcdb:2248/BC220/ODataV4"
    BC_COMPANY: str = "Netsmartz Infotech (India) Pri"

    # Posting groups BC may require on a vendor/customer insert. Left blank by
    # default (an existing vendor in BC has Gen/VAT groups empty), sent only
    # when set. Gen/VAT posting groups are the same concept on both card
    # types; Customer/Vendor posting group are card-specific, hence separate.
    BC_GEN_BUS_POSTING_GROUP: str = ""
    BC_VAT_BUS_POSTING_GROUP: str = ""
    BC_VENDOR_POSTING_GROUP: str = ""
    BC_CUSTOMER_POSTING_GROUP: str = ""

    # -- Runtime data locations ----------------------------------------
    # Env overrides keep their historical VENDOR_* names.
    UPLOAD_DIR: Path = Field(
        default=APP_DIR / "uploads", validation_alias="VENDOR_UPLOAD_DIR"
    )
    OUTPUT_DIR: Path = Field(
        default=APP_DIR / "outputs", validation_alias="VENDOR_OUTPUT_DIR"
    )
    LOG_DIR: Path = APP_DIR / "logs"

    # -- OCR / extraction --------------------------------------------
    DEFAULT_OCR_MODELS: str = "small"
    DEFAULT_MAPPING: str = "vendor_creation_v1"
    # Render DPI for RapidOCR (the active default engine). The preserved
    # PaddleOCR fallback stays at extraction_pipeline.models.RENDER_DPI (200);
    # this is deliberately separate -- see ocr_engine / document_loader.
    RAPID_RENDER_DPI: int = Field(
        default=100, validation_alias="VENDOR_RAPID_RENDER_DPI"
    )

    # -- Logging ---------------------------------------------------
    LOG_LEVEL: str = Field(default="INFO", validation_alias="VENDOR_LOG_LEVEL")

    model_config = SettingsConfigDict(
        env_file=_ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @field_validator("ENV")
    @classmethod
    def _known_env(cls, v: str) -> str:
        v = (v or "local").strip().lower()
        if v not in VALID_ENVS:
            raise ValueError(f"ENV must be one of {VALID_ENVS}, got {v!r}")
        return v

    @property
    def is_production(self) -> bool:
        return self.ENV == "production"

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.CORS_ALLOW_ORIGINS.split(",") if o.strip()]


settings = Settings()

# A Production deploy must not run on the built-in SQLite default or a blank/
# obviously-placeholder secret -- fail fast rather than start up misconfigured.
if settings.is_production:
    problems = []
    if settings.DATABASE_URL.startswith("sqlite"):
        problems.append("DATABASE_URL is still the SQLite default")
    if len(settings.SECRET_KEY) < 32:
        problems.append("SECRET_KEY is too short (need >= 32 chars)")
    if not settings.cors_origin_list:
        problems.append("CORS_ALLOW_ORIGINS is empty")
    if problems:
        raise RuntimeError(
            "Production configuration is incomplete: " + "; ".join(problems)
        )

# --- Static / template dirs (not env-configurable) -----------------------
STATIC_DIR = FRONTEND_DIR / "static"
TEMPLATE_DIR = FRONTEND_DIR / "templates"

# --- Module-level aliases for the compatibility shim --------------------
ENV = settings.ENV
APP_TITLE = settings.APP_TITLE
DEBUG = settings.DEBUG
UPLOAD_DIR = settings.UPLOAD_DIR
OUTPUT_DIR = settings.OUTPUT_DIR
LOG_DIR = settings.LOG_DIR
DEFAULT_OCR_MODELS = settings.DEFAULT_OCR_MODELS
DEFAULT_MAPPING = settings.DEFAULT_MAPPING
RAPID_RENDER_DPI = settings.RAPID_RENDER_DPI
LOG_LEVEL = settings.LOG_LEVEL.upper()


_LOG_FORMAT = "%(asctime)s %(levelname)-8s [%(name)s] %(message)s"
_logging_configured = False


def configure_logging() -> None:
    """Configure root logging once: console + a rotating file in LOG_DIR.

    Every module uses ``logging.getLogger(__name__)`` and inherits this, so
    there is a single place that decides where log lines go. The log file is
    per-environment (``app.log`` / ``app.test.log`` / ``app.production.log``)
    so a shared host never interleaves environments in one file.
    """
    global _logging_configured
    if _logging_configured:
        return

    level = getattr(logging, LOG_LEVEL, logging.INFO)
    root = logging.getLogger()
    root.setLevel(level)

    formatter = logging.Formatter(_LOG_FORMAT)

    console = logging.StreamHandler()
    console.setLevel(level)
    console.setFormatter(formatter)
    root.addHandler(console)

    settings.LOG_DIR.mkdir(parents=True, exist_ok=True)
    suffix = "" if ENV == "local" else f".{ENV}"
    file_handler = RotatingFileHandler(
        filename=settings.LOG_DIR / f"app{suffix}.log",
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    _logging_configured = True
    logging.getLogger(__name__).info("logging configured (ENV=%s, level=%s)", ENV, LOG_LEVEL)


def ensure_directories() -> None:
    settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    settings.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    settings.LOG_DIR.mkdir(parents=True, exist_ok=True)
