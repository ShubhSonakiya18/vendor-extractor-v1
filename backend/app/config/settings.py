"""Backward-compatibility shim.

Configuration now lives in `app.config.config`. This module re-exports the
module-level names that existing imports (`from app.config.settings import
UPLOAD_DIR`, ...) still expect. New code should import from
`app.config.config` directly.
"""

from __future__ import annotations

from app.config.config import (  # noqa: F401
    APP_DIR,
    APP_TITLE,
    BACKEND_DIR,
    DEBUG,
    DEFAULT_MAPPING,
    DEFAULT_OCR_MODELS,
    ENV,
    FRONTEND_DIR,
    LOG_DIR,
    LOG_LEVEL,
    OUTPUT_DIR,
    PROJECT_ROOT,
    RAPID_RENDER_DPI,
    STATIC_DIR,
    TEMPLATE_DIR,
    UPLOAD_DIR,
    configure_logging,
    ensure_directories,
    settings,
)

__all__ = [
    "APP_DIR",
    "APP_TITLE",
    "BACKEND_DIR",
    "DEBUG",
    "DEFAULT_MAPPING",
    "DEFAULT_OCR_MODELS",
    "ENV",
    "FRONTEND_DIR",
    "LOG_DIR",
    "LOG_LEVEL",
    "OUTPUT_DIR",
    "PROJECT_ROOT",
    "RAPID_RENDER_DPI",
    "STATIC_DIR",
    "TEMPLATE_DIR",
    "UPLOAD_DIR",
    "configure_logging",
    "ensure_directories",
    "settings",
]
