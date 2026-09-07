"""Alembic environment.

The database URL and model metadata come from the app's own config, so
migrations always run against whatever `ENV_FILE` / `DATABASE_URL` the current
environment is set to -- there is no separate connection string to keep in
sync. Run from the backend/ directory:

    alembic upgrade head                    # apply migrations (uses .env)
    ENV_FILE=.env.test alembic upgrade head  # ... against Test
    alembic revision --autogenerate -m "add X"
"""

from logging.config import fileConfig

from sqlalchemy import engine_from_config, pool

from alembic import context

from app.config.config import settings
from app.database.db import Base
import app.models.model  # noqa: F401  -- registers every table on Base.metadata

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Feed the app's live DB URL to Alembic (overrides the placeholder in
# alembic.ini). SQLAlchemy URLs can contain '%', which ConfigParser would try
# to interpolate, so set it on the config object rather than via the .ini.
config.set_main_option("sqlalchemy.url", settings.DATABASE_URL)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    context.configure(
        url=settings.DATABASE_URL,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        is_sqlite = connection.dialect.name == "sqlite"
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
            # SQLite can't ALTER most things; batch mode rebuilds the table.
            # Harmless on Postgres, essential for local SQLite dev.
            render_as_batch=is_sqlite,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
