"""Seed the portal's login accounts.

Normal (email + password) authentication, admin-provisioned -- there is no
public sign-up. Run once per environment after the DB is created:

    python -m app.cli.seed_users

Idempotent: an email that already exists is skipped, not updated. Override the
default password list with SEED_USERS in the environment, formatted as
`email:password` pairs separated by commas:

    SEED_USERS="a@x.com:Pass1,b@x.com:Pass2" python -m app.cli.seed_users
"""

from __future__ import annotations

import os
import sys

# Accounts to create when SEED_USERS is not set. Passwords are placeholders --
# change them (or pass SEED_USERS) before using this anywhere real.
DEFAULT_USERS = [
    ("admin@netsmartz.com", "Admin@123"),
    ("agamjot@netsmartz.com", "Agam@2024"),
    ("you@netsmartz.com", "password123"),
]


def _parse_env(raw: str) -> list[tuple[str, str]]:
    pairs = []
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if ":" not in chunk:
            raise SystemExit(f"SEED_USERS entry {chunk!r} is not email:password")
        email, _, pw = chunk.partition(":")
        pairs.append((email.strip(), pw))
    return pairs


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass

    from app.database.db import Base, SessionLocal, engine
    import app.models.model  # noqa: F401  -- register tables
    from app.services.auth_services.crud import create_user, get_user_by_email

    Base.metadata.create_all(bind=engine)

    raw = os.environ.get("SEED_USERS", "").strip()
    users = _parse_env(raw) if raw else DEFAULT_USERS

    db = SessionLocal()
    created, skipped = 0, 0
    try:
        for email, password in users:
            if get_user_by_email(db, email):
                print(f"  skip    {email} (already exists)")
                skipped += 1
                continue
            user = create_user(db, email, password)
            print(f"  created {email} (id={user.id})")
            created += 1
    finally:
        db.close()

    print(f"\n{created} created, {skipped} skipped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
