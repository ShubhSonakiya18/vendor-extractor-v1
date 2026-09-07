"""Password hashing and JWT helpers for the auth flow.

Passwords are hashed with bcrypt directly (the `bcrypt` package), not through
passlib -- passlib 1.7.4 is unmaintained and its bcrypt backend breaks against
bcrypt >= 4.1 (it reads a `__about__.__version__` attribute that no longer
exists). Calling bcrypt straight is exactly what passlib would have done
underneath, with one less dependency.
"""

import uuid
from datetime import datetime, timedelta, timezone

import bcrypt
from jose import jwt, JWTError

from app.config.config import settings

# bcrypt rejects inputs longer than 72 bytes. Long passphrases and multi-byte
# UTF-8 hit that ceiling, so pre-hash to a fixed-length digest first (a widely
# used pattern; the sha256 output is 64 hex chars, always < 72 bytes).
import hashlib


def _prehash(password: str) -> bytes:
    return hashlib.sha256(password.encode("utf-8")).hexdigest().encode("ascii")


def hash_password(password: str) -> str:
    return bcrypt.hashpw(_prehash(password), bcrypt.gensalt()).decode("ascii")


def verify_password(plain: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(_prehash(plain), hashed.encode("ascii"))
    except (ValueError, TypeError):
        # Malformed hash in the DB (e.g. a row from the old sha256 scheme).
        return False


def create_access_token(user_id: int) -> str:
    expire = datetime.now(timezone.utc) + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    payload = {"sub": str(user_id), "exp": expire, "type": "access"}
    return jwt.encode(payload, settings.SECRET_KEY, algorithm=settings.ALGORITHM)


def create_refresh_token(user_id: int) -> tuple[str, datetime]:
    expire = datetime.now(timezone.utc) + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)
    payload = {"sub": str(user_id), "exp": expire, "type": "refresh", "jti": str(uuid.uuid4())}
    token = jwt.encode(payload, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
    return token, expire


def decode_token(token: str) -> dict | None:
    try:
        return jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
    except JWTError:
        return None
