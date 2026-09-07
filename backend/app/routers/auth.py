from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.database.db import get_db
from app.schemas.auth_schema import UserCreate, UserLogin, UserOut, TokenPair, RefreshRequest
from app.services.auth_services.security import verify_password, create_access_token, create_refresh_token, decode_token
from app.services.auth_services.crud import create_user, get_user_by_email, store_refresh_token, get_valid_refresh_token, revoke_refresh_token
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/register", response_model=UserOut, status_code=status.HTTP_201_CREATED)
def register(user_in: UserCreate, db: Session = Depends(get_db)):
    logger.info("Registration attempt: %s", user_in.email)

    try:
        if get_user_by_email(db, user_in.email):
            logger.warning("Registration failed - email already registered: %s", user_in.email)
            raise HTTPException(status_code=400, detail="Email already registered")

        user = create_user(db, user_in.email, user_in.password)
        logger.info("User registered successfully - user_id: %s, email: %s", user.id, user.email)

        return user

    except HTTPException:
        raise
    except Exception:
        logger.exception("Registration error: %s", user_in.email)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/login", response_model=TokenPair)
def login(credentials: UserLogin, db: Session = Depends(get_db)):
    logger.info("Login attempt: %s", credentials.email)

    try:
        user = get_user_by_email(db, credentials.email)

        if not user or not verify_password(credentials.password, user.hashed_password):
            logger.warning("Login failed - invalid credentials: %s", credentials.email)
            raise HTTPException(status_code=401, detail="Invalid email or password")

        access_token = create_access_token(user.id)
        refresh_token, expires_at = create_refresh_token(user.id)
        store_refresh_token(db, user.id, refresh_token, expires_at)

        logger.info("Login successful - user_id: %s, email: %s", user.id, user.email)

        return TokenPair(access_token=access_token, refresh_token=refresh_token)

    except HTTPException:
        raise
    except Exception:
        logger.exception("Login error: %s", credentials.email)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/refresh", response_model=TokenPair)
def refresh(payload: RefreshRequest, db: Session = Depends(get_db)):
    logger.info("Refresh token request received")

    try:
        token_data = decode_token(payload.refresh_token)

        if token_data is None or token_data.get("type") != "refresh":
            logger.warning("Invalid refresh token")
            raise HTTPException(status_code=401, detail="Invalid refresh token")

        db_token = get_valid_refresh_token(db, payload.refresh_token)

        if db_token is None:
            logger.warning("Refresh token expired or revoked")
            raise HTTPException(status_code=401, detail="Refresh token expired or revoked")

        user_id = int(token_data["sub"])

        revoke_refresh_token(db, payload.refresh_token)
        new_access = create_access_token(user_id)
        new_refresh, expires_at = create_refresh_token(user_id)
        store_refresh_token(db, user_id, new_refresh, expires_at)

        logger.info("Refresh token rotated successfully - user_id: %s", user_id)

        return TokenPair(access_token=new_access, refresh_token=new_refresh)

    except HTTPException:
        raise
    except Exception:
        logger.exception("Refresh token processing error")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/logout")
def logout(payload: RefreshRequest, db: Session = Depends(get_db)):
    logger.info("Logout request received")

    try:
        revoke_refresh_token(db, payload.refresh_token)
        logger.info("Logout successful - refresh token revoked")
        return {"message": "Logged out successfully"}

    except Exception:
        logger.exception("Logout error")
        raise HTTPException(status_code=500, detail="Internal server error")