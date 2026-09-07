"""Customer persistence endpoints.

Saves a customer record after it has been reviewed on the frontend. GSTIN is
unique across customers: a create that repeats one already stored returns 409
with the id of the existing record. `GET /customers?gst=...` lets the review
screen check for that before submitting.

Sending the record on to Business Central is a separate step (see the
integration plan) -- these endpoints only persist locally.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.database.db import get_db
from app.models.model import User
from app.schemas.customer_schema import CustomerCreate, CustomerOut, CustomerUpdate
from app.services import records_crud as crud
from app.services.auth_services.dependencies import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/customers", tags=["customers"])


@router.post("", response_model=CustomerOut, status_code=status.HTTP_201_CREATED)
def create_customer(
    payload: CustomerCreate,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    gst = payload.gst_registration_number
    if gst:
        existing = crud.get_customer_by_gst(db, gst)
        if existing:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail={
                    "message": "A customer with this GSTIN already exists.",
                    "gst_registration_number": gst,
                    "existing_customer_id": existing.id,
                },
            )

    try:
        customer = crud.create_customer(db, payload.model_dump(), user_id=user.id)
    except IntegrityError:
        db.rollback()
        existing = crud.get_customer_by_gst(db, gst)
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "message": "A customer with this GSTIN already exists.",
                "gst_registration_number": gst,
                "existing_customer_id": existing.id if existing else None,
            },
        )

    logger.info("customer created id=%s gst=%s by user=%s", customer.id, gst, user.id)
    return customer


@router.get("", response_model=list[CustomerOut])
def list_customers(
    gst: str | None = Query(default=None),
    limit: int = Query(default=100, le=500),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    if gst:
        match = crud.get_customer_by_gst(db, gst)
        return [match] if match else []
    return crud.list_customers(db, limit=limit, offset=offset)


@router.get("/{customer_id}", response_model=CustomerOut)
def get_customer(
    customer_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    customer = crud.get_customer(db, customer_id)
    if customer is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Customer not found")
    return customer


@router.patch("/{customer_id}", response_model=CustomerOut)
def update_customer(
    customer_id: int,
    payload: CustomerUpdate,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    customer = crud.get_customer(db, customer_id)
    if customer is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Customer not found")

    changes = payload.model_dump(exclude_unset=True)
    if not changes:
        return customer

    new_gst = changes.get("gst_registration_number")
    if new_gst and new_gst.strip() and new_gst != customer.gst_registration_number:
        clash = crud.get_customer_by_gst(db, new_gst)
        if clash and clash.id != customer.id:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail={
                    "message": "Another customer already has this GSTIN.",
                    "gst_registration_number": new_gst,
                    "existing_customer_id": clash.id,
                },
            )

    try:
        customer = crud.update_customer(db, customer, changes)
    except IntegrityError:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"message": "Another customer already has this GSTIN."},
        )

    logger.info("customer %s updated (%s) by user %s", customer.id, ", ".join(changes), user.id)
    return customer


@router.delete("/{customer_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_customer(
    customer_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    customer = crud.get_customer(db, customer_id)
    if customer is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Customer not found")
    crud.delete_customer(db, customer)
    logger.info("customer %s deleted by user %s", customer_id, user.id)
