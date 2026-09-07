"""Vendor persistence endpoints.

Saves a vendor record after it has been reviewed on the frontend. Both the
GSTIN and the Udyam (MSME) number are unique across vendors: a create or
update that repeats either one already stored returns 409 with the id of the
existing record. `GET /vendors?gst_no=...` / `?udyam_no=...` let the review
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
from app.schemas.vendor_schema import VendorCreate, VendorOut, VendorUpdate
from app.services import records_crud as crud
from app.services.auth_services.dependencies import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/vendors", tags=["vendors"])

# Both identifiers are unique across vendors; a clash on either returns 409.
_DUP_LABEL = {"gst_no": "GSTIN", "udyam_no": "Udyam number"}


def _conflict(kind: str, value: str | None, existing_id: int | None) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_409_CONFLICT,
        detail={
            "message": f"A vendor with this {_DUP_LABEL.get(kind, kind)} already exists.",
            kind: value,
            "existing_vendor_id": existing_id,
        },
    )


@router.post("", response_model=VendorOut, status_code=status.HTTP_201_CREATED)
def create_vendor(
    payload: VendorCreate,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    dup = crud.find_vendor_duplicate(db, payload.gst_no, payload.udyam_no)
    if dup:
        kind, existing = dup
        raise _conflict(kind, getattr(payload, kind), existing.id)

    try:
        vendor = crud.create_vendor(db, payload.model_dump(), user_id=user.id)
    except IntegrityError:
        # Lost a race between the check above and the insert.
        db.rollback()
        dup = crud.find_vendor_duplicate(db, payload.gst_no, payload.udyam_no)
        if dup:
            kind, existing = dup
            raise _conflict(kind, getattr(payload, kind), existing.id)
        raise _conflict("gst_no", payload.gst_no, None)

    logger.info(
        "vendor created id=%s gst=%s udyam=%s by user=%s",
        vendor.id, vendor.gst_no, vendor.udyam_no, user.id,
    )
    return vendor


@router.get("", response_model=list[VendorOut])
def list_vendors(
    gst_no: str | None = Query(default=None),
    udyam_no: str | None = Query(default=None),
    limit: int = Query(default=100, le=500),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    if gst_no:
        match = crud.get_vendor_by_gst(db, gst_no)
        return [match] if match else []
    if udyam_no:
        match = crud.get_vendor_by_udyam(db, udyam_no)
        return [match] if match else []
    return crud.list_vendors(db, limit=limit, offset=offset)


@router.get("/{vendor_id}", response_model=VendorOut)
def get_vendor(
    vendor_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    vendor = crud.get_vendor(db, vendor_id)
    if vendor is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Vendor not found")
    return vendor


@router.patch("/{vendor_id}", response_model=VendorOut)
def update_vendor(
    vendor_id: int,
    payload: VendorUpdate,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    vendor = crud.get_vendor(db, vendor_id)
    if vendor is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Vendor not found")

    changes = payload.model_dump(exclude_unset=True)
    if not changes:
        return vendor

    # If the update sets a new GSTIN / Udyam number, make sure no *other*
    # vendor already holds it.
    new_gst = changes.get("gst_no") if changes.get("gst_no") != vendor.gst_no else None
    new_udyam = changes.get("udyam_no") if changes.get("udyam_no") != vendor.udyam_no else None
    dup = crud.find_vendor_duplicate(db, new_gst, new_udyam, exclude_id=vendor.id)
    if dup:
        kind, existing = dup
        raise _conflict(kind, changes.get(kind), existing.id)

    try:
        vendor = crud.update_vendor(db, vendor, changes)
    except IntegrityError:
        db.rollback()
        dup = crud.find_vendor_duplicate(db, new_gst, new_udyam, exclude_id=vendor.id)
        kind = dup[0] if dup else "gst_no"
        raise _conflict(kind, changes.get(kind), dup[1].id if dup else None)

    logger.info("vendor %s updated (%s) by user %s", vendor.id, ", ".join(changes), user.id)
    return vendor


@router.delete("/{vendor_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_vendor(
    vendor_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    vendor = crud.get_vendor(db, vendor_id)
    if vendor is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Vendor not found")
    crud.delete_vendor(db, vendor)
    logger.info("vendor %s deleted by user %s", vendor_id, user.id)
