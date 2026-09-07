"""CRUD for Vendor and Customer records.

Both share the same shape: create from a reviewed payload, list, get by id,
and a GSTIN lookup used both to warn the frontend before submit and to turn a
duplicate insert into a clean 409.
"""

from __future__ import annotations

from sqlalchemy.orm import Session

from app.models.model import Customer, Vendor

# Keys on the *Create schema that are stored in dedicated columns; the rest
# (raw_extraction, source_documents, fields_needing_review) are passed through.
_PASSTHROUGH = {"raw_extraction", "source_documents", "fields_needing_review"}

# Blank values in these identity columns must land as NULL, not "", so the
# UNIQUE constraints permit many records without that identifier (SQL UNIQUE
# ignores NULLs; it does not ignore "").
_NULLABLE_UNIQUE_COLUMNS = {"gst_no", "gst_registration_number", "udyam_no"}


def _split_payload(data: dict) -> tuple[dict, dict]:
    columns = {k: v for k, v in data.items() if k not in _PASSTHROUGH}
    for key in _NULLABLE_UNIQUE_COLUMNS:
        if key in columns and not (columns[key] or "").strip():
            columns[key] = None
    extras = {k: data[k] for k in _PASSTHROUGH if k in data}
    return columns, extras


# --- Vendor -------------------------------------------------------------

def get_vendor(db: Session, vendor_id: int) -> Vendor | None:
    return db.get(Vendor, vendor_id)


def get_vendor_by_gst(db: Session, gst_no: str) -> Vendor | None:
    if not gst_no:
        return None
    return db.query(Vendor).filter(Vendor.gst_no == gst_no).first()


def get_vendor_by_udyam(db: Session, udyam_no: str) -> Vendor | None:
    if not udyam_no:
        return None
    return db.query(Vendor).filter(Vendor.udyam_no == udyam_no).first()


def find_vendor_duplicate(
    db: Session, gst_no: str | None, udyam_no: str | None, exclude_id: int | None = None
) -> tuple[str, Vendor] | None:
    """Return ("gst_no" | "udyam_no", existing_vendor) if either identifier is
    already used by another vendor, else None. GSTIN is checked first."""
    for kind, value, getter in (
        ("gst_no", gst_no, get_vendor_by_gst),
        ("udyam_no", udyam_no, get_vendor_by_udyam),
    ):
        if value and value.strip():
            match = getter(db, value.strip())
            if match and match.id != exclude_id:
                return kind, match
    return None


def list_vendors(db: Session, limit: int = 100, offset: int = 0) -> list[Vendor]:
    # id is the tie-break so rows created within the same clock second still
    # come back newest-first in a stable order.
    return (
        db.query(Vendor)
        .order_by(Vendor.created_at.desc(), Vendor.id.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )


def create_vendor(db: Session, data: dict, user_id: int | None) -> Vendor:
    columns, extras = _split_payload(data)
    vendor = Vendor(**columns, **extras, created_by_user_id=user_id)
    db.add(vendor)
    db.commit()
    db.refresh(vendor)
    return vendor


def update_vendor(db: Session, vendor: Vendor, data: dict) -> Vendor:
    """Apply a partial update. Only the keys present in `data` are changed;
    provenance/BC columns are not touched here."""
    columns, _ = _split_payload(data)
    for key, value in columns.items():
        setattr(vendor, key, value)
    db.commit()
    db.refresh(vendor)
    return vendor


def delete_vendor(db: Session, vendor: Vendor) -> None:
    db.delete(vendor)
    db.commit()


# --- Customer -----------------------------------------------------------

def get_customer(db: Session, customer_id: int) -> Customer | None:
    return db.get(Customer, customer_id)


def get_customer_by_gst(db: Session, gst: str) -> Customer | None:
    if not gst:
        return None
    return (
        db.query(Customer)
        .filter(Customer.gst_registration_number == gst)
        .first()
    )


def list_customers(db: Session, limit: int = 100, offset: int = 0) -> list[Customer]:
    return (
        db.query(Customer)
        .order_by(Customer.created_at.desc(), Customer.id.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )


def create_customer(db: Session, data: dict, user_id: int | None) -> Customer:
    columns, extras = _split_payload(data)
    customer = Customer(**columns, **extras, created_by_user_id=user_id)
    db.add(customer)
    db.commit()
    db.refresh(customer)
    return customer


def update_customer(db: Session, customer: Customer, data: dict) -> Customer:
    columns, _ = _split_payload(data)
    for key, value in columns.items():
        setattr(customer, key, value)
    db.commit()
    db.refresh(customer)
    return customer


def delete_customer(db: Session, customer: Customer) -> None:
    db.delete(customer)
    db.commit()
