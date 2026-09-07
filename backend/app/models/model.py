from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    ForeignKey,
    Integer,
    String,
)
from sqlalchemy.sql import func

from app.database.db import Base


class TimestampMixin:
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class RefreshToken(Base):
    __tablename__ = "refresh_tokens"

    id = Column(Integer, primary_key=True, index=True)
    token = Column(String, unique=True, index=True, nullable=False)
    user_id = Column(Integer, nullable=False)
    expires_at = Column(DateTime(timezone=True), nullable=False)
    revoked = Column(Boolean, default=False)


class BusinessCentralSyncMixin:
    """Columns tracking whether this record has been pushed to Business Central.

    Unused until the BC push feature lands (see the integration plan, step 6) --
    defined now so adding it later is not a schema migration.
    """

    bc_status = Column(String, default="not_pushed", nullable=False)  # not_pushed | pushed | failed
    bc_no = Column(String)          # the No. BC assigns to the created card
    bc_synced_at = Column(DateTime(timezone=True))
    bc_error = Column(String)       # last push failure message, if any


class Vendor(Base, TimestampMixin, BusinessCentralSyncMixin):
    """A vendor as reviewed on the frontend before it is sent to Business Central.

    Fields mirror the Vendor Creation Request Form (supplier-filled block,
    rows 37-63). Values are whatever the user confirmed on the review screen --
    OCR-extracted, corrected, or hand-entered for the fields no document
    supplies (website, TAN, ESIC).
    """

    __tablename__ = "vendors"

    id = Column(Integer, primary_key=True, index=True)

    vendor_name = Column(String, nullable=False, index=True)
    address_1 = Column(String)
    address_2 = Column(String)
    address_3 = Column(String)
    address_4 = Column(String)
    city = Column(String)
    state = Column(String)
    country = Column(String)
    pin_code = Column(String)
    telephone_1 = Column(String)
    telephone_2 = Column(String)
    email = Column(String)
    website = Column(String)
    company_type = Column(String)          # Company / Non-Company
    nature_of_business = Column(String)
    tan_no = Column(String)
    pan = Column(String, index=True)
    tds_applicable = Column(String)        # Yes / No
    # GSTIN and Udyam number are both unique across vendors -- the same
    # registration / MSME number cannot be onboarded twice. NULL is allowed and
    # not covered by the constraint (not every vendor has both at creation).
    gst_no = Column(String, unique=True, index=True)
    esic_no = Column(String)
    udyam_no = Column(String, unique=True, index=True)
    bank_name = Column(String)
    branch_address = Column(String)
    ifsc_swift_code = Column(String)
    account_type = Column(String)          # CA / CC / SB
    account_number = Column(String)

    # Full extraction response as received, so nothing the pipeline produced is
    # lost even though only the fields above are promoted to columns.
    raw_extraction = Column(JSON)
    source_documents = Column(JSON)        # [{file_name, document_type, confidence}]
    fields_needing_review = Column(JSON)   # list[str] carried from extraction

    created_by_user_id = Column(Integer, ForeignKey("users.id"), index=True)


class Customer(Base, TimestampMixin, BusinessCentralSyncMixin):
    """A customer as reviewed on the frontend before it is sent to Business
    Central. Fields mirror the customer creation form (17 fields).

    The five business fields -- payment_terms, salesperson, region,
    customer_agreement, type -- are never in an uploaded document; they are
    entered by the user on the review screen and stored as-is.
    """

    __tablename__ = "customers"

    id = Column(Integer, primary_key=True, index=True)

    company_name = Column(String, nullable=False, index=True)
    contact_name = Column(String)
    billing_address = Column(String)
    city = Column(String)
    state = Column(String)
    zip_code = Column(String)
    country = Column(String)
    gst_registration_number = Column(String, unique=True, index=True)
    pan_number = Column(String, index=True)
    email_id_to = Column(String)
    email_id_cc = Column(String)
    phone_number = Column(String)
    payment_terms = Column(String)
    salesperson = Column(String)
    region = Column(String)
    customer_agreement = Column(String)
    type = Column(String, default="Services")   # Services / License

    raw_extraction = Column(JSON)
    source_documents = Column(JSON)
    fields_needing_review = Column(JSON)

    created_by_user_id = Column(Integer, ForeignKey("users.id"), index=True)
