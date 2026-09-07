"""Request / response models for the vendor endpoints.

Field names match the Vendor Creation Request Form and the frontend's vendor
review screen. Every field except `vendor_name` is optional -- a reviewer can
submit a partially complete record.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_serializer, model_validator


class VendorBase(BaseModel):
    vendor_name: str = Field(..., min_length=1)
    address_1: str = ""
    address_2: str = ""
    address_3: str = ""
    address_4: str = ""
    city: str = ""
    state: str = ""
    country: str = ""
    pin_code: str = ""
    telephone_1: str = ""
    telephone_2: str = ""
    email: str = ""
    website: str = ""
    company_type: str = ""
    nature_of_business: str = ""
    tan_no: str = ""
    pan: str = ""
    tds_applicable: str = ""
    gst_no: str = ""
    esic_no: str = ""
    udyam_no: str = ""
    bank_name: str = ""
    branch_address: str = ""
    ifsc_swift_code: str = ""
    account_type: str = ""
    account_number: str = ""

    @model_validator(mode="before")
    @classmethod
    def _null_strings_to_blank(cls, data: Any) -> Any:
        # The frontend sends null for fields the extraction did not fill.
        # Treat null the same as omitted -> "".
        if isinstance(data, dict):
            str_fields = {
                name for name, f in cls.model_fields.items()
                if f.annotation is str
            }
            return {
                k: ("" if (k in str_fields and v is None) else v)
                for k, v in data.items()
            }
        return data


class VendorCreate(VendorBase):
    # Optional provenance carried from the extraction step, stored but not
    # required. `raw_extraction` is the untouched extract response.
    raw_extraction: dict[str, Any] | None = None
    source_documents: list[dict[str, Any]] | None = None
    fields_needing_review: list[str] | None = None

    @model_validator(mode="before")
    @classmethod
    def _normalize_review_list(cls, data: Any) -> Any:
        # The extraction pipeline's needs_review is a list of {field, reason}
        # objects; accept that and keep just the field names.
        if isinstance(data, dict):
            nr = data.get("fields_needing_review")
            if isinstance(nr, list):
                data = {
                    **data,
                    "fields_needing_review": [
                        x["field"] if isinstance(x, dict) and "field" in x else x
                        for x in nr
                    ],
                }
        return data


class VendorUpdate(BaseModel):
    """Partial update -- every field optional; only what is sent is changed."""
    model_config = ConfigDict(extra="ignore")

    vendor_name: str | None = Field(default=None, min_length=1)
    address_1: str | None = None
    address_2: str | None = None
    address_3: str | None = None
    address_4: str | None = None
    city: str | None = None
    state: str | None = None
    country: str | None = None
    pin_code: str | None = None
    telephone_1: str | None = None
    telephone_2: str | None = None
    email: str | None = None
    website: str | None = None
    company_type: str | None = None
    nature_of_business: str | None = None
    tan_no: str | None = None
    pan: str | None = None
    tds_applicable: str | None = None
    gst_no: str | None = None
    esic_no: str | None = None
    udyam_no: str | None = None
    bank_name: str | None = None
    branch_address: str | None = None
    ifsc_swift_code: str | None = None
    account_type: str | None = None
    account_number: str | None = None


class VendorOut(VendorBase):
    id: int
    # NULL in the DB (identifier not supplied) is surfaced as "" for the
    # frontend, which treats every field as a string.
    gst_no: str | None = ""
    udyam_no: str | None = ""
    bc_status: str
    bc_no: str | None = None
    bc_synced_at: datetime | None = None
    fields_needing_review: list[str] | None = None
    created_by_user_id: int | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None

    model_config = ConfigDict(from_attributes=True)

    @field_serializer("gst_no", "udyam_no")
    def _none_to_blank(self, v: str | None) -> str:
        return v or ""
