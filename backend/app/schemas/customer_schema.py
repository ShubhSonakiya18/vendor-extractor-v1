"""Request / response models for the customer endpoints.

Field names match the customer creation form (17 fields) and the frontend's
CustomerReviewPage. `payment_terms`, `salesperson`, `region`,
`customer_agreement` and `type` are never extracted from a document -- they are
entered on the review screen.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_serializer, model_validator


class CustomerBase(BaseModel):
    company_name: str = Field(..., min_length=1)
    contact_name: str = ""
    billing_address: str = ""
    city: str = ""
    state: str = ""
    zip_code: str = ""
    country: str = ""
    gst_registration_number: str = ""
    pan_number: str = ""
    email_id_to: str = ""
    email_id_cc: str = ""
    phone_number: str = ""
    payment_terms: str = ""
    salesperson: str = ""
    region: str = ""
    customer_agreement: str = ""
    type: str = "Services"

    @model_validator(mode="before")
    @classmethod
    def _null_strings_to_blank(cls, data: Any) -> Any:
        # The frontend sends null for fields the extraction did not fill.
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


class CustomerCreate(CustomerBase):
    raw_extraction: dict[str, Any] | None = None
    source_documents: list[dict[str, Any]] | None = None
    fields_needing_review: list[str] | None = None

    @model_validator(mode="before")
    @classmethod
    def _normalize_review_list(cls, data: Any) -> Any:
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


class CustomerUpdate(BaseModel):
    """Partial update -- every field optional; only what is sent is changed."""
    model_config = ConfigDict(extra="ignore")

    company_name: str | None = Field(default=None, min_length=1)
    contact_name: str | None = None
    billing_address: str | None = None
    city: str | None = None
    state: str | None = None
    zip_code: str | None = None
    country: str | None = None
    gst_registration_number: str | None = None
    pan_number: str | None = None
    email_id_to: str | None = None
    email_id_cc: str | None = None
    phone_number: str | None = None
    payment_terms: str | None = None
    salesperson: str | None = None
    region: str | None = None
    customer_agreement: str | None = None
    type: str | None = None


class CustomerOut(CustomerBase):
    id: int
    gst_registration_number: str | None = ""
    bc_status: str
    bc_no: str | None = None
    bc_synced_at: datetime | None = None
    fields_needing_review: list[str] | None = None
    created_by_user_id: int | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None

    model_config = ConfigDict(from_attributes=True)

    @field_serializer("gst_registration_number")
    def _gst_none_to_blank(self, v: str | None) -> str:
        return v or ""
