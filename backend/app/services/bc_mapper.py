"""Map a saved Vendor or Customer row to a Business Central OData payload
(VendorCard / CustomerCard respectively).

Field names are BC's own (from GET .../VendorCard or .../CustomerCard on the
BC220 server), which differ from the portal's field names and the Excel/
onboarding form labels. `No` is sent empty -- BC assigns it from its own No.
Series.

Only fields the portal actually has are included. Blank values are omitted
rather than sent as "", so BC keeps its own defaults. Posting groups come from
config and are sent only when configured.

TODO (needs confirmation against the live BC company before enabling POST):
  - Is `Vendor_Posting_Group` / `Customer_Posting_Group` mandatory on insert?
    An existing vendor had it populated ("EMPLOAN") while Gen/VAT groups were
    empty -- the customer equivalent has never been checked against a real
    BC220 CustomerCard at all, since (unlike the vendor payload) nothing has
    exercised this one against a live company yet.
  - `nature_of_business` has no obvious VendorCard field -- dropped for now.
  - Bank details (bank_name / ifsc / account_number) are not on VendorCard;
    they belong to a separate VendorBankAccount entity, handled later.
  - Customer's salesperson/payment_terms are BC "Code" fields needing real
    master-data codes, not free text -- see _CUSTOMER_FIELD_MAP below for why
    they're deliberately left unmapped.
"""

from __future__ import annotations

from app.config.config import settings
from app.models.model import Customer, Vendor

# portal Vendor attribute  ->  BC VendorCard field
_FIELD_MAP: dict[str, str] = {
    "vendor_name": "Name",
    "address_1": "Address",
    "address_2": "Address_2",
    "city": "City",
    "state": "County",
    "country": "Country_Region_Code",
    "pin_code": "Post_Code",
    "telephone_1": "Phone_No",
    "telephone_2": "MobilePhoneNo",
    "email": "E_Mail",
    "website": "Home_Page",
    "pan": "PAN_Number",
    "gst_no": "GST_Number",
}


def vendor_to_bc_payload(vendor: Vendor) -> dict:
    """Build the JSON body for a POST to .../VendorCard."""
    payload: dict[str, str] = {"No": ""}

    for attr, bc_field in _FIELD_MAP.items():
        value = getattr(vendor, attr, None)
        if value:
            payload[bc_field] = str(value).strip()

    if settings.BC_GEN_BUS_POSTING_GROUP:
        payload["Gen_Bus_Posting_Group"] = settings.BC_GEN_BUS_POSTING_GROUP
    if settings.BC_VAT_BUS_POSTING_GROUP:
        payload["VAT_Bus_Posting_Group"] = settings.BC_VAT_BUS_POSTING_GROUP
    if settings.BC_VENDOR_POSTING_GROUP:
        payload["Vendor_Posting_Group"] = settings.BC_VENDOR_POSTING_GROUP

    return payload


def vendor_card_url() -> str:
    """The OData URL an operator POSTs the payload to (used by push_to_bc.ps1
    and shown in the UI)."""
    company = settings.BC_COMPANY.replace("'", "''")
    return f"{settings.BC_ODATA_BASE}/Company('{company}')/VendorCard"


# portal Customer attribute -> BC CustomerCard field. Same TODO as
# _FIELD_MAP above: field names are standard BC Customer-table captions, not
# yet confirmed against the live BC220 CustomerCard page.
#
# Deliberately NOT mapped here, same reasoning as vendor's dropped
# nature_of_business:
#   - salesperson, payment_terms map to BC "Code" fields (Salesperson_Code,
#     Payment_Terms_Code) that must match existing BC master-data codes --
#     sending the portal's free-text value would likely fail validation
#     rather than silently do the wrong thing, so it's left out until there's
#     a real code list to map against.
#   - region, customer_agreement, type have no obvious CustomerCard field.
_CUSTOMER_FIELD_MAP: dict[str, str] = {
    "company_name": "Name",
    "contact_name": "Contact",
    "billing_address": "Address",
    "city": "City",
    "state": "County",
    "country": "Country_Region_Code",
    "zip_code": "Post_Code",
    "phone_number": "Phone_No",
    "email_id_to": "E_Mail",
    "pan_number": "PAN_Number",
    "gst_registration_number": "GST_Number",
}


def customer_to_bc_payload(customer: Customer) -> dict:
    """Build the JSON body for a POST to .../CustomerCard. Mirrors
    vendor_to_bc_payload: blank values omitted rather than sent as "", `No`
    sent empty so BC assigns it from its own No. Series."""
    payload: dict[str, str] = {"No": ""}

    for attr, bc_field in _CUSTOMER_FIELD_MAP.items():
        value = getattr(customer, attr, None)
        if value:
            payload[bc_field] = str(value).strip()

    if settings.BC_GEN_BUS_POSTING_GROUP:
        payload["Gen_Bus_Posting_Group"] = settings.BC_GEN_BUS_POSTING_GROUP
    if settings.BC_VAT_BUS_POSTING_GROUP:
        payload["VAT_Bus_Posting_Group"] = settings.BC_VAT_BUS_POSTING_GROUP
    if settings.BC_CUSTOMER_POSTING_GROUP:
        payload["Customer_Posting_Group"] = settings.BC_CUSTOMER_POSTING_GROUP

    return payload


def customer_card_url() -> str:
    """The OData URL an operator POSTs the customer payload to."""
    company = settings.BC_COMPANY.replace("'", "''")
    return f"{settings.BC_ODATA_BASE}/Company('{company}')/CustomerCard"
