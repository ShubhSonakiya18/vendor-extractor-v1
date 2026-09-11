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

# Standard Business Central field widths (base VendorCard/CustomerCard --
# Vendor and Customer tables both use these same widths for the same-named
# fields). Fixed inside BC's own schema: the OData endpoint rejects anything
# longer with Application_StringExceededLength, and this cannot be changed
# from the portal side -- only a BC AL extension widening the base table
# column could raise it. A field with no entry here is assumed unbounded
# (safe default; BC will reject it just the same if that assumption is wrong,
# and the failure will now show the real reason -- see push_to_bc.ps1).
_BC_FIELD_MAX_LEN: dict[str, int] = {
    "Name": 50,
    "Address": 50,
    "Address_2": 50,
    "City": 30,
    "County": 30,
    "Contact": 50,
    "Phone_No": 30,
    "MobilePhoneNo": 30,
    "E_Mail": 80,
    "Home_Page": 80,
    "Post_Code": 20,
    "PAN_Number": 20,
    "GST_Number": 20,
}


def _fit_to_bc_width(bc_field: str, value: str, truncated: list[str]) -> str:
    """Cut `value` to bc_field's known BC column width, word-boundary aware,
    and record the field name in `truncated` when a cut actually happened.
    A field with no configured width is returned unchanged -- BC still
    enforces its own limit server-side if this table is ever wrong or
    incomplete; this is a best-effort pre-check, not the source of truth."""
    limit = _BC_FIELD_MAX_LEN.get(bc_field)
    if limit is None or len(value) <= limit:
        return value
    # Prefer cutting at the last whitespace inside the limit so a word isn't
    # split mid-token; fall back to a hard cut only if there's no whitespace
    # to break on (one very long unbroken token).
    head = value[:limit]
    last_space = head.rfind(" ")
    cut = head[:last_space].rstrip() if last_space > 0 else head
    cut = cut.rstrip(",;").rstrip() or head  # don't leave a trailing separator
    truncated.append(bc_field)
    return cut


# portal Vendor attribute  ->  BC VendorCard field
# address_2/3/4 are NOT mapped here individually -- a standard BC VendorCard
# has only Address and Address_2, no Address_3/Address_4. They are joined
# into Address_2 explicitly in vendor_to_bc_payload() below, so a vendor whose
# address_resolver-driven segmentation produced 3 or 4 lines (see
# address_segmenter.py) still has every line represented on the BC push --
# nothing is silently dropped for lack of a field to put it in. If the target
# BC tenant exposes custom Address_3/Address_4 fields, change this to a
# straight 1:1 map instead of a join.
_FIELD_MAP: dict[str, str] = {
    "vendor_name": "Name",
    "address_1": "Address",
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

# Vendor attributes joined (in order) into BC's single Address_2 field.
_ADDRESS_2_JOIN_FIELDS = ("address_2", "address_3", "address_4")


def vendor_to_bc_payload(vendor: Vendor) -> dict:
    """Build the JSON body for a POST to .../VendorCard, plus a
    `_truncated_fields` list (BC field names cut to fit BC's own column
    width -- see _BC_FIELD_MAX_LEN). Non-empty `_truncated_fields` means the
    push will still succeed, but the sent value is not the FULL extracted
    address; the caller should flag the record for a human to check/complete
    directly in BC. Strip `_truncated_fields` before sending -- it is not a
    BC field."""
    payload: dict[str, str] = {"No": ""}
    truncated: list[str] = []

    for attr, bc_field in _FIELD_MAP.items():
        value = getattr(vendor, attr, None)
        if value:
            payload[bc_field] = _fit_to_bc_width(bc_field, str(value).strip(), truncated)

    address_2_parts = [
        str(getattr(vendor, attr, "") or "").strip() for attr in _ADDRESS_2_JOIN_FIELDS
    ]
    address_2 = ", ".join(p for p in address_2_parts if p)
    if address_2:
        payload["Address_2"] = _fit_to_bc_width("Address_2", address_2, truncated)

    if settings.BC_GEN_BUS_POSTING_GROUP:
        payload["Gen_Bus_Posting_Group"] = settings.BC_GEN_BUS_POSTING_GROUP
    if settings.BC_VAT_BUS_POSTING_GROUP:
        payload["VAT_Bus_Posting_Group"] = settings.BC_VAT_BUS_POSTING_GROUP
    if settings.BC_VENDOR_POSTING_GROUP:
        payload["Vendor_Posting_Group"] = settings.BC_VENDOR_POSTING_GROUP

    payload["_truncated_fields"] = truncated
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
    sent empty so BC assigns it from its own No. Series, and BC-column-width
    overflow is truncated with the cut fields reported in
    `_truncated_fields` (strip before sending -- not a BC field)."""
    payload: dict[str, str] = {"No": ""}
    truncated: list[str] = []

    for attr, bc_field in _CUSTOMER_FIELD_MAP.items():
        value = getattr(customer, attr, None)
        if value:
            payload[bc_field] = _fit_to_bc_width(bc_field, str(value).strip(), truncated)

    if settings.BC_GEN_BUS_POSTING_GROUP:
        payload["Gen_Bus_Posting_Group"] = settings.BC_GEN_BUS_POSTING_GROUP
    if settings.BC_VAT_BUS_POSTING_GROUP:
        payload["VAT_Bus_Posting_Group"] = settings.BC_VAT_BUS_POSTING_GROUP
    if settings.BC_CUSTOMER_POSTING_GROUP:
        payload["Customer_Posting_Group"] = settings.BC_CUSTOMER_POSTING_GROUP

    payload["_truncated_fields"] = truncated
    return payload


def customer_card_url() -> str:
    """The OData URL an operator POSTs the customer payload to."""
    company = settings.BC_COMPANY.replace("'", "''")
    return f"{settings.BC_ODATA_BASE}/Company('{company}')/CustomerCard"
