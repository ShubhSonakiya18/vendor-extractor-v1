"""Reshapes a V2 ExtractionResult onto the fixed customer-onboarding schema.

The shared extraction pipeline (extraction_pipeline/*) already does the real
work here -- OCR, PDF rasterization, document classification, label-proximity
field matching, GSTIN/PAN/IFSC/PIN validation, cross-document merging -- it
just returns its own canonical JSON shape (the keys in
config/field_dictionary.yaml, e.g. `vendor_name` for the business's legal
name -- that pipeline was originally built for a different, vendor-facing
use case). This module is the one place that translates that result into the
customer-onboarding form's own keys, so the shared engine and its config stay
untouched.

Two onboarding keys have no source anywhere in field_dictionary.yaml
(`contact_name`, `email_id_cc`) and are always returned as "" rather than
guessed -- there is nothing in a cheque/GST/Udyam/PAN document set that could
fill them.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Any

from rapidfuzz import fuzz

from .extraction_pipeline.models import ExtractionResult

# Address lines are positional in field_dictionary.yaml (address_1 = building/
# flat/road, address_2 = locality, address_3/4 = template overflow rows that
# are almost never populated) -- concatenating them is exactly the "Building +
# Road/Street" join the onboarding form's billing_address expects.
_ADDRESS_FIELDS = ["address_1", "address_2", "address_3", "address_4"]

# onboarding key -> canonical field_dictionary.yaml key. One-to-one only;
# billing_address (see _ADDRESS_FIELDS) and the bank sub-object are handled
# separately below.
_SIMPLE_FIELDS = {
    "company_name": "vendor_name",
    "city": "city",
    "state": "state",
    "country": "country",
    "email_id_to": "email",
    "phone_number": "telephone",
}

_BANK_FIELDS = {
    "bank_name": "bank_name",
    "account_number": "account_number",
    "branch": "branch_address",
}

# Friendly labels for source_documents[].document_type. Anything not listed
# here (a profile added later to document_profiles.yaml, or the "other"
# bucket) falls back to a titlecased version of the profile key, so a new
# document type never comes back blank.
_DOC_TYPE_LABELS = {
    "gst_certificate": "GST Certificate",
    "udyam_certificate": "Udyam Certificate",
    "cancelled_cheque": "Cancelled Cheque",
    "pan_card": "PAN Card",
    "bank_statement": "Bank Statement",
    "other": "Other",
}

# A GSTIN embeds its holder's PAN at characters 3-12 (0-indexed 2:12) -- this
# mirrors the `gstin_contains_pan` cross-check already in validation_rules.yaml,
# just used here as a fallback fill instead of a consistency check.
_PAN_RE = re.compile(r"^[A-Z]{5}[0-9]{4}[A-Z]$")

# A value ending "..., if any" is a GST/Udyam form printing an optional
# field's own caption ("Trade Name, if any", "Additional Trade Name, if
# any") -- never a real answer, whatever field it landed under.
_CAPTION_SUFFIX_RE = re.compile(r",?\s*if\s+any\s*$", re.IGNORECASE)

# Below this, a candidate value is too short for the partial-ratio check to
# mean anything -- a 5-character value scoring 90 against a 40-character
# label is just a coincidence, not a truncated read of it.
_MIN_CAPTION_LEAK_CHARS = 8

# Words that open a large share of Indian statutory-certificate captions --
# "Date of Liability", "Period of Validity", "Type of Registration",
# "Particulars of Approving Authority", "Nature of Business", "Category of
# Enterprise", "Constitution of Business", "Jurisdictional Office" -- but
# essentially never open a company's actual legal name. This is what actually
# stops the whack-a-mole of blocklisting one caption string at a time: a GST
# REG-06 / Udyam certificate's caption column draws from a small, closed
# vocabulary of lead words no matter how many different rows it has, so one
# rule generalises across all of them -- including ones no field in
# field_dictionary.yaml has a label for at all ("Date of Liability" isn't
# anyone's caption there, so _all_dictionary_labels() alone can never catch
# it, however many entries get added to it).
_CAPTION_LEAD_WORDS = {
    "date", "period", "type", "nature", "particulars", "category",
    "constitution", "jurisdictional", "registration", "validity", "address",
}
# A caption this shape is short -- a real value that happens to start with
# one of the words above but runs well past this length (a sentence, not a
# label) is treated as a real answer instead.
_MAX_CAPTION_LEAD_WORDS = 8


@lru_cache(maxsize=1)
def _all_dictionary_labels() -> tuple[str, ...]:
    """Every caption printed on a source document, across every field in
    field_dictionary.yaml -- not just the field being checked.

    A layout mis-read (the adjacent-row/adjacent-cell search landing one row
    off on a table-shaped certificate) can hand a field's own caption text to
    a *different* field as its value: "Trade Name, if any" landing in
    vendor_name, or address_1's "Address of Principal Place of Business"
    label landing there too. field_matcher's own `value_looks_like_a_label`
    penalty only ever checks a field against its *own* configured labels for
    exactly this reason, so it does not catch a foreign caption leaking
    through. This mirrors that check but against the whole dictionary, and is
    used here to reject outright rather than merely penalise.

    Lazily loaded and cached so importing this module for something that
    never calls `to_onboarding_schema` (e.g. a test exercising only the
    address-splitting helpers) never pays a YAML parse it doesn't need.
    """
    from .extraction_pipeline.config_loader import load_field_dictionary
    from .extraction_pipeline.extract.normalizer import clean_label

    return tuple(sorted({
        clean_label(label) for spec in load_field_dictionary() for label in spec.labels
    } - {""}))


def _is_caption_leak(value: str) -> bool:
    """True when `value` looks like a form's own printed caption rather than
    an answer it holds. Three independent checks, any one of which is enough:

    1. A known "...if any" caption suffix.
    2. Its first word is one of _CAPTION_LEAD_WORDS and it's caption-length --
       catches any statutory-certificate caption, whether or not
       field_dictionary.yaml has a field for it at all (see that set's
       docstring for why this is the one that generalises).
    3. A close (including truncated, e.g. OCR line-wrap dropping "Business"
       off the end of "Address of Principal Place of Business") match against
       any label actually configured in field_dictionary.yaml.
    """
    from .extraction_pipeline.extract.normalizer import clean_label

    v = value.strip()
    if not v:
        return False
    if _CAPTION_SUFFIX_RE.search(v):
        return True

    cleaned = clean_label(v)
    words = cleaned.split(" ")
    if words[0] in _CAPTION_LEAD_WORDS and len(words) <= _MAX_CAPTION_LEAD_WORDS:
        return True

    if len(cleaned) < _MIN_CAPTION_LEAK_CHARS:
        return False
    for label in _all_dictionary_labels():
        if len(label) < _MIN_CAPTION_LEAK_CHARS:
            continue
        # ratio: the value IS the caption (possibly with minor OCR damage).
        # partial_ratio: the value is a contiguous chunk OF the caption (a
        # line-wrapped OCR read that dropped a leading/trailing word).
        if fuzz.ratio(cleaned, label) >= 88 or fuzz.partial_ratio(cleaned, label) >= 90:
            return True
    return False


# Indian states/UTs, mirroring validation_rules.yaml's `indian_state` enum --
# duplicated (rather than loaded from there) because this list is only used
# here to recognise a state name sitting at the tail of a combined address
# string, a different job from validating an already-isolated state field.
_INDIAN_STATES = {
    s.casefold() for s in (
        "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh",
        "Goa", "Gujarat", "Haryana", "Himachal Pradesh", "Jharkhand", "Karnataka",
        "Kerala", "Madhya Pradesh", "Maharashtra", "Manipur", "Meghalaya",
        "Mizoram", "Nagaland", "Odisha", "Punjab", "Rajasthan", "Sikkim",
        "Tamil Nadu", "Telangana", "Tripura", "Uttar Pradesh", "Uttarakhand",
        "West Bengal", "Andaman and Nicobar Islands", "Chandigarh",
        "Dadra and Nagar Haveli and Daman and Diu", "Delhi",
        "Jammu and Kashmir", "Ladakh", "Lakshadweep", "Puducherry",
    )
}

_PIN_RE = re.compile(r"^[1-9][0-9]{5}$")

# Reverse map: canonical field key -> the onboarding field name it feeds, used
# only to translate needs_review entries so a flag on "gst_number" is reported
# to the caller as "gst_registration_number" -- the name they actually asked
# for, not the internal one.
_REVIEW_TARGET: dict[str, str] = {"vendor_name": "company_name", "pin_code": "zip_code",
                                  "gst_number": "gst_registration_number", "pan": "pan_number"}
for _onboarding_key, _canonical_key in _SIMPLE_FIELDS.items():
    _REVIEW_TARGET.setdefault(_canonical_key, _onboarding_key)
for _onboarding_key, _canonical_key in _BANK_FIELDS.items():
    _REVIEW_TARGET.setdefault(_canonical_key, f"bank_details.{_onboarding_key}")
_REVIEW_TARGET.setdefault("ifsc", "bank_details.ifsc_code")
for _addr_key in _ADDRESS_FIELDS:
    _REVIEW_TARGET.setdefault(_addr_key, "billing_address")


def _value(result: ExtractionResult, key: str) -> str:
    field = result.fields.get(key)
    return (field.value or "") if field else ""


def _valid_value(result: ExtractionResult, key: str) -> str:
    """Like `_value`, but a value the engine already marked format-invalid
    (failed its regex validator) comes back as "" instead of as-is.

    Scoped to the identifiers the onboarding form calls out by name --
    GSTIN, PAN, IFSC, PIN -- because a confidently-wrong 15-character string
    sitting in `gst_registration_number` is worse than an empty field: a
    human reviewing `fields_needing_review` needs to know to go re-check the
    source document, not be handed a value that merely looks plausible.
    """
    field = result.fields.get(key)
    if field is None or field.validation_status == "invalid":
        return ""
    return field.value or ""


def _best_text_value(result: ExtractionResult, key: str, review: list[str], review_target: str) -> str:
    """Like `_value`, but rejects a chosen value that is actually a form
    caption leaking through (see _is_caption_leak) and falls back through the
    engine's own runner-up `alternatives` for the same field until it finds
    one that isn't, or gives up and returns "".

    This exists because a caption leak is a *wrong, confident* answer, not a
    missing one -- field.alternatives (up to 5 runner-up candidates the
    engine already scored, see semantic_engine.SemanticEngine.extract) is the
    only place another guess can come from; nothing else about `result`
    carries raw document text for a fresh attempt. Falling back to a
    lower-scored alternative is itself flagged for review, same as the
    GST-derived PAN below -- a human should confirm it rather than trust it
    silently.
    """
    field = result.fields.get(key)
    if field is None:
        return ""

    candidates = [field.value] + [alt.get("value") for alt in field.alternatives]
    for i, candidate in enumerate(candidates):
        if candidate and not _is_caption_leak(candidate):
            if i > 0 and review_target not in review:
                review.append(review_target)
            return candidate
    return ""


def _split_trailing_location(address: str) -> tuple[str, str, str, str]:
    """Peel a trailing ", <city>, <state>[, <pincode>]" off a combined,
    comma-separated address string.

    Only exists to backfill city/state/pin_code from a document that prints
    them as one address line with no separate City/State caption to match
    against -- a GST REG-06 certificate's "Address of Principal Place of
    Business" (e.g. "...IT PARK, SAS Nagar, Punjab, 160068") is exactly this
    shape. State is matched against the same Indian-states list
    validation_rules.yaml's `indian_state` validator uses; city is only taken
    when a state was actually found immediately after it, since a bare
    trailing comma-segment with no state to anchor it is too easily some
    other part of the address, not a city.

    Returns (remaining_address, city, state, pin_code); any of the three may
    come back "" if that piece wasn't found. `remaining_address` is the input
    with whatever was matched removed from the tail.
    """
    parts = [p.strip() for p in address.split(",") if p.strip()]

    pin = ""
    if parts and _PIN_RE.match(parts[-1]):
        pin = parts.pop()

    state = ""
    if parts and parts[-1].casefold() in _INDIAN_STATES:
        state = parts.pop()

    city = ""
    if state and parts:
        city = parts.pop()

    return ", ".join(parts), city, state, pin


def _billing_address(result: ExtractionResult) -> str:
    parts = [_value(result, key) for key in _ADDRESS_FIELDS]
    return ", ".join(p for p in parts if p)


def _location_fields(result: ExtractionResult, review: list[str]) -> tuple[str, str, str, str]:
    """Resolve (billing_address, city, state, zip_code), preferring whatever
    the engine matched under its own caption and only reaching for
    _split_trailing_location when one of city/state/pin_code came back empty
    -- see that function's docstring for why a combined address needs this at
    all. `review` is mutated to flag anything filled this way, same
    convention as the GST-derived PAN below: it was inferred, not read
    directly off a labelled field, so a human should be able to see that.
    """
    address = _billing_address(result)
    city = _value(result, "city")
    state = _value(result, "state")
    zip_code = _valid_value(result, "pin_code")

    if address and not (city and state and zip_code):
        remaining, split_city, split_state, split_pin = _split_trailing_location(address)
        if split_city or split_state or split_pin:
            address = remaining or address
            if not city and split_city:
                city = split_city
                if "city" not in review:
                    review.append("city")
            if not state and split_state:
                state = split_state
                if "state" not in review:
                    review.append("state")
            if not zip_code and split_pin:
                zip_code = split_pin
                if "zip_code" not in review:
                    review.append("zip_code")

    return address, city, state, zip_code


def _doc_type_label(doc_type: str) -> str:
    return _DOC_TYPE_LABELS.get(doc_type, doc_type.replace("_", " ").title())


def _source_documents(result: ExtractionResult) -> list[dict[str, Any]]:
    out = []
    for doc in result.documents:
        # classification_score is keyword-weighted (content_weight=2.0 per
        # content keyword hit, see document_profiles.yaml) with no fixed
        # ceiling, so several strong hits can exceed 1.0 -- capped here since
        # "confidence" in this schema is documented as a 0-1 value.
        confidence = min(1.0, round(float(doc.get("classification_score", 0.0)), 2))
        out.append({
            "file_name": doc.get("document", ""),
            "document_type": _doc_type_label(doc.get("doc_type", "other")),
            "confidence": confidence,
        })
    return out


def _fields_needing_review(result: ExtractionResult) -> list[str]:
    """Translate the engine's needs_review entries into onboarding field
    names, deduplicated and in first-seen order.

    Only entries whose canonical field actually feeds this schema are kept --
    the underlying engine also tracks fields (company_type, tan, website, ...)
    that have no place in the onboarding form and would be noise here.
    """
    out: list[str] = []
    for item in result.needs_review:
        target = _REVIEW_TARGET.get(item.get("field", ""))
        if target and target not in out:
            out.append(target)
    return out


def to_onboarding_schema(result: ExtractionResult) -> dict[str, Any]:
    """Reshape a pipeline ExtractionResult into the onboarding form's fixed
    JSON schema. Never invents a value: every field either comes from an
    extracted/validated candidate, a documented derivation (PAN-from-GSTIN),
    or is left as "" when nothing in the uploaded documents supports it.
    """
    gst = _valid_value(result, "gst_number")
    pan = _valid_value(result, "pan")
    review = _fields_needing_review(result)

    if not pan and gst:
        derived = gst[2:12]
        if _PAN_RE.match(derived):
            pan = derived
            # Computed, not read off a page -- flag it like the engine's own
            # derive_from fields do (see semantic_engine.py), so a human can
            # see this PAN was inferred from the GSTIN rather than printed on
            # a PAN card.
            if "pan_number" not in review:
                review.append("pan_number")

    billing_address, city, state, zip_code = _location_fields(result, review)

    return {
        "company_name": _best_text_value(result, "vendor_name", review, "company_name"),
        "contact_name": "",
        "billing_address": billing_address,
        "city": city,
        "state": state,
        "zip_code": zip_code,
        "country": _value(result, "country"),
        "gst_registration_number": gst,
        "pan_number": pan,
        "email_id_to": _value(result, "email"),
        "email_id_cc": "",
        "phone_number": _value(result, "telephone"),
        # Business fields that no uploaded document contains -- returned as
        # empty (type defaults to "Services") so the schema matches the
        # customer form's 17 fields. The reviewer fills these in on-screen.
        "payment_terms": "",
        "salesperson": "",
        "region": "",
        "customer_agreement": "",
        "type": "Services",
        "bank_details": {
            "bank_name": _value(result, "bank_name"),
            "account_number": _value(result, "account_number"),
            "ifsc_code": _valid_value(result, "ifsc"),
            "branch": _value(result, "branch_address"),
        },
        "source_documents": _source_documents(result),
        "fields_needing_review": review,
    }
