"""onboarding_mapper: ExtractionResult -> the fixed onboarding JSON schema.

No OCR runs here at all -- ExtractionResult is built directly from FieldResult
objects, the same way test_validator.py pins the validation engine's own
semantics with synthetic specs rather than real documents. That is also the
"mock the OCR call" the task asked for: the thing being tested is the
translation layer, not RapidOCR itself.
"""

from __future__ import annotations

from app.services.extraction_pipeline.models import ExtractionResult, FieldResult
from app.services.onboarding_mapper import to_onboarding_schema


def _result(fields: dict, documents: list | None = None, needs_review: list | None = None) -> ExtractionResult:
    """Build an ExtractionResult from {canonical_key: (value, validation_status)}."""
    result = ExtractionResult(documents=documents or [], needs_review=needs_review or [])
    for key, spec in fields.items():
        value, status = spec if isinstance(spec, tuple) else (spec, "valid")
        result.fields[key] = FieldResult(key=key, value=value, confidence=0.9, validation_status=status)
    return result


# A fully-populated, all-valid result -- one of each source document, no
# review flags -- to pin the "happy path" shape field-by-field.
FULL_FIELDS = {
    "vendor_name": "M B Control Systems Pvt Ltd",
    "address_1": "12 Park Street",
    "address_2": "Ballygunge",
    "city": "Kolkata",
    "state": "West Bengal",
    "country": "India",
    "pin_code": "700019",
    "gst_number": "19AAAAA0000A1Z5",
    "pan": "AAAAA0000A",
    "email": "accounts@mbcontrol.example",
    "telephone": "9876543210",
    "bank_name": "ICICI Bank",
    "account_number": "123456789012",
    "ifsc": "ICIC0001234",
    "branch_address": "Kolkata Gariahat Branch",
}

FULL_DOCUMENTS = [
    {"document": "gst_cert.pdf", "doc_type": "gst_certificate", "classification_score": 8.0},
    {"document": "cheque.jpg", "doc_type": "cancelled_cheque", "classification_score": 6.0},
]


class TestHappyPath:
    def setup_method(self):
        self.result = _result(FULL_FIELDS, documents=FULL_DOCUMENTS)
        self.out = to_onboarding_schema(self.result)

    def test_top_level_keys_match_schema_exactly(self):
        assert set(self.out) == {
            "company_name", "contact_name", "billing_address", "city", "state",
            "zip_code", "country", "gst_registration_number", "pan_number",
            "email_id_to", "email_id_cc", "phone_number",
            "payment_terms", "salesperson", "region", "customer_agreement", "type",
            "bank_details", "source_documents", "fields_needing_review",
        }
        assert set(self.out["bank_details"]) == {
            "bank_name", "account_number", "ifsc_code", "branch",
        }

    def test_business_fields_present_but_empty(self):
        # No document supplies these; type defaults to "Services".
        assert self.out["payment_terms"] == ""
        assert self.out["salesperson"] == ""
        assert self.out["region"] == ""
        assert self.out["customer_agreement"] == ""
        assert self.out["type"] == "Services"

    def test_identity_and_address_fields(self):
        assert self.out["company_name"] == "M B Control Systems Pvt Ltd"
        assert self.out["billing_address"] == "12 Park Street, Ballygunge"
        assert self.out["city"] == "Kolkata"
        assert self.out["state"] == "West Bengal"
        assert self.out["zip_code"] == "700019"
        assert self.out["country"] == "India"

    def test_statutory_and_contact_fields(self):
        assert self.out["gst_registration_number"] == "19AAAAA0000A1Z5"
        assert self.out["pan_number"] == "AAAAA0000A"
        assert self.out["email_id_to"] == "accounts@mbcontrol.example"
        assert self.out["phone_number"] == "9876543210"

    def test_bank_details(self):
        assert self.out["bank_details"] == {
            "bank_name": "ICICI Bank",
            "account_number": "123456789012",
            "ifsc_code": "ICIC0001234",
            "branch": "Kolkata Gariahat Branch",
        }

    def test_fields_with_no_source_document_are_never_hallucinated(self):
        assert self.out["contact_name"] == ""
        assert self.out["email_id_cc"] == ""

    def test_nothing_flagged_when_everything_is_valid(self):
        assert self.out["fields_needing_review"] == []

    def test_source_documents_carries_file_name_type_and_capped_confidence(self):
        assert self.out["source_documents"] == [
            {"file_name": "gst_cert.pdf", "document_type": "GST Certificate", "confidence": 1.0},
            {"file_name": "cheque.jpg", "document_type": "Cancelled Cheque", "confidence": 1.0},
        ]


class TestMissingFields:
    def test_absent_field_is_empty_string_not_missing_key(self):
        out = to_onboarding_schema(_result({"vendor_name": "Acme"}))
        assert out["gst_registration_number"] == ""
        assert out["bank_details"]["ifsc_code"] == ""
        assert out["billing_address"] == ""

    def test_unrecognised_document_type_falls_back_to_titlecase(self):
        out = to_onboarding_schema(_result(
            {}, documents=[{"document": "x.pdf", "doc_type": "bank_statement", "classification_score": 3.0}],
        ))
        assert out["source_documents"][0]["document_type"] == "Bank Statement"


class TestInvalidIdentifiersAreBlankedNotPassedThrough:
    """GSTIN/PAN/IFSC/PIN: a value the engine already marked invalid must not
    reach the caller as though it were trustworthy."""

    def test_invalid_gstin_is_blanked(self):
        out = to_onboarding_schema(_result({"gst_number": ("NOT-A-GSTIN", "invalid")}))
        assert out["gst_registration_number"] == ""

    def test_invalid_pan_is_blanked(self):
        out = to_onboarding_schema(_result({"pan": ("GARBLED123", "invalid")}))
        assert out["pan_number"] == ""

    def test_invalid_ifsc_is_blanked(self):
        out = to_onboarding_schema(_result({"ifsc": ("BADCODE", "invalid")}))
        assert out["bank_details"]["ifsc_code"] == ""

    def test_invalid_pin_is_blanked(self):
        out = to_onboarding_schema(_result({"pin_code": ("ABCDEF", "invalid")}))
        assert out["zip_code"] == ""

    def test_warning_status_still_passes_through(self):
        # A "warning" (e.g. an unrecognised state name) is still the best
        # evidence available -- only "invalid" format failures are blanked.
        out = to_onboarding_schema(_result({"state": ("Atlantis", "warning")}))
        assert out["state"] == "Atlantis"


class TestNeedsReviewTranslation:
    def test_engine_field_names_translate_to_onboarding_names(self):
        needs_review = [
            {"field": "gst_number", "reason": "low_confidence", "confidence": 0.4, "severity": "error"},
            {"field": "ifsc", "reason": "failed_validation", "confidence": 0.5, "severity": "error"},
        ]
        out = to_onboarding_schema(_result(FULL_FIELDS, needs_review=needs_review))
        assert out["fields_needing_review"] == ["gst_registration_number", "bank_details.ifsc_code"]

    def test_irrelevant_engine_fields_are_dropped(self):
        # `company_type` / `tan` / `website` etc. exist in field_dictionary.yaml
        # but have no place in this schema -- flagging them would be noise.
        needs_review = [{"field": "company_type", "reason": "low_confidence", "confidence": 0.3, "severity": "error"}]
        out = to_onboarding_schema(_result(FULL_FIELDS, needs_review=needs_review))
        assert out["fields_needing_review"] == []

    def test_duplicate_targets_are_not_repeated(self):
        needs_review = [
            {"field": "address_1", "reason": "low_confidence", "confidence": 0.4, "severity": "error"},
            {"field": "address_2", "reason": "low_confidence", "confidence": 0.4, "severity": "error"},
        ]
        out = to_onboarding_schema(_result(FULL_FIELDS, needs_review=needs_review))
        assert out["fields_needing_review"] == ["billing_address"]


class TestPanFallbackFromGstin:
    def test_pan_is_derived_from_gstin_when_no_pan_field_present(self):
        # Characters 3-12 (0-indexed 2:12) of a GSTIN are the holder's PAN.
        out = to_onboarding_schema(_result({"gst_number": "19AAAAA0000A1Z5"}))
        assert out["pan_number"] == "AAAAA0000A"
        assert out["fields_needing_review"] == ["pan_number"]

    def test_real_pan_field_wins_over_derivation(self):
        out = to_onboarding_schema(_result({
            "gst_number": "19AAAAA0000A1Z5",
            "pan": "BBBBB1111B",
        }))
        assert out["pan_number"] == "BBBBB1111B"
        assert out["fields_needing_review"] == []

    def test_never_fabricates_a_pan_shaped_value_from_a_malformed_gstin(self):
        # gst_number here is marked invalid, so _valid_value already blanks it
        # before the derivation ever runs -- nothing to derive from.
        out = to_onboarding_schema(_result({"gst_number": ("19AAAAA0000A1Z5EXTRA", "invalid")}))
        assert out["pan_number"] == ""
        assert "pan_number" not in out["fields_needing_review"]
