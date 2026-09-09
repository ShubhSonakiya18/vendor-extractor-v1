"""Validation engine: rule execution, status derivation, cross-document checks.

Two layers are covered. The first builds synthetic specs to pin the engine's
own semantics -- particularly `status_of`, whose four outcomes drive what the
results page shows. The second runs the real validators from
validation_rules.yaml against real Indian identifier formats, so a change to
the YAML that loosens a pattern fails here rather than in production.
"""

import re

import pytest

from app.services.extraction_pipeline.config_loader import CrossDocumentSpec, ValidationRules, ValidatorSpec, load_config
from app.services.extraction_pipeline.extract.validator import Validator


def spec(name, **kw):
    kw.setdefault("type", "regex")
    kw.setdefault("severity", "error")
    kw.setdefault("message", f"{name} failed")
    return ValidatorSpec(name=name, **kw)


def rules_with(*specs, threshold=0.85):
    return ValidationRules(
        validators={s.name: s for s in specs},
        cross_document=CrossDocumentSpec(
            similarity_threshold=threshold,
            source_precedence=["gst_certificate", "udyam_certificate", "cancelled_cheque"],
            severity="warning",
            message="documents disagree",
        ),
    )


# ---------------------------------------------------------------------------
# status_of -- the four outcomes shown in the UI
# ---------------------------------------------------------------------------

class TestStatusOf:
    def setup_method(self):
        self.v = Validator(rules_with())

    def test_no_findings_is_not_validated(self):
        """`not_validated` means no check ran -- not that a check passed. Fields
        configured with `validators: []` land here permanently."""
        assert self.v.status_of([]) == "not_validated"

    def test_all_passing_is_valid(self):
        v = Validator(rules_with(spec("ok_rule", pattern=re.compile(r"[A-Z]+"))))
        assert v.status_of(v.check(["ok_rule"], "ABC", {})) == "valid"

    def test_failing_error_is_invalid(self):
        v = Validator(rules_with(spec("strict", pattern=re.compile(r"\d+"), severity="error")))
        assert v.status_of(v.check(["strict"], "NOTDIGITS", {})) == "invalid"

    def test_failing_warning_is_warning(self):
        v = Validator(rules_with(spec("soft", pattern=re.compile(r"\d+"), severity="warning")))
        assert v.status_of(v.check(["soft"], "NOTDIGITS", {})) == "warning"

    def test_error_outranks_warning(self):
        v = Validator(rules_with(
            spec("soft", pattern=re.compile(r"\d+"), severity="warning"),
            spec("strict", pattern=re.compile(r"\d+"), severity="error"),
        ))
        assert v.status_of(v.check(["soft", "strict"], "X", {})) == "invalid"


# ---------------------------------------------------------------------------
# rule types
# ---------------------------------------------------------------------------

class TestRuleTypes:
    def test_length_respects_bounds(self):
        v = Validator(rules_with(spec("len", type="length", min=3, max=5)))
        assert v.check(["len"], "abcd", {})[0].ok is True
        assert v.check(["len"], "ab", {})[0].ok is False
        assert v.check(["len"], "abcdef", {})[0].ok is False

    def test_length_digits_only_rejects_letters(self):
        v = Validator(rules_with(spec("len", type="length", min=1, digits_only=True)))
        assert v.check(["len"], "12345", {})[0].ok is True
        assert v.check(["len"], "12a45", {})[0].ok is False

    def test_enum_is_case_insensitive(self):
        v = Validator(rules_with(spec("kind", type="enum", values=["Current", "Savings"])))
        assert v.check(["kind"], "CURRENT", {})[0].ok is True
        assert v.check(["kind"], "  savings ", {})[0].ok is True
        assert v.check(["kind"], "Loan", {})[0].ok is False

    def test_non_empty_rejects_whitespace(self):
        v = Validator(rules_with(spec("filled", type="non_empty")))
        assert v.check(["filled"], "x", {})[0].ok is True
        assert v.check(["filled"], "   ", {})[0].ok is False

    def test_unimplemented_type_raises(self):
        v = Validator(rules_with(spec("weird", type="does_not_exist")))
        with pytest.raises(KeyError):
            v.check(["weird"], "x", {})


class TestDerivedRules:
    """Derived rules compare a field against something computed from ANOTHER
    field, and return None -- 'cannot judge' -- when the counterpart is
    missing. That None is why a field can read `not_validated` despite having
    validators configured."""

    def test_substring_equals_matches(self):
        v = Validator(rules_with(spec(
            "gstin_has_pan", type="derived", rule="substring_equals",
            source="gst_number", target="pan", start=2, end=12,
        )))
        values = {"gst_number": "19AABCM7980K1ZU", "pan": "AABCM7980K"}
        assert v.check(["gstin_has_pan"], "", values)[0].ok is True

    def test_substring_equals_detects_mismatch(self):
        v = Validator(rules_with(spec(
            "gstin_has_pan", type="derived", rule="substring_equals",
            source="gst_number", target="pan", start=2, end=12,
        )))
        values = {"gst_number": "19AABCM7980K1ZU", "pan": "ZZZZZ9999Z"}
        assert v.check(["gstin_has_pan"], "", values)[0].ok is False

    def test_missing_counterpart_yields_no_finding(self):
        v = Validator(rules_with(spec(
            "gstin_has_pan", type="derived", rule="substring_equals",
            source="gst_number", target="pan", start=2, end=12,
        )))
        findings = v.check(["gstin_has_pan"], "", {"gst_number": "19AABCM7980K1ZU"})
        assert findings == []
        assert v.status_of(findings) == "not_validated"

    @pytest.mark.parametrize("gstin,expected", [
        ("19AABCM7980K1ZU", True),    # 19 = West Bengal
        ("01AABCM7980K1ZU", True),    # lowest assigned
        ("38AABCM7980K1ZU", True),    # highest assigned
        ("00AABCM7980K1ZU", False),   # unallocated
        ("39AABCM7980K1ZU", False),   # unallocated
    ])
    def test_state_code_range(self, gstin, expected):
        v = Validator(rules_with(spec(
            "state_code", type="derived", rule="state_code_valid",
            source="gst_number", start=0, end=2,
        )))
        assert v.check(["state_code"], "", {"gst_number": gstin})[0].ok is expected


class TestPinMatchesState:
    """pin_matches_state cross-checks the extracted state against the state the
    PIN directory says the 6-digit PIN belongs to. Skips (None) when either
    field is empty or the PIN is not in the directory."""

    def _v(self):
        return Validator(rules_with(spec(
            "pin_state", type="derived", rule="pin_matches_state",
            source="pin_code", severity="warning",
        )))

    def test_agreeing_pin_and_state_pass(self):
        # 711302 -> West Bengal in backend/data/address/pin_directory.csv
        f = self._v().check(["pin_state"], "West Bengal", {"pin_code": "711302"})
        assert f and f[0].ok is True

    def test_case_and_alias_insensitive(self):
        f = self._v().check(["pin_state"], "west bengal", {"pin_code": "711302"})
        assert f[0].ok is True

    def test_disagreeing_pin_and_state_fail_as_warning(self):
        f = self._v().check(["pin_state"], "Karnataka", {"pin_code": "711302"})
        assert f[0].ok is False
        assert f[0].severity == "warning"

    @pytest.mark.parametrize("state,pin", [
        ("", "711302"),          # no state to judge
        ("West Bengal", ""),     # no pin
        ("West Bengal", "abc"),  # not a 6-digit pin
        ("West Bengal", "999999"),  # pin not in the directory
    ])
    def test_insufficient_data_is_skipped(self, state, pin):
        assert self._v().check(["pin_state"], state, {"pin_code": pin}) == []


# ---------------------------------------------------------------------------
# cross-document comparison
# ---------------------------------------------------------------------------

class TestCrossDocument:
    def setup_method(self):
        self.v = Validator(rules_with())

    def test_single_value_is_not_a_disagreement(self):
        status, notes = self.v.compare_across_documents({"gst": "ACME PVT LTD"})
        assert status == "single_source"
        assert notes == []

    def test_no_values_is_not_checked(self):
        status, notes = self.v.compare_across_documents({})
        assert status == "not_checked"

    def test_punctuation_variants_agree(self):
        """The whole point of a fuzzy threshold: the same company spelled three
        ways across three documents must not raise a mismatch."""
        status, notes = self.v.compare_across_documents({
            "gst": "M B CONTROL & SYSTEMS PVT LTD",
            "udyam": "M.B. CONTROL & SYSTEMS PVT LTD",
        })
        assert status == "consistent"
        assert notes == []

    def test_truncated_ocr_read_is_flagged(self):
        """The real observed failure: OCR truncated the Udyam name."""
        status, notes = self.v.compare_across_documents({
            "gst": "M B CONTROL & SYSTEMS PVT LTD",
            "udyam": "SYSTEMS PVT L",
        })
        assert status == "inconsistent"
        assert len(notes) == 1

    def test_empty_values_are_ignored(self):
        status, _ = self.v.compare_across_documents({"gst": "ACME PVT LTD", "cheque": ""})
        assert status == "single_source"

    def test_threshold_is_honoured(self):
        strict = Validator(rules_with(threshold=0.99))
        status, _ = strict.compare_across_documents({
            "a": "M B CONTROL & SYSTEMS PVT LTD",
            "b": "M.B. CONTROL & SYSTEMS PVT LTD",
        })
        assert status == "inconsistent"


# ---------------------------------------------------------------------------
# the real configured validators
# ---------------------------------------------------------------------------

class TestRealValidators:
    """Runs the shipped validation_rules.yaml. These assertions encode what the
    Indian identifier formats actually are, so loosening a pattern in YAML
    fails here."""

    @classmethod
    def setup_class(cls):
        _, rules = load_config()
        cls.v = Validator(rules)

    def ok(self, name, value, values=None):
        findings = self.v.check([name], value, values or {})
        assert findings, f"{name} produced no finding for {value!r}"
        return findings[0].ok

    @pytest.mark.parametrize("value", ["19AABCM7980K1ZU", "27AAPFU0939F1ZV"])
    def test_gstin_accepts_valid(self, value):
        assert self.ok("gstin_format", value) is True

    @pytest.mark.parametrize("value", [
        "19AABCM7980K1Z",     # too short
        "1AABCM7980K1ZU",     # one-digit state code
        "19aabcm7980k1zu",    # lowercase
        "",
    ])
    def test_gstin_rejects_invalid(self, value):
        assert self.ok("gstin_format", value) is False

    @pytest.mark.parametrize("value", ["AABCM7980K", "ABCPD1234E"])
    def test_pan_accepts_valid(self, value):
        assert self.ok("pan_format", value) is True

    @pytest.mark.parametrize("value", ["AABCM7980", "AABCM79801", "1ABCM7980K", ""])
    def test_pan_rejects_invalid(self, value):
        assert self.ok("pan_format", value) is False

    @pytest.mark.parametrize("value", ["ICIC0006278", "BKID0004035", "SBIN0001234"])
    def test_ifsc_accepts_valid(self, value):
        assert self.ok("ifsc_format", value) is True

    @pytest.mark.parametrize("value", [
        "ICIC1006278",   # fifth character must be 0
        "ICIC000627",    # too short
        "IC1C0006278",   # digit in bank code
    ])
    def test_ifsc_rejects_invalid(self, value):
        assert self.ok("ifsc_format", value) is False

    @pytest.mark.parametrize("value", ["700019", "110001"])
    def test_pin_accepts_valid(self, value):
        assert self.ok("pin_format", value) is True

    @pytest.mark.parametrize("value", ["70001", "7000199", "0700019", "abcdef"])
    def test_pin_rejects_invalid(self, value):
        assert self.ok("pin_format", value) is False

    def test_gstin_embeds_its_pan(self):
        """Characters 2-12 of a GSTIN are the holder's PAN. This cross-field
        rule is what catches a GSTIN and PAN that were read off different
        documents and do not belong together."""
        values = {"gst_number": "19AABCM7980K1ZU", "pan": "AABCM7980K"}
        findings = self.v.check(["gstin_contains_pan"], "", values)
        assert findings[0].ok is True

        bad = {"gst_number": "19AABCM7980K1ZU", "pan": "ABCPD1234E"}
        assert self.v.check(["gstin_contains_pan"], "", bad)[0].ok is False
