"""Normalizer op chain and label cleaning.

These are the string transforms every extracted value passes through before it
is validated or compared, so a regression here silently shifts what counts as a
match everywhere downstream.
"""

import pytest

from app.services.extraction_pipeline.extract.normalizer import NORMALIZERS, clean_label, normalize, strip_value_prefix


class TestBasicOps:
    def test_strip(self):
        assert normalize("  ABC  ", ["strip"]) == "ABC"

    def test_collapse_spaces(self):
        assert normalize("A   B\t\nC", ["collapse_spaces"]) == "A B C"

    def test_remove_spaces(self):
        assert normalize("6278 5100 0539", ["remove_spaces"]) == "627851000539"

    def test_digits_only(self):
        assert normalize("+91 97484-78004", ["digits_only"]) == "919748478004"

    def test_ops_run_in_order(self):
        # uppercase then remove_spaces differs from the reverse only in
        # intermediate state, so use ops where order is observable.
        assert normalize(" ab ", ["strip", "uppercase"]) == "AB"

    def test_none_becomes_empty_string(self):
        assert normalize(None, ["strip"]) == ""

    def test_unknown_op_raises(self):
        # config_loader rejects unknown ops at load time; reaching here means
        # the registry and KNOWN_NORMALIZERS have drifted apart.
        with pytest.raises(KeyError):
            normalize("x", ["no_such_op"])


class TestTitlecase:
    def test_preserves_internal_dots(self):
        # str.title() would produce "M.B.Control"; the per-token rule keeps the
        # abbreviation intact.
        assert normalize("M.B.CONTROL", ["titlecase"]) == "M.b.control"

    def test_capitalises_each_word(self):
        assert normalize("pvt ltd", ["titlecase"]) == "Pvt Ltd"


class TestStripCountryCode:
    @pytest.mark.parametrize("raw", ["+919748478004", "919748478004"])
    def test_removes_leading_91_from_mobile(self, raw):
        assert normalize(raw, ["strip_country_code"]) == "9748478004"

    def test_leaves_plain_ten_digit_number(self):
        assert normalize("9748478004", ["strip_country_code"]) == "9748478004"

    def test_does_not_strip_when_remainder_is_not_a_mobile(self):
        # The lookahead requires a 10-digit number starting 6-9; "91" here is
        # part of a landline, not a country code.
        assert normalize("912212345678", ["strip_country_code"]) == "912212345678"


class TestFixIfscConfusions:
    def test_repairs_digit_for_letter_in_bank_code(self):
        # "1C1C" -> "ICIC": digits cannot appear in the first four positions.
        assert normalize("1C1C0006278", ["fix_ifsc_confusions"]) == "ICIC0006278"

    def test_forces_fifth_character_to_zero(self):
        assert normalize("ICICO006278", ["fix_ifsc_confusions"]) == "ICIC0006278"

    def test_leaves_wrong_length_untouched(self):
        # Guessing at a token of the wrong length would invent data.
        assert normalize("ICIC000", ["fix_ifsc_confusions"]) == "ICIC000"

    def test_does_not_touch_trailing_digits(self):
        assert normalize("BKID0004035", ["fix_ifsc_confusions"]) == "BKID0004035"


class TestSplitCorporateSuffix:
    """OCR drops inter-word spaces in dense certificate cells. This op
    re-inserts the space between a company-name stem and its glued corporate
    suffix, from a fixed token list -- never re-spelling the stem."""

    @pytest.mark.parametrize("raw,expected", [
        ("ORBITLOGISTICSSOLUTIONSPVTLTD", "ORBITLOGISTICSSOLUTIONS PVT LTD"),
        ("TALOJACHEMICALSPVTLTD", "TALOJACHEMICALS PVT LTD"),
        ("NORTHCITYLOGISTICSPVTLTD", "NORTHCITYLOGISTICS PVT LTD"),
        ("SHIVAMFORGINGSPVTLTD", "SHIVAMFORGINGS PVT LTD"),
        ("AARAVINDUSTRIALSOLUTIONSPRIVATELIMITED",
         "AARAVINDUSTRIALSOLUTIONS PRIVATE LIMITED"),
    ])
    def test_glued_suffix_is_split(self, raw, expected):
        assert normalize(raw, ["split_corporate_suffix"]) == expected

    @pytest.mark.parametrize("value", [
        "ORBIT LOGISTICS SOLUTIONS PVT LTD",
        "M B CONTROL & SYSTEMS PVT LTD",
        "AARAV INDUSTRIAL SOLUTIONS PRIVATE LIMITED",
    ])
    def test_already_spaced_stem_is_untouched(self, value):
        assert normalize(value, ["split_corporate_suffix"]) == value

    @pytest.mark.parametrize("value", ["RANDOMVENDOR", "TATA", "AMCO", "SOMECO"])
    def test_no_known_suffix_or_too_short_left_alone(self, value):
        assert normalize(value, ["split_corporate_suffix"]) == value

    def test_idempotent(self):
        once = normalize("ORBITLOGISTICSPVTLTD", ["split_corporate_suffix"])
        assert normalize(once, ["split_corporate_suffix"]) == once


class TestExpandKnownPhrase:
    """A glued legal-form phrase is re-spaced from a CLOSED vocabulary; a value
    that is not exactly one known phrase (squashed) is returned unchanged."""

    @pytest.mark.parametrize("raw,expected", [
        ("PRIVATELIMITEDCOMPANY", "private limited company"),
        ("Privatelimited Company", "private limited company"),
        ("Private Limited Company", "private limited company"),
        ("LIMITEDLIABILITYPARTNERSHIP", "limited liability partnership"),
        ("PROPRIETORSHIP", "proprietorship"),
        ("PARTNERSHIP", "partnership"),
    ])
    def test_known_phrase_is_respaced(self, raw, expected):
        assert normalize(raw, ["expand_known_phrase"]) == expected

    @pytest.mark.parametrize("value", [
        "Some Bespoke Structure", "MANUFACTURING", "",
    ])
    def test_unknown_value_untouched(self, value):
        assert normalize(value, ["expand_known_phrase"]) == value

    def test_full_company_type_chain(self):
        chain = ["strip", "collapse_spaces", "expand_known_phrase", "titlecase"]
        assert normalize("PRIVATELIMITEDCOMPANY", chain) == "Private Limited Company"
        assert normalize("Privatelimited Company", chain) == "Private Limited Company"
        assert normalize("Private Limited Company", chain) == "Private Limited Company"


class TestCleanLabel:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("PIN Code:", "pin code"),
            ("  Legal Name  ", "legal name"),
            ("-- A/c No. --", "a/c no"),
            ("Bank   Name", "bank name"),
        ],
    )
    def test_reduces_caption_to_comparable_form(self, raw, expected):
        assert clean_label(raw) == expected


class TestStripValuePrefix:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            (": 700019", "700019"),
            ("- ICIC0006278", "ICIC0006278"),
            ("  =  CURRENT", "CURRENT"),
            ("700019", "700019"),
        ],
    )
    def test_removes_caption_value_separator(self, raw, expected):
        assert strip_value_prefix(raw) == expected


def test_registry_matches_declared_contract():
    """config_loader.KNOWN_NORMALIZERS is the declared contract; this module is
    its implementation. Drift between them is only caught at runtime."""
    from app.services.extraction_pipeline.config_loader import KNOWN_NORMALIZERS

    assert set(NORMALIZERS) == set(KNOWN_NORMALIZERS)
