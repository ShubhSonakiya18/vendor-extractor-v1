"""
V2 Normalizer
=============
Applies the `normalization:` op chain from field_dictionary.yaml to a raw
extracted string. Ops are looked up in a registry, so the set of available ops
is data to the rest of the system -- config_loader.KNOWN_NORMALIZERS is the
declared contract and this module is the implementation of it.

No field names appear here. `fix_ifsc_confusions` is named for the shape it
repairs, not for the field that happens to use it.
"""

from __future__ import annotations

import re
from typing import Callable

_MULTISPACE = re.compile(r"\s+")
_NON_DIGIT = re.compile(r"\D")
_LEADING_91 = re.compile(r"^(?:\+?91)[\s\-]?(?=[6-9]\d{9}$)")

# OCR confuses these glyph pairs constantly on scanned documents. The mapping
# is only ever applied at positions where the field's format guarantees a
# letter or a digit, never blindly across the whole string.
_DIGIT_TO_LETTER = {"0": "O", "1": "I", "5": "S", "8": "B", "2": "Z", "6": "G"}


def _titlecase(value: str) -> str:
    # str.title() mangles "PVT LTD" -> "Pvt Ltd" acceptably but also turns
    # "M.B.CONTROL" into "M.B.Control"; capitalize per whitespace token only.
    return " ".join(w[:1].upper() + w[1:].lower() if w else w for w in value.split(" "))


# -- OCR word-glue repair for name / constitution fields --------------------
# Dense table cells on a scanned Udyam/GST certificate are frequently read by
# OCR with every inter-word space dropped ("ORBITLOGISTICSSOLUTIONSPVTLTD",
# "PRIVATELIMITEDCOMPANY"). These two ops re-insert the spaces DETERMINISTICALLY
# from a fixed vocabulary -- no field names, no guessing at the spelling of the
# proper-noun stem.

# Legal-form phrases as a closed set. Longest concatenation match wins, so
# "privatelimitedcompany" resolves before "limitedcompany" / "company".
_KNOWN_PHRASES: tuple[str, ...] = (
    "private limited company",
    "public limited company",
    "one person company",
    "limited liability partnership",
    "private limited",
    "public limited",
    "partnership firm",
    "sole proprietorship",
    "proprietorship firm",
    "proprietorship",
    "partnership",
    "hindu undivided family",
    "co operative society",
    "cooperative society",
    "trust",
    "society",
    "government",
    "limited",
)
_PHRASE_BY_SQUASH = {re.sub(r"\s+", "", p): p for p in _KNOWN_PHRASES}
_PHRASE_SQUASHES = sorted(_PHRASE_BY_SQUASH, key=len, reverse=True)

# Corporate-suffix tokens peeled off a GLUED tail, longest first. Deliberately
# excludes the 2-char "co"/"inc" -- too many real name stems end in those
# letters ("...AMCO", "...FINE"). The stem in front is never re-spelt.
_CORP_SUFFIX_TOKENS: tuple[str, ...] = (
    "privatelimited", "private", "limited", "pvtltd", "pvt", "ltd", "llp",
    "opc", "plc", "company", "corporation", "corp", "incorporated",
    "enterprises", "enterprise", "industries",
)
# after peeling, the remaining stem must be at least this long to accept the
# split -- guards against turning a short word into "<3 letters> + SUFFIX".
_CORP_MIN_STEM = 4
_CORP_CANON = {
    "pvt": "PVT", "pvtltd": "PVT LTD", "ltd": "LTD", "limited": "LIMITED",
    "private": "PRIVATE", "privatelimited": "PRIVATE LIMITED", "llp": "LLP",
    "opc": "OPC", "plc": "PLC", "company": "COMPANY", "co": "CO",
    "corporation": "CORPORATION", "corp": "CORP",
    "incorporated": "INCORPORATED", "inc": "INC",
    "enterprises": "ENTERPRISES", "enterprise": "ENTERPRISE",
    "industries": "INDUSTRIES", "udyog": "UDYOG",
}


def _expand_known_phrase(value: str) -> str:
    """Re-space a value that is (a squash of) one known legal-form phrase.

    'PRIVATELIMITEDCOMPANY' -> 'private limited company'. A value that is not a
    clean concatenation of one phrase is returned unchanged, so an already
    correct 'Private Limited Company' or an unrelated string is never touched.
    """
    squash = re.sub(r"\s+", "", value).lower()
    if not squash:
        return value
    for cand in _PHRASE_SQUASHES:
        if squash == cand:
            return _PHRASE_BY_SQUASH[cand]
    return value


def _split_corporate_suffix(value: str) -> str:
    """Insert the missing space(s) between a company name stem and its glued
    corporate suffix: 'ORBITLOGISTICSSOLUTIONSPVTLTD' -> 'ORBITLOGISTICSSOLUTIONS
    PVT LTD'. Only recognised suffix tokens are peeled; the stem's own internal
    spacing is left exactly as received (so an already-spaced value keeps its
    stem and only its suffix tokens are re-canonicalised)."""
    if not value or " " in value.strip():
        # already has spaces -- only re-canonicalise a trailing suffix run,
        # don't attempt to re-split the stem.
        parts = value.split()
        out: list[str] = []
        i = len(parts)
        while i > 0 and parts[i - 1].lower() in _CORP_CANON:
            i -= 1
        out = parts[:i] + [_CORP_CANON[p.lower()] for p in parts[i:]]
        return " ".join(out) if out else value

    tail = value
    suffixes: list[str] = []
    changed = True
    while changed and len(tail) > _CORP_MIN_STEM:
        changed = False
        low = tail.lower()
        for tok in _CORP_SUFFIX_TOKENS:
            if low.endswith(tok) and len(tail) - len(tok) >= _CORP_MIN_STEM:
                suffixes.append(_CORP_CANON.get(tok, tok.upper()))
                tail = tail[: -len(tok)]
                changed = True
                break
    if not suffixes:
        return value
    return " ".join([tail] + list(reversed(suffixes)))


def _fix_ifsc_confusions(value: str) -> str:
    """Repair an 11-character IFSC-shaped token.

    IFSC is 4 letters, then a literal '0', then 6 alphanumerics. That fixed
    shape means a digit appearing in the first four positions is certainly an
    OCR error and can be mapped back to its letter, and position 4 can be
    forced to '0'. Anything that is not 11 characters is left alone -- guessing
    at a token of the wrong length would invent data.
    """
    if len(value) != 11:
        return value
    chars = list(value.upper())
    for i in range(4):
        chars[i] = _DIGIT_TO_LETTER.get(chars[i], chars[i])
    chars[4] = "0"
    return "".join(chars)


NORMALIZERS: dict[str, Callable[[str], str]] = {
    "strip": lambda v: v.strip(),
    "collapse_spaces": lambda v: _MULTISPACE.sub(" ", v).strip(),
    "remove_spaces": lambda v: _MULTISPACE.sub("", v),
    "uppercase": lambda v: v.upper(),
    "lowercase": lambda v: v.lower(),
    "titlecase": _titlecase,
    "digits_only": lambda v: _NON_DIGIT.sub("", v),
    "strip_country_code": lambda v: _LEADING_91.sub("", v),
    "fix_ifsc_confusions": _fix_ifsc_confusions,
    "split_corporate_suffix": _split_corporate_suffix,
    "expand_known_phrase": _expand_known_phrase,
}


def normalize(value: str | None, ops: list[str]) -> str:
    """Run `value` through the named ops in order."""
    if value is None:
        return ""
    out = str(value)
    for op in ops:
        fn = NORMALIZERS.get(op)
        if fn is None:
            # config_loader rejects unknown ops at load time, so reaching here
            # means the registry and KNOWN_NORMALIZERS have drifted apart.
            raise KeyError(f"Normalizer op not implemented: {op!r}")
        out = fn(out)
    return out


def clean_label(text: str) -> str:
    """Reduce a caption to a comparable form: no trailing punctuation, no
    separators, collapsed whitespace, lowercased."""
    text = text.strip()
    text = re.sub(r"^[\s\-:.|*]+", "", text)
    text = re.sub(r"[\s\-:.|*]+$", "", text)
    text = _MULTISPACE.sub(" ", text)
    return text.lower()


def strip_value_prefix(text: str) -> str:
    """Remove the separator sitting between a caption and its value when both
    share one span, e.g. 'PIN Code: 700019' -> '700019' once the caption part
    has been sliced off."""
    return re.sub(r"^[\s:\-–—=|.,)#]+", "", text).strip()
