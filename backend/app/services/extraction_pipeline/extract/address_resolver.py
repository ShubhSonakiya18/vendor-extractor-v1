"""Deterministic address segmentation.

Turns a single combined address string -- the shape a document gives when it
prints the whole address on one line with no per-part captions (a Udyam
certificate's address block, a cancelled cheque, a GST REG-06 "Address of
Principal Place of Business" line) -- into the fixed fields the vendor /
customer creation form needs:

    address_1  city  state  pin_code        (address_2 is genuine leftover only)

This is the Phase 2 module from the address-segmentation plan. It absorbs
onboarding_mapper._split_trailing_location (peel a trailing ", city, state,
pin") and adds:

  * PIN-directory lookup -- a 6-digit PIN found ANYWHERE in the string is
    looked up in backend/data/address/pin_directory.csv for the authoritative
    (state, district). This fixes an OCR-misspelled state ("WEST BANGAL"),
    supplies a state when the string never names one, and gives a district to
    fall back on for the city.
  * state token matching against the data-file list (exact / alias / OCR
    misread), including a two-segment join for "WEST", "BENGAL" split across
    OCR lines; PIN's state as the fallback.
  * smart city selection -- the token nearest the state/PIN end of the string
    that is a known city or equals the PIN's district, so locality names
    sitting between the street and the real city ("..., ANDUL, NATIBPUR,
    HOWRAH, ...") don't get grabbed instead.
  * confidence scoring -- high / medium / low, to drive review flags.

`interpretation B`: address_1 is the whole premises/building/street portion
kept as one string (no building-vs-street sub-split); address_2 is "" unless
there is genuine leftover text.

Everything is local and deterministic -- no network, no model. If the lookup
data files were never built, `data_files_present()` is False and the module
degrades to the token-only path (roughly the old behaviour).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .address_lookups import (
    canonical_state,
    data_files_present,
    is_known_city,
    pin_state_district,
)

# A 6-digit Indian PIN: first digit 1-9. Matched anywhere in the string, not
# just as the last comma-segment, so "..., Punjab 160055" and "..., 711302,
# INDIA" both yield the PIN.
_PIN_RE = re.compile(r"(?<!\d)([1-9]\d{5})(?!\d)")

# Trailing tokens that are a country, not an address part -- dropped so they
# never land in a field or get mistaken for a city.
_COUNTRY_TOKENS = {"india", "bharat", "in", "ind"}

# Words that, alone, are never a city even if they sneak into the city slot.
_NON_CITY_TOKENS = _COUNTRY_TOKENS | {
    "near", "opp", "opposite", "behind", "beside", "above", "below",
    "road", "street", "lane", "marg", "nagar", "colony", "sector", "block",
    "floor", "plot", "no", "gala", "unit", "phase", "part",
}

_CONF_HIGH = "high"
_CONF_MEDIUM = "medium"
_CONF_LOW = "low"


@dataclass
class ResolvedAddress:
    address_1: str = ""
    address_2: str = ""
    city: str = ""
    state: str = ""
    pin_code: str = ""
    confidence: str = _CONF_LOW
    # what each field was resolved from, for review flags / debugging
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, str]:
        return {
            "address_1": self.address_1,
            "address_2": self.address_2,
            "city": self.city,
            "state": self.state,
            "pin_code": self.pin_code,
        }


def _segments(address: str) -> list[str]:
    """Comma-split, trim, drop empties. Newlines are treated as commas so an
    OCR line break behaves the same as a separator."""
    raw = (address or "").replace("\n", ",").replace("\r", ",")
    return [s.strip() for s in raw.split(",") if s.strip()]


def _titlecase_city(token: str) -> str:
    """'kolkata' -> 'Kolkata', 'NEW DELHI' -> 'New Delhi'. Leaves an
    already-mixed-case token alone (it is probably deliberate)."""
    t = token.strip()
    if t.isupper() or t.islower():
        return " ".join(w.capitalize() for w in t.split())
    return t


def _strip_pin(segments: list[str]) -> tuple[list[str], str]:
    """Remove the PIN from wherever it sits (usually the tail, sometimes glued
    to the state token: 'Punjab 160055'). Returns (segments_without_pin, pin)."""
    for i, seg in enumerate(segments):
        m = _PIN_RE.search(seg)
        if not m:
            continue
        pin = m.group(1)
        remainder = (seg[: m.start()] + " " + seg[m.end():]).strip(" ,")
        new = segments[:i] + ([remainder] if remainder else []) + segments[i + 1:]
        return new, pin
    return segments, ""


def _drop_trailing_country(segments: list[str]) -> list[str]:
    while segments and segments[-1].casefold() in _COUNTRY_TOKENS:
        segments = segments[:-1]
    return segments


def _match_state(segments: list[str]) -> tuple[list[str], str, int]:
    """Find a state token, scanning from the END (the state sits near the tail).

    Tries each single segment, then a join of two adjacent segments (handles
    'WEST', 'BENGAL' from an OCR line break). Returns
    (segments_without_state, canonical_state_or_'', index_it_was_at).
    """
    n = len(segments)
    # single-segment, from the end
    for i in range(n - 1, -1, -1):
        canon = canonical_state(segments[i])
        if canon:
            return segments[:i] + segments[i + 1:], canon, i
    # two adjacent segments joined, from the end
    for i in range(n - 2, -1, -1):
        canon = canonical_state(f"{segments[i]} {segments[i + 1]}")
        if canon:
            return segments[:i] + segments[i + 2:], canon, i
    return segments, "", -1


def _pick_city(
    segments: list[str],
    pin_district: str,
    have_state_anchor: bool,
) -> tuple[list[str], str]:
    """Choose the city token and remove it from `segments`.

    Preference, strongest first:
      1. the last segment that is a known city name (the string's own city
         token wins -- if the document says MOHALI, the city is Mohali even
         when the PIN's district is "S.A.S Nagar")
      2. the last segment that equals the PIN's district
      3. if a state/PIN anchor exists, the last remaining segment (the old
         "segment right before the state" behaviour) -- but only if it isn't
         obviously a street/landmark token

    After the city is chosen, any *adjacent* trailing segment that just
    repeats the city or names the PIN's district is dropped too -- it is a
    redundant district label ("KOLKATA, KOLKATA" / "MOHALI, S.A.S Nagar"),
    not part of the street.
    Returns (segments_without_city, city_or_'').
    """
    n = len(segments)
    dist_norm = (pin_district or "").casefold().replace(" ", "")

    def _norm(s: str) -> str:
        return s.casefold().replace(" ", "").replace(".", "")

    # Known-city tokens, end-first. When more than one exists and one of them
    # is just the PIN's district, prefer the *other* -- the string naming both
    # MOHALI and its district "S.A.S Nagar" means the city is Mohali.
    known = [i for i in range(n - 1, -1, -1) if is_known_city(segments[i])]
    chosen = -1
    if known:
        non_district = [
            i for i in known
            if not dist_norm or _norm(segments[i]) != _norm(pin_district)
        ]
        chosen = non_district[0] if non_district else known[0]
    if chosen == -1 and dist_norm:
        for i in range(n - 1, -1, -1):
            if _norm(segments[i]) == _norm(pin_district):
                chosen = i
                break

    if chosen == -1 and have_state_anchor and segments:
        last = segments[-1]
        words = last.casefold().split()
        if (
            last.casefold() not in _NON_CITY_TOKENS
            and not last.isdigit()
            and not any(w in _NON_CITY_TOKENS for w in words)
        ):
            chosen = n - 1

    if chosen == -1:
        return segments, ""

    city_token = segments[chosen]
    rest = segments[:chosen] + segments[chosen + 1:]

    # drop redundant trailing labels that just repeat the city or name the
    # PIN's district ("KOLKATA, KOLKATA" / "MOHALI, S.A.S Nagar") -- they are
    # not part of the street.
    city_norm = _norm(city_token)
    while rest and (
        _norm(rest[-1]) == city_norm
        or (dist_norm and _norm(rest[-1]) == _norm(pin_district))
    ):
        rest = rest[:-1]

    return rest, _titlecase_city(city_token)


def resolve_address_blob(address: str) -> ResolvedAddress:
    """Segment one combined address string. See module docstring."""
    out = ResolvedAddress()
    segments = _drop_trailing_country(_segments(address))
    if not segments:
        return out

    segments, pin = _strip_pin(segments)
    segments = _drop_trailing_country(segments)
    out.pin_code = pin

    pin_state, pin_district = "", ""
    if pin and data_files_present():
        looked_up = pin_state_district(pin)
        if looked_up:
            pin_state, pin_district = looked_up
            out.notes.append(f"pin {pin} -> {pin_state}/{pin_district}")

    segments, token_state, _ = _match_state(segments)

    if token_state:
        out.state = token_state
        out.notes.append(f"state from token: {token_state}")
        if pin_state and pin_state != token_state:
            out.notes.append(f"state token {token_state} != pin state {pin_state}")
    elif pin_state:
        out.state = pin_state
        out.notes.append(f"state from pin fallback: {pin_state}")

    have_anchor = bool(out.state or pin)
    segments, city = _pick_city(segments, pin_district, have_anchor)
    if city:
        out.city = city
        out.notes.append(f"city: {city}")

    out.address_1 = ", ".join(segments).strip(" ,")

    out.confidence = _score(out, pin, token_state, pin_state)
    return out


def _score(out: ResolvedAddress, pin: str, token_state: str, pin_state: str) -> str:
    """high  -- PIN present and (no state token, or it agrees) and a city found
       medium -- 2 of {pin, state, city} resolved
       low   -- PIN missing, or state token and PIN's state disagree, or
                almost nothing resolved."""
    if token_state and pin_state and token_state != pin_state:
        return _CONF_LOW
    resolved = sum(bool(x) for x in (out.pin_code, out.state, out.city))
    if pin and out.state and out.city:
        return _CONF_HIGH
    if resolved >= 2:
        return _CONF_MEDIUM
    return _CONF_LOW


# ---------------------------------------------------------------------------
# Back-compat shim
# ---------------------------------------------------------------------------

def split_trailing_location(address: str) -> tuple[str, str, str, str]:
    """Drop-in replacement for onboarding_mapper._split_trailing_location.

    Same 4-tuple contract: (remaining_address, city, state, pin_code). The
    caller (onboarding_mapper._location_fields) only fills a field it did not
    already have, so returning more here is safe -- it just means fewer blanks.
    """
    r = resolve_address_blob(address)
    return r.address_1, r.city, r.state, r.pin_code
