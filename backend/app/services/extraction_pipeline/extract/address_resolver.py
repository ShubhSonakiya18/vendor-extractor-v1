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
    # address_3/address_4 are populated ONLY when resolve_address_blob() is
    # called with multiline=True (vendor path) -- see that function's
    # docstring. Left "" (their dataclass default) on the legacy path, so
    # every existing caller and as_dict()'s 5-key shape stay byte-identical.
    address_3: str = ""
    address_4: str = ""
    city: str = ""
    state: str = ""
    pin_code: str = ""
    confidence: str = _CONF_LOW
    # what each field was resolved from, for review flags / debugging
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, str]:
        """The ORIGINAL 5-key shape, unchanged since before address_3/4
        existed. Every current caller (semantic_engine's legacy path,
        eval_address.py's default mode, split_trailing_location) reads
        exactly these keys -- do not add address_3/4 here; use
        as_dict_full() instead, which is additive."""
        return {
            "address_1": self.address_1,
            "address_2": self.address_2,
            "city": self.city,
            "state": self.state,
            "pin_code": self.pin_code,
        }

    def as_dict_full(self) -> dict[str, str]:
        """as_dict() plus address_3/address_4 -- for the multiline (vendor)
        path only. A separate method rather than conditionally including the
        extra keys in as_dict() itself, so as_dict()'s shape never varies."""
        return {**self.as_dict(), "address_3": self.address_3, "address_4": self.address_4}


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


def _looks_like_premises(segment: str) -> bool:
    """True when a segment is structurally part of the PROPERTY (a unit/floor
    designator, a named building, an industrial estate) or a street/landmark,
    rather than a place name that could be a city.

    Used only to veto the weakest city rule -- the positional "whatever sits
    last" guess. Without it, "..., METRO LOGISTICS PARK, BAWANA INDUSTRIAL
    AREA, DELHI" resolved city to "Bawana Industrial Area", promoting an
    industrial estate into the city field. Localities and keyword-less
    fragments still pass, since those genuinely can be the city.
    """
    try:
        from .address_segmenter import classify_fragment, _UNKNOWN_TIER
    except Exception:       # segmenter data unavailable -- keep old behaviour
        return False
    tier, _level, _conf, _ev = classify_fragment(segment)
    return tier not in (_UNKNOWN_TIER, "locality", "village_po")


def _split_known_city_prefix(segment: str) -> tuple[str, str]:
    """Split "NOIDA SECTOR 62" into ("Noida", "SECTOR 62").

    A single segment can carry the city AND a locality when the source wrote
    them without a comma. Only a KNOWN city name is ever peeled this way, and
    only from the FRONT, so nothing is invented and the remainder keeps its
    original text. Returns ("", segment) when there is no such prefix.

    Guarded to the locality shape specifically: the remainder must classify
    as `locality`/`village_po`/`unknown` (a plausible "which part of the
    city" suffix), not a street/estate/premises shape. Without this guard,
    "GHAZIPUR ROAD" -- a street named after the Ghazipur area, not "the city
    Ghazipur plus a locality" -- had its city-shaped first word peeled off,
    hallucinating city=Ghazipur onto an address whose actual city was
    already known from the text (DELHI, matched and removed earlier by
    _match_state). "NOIDA SECTOR 62" passes the guard because "SECTOR 62"
    classifies as `locality`; "GHAZIPUR ROAD" fails it because "ROAD"
    classifies as `thoroughfare`.
    """
    from .address_segmenter import classify_fragment

    words = segment.split()
    # Longest prefix first, so "NAVI MUMBAI ..." beats a bare "NAVI".
    for cut in range(min(3, len(words) - 1), 0, -1):
        head, tail = " ".join(words[:cut]), " ".join(words[cut:])
        if not tail or not is_known_city(head):
            continue
        tail_tier, _lvl, _conf, _ev = classify_fragment(tail)
        if tail_tier in ("locality", "village_po", "unknown"):
            return head, tail
    return "", segment


def _pick_city(
    segments: list[str],
    pin_district: str,
    have_state_anchor: bool,
    state: str = "",
) -> tuple[list[str], str, bool]:
    """Choose the city token and remove it from `segments`.

    Preference, strongest first:
      1. the last segment that is a known city name (the string's own city
         token wins -- if the document says MOHALI, the city is Mohali even
         when the PIN's district is "S.A.S Nagar")
      2. the last segment that equals the PIN's district
      3. the PIN's own district, verbatim, when nothing in the remaining
         text names a city or the district at all -- e.g. "...ANDUL,
         Natibpur, 711302" where 711302 -> Howrah, but no segment says
         "Howrah" and "Natibpur" is a locality, not a known city. Trusting
         the PIN-directory lookup here is more reliable than guessing from
         position (step 4): a PIN is a hard, verified fact about the
         document; "whatever token sits last" is not. This step introduces
         a district name that was never actually written on the page, so
         the caller should treat it as a lower-confidence resolution than
         steps 1-2 (see resolve_address_blob's confidence scoring) --
         nothing is removed from `segments` for this step, since the
         district text isn't literally present to remove.
      4. if a state/PIN anchor exists, the last remaining segment (the old
         "segment right before the state" behaviour) -- but only if it isn't
         obviously a street/landmark token. This is the weakest signal and
         only fires when steps 1-3 all come up empty (no PIN, or a PIN not
         in the directory).

    After the city is chosen, any *adjacent* trailing segment that just
    repeats the city or names the PIN's district is dropped too -- it is a
    redundant district label ("KOLKATA, KOLKATA" / "MOHALI, S.A.S Nagar"),
    not part of the street.
    Returns (segments_without_city, city_or_'', from_pin_fallback) -- the
    third element is True only for step 3 (the city text was never actually
    on the page), so the caller can note and score it differently.
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

    # Step 3: nothing in the text names the district or a known city, but the
    # PIN told us the district anyway -- use it rather than falling through
    # to the positional guess in step 4. Nothing to remove from `segments`
    # since this text was never there.
    if chosen == -1 and dist_norm:
        return segments, _titlecase_city(pin_district), True

    # Step 3b: a segment that carries the city joined onto a locality with no
    # comma ("NOIDA SECTOR 62"). Peel the known city off the front and keep the
    # remainder as an ordinary fragment. Scanned end-first like step 1, and
    # only tried once steps 1-3 have found nothing to prefer.
    if chosen == -1:
        for i in range(n - 1, -1, -1):
            head, tail = _split_known_city_prefix(segments[i])
            if head:
                rest = segments[:i] + [tail] + segments[i + 1:]
                return rest, _titlecase_city(head), False

    # Step 3c: the state is itself a city (Delhi, Chandigarh, Puducherry). If
    # nothing in the text named a city, the city IS the state. This outranks
    # the positional guess below deliberately: in "..., KONDLI, DELHI" the
    # positional rule would promote the locality KONDLI, when the document has
    # actually told us the city outright.
    if chosen == -1 and state and is_known_city(state):
        return segments, _titlecase_city(state), True

    if chosen == -1 and have_state_anchor and segments:
        last = segments[-1]
        words = last.casefold().split()
        if (
            last.casefold() not in _NON_CITY_TOKENS
            and not last.isdigit()
            and not any(w in _NON_CITY_TOKENS for w in words)
            # An estate/building/street fragment is never the city, however
            # conveniently it sits in the last slot.
            and not _looks_like_premises(last)
        ):
            chosen = n - 1

    if chosen == -1:
        return segments, "", False

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

    return rest, _titlecase_city(city_token), False


def resolve_address_blob(address: str, *, multiline: bool = False) -> ResolvedAddress:
    """Segment one combined address string. See module docstring.

    `multiline` (default False, opt-in): when True, the leftover after the
    pin/state/city/country peel is additionally run through
    `address_segmenter.segment_leftover()` to split it into up to 4 ordered
    lines (address_1..address_4) instead of joining everything into
    address_1 as one string. This is the VENDOR-path behaviour; the customer
    path and every other existing caller keep `multiline=False` so their
    output is byte-identical to before this parameter existed --
    `split_trailing_location()` below always calls with the default.
    """
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
    segments, city, city_from_pin_fallback = _pick_city(
        segments, pin_district, have_anchor, out.state
    )
    if city:
        out.city = city
        if city_from_pin_fallback:
            out.notes.append(f"city: {city} (inferred from PIN's district, not present in the text)")
        else:
            out.notes.append(f"city: {city}")

    resolver_confidence = _score(out, pin, token_state, pin_state, city_from_pin_fallback)

    if multiline and segments:
        from .address_segmenter import _LEVEL_ORDER, segment_leftover

        seg_result = segment_leftover(segments, pin=pin, district=pin_district)
        lines = (seg_result.lines + ["", "", "", ""])[:4]
        out.address_1, out.address_2, out.address_3, out.address_4 = lines
        if seg_result.notes:
            out.notes.append(f"segmenter: {'; '.join(seg_result.notes)}")
        # Overall confidence is the WEAKER of the two independent judgements
        # (resolver's pin/state/city confidence, segmenter's line-grouping
        # confidence) -- a confidently-resolved city/state/pin does not
        # license confidence in the line split, and vice versa.
        out.confidence = min(
            (resolver_confidence, seg_result.confidence),
            key=lambda c: _LEVEL_ORDER[c],
        )
    else:
        out.address_1 = ", ".join(segments).strip(" ,")
        out.confidence = resolver_confidence

    return out


def _score(
    out: ResolvedAddress,
    pin: str,
    token_state: str,
    pin_state: str,
    city_from_pin_fallback: bool = False,
) -> str:
    """high  -- PIN present and (no state token, or it agrees) and a city found
                IN THE TEXT (not inferred purely from the PIN lookup)
       medium -- 2 of {pin, state, city} resolved, OR all three resolved but
                the city was never actually written on the page and only
                came from the PIN-directory district (a real, useful value,
                but a lower-trust one than a city the OCR actually read)
       low   -- PIN missing, or state token and PIN's state disagree, or
                almost nothing resolved."""
    if token_state and pin_state and token_state != pin_state:
        return _CONF_LOW
    resolved = sum(bool(x) for x in (out.pin_code, out.state, out.city))
    if pin and out.state and out.city and not city_from_pin_fallback:
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
