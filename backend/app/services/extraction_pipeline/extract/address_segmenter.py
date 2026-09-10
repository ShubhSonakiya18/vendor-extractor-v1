"""Address LINE segmentation -- address_1 / address_2 / address_3(+overflow_4).

Turns the LEFTOVER of `address_resolver.resolve_address_blob()` -- i.e. every
fragment that is NOT the pin code, state, city or country, which have already
been peeled off before this module ever sees the input -- into up to four
ordered address lines, for the vendor path only.

Why simple comma splitting is insufficient
-------------------------------------------
A raw address blob like

    "3RD FLOOR, PART A BLOCK B, SRIJAN INDUSTRIAL LOGISTIC PARK, MOHIARY,
     CHANDIBAGAN, ANDUL, NATIBPUR"

has 7 comma-separated segments that mix two genuinely different kinds of
information: WHERE INSIDE THE PROPERTY (floor/block/building/estate name) and
WHICH NEIGHBOURHOOD (a run of locality names). `address.split(",")[:N]` or
"first N segments / last N segments" cannot distinguish these -- the boundary
between them moves depending on how many premises-tokens and how many
locality-tokens a given address happens to have, and neither count is fixed.

The chosen algorithm: classify, then find boundaries, then group in place
---------------------------------------------------------------------------
1. **normalize** -- comma/newline-split segments arrive as given from the
   caller (already produced by `address_resolver._segments()`).
2. **explicit boundary detection** -- commas are AUTHORITATIVE. Only when the
   caller hands us a SINGLE comma-less segment (the whole leftover had no
   commas at all) do we attempt to repair missing boundaries -- see
   `inject_boundaries()`. A comma the source document actually wrote is never
   second-guessed.
3. **semantic component classification** -- `classify_fragment()` scores each
   fragment independently, from its own text alone, against the keyword tiers
   in `segmentation_keywords.yaml` (premises_unit, structural, building_name,
   estate_zone, thoroughfare, landmark, locality, village_po). A fragment with
   no keyword match is `unknown`, treated as a likely locality name (see the
   module-level note on why that default is deliberate, not a failure).
4. **logical boundary inference** -- `_gap_strength()` scores the GAP between
   each pair of ADJACENT fragments (never fragments further apart), based on
   whether the semantic level increases, whether the tier changes, and
   whether either side was comma-less-injected. A gap scoring above
   `boundary_strength` (default 0.5, tunable in the YAML) is a cut.
5. **source-order-preserving grouping** -- groups are the CONTIGUOUS runs
   between cuts, in the fragments' original order. There is no bucketing step
   and no sort: fragments are never moved past one another. Concatenating the
   groups back together reproduces the input sequence exactly (see the
   `test_address_segmenter.py` invariant sweep, I3).
6. **packing** -- surviving groups become `address_1`, `address_2`,
   `address_3` in order; a 4th group (rare) becomes `address_4`, used only as
   an overflow/safety valve, never a normal target. If confidence is low, or
   there are more than 4 groups, adjacent groups are MERGED (never dropped)
   starting with the weakest boundary.

Ordering
--------
Fragments are never reordered relative to the source. `"ANDUL, 3RD FLOOR"`
comes back as `address_1="ANDUL"`, `address_2="3RD FLOOR"` -- the unusual
order is preserved because the source document wrote it that way.
Classification is a PURE function of each fragment's own text (no neighbour,
no index) -- see `classify_fragment()` -- which is what makes two differently
-ordered addresses ("3RD FLOOR, BLOCK B, UNIT 302" vs "UNIT 302, BLOCK B,
3RD FLOOR") group SEMANTICALLY identically: same multiset of components per
line, each rendered in its own source order.

Ambiguity and ambiguous fragments
----------------------------------
A fragment matching more than one tier's keywords (e.g. "NEAR GIDC ROAD")
is resolved via a fixed conflict-priority list and its confidence is capped
at 0.5 (see `_CONFLICT_CONFIDENCE_CAP`). Overall confidence is `high` /
`medium` / `low` -- the SAME three-level vocabulary `ResolvedAddress` and
`app/eval/eval_address.py` already use. Low confidence caps the address at 2
lines; any comma-less-injected boundary or any locality-tail-split caps it at
3 -- see `_LINE_CAP` and the explicit caps applied in `segment_leftover()`.
At the degenerate limit (no keyword evidence anywhere, fully injected) the
result collapses to a SINGLE line -- byte-identical to this project's
pre-segmenter behaviour. That is a deliberate safety property: the worst case
of this module is the behaviour that shipped before it existed.

The locality-tail-split -- a heuristic, not a geographic rule
----------------------------------------------------------------
Indian addresses often narrow -> broaden toward the city, so the LAST of a
long run of locality-level fragments is sometimes the village/post-office
tier. `_maybe_tail_split()` peels that last fragment into its own group, but
ONLY when the trailing locality run has at least `min_localities_for_tail_split`
members (default 4) -- so the common 2-3-locality case
("Koramangala, HSR Layout, Bengaluru") is left together, untouched, by
design. This is explicitly a positional guess with no gazetteer behind it in
the default configuration; it is not claimed to generalize to "the last
locality before a city is always administratively distinct," and any address
it fires on is capped at `medium` confidence and flagged
`locality_tail_split_inferred` so it is reviewable, never presented as a
confident claim.

Performance
-----------
Classification is a handful of dict lookups per token; grouping is O(n) over
at most a few dozen fragments. No regex scanning over the full keyword set,
no network, no model. See `test_address_segmenter.py::TestPerformance` for
the measured latency (target: sub-millisecond median per address).

Known limitations
------------------
- No fuzzy/OCR-tolerant keyword matching (a misspelled "FLR" variant that
  isn't in the keyword list falls through to `unknown`) -- deliberate, to
  hold the latency budget and avoid false positives on short locality names.
- Keyword coverage is not exhaustive for every Indian region; extending it is
  a data change in `segmentation_keywords.yaml`, not a code change.
- `address_lookups.is_known_city()`'s city list is district-level (~767
  names), so a town-level city may reach this module as an ordinary locality
  fragment -- an upstream limitation this module cannot detect or correct.
- Comma-less repair is best-effort on proper nouns that embed a keyword
  ("Park Street", "Lake Gardens") -- guarded (see `inject_boundaries()`), but
  any address requiring repair is capped at `medium` confidence, never
  presented as fully trusted.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field as _dc_field
from functools import lru_cache
from typing import Optional

from .address_lookups import segmentation_keywords

# ---------------------------------------------------------------------------
# text normalisation / shape patterns
# ---------------------------------------------------------------------------
_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")
_DESIGNATOR_TOKEN_RE = re.compile(r"^[a-z0-9]{1,3}$")
_DESIGNATOR_FRAGMENT_RE = re.compile(r"^[a-z]?[-/]?\d+[a-z]?([-/]\d+[a-z]?)*$")
_ORDINAL_RE = re.compile(r"^\d+(st|nd|rd|th)$")
_HASH_NO_RE = re.compile(r"^#|^no\.?\s*\d")

_CONFLICT_CONFIDENCE_CAP = 0.5
_UNKNOWN_TIER = "unknown"
# A keyword-less fragment is treated as a locality (see the module docstring on
# why that is positive evidence, not a failure), so `unknown` must sit at
# whatever level the data assigns `locality`. Kept in step with
# segmentation_keywords.yaml by _locality_level(); this constant is only the
# fallback for when the data file is missing entirely.
_UNKNOWN_LEVEL = 4

# The taxonomy groups its levels into two bands (see segmentation_keywords.yaml):
# levels 1-2 describe THE PROPERTY (unit/block/floor designators, then the
# building or estate that names it), levels 3+ describe ITS GEOGRAPHY (street,
# locality, village). _gap_strength() uses this to tell an ordinary
# within-premises step back ("BUILDING D, PART C") from a genuine reordering
# ("KORAMANGALA, MG ROAD"). Tied to the tier levels in that file.
_PREMISES_BAND_MAX = 2

_LEVEL_ORDER = {"low": 0, "medium": 1, "high": 2}
_LINE_CAP = {"high": 4, "medium": 3, "low": 2}


def _locality_level(kw: dict | None) -> int:
    """The level the data assigns to `locality`, which is also the level an
    `unknown` fragment takes. Read from the data rather than hardcoded so a
    future renumbering of the tier levels stays a data-only change."""
    tiers = (kw or {}).get("tiers") or {}
    lvl = (tiers.get("locality") or {}).get("level")
    return lvl if isinstance(lvl, int) else _UNKNOWN_LEVEL



def _word_norm(tok: str) -> str:
    """Casefold and strip ALL non-alphanumeric characters (not just edges) --
    matching address_lookups._norm's existing convention. This is a lookup
    key only; `Fragment.text` always carries the untouched original ("31/1"
    stays "31/1" in the output even though its norm form is "311"). Stripping
    internal punctuation, not just edges, is what lets a keyword like "po"
    match an abbreviation written "P.O." -- an edge-only strip would leave a
    stray internal dot ("p.o") that never matches the plain "po" keyword."""
    return _NON_ALNUM_RE.sub("", tok.casefold())


def _min_level(a: str, b: str) -> str:
    return a if _LEVEL_ORDER[a] <= _LEVEL_ORDER[b] else b


# ---------------------------------------------------------------------------
# data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Fragment:
    text: str
    index: int
    tier: str
    level: int
    confidence: float
    evidence: str
    injected: bool = False


@dataclass
class SegmentedAddress:
    lines: list[str] = _dc_field(default_factory=list)      # up to 4, non-empty, in order
    groups: list[list[Fragment]] = _dc_field(default_factory=list)  # parallel to `lines`
    fragments: list[Fragment] = _dc_field(default_factory=list)     # flat, source order
    confidence: str = "low"
    score: float = 0.0
    notes: list[str] = _dc_field(default_factory=list)

    def get(self, n: int) -> str:
        """1-based line accessor (`get(1)` == address_1); '' past the end."""
        return self.lines[n - 1] if 0 < n <= len(self.lines) else ""


# ---------------------------------------------------------------------------
# classification -- a pure function of each fragment's own text
# ---------------------------------------------------------------------------

def _role_sets(kw: dict, role: str) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for tier, spec in (kw.get("tiers") or {}).items():
        out[tier] = {w.casefold() for w in (spec.get(role) or [])}
    return out


def _flat_role_set(kw: dict, role: str) -> set[str]:
    flat: set[str] = set()
    for words in _role_sets(kw, role).values():
        flat |= words
    return flat


# ---------------------------------------------------------------------------
# OCR word re-segmentation -- one scored search, no per-address rules
# ---------------------------------------------------------------------------
# PaddleOCR on a small, dense combined-address line routinely drops every
# inter-word space: "2ND FLOOR, PART A, TALOJA INDUSTRIAL AREA" comes back as
# "2NDFLOOR,PARTA,TALOJAINDUSTRIALAREA". The comma split still works, but each
# fragment is then a space-less run that matches no keyword, so
# classify_fragment() returns `unknown` for everything and the boundary scorer
# has no evidence -- the whole address collapses onto one line.
#
# `_desegment_segments()` repairs this with a SINGLE dynamic-programming
# word-break over ONE additive scoring function. There is no ordered rule
# cascade, no per-word table, and no address-specific exception; every signal
# is read from `segmentation_keywords.yaml`.
#
#   piece_score(core):
#     designator shape (ordinal / A-1 / roman / single letter)        +2.0 strong
#     keyword, HEAD role, facility/premises tier (level <= 3)          +2.0 strong
#     keyword, TAIL role, facility/premises tier, ending a component   +2.0 strong
#     keyword, locality / village_po tier (level >= 4)                 +0.8 weak
#     keyword matched only as an `any`-phrase word                    +0.8 weak
#     plausible word: >= 4 chars, has a vowel, no >= 4-consonant run   +0.4 plausible
#     anything else (short, no anchor)                                -3.0 sliver
#   seg_score = sum(piece_score) - _DESEG_PER_CUT * (n_pieces - 1)   # Occam bias
#
# A split replaces the run ONLY when ALL of these hold (else return [run]):
#   * at least one piece is a STRONG anchor. Weak evidence alone -- a plausible
#     word, a place-name morpheme ("-gram", "-pur", "-nagar"), an `any`-phrase
#     word -- never justifies a cut, so "GURUGRAM", "KANPUR", "MAHALAXMI",
#     "NEWBARRACKPORE", "SRILAKSHMI" fall through unchanged with no explicit
#     exclusion for any of them;
#   * no cut sits between two same-role keywords (tail|tail or head|head) --
#     the exact guard inject_boundaries() applies to spaced input, so
#     "PARK STREET", "NAGAR ROAD", "LAKE GARDENS", "TOWER ROAD" stay whole;
#   * seg_score beats the no-split score by at least _DESEG_MARGIN. Ties and
#     near-ties resolve to the run untouched (fewer pieces preferred).
#
# This is conservative on purpose: at worst a fragment stays glued (and is
# classified `unknown`, exactly as before this function existed). It is never
# mangled. (I9 conservative-under-low-confidence; I10 safe fallback.)

_VOWEL_RE = re.compile(r"[aeiouy]")
_CONSONANT_RUN_RE = re.compile(r"[bcdfghjklmnpqrstvwxz]{4,}")
# A "designator" is a numbered / lettered unit label. The first two shapes
# carry a digit and are unambiguous anchors anywhere. The last two -- a lone
# letter, a bare roman numeral -- are anchors ONLY when they follow a
# structural/premises head keyword ("BLOCK B", "PHASE II"); standing alone
# they are just one letter of a proper noun and must not drive a split, or the
# DP would shatter every name into single characters.
_DESEG_SHAPES_NUMERIC = (
    re.compile(r"^\d+(?:st|nd|rd|th)$"),                     # ordinal
    re.compile(r"^\d{2,}[a-z]?(?:[-/]\d+[a-z]?)*$"),         # 302, 45/2, 12b
    re.compile(r"^[a-z]-\d+[a-z]?$"),                        # B-14  (letter, HYPHEN, digits)
    re.compile(r"^[a-z]?\d[a-z]$"),                          # 5A, B7 -- single digit between/after a letter
)
# a bare single digit ("BUILDING 4", "TOWER 4") -- a real unit number, but
# WEAK: on its own it is as likely to be one digit of a larger number, so it
# never justifies a cut by itself (a preceding head keyword must carry it).
_DESEG_SHAPE_LONE_DIGIT = re.compile(r"^\d$")
_DESEG_SHAPES_CONTEXTUAL = (
    # a single letter, EXCEPT "s" -- a trailing "S" is the English plural that
    # turns a real word into itself ("TOWERS" -> "TOWER" + "S" would corrupt
    # the word) and is never written as a unit label. "A"/"B"/"C" ARE common
    # block designators and stay in.
    re.compile(r"^[a-rt-z]$"),                                # BLOCK "B", WING "C", PART "A"
    re.compile(r"^(?:ii|iii|iv|vi|vii|viii|ix|xi|xii)$"),     # PHASE "II" (not lone "v"/"x")
)
_DESEG_PER_CUT = 0.5      # Occam penalty per extra piece
_DESEG_MARGIN = 0.75      # a split must beat no-split by at least this
_DESEG_MIN_RUN = 6        # shorter space-free runs are left alone
_DESEG_MAX_PIECE = 18     # longest candidate piece considered
_DESEG_MAX_RUN = 40       # runs longer than this are left whole (enumeration guard)

_DESEG_STRONG = 2.0
_DESEG_WEAK = 0.8
# A plausible word scores ZERO: it is admissible in a cover (it is not a
# sliver) but it is never itself a reason to cut. Only a keyword or a
# designator earns positive credit, so a split has to be "paid for" by a real
# anchor against the _DESEG_PER_CUT Occam penalty -- this is what keeps
# "GREENFIELDBUSINESS" from being chopped into two plausible halves.
_DESEG_PLAUSIBLE = 0.0
_DESEG_SLIVER = -3.0
# Tiers at or below this level name a distinct address component ("ROAD",
# "PARK", "FLOOR"); tiers above it are place-name morphemes ("-GRAM", "-PUR")
# that fuse into one proper noun. Derived from the YAML level numbers, not a
# hardcoded word list -- a future renumbering stays a data-only change.
_DESEG_FACILITY_MAX_LEVEL = 3


@lru_cache(maxsize=1)
def _desegment_vocab() -> dict:
    """casefold keyword word -> (tier, role, level) for every >=3-char word in
    the head / tail / any lists of segmentation_keywords.yaml. Pure data; the
    scorer never enumerates words itself."""
    kw = segmentation_keywords()
    tiers = (kw.get("tiers") or {}).items()
    out: dict[str, tuple[str, str, int]] = {}
    # A word that appears in ANY tier's `tail` list at a locality/village level
    # (> _DESEG_FACILITY_MAX_LEVEL) is a place-name-forming morpheme
    # ("-wadi", "-nagar", "-pur", "-halli"). It fuses into one proper noun far
    # more often than it marks a component boundary -- so even if a LOWER tier
    # also claims it (e.g. "wadi" is in building_name.tail too), it must not
    # anchor an OCR-deglue split. Collected first, consulted in Pass 1.
    morpheme_tails = {
        w.casefold()
        for _t, spec in tiers
        if spec.get("level", _UNKNOWN_LEVEL) > _DESEG_FACILITY_MAX_LEVEL
        for w in (spec.get("tail") or [])
        if len(w) >= 3
    }
    # Pass 1: head/tail roles. These are the strong, positionally-meaningful
    # signals and must win any word that also appears inside a multi-word
    # `any` phrase ("nagar" is a locality TAIL first, even though it also
    # occurs in the estate_zone any-phrase "udyog nagar").
    for tier, spec in tiers:
        level = spec.get("level", _UNKNOWN_LEVEL)
        for role in ("head", "tail"):
            for w in (spec.get(role) or []):
                wl = w.casefold()
                if len(wl) >= 3 and wl not in out:
                    # a morpheme tail is pinned to a locality level so
                    # _deseg_piece's "level > MAX and role == tail -> zero
                    # credit" rule catches it regardless of which tier listed it
                    lv = _UNKNOWN_LEVEL if (role == "tail" and wl in morpheme_tails) else level
                    rl = "tail" if wl in morpheme_tails and role == "tail" else role
                    out[wl] = (tier, rl, lv)
    # Pass 2: words that only ever appear inside an `any` phrase -> weak.
    for tier, spec in tiers:
        level = spec.get("level", _UNKNOWN_LEVEL)
        for phrase in (spec.get("any") or []):
            for w in phrase.split():
                wl = w.casefold()
                if len(wl) >= 3 and wl not in out:
                    out[wl] = (tier, "any", level)
    # Pass 3: connectors ("no", "nos") -- admissible glue between a head
    # keyword and its number ("FLAT NO 302"), scored as a plausible word (zero)
    # so it never drives a cut but also never blocks one as a sliver.
    for w in (kw.get("connectors") or []):
        wl = w.casefold()
        if wl.isalpha() and wl not in out:
            out[wl] = ("_connector", "any", _UNKNOWN_LEVEL)
    return out


def _deseg_is_plausible(core: str) -> bool:
    return (len(core) >= 4
            and bool(_VOWEL_RE.search(core))
            and not _CONSONANT_RUN_RE.search(core))


@lru_cache(maxsize=8192)
def _deseg_piece_cached(core: str, prev_core: str) -> tuple[float, str, str]:
    return _deseg_piece_impl(core, _desegment_vocab(), prev_core)


def _deseg_piece(core: str, vocab: dict, prev_core: str = "") -> tuple[float, str, str]:
    """(score, kind, role) for one candidate piece core (already casefolded,
    alnum-only). `prev_core` is the piece immediately before it in the current
    DP path ("" at the start). kind in strong|weak|plausible|sliver; role is
    the keyword role ("head"/"tail"/"any") or "" when not a keyword. Thin
    memoised wrapper -- `vocab` is always the _desegment_vocab() singleton."""
    return _deseg_piece_cached(core, prev_core)


def _deseg_piece_impl(core: str, vocab: dict, prev_core: str = "") -> tuple[float, str, str]:
    if any(rx.match(core) for rx in _DESEG_SHAPES_NUMERIC):
        # A multi-char unit label ("18", "302", "5A", "B-14") is a strong anchor.
        return _DESEG_STRONG, "strong", ""
    if _DESEG_SHAPE_LONE_DIGIT.match(core):
        return _DESEG_WEAK, "weak", ""
    if any(rx.match(core) for rx in _DESEG_SHAPES_CONTEXTUAL):
        prev = vocab.get(prev_core)
        # a lone letter / roman numeral is a unit label only right after a
        # STRUCTURAL head ("BLOCK B", "WING C", "PHASE II", "TOWER A") -- the
        # tier whose designator is a letter. A premises head ("FLAT", "PLOT",
        # "ROOM") takes a NUMBER, never a bare letter, so "FLAT N..." must not
        # treat "N" as a designator.
        if prev is not None and prev[0] == "structural" and prev[1] == "head":
            return _DESEG_STRONG, "strong", ""
        return _DESEG_SLIVER, "sliver", ""         # a lone letter of a proper noun
    ent = vocab.get(core)
    if ent is not None:
        _tier, role, level = ent
        if len(core) < 4:
            # A 3-char keyword ("tal", "gat", "rs") occurs as a substring
            # inside ordinary place names far too often to carry any weight --
            # the same reason the earlier de-glue vocab excluded <4-char
            # words. Score it as a plausible word (zero), never an anchor.
            return _DESEG_PLAUSIBLE, "plausible", ""
        if level > _DESEG_FACILITY_MAX_LEVEL and role == "tail":
            # A locality / village TAIL morpheme ("nagar", "halli", "pur").
            # These FUSE into one proper noun far more often than they mark a
            # component boundary -- zero credit, so "BOMMANA|HALLI" cannot
            # outscore the whole "BOMMANAHALLI". (Locality HEAD words like
            # "sector"/"pocket" are real anchors and fall through below.)
            return _DESEG_PLAUSIBLE, "plausible", ""
        if role == "any":
            return _DESEG_WEAK, "weak", role       # multi-word acronym / phrase word
        return _DESEG_STRONG, "strong", role       # head/tail keyword, real component
    if len(core) < 3:
        return _DESEG_SLIVER, "sliver", ""         # too short to be a word or a name part
    if _deseg_is_plausible(core):
        return _DESEG_PLAUSIBLE, "plausible", ""
    return _DESEG_SLIVER, "sliver", ""


@lru_cache(maxsize=1)
def _deseg_role_index() -> dict:
    """casefold word -> set of every head/tail role it holds ANYWHERE in the
    taxonomy. A word that is both a head and a tail somewhere ("tower",
    "annexe") is positionally ambiguous and the adjacency guard treats it
    conservatively on both sides."""
    kw = segmentation_keywords()
    idx: dict[str, set[str]] = {}
    for _tier, spec in (kw.get("tiers") or {}).items():
        for role in ("head", "tail"):
            for w in (spec.get(role) or []):
                wl = w.casefold()
                if len(wl) >= 3:
                    idx.setdefault(wl, set()).add(role)
    return idx


@lru_cache(maxsize=4096)
def _deseg_edge_roles(core: str, which: str) -> frozenset:
    """The set of head/tail roles carried by the keyword at one end of a piece
    -- either the whole piece IS such a keyword, or it starts/ends with one as
    a morpheme ("GREENPARK" ends with tail:park). Empty when neither end is a
    keyword. Memoised: called in the DP inner loop."""
    idx = _deseg_role_index()
    if core in idx:
        return frozenset(idx[core])
    roles: set[str] = set()
    for w, rs in idx.items():
        if len(w) >= len(core):
            continue
        if which == "end" and core.endswith(w):
            roles |= rs
        elif which == "start" and core.startswith(w):
            roles |= rs
    return frozenset(roles)


def _deseg_same_role_block(left_core: str, right_core: str, vocab: dict) -> bool:
    """True when a cut between these two pieces is disallowed:
      * they share a head/tail keyword role ("PARK|STREET" tail&tail,
        "GREENPARK|EXTENSION" tail&tail, "TOWER|ROAD" -- tower is head&tail,
        road is tail); mirrors the adjacency guard in inject_boundaries();
      * both pieces are all-digit ("3|02") -- almost always one number OCR
        broke, never two adjacent unit labels.
    """
    if left_core.isdigit() and right_core.isdigit():
        return True
    return bool(_deseg_edge_roles(left_core, "end") & _deseg_edge_roles(right_core, "start"))


def _deseg_has_strong(cores: list[str], vocab: dict) -> bool:
    """At least one STRONG anchor in the chosen segmentation, positioned the
    way its role implies:

      * a TAIL keyword is a real component tail only when it ENDS the
        segmentation or is followed by another anchor -- one sitting mid-run in
        front of a non-anchor ("MAHAL" before "AXMI") is a fused prefix
        morpheme, so "MAHALAXMI" yields no strong anchor and stays whole;
      * a HEAD keyword is a real component head only when it STARTS the
        segmentation or is preceded by an anchor -- one at the END in front of
        nothing ("GALA" after "KORAMAN") is a fused suffix, so "KORAMANGALA"
        stays whole;
      * a designator shape is always a strong anchor.
    """
    n = len(cores)
    for idx, core in enumerate(cores):
        prev = cores[idx - 1] if idx else ""
        _sc, kind, role = _deseg_piece(core, vocab, prev)
        if kind != "strong":
            continue
        if role == "tail":
            if idx == n - 1:
                return True
            nxt = cores[idx + 1]
            _nsc, nkind, nrole = _deseg_piece(nxt, vocab, core)
            # a following anchor, a following head, or a bare unit number
            # ("BUILDING|4", "TOWER|C") all confirm this tail is a real
            # component tail rather than a fused prefix.
            if nkind == "strong" or nrole == "head" or nxt.isdigit():
                return True
        elif role == "head":
            if idx == 0:
                return True
            _psc, pkind, _pr = _deseg_piece(prev, vocab, cores[idx - 2] if idx >= 2 else "")
            if pkind == "strong":
                return True
        else:                                   # designator (role == "")
            return True
    return False


def _deseg_effective_score(core: str, prev_core: str, next_core: str,
                           is_first: bool, is_last: bool, vocab: dict) -> float:
    """The score one piece contributes to a cover, WITH the positional test a
    keyword must pass to earn its +STRONG: a TAIL keyword keeps its +STRONG
    only when it ends the cover or is followed by an anchor / a head / a bare
    number ("BUILDING|4"); a HEAD keyword only when it starts the cover or is
    preceded by a strong anchor. A keyword landing mid-name in front of /
    behind an ordinary word ("HALL" inside "BOMMANAHALLI", "GALA" after
    "KORAMAN") is demoted to zero, so a mis-aligned cover cannot outscore the
    clean one. Both neighbours are needed, so the DP applies this only once a
    piece's successor is known (and once more at the end for the last piece)."""
    sc, kind, role = _deseg_piece(core, vocab, prev_core)
    if kind == "strong" and role == "tail":
        _n_sc, nkind, nrole = (_deseg_piece(next_core, vocab, core)
                               if next_core else (0.0, "", ""))
        if not (is_last or nkind == "strong" or nrole == "head" or next_core.isdigit()):
            return _DESEG_PLAUSIBLE
    elif kind == "strong" and role == "head":
        _p_sc, pkind, _pr = (_deseg_piece(prev_core, vocab, "")
                             if prev_core else (0.0, "", ""))
        if not (is_first or pkind == "strong"):
            return _DESEG_PLAUSIBLE
    return sc


def _desegment_run(run: str, vocab: dict) -> list[str]:
    """Split one space-free run into keyword/designator-anchored pieces, or
    return [run] unchanged. See the section comment for the full rule set.

    A dynamic-programming word-break maximises the cover score (see
    _deseg_effective_score) over all sliver-free covers. The DP state carries
    the previous piece so a keyword's +STRONG is credited only once its
    successor is known and the positional test (tail last / before an anchor;
    head first / after an anchor) can be applied -- a keyword landing mid-name
    earns nothing, so a mis-aligned cover never outscores the clean one. Take
    the split only when the best cover has >= 2 pieces, holds a
    properly-positioned strong anchor, and beats leaving the run whole by at
    least _DESEG_MARGIN.
    """
    n = len(run)
    if n < _DESEG_MIN_RUN or n > _DESEG_MAX_RUN:
        return [run]

    def core_of(a: int, b: int) -> str:
        return _NON_ALNUM_RE.sub("", run[a:b].casefold())

    # dp[i] maps a compact prune key -> best state for covering run[:i]. A
    # state is (settled_score, last_core, prev_core, pieces) where
    # `settled_score` is the cover score for pieces[:-1] fully credited
    # (positional bonus in, per-cut penalties out) but EXCLUDING the last
    # piece's own score -- that needs the last piece's successor, so it is
    # added when we extend past i, or once at the end. The prune key collapses
    # paths that are interchangeable for all future scoring: position `i`, the
    # last piece's core (its pending score + the adjacency guard read it), and
    # whether that last piece is a strong anchor (all a FOLLOWING head
    # keyword's positional test needs from its predecessor).
    def prune_key(pos: int, last_core: str, first: bool) -> tuple:
        _s, k, _r = _deseg_piece(last_core, vocab, "") if last_core else (0.0, "", "")
        return (pos, last_core, k == "strong", first)

    dp: dict[int, dict] = {0: {(0, "", False, True): (0.0, "", "", [])}}
    for i in range(0, n):
        if i not in dp:
            continue
        for (settled, last_core, prev_core, pieces) in list(dp[i].values()):
            for k in range(i + 1, min(n, i + _DESEG_MAX_PIECE) + 1):
                core = core_of(i, k)
                if not core:
                    continue
                _sc, kind, _r = _deseg_piece(core, vocab, last_core)
                if kind == "sliver":
                    continue
                if last_core and _deseg_same_role_block(last_core, core, vocab):
                    continue
                if pieces:
                    add = _deseg_effective_score(
                        last_core, prev_core, core,
                        is_first=(len(pieces) == 1), is_last=False, vocab=vocab)
                    new_settled = settled + add - _DESEG_PER_CUT
                else:
                    new_settled = settled
                new_pieces = pieces + [run[i:k]]
                key = prune_key(k, core, len(new_pieces) == 1)
                slot = dp.setdefault(k, {}).get(key)
                if slot is None or new_settled > slot[0]:
                    dp[k][key] = (new_settled, core, last_core, new_pieces)

    best_score = float("-inf")
    best_pieces: list[str] | None = None
    for (settled, last_core, prev_core, pieces) in dp.get(n, {}).values():
        if not pieces:
            continue
        final = settled + _deseg_effective_score(
            last_core, prev_core, "",
            is_first=(len(pieces) == 1), is_last=True, vocab=vocab)
        if final > best_score:
            best_score, best_pieces = final, pieces

    if not best_pieces or len(best_pieces) < 2:
        return [run]
    cores = [_NON_ALNUM_RE.sub("", p.casefold()) for p in best_pieces]
    if not _deseg_has_strong(cores, vocab):
        return [run]
    no_split = _deseg_piece(_NON_ALNUM_RE.sub("", run.casefold()), vocab)[0]
    if best_score < no_split + _DESEG_MARGIN:
        return [run]
    return best_pieces


def _desegment_fragment(fragment: str, vocab: dict) -> str:
    parts: list[str] = []
    changed = False
    for chunk in fragment.split():
        pieces = _desegment_run(chunk, vocab)
        if pieces != [chunk]:
            changed = True
        parts.extend(pieces)
    return " ".join(parts) if changed else fragment


def _desegment_segments(segments: list[str]) -> list[str]:
    """Re-insert OCR-dropped spaces across a whole address, conservatively.
    A no-op on already-spaced input and on any run without a strong keyword or
    designator anchor. See the section comment above for the full rule set and
    safety argument."""
    if not segmentation_keywords():
        return segments
    vocab = _desegment_vocab()
    if not vocab:
        return segments
    return [_desegment_fragment(s, vocab) for s in segments]


def classify_fragment(text: str) -> tuple[str, int, float, str]:
    """(tier, level, confidence, evidence). Depends ONLY on `text` -- no
    neighbour, no index, no position -- so permuting a set of fragments never
    changes any individual fragment's classification (see the module
    docstring's "Ordering" section)."""
    kw = segmentation_keywords()
    tokens = text.split()
    if not kw or not tokens:
        return (_UNKNOWN_TIER, _UNKNOWN_LEVEL, 0.55, "no_data" if not kw else "empty")

    tiers = kw.get("tiers") or {}
    norm_tokens = [_word_norm(t) for t in tokens]
    norm_fragment = " ".join(norm_tokens)
    first, last = norm_tokens[0], norm_tokens[-1]

    matches: list[tuple[str, float, str]] = []
    for tier, spec in tiers.items():
        head = {w.casefold() for w in (spec.get("head") or [])}
        tail = {w.casefold() for w in (spec.get("tail") or [])}
        anyphr = [w.casefold() for w in (spec.get("any") or [])]
        if first in head:
            matches.append((tier, 1.0, f"head:{first}"))
        if last in tail:
            matches.append((tier, 1.0, f"tail:{last}"))
        for phrase in anyphr:
            if phrase and phrase in norm_fragment:
                matches.append((tier, 0.9, f"any:{phrase}"))

    # OCR fallback: the same keywords, but matched as a PREFIX/SUFFIX of a
    # merged token rather than requiring a whole space-delimited token to
    # equal the keyword. PaddleOCR on a long, small combined-address line
    # routinely drops inter-word spaces ("5TH FLOOR, GREENFIELD BUSINESS
    # PARK" reads as "5THFLOOR, GREENFIELDBUSINESSPARK") -- sometimes merging
    # every word of a fragment, sometimes only some of them, unpredictably.
    # Without this every such fragment fell through to `unknown`, giving the
    # boundary scorer no evidence at all: the whole address collapsed onto
    # address_1.
    #
    # Tried token-by-token (not just when the WHOLE fragment is one token),
    # since a fragment can have some words merged and others still space-
    # separated ("GREENFIELD BUSINESSPARK"). Only for keywords of at least 4
    # characters, and only requiring the CANDIDATE token itself to be at
    # least 5: a merged fragment will contain many 3-letter substrings by
    # chance ("sec", "pur", "ext" all appear inside ordinary place names with
    # no relation to the keyword), and a false match here corrupts
    # classification instead of just missing it. Confidence is capped below
    # the exact-match values (0.75, vs 1.0/0.9) so this evidence is visibly
    # weaker and easy to filter out in review.
    if not matches:
        head_kw = {
            t: {w.casefold() for w in (s.get("head") or []) if len(w) >= 4}
            for t, s in tiers.items()
        }
        tail_kw = {
            t: {w.casefold() for w in (s.get("tail") or []) if len(w) >= 4}
            for t, s in tiers.items()
        }
        for tok in norm_tokens:
            if len(tok) < 5:
                continue
            for tier in tiers:
                for w in head_kw[tier]:
                    if tok.startswith(w) and len(tok) > len(w):
                        matches.append((tier, 0.75, f"head_merged:{w}"))
                for w in tail_kw[tier]:
                    if tok.endswith(w) and len(tok) > len(w):
                        matches.append((tier, 0.75, f"tail_merged:{w}"))
            if matches:
                break

    if not matches:
        joined = "".join(norm_tokens)
        if len(tokens) == 1 and _ORDINAL_RE.match(norm_tokens[0]):
            return ("premises_unit", 1, 0.85, "shape:ordinal")
        if _HASH_NO_RE.match(text.strip().casefold()):
            return ("premises_unit", 1, 0.80, "shape:hash_no")
        if _DESIGNATOR_FRAGMENT_RE.match(joined) or (
            len(tokens) == 1 and _DESIGNATOR_FRAGMENT_RE.match(norm_tokens[0])
        ):
            return ("premises_unit", 1, 0.70, "shape:designator")
        # "<STRUCTURE WORD> <short designator>" -- "BUILDING D", "COMPLEX B".
        # Words like `building` are carried as TAIL keywords ("SUNRISE
        # BUILDING"), but Indian industrial addresses also use them as HEADS
        # with a bare designator, exactly like "BLOCK B" or "TOWER 3". Without
        # this the fragment matches nothing and defaults to `unknown` at
        # locality level, which put "BUILDING D" three levels above the
        # "PART C, UNIT NO. 12" that follows it and cut a boundary straight
        # through one premises description.
        #
        # Handled as a SHAPE rule rather than by adding `building` to a head
        # list: a word sitting in both a head list and a tail list makes
        # inject_boundaries() propose a cut between the word and its own
        # designator (the documented "UNIT 7" -> "UNIT" | "7" bug), so the
        # keyword lists deliberately stay collision-free.
        if len(tokens) == 2 and _DESIGNATOR_TOKEN_RE.match(norm_tokens[1]):
            if first in _flat_role_set(kw, "tail"):
                return ("structural", 1, 0.80, f"shape:designated:{first}")
        th = kw.get("thresholds", {})
        return (_UNKNOWN_TIER, _UNKNOWN_LEVEL, th.get("unknown_fragment_confidence", 0.55), "default:locality")

    tier_set = {m[0] for m in matches}
    if len(tier_set) == 1:
        tier = next(iter(tier_set))
        conf = max(m[1] for m in matches)
        ev = "|".join(sorted({m[2] for m in matches}))
    else:
        priority = kw.get("conflict_priority") or []
        tier = next((t for t in priority if t in tier_set), sorted(tier_set)[0])
        conf = _CONFLICT_CONFIDENCE_CAP
        ev = "ambiguous:" + "|".join(sorted(tier_set))

    level = (tiers.get(tier) or {}).get("level", _UNKNOWN_LEVEL)
    return (tier, level, conf, ev)


# ---------------------------------------------------------------------------
# comma-less boundary injection -- ONLY used on a single comma-less segment
# ---------------------------------------------------------------------------

def inject_boundaries(segment: str, kw: Optional[dict] = None) -> list[tuple[str, bool]]:
    """Re-tokenise a single comma-less segment into (text, injected) parts.

    PURE re-tokenisation: only inserts boundaries, never edits, drops,
    reorders or normalises text -- `" ".join(tokens(result)) == tokens(segment)`
    always holds (asserted in tests as invariant I9). Cuts BEFORE a `head`
    keyword and AFTER a `tail` keyword, with adjacency guards so a compound of
    two same-role keywords ("PARK STREET", "NAGAR ROAD", "LAKE GARDENS")
    is never split -- this is what stops generic words from shredding proper
    names, per this module's design brief.
    """
    kw = kw if kw is not None else segmentation_keywords()
    tokens = segment.split()
    n = len(tokens)
    if not kw or n < 2:
        return [(segment, False)]

    norm = [_word_norm(t) for t in tokens]
    head_set = _flat_role_set(kw, "head")
    tail_set = _flat_role_set(kw, "tail")
    connectors = {c.casefold() for c in (kw.get("connectors") or [])}
    max_designator = kw.get("thresholds", {}).get("max_designator_tokens", 4)

    cut_after: set[int] = set()
    for i, w in enumerate(norm):
        if w in tail_set and i < n - 1:
            nxt = norm[i + 1]
            if nxt not in tail_set:
                cut_after.add(i)
        if w in head_set:
            if i > 0:
                prev = norm[i - 1]
                if prev not in head_set and prev not in connectors:
                    cut_after.add(i - 1)
            # designator absorption: PART A -> absorb "A"; BLOCK B -> absorb
            # "B"; cut immediately after the absorbed run so "BLOCK B" and
            # whatever follows it split apart.
            e, j, absorbed = i, i + 1, 0
            while (
                j < n
                and absorbed < max_designator
                and _DESIGNATOR_TOKEN_RE.match(norm[j])
                and norm[j] not in head_set
                and norm[j] not in tail_set
            ):
                e = j
                j += 1
                absorbed += 1
            if e > i and e < n - 1:
                cut_after.add(e)

    if not cut_after:
        return [(segment, False)]

    parts: list[str] = []
    start = 0
    for idx in sorted(cut_after):
        if idx + 1 <= start:
            continue
        chunk = tokens[start:idx + 1]
        if chunk:
            parts.append(" ".join(chunk))
        start = idx + 1
    tail_chunk = tokens[start:]
    if tail_chunk:
        parts.append(" ".join(tail_chunk))

    parts = [p for p in parts if p.strip(" ,")]
    if len(parts) <= 1:
        return [(segment, False)]
    return [(p, True) for p in parts]


# ---------------------------------------------------------------------------
# boundary inference + grouping
# ---------------------------------------------------------------------------

def _gap_strength(a: Fragment, b: Fragment) -> float:
    """Score the gap between two adjacent fragments. >= threshold means cut.

    A level change always implies a tier change in this taxonomy (distinct
    levels come from distinct tiers), so both weights are counted together.

    Level INCREASE (narrow -> broad) is the common, expected direction. Its
    weight is graded by distance: a one-level step scores 0.5, a longer step
    0.6. The grading matters at pack time, not at cut time -- both clear the
    0.5 threshold, but when the confidence cap leaves fewer lines than groups
    the shallowest boundary is the one the merge loop gives back first.

    Level DECREASE (broad -> narrow) is unusual but real: OCR and manual entry
    do write addresses out of order. It is scored 0.2 per level stepped back,
    capped at three levels, so the size of the anomaly decides:

        1 level   "BUILDING D, PART C"   0.2 + 0.2 tier = 0.4  -> no cut,
                  ordinary within-premises detail
        3 levels  "ANDUL, 3RD FLOOR"     0.6 + 0.2 tier = 0.8  -> cut,
                  a genuinely out-of-order address

    A flat 0.4 decrease weight (used until 2026-09-10) could not tell those
    apart and isolated "BUILDING D" on a line of its own. Order is always
    preserved either way -- see the module docstring's "Ordering" section --
    but that does not mean adjacent out-of-order components should merge."""
    strength = 0.0
    delta = b.level - a.level
    if delta > 0:
        # A single-level step (premises -> the named property it sits in) is a
        # real boundary but the SHALLOWEST one, so it scores just at threshold
        # and is the first to give way when there are more groups than lines.
        # That is what lets an address with a long locality run pack
        # premises+property onto one line while an address with room to spare
        # keeps them apart -- see the merge loop in segment_leftover().
        strength += 0.5 if delta == 1 else 0.6
    elif delta < 0:
        # A step back WITHIN the premises band (levels 1-2: the unit/block/floor
        # designators and the building they name) is ordinary detail, not a
        # reordering -- "BUILDING D, PART C" is one premises description and
        # must not cut. Any other step back crosses from geography into
        # premises, or reorders within geography, and is a real boundary:
        # "KORAMANGALA, MG ROAD" and "ANDUL, 3RD FLOOR" both cut.
        within_premises = a.level <= _PREMISES_BAND_MAX and b.level <= _PREMISES_BAND_MAX
        strength += 0.2 if within_premises else 0.4
    if b.tier != a.tier:
        strength += 0.2
    if a.injected or b.injected:
        strength -= 0.15
    return max(0.0, strength)


def _initial_groups(fragments: list[Fragment], threshold: float) -> list[list[Fragment]]:
    if not fragments:
        return []
    groups: list[list[Fragment]] = [[fragments[0]]]
    for i in range(1, len(fragments)):
        prev, cur = fragments[i - 1], fragments[i]
        if _gap_strength(prev, cur) >= threshold:
            groups.append([cur])
        else:
            groups[-1].append(cur)
    return groups


def _maybe_tail_split(
    groups: list[list[Fragment]], min_run: int, locality_level: int = _UNKNOWN_LEVEL
) -> tuple[list[list[Fragment]], bool]:
    """Peel the LAST fragment of a trailing locality-level run in
    the final group into its own group -- ONLY when that run has at least
    `min_run` members AND at least one other group exists before it. See the
    module docstring's "locality-tail-split" section: this is a heuristic,
    not a geographic rule, and the common 2-3-locality case is deliberately
    left untouched.

    The `len(groups) > 1` requirement is a deliberate, additional
    conservatism check, not just the count threshold: a bare list of 4+
    locality-like names with NOTHING else in the address (no premises,
    street, or estate content establishing a "head" for the address) reads
    as a set of co-equal area names, not a narrow-to-broad progression --
    there is no signal here that the LAST one specifically is
    administratively distinct from the other three, so this heuristic does
    not guess. It only fires when an earlier, already-distinguished group
    exists, giving the trailing run something to narrow FROM. This is what
    keeps a bare "Andheri, Vile Parle, Santacruz, Bandra" (four co-equal
    Mumbai localities, no other content) together as one line, while the
    worked example (which has real premises content before its 4-locality
    run) still splits -- see test_address_segmenter.py's dedicated
    "localities stay together" cases, including a 4-fragment one."""
    if len(groups) <= 1:
        return groups, False
    last = groups[-1]
    run_len = 0
    for f in reversed(last):
        if f.level >= locality_level:
            run_len += 1
        else:
            break
    if run_len < min_run:
        return groups, False
    remainder, peeled = last[:-1], last[-1:]
    if not remainder:
        return groups, False
    return groups[:-1] + [remainder, peeled], True


def _confidence_level(score: float, kw: dict) -> str:
    th = kw.get("thresholds", {})
    hi = th.get("confidence_high", 0.70)
    lo = th.get("confidence_medium", 0.45)
    if score >= hi:
        return "high"
    if score >= lo:
        return "medium"
    return "low"


# ---------------------------------------------------------------------------
# public entry point
# ---------------------------------------------------------------------------

def segment_leftover(segments: list[str], *, pin: str = "", district: str = "") -> SegmentedAddress:
    """Segment the leftover fragments (post pin/state/city/country peel) into
    up to four ordered address lines. See the module docstring for the full
    algorithm, ordering guarantee, and confidence model.

    `pin` / `district` are accepted for interface symmetry with
    `address_resolver` and future gazetteer-backed refinements, but are not
    currently used by the default (positional) tail-split -- see the module
    docstring.
    """
    kw = segmentation_keywords()
    notes: list[str] = []

    clean_segments = [(s or "").strip() for s in segments]
    clean_segments = [s for s in clean_segments if s]
    if not clean_segments:
        return SegmentedAddress(confidence="low", notes=["no address lines"])

    # Repair OCR-glued fragments ("2NDFLOOR" -> "2ND FLOOR") before anything
    # downstream tries to classify them. Deterministic, keyword-driven,
    # per-fragment: each space-free run is re-segmented only when a confident,
    # strong-anchored, Occam-beating split exists (see _desegment_run) -- a
    # lone long single-word locality in a clean address scores no split and is
    # left untouched. Recorded in notes when it fired, so a reviewer sees the
    # input was noisy.
    deglued = _desegment_segments(clean_segments)
    if deglued != clean_segments:
        notes.append("ocr_deglue_applied")
        clean_segments = deglued

    th = kw.get("thresholds", {}) if kw else {}
    min_tokens_for_injection = th.get("min_tokens_for_injection", 4)
    boundary_strength = th.get("boundary_strength", 0.5)
    min_run = th.get("min_localities_for_tail_split", 4)

    # Commas are authoritative: comma-less repair is only ever attempted when
    # the WHOLE leftover arrived as a single segment (i.e. the source string
    # had zero commas in this part at all) -- never on an individual piece
    # that a real comma already delimited from its neighbours. (An earlier
    # revision also injected into de-glued fragments to recover OCR-dropped
    # commas, but inject_boundaries cannot tell a dropped comma from two
    # keywords that legitimately sit adjacent in one name -- "RING ROAD
    # INDUSTRIAL COMPLEX" -- so it did more harm than good and was reverted.
    # An OCR-merged fragment is left whole; the address is still segmented,
    # just occasionally one group coarser than ideal.)
    comma_less_mode = len(clean_segments) == 1

    fragments: list[Fragment] = []
    idx = 0
    for seg in clean_segments:
        if comma_less_mode and kw and len(seg.split()) >= min_tokens_for_injection:
            parts = inject_boundaries(seg, kw)
        else:
            parts = [(seg, False)]
        for text, injected in parts:
            text = text.strip()
            if not text:
                continue
            tier, level, conf, evidence = classify_fragment(text)
            if injected:
                conf = max(0.0, conf - 0.10)
                evidence = f"{evidence}+injected"
            fragments.append(Fragment(text, idx, tier, level, conf, evidence, injected))
            idx += 1

    if not fragments:
        return SegmentedAddress(confidence="low", notes=["no address lines"])

    groups = _initial_groups(fragments, boundary_strength)

    any_injected = any(f.injected for f in fragments)
    all_unknown = all(f.tier == _UNKNOWN_TIER for f in fragments)

    base_score = sum(f.confidence for f in fragments) / len(fragments)
    level = _confidence_level(base_score, kw or {})
    if any_injected:
        level = _min_level(level, "medium")
        notes.append("boundaries_inferred")
    if any_injected and all_unknown:
        level = "low"

    tail_split_applied = False
    peeled_group_id: Optional[int] = None
    if level != "low":
        groups, tail_split_applied = _maybe_tail_split(
            groups, min_run, _locality_level(kw)
        )
        if tail_split_applied:
            level = _min_level(level, "medium")
            notes.append("locality_tail_split_inferred")
            # Remember which group the split peeled off. Its boundary is
            # between two same-tier, same-level locality fragments, so
            # _gap_strength scores it 0.0 -- it would ALWAYS be picked as the
            # weakest and the merge below would silently undo the split it was
            # just asked to make. Protected explicitly rather than by giving
            # the peel a fake gap score, so the scoring function stays a pure
            # function of the two fragments it is given.
            peeled_group_id = id(groups[-1])

    cap = min(_LINE_CAP[level], 4)
    while len(groups) > cap:
        weakest_i, weakest_val = 0, None
        for i in range(len(groups) - 1):
            if peeled_group_id is not None and id(groups[i + 1]) == peeled_group_id:
                continue
            val = _gap_strength(groups[i][-1], groups[i + 1][0])
            if weakest_val is None or val < weakest_val:
                weakest_val, weakest_i = val, i
        if weakest_val is None:
            # Every remaining boundary is the protected one; give it up rather
            # than loop forever or drop content.
            weakest_i = len(groups) - 2
        groups[weakest_i] = groups[weakest_i] + groups[weakest_i + 1]
        del groups[weakest_i + 1]
        notes.append(f"merged_group_at_{weakest_i}")

    lines = [", ".join(f.text for f in g) for g in groups]
    return SegmentedAddress(
        lines=lines, groups=groups, fragments=fragments,
        confidence=level, score=round(base_score, 3), notes=notes,
    )
