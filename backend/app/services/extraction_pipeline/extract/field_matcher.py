"""
V2 Field Matcher
================
Generates scored candidate values for one field within one document.

Two ways a value is found, both needed on real documents:

  inline    caption and value share a span
            'PIN Code: 700019', 'RTGS / NEFT /IFS Code:ICIC0006278'
  adjacent  caption is its own span and the value sits beside or below it
            'A/c No.' | '627851000539'

and one fallback:

  pattern-only   a span matches the field's pattern with no caption anywhere
                 near it. Scored low on purpose -- it is evidence, not proof.

Scoring blends label similarity, spatial proximity, pattern compatibility and
OCR confidence using the weights from field_dictionary.yaml. Nothing here is
field-specific; a new field behaves correctly by virtue of its YAML entry.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field as dc_field
from typing import Optional

from rapidfuzz import fuzz

from ..config_loader import FieldSpec
from ..ingest.layout_engine import DocumentLayout, PageLayout
from ..models import TextSpan
from .normalizer import clean_label, normalize, strip_value_prefix

# A candidate value that is itself a caption for some other field is almost
# certainly a mis-read of the layout -- penalise rather than exclude, because
# a genuine value can coincide with a caption word ("Current" is both).
LABEL_LIKE_PENALTY = 0.55

# A free-text field whose value has the exact shape of some *other* field is
# a layout mis-read: "Bank Name = BKID0004035" is an IFSC that happened to sit
# near the word "Bank".
FOREIGN_PATTERN_PENALTY = 0.40

# Pattern-only candidates get a nominal label score so that a labelled match
# always outranks an unlabelled one of equal pattern quality.
PATTERN_ONLY_LABEL_SCORE = 0.0

# How much each way of finding a value is worth beyond its component score.
# Inline ranks highest deliberately: when a caption and its value occupy one
# span ("State: West Bengal") the pairing is certain, whereas an adjacent span
# is an inference about layout that can be wrong.
KIND_BONUS = {
    "inline": 0.30,
    "adjacent_same_line": 0.18,
    "adjacent": 0.0,
    "pattern_only": -0.10,
}

# Ranking of match kinds, used to break score ties deterministically.
KIND_RANK = {"inline": 0, "adjacent": 1, "pattern_only": 2}

# Very short values are rarely real answers for a free-text field and are
# usually OCR debris ("Rvic") sitting next to a caption.
MIN_PLAUSIBLE_TEXT_LEN = 3

# Punctuation a form uses between a caption and its value.
#
# "." is deliberately EXCLUDED from the character class and handled by
# _is_separator() below instead of folding it in here. A bare "." is
# ambiguous in a way the other separator characters are not: "State. West
# Bengal" (an OCR-misread colon) is a real caption+value split, but "FLAT
# NO. 302" is an abbreviation period followed by the value's own leading
# digit, not a separator at all. Matching "." unconditionally here let a
# free-text field caption itself (matched via a fuzzy/inline hit on its own
# opening words, e.g. "Flat No" matching inside "FLAT NO. 302, BLOCK C, ...")
# truncate to whatever followed that period -- address_1 came back as "302"
# instead of the whole address. See _is_separator's docstring.
_SEPARATOR_RE = re.compile(r"^\s*[:\-–—=|>]")
_ABBREVIATION_DOT_RE = re.compile(r"^\s*\.\s*\d")


@dataclass
class Candidate:
    """One possible value for one field, with a full audit trail."""

    field_key: str
    value: str
    raw_text: str
    score: float
    components: dict[str, float]
    source_document: str
    page: int
    span: TextSpan
    matched_label: Optional[str] = None
    match_kind: str = "pattern_only"   # inline | adjacent | pattern_only
    direction: str = ""
    pattern_matched: bool = False
    notes: list[str] = dc_field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "value": self.value,
            "score": round(self.score, 4),
            "components": {k: round(v, 4) for k, v in self.components.items()},
            "source_document": self.source_document,
            "page": self.page,
            "matched_label": self.matched_label,
            "match_kind": self.match_kind,
            "direction": self.direction,
            "pattern_matched": self.pattern_matched,
            "bbox": [round(v, 1) for v in self.span.bbox.as_list()],
            "raw_text": self.raw_text,
            "notes": self.notes,
        }


class FieldMatcher:
    """Finds candidates for every configured field in one document."""

    def __init__(self, specs: list[FieldSpec], all_labels: set[str]):
        self.specs = specs
        # Every caption known to the dictionary, used to spot when a candidate
        # value is actually the next caption along.
        self.all_labels = all_labels
        # Every field's shape, so a free-text field can notice it has captured
        # something that clearly belongs to a different field.
        self._foreign_patterns = [(s.key, s.patterns) for s in specs if s.patterns]

    @staticmethod
    def _min_similarity(spec: FieldSpec, label: str) -> float:
        """Short captions need to match almost exactly.

        A 4-character caption like "Town" or "Bank" clears an 82% fuzzy
        threshold against a great deal of unrelated text, and every false
        caption hit drags an arbitrary neighbouring span in as a value. Longer
        captions carry enough signal to tolerate OCR damage.
        """
        base = spec.label_match.min_similarity
        n = len(label)
        if n <= 4:
            return max(base, 0.97)
        if n <= 6:
            return max(base, 0.92)
        return base

    # -- public -------------------------------------------------------------

    def match_document(
        self, layout: DocumentLayout, doc_type: str = "other"
    ) -> dict[str, list[Candidate]]:
        out: dict[str, list[Candidate]] = {}
        for spec in self.specs:
            found: list[Candidate] = []
            for page_layout in layout:
                found.extend(self._match_page(spec, page_layout, layout.name, doc_type))
            if found:
                found.sort(key=lambda c: -c.score)
                out[spec.key] = found
        return out

    # -- per page -----------------------------------------------------------

    def _match_page(
        self, spec: FieldSpec, layout: PageLayout, doc_name: str, doc_type: str
    ) -> list[Candidate]:
        candidates: list[Candidate] = []
        labelled_spans: set[int] = set()

        for span in layout.spans:
            label, similarity, remainder = self._label_hit(spec, span)
            if label is None:
                continue
            labelled_spans.add(id(span))

            if remainder:
                # Caption and value in one span.
                cand = self._build(
                    spec, remainder, span, doc_name, doc_type,
                    label_similarity=similarity, proximity=1.0,
                    matched_label=label, kind="inline", direction="inline",
                    same_line=True,
                )
                if cand:
                    candidates.append(cand)

            # Caption alone -> look outward for the value.
            for neighbour in layout.neighbours(span, spec.search.directions, spec.search.max_distance):
                if neighbour.direction not in spec.search.directions:
                    continue
                # The neighbour span itself must not be another field's own
                # caption. A table row ("Block | Road | City | Pin | State")
                # puts several dictionary captions on one line; a short label
                # like address_2's "Road" fuzzy-matching that ROW's "Road"
                # header (rather than a genuine "Road: <value>" caption) then
                # picked the very next cell, "City", as if it were the value
                # -- a caption, not data. _join_value already guards against
                # a caption appearing as a CONTINUATION; this is the same
                # check applied to the starting span, which it never covered.
                if self._is_label_like(neighbour.span.text):
                    continue
                text = self._strip_own_labels(spec, self._join_value(layout, neighbour.span, spec))
                cand = self._build(
                    spec, text, neighbour.span, doc_name, doc_type,
                    label_similarity=similarity, proximity=neighbour.proximity,
                    matched_label=label, kind="adjacent", direction=neighbour.direction,
                    same_line=neighbour.same_line,
                )
                if cand:
                    candidates.append(cand)

        # Fallback: pattern hits with no caption in sight.
        if spec.patterns:
            for span in layout.spans:
                if id(span) in labelled_spans:
                    continue
                for match in spec.find_pattern_matches(span.text):
                    cand = self._build(
                        spec, match, span, doc_name, doc_type,
                        label_similarity=PATTERN_ONLY_LABEL_SCORE, proximity=0.0,
                        matched_label=None, kind="pattern_only", direction="",
                        same_line=False,
                    )
                    if cand:
                        candidates.append(cand)

        return candidates

    # -- helpers ------------------------------------------------------------

    def _label_hit(
        self, spec: FieldSpec, span: TextSpan
    ) -> tuple[Optional[str], float, str]:
        """Does this span carry one of the field's captions?

        Returns (label, similarity, remainder-after-the-label). The remainder
        is non-empty when caption and value share the span.
        """
        text = span.text.strip()
        if not text or len(text) > spec.label_match.max_label_chars * 3:
            return None, 0.0, ""

        cleaned = clean_label(text)
        # Alignment offsets must be computed against a string that maps 1:1
        # onto `text`, so only case is changed here. Using clean_label's output
        # (which strips punctuation and collapses runs of whitespace) shifted
        # the indices and sliced the value at the wrong position.
        lowered = text.lower()
        best: tuple[Optional[str], float, str] = (None, 0.0, "")

        for label in spec.labels:
            target = clean_label(label)
            if not target:
                continue
            threshold = self._min_similarity(spec, target)

            # Whole span is the caption.
            whole = fuzz.ratio(cleaned, target) / 100.0
            if whole >= threshold and whole > best[1]:
                best = (label, whole, "")

            # Caption is a prefix of the span, value follows.
            if len(lowered) > len(target):
                alignment = fuzz.partial_ratio_alignment(target, lowered)
                if alignment is not None:
                    partial = alignment.score / 100.0
                    # Only trust an inline split when the caption starts at the
                    # beginning of the span; a caption word appearing mid-prose
                    # is not a caption.
                    if partial >= threshold and alignment.dest_start <= 2:
                        tail = text[alignment.dest_end:]
                        remainder = strip_value_prefix(tail)
                        if remainder and self._is_inline_split(spec, tail, remainder):
                            # Prefer the longest caption that fits, so
                            # "Building No./Flat No.: 1ST FLOOR" splits on the
                            # full caption rather than a shorter synonym that
                            # would leave caption text inside the value.
                            if partial > best[1] or (
                                partial >= best[1] - 0.02
                                and len(target) > len(clean_label(best[0] or ""))
                            ):
                                best = (label, partial, remainder)

        return best

    @staticmethod
    def _is_inline_split(spec: FieldSpec, tail: str, remainder: str) -> bool:
        """Is what follows the caption really a value, or just more prose?

        A form prints "State: West Bengal" -- caption, separator, value. Plain
        running text does not: the span "Address of Principal Place of" fuzzy
        matches the caption "Address", and without this check the rest of the
        sentence gets stored as the address. Requiring an explicit separator,
        or a remainder that already has the field's shape, tells the two apart.

        A "." right before a digit is excluded from that separator check --
        see _SEPARATOR_RE's comment. "FLAT NO. 302, BLOCK C, 3RD FLOOR, SAI
        INDUSTRIAL PARK..." must not split into caption "FLAT NO" + value
        "302, BLOCK C, ..." just because the abbreviation happens to fuzzy-
        match a field's own label ("Flat No") at the start of the address's
        own free text -- there is no separate caption on the page here at
        all, and address_1 came back as "302" for real (Chrome/OCR-rendered)
        combined-address documents before this guard existed.
        """
        if _ABBREVIATION_DOT_RE.match(tail):
            return False
        if _SEPARATOR_RE.match(tail) or tail.lstrip().startswith("."):
            return True
        return bool(spec.patterns) and spec.matches_pattern(remainder)

    def _strip_own_labels(self, spec: FieldSpec, text: str) -> str:
        """Remove a caption that is repeated inside an adjacent value span.

        GST certificates lay out an address as a caption cell ("Address of
        Principal Place of") whose neighbouring cell repeats a sub-caption
        ("Building No./Flat No.: 1ST FLOOR"). Without this the sub-caption ends
        up stored as part of the address.
        """
        current = text.strip()
        for _ in range(2):
            lowered = current.lower()
            stripped = current
            for label in spec.labels:
                target = clean_label(label)
                if not target or len(lowered) <= len(target):
                    continue
                alignment = fuzz.partial_ratio_alignment(target, lowered)
                if alignment is None:
                    continue
                if alignment.score / 100.0 >= self._min_similarity(spec, target) and alignment.dest_start <= 2:
                    tail = current[alignment.dest_end:]
                    candidate = strip_value_prefix(tail)
                    if (
                        candidate
                        and len(candidate) < len(stripped)
                        and self._is_inline_split(spec, tail, candidate)
                    ):
                        stripped = candidate
            if stripped == current:
                break
            current = stripped
        return current

    def _join_value(self, layout: PageLayout, span: TextSpan, spec: FieldSpec) -> str:
        """For free-text fields, stitch together spans that continue the value
        on the same line (an address broken into several boxes), stopping at a
        wide gap or at the next caption -- then do the same DOWN the page for a
        value that wraps onto a second (or third) visual row.

        A combined GST/Udyam address line often does not fit one table row and
        wraps, e.g.

            FLAT NO. 302, BLOCK C, 3RD FLOOR, SAI INDUSTRIAL PARK,
            BUILDING 4, SECTOR 18, SERVICE ROAD, UDYOG VIHAR EXTENSION,
            SIKANDERPUR, SARHOL, GURUGRAM, HARYANA

        Before this, only the first row was ever captured -- address_1 (and
        the segmenter fed from it) silently lost every fragment after the
        first wrap. Continuation rows are accepted only while ALL of these
        hold, so an ordinary next FIELD below is never swallowed:
          - the row starts within one caption-column-width of this value's
            own left edge (a genuine wrap keeps the value's indentation; a new
            field's value starts in the caption's column, further left)
          - the row is not itself a caption (_is_label_like)
          - the vertical gap to the previous row is tight (a real wrap has
            ordinary line spacing; a new table row has a bigger one)
        """
        if spec.patterns or spec.value_type != "text":
            return span.text

        line = layout.line_of(span)
        if line is None:
            return span.text

        unit = span.bbox.height or layout.median_height or 1.0
        row0 = [span.text]
        prev = span
        for other in line.spans:
            if other.bbox.x1 <= span.bbox.x1 or other is span:
                continue
            gap = (other.bbox.x1 - prev.bbox.x2) / unit
            if gap > 1.2 or self._is_label_like(other.text):
                break
            row0.append(other.text)
            prev = other

        # `value` is built up row by row. Each wrapped continuation row is
        # spliced on with a separator chosen from THAT row's own provenance
        # (its internal punctuation) -- never from a comma elsewhere in the
        # value so far. See _wrap_separator.
        value = " ".join(row0)
        prev_row_text = value

        row_bottom = span.bbox.y2
        row_centre = span.bbox.cy
        row_left = span.bbox.x1
        for candidate_line in layout.lines:
            # A wrapped continuation often sits close enough to the caption
            # row that their bounding boxes slightly OVERLAP (e.g. a value
            # ending at y2=365 and its very next visual row starting at
            # y1=364) -- ordinary tight line-spacing, not the same row. Using
            # the row's own CENTRE rather than a strict "starts after my
            # bottom edge" cutoff is what tells that apart from a line that
            # actually IS the same row as the caption (whose centre sits at or
            # above row_centre, since the caption's own line is scanned here
            # too and must be skipped).
            if candidate_line.bbox.cy <= row_centre:
                continue
            vgap = (candidate_line.bbox.y1 - row_bottom) / unit
            if vgap > 1.6:
                break
            starters = [s for s in candidate_line.spans if s.bbox.x1 <= row_left + unit * 2.0]
            if not starters:
                break
            first = min(starters, key=lambda s: s.bbox.x1)
            if abs(first.bbox.x1 - row_left) > unit * 2.0 or self._is_label_like(first.text):
                break
            row_parts = [first.text]
            prev = first
            for other in candidate_line.spans:
                if other.bbox.x1 <= first.bbox.x1 or other is first:
                    continue
                gap = (other.bbox.x1 - prev.bbox.x2) / unit
                if gap > 1.2 or self._is_label_like(other.text):
                    break
                row_parts.append(other.text)
                prev = other
            row_text = " ".join(row_parts)
            value += self._wrap_separator(prev_row_text, row_text) + row_text
            row_bottom = candidate_line.bbox.y2
            row_centre = candidate_line.bbox.cy
            prev_row_text = row_text

        return value

    @staticmethod
    def _wrap_separator(prev_row_text: str, row_text: str) -> str:
        """Separator to splice a wrapped continuation row onto the value.

        A rendered line-wrap INSIDE a comma-delimited list eats the comma that
        sat at the wrap point; OCR then hands us two rows with nothing between
        them, and a plain-space join fuses two list items into one fragment
        ("...PARK, HOSUR ROAD" -> "...PARK HOSUR ROAD"). Restore ", " -- which
        survives `collapse_spaces` normalisation, unlike a "\\n".

        Every test is provenance of THE WRAPPED ROW ITSELF, never of the
        accumulated value -- a comma appearing earlier elsewhere in the value
        can not, by itself, force a separator here. Restore ", " only when ALL:

          1. `prev_row_text` (the row that ends at this wrap) carries >= 2 of
             its own internal commas -- it is unambiguously a delimited list
             of 3+ items, and its continuation on the next visual row is the
             same list. One comma is ambiguous ("City, State" tail of a
             company name) and is left as a plain-space join.
          2. that row does not already end in separator punctuation, and the
             continuation row does not already begin with one.
        """
        tail = prev_row_text.rstrip()
        head = row_text.lstrip()
        if not tail or not head:
            return " "
        if tail[-1] in ",;–—" or head[0] in ",;":
            return " "
        if prev_row_text.count(",") < 2:
            return " "
        return ", "

    def _is_label_like(self, text: str) -> bool:
        cleaned = clean_label(text)
        if not cleaned:
            return False
        return any(fuzz.ratio(cleaned, lbl) >= 88 for lbl in self.all_labels)

    def _foreign_field(self, spec: FieldSpec, value: str) -> Optional[str]:
        """Name the field whose shape this value has, if it is not this one."""
        if spec.patterns or not value:
            return None
        for key, patterns in self._foreign_patterns:
            if key == spec.key:
                continue
            if any(p.fullmatch(value) for p in patterns):
                return key
        return None

    def _build(
        self,
        spec: FieldSpec,
        raw: str,
        span: TextSpan,
        doc_name: str,
        doc_type: str,
        label_similarity: float,
        proximity: float,
        matched_label: Optional[str],
        kind: str,
        direction: str,
        same_line: bool,
    ) -> Optional[Candidate]:
        raw = (raw or "").strip()
        if not raw:
            return None

        value, pattern_matched = self._derive_value(spec, raw)
        if not value:
            return None
        if spec.is_excluded(value):
            # Structurally valid but semantically wrong -- e.g. the issuing
            # authority's portal URL in certificate boilerplate, which is a
            # perfectly well-formed website that is never this vendor's.
            return None
        if spec.patterns and not pattern_matched:
            # A field with a declared shape should not accept something of a
            # different shape just because it sat next to the caption.
            return None

        notes: list[str] = []
        components = {
            "label_similarity": label_similarity,
            "spatial_proximity": proximity,
            "pattern_match": 1.0 if pattern_matched else (0.5 if not spec.patterns else 0.0),
            "ocr_confidence": float(span.confidence),
        }
        score = sum(components[k] * w for k, w in spec.confidence.weights.items() if k in components)

        bonus_key = "adjacent_same_line" if (kind == "adjacent" and same_line) else kind
        bonus = KIND_BONUS.get(bonus_key, 0.0)
        if bonus:
            score += bonus
            notes.append(f"{bonus_key}:{bonus:+.2f}")

        if doc_type and spec.expected_documents:
            if doc_type in spec.expected_documents:
                score += 0.05
                notes.append(f"expected_document:{doc_type}")
            else:
                score -= 0.05
                notes.append(f"unexpected_document:{doc_type}")

        if kind != "inline" and self._is_label_like(raw) and not pattern_matched:
            score *= LABEL_LIKE_PENALTY
            notes.append("value_looks_like_a_label")

        foreign = self._foreign_field(spec, value)
        if foreign:
            score *= FOREIGN_PATTERN_PENALTY
            notes.append(f"value_matches_{foreign}_pattern")

        if not spec.patterns and spec.value_type == "text" and len(value) < MIN_PLAUSIBLE_TEXT_LEN:
            score *= 0.5
            notes.append("implausibly_short_value")

        return Candidate(
            field_key=spec.key,
            value=value,
            raw_text=raw,
            score=max(0.0, min(1.0, score)),
            components=components,
            source_document=doc_name,
            page=span.page,
            span=span,
            matched_label=matched_label,
            match_kind=kind,
            direction=direction,
            pattern_matched=pattern_matched,
            notes=notes,
        )

    def _derive_value(self, spec: FieldSpec, raw: str) -> tuple[str, bool]:
        """Pull the value out of the raw text and normalize it.

        When the field declares patterns, prefer the matched substring over the
        whole string: the span 'RTGS / NEFT /IFS Code:ICIC0006278' should yield
        'ICIC0006278', not the entire caption-plus-value blob.
        """
        if spec.patterns:
            for text in (raw, normalize(raw, spec.normalization)):
                matches = spec.find_pattern_matches(text)
                if matches:
                    return spec.canonicalize(normalize(matches[0], spec.normalization)), True
            # Normalization can create a match that the raw text lacked, e.g.
            # 'ICIC 0006278' only becomes an IFSC once spaces are removed.
            normalized = normalize(raw, spec.normalization)
            if spec.matches_pattern(normalized):
                return spec.canonicalize(normalized), True
            return spec.canonicalize(normalized), False

        return spec.canonicalize(normalize(raw, spec.normalization)), False
