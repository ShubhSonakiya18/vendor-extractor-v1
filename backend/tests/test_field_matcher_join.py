"""Exhaustive coverage of field_matcher._join_value / _wrap_separator.

The concern (approved-plan constraint 3 / B): when a rendered address wraps
across lines, OCR drops the trailing comma at the wrap point and _join_value
must NOT fuse the two list items with a bare space. The fix restores ", " --
but ONLY from provenance of the wrapped row itself (>= 2 internal commas),
never because a comma appeared earlier elsewhere in the value.

Matrix covered here:
  - comma-immediately-before-wrap (the fix must fire)
  - no comma before wrap
  - a comma earlier in the value but NOT in the wrapped row (must NOT fire)
  - one comma in the wrapped row -- ambiguous, must NOT fire
  - multiple consecutive wraps
  - wrapped row already ends in "," / continuation already starts with ","
  - an address text field
  - a non-address text field (company name)
"""
from __future__ import annotations

import pytest

from app.services.extraction_pipeline.extract.field_matcher import FieldMatcher
from app.services.extraction_pipeline.ingest.layout_engine import PageLayout
from app.services.extraction_pipeline.models import BBox, Page, TextSpan


class _Spec:
    """Minimal FieldSpec stand-in for _join_value (only these attrs read)."""

    def __init__(self, value_type: str = "text"):
        self.patterns = []
        self.value_type = value_type
        self.labels = []


def _span(text: str, x1: float, y: float, x2: float) -> TextSpan:
    return TextSpan(text=text, page=1, bbox=BBox(x1, y, x2, y + 16.0),
                    source_document="t.pdf")


def _layout(*rows: list[TextSpan]) -> tuple[PageLayout, TextSpan]:
    spans = [s for row in rows for s in row]
    layout = PageLayout(Page(number=1, width=900.0, height=1200.0, spans=spans))
    return layout, rows[0][0]


# ---------------------------------------------------------------------------
# _wrap_separator -- the decision in isolation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("prev_row,next_row,want", [
    # comma-immediately-before-wrap: prev row is a 3+-item list -> restore ", "
    ("BUILDINGB,3RDFLOOR,PARTC,UNIT312,NORTHCITYPARK", "BARASATROAD,ZONE", ", "),
    # no comma before wrap -> plain space
    ("1ST FLOOR", "MG ROAD", " "),
    # a comma earlier in the value, but the WRAPPED ROW itself has none
    ("SUNRISE APARTMENTS", "BANJARA HILLS", " "),
    # exactly one comma in the wrapped row -- ambiguous (could be "City, State"
    # tail of a name) -> plain space
    ("NARMADA ENGINEERING WORKS, KOLKATA", "PRIVATE LIMITED", " "),
    ("UNIT 5, ORBIT PARK", "NH-8, BHIWANDI", " "),
    # wrapped row already ends with a separator -> nothing to restore
    ("FLAT 1, MG ROAD, SECTOR 5,", "KORAMANGALA", " "),
    # continuation row already begins with a separator
    ("FLAT 1, MG ROAD, SECTOR 5", ", KORAMANGALA", " "),
    # empty inputs are safe
    ("", "ANYTHING", " "),
    ("A,B,C", "", " "),
])
def test_wrap_separator_matrix(prev_row, next_row, want):
    assert FieldMatcher._wrap_separator(prev_row, next_row) == want


def test_wrap_separator_is_row_local_not_value_wide():
    """A comma-rich head followed by a comma-free wrapped row must NOT
    inherit the head's list-ness."""
    # row that ends at the wrap has zero commas of its own
    assert FieldMatcher._wrap_separator("VILLAGE KOTHRUD PUNE MAHARASHTRA", "X") == " "
    # ...even though an earlier row of the same value was a list
    assert FieldMatcher._wrap_separator("PLAIN CONTINUATION TEXT", "MORE") == " "


# ---------------------------------------------------------------------------
# _join_value -- end to end over a built layout
# ---------------------------------------------------------------------------

def _run(fm: FieldMatcher, layout: PageLayout, first: TextSpan, vt: str = "text") -> str:
    return fm._join_value(layout, first, _Spec(vt))


@pytest.fixture
def fm():
    return FieldMatcher(specs=[], all_labels=set())


def test_join_wrapped_address_restores_lost_comma(fm):
    # three visual rows of one combined address; row 1 & 2 lost their wrap comma
    layout, first = _layout(
        [_span("FLATNO.302,BLOCKC,3RDFLOOR,SAIPARK", 300, 380, 700)],
        [_span("SECTOR18,SERVICEROAD,UDYOGVIHAR", 300, 400, 690)],
        [_span("SIKANDERPUR,SARHOL", 300, 420, 470)],
    )
    out = _run(fm, layout, first)
    # boundaries between the three rows are restored as ", "
    assert out == ("FLATNO.302,BLOCKC,3RDFLOOR,SAIPARK, "
                   "SECTOR18,SERVICEROAD,UDYOGVIHAR, SIKANDERPUR,SARHOL")


def test_join_wrapped_no_comma_stays_space(fm):
    layout, first = _layout(
        [_span("1ST FLOOR", 300, 380, 360)],
        [_span("MG ROAD", 300, 400, 360)],
    )
    assert _run(fm, layout, first) == "1ST FLOOR MG ROAD"


def test_join_non_address_name_not_split_by_stray_comma(fm):
    # a company name whose first row happens to carry one comma, wrapping to a
    # continuation -- must stay space-joined (one comma is ambiguous)
    layout, first = _layout(
        [_span("NARMADA ENGINEERING WORKS, KOLKATA", 300, 380, 640)],
        [_span("PRIVATE LIMITED", 300, 400, 400)],
    )
    assert _run(fm, layout, first) == "NARMADA ENGINEERING WORKS, KOLKATA PRIVATE LIMITED"


def test_join_single_row_unchanged(fm):
    layout, first = _layout([_span("PLOT 4, MG ROAD, BENGALURU", 300, 380, 560)])
    assert _run(fm, layout, first) == "PLOT 4, MG ROAD, BENGALURU"


def test_join_stops_at_caption_row(fm):
    # a following row that looks like a caption ends the join
    fm2 = FieldMatcher(specs=[], all_labels={"date of liability"})
    layout, first = _layout(
        [_span("A,B,C,D,E", 300, 380, 500)],
        [_span("Date of Liability", 60, 410, 200)],
    )
    assert fm2._join_value(layout, first, _Spec()) == "A,B,C,D,E"
