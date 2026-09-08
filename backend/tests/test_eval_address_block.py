"""eval_extraction's address-block roll-up.

The address is filled as one unit (address_1..4 + city + state + pin_code). A
combined address line the resolver fails to segment shows up as several
independent `missed`/`wrong` fields, which is easy to miss when it's averaged
across 24 fields. `_address_block` collapses those seven into one verdict so an
address regression reads as a single PASS/FAIL. These tests pin that roll-up.
"""

from app.eval.eval_extraction import _address_block, evaluate, summarize

_GT = {
    "fields": {
        "address_1": {"value": "IT PARK, PHASE 8"},
        "address_2": {"absent": True},
        "address_3": {"absent": True},
        "address_4": {"absent": True},
        "city": {"value": "Mohali"},
        "state": {"value": "Punjab"},
        "pin_code": {"value": "160059"},
        "gst_number": {"value": "03AAAAA0000A1Z5"},  # a non-address field
    }
}


def _block(extracted):
    return summarize(evaluate(_GT, extracted))["address_block"]


def test_perfectly_segmented_address_is_exact():
    block = _block({
        "address_1": "IT PARK, PHASE 8",
        "city": "Mohali", "state": "Punjab", "pin_code": "160059",
        "gst_number": "03AAAAA0000A1Z5",
    })
    assert block["exact"] is True
    assert block["field_accuracy"] == 100.0
    assert block["fields"] == 7                # only the address fields
    assert block["wrong"] == 0 and block["missed"] == 0


def test_unsegmented_blob_fails_the_block():
    # whole address dumped in address_1, city/state/pin never filled
    block = _block({
        "address_1": "IT PARK, PHASE 8, MOHALI, S.A.S NAGAR, PUNJAB, 160059",
        "gst_number": "03AAAAA0000A1Z5",
    })
    assert block["exact"] is False
    assert block["wrong"] == 1                 # address_1 differs from GT
    assert block["missed"] == 3                # city, state, pin_code
    assert block["field_accuracy"] == 0.0


def test_partial_segmentation_is_not_exact():
    # pin recovered, city/state still missing
    block = _block({
        "address_1": "IT PARK, PHASE 8, MOHALI, S.A.S NAGAR, PUNJAB",
        "pin_code": "160059",
        "gst_number": "03AAAAA0000A1Z5",
    })
    assert block["exact"] is False
    assert block["correct"] == 1              # pin_code
    assert block["missed"] == 2               # city, state


def test_absent_when_ground_truth_has_no_address_fields():
    gt = {"fields": {"gst_number": {"value": "03AAAAA0000A1Z5"}}}
    assert _address_block(evaluate(gt, {"gst_number": "03AAAAA0000A1Z5"})) is None
