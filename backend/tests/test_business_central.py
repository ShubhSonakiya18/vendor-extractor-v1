"""Business Central manual-push endpoints (vendor)."""

from __future__ import annotations

import pytest

from app.config.config import settings

VENDOR_PAYLOAD = {
    "vendor_name": "M.B. Control & Systems Pvt. Ltd.",
    "address_1": "Srijan Industrial Logistic Park",
    "address_2": "Block B, 3rd Floor",
    "city": "Howrah",
    "state": "West Bengal",
    "country": "India",
    "pin_code": "711302",
    "telephone_1": "9831330473",
    "email": "enquiry@mbcontrol.com",
    "pan": "AABCM7980K",
    "gst_no": "19AABCM7980K1ZU",
}


@pytest.fixture
def bc_enabled():
    prev = settings.BC_ENABLED
    settings.BC_ENABLED = True
    yield
    settings.BC_ENABLED = prev


@pytest.fixture
def bc_disabled():
    # Force the disabled state regardless of what the loaded .env has.
    prev = settings.BC_ENABLED
    settings.BC_ENABLED = False
    yield
    settings.BC_ENABLED = prev


def _create_vendor(auth_client) -> int:
    r = auth_client.post("/vendors", json=VENDOR_PAYLOAD)
    assert r.status_code == 201, r.text
    return r.json()["id"]


class TestDisabled:
    def test_payload_503_when_disabled(self, auth_client, bc_disabled):
        vid = _create_vendor(auth_client)
        assert auth_client.get(f"/business-central/vendors/{vid}/payload").status_code == 503

    def test_mark_pushed_503_when_disabled(self, auth_client, bc_disabled):
        vid = _create_vendor(auth_client)
        r = auth_client.patch(f"/business-central/vendors/{vid}/mark-pushed", json={"bc_no": "X"})
        assert r.status_code == 503


class TestPayload:
    def test_requires_auth(self, client, bc_enabled):
        assert client.get("/business-central/vendors/1/payload").status_code == 401

    def test_payload_shape_and_field_mapping(self, auth_client, bc_enabled):
        vid = _create_vendor(auth_client)
        r = auth_client.get(f"/business-central/vendors/{vid}/payload")
        assert r.status_code == 200
        body = r.json()

        assert body["vendor_id"] == vid
        assert body["already_pushed"] is False
        assert body["method"] == "POST"
        assert body["target_url"].endswith("/VendorCard")

        p = body["payload"]
        assert p["No"] == ""
        assert p["Name"] == VENDOR_PAYLOAD["vendor_name"]
        assert p["Address"] == VENDOR_PAYLOAD["address_1"]
        assert p["Address_2"] == VENDOR_PAYLOAD["address_2"]
        assert p["City"] == "Howrah"
        assert p["County"] == "West Bengal"          # BC calls state "County"
        assert p["Country_Region_Code"] == "India"
        assert p["Post_Code"] == "711302"
        assert p["Phone_No"] == "9831330473"
        assert p["E_Mail"] == VENDOR_PAYLOAD["email"]
        assert p["PAN_Number"] == "AABCM7980K"
        assert p["GST_Number"] == "19AABCM7980K1ZU"

    def test_blank_fields_are_omitted(self, auth_client, bc_enabled):
        r = auth_client.post("/vendors", json={"vendor_name": "Bare Co"})
        vid = r.json()["id"]
        p = auth_client.get(f"/business-central/vendors/{vid}/payload").json()["payload"]
        assert "E_Mail" not in p and "GST_Number" not in p and "Post_Code" not in p
        assert p["Name"] == "Bare Co"

    def test_posting_groups_included_only_when_configured(self, auth_client, bc_enabled, monkeypatch):
        vid = _create_vendor(auth_client)
        p = auth_client.get(f"/business-central/vendors/{vid}/payload").json()["payload"]
        assert "Vendor_Posting_Group" not in p

        monkeypatch.setattr(settings, "BC_VENDOR_POSTING_GROUP", "EMPLOAN")
        p = auth_client.get(f"/business-central/vendors/{vid}/payload").json()["payload"]
        assert p["Vendor_Posting_Group"] == "EMPLOAN"

    def test_payload_404_for_missing_vendor(self, auth_client, bc_enabled):
        assert auth_client.get("/business-central/vendors/999999/payload").status_code == 404


class TestMarkPushed:
    def test_marks_and_is_idempotent(self, auth_client, bc_enabled):
        vid = _create_vendor(auth_client)

        r = auth_client.patch(f"/business-central/vendors/{vid}/mark-pushed", json={"bc_no": "EMPV/0123"})
        assert r.status_code == 200
        assert r.json() == {"vendor_id": vid, "bc_status": "pushed", "bc_no": "EMPV/0123"}

        # reflected on the record
        v = auth_client.get(f"/vendors/{vid}").json()
        assert v["bc_status"] == "pushed" and v["bc_no"] == "EMPV/0123"

        # payload endpoint now flags it
        assert auth_client.get(f"/business-central/vendors/{vid}/payload").json()["already_pushed"] is True

        # second mark -> 409
        r2 = auth_client.patch(f"/business-central/vendors/{vid}/mark-pushed", json={"bc_no": "EMPV/9999"})
        assert r2.status_code == 409

    def test_mark_pushed_404_for_missing_vendor(self, auth_client, bc_enabled):
        r = auth_client.patch("/business-central/vendors/999999/mark-pushed", json={"bc_no": "X"})
        assert r.status_code == 404

    def test_bc_no_required(self, auth_client, bc_enabled):
        vid = _create_vendor(auth_client)
        assert auth_client.patch(f"/business-central/vendors/{vid}/mark-pushed", json={}).status_code == 422
