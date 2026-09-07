"""Vendor / customer persistence endpoints: create, list, get, and the
GSTIN-uniqueness 409 path.
"""

from __future__ import annotations

VENDOR_PAYLOAD = {
    "vendor_name": "M.B. Control & Systems Pvt. Ltd.",
    "address_1": "Srijan Industrial Logistic Park",
    "city": "Howrah",
    "state": "West Bengal",
    "country": "India",
    "pin_code": "711302",
    "email": "enquiry@mbcontrol.com",
    "pan": "AABCM7980K",
    "gst_no": "19AABCM7980K1ZU",
    "udyam_no": "UDYAM-WB-10-0003543",
    "bank_name": "ICICI Bank",
    "ifsc_swift_code": "ICIC0006278",
    "account_number": "627851000539",
    "fields_needing_review": ["pan_number"],
    "raw_extraction": {"anything": "kept verbatim"},
}

CUSTOMER_PAYLOAD = {
    "company_name": "Acme Engineering",
    "billing_address": "12 MG Road",
    "city": "Pune",
    "state": "Maharashtra",
    "zip_code": "411001",
    "country": "India",
    "gst_registration_number": "27AABCU9603R1ZM",
    "pan_number": "AABCU9603R",
    "email_id_to": "accounts@acme-eng.in",
    "payment_terms": "NET30",
    "salesperson": "R. Sharma",
    "region": "West",
    "type": "License",
}


class TestAuthRequired:
    def test_create_vendor_requires_auth(self, client):
        assert client.post("/vendors", json=VENDOR_PAYLOAD).status_code == 401

    def test_create_customer_requires_auth(self, client):
        assert client.post("/customers", json=CUSTOMER_PAYLOAD).status_code == 401


class TestVendor:
    def test_create_then_get_and_list(self, auth_client):
        r = auth_client.post("/vendors", json=VENDOR_PAYLOAD)
        assert r.status_code == 201, r.text
        body = r.json()
        vid = body["id"]
        assert body["vendor_name"] == VENDOR_PAYLOAD["vendor_name"]
        assert body["gst_no"] == VENDOR_PAYLOAD["gst_no"]
        assert body["bc_status"] == "not_pushed"
        assert body["fields_needing_review"] == ["pan_number"]

        got = auth_client.get(f"/vendors/{vid}")
        assert got.status_code == 200
        assert got.json()["id"] == vid

        listed = auth_client.get("/vendors")
        assert listed.status_code == 200
        assert any(v["id"] == vid for v in listed.json())

    def test_duplicate_gstin_returns_409_with_existing_id(self, auth_client):
        first = auth_client.post("/vendors", json=VENDOR_PAYLOAD)
        assert first.status_code == 201
        existing_id = first.json()["id"]

        # different Udyam so only the GSTIN clashes
        dup = auth_client.post("/vendors", json={
            **VENDOR_PAYLOAD, "vendor_name": "Different Name", "udyam_no": "UDYAM-XX-99-9999999",
        })
        assert dup.status_code == 409
        detail = dup.json()["detail"]
        assert detail["existing_vendor_id"] == existing_id
        assert detail["gst_no"] == VENDOR_PAYLOAD["gst_no"]

    def test_duplicate_udyam_returns_409(self, auth_client):
        first = auth_client.post("/vendors", json=VENDOR_PAYLOAD)
        assert first.status_code == 201

        # different GSTIN, same Udyam number
        dup = auth_client.post("/vendors", json={
            **VENDOR_PAYLOAD, "vendor_name": "Other", "gst_no": "27AABCU9603R1ZM",
        })
        assert dup.status_code == 409
        detail = dup.json()["detail"]
        assert detail["existing_vendor_id"] == first.json()["id"]
        assert detail["udyam_no"] == VENDOR_PAYLOAD["udyam_no"]
        assert "Udyam" in detail["message"]

    def test_blank_identifiers_are_not_deduplicated(self, auth_client):
        p = {k: v for k, v in VENDOR_PAYLOAD.items() if k not in ("gst_no", "udyam_no")}
        r1 = auth_client.post("/vendors", json=p)
        r2 = auth_client.post("/vendors", json=p)
        assert r1.status_code == 201 and r2.status_code == 201
        assert r1.json()["id"] != r2.json()["id"]

    def test_lookup_by_udyam_query(self, auth_client):
        auth_client.post("/vendors", json=VENDOR_PAYLOAD)
        hit = auth_client.get("/vendors", params={"udyam_no": VENDOR_PAYLOAD["udyam_no"]})
        assert hit.status_code == 200 and len(hit.json()) == 1
        miss = auth_client.get("/vendors", params={"udyam_no": "UDYAM-NO-00-0000000"})
        assert miss.status_code == 200 and miss.json() == []

    def test_lookup_by_gst_query(self, auth_client):
        auth_client.post("/vendors", json=VENDOR_PAYLOAD)
        hit = auth_client.get("/vendors", params={"gst_no": VENDOR_PAYLOAD["gst_no"]})
        assert hit.status_code == 200 and len(hit.json()) == 1
        miss = auth_client.get("/vendors", params={"gst_no": "00XXXXX0000X0X0"})
        assert miss.status_code == 200 and miss.json() == []

    def test_get_missing_vendor_404(self, auth_client):
        assert auth_client.get("/vendors/999999").status_code == 404

    def test_list_is_newest_first(self, auth_client):
        for n in ("First", "Second", "Third"):
            auth_client.post("/vendors", json={"vendor_name": n})
        names = [v["vendor_name"] for v in auth_client.get("/vendors").json()]
        assert names[:3] == ["Third", "Second", "First"]

    def test_vendor_name_required(self, auth_client):
        bad = {k: v for k, v in VENDOR_PAYLOAD.items() if k != "vendor_name"}
        assert auth_client.post("/vendors", json=bad).status_code == 422


class TestCustomer:
    def test_create_persists_all_17_fields(self, auth_client):
        r = auth_client.post("/customers", json=CUSTOMER_PAYLOAD)
        assert r.status_code == 201, r.text
        body = r.json()
        for key, value in CUSTOMER_PAYLOAD.items():
            assert body[key] == value
        assert body["bc_status"] == "not_pushed"

    def test_business_fields_default_when_omitted(self, auth_client):
        minimal = {"company_name": "Bare Co", "gst_registration_number": "29AAAAA0000A1Z5"}
        body = auth_client.post("/customers", json=minimal).json()
        assert body["payment_terms"] == ""
        assert body["salesperson"] == ""
        assert body["region"] == ""
        assert body["customer_agreement"] == ""
        assert body["type"] == "Services"

    def test_duplicate_gstin_returns_409(self, auth_client):
        first = auth_client.post("/customers", json=CUSTOMER_PAYLOAD)
        assert first.status_code == 201
        dup = auth_client.post("/customers", json={**CUSTOMER_PAYLOAD, "company_name": "Other"})
        assert dup.status_code == 409
        assert dup.json()["detail"]["existing_customer_id"] == first.json()["id"]

    def test_get_and_list(self, auth_client):
        cid = auth_client.post("/customers", json=CUSTOMER_PAYLOAD).json()["id"]
        assert auth_client.get(f"/customers/{cid}").json()["id"] == cid
        assert any(c["id"] == cid for c in auth_client.get("/customers").json())


class TestVendorUpdateDelete:
    def test_patch_changes_only_sent_fields(self, auth_client):
        vid = auth_client.post("/vendors", json=VENDOR_PAYLOAD).json()["id"]
        r = auth_client.patch(f"/vendors/{vid}", json={"city": "Kolkata", "email": "new@mb.com"})
        assert r.status_code == 200
        body = r.json()
        assert body["city"] == "Kolkata"
        assert body["email"] == "new@mb.com"
        assert body["vendor_name"] == VENDOR_PAYLOAD["vendor_name"]  # untouched

    def test_patch_requires_auth(self, client):
        assert client.patch("/vendors/1", json={"city": "X"}).status_code == 401

    def test_patch_missing_vendor_404(self, auth_client):
        assert auth_client.patch("/vendors/999999", json={"city": "X"}).status_code == 404

    def test_patch_gstin_clash_409(self, auth_client):
        a = auth_client.post("/vendors", json=VENDOR_PAYLOAD).json()["id"]
        b = auth_client.post("/vendors", json={
            **VENDOR_PAYLOAD, "gst_no": "27AABCU9603R1ZM",
            "udyam_no": "UDYAM-XX-99-9999999", "vendor_name": "B",
        }).json()["id"]
        r = auth_client.patch(f"/vendors/{b}", json={"gst_no": VENDOR_PAYLOAD["gst_no"]})
        assert r.status_code == 409
        assert r.json()["detail"]["existing_vendor_id"] == a

    def test_patch_udyam_clash_409(self, auth_client):
        a = auth_client.post("/vendors", json=VENDOR_PAYLOAD).json()["id"]
        b = auth_client.post("/vendors", json={
            **VENDOR_PAYLOAD, "gst_no": "27AABCU9603R1ZM",
            "udyam_no": "UDYAM-XX-99-9999999", "vendor_name": "B",
        }).json()["id"]
        r = auth_client.patch(f"/vendors/{b}", json={"udyam_no": VENDOR_PAYLOAD["udyam_no"]})
        assert r.status_code == 409
        assert r.json()["detail"]["existing_vendor_id"] == a
        assert "Udyam" in r.json()["detail"]["message"]

    def test_patch_same_gstin_on_self_ok(self, auth_client):
        vid = auth_client.post("/vendors", json=VENDOR_PAYLOAD).json()["id"]
        r = auth_client.patch(f"/vendors/{vid}", json={"gst_no": VENDOR_PAYLOAD["gst_no"], "city": "Pune"})
        assert r.status_code == 200 and r.json()["city"] == "Pune"

    def test_delete_removes_record(self, auth_client):
        vid = auth_client.post("/vendors", json=VENDOR_PAYLOAD).json()["id"]
        assert auth_client.delete(f"/vendors/{vid}").status_code == 204
        assert auth_client.get(f"/vendors/{vid}").status_code == 404

    def test_delete_missing_vendor_404(self, auth_client):
        assert auth_client.delete("/vendors/999999").status_code == 404

    def test_delete_requires_auth(self, client):
        assert client.delete("/vendors/1").status_code == 401


class TestCustomerUpdateDelete:
    def test_patch_and_delete(self, auth_client):
        cid = auth_client.post("/customers", json=CUSTOMER_PAYLOAD).json()["id"]
        r = auth_client.patch(f"/customers/{cid}", json={"region": "South", "salesperson": "K"})
        assert r.status_code == 200
        assert r.json()["region"] == "South" and r.json()["salesperson"] == "K"
        assert r.json()["company_name"] == CUSTOMER_PAYLOAD["company_name"]

        assert auth_client.delete(f"/customers/{cid}").status_code == 204
        assert auth_client.get(f"/customers/{cid}").status_code == 404

    def test_patch_gstin_clash_409(self, auth_client):
        a = auth_client.post("/customers", json=CUSTOMER_PAYLOAD).json()["id"]
        b = auth_client.post("/customers", json={**CUSTOMER_PAYLOAD, "gst_registration_number": "29AAAAA0000A1Z5", "company_name": "B"}).json()["id"]
        r = auth_client.patch(f"/customers/{b}", json={"gst_registration_number": CUSTOMER_PAYLOAD["gst_registration_number"]})
        assert r.status_code == 409
        assert r.json()["detail"]["existing_customer_id"] == a
