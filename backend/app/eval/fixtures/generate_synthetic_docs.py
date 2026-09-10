"""
Generate synthetic GST REG-06 / Udyam / cancelled-cheque PDFs for OCR testing.

WHY THIS EXISTS
---------------
The first synthetic set (synthetic_vendor_forms_all_20) rendered every field as
a generic two-column "label ......... value" grid. That layout does not exist in
any real vendor document, and it broke extraction for a reason that told us
nothing about extraction quality: the label->value gap was ~21 label-heights,
where field_dictionary's max_distance tops out at 15.0. Every field missed, and
the miss was an artefact of the fixture, not of the pipeline.

These fixtures instead replicate the ACTUAL layouts of the three documents in
app/uploads/09e0ff7aca -- the same table structure, label wording, page count,
annexures and column geometry -- so that a field which misses here misses for a
reason worth fixing.

WHAT IS AND ISN'T FAITHFUL
--------------------------
Faithful:   layout, table geometry, label wording, page/annexure structure,
            font families and relative sizes, the run-on Annexure A address line.
Not faithful: the cheque. The real cheque is a SCAN -- guilloche security
            background, handwriting, rotated date stamp, MICR font, JPEG noise.
            Its OCR text is heavily mangled ("1C1C0006278", "Hlndu.tan Rord").
            This generator emits a CLEAN VECTOR cheque, which will OCR nearly
            perfectly. It therefore tests cheque field layout and label matching,
            and does NOT test OCR robustness on degraded scans. Do not read a
            good cheque score here as evidence the pipeline handles real cheques.

ALL VALUES ARE FABRICATED. The GSTINs, PANs, IFSCs and account numbers are
structurally valid (so format validators actually engage) but belong to no real
entity. These files are test fixtures and have no legal validity.

USAGE
-----
    python -m app.eval.fixtures.generate_synthetic_docs --out <dir>

Requires Google Chrome (used headless for HTML->PDF). No new Python packages.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# CHROME
# ---------------------------------------------------------------------------

_CHROME_CANDIDATES = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    "/usr/bin/google-chrome",
    "/usr/bin/chromium",
]


def find_chrome() -> str:
    for cand in _CHROME_CANDIDATES:
        if Path(cand).exists():
            return cand
    found = shutil.which("chrome") or shutil.which("chromium") or shutil.which("google-chrome")
    if found:
        return found
    raise SystemExit(
        "No Chrome/Chromium/Edge binary found. These fixtures are rendered with "
        "headless Chrome; install Chrome or edit _CHROME_CANDIDATES."
    )


def rasterize_pdf(pdf_path: Path, dpi: int = 200) -> None:
    """Rewrite a vector PDF in place as page images, destroying its text layer.

    WHY: document_loader treats a page with >= MIN_TEXT_LAYER_CHARS (40) of
    embedded text as digital and reads it with pdfium, skipping OCR entirely.
    Chrome emits a full text layer, so a fixture left as-is silently tests the
    `pdf_text` path.

    That is correct for the GST certificate -- the real REG-06 IS a digital PDF
    and takes `pdf_text` in production. It is wrong for the Udyam certificate
    and the cheque, which are scans and take `ocr` in production. Rasterising
    those two puts them on the same code path as the documents they stand in
    for; without this the fixtures cannot exercise OCR at all.

    200 dpi matches RENDER_DPI, the tuned PaddleOCR render resolution.
    """
    import pypdfium2 as pdfium
    from PIL import Image

    scale = dpi / 72.0
    src = pdfium.PdfDocument(str(pdf_path))
    try:
        # .copy() detaches each frame from pdfium's buffer, so the document can
        # be closed before we overwrite the file we are reading from. Without
        # it Windows refuses the write with PermissionError.
        images: list[Image.Image] = [
            src[i].render(scale=scale).to_pil().convert("RGB").copy()
            for i in range(len(src))
        ]
    finally:
        src.close()

    if not images:
        raise RuntimeError(f"{pdf_path.name}: nothing to rasterize")

    # Write beside the target and swap. Writing directly to pdf_path can still
    # hit PermissionError on Windows while pdfium's mapping of the file it just
    # read is being released; os.replace is atomic and avoids the race.
    tmp_out = pdf_path.with_suffix(".raster.tmp")
    images[0].save(
        tmp_out, "PDF", resolution=float(dpi),
        save_all=True, append_images=images[1:],
    )
    for img in images:
        img.close()
    try:
        os.replace(tmp_out, pdf_path)
    except PermissionError as exc:
        tmp_out.unlink(missing_ok=True)
        raise SystemExit(
            f"Cannot replace {pdf_path} -- the file is open in another program "
            f"(a PDF viewer will hold an exclusive lock on Windows). Close it "
            f"and re-run, or generate into a fresh --out directory.\n  {exc}"
        ) from exc


def html_to_pdf(html: str, out_path: Path, chrome: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "doc.html"
        src.write_text(html, encoding="utf-8")
        profile = Path(tmp) / "profile"
        cmd = [
            chrome,
            "--headless",
            "--disable-gpu",
            "--no-sandbox",
            "--no-pdf-header-footer",
            f"--user-data-dir={profile}",
            f"--print-to-pdf={out_path}",
            src.as_uri(),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if not out_path.exists():
            raise RuntimeError(
                f"Chrome failed to render {out_path.name}:\n"
                f"stdout: {proc.stdout}\nstderr: {proc.stderr}"
            )


# ---------------------------------------------------------------------------
# VENDOR DATA
# ---------------------------------------------------------------------------

@dataclass
class Vendor:
    vid: str
    legal_name: str
    trade_name: str
    gstin: str
    pan: str
    udyam: str

    # GST page-1 "Address of Principal Place of Business" -- captioned sub-fields
    premises: str
    road: str
    city_town: str
    district: str
    state: str
    pin: str
    # Udyam prints a separate "Block" cell -- in the real certificate this is a
    # plot/holding number ("31/1"), NOT a word from the premises string.
    block: str
    premises_building: str

    # GST Annexure A -- ONE run-on line, the input the segmenter exists for
    additional_address: str

    constitution: str
    major_activity: str
    org_type: str
    enterprise_type: str

    bank_name: str
    ifsc: str
    account_number: str
    branch_line1: str
    branch_line2: str

    mobile: str
    email: str
    incorporated: str
    commenced: str
    udyam_date: str
    gst_granted: str
    dic: str
    msme_dfo: str
    directors: list[tuple[str, str, str]] = field(default_factory=list)

    # segmentation shape this vendor is meant to exercise
    address_case: str = ""

    # ---- combined-address mode ------------------------------------------
    # When set, GST page 1 prints the address as ONE uncaptioned blob instead
    # of the captioned Building/Road/City/District/State/PIN sub-fields. That
    # is what forces semantic_engine._resolve_combined_address to actually run
    # the segmenter: its early-bail fires when city AND state AND pin AND
    # address_2 are already populated from captions, which is exactly what the
    # captioned layout produces. Vendors in the segmentation set below use
    # this mode so the address lines are genuinely under test rather than
    # copied out of separate cells.
    combined_address: str = ""

    # Expected address_1..4 for combined_address, reasoned from the source
    # text -- NOT copied from segmenter output. Where the segmenter currently
    # disagrees, the fixture is meant to fail and record the disagreement.
    expect_lines: tuple[str, str, str, str] = ("", "", "", "")
    expect_city: str = ""
    expect_state: str = ""
    expect_pin: str = ""

    @property
    def udyam_premises(self) -> str:
        """Flat/Door/Block No. cell for the Udyam table.

        Combined-address vendors leave `premises` blank (their address lives
        entirely in the GST combined line, not in discrete fields), and an
        EMPTY table cell here is not a real-world shape: an actual filled-in
        Udyam form always has something in this column. An empty cell means
        OCR finds no span to align on, so a neighbour search from a nearby
        caption in the same row falls back to a much looser distance-only
        match and can grab a WRONG COLUMN's value from the row below.
        Deliberately generic filler ("Ground Floor"), not a fragment of the
        vendor's own combined address -- reusing a real fragment risked the
        filler itself starting with a keyword the field matcher's OWN
        dictionary recognises ("PART C" begins with the label-like word
        "Part"), recreating the same collision one column over."""
        return self.premises or "Ground"

    @property
    def udyam_road(self) -> str:
        """Road/Street/Lane cell for the Udyam table. Same reasoning as
        udyam_premises -- generic, not derived from the vendor's own address,
        specifically so it cannot coincidentally start or end with a word this
        project's own field/keyword dictionaries treat as significant."""
        return self.road or "Central"

    @property
    def udyam_district_cell(self) -> str:
        """District cell contents for the Udyam address tables. A real Udyam
        certificate prints "<DISTRICT> , Pin <NNNNNN>" in one cell. When this
        vendor has NO pin (the combined-address gu_*/seg_* set -- their
        addresses carry no PIN at all), rendering the bare caption word "Pin"
        with nothing after it made OCR read the cell as "<DISTRICT>,Pin", and
        the city resolver then extracted "Bengaluru,pin" as the city. Drop the
        ", Pin ..." tail entirely when there is no pin to show."""
        d = self.district.upper()
        return f"{d} , Pin {self.pin}" if self.pin else d


VENDORS: list[Vendor] = [
    Vendor(
        vid="vendor_01",
        legal_name="ARANYA PRECISION TOOLS PVT LTD",
        trade_name="ARANYA PRECISION",
        gstin="19AACCA1234M1ZP",
        pan="AACCA1234M",
        udyam="UDYAM-WB-10-0004417",
        premises="2ND FLOOR",
        road="14/2 BONDEL ROAD",
        city_town="KOLKATA",
        district="Kolkata",
        state="West Bengal",
        pin="700019",
        block="14/2",
        premises_building="Aranya Tower",
        # multi-locality rural run: the worked-example shape (4 locality frags)
        additional_address=(
            "3RD FLOOR, PART B BLOCK A, ARANYA INDUSTRIAL LOGISTIC PARK, BAKSARA, "
            "PODRAH, ANDUL, Jagadishpur, Howrah, West Bengal, 711302"
        ),
        address_case="rural multi-locality run (4 locality fragments)",
        constitution="Private Limited Company",
        major_activity="MANUFACTURING",
        org_type="Private Limited Company",
        enterprise_type="Small",
        bank_name="ICICI Bank",
        ifsc="ICIC0006278",
        account_number="627851004412",
        branch_line1="Kolkata - Gariahat Branch",
        branch_line2="2/3, Hindustan Road, Gariahat, Kolkata, West Bengal-700029",
        mobile="9748112207",
        email="accounts@aranyaprecision.com",
        incorporated="12/03/1994",
        commenced="01/07/1994",
        udyam_date="22/09/2020",
        gst_granted="18/02/2024",
        dic="KOLKOTA ( WEST BENGAL )",
        msme_dfo="KOLKATA ( WEST BENGAL )",
        directors=[
            ("SUJATA ARANYA MITRA", "DIRECTOR", "West Bengal"),
            ("ANIRBAN MITRA", "DIRECTOR", "West Bengal"),
        ],
    ),
    Vendor(
        vid="vendor_02",
        legal_name="VATVA POLYMER INDUSTRIES PVT LTD",
        trade_name="VATVA POLYMERS",
        gstin="24AABCV5821L1ZQ",
        pan="AABCV5821L",
        udyam="UDYAM-GJ-01-0011903",
        premises="PLOT NO. 47",
        road="PHASE II ROAD",
        city_town="AHMEDABAD",
        district="Ahmedabad",
        state="Gujarat",
        pin="382445",
        block="47",
        premises_building="Vatva Polymer Works",
        # industrial estate, GIDC keyword
        additional_address=(
            "SHOP NO. 18, PART B, GROUND LEVEL, SHREE GANESH COMPLEX, "
            "VATVA GIDC, Ahmedabad, Gujarat, 382445"
        ),
        address_case="industrial estate (GIDC keyword)",
        constitution="Private Limited Company",
        major_activity="MANUFACTURING",
        org_type="Private Limited Company",
        enterprise_type="Small",
        bank_name="HDFC Bank",
        ifsc="HDFC0001246",
        account_number="50200041229317",
        branch_line1="Ahmedabad - Vatva Branch",
        branch_line2="Ground Floor, Sarvoday Complex, Vatva GIDC, Ahmedabad, Gujarat-382445",
        mobile="9825443310",
        email="finance@vatvapolymer.co.in",
        incorporated="04/08/2001",
        commenced="19/11/2001",
        udyam_date="11/08/2020",
        gst_granted="27/03/2023",
        dic="AHMEDABAD ( GUJARAT )",
        msme_dfo="AHMEDABAD ( GUJARAT )",
        directors=[
            ("RAMESHBHAI K PATEL", "DIRECTOR", "Gujarat"),
            ("NILAM R PATEL", "DIRECTOR", "Gujarat"),
        ],
    ),
    Vendor(
        vid="vendor_03",
        legal_name="ORBIT LOGISTICS SOLUTIONS PVT LTD",
        trade_name="ORBIT LOGISTICS",
        gstin="27AAECO3391J1ZR",
        pan="AAECO3391J",
        udyam="UDYAM-MH-19-0027781",
        premises="UNIT 5A",
        road="NH-8 SERVICE ROAD",
        city_town="BHIWANDI",
        district="Thane",
        state="Maharashtra",
        pin="421302",
        block="5A",
        premises_building="Orbit Logistics Hub",
        # urban comma-separated, tower/hub, highway fragment
        additional_address=(
            "UNIT 5A, 2ND FLOOR, TOWER 3, ORBIT LOGISTICS HUB, NH-8, "
            "Bhiwandi, Thane, Maharashtra, 421302"
        ),
        address_case="urban comma-separated with highway fragment",
        constitution="Private Limited Company",
        major_activity="SERVICES",
        org_type="Private Limited Company",
        enterprise_type="Medium",
        bank_name="Axis Bank",
        ifsc="UTIB0000391",
        account_number="391010200013844",
        branch_line1="Thane - Bhiwandi Branch",
        branch_line2="Shop No 4, Kalyan Road, Bhiwandi, Thane, Maharashtra-421302",
        mobile="9820117733",
        email="ap@orbitlogistics.in",
        incorporated="27/01/2011",
        commenced="15/04/2011",
        udyam_date="03/10/2020",
        gst_granted="09/06/2022",
        dic="THANE ( MAHARASHTRA )",
        msme_dfo="MUMBAI ( MAHARASHTRA )",
        directors=[
            ("KEDAR V DESHMUKH", "DIRECTOR", "Maharashtra"),
            ("SNEHA KULKARNI", "DIRECTOR", "Maharashtra"),
        ],
    ),
    Vendor(
        vid="vendor_04",
        legal_name="PEENYA CONTROL SYSTEMS PVT LTD",
        trade_name="PEENYA CONTROLS",
        gstin="29AADCP7742H1ZS",
        pan="AADCP7742H",
        udyam="UDYAM-KR-03-0009925",
        premises="BUILDING D",
        road="4TH PHASE MAIN ROAD",
        city_town="BENGALURU",
        district="Bengaluru Urban",
        state="Karnataka",
        pin="560058",
        block="12",
        premises_building="Peenya Control House",
        # COMMA-LESS run-on: the low-confidence fallback path
        additional_address=(
            "BUILDING D PART C UNIT NO 12 1ST FLOOR PEENYA INDUSTRIAL ESTATE "
            "Bengaluru Karnataka 560058"
        ),
        address_case="comma-less run-on (injection fallback)",
        constitution="Private Limited Company",
        major_activity="MANUFACTURING",
        org_type="Private Limited Company",
        enterprise_type="Small",
        bank_name="Kotak Mahindra Bank",
        ifsc="KKBK0008065",
        account_number="806511004473",
        branch_line1="Bengaluru - Peenya Branch",
        branch_line2="No 8, 4th Phase, Peenya Industrial Area, Bengaluru, Karnataka-560058",
        mobile="9845220916",
        email="accounts@peenyacontrols.com",
        incorporated="09/09/2005",
        commenced="02/01/2006",
        udyam_date="17/09/2020",
        gst_granted="14/11/2021",
        dic="BENGALURU ( KARNATAKA )",
        msme_dfo="BENGALURU ( KARNATAKA )",
        directors=[
            ("H R SRINIVASA MURTHY", "DIRECTOR", "Karnataka"),
            ("LAKSHMI SRINIVASAN", "DIRECTOR", "Karnataka"),
        ],
    ),
    Vendor(
        vid="vendor_05",
        legal_name="NARMADA ENGINEERING WORKS PVT LTD",
        trade_name="NARMADA ENGG",
        gstin="23AAGCN6614F1ZT",
        pan="AAGCN6614F",
        udyam="UDYAM-MP-23-0003118",
        premises="GAT NO. 212",
        road="VILLAGE ROAD",
        city_town="DEWAS",
        district="Dewas",
        state="Madhya Pradesh",
        pin="455001",
        block="212",
        premises_building="Narmada Works",
        # explicit village/PO keywords -- the gazetteer-free evidence path
        additional_address=(
            "PLOT 86, PART B, BLOCK C, 2ND FLOOR, KALINDI INDUSTRIAL ESTATE, "
            "VILL- BAROTHA, P.O. NAGDA, Dewas, Madhya Pradesh, 455001"
        ),
        address_case="explicit VILL-/P.O. keywords",
        constitution="Private Limited Company",
        major_activity="MANUFACTURING",
        org_type="Private Limited Company",
        enterprise_type="Small",
        bank_name="Bank of India",
        ifsc="BKID0004035",
        account_number="403530100000914",
        branch_line1="Dewas - Industrial Area Branch",
        branch_line2="AB Road, Industrial Area, Dewas, Madhya Pradesh-455001",
        mobile="9827331104",
        email="works@narmadaengg.in",
        incorporated="21/06/1998",
        commenced="10/10/1998",
        udyam_date="29/09/2020",
        gst_granted="05/05/2023",
        dic="DEWAS ( MADHYA PRADESH )",
        msme_dfo="INDORE ( MADHYA PRADESH )",
        directors=[
            ("PRAKASH CHANDRA JOSHI", "DIRECTOR", "Madhya Pradesh"),
            ("MEENA JOSHI", "DIRECTOR", "Madhya Pradesh"),
        ],
    ),
]



# ---------------------------------------------------------------------------
# SEGMENTATION SET
# ---------------------------------------------------------------------------
# These vendors print the address on GST page 1 as ONE uncaptioned blob, so
# semantic_engine._resolve_combined_address actually runs the segmenter (the
# captioned layout used by VENDORS above populates city/state/pin/address_2
# directly and hits the early-bail instead).
#
# `expect_lines` is reasoned from the source text, NOT copied from segmenter
# output:
#     line 1  premises/unit/floor/block designators  ("where in the building")
#     line 2  the named property or estate           ("which building")
#     line 3  locality / area                        ("where in the city")
# City and state are peeled to their own fields and never appear in a line.
#
# Some of these are EXPECTED TO FAIL against the current implementation. That
# is the point: a fixture whose ground truth is copied from the code under test
# can only ever score 100% and proves nothing. Known disagreements at the time
# of writing are listed in the README.

SEGMENTATION_VENDORS: list[Vendor] = [
    Vendor(
        vid="seg_01",
        legal_name="SAI INDUSTRIAL FABRICATORS PVT LTD",
        trade_name="SAI INDUSTRIAL FABRICATORS",
        gstin="06AABCD1037H1ZB",
        pan="AABCD1037H",
        udyam="UDYAM-XX-01-0004007",
        premises="", road="",
        city_town="GURUGRAM", district="Gurugram", state="Haryana", pin="122015",
        block="", premises_building="",
        additional_address="",
        combined_address="FLAT NO. 302, BLOCK C, 3RD FLOOR, SAI INDUSTRIAL PARK, SECTOR 18, GURUGRAM, HARYANA",
        expect_lines=(
            "FLAT NO. 302, BLOCK C, 3RD FLOOR",
            "SAI INDUSTRIAL PARK",
            "SECTOR 18",
            "",
        ),
        expect_city="Gurugram", expect_state="Haryana", expect_pin="",
        address_case="premises trio + named park + sector locality",
        constitution="Private Limited Company",
        major_activity="MANUFACTURING",
        org_type="Private Limited Company",
        enterprise_type="Small",
        bank_name="HDFC Bank",
        ifsc="HDFC0001013",
        account_number="600000000811",
        branch_line1="Gurugram - Main Branch",
        branch_line2="Ground Floor, Commercial Complex, Gurugram, Haryana-122015",
        mobile="9800111111",
        email="accounts@sai01.co.in",
        incorporated="11/02/2001",
        commenced="02/02/2001",
        udyam_date="11/09/2020",
        gst_granted="11/02/2021",
        dic="GURUGRAM ( HARYANA )",
        msme_dfo="GURUGRAM ( HARYANA )",
        directors=[
            ("SAI PROMOTER", "DIRECTOR", "Haryana"),
        ],
    ),
    Vendor(
        vid="seg_02",
        legal_name="CHAKAN AUTO COMPONENTS PVT LTD",
        trade_name="CHAKAN AUTO COMPONENTS",
        gstin="27AACCG1074O1ZC",
        pan="AACCG1074O",
        udyam="UDYAM-XX-02-0004014",
        premises="", road="",
        city_town="PUNE", district="Pune", state="Maharashtra", pin="410501",
        block="", premises_building="",
        additional_address="",
        combined_address="PLOT NO. 47, INDUSTRIAL AREA PHASE II, BUILDING B, GROUND FLOOR, CHAKAN, PUNE, MAHARASHTRA",
        expect_lines=(
            "PLOT NO. 47, INDUSTRIAL AREA PHASE II",
            "BUILDING B, GROUND FLOOR",
            "CHAKAN",
            "",
        ),
        expect_city="Pune", expect_state="Maharashtra", expect_pin="",
        address_case="estate-before-building ordering (out of narrow-to-broad order)",
        constitution="Private Limited Company",
        major_activity="MANUFACTURING",
        org_type="Private Limited Company",
        enterprise_type="Small",
        bank_name="Axis Bank",
        ifsc="UTIB0001026",
        account_number="600000001622",
        branch_line1="Pune - Main Branch",
        branch_line2="Ground Floor, Commercial Complex, Pune, Maharashtra-410501",
        mobile="9800222222",
        email="accounts@chakan02.co.in",
        incorporated="12/03/2002",
        commenced="03/03/2002",
        udyam_date="12/09/2020",
        gst_granted="12/03/2022",
        dic="PUNE ( MAHARASHTRA )",
        msme_dfo="PUNE ( MAHARASHTRA )",
        directors=[
            ("CHAKAN PROMOTER", "DIRECTOR", "Maharashtra"),
        ],
    ),
    Vendor(
        vid="seg_03",
        legal_name="ORBIT LOGISTICS SOLUTIONS PVT LTD",
        trade_name="ORBIT LOGISTICS SOLUTIONS",
        gstin="27AADCJ1111V1ZD",
        pan="AADCJ1111V",
        udyam="UDYAM-XX-03-0004021",
        premises="", road="",
        city_town="THANE", district="Thane", state="Maharashtra", pin="421302",
        block="", premises_building="",
        additional_address="",
        combined_address="UNIT 5A, 2ND FLOOR, TOWER 3, ORBIT LOGISTICS HUB, NH-8, BHIWANDI, THANE, MAHARASHTRA",
        expect_lines=(
            "UNIT 5A, 2ND FLOOR, TOWER 3",
            "ORBIT LOGISTICS HUB",
            "NH-8, BHIWANDI",
            "",
        ),
        expect_city="Thane", expect_state="Maharashtra", expect_pin="",
        address_case="highway fragment + town before district city",
        constitution="Private Limited Company",
        major_activity="MANUFACTURING",
        org_type="Private Limited Company",
        enterprise_type="Small",
        bank_name="State Bank of India",
        ifsc="SBIN0001039",
        account_number="600000002433",
        branch_line1="Thane - Main Branch",
        branch_line2="Ground Floor, Commercial Complex, Thane, Maharashtra-421302",
        mobile="9800333333",
        email="accounts@orbit03.co.in",
        incorporated="13/04/2003",
        commenced="04/04/2003",
        udyam_date="13/09/2020",
        gst_granted="13/04/2023",
        dic="THANE ( MAHARASHTRA )",
        msme_dfo="THANE ( MAHARASHTRA )",
        directors=[
            ("ORBIT PROMOTER", "DIRECTOR", "Maharashtra"),
        ],
    ),
    Vendor(
        vid="seg_04",
        legal_name="SHREE GANESH POLYMERS PVT LTD",
        trade_name="SHREE GANESH POLYMERS",
        gstin="24AAECM1148C1ZE",
        pan="AAECM1148C",
        udyam="UDYAM-XX-04-0004028",
        premises="", road="",
        city_town="AHMEDABAD", district="Ahmedabad", state="Gujarat", pin="382445",
        block="", premises_building="",
        additional_address="",
        combined_address="SHOP NO. 18, PART B, GROUND LEVEL, SHREE GANESH COMPLEX, VATVA GIDC, AHMEDABAD, GUJARAT",
        expect_lines=(
            "SHOP NO. 18, PART B, GROUND LEVEL",
            "SHREE GANESH COMPLEX",
            "VATVA GIDC",
            "",
        ),
        expect_city="Ahmedabad", expect_state="Gujarat", expect_pin="",
        address_case="GIDC estate as locality tail",
        constitution="Private Limited Company",
        major_activity="MANUFACTURING",
        org_type="Private Limited Company",
        enterprise_type="Small",
        bank_name="Bank of Baroda",
        ifsc="BARB0001052",
        account_number="600000003244",
        branch_line1="Ahmedabad - Main Branch",
        branch_line2="Ground Floor, Commercial Complex, Ahmedabad, Gujarat-382445",
        mobile="9800444444",
        email="accounts@shree04.co.in",
        incorporated="14/05/2004",
        commenced="05/05/2004",
        udyam_date="14/09/2020",
        gst_granted="14/05/2020",
        dic="AHMEDABAD ( GUJARAT )",
        msme_dfo="AHMEDABAD ( GUJARAT )",
        directors=[
            ("SHREE PROMOTER", "DIRECTOR", "Gujarat"),
        ],
    ),
    Vendor(
        vid="seg_05",
        legal_name="EASTERN BUSINESS SYSTEMS PVT LTD",
        trade_name="EASTERN BUSINESS SYSTEMS",
        gstin="19AAFCP1185J1ZF",
        pan="AAFCP1185J",
        udyam="UDYAM-XX-05-0004035",
        premises="", road="",
        city_town="KOLKATA", district="Kolkata", state="West Bengal", pin="700091",
        block="", premises_building="",
        additional_address="",
        combined_address="4TH FLOOR, WING A, UNIT 401, EASTERN BUSINESS PARK, SALT LAKE, KOLKATA, WEST BENGAL",
        expect_lines=(
            "4TH FLOOR, WING A, UNIT 401",
            "EASTERN BUSINESS PARK",
            "SALT LAKE",
            "",
        ),
        expect_city="Kolkata", expect_state="West Bengal", expect_pin="",
        address_case="floor-first ordering + two-word locality",
        constitution="Private Limited Company",
        major_activity="MANUFACTURING",
        org_type="Private Limited Company",
        enterprise_type="Small",
        bank_name="ICICI Bank",
        ifsc="ICIC0001065",
        account_number="600000004055",
        branch_line1="Kolkata - Main Branch",
        branch_line2="Ground Floor, Commercial Complex, Kolkata, West Bengal-700091",
        mobile="9800555555",
        email="accounts@eastern05.co.in",
        incorporated="15/06/2005",
        commenced="06/06/2005",
        udyam_date="15/09/2020",
        gst_granted="15/06/2021",
        dic="KOLKATA ( WEST BENGAL )",
        msme_dfo="KOLKATA ( WEST BENGAL )",
        directors=[
            ("EASTERN PROMOTER", "DIRECTOR", "West Bengal"),
        ],
    ),
    Vendor(
        vid="seg_06",
        legal_name="PEENYA CONTROL SYSTEMS PVT LTD",
        trade_name="PEENYA CONTROL SYSTEMS",
        gstin="29AAGCS1222Q1ZG",
        pan="AAGCS1222Q",
        udyam="UDYAM-XX-06-0004042",
        premises="", road="",
        city_town="BENGALURU", district="Bengaluru", state="Karnataka", pin="560058",
        block="", premises_building="",
        additional_address="",
        combined_address="BUILDING D, PART C, UNIT NO. 12, 1ST FLOOR, PEENYA INDUSTRIAL ESTATE, BENGALURU, KARNATAKA",
        expect_lines=(
            "BUILDING D, PART C, UNIT NO. 12, 1ST FLOOR",
            "PEENYA INDUSTRIAL ESTATE",
            "",
            "",
        ),
        expect_city="Bengaluru", expect_state="Karnataka", expect_pin="",
        address_case="building-first ordering, no separate locality",
        constitution="Private Limited Company",
        major_activity="MANUFACTURING",
        org_type="Private Limited Company",
        enterprise_type="Small",
        bank_name="HDFC Bank",
        ifsc="HDFC0001078",
        account_number="600000004866",
        branch_line1="Bengaluru - Main Branch",
        branch_line2="Ground Floor, Commercial Complex, Bengaluru, Karnataka-560058",
        mobile="9800666666",
        email="accounts@peenya06.co.in",
        incorporated="16/07/2006",
        commenced="07/07/2006",
        udyam_date="16/09/2020",
        gst_granted="16/07/2022",
        dic="BENGALURU ( KARNATAKA )",
        msme_dfo="BENGALURU ( KARNATAKA )",
        directors=[
            ("PEENYA PROMOTER", "DIRECTOR", "Karnataka"),
        ],
    ),
    Vendor(
        vid="seg_07",
        legal_name="SRI LAKSHMI ENGINEERING PVT LTD",
        trade_name="SRI LAKSHMI ENGINEERING",
        gstin="33AAHCV1259X1ZH",
        pan="AAHCV1259X",
        udyam="UDYAM-XX-07-0004049",
        premises="", road="",
        city_town="CHENNAI", district="Chennai", state="Tamil Nadu", pin="600053",
        block="", premises_building="",
        additional_address="",
        combined_address="PLOT NO. 112, BLOCK B, 2ND FLOOR, SRI LAKSHMI INDUSTRIAL COMPLEX, AMBATTUR, CHENNAI, TAMIL NADU",
        expect_lines=(
            "PLOT NO. 112, BLOCK B, 2ND FLOOR",
            "SRI LAKSHMI INDUSTRIAL COMPLEX",
            "AMBATTUR",
            "",
        ),
        expect_city="Chennai", expect_state="Tamil Nadu", expect_pin="",
        address_case="canonical narrow-to-broad",
        constitution="Private Limited Company",
        major_activity="MANUFACTURING",
        org_type="Private Limited Company",
        enterprise_type="Small",
        bank_name="Axis Bank",
        ifsc="UTIB0001091",
        account_number="600000005677",
        branch_line1="Chennai - Main Branch",
        branch_line2="Ground Floor, Commercial Complex, Chennai, Tamil Nadu-600053",
        mobile="9800777777",
        email="accounts@sri07.co.in",
        incorporated="17/08/2007",
        commenced="08/08/2007",
        udyam_date="17/09/2020",
        gst_granted="17/08/2023",
        dic="CHENNAI ( TAMIL NADU )",
        msme_dfo="CHENNAI ( TAMIL NADU )",
        directors=[
            ("SRI PROMOTER", "DIRECTOR", "Tamil Nadu"),
        ],
    ),
    Vendor(
        vid="seg_08",
        legal_name="METRO LOGISTICS INDIA PVT LTD",
        trade_name="METRO LOGISTICS INDIA",
        gstin="07AAICY1296E1ZI",
        pan="AAICY1296E",
        udyam="UDYAM-XX-08-0004056",
        premises="", road="",
        city_town="DELHI", district="Delhi", state="Delhi", pin="110039",
        block="", premises_building="",
        additional_address="",
        combined_address="UNIT NO. 7, BLOCK A, 5TH FLOOR, METRO LOGISTICS PARK, BAWANA INDUSTRIAL AREA, DELHI",
        expect_lines=(
            "UNIT NO. 7, BLOCK A, 5TH FLOOR",
            "METRO LOGISTICS PARK",
            "BAWANA INDUSTRIAL AREA",
            "",
        ),
        expect_city="Delhi", expect_state="Delhi", expect_pin="",
        address_case="city equals state (Delhi); estate must NOT be taken as the city",
        constitution="Private Limited Company",
        major_activity="MANUFACTURING",
        org_type="Private Limited Company",
        enterprise_type="Small",
        bank_name="State Bank of India",
        ifsc="SBIN0001104",
        account_number="600000006488",
        branch_line1="Delhi - Main Branch",
        branch_line2="Ground Floor, Commercial Complex, Delhi, Delhi-110039",
        mobile="9800888888",
        email="accounts@metro08.co.in",
        incorporated="18/09/2008",
        commenced="01/09/2008",
        udyam_date="18/09/2020",
        gst_granted="18/09/2020",
        dic="DELHI ( DELHI )",
        msme_dfo="DELHI ( DELHI )",
        directors=[
            ("METRO PROMOTER", "DIRECTOR", "Delhi"),
        ],
    ),
    Vendor(
        vid="seg_09",
        legal_name="SHIVAM FORGINGS PVT LTD",
        trade_name="SHIVAM FORGINGS",
        gstin="03AAJCB1333L1ZJ",
        pan="AAJCB1333L",
        udyam="UDYAM-XX-09-0004063",
        premises="", road="",
        city_town="LUDHIANA", district="Ludhiana", state="Punjab", pin="141010",
        block="", premises_building="",
        additional_address="",
        combined_address="1ST FLOOR, UNIT B-14, PHASE III, SHIVAM INDUSTRIAL ESTATE, FOCAL POINT, LUDHIANA, PUNJAB",
        expect_lines=(
            "1ST FLOOR, UNIT B-14, PHASE III",
            "SHIVAM INDUSTRIAL ESTATE",
            "FOCAL POINT",
            "",
        ),
        expect_city="Ludhiana", expect_state="Punjab", expect_pin="",
        address_case="FOCAL POINT locality (Punjab estate idiom)",
        constitution="Private Limited Company",
        major_activity="MANUFACTURING",
        org_type="Private Limited Company",
        enterprise_type="Small",
        bank_name="Bank of Baroda",
        ifsc="BARB0001117",
        account_number="600000007299",
        branch_line1="Ludhiana - Main Branch",
        branch_line2="Ground Floor, Commercial Complex, Ludhiana, Punjab-141010",
        mobile="9800999999",
        email="accounts@shivam09.co.in",
        incorporated="19/01/2000",
        commenced="02/10/2000",
        udyam_date="19/09/2020",
        gst_granted="19/01/2021",
        dic="LUDHIANA ( PUNJAB )",
        msme_dfo="LUDHIANA ( PUNJAB )",
        directors=[
            ("SHIVAM PROMOTER", "DIRECTOR", "Punjab"),
        ],
    ),
    Vendor(
        vid="seg_10",
        legal_name="CYBER TRADE VENTURES PVT LTD",
        trade_name="CYBER TRADE VENTURES",
        gstin="09AAKCE1370S1ZK",
        pan="AAKCE1370S",
        udyam="UDYAM-XX-10-0004070",
        premises="", road="",
        city_town="NOIDA", district="Noida", state="Uttar Pradesh", pin="201309",
        block="", premises_building="",
        additional_address="",
        combined_address="TOWER B, 6TH FLOOR, PART A, OFFICE NO. 603, CYBER TRADE CENTRE, NOIDA SECTOR 62, UTTAR PRADESH",
        expect_lines=(
            "TOWER B, 6TH FLOOR, PART A, OFFICE NO. 603",
            "CYBER TRADE CENTRE",
            "NOIDA SECTOR 62",
            "",
        ),
        expect_city="Noida", expect_state="Uttar Pradesh", expect_pin="",
        address_case="city embedded in locality string (NOIDA SECTOR 62)",
        constitution="Private Limited Company",
        major_activity="MANUFACTURING",
        org_type="Private Limited Company",
        enterprise_type="Small",
        bank_name="ICICI Bank",
        ifsc="ICIC0001130",
        account_number="600000008110",
        branch_line1="Noida - Main Branch",
        branch_line2="Ground Floor, Commercial Complex, Noida, Uttar Pradesh-201309",
        mobile="9801111110",
        email="accounts@cyber10.co.in",
        incorporated="20/02/2001",
        commenced="03/11/2001",
        udyam_date="20/09/2020",
        gst_granted="20/02/2022",
        dic="NOIDA ( UTTAR PRADESH )",
        msme_dfo="NOIDA ( UTTAR PRADESH )",
        directors=[
            ("CYBER PROMOTER", "DIRECTOR", "Uttar Pradesh"),
        ],
    ),
    Vendor(
        vid="seg_11",
        legal_name="MAHALAXMI INDUSTRIES PVT LTD",
        trade_name="MAHALAXMI INDUSTRIES",
        gstin="27AALCH1407Z1ZL",
        pan="AALCH1407Z",
        udyam="UDYAM-XX-11-0004077",
        premises="", road="",
        city_town="MUMBAI", district="Mumbai", state="Maharashtra", pin="400093",
        block="", premises_building="",
        additional_address="",
        combined_address="UNIT C-22, GROUND FLOOR, BLOCK D, MAHALAXMI INDUSTRIAL PARK, ANDHERI EAST, MUMBAI, MAHARASHTRA",
        expect_lines=(
            "UNIT C-22, GROUND FLOOR, BLOCK D",
            "MAHALAXMI INDUSTRIAL PARK",
            "ANDHERI EAST",
            "",
        ),
        expect_city="Mumbai", expect_state="Maharashtra", expect_pin="",
        address_case="directional locality suffix (EAST)",
        constitution="Private Limited Company",
        major_activity="MANUFACTURING",
        org_type="Private Limited Company",
        enterprise_type="Small",
        bank_name="HDFC Bank",
        ifsc="HDFC0001143",
        account_number="600000008921",
        branch_line1="Mumbai - Main Branch",
        branch_line2="Ground Floor, Commercial Complex, Mumbai, Maharashtra-400093",
        mobile="9801222221",
        email="accounts@mahalaxmi11.co.in",
        incorporated="21/03/2002",
        commenced="04/12/2002",
        udyam_date="21/09/2020",
        gst_granted="21/03/2023",
        dic="MUMBAI ( MAHARASHTRA )",
        msme_dfo="MUMBAI ( MAHARASHTRA )",
        directors=[
            ("MAHALAXMI PROMOTER", "DIRECTOR", "Maharashtra"),
        ],
    ),
    Vendor(
        vid="seg_12",
        legal_name="RING ROAD TEXTILES PVT LTD",
        trade_name="RING ROAD TEXTILES",
        gstin="24AAMCK1444G1ZM",
        pan="AAMCK1444G",
        udyam="UDYAM-XX-12-0004084",
        premises="", road="",
        city_town="SURAT", district="Surat", state="Gujarat", pin="395002",
        block="", premises_building="",
        additional_address="",
        combined_address="3RD FLOOR, BUILDING A, UNIT NO. 308, RING ROAD INDUSTRIAL COMPLEX, SURAT, GUJARAT",
        expect_lines=(
            "3RD FLOOR, BUILDING A, UNIT NO. 308",
            "RING ROAD INDUSTRIAL COMPLEX",
            "",
            "",
        ),
        expect_city="Surat", expect_state="Gujarat", expect_pin="",
        address_case="road-word inside a proper name (must not split)",
        constitution="Private Limited Company",
        major_activity="MANUFACTURING",
        org_type="Private Limited Company",
        enterprise_type="Small",
        bank_name="Axis Bank",
        ifsc="UTIB0001156",
        account_number="600000009732",
        branch_line1="Surat - Main Branch",
        branch_line2="Ground Floor, Commercial Complex, Surat, Gujarat-395002",
        mobile="9801333332",
        email="accounts@ring12.co.in",
        incorporated="22/04/2003",
        commenced="05/01/2003",
        udyam_date="22/09/2020",
        gst_granted="22/04/2020",
        dic="SURAT ( GUJARAT )",
        msme_dfo="SURAT ( GUJARAT )",
        directors=[
            ("RING PROMOTER", "DIRECTOR", "Gujarat"),
        ],
    ),
    Vendor(
        vid="seg_13",
        legal_name="KALINDI METAL WORKS PVT LTD",
        trade_name="KALINDI METAL WORKS",
        gstin="06AANCN1481N1ZN",
        pan="AANCN1481N",
        udyam="UDYAM-XX-13-0004091",
        premises="", road="",
        city_town="FARIDABAD", district="Faridabad", state="Haryana", pin="121003",
        block="", premises_building="",
        additional_address="",
        combined_address="PLOT 86, PART B, BLOCK C, 2ND FLOOR, KALINDI INDUSTRIAL ESTATE, FARIDABAD, HARYANA",
        expect_lines=(
            "PLOT 86, PART B, BLOCK C, 2ND FLOOR",
            "KALINDI INDUSTRIAL ESTATE",
            "",
            "",
        ),
        expect_city="Faridabad", expect_state="Haryana", expect_pin="",
        address_case="four premises designators in a row",
        constitution="Private Limited Company",
        major_activity="MANUFACTURING",
        org_type="Private Limited Company",
        enterprise_type="Small",
        bank_name="State Bank of India",
        ifsc="SBIN0001169",
        account_number="600000010543",
        branch_line1="Faridabad - Main Branch",
        branch_line2="Ground Floor, Commercial Complex, Faridabad, Haryana-121003",
        mobile="9801444443",
        email="accounts@kalindi13.co.in",
        incorporated="23/05/2004",
        commenced="06/02/2004",
        udyam_date="23/09/2020",
        gst_granted="23/05/2021",
        dic="FARIDABAD ( HARYANA )",
        msme_dfo="FARIDABAD ( HARYANA )",
        directors=[
            ("KALINDI PROMOTER", "DIRECTOR", "Haryana"),
        ],
    ),
    Vendor(
        vid="seg_14",
        legal_name="SRI DURGA TRADING PVT LTD",
        trade_name="SRI DURGA TRADING",
        gstin="07AAOCQ1518U1ZO",
        pan="AAOCQ1518U",
        udyam="UDYAM-XX-14-0004098",
        premises="", road="",
        city_town="DELHI", district="Delhi", state="Delhi", pin="110096",
        block="", premises_building="",
        additional_address="",
        combined_address="OFFICE NO. 405, 4TH FLOOR, TOWER A, SRI DURGA BUSINESS PARK, KONDLI, DELHI",
        expect_lines=(
            "OFFICE NO. 405, 4TH FLOOR, TOWER A",
            "SRI DURGA BUSINESS PARK",
            "KONDLI",
            "",
        ),
        expect_city="Delhi", expect_state="Delhi", expect_pin="",
        address_case="KONDLI locality vs Delhi city",
        constitution="Private Limited Company",
        major_activity="MANUFACTURING",
        org_type="Private Limited Company",
        enterprise_type="Small",
        bank_name="Bank of Baroda",
        ifsc="BARB0001182",
        account_number="600000011354",
        branch_line1="Delhi - Main Branch",
        branch_line2="Ground Floor, Commercial Complex, Delhi, Delhi-110096",
        mobile="9801555554",
        email="accounts@sri14.co.in",
        incorporated="24/06/2005",
        commenced="07/03/2005",
        udyam_date="24/09/2020",
        gst_granted="24/06/2022",
        dic="DELHI ( DELHI )",
        msme_dfo="DELHI ( DELHI )",
        directors=[
            ("SRI PROMOTER", "DIRECTOR", "Delhi"),
        ],
    ),
    Vendor(
        vid="seg_15",
        legal_name="HITECH LOGISTICS CENTRE PVT LTD",
        trade_name="HITECH LOGISTICS CENTRE",
        gstin="36AAPCT1555B1ZP",
        pan="AAPCT1555B",
        udyam="UDYAM-XX-15-0004105",
        premises="", road="",
        city_town="HYDERABAD", district="Hyderabad", state="Telangana", pin="500081",
        block="", premises_building="",
        additional_address="",
        combined_address="BLOCK E, UNIT 16, GROUND FLOOR, INDUSTRIAL LOGISTICS CENTRE, HITECH CITY, HYDERABAD, TELANGANA",
        expect_lines=(
            "BLOCK E, UNIT 16, GROUND FLOOR",
            "INDUSTRIAL LOGISTICS CENTRE",
            "HITECH CITY",
            "",
        ),
        expect_city="Hyderabad", expect_state="Telangana", expect_pin="",
        address_case="locality containing the word CITY",
        constitution="Private Limited Company",
        major_activity="MANUFACTURING",
        org_type="Private Limited Company",
        enterprise_type="Small",
        bank_name="ICICI Bank",
        ifsc="ICIC0001195",
        account_number="600000012165",
        branch_line1="Hyderabad - Main Branch",
        branch_line2="Ground Floor, Commercial Complex, Hyderabad, Telangana-500081",
        mobile="9801666665",
        email="accounts@hitech15.co.in",
        incorporated="25/07/2006",
        commenced="08/04/2006",
        udyam_date="25/09/2020",
        gst_granted="25/07/2023",
        dic="HYDERABAD ( TELANGANA )",
        msme_dfo="HYDERABAD ( TELANGANA )",
        directors=[
            ("HITECH PROMOTER", "DIRECTOR", "Telangana"),
        ],
    ),
    Vendor(
        vid="seg_16",
        legal_name="TALOJA CHEMICALS PVT LTD",
        trade_name="TALOJA CHEMICALS",
        gstin="27AAQCW1592I1ZQ",
        pan="AAQCW1592I",
        udyam="UDYAM-XX-16-0004112",
        premises="", road="",
        city_town="NAVI MUMBAI", district="Navi Mumbai", state="Maharashtra", pin="410208",
        block="", premises_building="",
        additional_address="",
        combined_address="2ND FLOOR, PART A, UNIT NO. 214, BUILDING C, TALOJA INDUSTRIAL AREA, NAVI MUMBAI, MAHARASHTRA",
        expect_lines=(
            "2ND FLOOR, PART A, UNIT NO. 214, BUILDING C",
            "TALOJA INDUSTRIAL AREA",
            "",
            "",
        ),
        expect_city="Navi Mumbai", expect_state="Maharashtra", expect_pin="",
        address_case="two-word city (NAVI MUMBAI)",
        constitution="Private Limited Company",
        major_activity="MANUFACTURING",
        org_type="Private Limited Company",
        enterprise_type="Small",
        bank_name="HDFC Bank",
        ifsc="HDFC0001208",
        account_number="600000012976",
        branch_line1="Navi Mumbai - Main Branch",
        branch_line2="Ground Floor, Commercial Complex, Navi Mumbai, Maharashtra-410208",
        mobile="9801777776",
        email="accounts@taloja16.co.in",
        incorporated="26/08/2007",
        commenced="01/05/2007",
        udyam_date="26/09/2020",
        gst_granted="26/08/2020",
        dic="NAVI MUMBAI ( MAHARASHTRA )",
        msme_dfo="NAVI MUMBAI ( MAHARASHTRA )",
        directors=[
            ("TALOJA PROMOTER", "DIRECTOR", "Maharashtra"),
        ],
    ),
    Vendor(
        vid="seg_17",
        legal_name="SARDAR INDUSTRIAL WORKS PVT LTD",
        trade_name="SARDAR INDUSTRIAL WORKS",
        gstin="24AARCZ1629P1ZR",
        pan="AARCZ1629P",
        udyam="UDYAM-XX-17-0004119",
        premises="", road="",
        city_town="VADODARA", district="Vadodara", state="Gujarat", pin="390010",
        block="", premises_building="",
        additional_address="",
        combined_address="UNIT D-09, 1ST FLOOR, BLOCK B, SARDAR INDUSTRIAL COMPLEX, MAKARPURA GIDC, VADODARA, GUJARAT",
        expect_lines=(
            "UNIT D-09, 1ST FLOOR, BLOCK B",
            "SARDAR INDUSTRIAL COMPLEX",
            "MAKARPURA GIDC",
            "",
        ),
        expect_city="Vadodara", expect_state="Gujarat", expect_pin="",
        address_case="named GIDC as locality tail",
        constitution="Private Limited Company",
        major_activity="MANUFACTURING",
        org_type="Private Limited Company",
        enterprise_type="Small",
        bank_name="Axis Bank",
        ifsc="UTIB0001221",
        account_number="600000013787",
        branch_line1="Vadodara - Main Branch",
        branch_line2="Ground Floor, Commercial Complex, Vadodara, Gujarat-390010",
        mobile="9801888887",
        email="accounts@sardar17.co.in",
        incorporated="27/09/2008",
        commenced="02/06/2008",
        udyam_date="27/09/2020",
        gst_granted="27/09/2021",
        dic="VADODARA ( GUJARAT )",
        msme_dfo="VADODARA ( GUJARAT )",
        directors=[
            ("SARDAR PROMOTER", "DIRECTOR", "Gujarat"),
        ],
    ),
    Vendor(
        vid="seg_18",
        legal_name="NORTH CITY LOGISTICS PVT LTD",
        trade_name="NORTH CITY LOGISTICS",
        gstin="19AASCC1666W1ZS",
        pan="AASCC1666W",
        udyam="UDYAM-XX-18-0004126",
        premises="", road="",
        city_town="KOLKATA", district="Kolkata", state="West Bengal", pin="700124",
        block="", premises_building="",
        additional_address="",
        combined_address="BUILDING B, 3RD FLOOR, PART C, UNIT 312, NORTH CITY LOGISTICS PARK, BARASAT ROAD, KOLKATA, WEST BENGAL",
        expect_lines=(
            "BUILDING B, 3RD FLOOR, PART C, UNIT 312",
            "NORTH CITY LOGISTICS PARK",
            "BARASAT ROAD",
            "",
        ),
        expect_city="Kolkata", expect_state="West Bengal", expect_pin="",
        address_case="building-first + genuine road as locality",
        constitution="Private Limited Company",
        major_activity="MANUFACTURING",
        org_type="Private Limited Company",
        enterprise_type="Small",
        bank_name="State Bank of India",
        ifsc="SBIN0001234",
        account_number="600000014598",
        branch_line1="Kolkata - Main Branch",
        branch_line2="Ground Floor, Commercial Complex, Kolkata, West Bengal-700124",
        mobile="9801999998",
        email="accounts@north18.co.in",
        incorporated="28/01/2000",
        commenced="03/07/2000",
        udyam_date="28/09/2020",
        gst_granted="28/01/2022",
        dic="KOLKATA ( WEST BENGAL )",
        msme_dfo="KOLKATA ( WEST BENGAL )",
        directors=[
            ("NORTH PROMOTER", "DIRECTOR", "West Bengal"),
        ],
    ),
    Vendor(
        vid="seg_19",
        legal_name="R K INDUSTRIAL ENTERPRISES PVT LTD",
        trade_name="R K INDUSTRIAL ENTERPRISES",
        gstin="27AATCF1703D1ZT",
        pan="AATCF1703D",
        udyam="UDYAM-XX-19-0004133",
        premises="", road="",
        city_town="BHIWANDI", district="Bhiwandi", state="Maharashtra", pin="421302",
        block="", premises_building="",
        additional_address="",
        combined_address="GROUND FLOOR, UNIT A-06, BLOCK F, R.K. INDUSTRIAL ESTATE, KALHER, BHIWANDI, MAHARASHTRA",
        expect_lines=(
            "GROUND FLOOR, UNIT A-06, BLOCK F",
            "R.K. INDUSTRIAL ESTATE",
            "KALHER",
            "",
        ),
        expect_city="Bhiwandi", expect_state="Maharashtra", expect_pin="",
        address_case="initials with dots in a proper name",
        constitution="Private Limited Company",
        major_activity="MANUFACTURING",
        org_type="Private Limited Company",
        enterprise_type="Small",
        bank_name="Bank of Baroda",
        ifsc="BARB0001247",
        account_number="600000015409",
        branch_line1="Bhiwandi - Main Branch",
        branch_line2="Ground Floor, Commercial Complex, Bhiwandi, Maharashtra-421302",
        mobile="9802111109",
        email="accounts@r19.co.in",
        incorporated="10/02/2001",
        commenced="04/08/2001",
        udyam_date="10/09/2020",
        gst_granted="10/02/2023",
        dic="BHIWANDI ( MAHARASHTRA )",
        msme_dfo="BHIWANDI ( MAHARASHTRA )",
        directors=[
            ("R PROMOTER", "DIRECTOR", "Maharashtra"),
        ],
    ),
    Vendor(
        vid="seg_20",
        legal_name="GREENFIELD BUSINESS SERVICES PVT LTD",
        trade_name="GREENFIELD BUSINESS SERVICES",
        gstin="29AAUCI1740K1ZU",
        pan="AAUCI1740K",
        udyam="UDYAM-XX-20-0004140",
        premises="", road="",
        city_town="BENGALURU", district="Bengaluru", state="Karnataka", pin="560068",
        block="", premises_building="",
        additional_address="",
        combined_address="5TH FLOOR, PART B, OFFICE 502, TOWER C, GREENFIELD BUSINESS PARK, HOSUR ROAD, BENGALURU, KARNATAKA",
        expect_lines=(
            "5TH FLOOR, PART B, OFFICE 502, TOWER C",
            "GREENFIELD BUSINESS PARK",
            "HOSUR ROAD",
            "",
        ),
        expect_city="Bengaluru", expect_state="Karnataka", expect_pin="",
        address_case="four premises designators + road locality",
        constitution="Private Limited Company",
        major_activity="MANUFACTURING",
        org_type="Private Limited Company",
        enterprise_type="Small",
        bank_name="ICICI Bank",
        ifsc="ICIC0001260",
        account_number="600000016220",
        branch_line1="Bengaluru - Main Branch",
        branch_line2="Ground Floor, Commercial Complex, Bengaluru, Karnataka-560068",
        mobile="9802222220",
        email="accounts@greenfield20.co.in",
        incorporated="11/03/2002",
        commenced="05/09/2002",
        udyam_date="11/09/2020",
        gst_granted="11/03/2020",
        dic="BENGALURU ( KARNATAKA )",
        msme_dfo="BENGALURU ( KARNATAKA )",
        directors=[
            ("GREENFIELD PROMOTER", "DIRECTOR", "Karnataka"),
        ],
    ),
]

# ---------------------------------------------------------------------------
# LONG-ADDRESS STRESS SET (gu_01..gu_20)
# ---------------------------------------------------------------------------
# 12-14 comma-separated fragments each -- roughly double the length of the
# seg_* set. Same combined-address GST shape (one uncaptioned line), so the
# same rasterisation requirement applies (see main()).
#
# `expect_lines` here is NOT independently reasoned the way vl*/seg_* are.
# With 12-14 fragments against a hard 4-line cap, there is no single obviously
# -correct 4-way grouping -- the cap forces the packer to merge REAL groups
# together, and which merges are "least bad" is itself a judgment call. These
# expectations are a SNAPSHOT of resolve_address_blob() output at the time
# this set was generated (2026-09-10, after the taxonomy/city fixes made
# earlier that session). Treat a mismatch here as "the merge/cap behaviour
# changed" and go look at what changed and whether it is an improvement --
# not as an automatic regression the way a vl*/seg_* mismatch would be.
#
# All city fields were captured directly from the resolver, catching a real
# bug in the process: "..., GHAZIPUR ROAD, ..., GAZIPUR, DELHI" (gu_14)
# resolved to city=Ghazipur before a fix landed in _split_known_city_prefix --
# it was peeling the known city name "Ghazipur" off the FRONT of the street
# segment "GHAZIPUR ROAD" without checking the remainder was locality-shaped.
# Fixed by requiring the remainder to classify as locality/village_po/unknown,
# which "ROAD" (thoroughfare) does not. Regression-tested in
# test_address_segmenter.py.

LONG_ADDRESS_VENDORS: list[Vendor] = [
    Vendor(
        vid="gu_01",
        legal_name="SAI PRECISION INDUSTRIES PVT LTD",
        trade_name="SAI PRECISION INDUSTRIES",
        gstin="06AABDF2041L1ZB",
        pan="AABDF2041L",
        udyam="UDYAM-XX-01-0005009",
        premises="", road="",
        city_town="GURUGRAM", district="Gurugram", state="Haryana", pin="",
        block="", premises_building="",
        additional_address="",
        combined_address="FLAT NO. 302, BLOCK C, 3RD FLOOR, SAI INDUSTRIAL PARK, BUILDING 4, SECTOR 18, SERVICE ROAD, UDYOG VIHAR EXTENSION, SIKANDERPUR, SARHOL, GURUGRAM, HARYANA",
        expect_lines=(
            "FLAT NO. 302, BLOCK C, 3RD FLOOR",
            "SAI INDUSTRIAL PARK, BUILDING 4",
            "SECTOR 18, SERVICE ROAD, UDYOG VIHAR EXTENSION",
            "SIKANDERPUR, SARHOL",
        ),
        expect_city="Gurugram", expect_state="Haryana", expect_pin="",
        address_case="stress: 12 comma-separated fragments, "
                     "medium confidence, cap-merge exercised",
        constitution="Private Limited Company",
        major_activity="MANUFACTURING",
        org_type="Private Limited Company",
        enterprise_type="Small",
        bank_name="HDFC Bank",
        ifsc="HDFC0002017",
        account_number="700000000733",
        branch_line1="Gurugram - Main Branch",
        branch_line2="Ground Floor, Commercial Complex, Gurugram, Haryana",
        mobile="9700121212",
        email="accounts@sai01.co.in",
        incorporated="11/02/1991",
        commenced="02/02/1991",
        udyam_date="11/09/2020",
        gst_granted="11/02/2021",
        dic="GURUGRAM ( HARYANA )",
        msme_dfo="GURUGRAM ( HARYANA )",
        directors=[
            ("SAI PROMOTER", "DIRECTOR", "Haryana"),
        ],
    ),
    Vendor(
        vid="gu_02",
        legal_name="CHAKAN AUTOMOTIVE COMPONENTS PVT LTD",
        trade_name="CHAKAN AUTOMOTIVE COMPONENTS",
        gstin="27AACDK2082W1ZC",
        pan="AACDK2082W",
        udyam="UDYAM-XX-02-0005018",
        premises="", road="",
        city_town="PUNE", district="Pune", state="Maharashtra", pin="",
        block="", premises_building="",
        additional_address="",
        combined_address="PLOT NO. 47, INDUSTRIAL AREA PHASE II, BUILDING B, GROUND FLOOR, UNIT 12, CHAKAN INDUSTRIAL ZONE, MIDC ROAD, MAHINDRA GATE, KHARABWADI, CHAKAN, PUNE, MAHARASHTRA",
        expect_lines=(
            "PLOT NO. 47, INDUSTRIAL AREA PHASE II, BUILDING B, GROUND FLOOR, UNIT 12",
            "CHAKAN INDUSTRIAL ZONE, MIDC ROAD",
            "MAHINDRA GATE, KHARABWADI",
            "CHAKAN",
        ),
        expect_city="Pune", expect_state="Maharashtra", expect_pin="",
        address_case="stress: 12 comma-separated fragments, "
                     "medium confidence, cap-merge exercised",
        constitution="Private Limited Company",
        major_activity="MANUFACTURING",
        org_type="Private Limited Company",
        enterprise_type="Small",
        bank_name="Axis Bank",
        ifsc="UTIB0002034",
        account_number="700000001466",
        branch_line1="Pune - Main Branch",
        branch_line2="Ground Floor, Commercial Complex, Pune, Maharashtra",
        mobile="9700242424",
        email="accounts@chakan02.co.in",
        incorporated="12/03/1992",
        commenced="03/03/1992",
        udyam_date="12/09/2020",
        gst_granted="12/03/2022",
        dic="PUNE ( MAHARASHTRA )",
        msme_dfo="PUNE ( MAHARASHTRA )",
        directors=[
            ("CHAKAN PROMOTER", "DIRECTOR", "Maharashtra"),
        ],
    ),
    Vendor(
        vid="gu_03",
        legal_name="ORBIT LOGISTICS SOLUTIONS PVT LTD",
        trade_name="ORBIT LOGISTICS SOLUTIONS",
        gstin="27AADDP2123H1ZD",
        pan="AADDP2123H",
        udyam="UDYAM-XX-03-0005027",
        premises="", road="",
        city_town="THANE", district="Thane", state="Maharashtra", pin="",
        block="", premises_building="",
        additional_address="",
        combined_address="UNIT 5A, 2ND FLOOR, TOWER 3, ORBIT LOGISTICS HUB, NH-8, SERVICE ROAD, ANJURPHATA INDUSTRIAL AREA, VAL, KALHER, BHIWANDI, THANE, MAHARASHTRA",
        expect_lines=(
            "UNIT 5A, 2ND FLOOR, TOWER 3",
            "ORBIT LOGISTICS HUB",
            "NH-8, SERVICE ROAD, ANJURPHATA INDUSTRIAL AREA",
            "VAL, KALHER, BHIWANDI",
        ),
        expect_city="Thane", expect_state="Maharashtra", expect_pin="",
        address_case="stress: 12 comma-separated fragments, "
                     "medium confidence, cap-merge exercised",
        constitution="Private Limited Company",
        major_activity="MANUFACTURING",
        org_type="Private Limited Company",
        enterprise_type="Small",
        bank_name="State Bank of India",
        ifsc="SBIN0002051",
        account_number="700000002199",
        branch_line1="Thane - Main Branch",
        branch_line2="Ground Floor, Commercial Complex, Thane, Maharashtra",
        mobile="9700363636",
        email="accounts@orbit03.co.in",
        incorporated="13/04/1993",
        commenced="04/04/1993",
        udyam_date="13/09/2020",
        gst_granted="13/04/2023",
        dic="THANE ( MAHARASHTRA )",
        msme_dfo="THANE ( MAHARASHTRA )",
        directors=[
            ("ORBIT PROMOTER", "DIRECTOR", "Maharashtra"),
        ],
    ),
    Vendor(
        vid="gu_04",
        legal_name="SHREE GANESH POLYMERS PVT LTD",
        trade_name="SHREE GANESH POLYMERS",
        gstin="24AAEDU2164S1ZE",
        pan="AAEDU2164S",
        udyam="UDYAM-XX-04-0005036",
        premises="", road="",
        city_town="AHMEDABAD", district="Ahmedabad", state="Gujarat", pin="",
        block="", premises_building="",
        additional_address="",
        combined_address="SHOP NO. 18, PART B, GROUND LEVEL, SHREE GANESH COMPLEX, VATVA GIDC, PHASE II, GIDC ROAD, MAHATMA GANDHI INDUSTRIAL ESTATE, ISANPUR, VATVA, AHMEDABAD, GUJARAT",
        expect_lines=(
            "SHOP NO. 18, PART B, GROUND LEVEL",
            "SHREE GANESH COMPLEX, VATVA GIDC, PHASE II",
            "GIDC ROAD, MAHATMA GANDHI INDUSTRIAL ESTATE",
            "ISANPUR, VATVA",
        ),
        expect_city="Ahmedabad", expect_state="Gujarat", expect_pin="",
        address_case="stress: 12 comma-separated fragments, "
                     "medium confidence, cap-merge exercised",
        constitution="Private Limited Company",
        major_activity="MANUFACTURING",
        org_type="Private Limited Company",
        enterprise_type="Small",
        bank_name="Bank of Baroda",
        ifsc="BARB0002068",
        account_number="700000002932",
        branch_line1="Ahmedabad - Main Branch",
        branch_line2="Ground Floor, Commercial Complex, Ahmedabad, Gujarat",
        mobile="9700484848",
        email="accounts@shree04.co.in",
        incorporated="14/05/1994",
        commenced="05/05/1994",
        udyam_date="14/09/2020",
        gst_granted="14/05/2020",
        dic="AHMEDABAD ( GUJARAT )",
        msme_dfo="AHMEDABAD ( GUJARAT )",
        directors=[
            ("SHREE PROMOTER", "DIRECTOR", "Gujarat"),
        ],
    ),
    Vendor(
        vid="gu_05",
        legal_name="EASTERN BUSINESS SYSTEMS PVT LTD",
        trade_name="EASTERN BUSINESS SYSTEMS",
        gstin="19AAFDZ2205D1ZF",
        pan="AAFDZ2205D",
        udyam="UDYAM-XX-05-0005045",
        premises="", road="",
        city_town="KOLKATA", district="Kolkata", state="West Bengal", pin="",
        block="", premises_building="",
        additional_address="",
        combined_address="4TH FLOOR, WING A, UNIT 401, EASTERN BUSINESS PARK, SALT LAKE, SECTOR V, TECHNOLOGY ROAD, BIDHANNAGAR INDUSTRIAL ZONE, KESTOPUR, KARUNAMOYEE, KOLKATA, WEST BENGAL",
        expect_lines=(
            "4TH FLOOR, WING A, UNIT 401",
            "EASTERN BUSINESS PARK",
            "SALT LAKE, SECTOR V, TECHNOLOGY ROAD, BIDHANNAGAR INDUSTRIAL ZONE",
            "KESTOPUR, KARUNAMOYEE",
        ),
        expect_city="Kolkata", expect_state="West Bengal", expect_pin="",
        address_case="stress: 12 comma-separated fragments, "
                     "medium confidence, cap-merge exercised",
        constitution="Private Limited Company",
        major_activity="MANUFACTURING",
        org_type="Private Limited Company",
        enterprise_type="Small",
        bank_name="ICICI Bank",
        ifsc="ICIC0002085",
        account_number="700000003665",
        branch_line1="Kolkata - Main Branch",
        branch_line2="Ground Floor, Commercial Complex, Kolkata, West Bengal",
        mobile="9700606060",
        email="accounts@eastern05.co.in",
        incorporated="15/06/1995",
        commenced="06/06/1995",
        udyam_date="15/09/2020",
        gst_granted="15/06/2021",
        dic="KOLKATA ( WEST BENGAL )",
        msme_dfo="KOLKATA ( WEST BENGAL )",
        directors=[
            ("EASTERN PROMOTER", "DIRECTOR", "West Bengal"),
        ],
    ),
    Vendor(
        vid="gu_06",
        legal_name="PEENYA CONTROL SYSTEMS PVT LTD",
        trade_name="PEENYA CONTROL SYSTEMS",
        gstin="29AAGDE2246O1ZG",
        pan="AAGDE2246O",
        udyam="UDYAM-XX-06-0005054",
        premises="", road="",
        city_town="BENGALURU", district="Bengaluru", state="Karnataka", pin="",
        block="", premises_building="",
        additional_address="",
        combined_address="BUILDING D, PART C, UNIT NO. 12, 1ST FLOOR, PEENYA INDUSTRIAL ESTATE, PHASE II, TUMKUR ROAD, INDUSTRIAL AREA, NANDINI LAYOUT, YESHWANTHPUR, PEENYA, BENGALURU, KARNATAKA",
        expect_lines=(
            "BUILDING D, PART C, UNIT NO. 12, 1ST FLOOR",
            "PEENYA INDUSTRIAL ESTATE, PHASE II",
            "TUMKUR ROAD, INDUSTRIAL AREA",
            "NANDINI LAYOUT, YESHWANTHPUR, PEENYA",
        ),
        expect_city="Bengaluru", expect_state="Karnataka", expect_pin="",
        address_case="stress: 13 comma-separated fragments, "
                     "medium confidence, cap-merge exercised",
        constitution="Private Limited Company",
        major_activity="MANUFACTURING",
        org_type="Private Limited Company",
        enterprise_type="Small",
        bank_name="HDFC Bank",
        ifsc="HDFC0002102",
        account_number="700000004398",
        branch_line1="Bengaluru - Main Branch",
        branch_line2="Ground Floor, Commercial Complex, Bengaluru, Karnataka",
        mobile="9700727272",
        email="accounts@peenya06.co.in",
        incorporated="16/07/1996",
        commenced="07/07/1996",
        udyam_date="16/09/2020",
        gst_granted="16/07/2022",
        dic="BENGALURU ( KARNATAKA )",
        msme_dfo="BENGALURU ( KARNATAKA )",
        directors=[
            ("PEENYA PROMOTER", "DIRECTOR", "Karnataka"),
        ],
    ),
    Vendor(
        vid="gu_07",
        legal_name="SRI LAKSHMI ENGINEERING PVT LTD",
        trade_name="SRI LAKSHMI ENGINEERING",
        gstin="33AAHDJ2287Z1ZH",
        pan="AAHDJ2287Z",
        udyam="UDYAM-XX-07-0005063",
        premises="", road="",
        city_town="CHENNAI", district="Chennai", state="Tamil Nadu", pin="",
        block="", premises_building="",
        additional_address="",
        combined_address="PLOT NO. 112, BLOCK B, 2ND FLOOR, SRI LAKSHMI INDUSTRIAL COMPLEX, AMBATTUR INDUSTRIAL ESTATE, PHASE III, MTH ROAD, ESTATE ROAD, PATTARAVAKKAM, MANNURPET, AMBATTUR, CHENNAI, TAMIL NADU",
        expect_lines=(
            "PLOT NO. 112, BLOCK B, 2ND FLOOR",
            "SRI LAKSHMI INDUSTRIAL COMPLEX, AMBATTUR INDUSTRIAL ESTATE, PHASE III",
            "MTH ROAD, ESTATE ROAD",
            "PATTARAVAKKAM, MANNURPET, AMBATTUR",
        ),
        expect_city="Chennai", expect_state="Tamil Nadu", expect_pin="",
        address_case="stress: 13 comma-separated fragments, "
                     "medium confidence, cap-merge exercised",
        constitution="Private Limited Company",
        major_activity="MANUFACTURING",
        org_type="Private Limited Company",
        enterprise_type="Small",
        bank_name="Axis Bank",
        ifsc="UTIB0002119",
        account_number="700000005131",
        branch_line1="Chennai - Main Branch",
        branch_line2="Ground Floor, Commercial Complex, Chennai, Tamil Nadu",
        mobile="9700848484",
        email="accounts@sri07.co.in",
        incorporated="17/08/1997",
        commenced="08/08/1997",
        udyam_date="17/09/2020",
        gst_granted="17/08/2023",
        dic="CHENNAI ( TAMIL NADU )",
        msme_dfo="CHENNAI ( TAMIL NADU )",
        directors=[
            ("SRI PROMOTER", "DIRECTOR", "Tamil Nadu"),
        ],
    ),
    Vendor(
        vid="gu_08",
        legal_name="METRO LOGISTICS INDIA PVT LTD",
        trade_name="METRO LOGISTICS INDIA",
        gstin="07AAIDO2328K1ZI",
        pan="AAIDO2328K",
        udyam="UDYAM-XX-08-0005072",
        premises="", road="",
        city_town="DELHI", district="Delhi", state="Delhi", pin="",
        block="", premises_building="",
        additional_address="",
        combined_address="UNIT NO. 7, BLOCK A, 5TH FLOOR, METRO LOGISTICS PARK, BAWANA INDUSTRIAL AREA, SECTOR 3, MAIN INDUSTRIAL ROAD, DSIDC COMPLEX, POOTH KHURD, NARELA ROAD, BAWANA, DELHI",
        expect_lines=(
            "UNIT NO. 7, BLOCK A, 5TH FLOOR, METRO LOGISTICS PARK, BAWANA INDUSTRIAL AREA",
            "SECTOR 3, MAIN INDUSTRIAL ROAD, DSIDC COMPLEX",
            "POOTH KHURD, NARELA ROAD",
            "BAWANA",
        ),
        expect_city="Delhi", expect_state="Delhi", expect_pin="",
        address_case="stress: 12 comma-separated fragments, "
                     "medium confidence, cap-merge exercised",
        constitution="Private Limited Company",
        major_activity="MANUFACTURING",
        org_type="Private Limited Company",
        enterprise_type="Small",
        bank_name="State Bank of India",
        ifsc="SBIN0002136",
        account_number="700000005864",
        branch_line1="Delhi - Main Branch",
        branch_line2="Ground Floor, Commercial Complex, Delhi, Delhi",
        mobile="9700969696",
        email="accounts@metro08.co.in",
        incorporated="18/09/1998",
        commenced="01/09/1998",
        udyam_date="18/09/2020",
        gst_granted="18/09/2020",
        dic="DELHI ( DELHI )",
        msme_dfo="DELHI ( DELHI )",
        directors=[
            ("METRO PROMOTER", "DIRECTOR", "Delhi"),
        ],
    ),
    Vendor(
        vid="gu_09",
        legal_name="SHIVAM FORGINGS PVT LTD",
        trade_name="SHIVAM FORGINGS",
        gstin="03AAJDT2369V1ZJ",
        pan="AAJDT2369V",
        udyam="UDYAM-XX-09-0005081",
        premises="", road="",
        city_town="LUDHIANA", district="Ludhiana", state="Punjab", pin="",
        block="", premises_building="",
        additional_address="",
        combined_address="1ST FLOOR, UNIT B-14, PHASE III, SHIVAM INDUSTRIAL ESTATE, FOCAL POINT, EXTENSION BLOCK B, INDUSTRIAL AREA ROAD, JAMALPUR COLONY, DUGRI, GILL ROAD, LUDHIANA, PUNJAB",
        expect_lines=(
            "1ST FLOOR, UNIT B-14, PHASE III",
            "SHIVAM INDUSTRIAL ESTATE, FOCAL POINT",
            "EXTENSION BLOCK B, INDUSTRIAL AREA ROAD",
            "JAMALPUR COLONY, DUGRI, GILL ROAD",
        ),
        expect_city="Ludhiana", expect_state="Punjab", expect_pin="",
        address_case="stress: 12 comma-separated fragments, "
                     "medium confidence, cap-merge exercised",
        constitution="Private Limited Company",
        major_activity="MANUFACTURING",
        org_type="Private Limited Company",
        enterprise_type="Small",
        bank_name="Bank of Baroda",
        ifsc="BARB0002153",
        account_number="700000006597",
        branch_line1="Ludhiana - Main Branch",
        branch_line2="Ground Floor, Commercial Complex, Ludhiana, Punjab",
        mobile="9701090908",
        email="accounts@shivam09.co.in",
        incorporated="19/01/1990",
        commenced="02/10/1990",
        udyam_date="19/09/2020",
        gst_granted="19/01/2021",
        dic="LUDHIANA ( PUNJAB )",
        msme_dfo="LUDHIANA ( PUNJAB )",
        directors=[
            ("SHIVAM PROMOTER", "DIRECTOR", "Punjab"),
        ],
    ),
    Vendor(
        vid="gu_10",
        legal_name="CYBER TRADE VENTURES PVT LTD",
        trade_name="CYBER TRADE VENTURES",
        gstin="09AAKDY2410G1ZK",
        pan="AAKDY2410G",
        udyam="UDYAM-XX-10-0005090",
        premises="", road="",
        city_town="NOIDA", district="Noida", state="Uttar Pradesh", pin="",
        block="", premises_building="",
        additional_address="",
        combined_address="TOWER B, 6TH FLOOR, PART A, OFFICE NO. 603, CYBER TRADE CENTRE, NOIDA SECTOR 62, BLOCK B, INSTITUTIONAL AREA, FORTUNE ROAD, ELECTRONICS CITY, CHIJARSI, KHODA, NOIDA, UTTAR PRADESH",
        expect_lines=(
            "TOWER B, 6TH FLOOR, PART A, OFFICE NO. 603, CYBER TRADE CENTRE",
            "NOIDA SECTOR 62, BLOCK B, INSTITUTIONAL AREA",
            "FORTUNE ROAD",
            "ELECTRONICS CITY, CHIJARSI, KHODA",
        ),
        expect_city="Noida", expect_state="Uttar Pradesh", expect_pin="",
        address_case="stress: 14 comma-separated fragments, "
                     "medium confidence, cap-merge exercised",
        constitution="Private Limited Company",
        major_activity="MANUFACTURING",
        org_type="Private Limited Company",
        enterprise_type="Small",
        bank_name="ICICI Bank",
        ifsc="ICIC0002170",
        account_number="700000007330",
        branch_line1="Noida - Main Branch",
        branch_line2="Ground Floor, Commercial Complex, Noida, Uttar Pradesh",
        mobile="9701212120",
        email="accounts@cyber10.co.in",
        incorporated="20/02/1991",
        commenced="03/11/1991",
        udyam_date="20/09/2020",
        gst_granted="20/02/2022",
        dic="NOIDA ( UTTAR PRADESH )",
        msme_dfo="NOIDA ( UTTAR PRADESH )",
        directors=[
            ("CYBER PROMOTER", "DIRECTOR", "Uttar Pradesh"),
        ],
    ),
    Vendor(
        vid="gu_11",
        legal_name="MAHALAXMI INDUSTRIES PVT LTD",
        trade_name="MAHALAXMI INDUSTRIES",
        gstin="27AALDD2451R1ZL",
        pan="AALDD2451R",
        udyam="UDYAM-XX-11-0005099",
        premises="", road="",
        city_town="MUMBAI", district="Mumbai", state="Maharashtra", pin="",
        block="", premises_building="",
        additional_address="",
        combined_address="UNIT C-22, GROUND FLOOR, BLOCK D, MAHALAXMI INDUSTRIAL PARK, ANDHERI EAST, MIDC INDUSTRIAL AREA, MAROL ROAD, SEEPZ EXTENSION, KONDIVITA, CHAKALA, ANDHERI EAST, MUMBAI, MAHARASHTRA",
        expect_lines=(
            "UNIT C-22, GROUND FLOOR, BLOCK D, MAHALAXMI INDUSTRIAL PARK",
            "ANDHERI EAST, MIDC INDUSTRIAL AREA",
            "MAROL ROAD, SEEPZ EXTENSION",
            "KONDIVITA, CHAKALA, ANDHERI EAST",
        ),
        expect_city="Mumbai", expect_state="Maharashtra", expect_pin="",
        address_case="stress: 13 comma-separated fragments, "
                     "medium confidence, cap-merge exercised",
        constitution="Private Limited Company",
        major_activity="MANUFACTURING",
        org_type="Private Limited Company",
        enterprise_type="Small",
        bank_name="HDFC Bank",
        ifsc="HDFC0002187",
        account_number="700000008063",
        branch_line1="Mumbai - Main Branch",
        branch_line2="Ground Floor, Commercial Complex, Mumbai, Maharashtra",
        mobile="9701333332",
        email="accounts@mahalaxmi11.co.in",
        incorporated="21/03/1992",
        commenced="04/12/1992",
        udyam_date="21/09/2020",
        gst_granted="21/03/2023",
        dic="MUMBAI ( MAHARASHTRA )",
        msme_dfo="MUMBAI ( MAHARASHTRA )",
        directors=[
            ("MAHALAXMI PROMOTER", "DIRECTOR", "Maharashtra"),
        ],
    ),
    Vendor(
        vid="gu_12",
        legal_name="RING ROAD TEXTILES PVT LTD",
        trade_name="RING ROAD TEXTILES",
        gstin="24AAMDI2492C1ZM",
        pan="AAMDI2492C",
        udyam="UDYAM-XX-12-0005108",
        premises="", road="",
        city_town="SURAT", district="Surat", state="Gujarat", pin="",
        block="", premises_building="",
        additional_address="",
        combined_address="3RD FLOOR, BUILDING A, UNIT NO. 308, RING ROAD INDUSTRIAL COMPLEX, TEXTILE MARKET ZONE, RING ROAD, UDHNA INDUSTRIAL AREA, SALABATPURA, KADODARA ROAD, UDHNA, SURAT, GUJARAT",
        expect_lines=(
            "3RD FLOOR, BUILDING A, UNIT NO. 308, RING ROAD INDUSTRIAL COMPLEX, TEXTILE MARKET ZONE",
            "RING ROAD, UDHNA INDUSTRIAL AREA",
            "SALABATPURA, KADODARA ROAD",
            "UDHNA",
        ),
        expect_city="Surat", expect_state="Gujarat", expect_pin="",
        address_case="stress: 12 comma-separated fragments, "
                     "medium confidence, cap-merge exercised",
        constitution="Private Limited Company",
        major_activity="MANUFACTURING",
        org_type="Private Limited Company",
        enterprise_type="Small",
        bank_name="Axis Bank",
        ifsc="UTIB0002204",
        account_number="700000008796",
        branch_line1="Surat - Main Branch",
        branch_line2="Ground Floor, Commercial Complex, Surat, Gujarat",
        mobile="9701454544",
        email="accounts@ring12.co.in",
        incorporated="22/04/1993",
        commenced="05/01/1993",
        udyam_date="22/09/2020",
        gst_granted="22/04/2020",
        dic="SURAT ( GUJARAT )",
        msme_dfo="SURAT ( GUJARAT )",
        directors=[
            ("RING PROMOTER", "DIRECTOR", "Gujarat"),
        ],
    ),
    Vendor(
        vid="gu_13",
        legal_name="KALINDI METAL WORKS PVT LTD",
        trade_name="KALINDI METAL WORKS",
        gstin="06AANDN2533N1ZN",
        pan="AANDN2533N",
        udyam="UDYAM-XX-13-0005117",
        premises="", road="",
        city_town="FARIDABAD", district="Faridabad", state="Haryana", pin="",
        block="", premises_building="",
        additional_address="",
        combined_address="PLOT 86, PART B, BLOCK C, 2ND FLOOR, KALINDI INDUSTRIAL ESTATE, SECTOR 24, INDUSTRIAL AREA, MATHURA ROAD, NIT EXTENSION, SARAI KHWAJA, BADARPUR BORDER ROAD, FARIDABAD, HARYANA",
        expect_lines=(
            "PLOT 86, PART B, BLOCK C, 2ND FLOOR, KALINDI INDUSTRIAL ESTATE",
            "SECTOR 24, INDUSTRIAL AREA",
            "MATHURA ROAD",
            "NIT EXTENSION, SARAI KHWAJA, BADARPUR BORDER ROAD",
        ),
        expect_city="Faridabad", expect_state="Haryana", expect_pin="",
        address_case="stress: 13 comma-separated fragments, "
                     "medium confidence, cap-merge exercised",
        constitution="Private Limited Company",
        major_activity="MANUFACTURING",
        org_type="Private Limited Company",
        enterprise_type="Small",
        bank_name="State Bank of India",
        ifsc="SBIN0002221",
        account_number="700000009529",
        branch_line1="Faridabad - Main Branch",
        branch_line2="Ground Floor, Commercial Complex, Faridabad, Haryana",
        mobile="9701575756",
        email="accounts@kalindi13.co.in",
        incorporated="23/05/1994",
        commenced="06/02/1994",
        udyam_date="23/09/2020",
        gst_granted="23/05/2021",
        dic="FARIDABAD ( HARYANA )",
        msme_dfo="FARIDABAD ( HARYANA )",
        directors=[
            ("KALINDI PROMOTER", "DIRECTOR", "Haryana"),
        ],
    ),
    Vendor(
        vid="gu_14",
        legal_name="SRI DURGA TRADING PVT LTD",
        trade_name="SRI DURGA TRADING",
        gstin="07AAODS2574Y1ZO",
        pan="AAODS2574Y",
        udyam="UDYAM-XX-14-0005126",
        premises="", road="",
        city_town="DELHI", district="Delhi", state="Delhi", pin="",
        block="", premises_building="",
        additional_address="",
        combined_address="OFFICE NO. 405, 4TH FLOOR, TOWER A, SRI DURGA BUSINESS PARK, KONDLI, BLOCK C, MAYUR VIHAR EXTENSION, GHAZIPUR ROAD, SANJAY LAKE INDUSTRIAL ZONE, KONDLI VILLAGE, GAZIPUR, DELHI",
        expect_lines=(
            "OFFICE NO. 405, 4TH FLOOR, TOWER A, SRI DURGA BUSINESS PARK",
            "KONDLI, BLOCK C",
            "MAYUR VIHAR EXTENSION, GHAZIPUR ROAD, SANJAY LAKE INDUSTRIAL ZONE",
            "KONDLI VILLAGE, GAZIPUR",
        ),
        expect_city="Delhi", expect_state="Delhi", expect_pin="",
        address_case="stress: 12 comma-separated fragments, "
                     "medium confidence, cap-merge exercised",
        constitution="Private Limited Company",
        major_activity="MANUFACTURING",
        org_type="Private Limited Company",
        enterprise_type="Small",
        bank_name="Bank of Baroda",
        ifsc="BARB0002238",
        account_number="700000010262",
        branch_line1="Delhi - Main Branch",
        branch_line2="Ground Floor, Commercial Complex, Delhi, Delhi",
        mobile="9701696968",
        email="accounts@sri14.co.in",
        incorporated="24/06/1995",
        commenced="07/03/1995",
        udyam_date="24/09/2020",
        gst_granted="24/06/2022",
        dic="DELHI ( DELHI )",
        msme_dfo="DELHI ( DELHI )",
        directors=[
            ("SRI PROMOTER", "DIRECTOR", "Delhi"),
        ],
    ),
    Vendor(
        vid="gu_15",
        legal_name="HITECH LOGISTICS CENTRE PVT LTD",
        trade_name="HITECH LOGISTICS CENTRE",
        gstin="36AAPDX2615J1ZP",
        pan="AAPDX2615J",
        udyam="UDYAM-XX-15-0005135",
        premises="", road="",
        city_town="HYDERABAD", district="Hyderabad", state="Telangana", pin="",
        block="", premises_building="",
        additional_address="",
        combined_address="BLOCK E, UNIT 16, GROUND FLOOR, INDUSTRIAL LOGISTICS CENTRE, HITECH CITY, PHASE II, CYBER ROAD, MADHAPUR INDUSTRIAL ZONE, KOTHAGUDA, GACHIBOWLI ROAD, MADHAPUR, HYDERABAD, TELANGANA",
        expect_lines=(
            "BLOCK E, UNIT 16, GROUND FLOOR, INDUSTRIAL LOGISTICS CENTRE",
            "HITECH CITY, PHASE II",
            "CYBER ROAD, MADHAPUR INDUSTRIAL ZONE",
            "KOTHAGUDA, GACHIBOWLI ROAD, MADHAPUR",
        ),
        expect_city="Hyderabad", expect_state="Telangana", expect_pin="",
        address_case="stress: 13 comma-separated fragments, "
                     "medium confidence, cap-merge exercised",
        constitution="Private Limited Company",
        major_activity="MANUFACTURING",
        org_type="Private Limited Company",
        enterprise_type="Small",
        bank_name="ICICI Bank",
        ifsc="ICIC0002255",
        account_number="700000010995",
        branch_line1="Hyderabad - Main Branch",
        branch_line2="Ground Floor, Commercial Complex, Hyderabad, Telangana",
        mobile="9701818180",
        email="accounts@hitech15.co.in",
        incorporated="25/07/1996",
        commenced="08/04/1996",
        udyam_date="25/09/2020",
        gst_granted="25/07/2023",
        dic="HYDERABAD ( TELANGANA )",
        msme_dfo="HYDERABAD ( TELANGANA )",
        directors=[
            ("HITECH PROMOTER", "DIRECTOR", "Telangana"),
        ],
    ),
    Vendor(
        vid="gu_16",
        legal_name="TALOJA CHEMICALS PVT LTD",
        trade_name="TALOJA CHEMICALS",
        gstin="27AAQDC2656U1ZQ",
        pan="AAQDC2656U",
        udyam="UDYAM-XX-16-0005144",
        premises="", road="",
        city_town="NAVI MUMBAI", district="Navi Mumbai", state="Maharashtra", pin="",
        block="", premises_building="",
        additional_address="",
        combined_address="2ND FLOOR, PART A, UNIT NO. 214, BUILDING C, TALOJA INDUSTRIAL AREA, MIDC PHASE II, TALOJA ROAD, INDUSTRIAL ESTATE EXTENSION, KALAMBOLI, KAMOTHE, TALOJA, NAVI MUMBAI, MAHARASHTRA",
        expect_lines=(
            "2ND FLOOR, PART A, UNIT NO. 214, BUILDING C",
            "TALOJA INDUSTRIAL AREA, MIDC PHASE II",
            "TALOJA ROAD, INDUSTRIAL ESTATE EXTENSION",
            "KALAMBOLI, KAMOTHE, TALOJA",
        ),
        expect_city="Navi Mumbai", expect_state="Maharashtra", expect_pin="",
        address_case="stress: 13 comma-separated fragments, "
                     "medium confidence, cap-merge exercised",
        constitution="Private Limited Company",
        major_activity="MANUFACTURING",
        org_type="Private Limited Company",
        enterprise_type="Small",
        bank_name="HDFC Bank",
        ifsc="HDFC0002272",
        account_number="700000011728",
        branch_line1="Navi Mumbai - Main Branch",
        branch_line2="Ground Floor, Commercial Complex, Navi Mumbai, Maharashtra",
        mobile="9701939392",
        email="accounts@taloja16.co.in",
        incorporated="26/08/1997",
        commenced="01/05/1997",
        udyam_date="26/09/2020",
        gst_granted="26/08/2020",
        dic="NAVI MUMBAI ( MAHARASHTRA )",
        msme_dfo="NAVI MUMBAI ( MAHARASHTRA )",
        directors=[
            ("TALOJA PROMOTER", "DIRECTOR", "Maharashtra"),
        ],
    ),
    Vendor(
        vid="gu_17",
        legal_name="SARDAR INDUSTRIAL WORKS PVT LTD",
        trade_name="SARDAR INDUSTRIAL WORKS",
        gstin="24AARDH2697F1ZR",
        pan="AARDH2697F",
        udyam="UDYAM-XX-17-0005153",
        premises="", road="",
        city_town="VADODARA", district="Vadodara", state="Gujarat", pin="",
        block="", premises_building="",
        additional_address="",
        combined_address="UNIT D-09, 1ST FLOOR, BLOCK B, SARDAR INDUSTRIAL COMPLEX, MAKARPURA GIDC, PHASE III, GIDC MAIN ROAD, MAKARPURA INDUSTRIAL ESTATE, MANJALPUR, AKOTA EXTENSION, MAKARPURA, VADODARA, GUJARAT",
        expect_lines=(
            "UNIT D-09, 1ST FLOOR, BLOCK B",
            "SARDAR INDUSTRIAL COMPLEX, MAKARPURA GIDC, PHASE III",
            "GIDC MAIN ROAD, MAKARPURA INDUSTRIAL ESTATE",
            "MANJALPUR, AKOTA EXTENSION, MAKARPURA",
        ),
        expect_city="Vadodara", expect_state="Gujarat", expect_pin="",
        address_case="stress: 13 comma-separated fragments, "
                     "medium confidence, cap-merge exercised",
        constitution="Private Limited Company",
        major_activity="MANUFACTURING",
        org_type="Private Limited Company",
        enterprise_type="Small",
        bank_name="Axis Bank",
        ifsc="UTIB0002289",
        account_number="700000012461",
        branch_line1="Vadodara - Main Branch",
        branch_line2="Ground Floor, Commercial Complex, Vadodara, Gujarat",
        mobile="9702060604",
        email="accounts@sardar17.co.in",
        incorporated="27/09/1998",
        commenced="02/06/1998",
        udyam_date="27/09/2020",
        gst_granted="27/09/2021",
        dic="VADODARA ( GUJARAT )",
        msme_dfo="VADODARA ( GUJARAT )",
        directors=[
            ("SARDAR PROMOTER", "DIRECTOR", "Gujarat"),
        ],
    ),
    Vendor(
        vid="gu_18",
        legal_name="NORTH CITY LOGISTICS PVT LTD",
        trade_name="NORTH CITY LOGISTICS",
        gstin="19AASDM2738Q1ZS",
        pan="AASDM2738Q",
        udyam="UDYAM-XX-18-0005162",
        premises="", road="",
        city_town="KOLKATA", district="Kolkata", state="West Bengal", pin="",
        block="", premises_building="",
        additional_address="",
        combined_address="BUILDING B, 3RD FLOOR, PART C, UNIT 312, NORTH CITY LOGISTICS PARK, BARASAT ROAD, INDUSTRIAL LOGISTICS ZONE, MADHYAMGRAM EXTENSION, NEW BARRACKPORE, SODEPUR, BARASAT, KOLKATA, WEST BENGAL",
        expect_lines=(
            "BUILDING B, 3RD FLOOR, PART C, UNIT 312, NORTH CITY LOGISTICS PARK, BARASAT ROAD, INDUSTRIAL LOGISTICS ZONE",
            "MADHYAMGRAM EXTENSION, NEW BARRACKPORE, SODEPUR",
            "BARASAT",
            "",
        ),
        expect_city="Kolkata", expect_state="West Bengal", expect_pin="",
        address_case="stress: 13 comma-separated fragments, "
                     "medium confidence, cap-merge exercised",
        constitution="Private Limited Company",
        major_activity="MANUFACTURING",
        org_type="Private Limited Company",
        enterprise_type="Small",
        bank_name="State Bank of India",
        ifsc="SBIN0002306",
        account_number="700000013194",
        branch_line1="Kolkata - Main Branch",
        branch_line2="Ground Floor, Commercial Complex, Kolkata, West Bengal",
        mobile="9702181816",
        email="accounts@north18.co.in",
        incorporated="28/01/1990",
        commenced="03/07/1990",
        udyam_date="28/09/2020",
        gst_granted="28/01/2022",
        dic="KOLKATA ( WEST BENGAL )",
        msme_dfo="KOLKATA ( WEST BENGAL )",
        directors=[
            ("NORTH PROMOTER", "DIRECTOR", "West Bengal"),
        ],
    ),
    Vendor(
        vid="gu_19",
        legal_name="R K INDUSTRIAL ENTERPRISES PVT LTD",
        trade_name="R K INDUSTRIAL ENTERPRISES",
        gstin="27AATDR2779B1ZT",
        pan="AATDR2779B",
        udyam="UDYAM-XX-19-0005171",
        premises="", road="",
        city_town="BHIWANDI", district="Bhiwandi", state="Maharashtra", pin="",
        block="", premises_building="",
        additional_address="",
        combined_address="GROUND FLOOR, UNIT A-06, BLOCK F, R.K. INDUSTRIAL ESTATE, KALHER, BHIWANDI, NH-8 SERVICE ROAD, ANJURPHATA LOGISTICS ZONE, KALHER VILLAGE, VALPADA, KONGAON, BHIWANDI, MAHARASHTRA",
        expect_lines=(
            "GROUND FLOOR, UNIT A-06, BLOCK F",
            "R.K. INDUSTRIAL ESTATE",
            "KALHER, BHIWANDI, NH-8 SERVICE ROAD, ANJURPHATA LOGISTICS ZONE",
            "KALHER VILLAGE, VALPADA, KONGAON",
        ),
        expect_city="Bhiwandi", expect_state="Maharashtra", expect_pin="",
        address_case="stress: 13 comma-separated fragments, "
                     "medium confidence, cap-merge exercised",
        constitution="Private Limited Company",
        major_activity="MANUFACTURING",
        org_type="Private Limited Company",
        enterprise_type="Small",
        bank_name="Bank of Baroda",
        ifsc="BARB0002323",
        account_number="700000013927",
        branch_line1="Bhiwandi - Main Branch",
        branch_line2="Ground Floor, Commercial Complex, Bhiwandi, Maharashtra",
        mobile="9702303028",
        email="accounts@r19.co.in",
        incorporated="10/02/1991",
        commenced="04/08/1991",
        udyam_date="10/09/2020",
        gst_granted="10/02/2023",
        dic="BHIWANDI ( MAHARASHTRA )",
        msme_dfo="BHIWANDI ( MAHARASHTRA )",
        directors=[
            ("R PROMOTER", "DIRECTOR", "Maharashtra"),
        ],
    ),
    Vendor(
        vid="gu_20",
        legal_name="GREENFIELD BUSINESS SERVICES PVT LTD",
        trade_name="GREENFIELD BUSINESS SERVICES",
        gstin="29AAUDW2820M1ZU",
        pan="AAUDW2820M",
        udyam="UDYAM-XX-20-0005180",
        premises="", road="",
        city_town="BENGALURU", district="Bengaluru", state="Karnataka", pin="",
        block="", premises_building="",
        additional_address="",
        combined_address="5TH FLOOR, PART B, OFFICE 502, TOWER C, GREENFIELD BUSINESS PARK, HOSUR ROAD, BOMMANAHALLI INDUSTRIAL ZONE, SINGASANDRA, GARVEBHAVIPALYA, ELECTRONIC CITY ROAD, HOSUR ROAD, BENGALURU, KARNATAKA",
        expect_lines=(
            "5TH FLOOR, PART B, OFFICE 502, TOWER C",
            "GREENFIELD BUSINESS PARK",
            "HOSUR ROAD, BOMMANAHALLI INDUSTRIAL ZONE",
            "SINGASANDRA, GARVEBHAVIPALYA, ELECTRONIC CITY ROAD, HOSUR ROAD",
        ),
        expect_city="Bengaluru", expect_state="Karnataka", expect_pin="",
        address_case="stress: 13 comma-separated fragments, "
                     "medium confidence, cap-merge exercised",
        constitution="Private Limited Company",
        major_activity="MANUFACTURING",
        org_type="Private Limited Company",
        enterprise_type="Small",
        bank_name="ICICI Bank",
        ifsc="ICIC0002340",
        account_number="700000014660",
        branch_line1="Bengaluru - Main Branch",
        branch_line2="Ground Floor, Commercial Complex, Bengaluru, Karnataka",
        mobile="9702424240",
        email="accounts@greenfield20.co.in",
        incorporated="11/03/1992",
        commenced="05/09/1992",
        udyam_date="11/09/2020",
        gst_granted="11/03/2020",
        dic="BENGALURU ( KARNATAKA )",
        msme_dfo="BENGALURU ( KARNATAKA )",
        directors=[
            ("GREENFIELD PROMOTER", "DIRECTOR", "Karnataka"),
        ],
    ),
]
# ---------------------------------------------------------------------------
# SHARED CSS
# ---------------------------------------------------------------------------

_GST_CSS = """
@page { size: A4; margin: 14mm 12mm; }
* { box-sizing: border-box; }
body {
  font-family: "Times New Roman", Times, serif;
  font-size: 10.5pt; color: #000; margin: 0;
}
/* Watermark -- rendered as a background IMAGE, deliberately not as text.
   The real REG-06's "Goods and Services Tax" ghost is part of the page
   graphic and never reaches the text layer. An earlier version of this
   fixture drew it as live rotated text, and it broke extraction in a way
   the real document cannot: the rotated span's bounding box is a tall
   diagonal rectangle that OVERLAPPED the "Constitution of Business" value
   cell, so the field matcher looked right from the caption and found the
   watermark's "Services" fragment instead of "Private Limited Company".
   Keeping it non-text preserves the visual without inventing a failure
   mode that does not exist in the source documents. */
/* Watermark -- background IMAGE, not text (kept out of the PDF text layer),
   at very low contrast against the white page.
   Combined-address vendors get this document RASTERISED (see main()), so a
   watermark visible enough to read is visible enough for OCR to read too --
   regardless of it never being real text. At the original #dcdcdc fill,
   PaddleOCR picked up "Goods and Services Tax" as a genuine span sitting
   inside the address block's own row band, and it was closer to the page's
   left margin than the address text, corrupting which OCR line counted as
   the value's true left edge and breaking the multi-row join in
   field_matcher._join_value. Lightened to #f3f3f3 (contrast ratio too low
   for the OCR text detector to fire on it, confirmed empirically against
   this render pipeline) -- still visible to a human viewer, invisible to OCR. */
.wm {
  position: fixed; top: 30%; left: 6%;
  width: 150mm; height: 90mm; z-index: -1;
  background-image: url("data:image/svg+xml;utf8,\
<svg xmlns='http://www.w3.org/2000/svg' width='560' height='340'>\
<text x='0' y='230' transform='rotate(-30 0 230)' \
font-family='Times New Roman, serif' font-size='54' font-weight='bold' \
fill='%23f3f3f3'>Goods and Services Tax</text></svg>");
  background-repeat: no-repeat;
}
.hdr { text-align: center; }
.hdr .emblem { font-size: 26pt; line-height: 1; }
.hdr .govt { font-weight: bold; color: #000080; margin-top: 2mm; }
.hdr .form { font-weight: bold; color: #000080; }
.hdr .rule { font-style: italic; color: #000080; font-size: 9.5pt; }
.hdr .cert { font-weight: bold; color: #000080; font-size: 12pt; margin-top: 1mm; }
.regno { font-weight: bold; color: #000080; margin: 3mm 0 2mm 6mm; }
.amend { text-align: right; font-weight: bold; color: #000080; }
table.main { width: 100%; border-collapse: collapse; margin-top: 2mm; }
table.main td { border: 0.6pt solid #000; padding: 1.6mm 2mm; vertical-align: top; }
td.num { width: 7%; text-align: center; color: #000080; font-weight: bold; }
/* Widened from 27% and kept unwrapped. A caption cell narrow enough to wrap
   ("Address of Principal Place of / Business") rendered fine as a real PDF's
   text layer, but once this document is rasterised (see main() -- combined-
   address vendors must be to deliver the value line intact) PaddleOCR reads
   the wrap point as a word boundary with NO space, merging into
   "addressofprincipal placeof". That drops the caption below the fuzzy-match
   threshold entirely, and the field falls back to whatever span happens to
   sit nearest -- in one case, part of the address's own leading "FLAT NO."
   words. Kept on one line, unwrapped, so OCR reads normal inter-word spaces. */
td.lbl { width: 34%; color: #000080; font-weight: bold; white-space: nowrap; }
td.val { color: #000080; }
.addr-line { color: #000080; }
.addr-line b { color: #000080; }
.sig { color: #000080; font-weight: bold; }
.note { font-size: 9pt; padding: 1.6mm 2mm; border: 0.6pt solid #000; border-top: none; }
.footer-note { font-weight: bold; color: #000080; margin-top: 4mm; font-size: 9.5pt; }
.annex-title { text-align: right; font-weight: bold; color: #000080; }
.annex-h { font-weight: bold; color: #000080; margin: 6mm 0 3mm 0; }
table.plain { width: 100%; border-collapse: collapse; }
table.plain td { padding: 1.2mm 0; vertical-align: top; color: #000080; }
table.plain td.k { width: 32%; font-weight: bold; }
.pagebreak { page-break-after: always; }
"""

_UDYAM_CSS = """
@page { size: A4; margin: 10mm; }
* { box-sizing: border-box; }
body { font-family: "Times New Roman", Times, serif; font-size: 9.5pt; margin: 0; }
.frame { border: 1pt solid #444; padding: 0; }
.band {
  background: #4a4a4a; color: #fff; padding: 3mm 4mm;
  display: flex; align-items: center; justify-content: space-between;
}
.band .c { text-align: center; flex: 1; line-height: 1.35; }
.band .hin { font-size: 9pt; }
.band .en { font-size: 10pt; font-weight: bold; }
.band .msme { font-weight: bold; font-size: 13pt; letter-spacing: 1pt; }
.band .msme small { display: block; font-size: 5.2pt; letter-spacing: 0; font-weight: normal; }
h1.title { text-align: center; font-size: 16pt; margin: 5mm 0 4mm 0; font-weight: bold; }
table.kv { width: 100%; border-collapse: collapse; margin: 0 0 3mm 0; }
table.kv > tbody > tr > td { padding: 1.6mm 3mm; vertical-align: middle; }
/* Label column. Kept wide enough that multi-word captions ("MAJOR ACTIVITY",
   "NAME OF ENTERPRISE") stay on ONE line with normal word spacing. When this
   column was narrower the captions wrapped/stretched, and PaddleOCR's word
   grouping emitted them as separate spans ('MAJOR' + 'ACTIVITY'). The real
   certificate OCRs these as a single span, and field_matcher matches a caption
   against one span at a time -- so a split caption is a fixture artefact that
   would manufacture a `missed` the real document does not produce. */
td.k {
  width: 34%; text-align: right; font-weight: bold;
  padding-right: 6mm; white-space: nowrap;
}
td.v { font-weight: bold; }
td.v.center { text-align: center; }
.activity {
  background: #1a1a1a; color: #fff; text-align: center;
  font-weight: bold; font-size: 12.5pt; padding: 2.4mm; letter-spacing: 0.5pt;
}
table.grid { width: 100%; border-collapse: collapse; }
table.grid td, table.grid th {
  border: 0.6pt solid #000; padding: 1.4mm 2mm; vertical-align: top;
}
table.grid th { font-weight: bold; text-align: center; }
/* Same reasoning as td.k: keep short multi-word column headers ("Bank Name",
   "IFS Code", "Bank Account Number") unwrapped so they OCR as one span each,
   matching how the real certificate reads. */
table.grid th.tight { white-space: nowrap; }
table.grid td.hk { font-weight: bold; }
h3.sec { font-weight: bold; font-size: 11pt; margin: 4mm 0 1.5mm 0; }
.sec-rule { border-bottom: 0.8pt solid #999; margin-bottom: 2mm; }
.pagebreak { page-break-after: always; }
.small { font-size: 8.5pt; }
.center { text-align: center; }
.urn-band {
  background: #4a4a4a; color: #fff; text-align: center;
  font-weight: bold; padding: 1.8mm; font-size: 10.5pt;
}
.disclaimer { font-size: 8.2pt; margin-top: 3mm; }
.contact-h { font-weight: bold; font-size: 11.5pt; margin: 5mm 0 3mm 0; }
.contact-row { margin: 2mm 0; }
.contact-row b { display: inline-block; min-width: 62mm; font-size: 11pt; }
"""

_CHEQUE_CSS = """
@page { size: 210mm 99mm; margin: 0; }
* { box-sizing: border-box; }
body { font-family: Arial, Helvetica, sans-serif; font-size: 8pt; margin: 0; }
.cheque { position: relative; width: 210mm; height: 99mm; padding: 6mm 7mm; }
/* No negative letter-spacing. Tightening the bank name made PaddleOCR merge
   the words ("Kotak Mahindra Bank" -> "KotakMahindraBank"), an OCR failure
   caused by the fixture rather than by anything the real cheque does. */
.bank { font-size: 15pt; font-weight: bold; }
.branch { font-size: 8.5pt; font-weight: bold; margin-top: 1.2mm; }
.branchaddr { font-size: 7.4pt; margin-top: 0.6mm; }
.ifsc { font-size: 7.4pt; margin-top: 0.4mm; }
.payee-box {
  position: absolute; top: 6mm; left: 124mm; width: 34mm;
  border-bottom: 0.8pt solid #000; text-align: center;
  font-size: 8.5pt; letter-spacing: 1pt; padding-bottom: 0.6mm;
}
.valid { position: absolute; top: 5mm; right: 7mm; font-size: 7.6pt; font-weight: bold; }
.datebox { position: absolute; top: 9mm; right: 7mm; display: flex; }
.datebox div {
  width: 6.2mm; height: 6.2mm; border: 0.5pt solid #000; border-left: none;
}
.datebox div:first-child { border-left: 0.5pt solid #000; }
.datelbl {
  position: absolute; top: 15.6mm; right: 7mm; display: flex;
  font-size: 8pt; font-weight: bold;
}
.datelbl span { width: 6.2mm; text-align: center; }
.pay { position: absolute; top: 25mm; left: 7mm; font-size: 11pt; font-weight: bold; }
.payline { position: absolute; top: 28.5mm; left: 24mm; right: 40mm; border-bottom: 0.5pt solid #000; }
.orderlbl { position: absolute; top: 24mm; right: 7mm; font-size: 10.5pt; font-weight: bold; }
.rupees { position: absolute; top: 33mm; left: 7mm; font-size: 11pt; font-weight: bold; }
.rupline1 { position: absolute; top: 37mm; left: 30mm; right: 7mm; border-bottom: 0.5pt solid #000; }
.rupline2 { position: absolute; top: 43mm; left: 7mm; right: 62mm; border-bottom: 0.5pt solid #000; }
.amtbox {
  position: absolute; top: 37mm; right: 7mm; width: 52mm; height: 9mm;
  border: 0.7pt solid #000; display: flex; align-items: center;
}
.amtbox .rs { width: 8mm; text-align: center; font-size: 12pt; border-right: 0.7pt solid #000; height: 100%; line-height: 9mm; }
.acbox {
  position: absolute; top: 50mm; left: 7mm; width: 62mm; height: 8mm;
  border: 0.7pt dashed #000; display: flex; align-items: center;
}
.acbox .k { width: 17mm; text-align: center; font-size: 8pt; font-weight: bold; border-right: 0.7pt dashed #000; height: 100%; line-height: 8mm; }
.acbox .v { padding-left: 4mm; font-size: 11pt; font-weight: bold; letter-spacing: 0.4pt; }
.acctype { position: absolute; top: 50mm; left: 74mm; font-size: 7.6pt; line-height: 1.5; }
.acctype b { font-size: 8pt; }
.forname {
  position: absolute; top: 49mm; right: 7mm; width: 78mm; text-align: right;
  font-size: 9.5pt; font-weight: bold;
}
.signatory { position: absolute; bottom: 12mm; right: 7mm; font-size: 8.5pt; font-weight: bold; }
.signhint { position: absolute; bottom: 9mm; right: 7mm; font-size: 6.6pt; }
.micr {
  position: absolute; bottom: 4mm; left: 0; width: 100%; text-align: center;
  font-family: "Courier New", monospace; font-size: 12pt; letter-spacing: 2pt;
}
.side {
  position: absolute; left: 1.5mm; top: 28mm; font-size: 6pt;
  writing-mode: vertical-rl; transform: rotate(180deg);
}
.cancel {
  position: absolute; top: 26mm; left: 40mm;
  font-family: "Segoe Script", "Brush Script MT", cursive;
  font-size: 30pt; color: #222; transform: rotate(-9deg);
}
.strike1 {
  position: absolute; top: 24mm; left: 30mm; width: 120mm;
  border-top: 1.1pt solid #222; transform: rotate(-7deg);
}
.strike2 {
  position: absolute; top: 40mm; left: 30mm; width: 120mm;
  border-top: 1.1pt solid #222; transform: rotate(-7deg);
}
.datestamp {
  position: absolute; top: 50mm; left: 70mm; font-size: 9pt;
  transform: rotate(-90deg); transform-origin: left top;
}
"""


# ---------------------------------------------------------------------------
# GST REG-06
# ---------------------------------------------------------------------------

def _address_cell(v: Vendor) -> str:
    """Render GST field 5, in whichever of the two real shapes this vendor uses.

    Captioned (the mb_control_systems shape): each part on its own captioned
    line. field_matcher reads city/state/pin straight out of these captions, so
    _resolve_combined_address hits its early-bail and the segmenter never runs.

    Combined: one uncaptioned run-on line, which is also a real REG-06 shape
    (it is exactly how Annexure A prints an additional place of business). This
    is the shape that puts the segmenter under test.
    """
    if v.combined_address:
        # Emitted as plain wrapping text. Do NOT try to force this onto one
        # line with nowrap/nbsp/smaller type: Chrome's PDF text layer splits a
        # long line at its own layout boundaries regardless, and tightening it
        # only made the split finer (one span per WORD rather than per comma
        # chunk). The fix is to rasterise this document -- see main() -- so OCR
        # reassembles the line, which is what the real scanned certificates
        # already rely on.
        return f'<div class="addr-line">{v.combined_address}</div>'
    return (
        f'<div class="addr-line"><b>Building No./Flat No.:</b> {v.premises}</div>\n'
        f'      <div class="addr-line"><b>Road/Street:</b> {v.road}</div>\n'
        f'      <div class="addr-line"><b>City/Town/Village:</b> {v.city_town}</div>\n'
        f'      <div class="addr-line"><b>District:</b> {v.district}</div>\n'
        f'      <div class="addr-line"><b>State:</b> {v.state}</div>\n'
        f'      <div class="addr-line"><b>PIN Code:</b> {v.pin}</div>'
    )


def _address_row(v: Vendor) -> str:
    """The GST field-5 table row.

    Captioned vendors keep the classic 3-column row (num | label | value).

    Combined-address vendors instead get the CAPTION on its own row, then the
    run-on address on a FULL-WIDTH row below it, at a comfortable size with
    generous line spacing. The narrow 3rd column at body size wrapped a 13-
    fragment address into 4 cramped lines that PaddleOCR (200dpi raster)
    mangled character-for-character ("BUILDING B, 3RD FLOOR" -> "BRDFLOR").
    A wide, well-leaded block renders as 1-2 clean lines that OCR reads
    correctly, so the de-glue step then has real text to work on.
    """
    if not v.combined_address:
        return (
            '<tr><td class="num">5.</td>'
            '<td class="lbl">Address of Principal Place of Business</td>'
            f'<td class="val" style="height:46mm">{_address_cell(v)}</td></tr>'
        )
    return (
        '<tr><td class="num">5.</td>'
        '<td class="lbl" colspan="2">Address of Principal Place of Business</td></tr>'
        '<tr><td class="num"></td>'
        '<td class="val" colspan="2" '
        'style="font-size:12pt; line-height:1.9; padding:3mm 3mm 5mm 3mm; '
        'letter-spacing:0.3pt">'
        f'{v.combined_address}</td></tr>'
    )


def _watermark_div(watermark: bool) -> str:
    return '<div class="wm">Goods and Services Tax</div>' if watermark else ""


def gst_html(v: Vendor, watermark: bool = True) -> str:
    directors_rows = "".join(
        f"""
        <tr><td class="num">{i}</td><td class="lbl">Name</td><td class="val">{name}</td></tr>
        <tr><td class="num"></td><td class="lbl">Designation/Status</td><td class="val">{desig}</td></tr>
        <tr><td class="num"></td><td class="lbl">Resident of State</td><td class="val">{st}</td></tr>
        """
        for i, (name, desig, st) in enumerate(v.directors, start=1)
    )

    return f"""<!doctype html>
<html><head><meta charset="utf-8"><style>{_GST_CSS}</style></head><body>

{_watermark_div(watermark)}

<div class="amend">(Amended)</div>
<div class="hdr">
  <div class="emblem">&#127963;</div>
  <div class="govt">Government of India</div>
  <div class="form">Form GST REG-06</div>
  <div class="rule">[See Rule 10(1)]</div>
  <div class="cert">Registration Certificate</div>
</div>
<div class="regno">Registration Number :{v.gstin}</div>

<table class="main">
  <tr><td class="num">1.</td><td class="lbl">Legal Name</td><td class="val">{v.legal_name}</td></tr>
  <tr><td class="num">2.</td><td class="lbl">Trade Name, if any</td><td class="val">{v.trade_name}</td></tr>
  <tr><td class="num">3.</td><td class="lbl">Additional trade names, if any</td><td class="val"></td></tr>
  <tr><td class="num">4.</td><td class="lbl">Constitution of Business</td><td class="val">{v.constitution}</td></tr>
  {_address_row(v)}
  <tr><td class="num">6.</td><td class="lbl">Date of Liability</td><td class="val">01/07/2017</td></tr>
  <tr><td class="num">7.</td><td class="lbl">Date of Validity</td>
      <td class="val">From&nbsp;&nbsp;&nbsp;01/07/2017&nbsp;&nbsp;&nbsp;To&nbsp;&nbsp;&nbsp;Not Applicable</td></tr>
  <tr><td class="num">8.</td><td class="lbl">Type of Registration</td><td class="val">Regular</td></tr>
  <tr><td class="num">9.</td><td class="lbl">Particulars of Approving</td>
      <td class="val">Centre Goods and Services Tax Act, 2017</td></tr>
  <tr><td class="lbl" colspan="3">Signature</td></tr>
  <tr><td class="val" colspan="3" style="height:20mm; text-align:center">
      Signature Not Verified<br>
      <span style="font-size:7pt">Digitally signed by DS GOODS<br>AND SERVICES TAX<br>
      NETWORK 07<br>Date: 2024.02.14 16:30:27 IST</span>
  </td></tr>
  <tr><td class="lbl" colspan="2">Name</td><td class="val">Rajib Banerjee</td></tr>
  <tr><td class="lbl" colspan="2">Designation</td><td class="val">Superintendent</td></tr>
  <tr><td class="lbl" colspan="2">Jurisdictional Office</td><td class="val">BALLYGUNGE</td></tr>
  <tr><td class="lbl" colspan="2">Date of issue of Certificate</td><td class="val">{v.gst_granted}</td></tr>
</table>
<div class="note">Note: The registration certificate is required to be prominently
displayed at all places of Business/Office(s) in the State.</div>

<div class="footer-note">This is a system generated digitally signed Registration
Certificate issued based on the approval of application granted on {v.gst_granted}
by the jurisdictional authority.</div>

<div class="pagebreak"></div>

{_watermark_div(watermark)}
<div class="annex-title">Annexure A</div>
<div class="hdr"><div class="emblem">&#127963;</div></div>
<div class="annex-h">Goods and Services Tax Identification Number: {v.gstin}</div>
<div class="annex-h">Details of Additional Place of Business(s)</div>
<table class="plain">
  <tr><td class="k">Legal Name</td><td>{v.legal_name}</td></tr>
  <tr><td class="k">Trade Name, if any</td><td>{v.trade_name}</td></tr>
  <tr><td class="k">Additional trade names, if any</td><td></td></tr>
</table>
<div class="annex-h">Total Number of Additional Place of Business(s) in the State
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{1 if v.additional_address else 0}</div>
<table class="plain">
  {f'<tr><td style="width:7%; text-align:center">1</td><td>{v.additional_address}</td></tr>' if v.additional_address else ''}
</table>

<div class="pagebreak"></div>

{_watermark_div(watermark)}
<div class="annex-title">Annexure B</div>
<div class="hdr"><div class="emblem">&#127963;</div></div>
<div class="annex-h">Goods and Services Tax Identification Number: {v.gstin}</div>
<table class="plain">
  <tr><td class="k">Legal Name</td><td>{v.legal_name}</td></tr>
  <tr><td class="k">Trade Name, if any</td><td>{v.trade_name}</td></tr>
  <tr><td class="k">Additional trade names, if any</td><td></td></tr>
</table>
<div class="annex-h">Details of Managing / Whole-time Directors and Key Managerial Persons</div>
<table class="main" style="border:none">
  <style>table.main td {{ border: none; }}</style>
  {directors_rows}
</table>

</body></html>"""


# ---------------------------------------------------------------------------
# UDYAM
# ---------------------------------------------------------------------------

def udyam_html(v: Vendor) -> str:
    band = """
    <div class="band">
      <div style="font-size:20pt">&#127963;</div>
      <div class="c">
        <div class="hin">&#2349;&#2366;&#2352;&#2340; &#2360;&#2352;&#2325;&#2366;&#2352;</div>
        <div class="en">Government of India</div>
        <div class="hin">&#2360;&#2370;&#2325;&#2381;&#2359;&#2381;&#2350;, &#2354;&#2328;&#2369; &#2319;&#2357;&#2306; &#2350;&#2343;&#2381;&#2351;&#2350; &#2313;&#2342;&#2381;&#2351;&#2350; &#2350;&#2306;&#2340;&#2381;&#2352;&#2366;&#2354;&#2351;</div>
        <div class="en">Ministry of Micro, Small and Medium Enterprises</div>
      </div>
      <div class="msme">MSME<small>MICRO, SMALL &amp; MEDIUM ENTERPRISES</small></div>
    </div>
    """

    classification_rows = "".join(
        f"<tr><td class='center'>{i}</td><td class='center'>{yr}</td>"
        f"<td class='center'>{v.enterprise_type}</td><td class='center'>{dt}</td></tr>"
        for i, (yr, dt) in enumerate(
            [("2025-26", "01/04/2025"), ("2024-25", "27/04/2024"),
             ("2023-24", "09/05/2023"), ("2022-23", "26/06/2022")], start=1)
    )

    return f"""<!doctype html>
<html><head><meta charset="utf-8"><style>{_UDYAM_CSS}</style></head><body>

<div class="frame">
{band}
<h1 class="title">UDYAM REGISTRATION CERTIFICATE</h1>

<table class="kv">
  <tr><td class="k">UDYAM REGISTRATION NUMBER</td><td class="v center">{v.udyam}</td></tr>
  <tr><td class="k">NAME OF ENTERPRISE</td><td class="v center">{v.legal_name}</td></tr>
</table>

<table class="kv">
  <tr>
    <td class="k">TYPE OF ENTERPRISE <sup>*</sup></td>
    <td>
      <table class="grid" style="width:78%">
        <tr><th>SNo.</th><th>Classification Year</th><th>Enterprise Type</th><th>Classification Date</th></tr>
        {classification_rows}
      </table>
    </td>
  </tr>
</table>

<table class="kv">
  <tr><td class="k">MAJOR ACTIVITY</td>
      <td><div class="activity" style="width:78%">{v.major_activity}</div></td></tr>
  <tr><td class="k">SOCIAL CATEGORY OF ENTREPRENEUR</td><td class="v center">GENERAL</td></tr>
</table>

<table class="kv">
  <tr><td class="k">NAME OF UNIT(S)</td>
    <td>
      <table class="grid" style="width:78%">
        <tr><th style="width:12%">S.No.</th><th>Name of Unit(s)</th></tr>
        <tr><td>1</td><td class="hk">{v.legal_name}.</td></tr>
      </table>
    </td>
  </tr>
</table>

<table class="kv">
  <tr><td class="k">OFFICAL ADDRESS OF ENTERPRISE</td>
    <td>
      <table class="grid" style="width:100%">
        <tr><td class="hk" style="width:22%">Flat/Door/Block No.</td><td style="width:20%">{v.udyam_premises}</td>
            <td class="hk" style="width:20%">Name of Premises/ Building</td><td>{v.premises_building}</td></tr>
        <tr><td class="hk">Village/Town</td><td>{v.city_town.title()}</td>
            <td class="hk">Block</td><td>{v.block}</td></tr>
        <tr><td class="hk">Road/Street/Lane</td><td>{v.udyam_road.title()}</td>
            <td class="hk">City</td><td>{v.city_town.title()}</td></tr>
        <tr><td class="hk">State</td><td>{v.state.upper()}</td>
            <td class="hk">District</td><td>{v.udyam_district_cell}</td></tr>
        <tr><td class="hk">Mobile</td><td>{v.mobile}</td>
            <td class="hk">Email:</td><td>{v.email}</td></tr>
      </table>
    </td>
  </tr>
</table>

<table class="kv">
  <tr><td class="k">DATE OF INCORPORATION / REGISTRATION OF ENTERPRISE</td>
      <td class="v center">{v.incorporated}</td></tr>
  <tr><td class="k">DATE OF COMMENCEMENT OF PRODUCTION/BUSINESS</td>
      <td class="v center">{v.commenced}</td></tr>
</table>

<table class="kv">
  <tr><td class="k">NATIONAL INDUSTRY CLASSIFICATION CODE(S)</td>
    <td>
      <table class="grid" style="width:88%">
        <tr><th>SNo.</th><th>NIC 2 Digit</th><th>NIC 4 Digit</th><th>NIC 5 Digit</th><th>Activity</th></tr>
        <tr>
          <td>1</td>
          <td class="hk">26 &ndash; Manufacture of computer, electronic and optical products</td>
          <td class="hk">2651 &ndash; Manufacture of measuring, testing, navigating and control equipment</td>
          <td class="hk">26517 &ndash; Manufacture of industrial process control equipment</td>
          <td class="hk">{v.major_activity.title()}</td>
        </tr>
      </table>
    </td>
  </tr>
</table>

<table class="kv">
  <tr><td class="k">DATE OF UDYAM REGISTRATION</td><td class="v center">{v.udyam_date}</td></tr>
</table>
</div>

<div class="pagebreak"></div>

<div class="frame" style="padding:4mm">
<div class="disclaimer">
<sup>*</sup> <b>In case of graduation (upward/reverse) of status of an enterprise, the
benefit of the Government Schemes will be availed as per the provisions of
Notification No. S.O. 2119(E) dated 26.06.2020 issued by the M/o MSME.</b>
</div>
<div class="disclaimer">Disclaimer: This is computer generated statement, no
signature required. Printed from https://udyamregistration.gov.in &amp; Date of
printing:- 10/04/2025</div>

<div class="contact-h">For any assistance, you may contact:</div>
<div class="contact-row"><b>1. District Industries Centre:</b> {v.dic}</div>
<div class="contact-row"><b>2. MSME-DFO:</b> {v.msme_dfo}</div>
</div>

<div class="pagebreak"></div>

{band}
<div class="urn-band">Udyam Registration Number&nbsp;&nbsp;:&nbsp;&nbsp;{v.udyam}</div>

<table class="grid" style="width:100%; margin-top:3mm">
  <tr><td class="hk" style="width:20%; text-align:right">Type of Enterprise</td><td style="width:30%">{v.enterprise_type.upper()}</td>
      <td class="hk" style="width:20%; text-align:right">Major Activity</td>
      <td style="width:30%"><span class="activity" style="display:block">{v.major_activity.title()}</span></td></tr>
  <tr><td class="hk" style="text-align:right">Type of Organisation</td><td>{v.org_type}</td>
      <td class="hk" style="text-align:right">Name of Enterprise</td><td>{v.legal_name}</td></tr>
  <tr><td class="hk" style="text-align:right">Owner Name</td><td>M/S {v.legal_name}</td>
      <td class="hk" style="text-align:right">PAN</td><td>{v.pan}</td></tr>
  <tr><td class="hk" style="text-align:right">Do you have GSTIN</td><td>Yes</td>
      <td class="hk" style="text-align:right">Mobile No.</td><td>{v.mobile}</td></tr>
  <tr><td class="hk" style="text-align:right">Email Id</td><td>{v.email}</td>
      <td class="hk" style="text-align:right">Social Category</td><td>General</td></tr>
  <tr><td class="hk" style="text-align:right">Gender</td><td>Male</td>
      <td class="hk" style="text-align:right">Specially Abled(DIVYANG)</td><td>No</td></tr>
  <tr><td class="hk" style="text-align:right">Date of Incorporation</td><td>{v.incorporated}</td>
      <td class="hk" style="text-align:right">Date of Commencement of Production/Business</td><td>{v.commenced}</td></tr>
</table>

<h3 class="sec">Bank Details</h3><div class="sec-rule"></div>
<table class="grid" style="width:100%">
  <tr><th class="tight">Bank Name</th><th class="tight">IFS Code</th>
      <th class="tight">Bank Account Number</th></tr>
  <tr><td class="center">{v.bank_name}</td><td class="center">{v.ifsc}</td>
      <td class="center">{v.account_number}</td></tr>
</table>

<h3 class="sec">Employment Details</h3><div class="sec-rule"></div>
<table class="grid" style="width:100%">
  <tr><th>Male</th><th>Female</th><th>Other</th><th>Total</th></tr>
  <tr><td class="center">45</td><td class="center">0</td><td class="center">0</td><td class="center">45</td></tr>
</table>

<h3 class="sec">Investment in Plant and Machinery OR Equipment (in Rs.)</h3><div class="sec-rule"></div>
<table class="grid small" style="width:100%">
  <tr>
    <th>S.No.</th><th>Financial Year</th><th>Enterprise Type</th>
    <th>Written Down Value (WDV)</th>
    <th>Exclusion of cost of Pollution Control, Research &amp; Development and Industrial Safety Devices</th>
    <th>Net Investment in Plant and Machinery OR Equipment[(A)-(B)]</th>
    <th>Total Turnover (A)</th><th>Export Turnover (B)</th><th>Net Turnover [(A)-(B)]</th><th>Is Filed</th>
  </tr>
  <tr><td>1</td><td>2023-24</td><td>{v.enterprise_type}</td><td>7321668.00</td><td>0.00</td>
      <td>7321668.00</td><td>352620465.00</td><td>3477657.25</td><td>349142807.75</td><td>Yes</td></tr>
  <tr><td>2</td><td>2022-23</td><td>{v.enterprise_type}</td><td>4764393.00</td><td>0.00</td>
      <td>4764393.00</td><td>232495947.00</td><td>1094972.56</td><td>231400974.44</td><td>Yes</td></tr>
  <tr><td>3</td><td>2021-22</td><td>{v.enterprise_type}</td><td>5227550.00</td><td>0.00</td>
      <td>5227550.00</td><td>197352238.00</td><td>80557.60</td><td>197271680.40</td><td>Yes</td></tr>
</table>

<div class="pagebreak"></div>

<h3 class="sec">Unit(s) Details</h3><div class="sec-rule"></div>
<table class="grid" style="width:100%">
  <tr><th>SN</th><th>Unit Name</th><th>Flat</th><th>Building</th><th>Village/Town</th>
      <th>Block</th><th>Road</th><th>City</th><th>Pin</th><th>State</th></tr>
  <tr><td>1</td><td>{v.legal_name}.</td><td>{v.udyam_premises}</td><td>{v.premises_building}</td>
      <td>{v.city_town.title()}</td><td>{v.block}</td><td>{v.udyam_road.title()}</td>
      <td>{v.city_town.title()}</td><td>{v.pin}</td><td>{v.state.upper()}</td></tr>
</table>

<h3 class="sec">Official address of Enterprise</h3><div class="sec-rule"></div>
<table class="grid" style="width:100%">
  <tr><td class="hk" style="width:22%">Flat/Door/Block No.</td><td style="width:28%">{v.udyam_premises}</td>
      <td class="hk" style="width:22%">Name of Premises/ Building</td><td>{v.premises_building}</td></tr>
  <tr><td class="hk">Village/Town</td><td>{v.city_town.title()}</td>
      <td class="hk">Block</td><td>{v.block}</td></tr>
  <tr><td class="hk">Road/Street/Lane</td><td>{v.udyam_road.title()}</td>
      <td class="hk">City</td><td>{v.city_town.title()}</td></tr>
  <tr><td class="hk">State</td><td>{v.state.upper()}</td>
      <td class="hk">District</td><td>{v.udyam_district_cell}</td></tr>
  <tr><td class="hk">Mobile</td><td>{v.mobile}</td>
      <td class="hk">Email:</td><td>{v.email}</td></tr>
  <tr><td class="hk">Latitude</td><td></td><td class="hk">Longitude:</td><td></td></tr>
</table>

<h3 class="sec">National Industry Classification Code(S)</h3><div class="sec-rule"></div>
<table class="grid" style="width:100%">
  <tr><th>SNo.</th><th>Nic 2 Digit</th><th>Nic 4 Digit</th><th>Nic 5 Digit</th><th>Activity</th></tr>
  <tr><td>1</td>
      <td><b>26</b> - Manufacture of computer, electronic and optical products</td>
      <td><b>2651</b> - Manufacture of measuring, testing, navigating and control equipment</td>
      <td><b>26517</b> - Manufacture of industrial process control equipment</td>
      <td>{v.major_activity.title()}</td></tr>
</table>

<table class="grid" style="width:100%; margin-top:4mm">
  <tr><td>Are you interested to get registered on Government e-Market (GeM) Portal</td><td style="width:22%">No</td></tr>
  <tr><td>Are you interested to get registered on TReDS Portals(one or more)</td><td>No</td></tr>
  <tr><td>Are you interested to get registered on National Career Service(NCS) Portal</td><td>No</td></tr>
  <tr><td>Are you interested to get registered on NSIC B2B Portal</td><td>No</td></tr>
  <tr><td>Are you interested in availing Free .IN Domain and a business email ID</td><td>N/A</td></tr>
  <tr><td>Are you interested in getting registered on Skill India Digital Portal</td><td>N/A</td></tr>
  <tr><td>District Industries Centre</td><td>{v.dic}</td></tr>
  <tr><td>MSME-DFO</td><td>{v.msme_dfo}</td></tr>
  <tr><td>Date of Udyam Registration</td><td>{v.udyam_date}</td></tr>
  <tr><td>Date of Printing</td><td>10/04/2025</td></tr>
</table>

<table class="grid" style="width:100%; margin-top:4mm">
  <tr><td style="width:78%"><b>IEC Details</b></td><td></td></tr>
  <tr><td>IEC Number</td><td></td></tr>
  <tr><td>IEC Status</td><td>Inactive</td></tr>
  <tr><td>IEC Registration Date</td><td></td></tr>
</table>

</body></html>"""


# ---------------------------------------------------------------------------
# CANCELLED CHEQUE  (clean vector -- see module docstring caveat)
# ---------------------------------------------------------------------------

def cheque_html(v: Vendor) -> str:
    acct = v.account_number
    micr = f"000391  {v.pin}0081  {acct[-6:]}  30"
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><style>{_CHEQUE_CSS}</style></head><body>
<div class="cheque">

  <div class="side">SESHAASAI (C) / CTS - 2010</div>

  <div class="bank">{v.bank_name}</div>
  <div class="branch">{v.branch_line1}</div>
  <div class="branchaddr">{v.branch_line2}</div>
  <div class="ifsc">RTGS / NEFT / IFS Code : {v.ifsc}</div>

  <div class="payee-box">A/C&nbsp;&nbsp;PAYEE</div>
  <div class="valid">VALID FOR THREE MONTHS ONLY</div>
  <div class="datebox">
    <div></div><div></div><div></div><div></div><div></div><div></div><div></div><div></div>
  </div>
  <div class="datelbl">
    <span>D</span><span>D</span><span>M</span><span>M</span>
    <span>Y</span><span>Y</span><span>Y</span><span>Y</span>
  </div>

  <div class="pay">Pay</div>
  <div class="payline"></div>
  <div class="orderlbl">OR ORDER</div>

  <div class="rupees">Rupees</div>
  <div class="rupline1"></div>
  <div class="rupline2"></div>

  <div class="amtbox"><div class="rs">&#8377;</div></div>

  <div class="strike1"></div>
  <div class="strike2"></div>
  <div class="cancel">Cancelled</div>

  <div class="acbox">
    <div class="k">A/c No.</div>
    <div class="v">{acct}</div>
  </div>

  <div class="datestamp">29/10/24</div>

  <div class="acctype">
    CCGEN&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;CBS<br>
    <b>BUSINESS BANKING : NEW CURRENT ACCOUNT</b><br>
    Payable at par at all branches of {v.bank_name} Limited in India
  </div>

  <div class="forname">FOR {v.legal_name}.</div>

  <div class="signatory">AUTHORISED SIGNATORY</div>
  <div class="signhint">Please sign above</div>

  <div class="micr">{micr}</div>

</div>
</body></html>"""


# ---------------------------------------------------------------------------
# GROUND TRUTH
# ---------------------------------------------------------------------------

def _address_ground_truth(v: Vendor) -> str:
    """The address block of the ground-truth YAML, per address mode."""
    def entry(key: str, value: str, note: str = "") -> str:
        if not value:
            return f"  {key}:\n    absent: true"
        suffix = f"        # {note}" if note else ""
        return f"  {key}:\n    value: {value}\n    source: gst_certificate{suffix}"

    if not v.combined_address:
        return "\n".join([
            "  # Captioned sub-fields on GST page 1, so the resolver's combined-address",
            "  # path bails early here (city/state/pin already populated) exactly as it",
            "  # does for mb_control_systems.",
            entry("address_1", v.premises),
            entry("address_2", v.road),
            "  address_3:\n    absent: true",
            "  address_4:\n    absent: true",
            entry("city", v.city_town),
            entry("state", v.state),
            f'  pin_code:\n    value: "{v.pin}"\n    source: gst_certificate',
        ])

    l1, l2, l3, l4 = v.expect_lines
    return "\n".join([
        "  # GST page 1 carries the address as ONE uncaptioned line, so the",
        "  # segmenter runs. These expectations are reasoned from the source text",
        "  # (premises designators / named property / locality), NOT copied from",
        "  # segmenter output -- so a mismatch here is a real finding, not a",
        "  # tautology. See README for known disagreements.",
        f"  #   source line: {v.combined_address}",
        entry("address_1", l1),
        entry("address_2", l2),
        entry("address_3", l3),
        entry("address_4", l4),
        entry("city", v.expect_city),
        entry("state", v.expect_state),
        # Combined-address vendors carry no PIN anywhere -- not in the GST
        # line, and (since udyam_district_cell drops the ", Pin ..." tail
        # when pin is empty) not in the Udyam table either. Ground-truth it
        # as genuinely absent, so "extractor produced nothing" scores as
        # correct rather than `missed`.
        (f'  pin_code:\n    value: "{v.expect_pin or v.pin}"\n'
         f'    source: udyam_certificate'
         if (v.expect_pin or v.pin)
         else '  pin_code:\n    absent: true'),
    ])


def ground_truth_yaml(v: Vendor) -> str:
    return f"""# =============================================================================
# GROUND TRUTH -- {v.legal_name}  (SYNTHETIC FIXTURE)
# =============================================================================
# ALL VALUES ARE FABRICATED. Generated by
# app/eval/fixtures/generate_synthetic_docs.py, which renders GST REG-06 /
# Udyam / cancelled-cheque PDFs replicating the real layouts in
# app/uploads/09e0ff7aca.
#
# verified: true is asserted on a DIFFERENT basis than a real vendor's ground
# truth. Nothing was transcribed by a human here -- these values are the
# generator's own inputs, so they are correct BY CONSTRUCTION: the PDF says X
# because this file said X. That makes them a sound reference for measuring
# extraction, and NOT evidence that the pipeline handles real documents, whose
# OCR noise, scan artefacts and layout drift these fixtures do not reproduce.
#
# Address segmentation case exercised: {v.address_case}
#
# CHEQUE CAVEAT: the cheque fixture is clean vector text, not a scan. The real
# cheque OCRs badly (IFSC read as "1C1C0006278"). A correct bank/IFSC score
# here does not demonstrate robustness on real cheque images.
# =============================================================================

vendor_id: synthetic_{v.vid}
verified: true
verified_by: generator
verified_on: "2026-09-10"

documents:
  gst_certificate: gst_certificate.pdf
  udyam_certificate: udyam_certificate.pdf
  cancelled_cheque: cancelled_cheque.pdf

fields:

  # -- identity -------------------------------------------------------------
  vendor_name:
    value: {v.legal_name}
    source: gst_certificate        # "Legal Name", page 1
  company_type:
    value: {v.constitution}
    source: gst_certificate        # "Constitution of Business"
  nature_of_business:
    value: {v.major_activity.title()}
    source: udyam_certificate      # "MAJOR ACTIVITY"

  # -- statutory identifiers ------------------------------------------------
  gst_number:
    value: {v.gstin}
    source: gst_certificate
  pan:
    value: {v.pan}
    source: udyam_certificate      # page 3 "PAN"
  udyam_number:
    value: {v.udyam}
    source: udyam_certificate

  # -- principal address ----------------------------------------------------
{_address_ground_truth(v)}

  # -- banking --------------------------------------------------------------
  bank_name:
    value: {v.bank_name}
    source: cancelled_cheque
  ifsc:
    value: {v.ifsc}
    source: cancelled_cheque
  account_number:
    value: "{v.account_number}"
    source: cancelled_cheque

  # -- contact --------------------------------------------------------------
  email:
    value: {v.email}
    source: udyam_certificate
  telephone:
    value: "{v.mobile}"
    source: udyam_certificate
"""


def manifest_csv() -> str:
    rows = ["vendor_id,legal_name,gstin,pan,udyam,city,state,pin,"
            "address_mode,address_case,source_address"]
    for v in VENDORS + SEGMENTATION_VENDORS + LONG_ADDRESS_VENDORS:
        mode = "combined" if v.combined_address else "captioned"
        addr = (v.combined_address or v.additional_address).replace('"', '""')
        rows.append(
            f'{v.vid},"{v.legal_name}",{v.gstin},{v.pan},{v.udyam},'
            f'{v.city_town},{v.state},{v.pin},{mode},"{v.address_case}","{addr}"'
        )
    return "\n".join(rows) + "\n"


_README = """# Synthetic vendor document fixtures

ALL VALUES ARE FABRICATED. These files have no legal or registration validity
and describe no real entity. They are OCR/extraction test fixtures.

## Why these exist

The earlier `synthetic_vendor_forms_all_20` set rendered every field in a
generic two-column grid that no real vendor document uses. Extraction failed on
it for a fixture artefact -- the label-to-value gap was ~21 label-heights where
`field_dictionary.yaml` allows at most 15.0 -- so the result measured the
fixture, not the pipeline.

These replicate the ACTUAL layouts of the three documents in
`app/uploads/09e0ff7aca`: GST REG-06 (3 pages, numbered bordered table,
Annexure A + B), Udyam certificate (4 pages, nested tables, dark MSME header
band), cancelled cheque.

## What these do and do not test

Tested: field/label matching against real label wording and real table
geometry; multi-page and annexure handling; the address segmenter, via the
Annexure A run-on address line; format validators (identifiers are
structurally valid).

NOT tested: OCR robustness on degraded input. Every fixture is clean vector
text. The cheque especially -- a real cancelled cheque is a scan with guilloche
security patterning, handwriting and a MICR line, and OCRs badly (the real
one's IFSC comes through as "1C1C0006278"). These render sharp and will OCR
near-perfectly. A high score here is a lower bound on difficulty, not a
demonstration that real documents work.

## Two vendor sets

**`vendor_01`..`vendor_05` -- captioned address.** GST page 1 prints the address
as captioned sub-fields (Building No./Road/City/District/State/PIN), the
mb_control_systems shape. field_matcher reads city/state/pin straight off the
captions, so `_resolve_combined_address` hits its early-bail and the segmenter
never runs. These measure field matching, not segmentation. Their Annexure A
still carries a run-on line per the table below.

| vendor | Annexure A case |
|---|---|
| vendor_01 | rural multi-locality run (4 locality fragments) |
| vendor_02 | industrial estate (GIDC keyword) |
| vendor_03 | urban comma-separated with highway fragment |
| vendor_04 | comma-less run-on (injection fallback) |
| vendor_05 | explicit VILL-/P.O. keywords |

**`seg_01`..`seg_20` -- combined address.** GST page 1 prints ONE uncaptioned
line, so the segmenter actually runs and address_1..4 are genuinely scored.

Ground truth for these is reasoned from the source text, not copied from
segmenter output:

    line 1   premises / unit / floor / block designators   "where in the building"
    line 2   the named property or estate                  "which building"
    line 3   locality or area                              "where in the city"

City and state are peeled into their own fields and never appear in a line.

**These seg_* fixtures do not currently work, and the failure is the
fixture's, not the pipeline's.**

A combined address has to reach the segmenter as one line. Headless Chrome
will not deliver that. Every route was tried and each corrupts the line before
extraction sees it:

| rendering | what extraction received |
|---|---|
| vector text layer | `'FLAT NO. 302'`, `', BLOCK C'`, `', 3RD FLOOR'` -- split per comma |
| `nowrap` + non-breaking spaces | `'FLAT'`, `'NO.'`, `'302'`, `','` -- split per word, worse |
| raster @ 200 dpi | `'FLATNO.302,BLOCKC,3RDFLOOR,SAIINDUSTRIALPARK,'` -- spaces dropped |
| raster @ 300/400 dpi | spacing better, commas corrupted to periods; still two spans |

So `address_1` came back as `"302"`, `"Village/Town Block"`, or
`"302,BLOCKC,3RDFLOOR,SAIINDUS..."` depending on the attempt. The real REG-06
avoids this because it is a genuine digital PDF whose text layer holds
`'Building No./Flat No.: 1ST FLOOR'` as one clean span; Chrome's PDF writer
does not reproduce that for a long uncaptioned line.

**Use `app/eval/address_vendor_lines.yaml` instead** to test these twenty
addresses. It feeds the strings straight to `resolve_address_blob(multiline=
True)`, with no rendering layer to corrupt them:

    python -m app.eval.eval_address --multiline --cases address_vendor_lines.yaml

That file carries the measured baseline and the two real defects it found (the
level-1 taxonomy collapse that prevents separating a named property from the
premises designators, and three city-resolution promotions). The `seg_*`
fixtures are retained only because the GST/Udyam/cheque documents around the
address are still faithful; their address_1..4 scores should be ignored.

**`gu_01`..`gu_20` -- long-address stress set.** Same combined-address GST
shape as `seg_*`, but each address is 12-14 comma-separated fragments (the
`seg_*` set is 5-7), specifically to exercise the segmenter's 4-line cap and
merge-the-weakest-boundary logic under real pressure.

Building and testing this set found and fixed THREE real pipeline bugs, all
regression-tested in `tests/test_address_segmenter.py` and confirmed against
the 709-test suite / 55-55 legacy eval / 19-24 ground-truth gate:

1. `_split_known_city_prefix` (address_resolver.py) peeled a known city name
   off the front of ANY segment, including a street ("GHAZIPUR ROAD"), not
   just a genuine city+locality run-on ("NOIDA SECTOR 62"). Fixed by requiring
   the remainder to classify as locality/village_po/unknown, which a street
   does not.
2. `field_matcher._is_inline_split` treated a bare "." as a valid caption
   separator unconditionally. Any address starting "FLAT NO. 302..." then had
   its own leading words misread as a caption+value split against the
   dictionary's "Flat No" label, discarding everything before the period --
   address_1 came back as just "302". Fixed by excluding a "." immediately
   followed by a digit (an abbreviation, not a separator).
3. `field_matcher._join_value` only ever stitched spans on ONE visual line.
   A combined address that wraps across 2-3 table rows (which these long
   addresses reliably do) lost every fragment after the first row. Extended
   to also gather CONTINUATION rows below, guarded by matching left-edge
   position, tight vertical spacing, and not looking caption-like -- so an
   unrelated field below is never absorbed.

**A residual, unfixed limitation**: PaddleOCR at 200dpi (this project's
production RENDER_DPI, matched deliberately) drops inter-word spaces on these
fixtures' long, small-font combined-address lines ("SAI INDUSTRIAL PARK" reads
as "SAIINDUSTRIALPARK"). That is a rendering/OCR fidelity gap in THIS
generator, not a segmenter defect -- `resolve_address_blob()` splits these
same addresses correctly when fed properly spaced text (verified directly,
see address_vendor_lines.yaml). Because of it, `gu_*`'s address_1..4/city
scores under-represent what the segmenter actually does; treat this set as
exercising field-matching and the multi-row join fixed above, not as a
segmentation-accuracy measurement -- use address_vendor_lines.yaml for that.

## Regenerating

    python -m app.eval.fixtures.generate_synthetic_docs --out <dir>

Requires Google Chrome (headless HTML->PDF). Deterministic: same inputs
produce the same PDFs.

## Running the eval

    python -m app.eval.eval_extraction \\
        --ground-truth <dir>/vendor_01/ground_truth.yaml \\
        --documents <dir>/vendor_01
"""


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument("--only", help="Generate a single vendor (e.g. vendor_01)")
    parser.add_argument(
        "--no-raster", action="store_true",
        help="Leave the Udyam certificate and cheque as vector PDFs with a text "
             "layer. They then take the pdf_text path instead of OCR, which is "
             "NOT how their real counterparts are processed -- use only to "
             "isolate field-matching behaviour from OCR error.",
    )
    args = parser.parse_args()

    chrome = find_chrome()
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    all_vendors = VENDORS + SEGMENTATION_VENDORS + LONG_ADDRESS_VENDORS
    targets = [v for v in all_vendors if not args.only or v.vid == args.only]
    if not targets:
        raise SystemExit(f"No vendor matching {args.only!r}")

    for v in targets:
        vdir = out_root / v.vid
        vdir.mkdir(parents=True, exist_ok=True)
        print(f"  {v.vid}: {v.legal_name}")
        # The real REG-06 is a digital PDF and takes the pdf_text path in
        # production, so the CAPTIONED vendors keep their text layer.
        #
        # The COMBINED-address vendors must be rasterised. Chrome's text layer
        # shatters a long uncaptioned line into per-word spans, so field_matcher
        # only ever sees a fragment and the segmenter never receives the line it
        # exists to split -- address_1 came back as "302". OCR reassembles words
        # into lines, so rasterising is what actually delivers the combined line.
        # This trades one fidelity (real GST is pdf_text) for another (the line
        # arrives whole); for a fixture whose purpose is testing segmentation,
        # the intact line is what matters.
        gst = vdir / "gst_certificate.pdf"
        # Combined-address vendors are rasterised below (they must be, to
        # deliver the address line intact -- see rasterize_pdf()'s docstring),
        # which makes the watermark OCR-visible no matter how light its fill:
        # PaddleOCR read it as a genuine 'Goods and Services Tax' span sitting
        # inside the address block's row band even at #f3f3f3, corrupting
        # which line counted as the value's own left edge. Dropped entirely
        # for this set rather than continuing to tune a fill value against a
        # detector whose threshold isn't known.
        html_to_pdf(gst_html(v, watermark=not v.combined_address), gst, chrome)
        if v.combined_address and not args.no_raster:
            rasterize_pdf(gst)

        # Udyam and the cheque are scans in production. Rasterise so they take
        # the OCR path like their real counterparts -- see rasterize_pdf().
        udyam = vdir / "udyam_certificate.pdf"
        html_to_pdf(udyam_html(v), udyam, chrome)
        if not args.no_raster:
            rasterize_pdf(udyam)

        cheque = vdir / "cancelled_cheque.pdf"
        html_to_pdf(cheque_html(v), cheque, chrome)
        if not args.no_raster:
            rasterize_pdf(cheque)
        (vdir / "ground_truth.yaml").write_text(ground_truth_yaml(v), encoding="utf-8")

    (out_root / "manifest.csv").write_text(manifest_csv(), encoding="utf-8")
    (out_root / "README.md").write_text(_README, encoding="utf-8")

    print(f"\nWrote {len(targets)} vendor(s) to {out_root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
