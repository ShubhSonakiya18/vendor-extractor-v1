"""Build the address-resolver lookup tables from source reference data.

One-off / occasional. Reads two source CSVs and writes three data files under
`backend/data/address/`:

    pin_directory.csv   pincode,district,state   (deduped; the authoritative
                        PIN -> district/state map used to fix OCR errors and
                        fall back when a state/city token is missing)
    states.txt          one canonical state/UT name per line, plus a
                        `<alias>\t<canonical>` section for OCR misreads and
                        abbreviations
    cities.txt          one city / district name per line, casefolded, used to
                        recognise a city token inside an address blob

`cities.txt` is built from the DISTRICT column of the PIN directory only
(~740 clean names) plus a small hand-curated list of common city names whose
own spelling differs from the district they sit in (Bengaluru vs "Bengaluru
Urban", Mohali vs "S.A.S Nagar", Gurgaon vs "Gurugram", ...). An earlier
version also folded in the PIN directory's `officename` column, but that is
~65k branch/sub-post-office names ("natibpur bo", "moti nagar so") -- not
cities -- and made `is_known_city` return True for localities it should
reject. The office column is deliberately no longer used.

Usage (only --pin-csv is needed now):
    python -m app.cli.build_address_lookups \
        --pin-csv "C:/path/to/india_post_pincode_directory.csv"

Or, to rebuild the tables in place from the already-committed
`backend/data/address/pin_directory.csv` (its `district`/`state` columns are
the same authoritative data the source CSV carries):
    python -m app.cli.build_address_lookups --pin-csv backend/data/address/pin_directory.csv

The large source CSV is NOT committed. The generated files ARE -- they are the
runtime input. Re-run this when a newer PIN directory is published.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

_OUT_DIR = Path(__file__).resolve().parents[2] / "data" / "address"

# OCR misreads / short forms seen on real documents, mapped to the canonical
# name. Extend as new ones show up. Keys are casefolded.
_STATE_ALIASES: dict[str, str] = {
    "west bangal": "West Bengal",
    "west bengol": "West Bengal",
    "wb": "West Bengal",
    "up": "Uttar Pradesh",
    "uttarpradesh": "Uttar Pradesh",
    "mh": "Maharashtra",
    "tn": "Tamil Nadu",
    "tamilnadu": "Tamil Nadu",
    "ka": "Karnataka",
    "ap": "Andhra Pradesh",
    "andhrapradesh": "Andhra Pradesh",
    "telengana": "Telangana",
    "orissa": "Odisha",
    "pondicherry": "Puducherry",
    "nct of delhi": "Delhi",
    "new delhi": "Delhi",
    "j&k": "Jammu and Kashmir",
    "jk": "Jammu and Kashmir",
}

# statename in the PIN CSV is ALL CAPS with a few non-standard forms; map to
# the clean canonical spelling.
_STATE_CANONICAL: dict[str, str] = {
    "ANDAMAN AND NICOBAR ISLANDS": "Andaman and Nicobar Islands",
    "ANDHRA PRADESH": "Andhra Pradesh",
    "ARUNACHAL PRADESH": "Arunachal Pradesh",
    "ASSAM": "Assam",
    "BIHAR": "Bihar",
    "CHANDIGARH": "Chandigarh",
    "CHHATTISGARH": "Chhattisgarh",
    "DELHI": "Delhi",
    "GOA": "Goa",
    "GUJARAT": "Gujarat",
    "HARYANA": "Haryana",
    "HIMACHAL PRADESH": "Himachal Pradesh",
    "JAMMU AND KASHMIR": "Jammu and Kashmir",
    "JHARKHAND": "Jharkhand",
    "KARNATAKA": "Karnataka",
    "KERALA": "Kerala",
    "LADAKH": "Ladakh",
    "LAKSHADWEEP": "Lakshadweep",
    "MADHYA PRADESH": "Madhya Pradesh",
    "MAHARASHTRA": "Maharashtra",
    "MANIPUR": "Manipur",
    "MEGHALAYA": "Meghalaya",
    "MIZORAM": "Mizoram",
    "NAGALAND": "Nagaland",
    "ODISHA": "Odisha",
    "PUDUCHERRY": "Puducherry",
    "PUNJAB": "Punjab",
    "RAJASTHAN": "Rajasthan",
    "SIKKIM": "Sikkim",
    "TAMIL NADU": "Tamil Nadu",
    "TELANGANA": "Telangana",
    "THE DADRA AND NAGAR HAVELI AND DAMAN AND DIU": "Dadra and Nagar Haveli and Daman and Diu",
    "TRIPURA": "Tripura",
    "UTTAR PRADESH": "Uttar Pradesh",
    "UTTARAKHAND": "Uttarakhand",
    "WEST BENGAL": "West Bengal",
}

_PIN_LEN = 6

# Common city names whose spelling differs from the PIN directory's district
# name for the same place, so a district-only city list would miss them. Left
# side is what a document actually prints; it is emitted into cities.txt as-is
# (the list is only used for is_known_city() boolean checks, not for
# canonicalising a value). Extend as real documents surface more.
_COMMON_CITY_ALIASES: set[str] = {
    "bangalore", "bengaluru",          # district: Bengaluru Urban / Rural
    "mohali", "sas nagar",             # district: S.A.S Nagar
    "gurgaon",                         # district: Gurugram
    "noida", "greater noida",          # district: Gautam Buddha Nagar
    "navi mumbai",                     # district: Thane / Raigad
    "secunderabad",                    # district: Hyderabad
    "new delhi", "delhi",              # districts are the zones (New Delhi, Central, ...)
    "kolkata", "calcutta",
    "mumbai", "bombay",
    "chennai", "madras",
    "pune", "poona",
    "prayagraj", "allahabad",          # district renamed
    "vadodara", "baroda",
    "kochi", "cochin",
    "thiruvananthapuram", "trivandrum",
    "puducherry", "pondicherry",
    "vijayawada",                      # district: Krishna / NTR
    "visakhapatnam", "vizag",
    "indore", "bhopal", "nagpur", "surat", "jaipur", "lucknow", "kanpur",
    "patna", "bhubaneswar", "guwahati", "chandigarh", "ludhiana", "agra",
    "nashik", "faridabad", "ghaziabad", "rajkot", "coimbatore", "madurai",
}


# canonical spelling keyed by casefolded-no-space, so an already-canonical
# value ("Dadra and Nagar Haveli and Daman and Diu") round-trips instead of
# being .title()'d into "...And...".
_STATE_BY_NORM: dict[str, str] = {
    v.casefold().replace(" ", ""): v for v in _STATE_CANONICAL.values()
}


def _canon_state(raw: str) -> str:
    raw = (raw or "").strip()
    if raw.upper() in _STATE_CANONICAL:
        return _STATE_CANONICAL[raw.upper()]
    hit = _STATE_BY_NORM.get(raw.casefold().replace(" ", ""))
    return hit or raw.title()


def build_pin_directory(pin_csv: Path, out: Path) -> tuple[int, int]:
    """pincode,district,state -- one row per (pincode, district). A PIN can map
    to more than one district only across state borders in rare cases; keep the
    first seen and count collisions."""
    seen: dict[str, tuple[str, str]] = {}
    collisions = 0
    with pin_csv.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            pin = (row.get("pincode") or "").strip()
            if len(pin) != _PIN_LEN or not pin.isdigit():
                continue
            district = (row.get("district") or "").strip().title()
            # source CSV calls it "statename"; the already-built
            # pin_directory.csv calls it "state" -- accept either so this can
            # rebuild in place.
            state = _canon_state(row.get("statename") or row.get("state") or "")
            if not state:
                continue
            if pin in seen:
                if seen[pin] != (district, state):
                    collisions += 1
                continue
            seen[pin] = (district, state)

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pincode", "district", "state"])
        for pin in sorted(seen):
            d, s = seen[pin]
            w.writerow([pin, d, s])
    return len(seen), collisions


def build_states(out: Path) -> int:
    canon = sorted(set(_STATE_CANONICAL.values()))
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        f.write("# Canonical Indian state / UT names, one per line.\n")
        for s in canon:
            f.write(s + "\n")
        f.write("\n# Aliases: <casefolded misread/abbrev>\\t<canonical>\n")
        for alias, target in sorted(_STATE_ALIASES.items()):
            f.write(f"{alias}\t{target}\n")
    return len(canon) + len(_STATE_ALIASES)


def build_cities(pin_csv: Path, out: Path) -> int:
    """City names, casefolded, deduped. The DISTRICT column of the PIN
    directory (authoritative, ~740 clean names) plus _COMMON_CITY_ALIASES for
    the well-known cities whose own name differs from their district. The
    `officename` column is intentionally NOT used -- see module docstring."""
    names: set[str] = set()
    with pin_csv.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames and "district" not in reader.fieldnames:
            raise SystemExit(
                f"{pin_csv} has no 'district' column (columns: {reader.fieldnames})"
            )
        for row in reader:
            v = (row.get("district") or "").strip()
            if v and not v.isdigit() and len(v) > 2:
                names.add(v.casefold())

    names |= _COMMON_CITY_ALIASES

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        f.write("# City / district names, casefolded, one per line. Built from the\n")
        f.write("# PIN directory's district column + a curated common-city alias list.\n")
        for n in sorted(names):
            f.write(n + "\n")
    return len(names)


def main() -> int:
    ap = argparse.ArgumentParser(description="Build address lookup tables")
    ap.add_argument("--pin-csv", required=True, type=Path,
                    help="India Post PIN directory, or the committed "
                         "backend/data/address/pin_directory.csv")
    ap.add_argument("--out-dir", type=Path, default=_OUT_DIR)
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass

    if not args.pin_csv.is_file():
        raise SystemExit(f"PIN CSV not found: {args.pin_csv}")

    out_pin = args.out_dir / "pin_directory.csv"
    n_pin, collisions = build_pin_directory(args.pin_csv, out_pin)
    n_states = build_states(args.out_dir / "states.txt")
    # cities come from the district column; read the just-written pin_directory
    # so this works whether --pin-csv was the raw source or the committed file.
    n_cities = build_cities(out_pin, args.out_dir / "cities.txt")

    print(f"pin_directory.csv : {n_pin} pincodes  ({collisions} cross-district collisions kept-first)")
    print(f"states.txt        : {n_states} entries (canonical + aliases)")
    print(f"cities.txt        : {n_cities} names")
    print(f"written to        : {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
