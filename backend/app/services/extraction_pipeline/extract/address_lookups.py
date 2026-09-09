"""In-memory address reference data, loaded once from backend/data/address/.

Three tables, all built by app.cli.build_address_lookups:

    pin_state_district(pin)  -> (state, district) | None
    canonical_state(token)   -> "West Bengal" | None   (exact, alias, or
                                whitespace/punctuation-insensitive match)
    is_known_city(token)     -> bool                    (normalised match
                                against ~140k district / town names)

Everything is deterministic and local -- no network, no model. Loading is
lazy + cached so importing this module (e.g. for a unit test) is free until a
lookup actually runs.
"""

from __future__ import annotations

import csv
import re
from functools import lru_cache
from pathlib import Path

_DATA_DIR = Path(__file__).resolve().parents[4] / "data" / "address"
_PIN_CSV = _DATA_DIR / "pin_directory.csv"
_STATES_TXT = _DATA_DIR / "states.txt"
_CITIES_TXT = _DATA_DIR / "cities.txt"

# Normalise a token for fuzzy-but-cheap matching: casefold, drop all
# non-alphanumerics (so "S.A.S Nagar" == "sas nagar" == "SAS  Nagar").
_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def _norm(s: str) -> str:
    return _NON_ALNUM.sub("", (s or "").casefold())


@lru_cache(maxsize=1)
def _pin_table() -> dict[str, tuple[str, str]]:
    out: dict[str, tuple[str, str]] = {}
    if not _PIN_CSV.is_file():
        return out
    with _PIN_CSV.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            pin = (row.get("pincode") or "").strip()
            if len(pin) == 6 and pin.isdigit():
                out[pin] = ((row.get("state") or "").strip(), (row.get("district") or "").strip())
    return out


@lru_cache(maxsize=1)
def _state_tables() -> tuple[dict[str, str], dict[str, str]]:
    """(normalised-canonical -> canonical, normalised-alias -> canonical)."""
    canonical: dict[str, str] = {}
    aliases: dict[str, str] = {}
    if not _STATES_TXT.is_file():
        return canonical, aliases
    for line in _STATES_TXT.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "\t" in line:
            alias, target = line.split("\t", 1)
            aliases[_norm(alias)] = target.strip()
        else:
            canonical[_norm(line)] = line
    return canonical, aliases


@lru_cache(maxsize=1)
def _city_set() -> frozenset[str]:
    if not _CITIES_TXT.is_file():
        return frozenset()
    return frozenset(
        _norm(line)
        for line in _CITIES_TXT.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    )


# --- public ----------------------------------------------------------------

def pin_state_district(pin: str) -> tuple[str, str] | None:
    """(state, district) for a 6-digit PIN, or None if unknown."""
    return _pin_table().get((pin or "").strip()) or None


def canonical_state(token: str) -> str | None:
    """Map a token to its canonical state/UT name. Tries exact canonical, then
    known aliases / abbreviations / OCR misreads. Returns None if not a state."""
    n = _norm(token)
    if not n:
        return None
    canonical, aliases = _state_tables()
    return canonical.get(n) or aliases.get(n)


def is_known_city(token: str) -> bool:
    """True if the token normalises to a known city / district / town name."""
    n = _norm(token)
    return bool(n) and n in _city_set()


def data_files_present() -> bool:
    """True when all three lookup files exist -- callers can degrade gracefully
    (skip the PIN-directory path) rather than crash if they were never built."""
    return _PIN_CSV.is_file() and _STATES_TXT.is_file() and _CITIES_TXT.is_file()
