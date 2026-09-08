"""
V2 Validation Engine
====================
One generic executor for every rule in validation_rules.yaml. V1 carried the
same six regexes in two files that could drift apart; here a rule exists once,
in YAML, and this module runs whatever it finds.

Rule types map to `_RULES` entries. `derived` rules compare a field against
something computed from ANOTHER field -- that is how "the PAN inside the GSTIN
must equal the PAN field" is expressed without either field name appearing in
Python.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from rapidfuzz import fuzz

from ..config_loader import ValidationRules, ValidatorSpec

# GST state codes 01-38 are assigned; the rest are unallocated.
_VALID_STATE_CODES = {f"{i:02d}" for i in range(1, 39)}


@dataclass
class Finding:
    validator: str
    severity: str      # error | warning
    message: str
    ok: bool

    def to_dict(self) -> dict:
        return {
            "validator": self.validator,
            "severity": self.severity,
            "message": self.message,
            "ok": self.ok,
        }


# ---------------------------------------------------------------------------
# RULE IMPLEMENTATIONS
# ---------------------------------------------------------------------------

def _rule_regex(spec: ValidatorSpec, value: str, values: dict) -> bool:
    return bool(spec.pattern and spec.pattern.fullmatch(value))


def _rule_length(spec: ValidatorSpec, value: str, values: dict) -> bool:
    if spec.digits_only and not value.isdigit():
        return False
    if spec.min is not None and len(value) < spec.min:
        return False
    if spec.max is not None and len(value) > spec.max:
        return False
    return True


def _rule_enum(spec: ValidatorSpec, value: str, values: dict) -> bool:
    target = value.strip().lower()
    return any(target == allowed.strip().lower() for allowed in spec.values)


def _rule_non_empty(spec: ValidatorSpec, value: str, values: dict) -> bool:
    return bool(value and value.strip())


def _derived_substring_equals(spec: ValidatorSpec, value: str, values: dict) -> Optional[bool]:
    """Compare a slice of `source` against `target`. Returns None ("cannot
    judge") when either side is absent -- a missing PAN is the PAN field's
    problem to report, not this rule's."""
    source_value = values.get(spec.source or "")
    target_value = values.get(spec.target or "")
    if not source_value or not target_value:
        return None
    start = spec.start or 0
    end = spec.end if spec.end is not None else len(source_value)
    return source_value[start:end].upper() == target_value.upper()


def _derived_state_code_valid(spec: ValidatorSpec, value: str, values: dict) -> Optional[bool]:
    source_value = values.get(spec.source or "")
    if not source_value:
        return None
    start = spec.start or 0
    end = spec.end if spec.end is not None else 2
    return source_value[start:end] in _VALID_STATE_CODES


def _derived_pin_matches_state(spec: ValidatorSpec, value: str, values: dict) -> Optional[bool]:
    """The PIN directory knows the state a 6-digit PIN belongs to. Compare it
    against the extracted state. Returns None ("cannot judge") when the state
    field is blank, the PIN field (`spec.source`) is blank, or the PIN is not
    in the directory -- an absent value is not this rule's to report, and a
    lookup miss is not evidence of a mismatch."""
    from .address_lookups import canonical_state, pin_state_district

    pin = (values.get(spec.source or "") or "").strip()
    if not value.strip() or len(pin) != 6 or not pin.isdigit():
        return None
    looked_up = pin_state_district(pin)
    if not looked_up:
        return None
    pin_state = looked_up[0]
    # canonicalise both sides so "west bengal" / "WB" / "West Bengal" agree.
    got = canonical_state(value) or value.strip()
    want = canonical_state(pin_state) or pin_state
    return got.casefold() == want.casefold()


_RULES: dict[str, Callable[[ValidatorSpec, str, dict], bool]] = {
    "regex": _rule_regex,
    "length": _rule_length,
    "enum": _rule_enum,
    "non_empty": _rule_non_empty,
}

_DERIVED: dict[str, Callable[[ValidatorSpec, str, dict], Optional[bool]]] = {
    "substring_equals": _derived_substring_equals,
    "state_code_valid": _derived_state_code_valid,
    "pin_matches_state": _derived_pin_matches_state,
}


# ---------------------------------------------------------------------------
# ENGINE
# ---------------------------------------------------------------------------

class Validator:
    def __init__(self, rules: ValidationRules):
        self.rules = rules

    def check(self, validator_names: list[str], value: str, all_values: dict) -> list[Finding]:
        """Run named validators against `value`. `all_values` gives derived
        rules access to the other extracted fields."""
        findings: list[Finding] = []
        for name in validator_names:
            spec = self.rules.get(name)

            if spec.type == "derived":
                fn = _DERIVED.get(spec.rule or "")
                if fn is None:
                    raise KeyError(f"Derived rule not implemented: {spec.rule!r}")
                outcome = fn(spec, value, all_values)
                if outcome is None:
                    continue  # not enough data to judge
                ok = outcome
            else:
                fn = _RULES.get(spec.type)
                if fn is None:
                    raise KeyError(f"Validator type not implemented: {spec.type!r}")
                ok = fn(spec, value, all_values)

            findings.append(
                Finding(validator=name, severity=spec.severity, message=spec.message, ok=ok)
            )
        return findings

    def status_of(self, findings: list[Finding]) -> str:
        if any(not f.ok and f.severity == "error" for f in findings):
            return "invalid"
        if any(not f.ok and f.severity == "warning" for f in findings):
            return "warning"
        return "valid" if findings else "not_validated"

    # -- cross-document -----------------------------------------------------

    def compare_across_documents(self, values_by_document: dict[str, str]) -> tuple[str, list[str]]:
        """Report whether the same field agrees across the documents it was
        found in. Comparison is fuzzy so that 'M B CONTROL & SYSTEMS PVT LTD'
        and 'M.B.CONTROL & SYSTEM PVT LTD' count as agreement, while a
        genuinely different company does not."""
        distinct = {k: v for k, v in values_by_document.items() if v}
        if len(distinct) <= 1:
            return ("single_source" if distinct else "not_checked"), []

        threshold = self.rules.cross_document.similarity_threshold * 100
        items = list(distinct.items())
        disagreements: list[str] = []
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                (doc_a, val_a), (doc_b, val_b) = items[i], items[j]
                if fuzz.ratio(val_a.lower(), val_b.lower()) < threshold:
                    disagreements.append(f"{doc_a}={val_a!r} vs {doc_b}={val_b!r}")

        return ("inconsistent" if disagreements else "consistent"), disagreements
