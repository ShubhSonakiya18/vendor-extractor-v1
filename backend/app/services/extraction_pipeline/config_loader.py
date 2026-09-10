"""
V2 Configuration Loader
=======================
Reads config/field_dictionary.yaml and config/validation_rules.yaml into typed
objects, and validates them on the way in.

The validation matters more than it looks. This config is the extension point:
someone adding their 500th field will typo a validator name or a normalizer op,
and the failure mode without checking is silent -- the field simply never
matches, and nobody notices until an Excel cell is blank. So loading is strict:
unknown keys, unresolvable validator references, bad regexes and mis-weighted
confidence all raise at load time, naming the field.

Nothing here knows what a GSTIN is. Field behaviour comes entirely from YAML.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field as dc_field
from pathlib import Path
from typing import Optional

import yaml

# backend/app/services/extraction_pipeline/config_loader.py -> backend/config
# parents: [0] extraction_pipeline  [1] services  [2] app  [3] backend
# This index is tied to the package's depth: move the package and it must change.
CONFIG_DIR = Path(__file__).resolve().parents[3] / "config"

# Ops the normalizer (Step 6) implements. Declared here so a typo in YAML is a
# load-time error rather than a silently skipped transformation.
KNOWN_NORMALIZERS = {
    "strip",
    "collapse_spaces",
    "remove_spaces",
    "uppercase",
    "lowercase",
    "titlecase",
    "digits_only",
    "strip_country_code",
    "fix_ifsc_confusions",
    "split_corporate_suffix",
    "expand_known_phrase",
}

KNOWN_VALIDATOR_TYPES = {"regex", "length", "enum", "non_empty", "derived"}

KNOWN_SEARCH_DIRECTIONS = {"right", "below", "left", "above"}

# Minimum length for the unspaced-match fallback in
# `FieldSpec._find_unspaced_matches`. Set above the 2-character "CC" / "OD"
# account-type codes, which are the shortest documented false positives the
# token-boundary guard exists to prevent. See that method for the full
# rationale before changing this.
MIN_UNSPACED_MATCH_LEN = 5

_FIELD_KEYS = {
    "description",
    "labels",
    "patterns",
    "normalization",
    "validators",
    "value_type",
    "search",
    "expected_documents",
    "priority",
    "required",
    "cross_document_consistency",
    "confidence",
    "label_match",
    "default",
    "aliases",
    "derive_from",
    "exclude_patterns",
}

# Mailbox providers whose domain says nothing about the vendor. Deriving a
# website from "vendor@gmail.com" would report gmail.com as the company's site.
FREE_EMAIL_DOMAINS = frozenset({
    "gmail.com", "googlemail.com", "yahoo.com", "yahoo.co.in", "yahoo.in",
    "hotmail.com", "outlook.com", "live.com", "msn.com",
    "rediffmail.com", "rediff.com", "sify.com", "vsnl.net", "vsnl.com",
    "bsnl.in", "airtelmail.in", "icloud.com", "me.com", "aol.com",
    "protonmail.com", "proton.me", "zoho.com", "yandex.com", "mail.com",
})


def derive_value(rule: str, source_value: str) -> Optional[str]:
    """Compute a field value from another field's value.

    Returns None when the rule cannot produce a trustworthy answer, which the
    caller must treat as "leave the field empty" rather than guessing.
    """
    if rule == "email_domain":
        _, _, domain = source_value.strip().lower().partition("@")
        domain = domain.strip().strip(".")
        if not domain or "." not in domain:
            return None
        if domain in FREE_EMAIL_DOMAINS:
            return None
        return f"www.{domain}"
    raise ConfigError(f"unknown derive_from rule {rule!r}")


class ConfigError(ValueError):
    """Raised when the YAML configuration is malformed or inconsistent."""


# ---------------------------------------------------------------------------
# SPECS
# ---------------------------------------------------------------------------

@dataclass
class SearchSpec:
    directions: list[str]
    max_distance: float
    same_line_bonus: float


@dataclass
class ConfidenceSpec:
    weights: dict[str, float]
    min_accept: float


@dataclass
class LabelMatchSpec:
    min_similarity: float
    max_label_chars: int


@dataclass
class FieldSpec:
    """One extractable field, fully resolved against the defaults block."""

    key: str
    description: str
    labels: list[str]
    patterns: list[re.Pattern]
    normalization: list[str]
    validators: list[str]
    value_type: str
    search: SearchSpec
    confidence: ConfidenceSpec
    label_match: LabelMatchSpec
    expected_documents: list[str]
    priority: int
    required: bool
    cross_document_consistency: bool
    default: Optional[str] = None
    # Same patterns as `patterns`, but guarded so they cannot match inside a
    # longer alphanumeric run. Built at load time; see _boundary_wrap.
    scan_patterns: list[re.Pattern] = dc_field(default_factory=list)
    # Canonical value -> patterns that should collapse to it. Lets a field
    # report one spelling regardless of how the document (or the OCR engine)
    # rendered it: a cheque's ICICI logo reads as "AICICIBank" and its
    # boilerplate as "ICiCI Bank Limited", both of which should surface as
    # "ICICI Bank". Applied after normalization; see `canonicalize`.
    aliases: list[tuple[str, re.Pattern]] = dc_field(default_factory=list)
    # Fill this field from another field when no candidate was found on any
    # document, e.g. {"field": "email", "rule": "email_domain"}. The result is
    # INFERRED, not read off a page -- see `derive_value` and the review flag
    # raised in semantic_engine.
    derive_from: Optional[dict] = None
    # Values matching any of these are rejected outright, even when they match
    # `patterns`. For boilerplate that is structurally valid but semantically
    # wrong -- a government portal URL is a perfectly well-formed website, just
    # never this vendor's.
    exclude_patterns: list[re.Pattern] = dc_field(default_factory=list)

    @property
    def canonical_label(self) -> str:
        return self.labels[0] if self.labels else self.key

    def matches_pattern(self, value: str) -> bool:
        """True if the value fully matches any configured pattern. A field with
        no patterns (a name, an address) accepts anything -- for those, label
        and position are the only evidence available."""
        if not self.patterns:
            return True
        return any(p.fullmatch(value) for p in self.patterns)

    def is_excluded(self, value: str) -> bool:
        """True if this value is boilerplate this field must never report."""
        return any(p.search(value) for p in self.exclude_patterns)

    def canonicalize(self, value: str) -> str:
        """Collapse a recognised spelling to its canonical form.

        Returns the value unchanged when no alias matches, so a field with no
        `aliases` configured is entirely unaffected.
        """
        for canonical, pattern in self.aliases:
            if pattern.search(value):
                return canonical
        return value

    def find_pattern_matches(self, text: str) -> list[str]:
        """All substrings of `text` matching any pattern, at token boundaries.

        Boundary guarding is essential rather than cosmetic. Scanning the real
        sample documents without it, the 6-digit PIN pattern matched "627851"
        inside the account number 627851000539, the phone pattern matched
        "6278510005" inside that same number, and the account-type pattern
        matched "CC" inside "CCGEN" and "OD" inside "PRODUCTION". Every one of
        those is a confident-looking value that is simply wrong.
        """
        found: list[str] = []
        for pattern in self.scan_patterns or self.patterns:
            found.extend(m.group(0) for m in pattern.finditer(text))
        if found:
            return found
        return self._find_unspaced_matches(text)

    def _find_unspaced_matches(self, text: str) -> list[str]:
        """Fallback for OCR engines that drop inter-word spaces.

        RapidOCR's shipped recogniser uses a Chinese charset, where inter-word
        spaces do not exist, and it runs Latin words together: it reads a cheque
        line as "BUSINESSBANKING:NEW CURRENTACCOUNT" where PaddleOCR reads
        "BUSINESS BANKING: NEW CURRENT ACCOUNT". The boundary guard above then
        correctly refuses to match "CURRENT" inside "CURRENTACCOUNT" -- correct,
        because that guard is what stops "CC" matching inside "CCGEN".

        So this relaxation is deliberately narrow. It applies only when the
        guarded scan found nothing at all, and only to matches that are
        **alphabetic and at least MIN_UNSPACED_MATCH_LEN characters**. That
        threshold is chosen against the exact false positives the guard exists
        to prevent:

            "CC" in "CCGEN"                  -- 2 chars, rejected
            "OD" in "PRODUCTION"             -- 2 chars, rejected
            "627851" in "627851000539"       -- numeric, rejected
            "6278510005" in "627851000539"   -- numeric, rejected
            "CURRENT" in "CURRENTACCOUNT"    -- 7 alphabetic chars, ACCEPTED

        Numeric patterns are excluded outright: a digit run colliding inside a
        longer digit run is exactly the PIN-inside-account-number failure, and
        no length threshold makes that safe. Identifier fields (GSTIN, PAN,
        IFSC, account number) are all digit-bearing, so none of them can reach
        this path -- the relaxation cannot affect the fields that matter most.
        """
        found: list[str] = []
        for pattern in self.patterns:          # unguarded originals
            for match in pattern.finditer(text):
                value = match.group(0)
                compact = value.replace(" ", "")
                if len(compact) >= MIN_UNSPACED_MATCH_LEN and compact.isalpha():
                    found.append(value)
        return found


@dataclass
class ValidatorSpec:
    name: str
    type: str
    severity: str
    message: str
    pattern: Optional[re.Pattern] = None
    values: list[str] = dc_field(default_factory=list)
    min: Optional[int] = None
    max: Optional[int] = None
    digits_only: bool = False
    rule: Optional[str] = None
    source: Optional[str] = None
    target: Optional[str] = None
    start: Optional[int] = None
    end: Optional[int] = None


@dataclass
class CrossDocumentSpec:
    similarity_threshold: float
    source_precedence: list[str]
    severity: str
    message: str


@dataclass
class ValidationRules:
    validators: dict[str, ValidatorSpec]
    cross_document: CrossDocumentSpec

    def get(self, name: str) -> ValidatorSpec:
        try:
            return self.validators[name]
        except KeyError:
            raise ConfigError(f"Unknown validator: {name!r}") from None


@dataclass
class FieldDictionary:
    fields: dict[str, FieldSpec]
    version: int = 1

    def __iter__(self):
        return iter(self.fields.values())

    def __len__(self) -> int:
        return len(self.fields)

    def __getitem__(self, key: str) -> FieldSpec:
        return self.fields[key]

    def get(self, key: str) -> Optional[FieldSpec]:
        return self.fields.get(key)

    @property
    def keys(self) -> list[str]:
        return list(self.fields)

    @property
    def required_fields(self) -> list[str]:
        return [f.key for f in self if f.required]

    def by_priority(self) -> list[FieldSpec]:
        return sorted(self, key=lambda f: -f.priority)

    def with_patterns(self) -> list[FieldSpec]:
        return [f for f in self if f.patterns]


# ---------------------------------------------------------------------------
# LOADING
# ---------------------------------------------------------------------------

def _read_yaml(path: Path) -> dict:
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path.name} is not valid YAML: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"{path.name} must contain a top-level mapping")
    return data


def _compile(pattern: str, where: str) -> re.Pattern:
    try:
        return re.compile(pattern)
    except re.error as exc:
        raise ConfigError(f"{where}: invalid regex {pattern!r}: {exc}") from exc


def _boundary_wrap(pattern: str, where: str) -> re.Pattern:
    """Compile a pattern that cannot match inside a longer alphanumeric token.

    Plain \\b is not enough here: many of these patterns begin or end with
    non-word characters (a URL scheme, a '+' country code), where \\b asserts
    the opposite of what is wanted. Explicit lookarounds on [A-Za-z0-9] give
    the intended "not part of a bigger token" meaning regardless of what the
    pattern starts with.
    """
    return _compile(rf"(?<![A-Za-z0-9])(?:{pattern})(?![A-Za-z0-9])", where)


def load_validation_rules(path: Optional[Path] = None) -> ValidationRules:
    path = path or CONFIG_DIR / "validation_rules.yaml"
    data = _read_yaml(path)

    raw_validators = data.get("validators") or {}
    if not raw_validators:
        raise ConfigError(f"{path.name} defines no validators")

    validators: dict[str, ValidatorSpec] = {}
    for name, cfg in raw_validators.items():
        if not isinstance(cfg, dict):
            raise ConfigError(f"validator {name!r} must be a mapping")

        vtype = cfg.get("type")
        if vtype not in KNOWN_VALIDATOR_TYPES:
            raise ConfigError(
                f"validator {name!r} has unknown type {vtype!r}; "
                f"expected one of {sorted(KNOWN_VALIDATOR_TYPES)}"
            )

        severity = cfg.get("severity", "error")
        if severity not in {"error", "warning"}:
            raise ConfigError(f"validator {name!r}: severity must be error or warning")

        spec = ValidatorSpec(
            name=name,
            type=vtype,
            severity=severity,
            message=cfg.get("message", f"{name} failed"),
            values=[str(v) for v in (cfg.get("values") or [])],
            min=cfg.get("min"),
            max=cfg.get("max"),
            digits_only=bool(cfg.get("digits_only", False)),
            rule=cfg.get("rule"),
            source=cfg.get("source"),
            target=cfg.get("target"),
            start=cfg.get("start"),
            end=cfg.get("end"),
        )

        if vtype == "regex":
            if not cfg.get("pattern"):
                raise ConfigError(f"validator {name!r} of type regex needs a pattern")
            spec.pattern = _compile(cfg["pattern"], f"validator {name!r}")
        elif vtype == "enum" and not spec.values:
            raise ConfigError(f"validator {name!r} of type enum needs values")
        elif vtype == "length" and spec.min is None and spec.max is None:
            raise ConfigError(f"validator {name!r} of type length needs min and/or max")
        elif vtype == "derived" and not spec.rule:
            raise ConfigError(f"validator {name!r} of type derived needs a rule")

        validators[name] = spec

    cd = data.get("cross_document") or {}
    cross = CrossDocumentSpec(
        similarity_threshold=float(cd.get("similarity_threshold", 0.85)),
        source_precedence=list(cd.get("source_precedence") or []),
        severity=cd.get("severity", "warning"),
        message=cd.get("message", "Field value differs across source documents"),
    )

    return ValidationRules(validators=validators, cross_document=cross)


def load_field_dictionary(
    path: Optional[Path] = None,
    rules: Optional[ValidationRules] = None,
) -> FieldDictionary:
    """Load the field dictionary, resolving defaults and checking references.

    `rules` is optional but recommended: passing it lets the loader verify that
    every validator a field names actually exists.
    """
    path = path or CONFIG_DIR / "field_dictionary.yaml"
    data = _read_yaml(path)

    defaults = data.get("defaults") or {}
    raw_fields = data.get("fields") or {}
    if not raw_fields:
        raise ConfigError(f"{path.name} defines no fields")

    d_search = defaults.get("search") or {}
    d_conf = defaults.get("confidence") or {}
    d_label = defaults.get("label_match") or {}

    fields: dict[str, FieldSpec] = {}
    for key, cfg in raw_fields.items():
        if not isinstance(cfg, dict):
            raise ConfigError(f"field {key!r} must be a mapping")

        unknown = set(cfg) - _FIELD_KEYS
        if unknown:
            raise ConfigError(
                f"field {key!r} has unknown key(s): {sorted(unknown)}; "
                f"allowed: {sorted(_FIELD_KEYS)}"
            )

        # -- search
        search_cfg = {**d_search, **(cfg.get("search") or {})}
        directions = list(search_cfg.get("directions") or ["right", "below"])
        bad_dirs = set(directions) - KNOWN_SEARCH_DIRECTIONS
        if bad_dirs:
            raise ConfigError(f"field {key!r}: unknown search direction(s) {sorted(bad_dirs)}")
        search = SearchSpec(
            directions=directions,
            max_distance=float(search_cfg.get("max_distance", 12.0)),
            same_line_bonus=float(search_cfg.get("same_line_bonus", 0.25)),
        )

        # -- confidence
        conf_cfg = {**d_conf, **(cfg.get("confidence") or {})}
        weights = {**(d_conf.get("weights") or {}), **((cfg.get("confidence") or {}).get("weights") or {})}
        if not weights:
            raise ConfigError(f"field {key!r}: no confidence weights configured")
        total = sum(float(v) for v in weights.values())
        if abs(total - 1.0) > 1e-6:
            raise ConfigError(
                f"field {key!r}: confidence weights sum to {total:.3f}, expected 1.0 "
                f"({weights})"
            )
        confidence = ConfidenceSpec(
            weights={k: float(v) for k, v in weights.items()},
            min_accept=float(conf_cfg.get("min_accept", 0.6)),
        )

        # -- label matching
        label_cfg = {**d_label, **(cfg.get("label_match") or {})}
        label_match = LabelMatchSpec(
            min_similarity=float(label_cfg.get("min_similarity", 0.82)),
            max_label_chars=int(label_cfg.get("max_label_chars", 60)),
        )

        # -- normalizers
        normalization = list(cfg.get("normalization") or [])
        bad_ops = [op for op in normalization if op not in KNOWN_NORMALIZERS]
        if bad_ops:
            raise ConfigError(
                f"field {key!r}: unknown normalization op(s) {bad_ops}; "
                f"allowed: {sorted(KNOWN_NORMALIZERS)}"
            )

        # -- validators
        validator_names = list(cfg.get("validators") or [])
        if rules is not None:
            missing = [v for v in validator_names if v not in rules.validators]
            if missing:
                raise ConfigError(
                    f"field {key!r} references undefined validator(s) {missing}; "
                    f"define them in validation_rules.yaml"
                )

        labels = [str(l) for l in (cfg.get("labels") or [])]
        raw_patterns = [str(p) for p in (cfg.get("patterns") or [])]
        patterns = [_compile(p, f"field {key!r}") for p in raw_patterns]
        scan_patterns = [_boundary_wrap(p, f"field {key!r}") for p in raw_patterns]
        if not labels and not patterns:
            raise ConfigError(
                f"field {key!r} has neither labels nor patterns, so nothing could "
                f"ever match it"
            )

        fields[key] = FieldSpec(
            key=key,
            description=cfg.get("description", ""),
            labels=labels,
            patterns=patterns,
            normalization=normalization,
            validators=validator_names,
            value_type=cfg.get("value_type", "text"),
            search=search,
            confidence=confidence,
            label_match=label_match,
            expected_documents=list(cfg.get("expected_documents") or []),
            aliases=[
                (canonical, _compile(pat, f"field {key!r} alias {canonical!r}"))
                for canonical, pats in (cfg.get("aliases") or {}).items()
                for pat in (pats if isinstance(pats, list) else [pats])
            ],
            derive_from=cfg.get("derive_from"),
            exclude_patterns=[
                _compile(p, f"field {key!r} exclude_pattern")
                for p in (cfg.get("exclude_patterns") or [])
            ],
            priority=int(cfg.get("priority", defaults.get("priority", 50))),
            required=bool(cfg.get("required", defaults.get("required", False))),
            cross_document_consistency=bool(
                cfg.get("cross_document_consistency", defaults.get("cross_document_consistency", False))
            ),
            default=cfg.get("default"),
            scan_patterns=scan_patterns,
        )

    return FieldDictionary(fields=fields, version=int(data.get("version", 1)))


def load_config(config_dir: Optional[Path] = None) -> tuple[FieldDictionary, ValidationRules]:
    """Load both config files together, cross-checking references."""
    base = config_dir or CONFIG_DIR
    rules = load_validation_rules(base / "validation_rules.yaml")
    dictionary = load_field_dictionary(base / "field_dictionary.yaml", rules=rules)
    return dictionary, rules
