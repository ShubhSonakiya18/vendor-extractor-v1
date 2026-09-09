"""
V2 Semantic Engine
==================
Ties the pieces together for a whole document set:

  classify each document  ->  match every field in every document
  ->  merge candidates across documents  ->  validate  ->  canonical result

Merging is where multi-document handling actually happens. The same field can
be found in several places with different confidence; the engine keeps the best
one, records the rest as alternatives, and reports whether the documents agreed.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import yaml

from ..config_loader import CONFIG_DIR, FieldDictionary, ValidationRules, derive_value
from .field_matcher import KIND_RANK, Candidate, FieldMatcher
from ..ingest.layout_engine import DocumentLayout
from ..models import Document, DocumentSet, ExtractionResult, FieldResult
from .validator import Validator

# Derived values are computed from another field, not observed on a document.
# Kept deliberately low so they never outrank an extracted value and always
# read as second-class in the report.
DERIVED_VALUE_CONFIDENCE = 0.40

# A Udyam certificate's "Type of Enterprise" table (column header "Enterprise
# Type") sits under "Name of Enterprise"; the classification cell -- the lone
# word "Small" / "Micro" / "Medium" -- fuzzy-matches close enough to be
# generated as a vendor_name candidate. It is never a company name, so it must
# not be compared as one in the cross-document consistency check (step 4) --
# otherwise "GST cert says <real name>, Udyam says SMALL" reads as a document
# disagreement. onboarding_mapper._is_caption_leak rejects the same values on
# the value-selection side.
_JUNK_VENDOR_NAMES = {
    "micro", "small", "medium",
    "micro enterprise", "small enterprise", "medium enterprise",
    "micro enterprises", "small enterprises", "medium enterprises",
}


def _looks_like_real_value(field_key: str, value: str) -> bool:
    if field_key == "vendor_name":
        return " ".join((value or "").split()).casefold() not in _JUNK_VENDOR_NAMES
    return True


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DOCUMENT CLASSIFICATION
# ---------------------------------------------------------------------------

class DocumentClassifier:
    """Scores each document against config/document_profiles.yaml."""

    def __init__(self, path: Optional[Path] = None):
        path = path or CONFIG_DIR / "document_profiles.yaml"
        data = yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else {}
        self.profiles: dict = data.get("profiles") or {}
        self.content_weight = float(data.get("content_weight", 2.0))
        self.filename_weight = float(data.get("filename_weight", 1.0))
        self.min_score = float(data.get("min_score", 1.0))

    def classify(self, document: Document) -> tuple[str, float]:
        name = document.name.lower()
        # Only the first pages are needed to tell what a document is, and
        # scanning a 50-page attachment in full would be wasted work.
        text = "\n".join(p.text for p in document.pages[:3]).lower()

        best, best_score = "other", 0.0
        for profile, cfg in self.profiles.items():
            score = 0.0
            for kw in cfg.get("filename_keywords") or []:
                if kw.lower() in name:
                    score += self.filename_weight
            for kw in cfg.get("content_keywords") or []:
                if kw.lower() in text:
                    score += self.content_weight
            if score > best_score:
                best, best_score = profile, score

        return (best, best_score) if best_score >= self.min_score else ("other", best_score)


# ---------------------------------------------------------------------------
# ENGINE
# ---------------------------------------------------------------------------

class SemanticEngine:
    def __init__(
        self,
        dictionary: FieldDictionary,
        rules: ValidationRules,
        classifier: Optional[DocumentClassifier] = None,
    ):
        self.dictionary = dictionary
        self.rules = rules
        self.classifier = classifier or DocumentClassifier()
        self.validator = Validator(rules)

        all_labels = {
            lbl.lower().strip()
            for spec in dictionary
            for lbl in spec.labels
        }
        self.matcher = FieldMatcher(list(dictionary), all_labels)

    # -- main ---------------------------------------------------------------

    def extract(self, doc_set: DocumentSet) -> ExtractionResult:
        result = ExtractionResult()

        # 1. classify + match, per document
        per_document: list[tuple[str, str, dict[str, list[Candidate]]]] = []
        for document in doc_set:
            doc_type, score = self.classifier.classify(document)
            layout = DocumentLayout(document)
            candidates = self.matcher.match_document(layout, doc_type)
            per_document.append((document.name, doc_type, candidates))
            result.documents.append(
                {
                    "document": document.name,
                    "doc_type": doc_type,
                    "classification_score": round(score, 2),
                    "pages": len(document.pages),
                    "spans": len(document.spans),
                    "fields_found": len(candidates),
                    "extraction_methods": document.metadata.get("extraction_methods", []),
                }
            )
            logger.info("%s -> %s (%d fields)", document.name, doc_type, len(candidates))

        # 2. merge across documents
        pooled: dict[str, list[Candidate]] = {}
        doc_types: dict[str, str] = {}
        for doc_name, doc_type, candidates in per_document:
            doc_types[doc_name] = doc_type
            for key, items in candidates.items():
                pooled.setdefault(key, []).extend(items)

        precedence = self.rules.cross_document.source_precedence

        for spec in self.dictionary:
            items = sorted(pooled.get(spec.key, []), key=lambda c: -c.score)
            # Drop candidates that can't be a real value for this field (a
            # Udyam "Enterprise Type" cell -- "Small" -- captured under
            # vendor_name). Only applied when it wouldn't leave the field with
            # nothing: if every candidate is junk, keep them so the field
            # still surfaces for review rather than silently vanishing.
            filtered = [c for c in items if _looks_like_real_value(spec.key, c.value)]
            if filtered:
                items = filtered
            field_result = FieldResult(key=spec.key)

            if items:
                items = self._apply_validation_bias(spec, items)
                best = self._choose(items, spec, doc_types, precedence)
                field_result.value = best.value
                field_result.confidence = best.score
                field_result.source_document = best.source_document
                field_result.source_page = best.page
                field_result.matched_label = best.matched_label
                field_result.match_kind = best.match_kind
                field_result.notes = list(best.notes)
                field_result.alternatives = [
                    c.to_dict() for c in items[:6] if c is not best
                ][:5]
            elif spec.default:
                field_result.value = spec.default
                field_result.confidence = 1.0
                field_result.source_document = "config_default"
                field_result.notes.append("filled_from_config_default")

            result.fields[spec.key] = field_result

        # 2a. segment a combined address line.
        #
        # A GST REG-06 certificate (and a Udyam cert / cheque) prints the whole
        # address on one line with no City/State/PIN caption, so the matcher
        # lands it all in address_1 and leaves city/state/pin_code empty. Split
        # that blob deterministically (PIN-directory lookup + state list +
        # district-aware city pick) and backfill only the pieces still missing.
        # Anything filled this way is flagged for review -- it was inferred from
        # one run-on string, not read off a labelled field.
        self._resolve_combined_address(result)

        # 2b. derive fields that nothing on the documents supplied.
        #
        # Runs after every field is resolved, because a derivation reads
        # another field's final value. The result is INFERRED, never read off a
        # page, so it is flagged for review and its provenance says so -- a
        # human signing the form should be able to see that the value was
        # computed rather than extracted.
        for spec in self.dictionary:
            field_result = result.fields[spec.key]
            if field_result.is_present or not spec.derive_from:
                continue
            source_key = spec.derive_from.get("field")
            rule = spec.derive_from.get("rule")
            source = result.fields.get(source_key)
            if not source or not source.value:
                continue
            derived = derive_value(rule, source.value)
            if not derived:
                continue
            field_result.value = derived
            # Never inherit the source field's confidence: this value was not
            # observed, and it must not outrank things that were.
            field_result.confidence = DERIVED_VALUE_CONFIDENCE
            field_result.source_document = f"derived_from_{source_key}"
            field_result.notes.append(f"derived_from_{source_key}_via_{rule}")
            # A warning, not an error: deriving was the configured intent, and
            # the point of the flag is to tell a human the value was computed
            # rather than read -- not to report a failure.
            self._flag(result, spec.key, "value_derived_not_printed",
                       DERIVED_VALUE_CONFIDENCE,
                       detail=f"computed from {source_key}; not printed on any document",
                       severity="warning")

        # 3. validate (after every value is known, so derived rules can see them)
        values = {k: (r.value or "") for k, r in result.fields.items()}
        for spec in self.dictionary:
            field_result = result.fields[spec.key]

            if not field_result.is_present:
                field_result.validation_status = "missing"
                if spec.required:
                    self._flag(result, spec.key, "missing_required_field", field_result.confidence)
                continue

            findings = self.validator.check(spec.validators, field_result.value or "", values)
            field_result.validation_status = self.validator.status_of(findings)
            field_result.validation_messages = [
                f"{f.validator}: {f.message}" for f in findings if not f.ok
            ]

            if field_result.validation_status == "invalid":
                # A configured default is a better answer than a value already
                # known to be wrong -- country is "India" far more reliably
                # than whatever sat beside the word "Country" on page 4.
                if spec.default:
                    field_result.notes.append(
                        f"replaced_invalid_value_{field_result.value!r}_with_config_default"
                    )
                    field_result.value = spec.default
                    field_result.validation_status = "valid"
                    field_result.source_document = "config_default"
                    field_result.source_page = None
                    field_result.matched_label = None
                    field_result.validation_messages = []
                    values[spec.key] = spec.default
                    continue
                self._flag(result, spec.key, "failed_validation", field_result.confidence,
                           detail="; ".join(field_result.validation_messages))
            elif field_result.validation_status == "warning":
                self._flag(result, spec.key, "validation_warning", field_result.confidence,
                           detail="; ".join(field_result.validation_messages), severity="warning")

            if field_result.confidence < spec.confidence.min_accept:
                self._flag(result, spec.key, "low_confidence", field_result.confidence)

        # 4. cross-document consistency
        for spec in self.dictionary:
            if not spec.cross_document_consistency:
                continue
            field_result = result.fields[spec.key]
            best_per_doc: dict[str, str] = {}
            # Only compare candidates the engine would actually have trusted.
            # Including rejects turns every low-scoring stray span into a
            # "documents disagree" warning and buries the real conflicts.
            trusted = [
                c for c in sorted(pooled.get(spec.key, []), key=lambda c: -c.score)
                if c.score >= spec.confidence.min_accept
                and _looks_like_real_value(spec.key, c.value)
            ]
            for cand in trusted:
                best_per_doc.setdefault(cand.source_document, cand.value)

            status, disagreements = self.validator.compare_across_documents(best_per_doc)
            field_result.consistency = status
            if status == "inconsistent":
                self._flag(
                    result, spec.key, "cross_document_mismatch", field_result.confidence,
                    detail="; ".join(disagreements), severity="warning",
                )

        # 4b. a PIN/state disagreement implicates BOTH fields, not just the
        # one the validator sits on -- the state validator has already flagged
        # `state`, so mirror that onto `pin_code` so a reviewer re-checks the
        # PIN too. Detected by the pin_matches_state finding on `state`.
        state_result = result.fields.get("state")
        pin_result = result.fields.get("pin_code")
        if (
            state_result and pin_result and pin_result.value
            and any("pin_matches_state" in m for m in state_result.validation_messages)
            and not any(
                e["field"] == "pin_code" and e["reason"] == "pin_state_mismatch"
                for e in result.needs_review
            )
        ):
            self._flag(
                result, "pin_code", "pin_state_mismatch", pin_result.confidence,
                detail=f"PIN {pin_result.value} belongs to a different state than "
                       f"{state_result.value!r}", severity="warning",
            )

        return result

    # -- helpers ------------------------------------------------------------

    def _resolve_combined_address(self, result: ExtractionResult) -> None:
        """Backfill city / state / pin_code (and tidy address_1) from a
        run-on address string. No-op unless address_1 looks combined and at
        least one of the three target fields is still empty."""
        addr = result.fields.get("address_1")
        if addr is None or not addr.value:
            return

        city = result.fields.get("city")
        state = result.fields.get("state")
        pin = result.fields.get("pin_code")
        already = lambda fr: bool(fr and fr.value)
        if already(city) and already(state) and already(pin):
            return
        # A bare "1ST FLOOR" with no commas and no digits-run isn't a blob to
        # split -- leave it alone.
        if "," not in addr.value and not any(ch.isdigit() for ch in addr.value):
            return

        from .address_resolver import resolve_address_blob

        r = resolve_address_blob(addr.value)
        if not (r.city or r.state or r.pin_code):
            return

        if r.address_1 and r.address_1 != addr.value:
            addr.value = r.address_1
            addr.notes.append("address_1_trimmed_by_address_resolver")

        for key, fr, val in (
            ("city", city, r.city),
            ("state", state, r.state),
            ("pin_code", pin, r.pin_code),
        ):
            if val and not already(fr):
                target = fr if fr is not None else FieldResult(key=key)
                target.value = val
                target.confidence = max(target.confidence, 0.55)
                target.source_document = "address_resolver"
                target.notes.append(f"resolved_from_combined_address ({r.confidence})")
                result.fields[key] = target
                self._flag(
                    result, key, "value_resolved_from_combined_address", 0.55,
                    detail=f"split out of address_1; resolver confidence {r.confidence}",
                    severity="warning",
                )

    def _apply_validation_bias(self, spec, items: list[Candidate]) -> list[Candidate]:
        """Let the field's own validators influence which candidate wins.

        Validation used to run only after selection, which meant a value that
        could never be correct still won on layout evidence alone -- the state
        field picked up OCR debris ("Rvic") over the genuine "West Bengal"
        sitting a few points lower. Checking each candidate first turns those
        rules into selection signal, which is exactly the knowledge they encode.

        Derived rules are skipped here: they compare fields against each other
        and nothing is decided yet, so they have nothing to judge.
        """
        checkable = [
            name for name in spec.validators
            if self.rules.validators[name].type != "derived"
        ]
        if not checkable:
            return items

        for cand in items:
            findings = self.validator.check(checkable, cand.value, {})
            if any(not f.ok and f.severity == "error" for f in findings):
                cand.score *= 0.35
                cand.notes.append("failed_validation")
            elif any(not f.ok for f in findings):
                cand.score *= 0.75
                cand.notes.append("validation_warning")
            elif findings:
                cand.score = min(1.0, cand.score + 0.10)
                cand.notes.append("passed_validation")

        return sorted(items, key=lambda c: -c.score)

    def _choose(
        self,
        items: list[Candidate],
        spec,
        doc_types: dict[str, str],
        precedence: list[str],
    ) -> Candidate:
        """Pick the winning candidate.

        Score decides, except when candidates are effectively tied -- and they
        often are, because scores clamp at 1.0. Ties break on:

        1. The field's own `expected_documents`. This has to outrank the global
           precedence list: that list puts the Udyam certificate above the
           cheque (right for identity fields), but the Udyam certificate also
           prints bank details, and it was winning IFSC and account number away
           from the cancelled cheque that the field explicitly expects.
        2. The global source precedence, for everything with no such preference.
        3. How the value was found -- inline beats adjacent beats bare pattern.
        """
        # `expected_documents` is a HARD restriction, not a tie-break. It used
        # to apply only inside the 0.05 window below, which meant a
        # better-scoring candidate from the wrong document won outright and the
        # preference never ran.
        #
        # That is how production came to emit a mixed-source bank record: this
        # vendor's Udyam certificate prints a labelled "Bank of India" (clean
        # inline match, high score) while the cancelled cheque carries ICICI
        # only as a logo OCR'd as "AICICIBank" (no label, low score). The gap
        # exceeded 0.05, so Udyam won `bank_name` while the cheque still won
        # `ifsc` and `account_number` -- ICICI's account under Bank of India's
        # name, belonging to no real account, and not flagged for review.
        #
        # When a field names the documents it belongs on and at least one
        # candidate comes from one, candidates from anywhere else are not
        # contenders at any score.
        if spec.expected_documents:
            preferred = [
                c for c in items
                if doc_types.get(c.source_document, "other") in spec.expected_documents
            ]
            if preferred:
                items = preferred

        best = items[0]
        close = [c for c in items if best.score - c.score <= 0.05]
        if len(close) <= 1:
            return best

        def expected_rank(c: Candidate) -> int:
            dtype = doc_types.get(c.source_document, "other")
            if spec.expected_documents:
                return 0 if dtype in spec.expected_documents else 1
            return 0

        def global_rank(c: Candidate) -> int:
            dtype = doc_types.get(c.source_document, "other")
            return precedence.index(dtype) if dtype in precedence else len(precedence)

        close.sort(
            key=lambda c: (
                expected_rank(c),
                global_rank(c),
                KIND_RANK.get(c.match_kind, 9),
                -c.score,
            )
        )
        return close[0]

    def _flag(
        self,
        result: ExtractionResult,
        field_key: str,
        reason: str,
        confidence: float,
        detail: str = "",
        severity: str = "error",
    ) -> None:
        entry = {
            "field": field_key,
            "reason": reason,
            "confidence": round(confidence, 4),
            "severity": severity,
        }
        if detail:
            entry["detail"] = detail
        result.needs_review.append(entry)
