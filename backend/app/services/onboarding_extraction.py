"""Orchestration for the customer-onboarding extraction endpoint.

Kept thin on purpose, mirroring services/extraction.py: the OCR / PDF
rasterization / document classification / field matching / validation work
all already exists in extraction_pipeline -- the same shared engine this
backend's other endpoints (/extract, /process-vendor) already use for their
own Excel-filling feature. This module only handles the upload bookkeeping
and calls onboarding_mapper to reshape the result into the customer-onboarding
form's fixed schema.
"""

from __future__ import annotations

import json
import logging
import shutil
import time
import traceback
from pathlib import Path

from fastapi import UploadFile

from . import run_state
from .onboarding_mapper import to_onboarding_schema
from ..config.settings import OUTPUT_DIR, UPLOAD_DIR

logger = logging.getLogger(__name__)


class OnboardingExtractionError(Exception):
    """A failure worth reporting to the caller, with enough context to act on."""

    def __init__(self, message: str, detail: str = ""):
        super().__init__(message)
        self.message = message
        self.detail = detail


def _save_upload(upload: UploadFile, dest_dir: Path) -> Path:
    dest = dest_dir / Path(upload.filename).name
    with dest.open("wb") as f:
        shutil.copyfileobj(upload.file, f)
    return dest


def process(documents: list[UploadFile]) -> dict:
    """Run OCR + field extraction over `documents` and return the onboarding
    JSON schema. Uploads and intermediate results are persisted under a run id
    (same convention as /extract) so a submission can be audited later, but
    the run id itself is not part of the returned schema.
    """
    # Imported lazily so importing this module (e.g. for a unit test that only
    # exercises onboarding_mapper) never pays the OCR engine's import cost.
    from .extraction_pipeline.ingest.document_loader import load_documents
    from .extraction_pipeline.ingest.ocr_engine import OCREngine
    from .extraction_pipeline.pipeline import extract_from_document_set

    started_at = time.perf_counter()
    run_id = run_state.new_run_id()
    upload_dir = UPLOAD_DIR / run_id
    upload_dir.mkdir(parents=True, exist_ok=True)

    saved = [_save_upload(doc, upload_dir) for doc in documents if doc.filename]
    if not saved:
        raise OnboardingExtractionError(
            "No documents were uploaded.",
            detail="Attach at least one PDF or image (cancelled cheque, GST "
                   "certificate, Udyam certificate, PAN card, etc.).",
        )

    t0 = time.perf_counter()
    try:
        doc_set = load_documents(saved, engine=OCREngine())
    except Exception:
        logger.exception("onboarding extraction: document loading failed for run %s", run_id)
        raise OnboardingExtractionError(
            "Failed while reading the uploaded documents.",
            detail=traceback.format_exc(limit=6),
        )

    if not doc_set.documents:
        raise OnboardingExtractionError(
            "None of the uploaded files could be read.",
            detail="Supported types are PDF, PNG, JPG, TIFF, BMP and WEBP.",
        )

    load_seconds = time.perf_counter() - t0
    result = extract_from_document_set(doc_set)
    onboarding = to_onboarding_schema(result)
    # Same shape /extract returns, so CustomerReviewPage can show the timing
    # pill the vendor VendorComparePage already does.
    onboarding["timings"] = {
        "load": round(load_seconds, 1),
        "extract": round(result.duration_s, 2),
        "total": round(time.perf_counter() - started_at, 1),
    }

    run_dir = OUTPUT_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    result.save_json(run_dir / "extraction.json")
    (run_dir / "onboarding_result.json").write_text(
        json.dumps(onboarding, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    logger.info(
        "onboarding extraction run %s: %d document(s) in %.1fs -- %d field(s) need review",
        run_id, len(saved), time.perf_counter() - t0, len(onboarding["fields_needing_review"]),
    )
    return onboarding
