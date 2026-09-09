"""
V2 Pipeline
===========
End-to-end, fully local:

    documents -> load/OCR -> layout -> semantic matching -> normalize
              -> validate -> canonical JSON -> Excel -> verify -> report

Every stage is importable on its own; this module is the wiring plus a CLI.

    python -m app.services.extraction_pipeline.pipeline uploads/861b7dc254 --out outputs/run
    python -m app.services.extraction_pipeline.pipeline --cache outputs/ocr_dump/document_set.json --out outputs/run
    python -m app.services.extraction_pipeline.pipeline uploads/861b7dc254 --template form.xlsx --sheet Sheet1
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Iterable, Optional

from .config_loader import FieldDictionary, ValidationRules, load_config
from .ingest.document_loader import IMAGE_SUFFIXES, _default_dpi_for, load_documents
from .models import RENDER_DPI, DocumentSet, ExtractionResult
from .ingest.ocr_engine import OCREngine, parse_ocr_tune
from .extract.semantic_engine import DocumentClassifier, SemanticEngine

logger = logging.getLogger(__name__)

SUPPORTED = {".pdf", ".docx"} | IMAGE_SUFFIXES


def collect_inputs(paths: Iterable[str | Path]) -> list[Path]:
    out: list[Path] = []
    for raw in paths:
        p = Path(raw)
        if p.is_dir():
            out.extend(sorted(f for f in p.iterdir() if f.suffix.lower() in SUPPORTED))
        elif p.suffix.lower() in SUPPORTED:
            out.append(p)
    return out


def extract_from_document_set(
    doc_set: DocumentSet,
    dictionary: Optional[FieldDictionary] = None,
    rules: Optional[ValidationRules] = None,
) -> ExtractionResult:
    """Run semantic extraction over already-loaded documents."""
    if dictionary is None or rules is None:
        dictionary, rules = load_config()
    engine = SemanticEngine(dictionary, rules, DocumentClassifier())
    t0 = time.perf_counter()
    result = engine.extract(doc_set)
    result.duration_s = time.perf_counter() - t0
    return result


def process_documents(
    paths: Iterable[str | Path],
    engine: Optional[OCREngine] = None,
    force_ocr: bool = False,
) -> tuple[DocumentSet, ExtractionResult]:
    """Load documents from disk and extract vendor fields from them."""
    files = collect_inputs(paths)
    if not files:
        raise ValueError(f"No supported documents found in {list(paths)}")

    doc_set = load_documents(files, engine=engine or OCREngine(), force_ocr=force_ocr)
    return doc_set, extract_from_document_set(doc_set)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _print_result(result: ExtractionResult) -> None:
    print("\n" + "=" * 96)
    print("EXTRACTED FIELDS")
    print("=" * 96)
    print(f"  {'field':<20} {'value':<34} {'conf':>5} {'status':<9} {'source':<22} label")
    print(f"  {'-'*20} {'-'*34} {'-'*5} {'-'*9} {'-'*22} {'-'*18}")
    for key, r in result.fields.items():
        value = (r.value or "")[:33]
        src = (r.source_document or "")[:21]
        label = (r.matched_label or "")[:18]
        print(f"  {key:<20} {value:<34} {r.confidence:>5.2f} {r.validation_status:<9} {src:<22} {label}")

    print("\n" + "=" * 96)
    print("DOCUMENTS")
    print("=" * 96)
    for d in result.documents:
        print(
            f"  {d['document'][:38]:<40} -> {d['doc_type']:<20} "
            f"score={d['classification_score']:<5} fields={d['fields_found']}"
        )

    if result.needs_review:
        print("\n" + "=" * 96)
        print("NEEDS REVIEW")
        print("=" * 96)
        for item in result.needs_review:
            detail = f"  [{item['detail'][:70]}]" if item.get("detail") else ""
            print(f"  {item['severity']:<8} {item['field']:<20} {item['reason']:<26} "
                  f"conf={item['confidence']:.2f}{detail}")

    s = result.summary()
    print("\n" + "=" * 96)
    print("SUMMARY")
    print("=" * 96)
    print(f"  fields filled : {s['filled']}/{s['total_fields']}  ({s['fill_rate']}%)")
    print(f"  valid         : {s['valid']}")
    print(f"  needs review  : {s['needs_review']}")
    print(f"  extraction    : {result.duration_s:.2f}s")


def main() -> int:
    parser = argparse.ArgumentParser(description="V2 local vendor extraction pipeline")
    parser.add_argument("inputs", nargs="*", help="Files and/or directories")
    parser.add_argument("--cache", help="Reuse a saved document_set.json instead of re-running OCR")
    parser.add_argument("--out", default="outputs/run", help="Output directory")
    parser.add_argument("--models", choices=["small", "medium", "tiny"], default="small")
    parser.add_argument(
        "--backend", choices=["paddleocr", "rapidocr", "rapidocr_openvino"], default=None,
        help="OCR backend. Defaults to $OCR_BACKEND, else rapidocr_openvino (the "
             "active engine as of 2026-09-09, ~2.3x faster than plain rapidocr on "
             "this machine's Intel CPU -- UNVALIDATED on AMD/ARM64, see "
             "ocr_engine.py's OCR_BACKEND comment; paddleocr is the preserved "
             "fallback, currently disabled -- see its PRESERVED FALLBACK banner).",
    )
    parser.add_argument(
        "--ocr-tune", action="append", metavar="NAME=VALUE",
        help="Tune a RapidOCR knob, repeatable. e.g. --ocr-tune det_box_thresh=0.3 "
             "--ocr-tune max_side_len=4000. Applies to both rapidocr and "
             "rapidocr_openvino (shared RapidOCRTuning); ignored by paddleocr. "
             "Composes with any OCR_RAPID_* environment variables.",
    )
    parser.add_argument(
        "--dpi", type=int, default=None,
        help=f"Page render DPI. Defaults to whichever DPI matches the active "
             f"OCR backend ({RENDER_DPI} for paddleocr, RAPID_RENDER_DPI for "
             f"rapidocr). Raising this must be paired with a matching "
             f"--ocr-tune max_side_len or RapidOCR downscales it straight back.",
    )
    parser.add_argument("--force-ocr", action="store_true")
    parser.add_argument("--template", help="Excel template to fill")
    parser.add_argument("--sheet", help="Sheet name inside the template")
    parser.add_argument("--mapping", default="vendor_creation_v1", help="Excel mapping config name")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    if not args.inputs and not args.cache:
        parser.error("provide input paths or --cache")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    if args.cache:
        print(f"Loading cached document set: {args.cache}")
        doc_set = DocumentSet.load_json(args.cache)
    else:
        files = collect_inputs(args.inputs)
        if not files:
            raise SystemExit(f"No supported documents found in: {args.inputs}")
        print(f"Loading {len(files)} document(s)...")
        for f in files:
            print(f"  - {f.name}")
        try:
            rapid_tuning = parse_ocr_tune(args.ocr_tune)
        except ValueError as exc:
            parser.error(str(exc))

        ocr = OCREngine(
            det_model=f"PP-OCRv6_{args.models}_det",
            rec_model=f"PP-OCRv6_{args.models}_rec",
            backend=args.backend,
            rapid_tuning=rapid_tuning,
        )
        print(f"  OCR backend: {ocr.backend}"
              + ("   [PRESERVED FALLBACK -- currently disabled, see "
                 "ocr_engine.py]" if ocr.backend == "paddleocr" else ""))

        if args.dpi is None:
            args.dpi = _default_dpi_for(ocr)

        if ocr.backend in ("rapidocr", "rapidocr_openvino"):
            tuning = ocr._rapid_tuning
            print(f"    max_side_len={tuning.max_side_len} "
                  f"text_score={tuning.text_score} use_cls={tuning.use_cls} "
                  f"det_box_thresh={tuning.det_box_thresh} "
                  f"intra_op={tuning.intra_op_num_threads}"
                  + (f" det_model={tuning.det_model_path}" if ocr.backend == "rapidocr_openvino" else ""))
            # Raising DPI without raising max_side_len in lockstep is the
            # classic way to pay 2x the render cost and measure no gain --
            # RapidOCR simply downscales the extra pixels away again. Applies
            # to rapidocr_openvino too -- it shares this exact tuning object,
            # even though max_side_len itself isn't a kwarg OpenVINO's older
            # parse_parameters.py reads (see to_rapidocr_openvino_kwargs()).
            needed = int(args.dpi * 11.69) + 1   # A4 long edge, in pixels
            if tuning.max_side_len < needed:
                print(f"    WARNING: at {args.dpi} DPI an A4 page is ~{needed}px tall, "
                      f"but max_side_len={tuning.max_side_len} will downscale it. "
                      f"Pass --ocr-tune max_side_len={needed} to actually use the "
                      f"extra resolution.")
        elif args.ocr_tune:
            print("    NOTE: --ocr-tune is ignored by the paddleocr backend.")

        doc_set = load_documents(files, engine=ocr, force_ocr=args.force_ocr, dpi=args.dpi)
        doc_set.save_json(out_dir / "document_set.json")

    print(f"  {len(doc_set)} documents, {len(doc_set.spans)} spans "
          f"({time.perf_counter() - t0:.1f}s)")

    # Phase 0 instrumentation (plan.md 0.1): break the load stage down further.
    # Populated only for freshly-loaded PDFs; a --cache run has no metadata to
    # sum, and this prints nothing in that case.
    pdf_open_s = sum(d.metadata.get("pdf_open_s", 0.0) for d in doc_set)
    render_s = sum(d.metadata.get("render_s", 0.0) for d in doc_set)
    if pdf_open_s or render_s:
        # render_s is nested inside doc_set.duration_s (it happens within each
        # page's own timer); pdf_open_s is not (it happens before the
        # per-page timers start), so only render_s is subtracted here.
        ocr_and_text_layer_s = doc_set.duration_s - render_s
        print(f"    pdf_open={pdf_open_s:.2f}s  render={render_s:.2f}s  "
              f"ocr_and_text_layer={ocr_and_text_layer_s:.2f}s")

    result = extract_from_document_set(doc_set)
    _print_result(result)

    result.save_json(out_dir / "extraction.json")
    (out_dir / "result.json").write_text(
        json.dumps(result.canonical(), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\n  -> {out_dir / 'extraction.json'}  (full metadata)")
    print(f"  -> {out_dir / 'result.json'}      (canonical vendor JSON)")

    # Optional Excel fill + verification (Steps 8-9).
    if args.template:
        from .excel.excel_mapper import ExcelMapper
        from .excel.verifier import verify_excel

        excel_t0 = time.perf_counter()
        mapper = ExcelMapper.load(args.mapping)
        xlsx_out = out_dir / "vendor_filled.xlsx"
        written = mapper.fill(result.canonical(), args.template, str(xlsx_out), sheet_name=args.sheet)
        fill_s = time.perf_counter() - excel_t0
        print(f"\n  Excel: wrote {written} field(s) -> {xlsx_out}  ({fill_s:.2f}s)")

        verify_t0 = time.perf_counter()
        report = verify_excel(
            result.canonical(), str(xlsx_out), mapper,
            sheet_name=args.sheet, report_path=str(out_dir / "verification_report.json"),
        )
        verify_s = time.perf_counter() - verify_t0
        passed = sum(1 for e in report if e["status"] == "PASS")
        print(f"  Verification: {passed}/{len(report)} PASS "
              f"-> {out_dir / 'verification_report.json'}  ({verify_s:.2f}s)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
