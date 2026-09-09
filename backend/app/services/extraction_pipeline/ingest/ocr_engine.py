"""
OCR Engine Wrapper (RapidOCR active / PaddleOCR preserved fallback)
====================================================================
The ONLY module the PRODUCTION PIPELINE imports paddleocr or
rapidocr_onnxruntime through. Everything downstream consumes backend-neutral
`TextSpan` objects, so the OCR engine is swappable without touching
classification, extraction, validation or Excel mapping. (app/eval/eval_sahi.py
imports paddleocr directly, by design -- it is a standalone detection-recall
experiment for a PaddleOCR-specific knob (text_det_limit_side_len) with no
RapidOCR equivalent, so it does not go through this abstraction.)

RapidOCR is the ACTIVE default (see OCR_BACKEND below). PaddleOCR's execution
path is preserved in this file, commented out and clearly marked, so it can
be restored if RapidOCR underperforms in deployment -- see the restore steps
at each "PRESERVED FALLBACK" banner. Targets the installed PaddleOCR 3.7 API
(`PaddleOCR.predict`); the 2.x style
`PaddleOCR(use_angle_cls=True).ocr(img, cls=True)` is NOT used -- those kwargs
no longer exist.

Model loading is expensive (seconds) and the models are stateless across
calls, so engines are cached per configuration and reused for the whole batch.
"""

from __future__ import annotations

import dataclasses
import logging
import os
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

from ..models import BBox, SpanSource, TextSpan

# backend/app/services/extraction_pipeline/ingest/ocr_engine.py -> backend/
_BACKEND_DIR = Path(__file__).resolve().parents[4]
_OCR_MODELS_DIR = _BACKEND_DIR / "config" / "ocr_models"

# OpenVINO IR (.xml/.bin) conversions of the same committed int8 weights above
# -- same detector/recognizer, same calibration, just run through OpenVINO's
# runtime instead of ONNX Runtime. Converted via `ovc` from the .onnx files
# (2026-09-09); see the OpenVINO backend section below for the benchmark this
# is based on. Not yet git-lfs-tracked (no *.xml/*.bin pattern in
# .gitattributes) -- add one before committing these for real.
_OCR_MODELS_DIR_OPENVINO = _OCR_MODELS_DIR

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# oneDNN WORKAROUND -- must run before paddlex is imported anywhere.
# ---------------------------------------------------------------------------
# PaddlePaddle 3.3.1 on CPU defaults to the oneDNN/MKL-DNN run mode, whose new
# PIR executor cannot convert some op attributes, and detection dies with:
#     NotImplementedError: (Unimplemented) ConvertPirAttribute2RuntimeAttribute
#     not support [pir::ArrayAttribute<pir::DoubleAttribute>]
#     (onednn_instruction.cc:118)
# Forcing the plain "paddle" run mode avoids that code path entirely. paddlex
# reads this flag at import time, so setting it later has no effect -- hence
# module scope. An explicit value from the environment always wins, so this
# can be re-enabled from outside once upstream fixes the executor.
os.environ.setdefault("PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT", "0")

# PaddleOCR is imported lazily so that importing models / document_loader
# (e.g. for a DOCX-only run, or in tests) does not pay the paddle import cost.
_ENGINE_CACHE: dict[tuple, Any] = {}

# Model selection is the dominant cost/accuracy lever on CPU. Measured on a
# 1592x723 cheque render, oneDNN disabled (see workaround above):
#
#   PP-OCRv6_medium   93.3s   mean conf 0.962   <- PaddleOCR's default
#   PP-OCRv6_small    10.7s   mean conf 0.948
#   PP-OCRv6_tiny      3.8s   mean conf 0.875   (hallucinated CJK glyphs)
#
# `small` is the default here: ~9x faster than the stock medium models for a
# marginal confidence cost, which is what makes multi-document batches
# practical. `tiny` is not recommended -- it invented characters that were not
# on the page, which is far more dangerous than a low score, because a
# confident wrong GSTIN silently passes validation. Override per call site.
DEFAULT_DET_MODEL = "PP-OCRv6_small_det"
DEFAULT_REC_MODEL = "PP-OCRv6_small_rec"

# PaddleOCR's own default is 10 threads. Benchmarked 1/2/4/8/10 on the fixed
# 3-document set (app/uploads/09e0ff7aca/): 4 threads was fastest at 104.4s
# load vs 133.9s at the PaddleOCR default -- ~22% faster, byte-identical
# canonical output at every thread count (plan.md task 1.3, 2026-08-27).
DEFAULT_CPU_THREADS = 4


# GPU support. Off by default -- this codebase was CPU-only when received
# (no `device` kwarg anywhere in this file). Controlled by an env var
# rather than a constructor param so it can be flipped without touching
# call sites: set OCR_ENGINE_DEVICE=gpu before starting the app. Falls
# back to CPU automatically if GPU init fails for any reason (no GPU
# present, paddlepaddle-gpu not installed, CUDA/cuBLAS runtime missing),
# so this is safe to leave enabled on a machine that turns out not to
# have a usable GPU.
_REQUESTED_DEVICE = os.environ.get("OCR_ENGINE_DEVICE", "cpu").strip().lower()


# OCR backend selection (plan.md task 2.1, superseded 2026-09-01, superseded
# again 2026-09-09). RapidOCR+OpenVINO is now the ACTIVE/DEFAULT engine.
# RapidOCR+ONNX Runtime and PaddleOCR both remain fully supported fallbacks
# -- PaddleOCR's execution path is PRESERVED (commented out below in `_load()`
# and `read_image()`), restorable via OCR_BACKEND=paddleocr plus uncommenting,
# see each "PRESERVED FALLBACK" banner for exact restore steps.
#
#   OCR_BACKEND=rapidocr_openvino   (default) -- RapidOCR on OpenVINO
#   OCR_BACKEND=rapidocr                      -- RapidOCR on ONNX Runtime
#   OCR_BACKEND=paddleocr                     -- PP-OCRv6 via PaddlePaddle, PRESERVED FALLBACK
#
# Why OpenVINO is the default (2026-09-09): the SAME committed int8
# detector/recognizer, converted to OpenVINO IR (.xml/.bin) via `ovc`, measured
# ~2.3x faster median latency than the ONNX Runtime path on this machine's
# Intel CPU -- 10 isolated-subprocess reps, 1.71s vs 3.93s median, zero overlap
# between the two distributions. Accuracy is IDENTICAL: scored against the same
# human-verified ground-truth file (app/eval/ground_truth/mb_control_systems.yaml)
# via app/eval/eval_extraction.py -- 19/24 correct, 1 wrong, 0 hallucinated, 0
# missed (95.8%), byte-identical on every GSTIN/PAN/IFSC/account-number field to
# both the ONNX Runtime and PaddleOCR paths. The one miss is the same
# pre-existing `nature_of_business` field-selection bug shared by every engine
# tested (Udyam certificate page 3's truncated "Manufactur" render), not an
# OCR-specific defect of any backend. The bank-account-consistency regression
# test (test_bank_record_consistency.py) also passes under this backend.
#
# CAVEAT, read before trusting this on different hardware: OpenVINO's fast
# paths are Intel-authored kernels tuned for Intel's own instruction-set
# extensions (AVX2/AVX-512/AMX). It is NOT validated on AMD or ARM64 (Apple
# Silicon, Qualcomm) -- third-party reports have it as SLOWER than ONNX
# Runtime on AMD Ryzen specifically, the opposite of this machine's result.
# Re-run this same isolated-subprocess latency+accuracy comparison on any
# non-Intel deployment target before trusting this default there; fall back
# to OCR_BACKEND=rapidocr (ONNX Runtime, no such caveat) if it doesn't hold.
#
# Requires the `rapidocr-openvino` and `openvino` packages (not yet in
# requirements.txt -- see that file's own comment on this).
#
# What is still NOT validated regardless of backend: the ground truth above
# covers ONE vendor's three documents. Multi-vendor divergence testing
# (plan.md 2.1) remains open before trusting any of these engines at scale.
_REQUESTED_BACKEND = os.environ.get("OCR_BACKEND", "rapidocr_openvino").strip().lower()

_VALID_BACKENDS = {"paddleocr", "rapidocr", "rapidocr_openvino"}
if _REQUESTED_BACKEND not in _VALID_BACKENDS:
    raise ValueError(
        f"OCR_BACKEND={_REQUESTED_BACKEND!r} is not recognised. "
        f"Valid values: {sorted(_VALID_BACKENDS)}"
    )


# ---------------------------------------------------------------------------
# RAPIDOCR TUNING
# ---------------------------------------------------------------------------
# Every kwarg name RapidOCR 1.4.4 actually reads, transcribed from
# `rapidocr_onnxruntime/utils/parse_parameters.py:init_args()`.
#
# This whitelist exists because RapidOCR's kwarg routing fails SILENTLY:
# `parse_kwargs()` routes by prefix (`det_*` stripped, `cls_*`/`rec_*` kept) and
# drops anything unrecognised into a Global dict that is never read. A typo, or
# a PaddleOCR-style name like `text_det_limit_side_len`, produces no error and
# no effect -- the tuning simply doesn't happen and the run looks normal.
# Validating against this set converts that into a loud failure, and also turns
# a future rapidocr-onnxruntime upgrade that renames a key into an exception
# rather than a quiet accuracy regression.
_RAPIDOCR_KNOWN_KWARGS = frozenset({
    # Global
    "text_score", "use_det", "use_cls", "use_rec", "print_verbose",
    "min_height", "width_height_ratio", "max_side_len", "min_side_len",
    "return_word_box", "intra_op_num_threads", "inter_op_num_threads",
    # Det
    "det_use_cuda", "det_use_dml", "det_model_path", "det_limit_side_len",
    "det_limit_type", "det_thresh", "det_box_thresh", "det_max_candidates",
    "det_unclip_ratio", "det_donot_use_dilation", "det_score_mode",
    # Cls
    "cls_use_cuda", "cls_use_dml", "cls_model_path", "cls_image_shape",
    "cls_batch_num", "cls_thresh", "cls_label_list",
    # Rec
    "rec_use_cuda", "rec_use_dml", "rec_model_path", "rec_keys_path",
    "rec_img_shape", "rec_batch_num",
})

# EXPERIMENTAL (2026-09-09): the kwarg surface rapidocr_openvino 1.3.25 (the
# version pip resolves alongside openvino-dev==2024.6.0) actually reads,
# transcribed from its own `rapidocr_openvino/utils/parse_parameters.py`.
# Older than rapidocr_onnxruntime 1.4.4's surface above and meaningfully
# different: no `intra_op_num_threads`/`inter_op_num_threads` (OpenVINO uses
# a single `inference_num_threads`, unprefixed so it is NOT det/cls/rec-
# routed), no `det_max_candidates`, no `*_use_cuda`/`*_use_dml`. Passing
# rapidocr_onnxruntime's kwarg names here would hit the exact silent-failure
# mode described above the ONNX whitelist -- hence a SEPARATE whitelist
# rather than reusing `_RAPIDOCR_KNOWN_KWARGS`.
_RAPIDOCR_OPENVINO_KNOWN_KWARGS = frozenset({
    # Global
    "text_score", "no_det", "no_cls", "no_rec", "print_verbose",
    "min_height", "width_height_ratio", "inference_num_threads",
    # Det
    "det_model_path", "det_limit_side_len", "det_limit_type", "det_thresh",
    "det_box_thresh", "det_unclip_ratio", "det_donot_use_dilation",
    "det_score_mode",
    # Cls
    "cls_model_path", "cls_image_shape", "cls_label_list", "cls_batch_num",
    "cls_thresh",
    # Rec
    "rec_model_path", "rec_keys_path", "rec_img_shape", "rec_batch_num",
})

_VALID_LIMIT_TYPES = {"min", "max"}
_VALID_SCORE_MODES = {"fast", "slow"}


def stock_fp32_paths() -> tuple[str, str]:
    """Absolute paths to RapidOCR's own bundled fp32 (det, rec) models.

    Resolved from the installed `rapidocr_onnxruntime` package rather than
    hardcoded, so it works regardless of venv location or OS. Used to opt
    back into fp32 without needing int8 uninstalled or int8's committed files
    removed -- see RapidOCRTuning's model-override comment.
    """
    import rapidocr_onnxruntime

    models_dir = Path(rapidocr_onnxruntime.__file__).resolve().parent / "models"
    det = models_dir / "ch_PP-OCRv4_det_infer.onnx"
    rec = models_dir / "ch_PP-OCRv4_rec_infer.onnx"
    for p in (det, rec):
        if not p.is_file():
            raise FileNotFoundError(
                f"Expected RapidOCR's stock model at {p}, but it is not there. "
                "The rapidocr-onnxruntime package layout may have changed -- "
                "check its installed version against what this function assumes."
            )
    return str(det), str(rec)


@dataclasses.dataclass(frozen=True)
class RapidOCRTuning:
    """RapidOCR knobs, in this project's vocabulary.

    Frozen so it is hashable and can live inside `OCREngine._cache_key` -- a
    tuning change must mint a new engine, not silently reuse one built with
    different settings.

    Fields left as `None` are filled in by `resolved()` from the owning
    `OCREngine`, so a RapidOCR run inherits this project's settings rather than
    RapidOCR's stock ones. Two of those inherited defaults are deliberate
    fairness fixes against the PaddleOCR path (see the class constants below).
    """

    # -- page geometry ------------------------------------------------------
    # RapidOCR's stock max_side_len is 2000, which silently downscales every
    # page whose long side exceeds it -- A4 at RENDER_DPI=200 is 1654x2338, so
    # ~15% of linear resolution is thrown away. Worse, `main.py` crops the
    # recognition inputs from that same downscaled image, so this costs
    # character legibility, not just detection.
    #
    # 3508 disables the resize up to 300 DPI on an A4 page (11.69in * 300 =
    # 3507px) -- comfortably above settings.RAPID_RENDER_DPI regardless of
    # which value that constant is currently set to, since a higher ceiling
    # never hurts. It only needs raising further if RAPID_RENDER_DPI (or a
    # page size beyond A4) is pushed past what this covers -- see that
    # constant's own comment for the lockstep math.
    max_side_len: int = 3508
    min_side_len: int = 30

    # -- filtering ----------------------------------------------------------
    # None -> inherit the engine's `drop_score` (0.30). RapidOCR's stock 0.5
    # deletes any recognition below it AND its bounding box, which is a second,
    # stricter filter stacked on top of ours -- it made `drop_score` dead below
    # 0.5 and filtered 67% harder than the PaddleOCR path. Aligning them makes
    # both backends filter identically and restores `drop_score` as the single
    # lever (which plan.md task 2.4 exists to tune).
    text_score: Optional[float] = None

    # None -> inherit the engine's `use_textline_orientation` (False). RapidOCR
    # runs a 180-degree textline classifier by default that our PaddleOCR path
    # does not, which is both an unequal-preprocessing gap and wasted time.
    use_cls: Optional[bool] = None

    # -- detection ----------------------------------------------------------
    # NOTE: det_limit_type stays "min". Under "max", RapidOCR DISCARDS
    # det_limit_side_len entirely and substitutes a hardcoded ladder capping at
    # 2000 (`ch_ppocr_det/text_detect.py:65-74`), which would silently undo the
    # max_side_len fix above. Under "min" the resize only ever upscales, so at
    # our page sizes it is a genuine no-op and detection runs at whatever
    # max_side_len hands it. Do not "fix" this to "max".
    det_limit_type: str = "min"
    det_limit_side_len: int = 736
    det_thresh: float = 0.3
    det_box_thresh: float = 0.5
    det_unclip_ratio: float = 1.6
    det_use_dilation: bool = True          # emitted NEGATED, see to_rapidocr_kwargs
    det_score_mode: str = "fast"

    # -- model overrides -----------------------------------------------------
    # SUPERSEDED 2026-09-07: int8-quantized weights are now the default,
    # committed under config/ocr_models/ (via git-lfs -- see .gitattributes).
    # Both were quantized from RapidOCR 1.4.4's own shipped PP-OCRv4 mobile
    # weights, calibrated on this project's real sample documents:
    #   - det: static QDQ int8 (calibrated on 9 real page renders through
    #     RapidOCR's own TextDetector.preprocess_op, so activation scales see
    #     genuine input, not synthetic data). 4.5MB -> 1.4MB.
    #   - rec: dynamic QDQ int8, weights-only, MatMul ops only (ONNX Runtime's
    #     CPU provider has no ConvInteger kernel, so a fully int8 QOperator
    #     rec model loads but cannot execute -- confirmed by hitting exactly
    #     that error). Static/activation-quantized rec was tried and
    #     REJECTED: recall collapsed 95%->45% (11/24 fields missed, 0 wrong)
    #     with only 78 calibration crops from 3 documents -- too narrow a
    #     calibration set for activation quantization specifically, not a
    #     problem with int8 itself. The weights-only dynamic variant has none
    #     of that risk and is what ships. 10.4MB -> 7.4MB.
    # Combined result vs stock fp32, 12 isolated reps each, real ground truth:
    # median latency 28.8% faster (11.75s vs 16.51s), 19/24 correct on both
    # (identical, the one miss is the pre-existing nature_of_business bug --
    # see field_dictionary.yaml), every identifier field byte-identical
    # across all 24 runs. See plan.md for the full writeup.
    #
    # Charset (rec_keys_path) is UNCHANGED -- quantization does not touch the
    # vocabulary, only the weights, so this stays None (RapidOCR's own
    # internal default) exactly as before.
    #
    # To run the original fp32 stock models instead (e.g. to compare, or if
    # int8 ever regresses on a different document set):
    #   - In code: RapidOCRTuning(det_model_path=None, rec_model_path=None) --
    #     to_rapidocr_kwargs() drops a None field entirely, so RapidOCR falls
    #     back to its own bundled fp32 default.
    #   - Via environment: from_env() treats a BLANK OCR_RAPID_*_MODEL_PATH as
    #     unset, which falls through to THIS class default (int8, not fp32) --
    #     a blank var does NOT get you fp32. Point it at the real stock files
    #     instead, resolved with stock_fp32_paths() below so nobody needs to
    #     know the venv's install path:
    #       python -c "from app.services.extraction_pipeline.ingest.ocr_engine \
    #         import stock_fp32_paths; d,r = stock_fp32_paths(); \
    #         print('OCR_RAPID_DET_MODEL_PATH=' + d); \
    #         print('OCR_RAPID_REC_MODEL_PATH=' + r)"
    det_model_path: Optional[str] = str(_OCR_MODELS_DIR / "rapidocr_det_int8.onnx")
    rec_model_path: Optional[str] = str(_OCR_MODELS_DIR / "rapidocr_rec_int8.onnx")
    rec_keys_path: Optional[str] = None

    # -- threading ----------------------------------------------------------
    # PORTABLE-BY-DEFAULT as of 2026-09-09 (see resolved() below): an
    # UNTOUCHED default (this field still exactly 8, meaning nobody explicitly
    # asked for a thread count) now resolves to "let the runtime decide" on
    # BOTH backends -- ORT's own "-1 = every core" sentinel for rapidocr,
    # `None` (kwarg omitted entirely) for rapidocr_openvino. Deliberately
    # symmetric: this project's own measured evidence (below) shows -1 is
    # WORSE than a pinned 8 on THIS machine's specific hybrid P-core/E-core
    # CPU -- but that finding does not generalize to other CPU topologies
    # (homogeneous-core servers, different hybrid layouts, other vendors
    # entirely), and shipping a number derived from one machine's measurement
    # as every deployment's default is exactly the non-portable pattern this
    # rework exists to avoid. An EXPLICIT override (via OCR_RAPID_THREADS,
    # `--ocr-tune`, or `RapidOCRTuning(intra_op_num_threads=N)`) is always
    # honored as given, on either backend -- nothing about configurability
    # changed, only what the untouched default resolves to. A deployment that
    # re-measures its own hardware and finds a specific pinned count is
    # genuinely better there should set OCR_RAPID_THREADS explicitly, not
    # rely on this class's own default.
    #
    # ---- ONNX Runtime (rapidocr): the measurement this default no longer assumes ----
    # SUPERSEDED 2026-09-01: 4 was chosen from an earlier pinned-vs-unleashed
    # sweep (see below) but was never checked against other pinned counts at
    # every DPI, so a thread x DPI interaction was invisible to it. A 12-rep,
    # 16-config (threads in {2,4,6,8} x DPI in {100,125,150,200}) isolated
    # sweep -- full min/median/mean/P90/max/stdev/CV per config, cold/warm
    # separated, identifier fields checked both for self-consistency across
    # a config's own 12 reps and agreement with a PaddleOCR reference -- found
    # threads=4 is NOT uniformly safe: at DPI=125 (the shipped default) it is
    # merely fine (CV=0.05), but threads=8 at DPI=125 is the single least
    # stable config in the entire grid (CV=0.40, one rep spiked to 70s vs a
    # 26s median). At DPI=100, however, threads=8 is both the fastest AND the
    # most stable config measured: 19.72s median, P90 19.90s (1% over median),
    # CV=0.01, zero identifier mismatches across all 12 reps. This is the
    # basis for also lowering RAPID_RENDER_DPI to 100 (see settings.py) --
    # the two changes are paired, not independent choices.
    #
    # Prior finding, still true in isolation, and the reason this was ever a
    # PINNED default rather than automatic: on THIS machine's hybrid
    # P-core/E-core CPU, -1 ("every core") forces threads across BOTH core
    # types simultaneously and is far slower/less consistent than any pinned
    # count tested (101.3s median / 110.1s spread vs 36.7s / 25.5s at
    # pinned-4, 8 runs each). RapidOCR/ONNX Runtime does not pin threads to
    # specific cores, so Windows' scheduler decides P-core vs E-core
    # placement per run, and a barrier-synchronised parallel op runs at the
    # speed of its slowest thread -- a run that lands on E-cores is gated by
    # them.
    #
    # This is a measurement of THIS machine's core topology, not a law -- a
    # homogeneous-core server should be re-measured before trusting 8 here.
    # As of 2026-09-09 that caveat is taken at face value: resolved() no
    # longer applies 8 automatically for an untouched default on this backend
    # either (see "PORTABLE-BY-DEFAULT" above) -- the -1-is-worse finding is
    # real ON THIS MACHINE but is not assumed to hold on whatever machine this
    # code actually deploys to. If this pipeline is deployed to hardware with
    # a similar hybrid-core topology and the same regression is confirmed
    # there, set OCR_RAPID_THREADS=8 (or whatever that hardware's own sweep
    # finds) explicitly for that deployment -- see Experiment 3 in the OCR
    # optimization report for how to run that sweep properly rather than
    # re-assuming this machine's number.
    #
    # ---- OpenVINO (rapidocr_openvino), 2026-09-09 -----------------------
    # UNLIKE the ONNX Runtime path above, no thread-count sweep was ever run
    # against rapidocr_openvino specifically -- the shipped default (8) was
    # carried over from the ONNX Runtime tuning unexamined (see the OCR
    # optimization report, Section 18C: "an assumption carried over, not a
    # value re-derived for OpenVINO"). OpenVINO's CPU plugin DOES manage its
    # own thread/stream configuration when left alone: confirmed by reading
    # rapidocr_openvino's own infer_engine.py -- inference_num_threads
    # defaults to -1 internally, and at -1 the code never calls
    # core.set_property("CPU", {"INFERENCE_NUM_THREADS": ...}) at all, leaving
    # OpenVINO's Core on its own automatic, hardware-aware configuration.
    # Given no evidence a manual pin beats that here, resolved() treats an
    # UNTOUCHED default (this field still exactly 8) as "auto" on the
    # OpenVINO backend -- swapped to None, which to_rapidocr_openvino_kwargs()
    # drops entirely from the emitted kwargs (omitting the kwarg is NOT
    # equivalent to passing a literal -1 or 8 -- it skips OpenVINO's
    # set_property call entirely, the genuinely-automatic path). The class
    # default stays 8, unchanged, so the ONNX Runtime path -- and anyone who
    # explicitly asks for 8 on purpose -- is completely unaffected; only the
    # OpenVINO backend's *unspecified* case changed. OCR_RAPID_THREADS,
    # --ocr-tune, or a direct RapidOCRTuning(intra_op_num_threads=N) still
    # overrides this on either backend, exactly as before -- see from_env()
    # below and resolved()'s docstring for the precise substitution rule.
    intra_op_num_threads: Optional[int] = 8
    # ORT defaults to sequential execution, so inter-op threads are inert;
    # pinning to 1 removes a variable rather than adding one. rapidocr_openvino
    # has no equivalent kwarg (not in _RAPIDOCR_OPENVINO_KNOWN_KWARGS), so this
    # is silently inert -- never emitted -- on the OpenVINO path already.
    inter_op_num_threads: Optional[int] = 1

    def __post_init__(self) -> None:
        if self.det_limit_type not in _VALID_LIMIT_TYPES:
            raise ValueError(
                f"det_limit_type={self.det_limit_type!r} invalid. "
                f"Valid: {sorted(_VALID_LIMIT_TYPES)}"
            )
        if self.det_score_mode not in _VALID_SCORE_MODES:
            raise ValueError(
                f"det_score_mode={self.det_score_mode!r} invalid. "
                f"Valid: {sorted(_VALID_SCORE_MODES)}"
            )

    # -- resolution ---------------------------------------------------------

    def resolved(
        self,
        drop_score: float,
        use_textline_orientation: bool,
        cpu_threads: Optional[int],
        backend: str = "rapidocr",
    ) -> "RapidOCRTuning":
        """Fill every inherited `None` from the owning engine's settings.

        Only resolved instances reach the cache key or the RapidOCR
        constructor, so no downstream code has to reason about `None`.

        Threading is PORTABLE BY DEFAULT on both backends as of 2026-09-09: if
        `intra_op_num_threads` is still exactly this class's UNTOUCHED default
        (8) -- i.e. nobody explicitly asked for a thread count, via
        `RapidOCRTuning(intra_op_num_threads=...)`, `--ocr-tune`, or
        `OCR_RAPID_THREADS` -- it resolves to "let the runtime decide" rather
        than to the literal 8, on EITHER backend. This mirrors the "swap only
        the untouched default" pattern `_load()` already uses for
        `det_model_path`/`rec_model_path`. What "let the runtime decide"
        concretely means differs by backend, because the two runtimes' own
        automatic-configuration sentinels differ:
        - `"rapidocr_openvino"`: resolves to `None`, which
          `to_rapidocr_openvino_kwargs()` drops entirely from the emitted
          kwargs -- confirmed by reading rapidocr_openvino's own
          infer_engine.py, omitting the kwarg skips OpenVINO's
          `set_property` call entirely, leaving its Core on its own
          automatic, hardware-aware thread/stream configuration.
        - every other backend (`"rapidocr"`, the ONNX Runtime path, and the
          default when `backend` is omitted): resolves to `-1`, RapidOCR/ONNX
          Runtime's own "use every core" sentinel -- ORT's bounds check
          rejects a bare `None`, so `-1` is the literal value that means
          "don't pin," not an omission the way OpenVINO's kwarg-drop is.
          This DIFFERS from this codebase's pre-2026-09-09 behavior, which
          pinned an untouched default to 8 on this backend specifically,
          based on a real sweep on the development machine (see the field's
          own comment) showing -1 performs worse THERE. That finding is
          preserved as a documented caveat, not silently discarded, but is no
          longer baked into the shipped default -- see the field's own
          comment for the full reasoning.

        `cpu_threads` (from the owning `OCREngine`, itself defaulted to
        `DEFAULT_CPU_THREADS` for the PaddleOCR path) is consulted only as a
        secondary fallback if `intra_op_num_threads` is `None` for a reason
        other than the untouched-default substitution above (i.e. it was set
        to `None` directly) -- an edge case with no current call site, kept
        for defensiveness rather than exercised in practice.

        An EXPLICIT value (anything other than the literal untouched default
        8) is always honored as given, on either backend -- nothing about
        configurability changed, only what leaving the field alone resolves
        to.
        """
        if self.intra_op_num_threads == 8:
            # Untouched class default -> genuinely automatic on both
            # backends, per this method's own docstring. What "automatic"
            # resolves to differs because the two runtimes' sentinels differ:
            # OpenVINO's kwarg-omission (None) is not equivalent to ORT's
            # literal -1, so this cannot be a single shared value.
            resolved_threads = None if backend == "rapidocr_openvino" else -1
        else:
            resolved_threads = (
                self.intra_op_num_threads
                if self.intra_op_num_threads is not None
                else (cpu_threads if cpu_threads is not None else -1)
            )

        return dataclasses.replace(
            self,
            text_score=self.text_score if self.text_score is not None else drop_score,
            use_cls=self.use_cls if self.use_cls is not None else use_textline_orientation,
            intra_op_num_threads=resolved_threads,
        )

    # -- translation --------------------------------------------------------

    def to_rapidocr_kwargs(self) -> dict[str, Any]:
        """Translate into the exact kwargs RapidOCR 1.4.4 reads.

        Drops `None`s (so the library's own default stands), inverts the
        dilation flag, and validates every emitted key against
        `_RAPIDOCR_KNOWN_KWARGS`.
        """
        kwargs: dict[str, Any] = {
            "max_side_len": self.max_side_len,
            "min_side_len": self.min_side_len,
            "text_score": self.text_score,
            "use_cls": self.use_cls,
            "det_limit_type": self.det_limit_type,
            "det_limit_side_len": self.det_limit_side_len,
            "det_thresh": self.det_thresh,
            "det_box_thresh": self.det_box_thresh,
            "det_unclip_ratio": self.det_unclip_ratio,
            # RapidOCR exposes the negative form only.
            "det_donot_use_dilation": not self.det_use_dilation,
            "det_score_mode": self.det_score_mode,
            "det_model_path": self.det_model_path,
            "rec_model_path": self.rec_model_path,
            "rec_keys_path": self.rec_keys_path,
            "intra_op_num_threads": self.intra_op_num_threads,
            "inter_op_num_threads": self.inter_op_num_threads,
        }
        kwargs = {k: v for k, v in kwargs.items() if v is not None}

        unknown = set(kwargs) - _RAPIDOCR_KNOWN_KWARGS
        if unknown:
            raise ValueError(
                f"Refusing to pass kwargs RapidOCR does not read: {sorted(unknown)}. "
                "RapidOCR would silently ignore these, so the tuning would not "
                "take effect and the run would look normal. Check "
                "_RAPIDOCR_KNOWN_KWARGS against the installed version."
            )
        return kwargs

    def to_rapidocr_openvino_kwargs(self) -> dict[str, Any]:
        """Translate into the kwargs rapidocr_openvino 1.3.25 reads.

        EXPERIMENTAL (2026-09-09). Shares every tuning field with
        `to_rapidocr_kwargs()` but the emitted kwarg names/set differ -- see
        `_RAPIDOCR_OPENVINO_KNOWN_KWARGS`'s comment for why a separate method
        exists rather than filtering the ONNX one. Notably `max_side_len` /
        `min_side_len` are DROPPED here: this older parse_parameters.py has no
        such global knob, so passing them would raise via the whitelist check
        below rather than silently doing nothing -- loud failure preferred
        over a quiet no-op, consistent with the ONNX path's own philosophy.
        det_model_path / rec_model_path are expected to point at OpenVINO IR
        (.xml) files here, not the .onnx files the ONNX path uses.
        """
        kwargs: dict[str, Any] = {
            "text_score": self.text_score,
            "use_cls": self.use_cls,  # 1.3.25 keeps "use_cls" (parity with ONNX), not "no_cls"
            "det_limit_type": self.det_limit_type,
            "det_limit_side_len": self.det_limit_side_len,
            "det_thresh": self.det_thresh,
            "det_box_thresh": self.det_box_thresh,
            "det_unclip_ratio": self.det_unclip_ratio,
            "det_donot_use_dilation": not self.det_use_dilation,
            "det_score_mode": self.det_score_mode,
            "det_model_path": self.det_model_path,
            "rec_model_path": self.rec_model_path,
            "rec_keys_path": self.rec_keys_path,
            "inference_num_threads": self.intra_op_num_threads,
        }
        kwargs = {k: v for k, v in kwargs.items() if v is not None}

        # "use_cls" itself is not in the transcribed whitelist (only "no_cls"
        # is documented in parse_parameters.py's argparse definitions), but
        # RapidOCR() constructors across both backends accept "use_cls" as a
        # direct override of the loaded config's Global.use_cls -- confirmed
        # by the smoke test (2026-09-09) actually disabling the cls step, not
        # silently ignoring it. Allow it explicitly alongside the whitelist
        # rather than expanding the whitelist itself, since it does not come
        # from parse_parameters.py's own argument list.
        unknown = set(kwargs) - _RAPIDOCR_OPENVINO_KNOWN_KWARGS - {"use_cls"}
        if unknown:
            raise ValueError(
                f"Refusing to pass kwargs rapidocr_openvino does not read: "
                f"{sorted(unknown)}. Check _RAPIDOCR_OPENVINO_KNOWN_KWARGS "
                "against the installed rapidocr-openvino version."
            )
        return kwargs

    # -- environment --------------------------------------------------------

    @classmethod
    def from_env(cls) -> "RapidOCRTuning":
        """Build from OCR_RAPID_* environment variables.

        Env is how bare `OCREngine()` constructions deep inside
        `document_loader.py` get tuned -- there is no parameter channel to
        those. Called lazily on the rapidocr branch only, so a malformed value
        can never break a PaddleOCR run, but at construction time so a typo
        fails immediately rather than mid-batch.
        """
        env_map = {
            "OCR_RAPID_MAX_SIDE_LEN": ("max_side_len", int),
            "OCR_RAPID_MIN_SIDE_LEN": ("min_side_len", int),
            "OCR_RAPID_TEXT_SCORE": ("text_score", float),
            "OCR_RAPID_USE_CLS": ("use_cls", _parse_bool),
            "OCR_RAPID_DET_LIMIT_TYPE": ("det_limit_type", str),
            "OCR_RAPID_DET_LIMIT_SIDE_LEN": ("det_limit_side_len", int),
            "OCR_RAPID_DET_THRESH": ("det_thresh", float),
            "OCR_RAPID_DET_BOX_THRESH": ("det_box_thresh", float),
            "OCR_RAPID_DET_UNCLIP_RATIO": ("det_unclip_ratio", float),
            "OCR_RAPID_DET_USE_DILATION": ("det_use_dilation", _parse_bool),
            "OCR_RAPID_DET_SCORE_MODE": ("det_score_mode", str),
            "OCR_RAPID_DET_MODEL_PATH": ("det_model_path", str),
            "OCR_RAPID_REC_MODEL_PATH": ("rec_model_path", str),
            "OCR_RAPID_REC_KEYS_PATH": ("rec_keys_path", str),
            "OCR_RAPID_THREADS": ("intra_op_num_threads", int),
            "OCR_RAPID_INTER_THREADS": ("inter_op_num_threads", int),
        }
        overrides: dict[str, Any] = {}
        for var, (field, caster) in env_map.items():
            raw = os.environ.get(var)
            if raw is None or raw.strip() == "":
                continue
            try:
                overrides[field] = caster(raw.strip())
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{var}={raw!r} is not a valid value: {exc}") from exc
        return cls(**overrides)


def parse_ocr_tune(pairs: Optional[list[str]]) -> Optional[RapidOCRTuning]:
    """Build a RapidOCRTuning from repeated `--ocr-tune name=value` arguments.

    Returns None when nothing was passed, so the caller falls through to the
    environment-derived default.

    Field names are validated against the dataclass itself, and types coerced
    from its annotations, so an unknown or misspelled name is a hard error here
    rather than a silently-ignored kwarg inside RapidOCR.
    """
    if not pairs:
        return None

    fields = {f.name: f for f in dataclasses.fields(RapidOCRTuning)}
    overrides: dict[str, Any] = {}

    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"--ocr-tune expects name=value, got {pair!r}")
        name, _, raw = pair.partition("=")
        name, raw = name.strip(), raw.strip()

        if name not in fields:
            raise ValueError(
                f"--ocr-tune {name!r} is not a tunable field. "
                f"Valid: {', '.join(sorted(fields))}"
            )

        # `from __future__ import annotations` makes these strings; matching on
        # the text is sufficient for this small, closed set of types.
        annotation = str(fields[name].type)
        if "bool" in annotation:
            caster: Any = _parse_bool
        elif "int" in annotation:
            caster = int
        elif "float" in annotation:
            caster = float
        else:
            caster = str

        try:
            overrides[name] = caster(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"--ocr-tune {name}={raw!r} is not valid: {exc}") from exc

    # Anything not overridden still comes from the environment, so
    # `--ocr-tune` composes with OCR_RAPID_* rather than replacing it.
    return dataclasses.replace(RapidOCRTuning.from_env(), **overrides)


def _parse_bool(raw: str) -> bool:
    lowered = raw.strip().lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"expected a boolean (true/false), got {raw!r}")


def clear_engine_cache() -> None:
    """Drop every cached engine.

    `_ENGINE_CACHE` is unbounded and each entry pins a full set of models, so a
    tuning sweep that builds many configurations in one process would otherwise
    accumulate all of them. Backend-neutral.
    """
    _ENGINE_CACHE.clear()


class OCREngine:
    """Thin, cached wrapper over RapidOCR (active) and PaddleOCR (preserved
    fallback, currently disabled -- see the "PRESERVED FALLBACK" banners in
    `_load()` and `read_image()`). `backend` selects which; see the
    OCR_BACKEND comment above `_REQUESTED_BACKEND` for how the default and
    per-call override interact.

    RapidOCR's own knobs live on `RapidOCRTuning` (passed via `rapid_tuning`),
    not on this class. The constructor parameters below mirror PaddleOCR 3.x's
    knobs and are ignored entirely when `backend == "rapidocr"`:

    - `use_textline_orientation`: rotates individual text lines. Cheques and
      phone-camera scans often have rotated lines; certificates rarely do.
    - `use_doc_orientation_classify` / `use_doc_unwarping`: whole-page
      deskew/dewarp. Both are off by default because they add a lot of
      latency and can distort already-flat renders of digital PDFs.
    - `text_det_limit_side_len`: detection input size. Larger recovers more
      small print (cheque account numbers) at the cost of speed.
    """

    def __init__(
        self,
        lang: str = "en",
        det_model: Optional[str] = DEFAULT_DET_MODEL,
        rec_model: Optional[str] = DEFAULT_REC_MODEL,
        use_textline_orientation: bool = False,
        use_doc_orientation_classify: bool = False,
        use_doc_unwarping: bool = False,
        text_det_limit_side_len: Optional[int] = None,
        drop_score: float = 0.30,
        cpu_threads: Optional[int] = DEFAULT_CPU_THREADS,
        backend: Optional[str] = None,
        rapid_tuning: Optional["RapidOCRTuning"] = None,
    ):
        # None means "take the process-wide default", so a caller can pin a
        # backend explicitly without the env var, and tests can pin PaddleOCR
        # regardless of how the environment is configured.
        self.backend = (backend or _REQUESTED_BACKEND).strip().lower()
        if self.backend not in _VALID_BACKENDS:
            raise ValueError(
                f"backend={self.backend!r} is not recognised. "
                f"Valid values: {sorted(_VALID_BACKENDS)}"
            )
        self.lang = lang
        self.det_model = det_model
        self.rec_model = rec_model
        self.use_textline_orientation = use_textline_orientation
        self.use_doc_orientation_classify = use_doc_orientation_classify
        self.use_doc_unwarping = use_doc_unwarping
        self.text_det_limit_side_len = text_det_limit_side_len
        # Pass None explicitly to fall back to PaddleOCR's own default (10).
        self.cpu_threads = cpu_threads
        # Spans below this recognition score are discarded outright -- they are
        # almost always watermark bleed or scan noise, and letting them through
        # pollutes spatial matching later.
        self.drop_score = drop_score

        # Resolved once, here, so the cache key and the RapidOCR constructor
        # never see a None-bearing tuning. `from_env()` is called lazily on the
        # rapidocr/rapidocr_openvino branches only -- a malformed OCR_RAPID_*
        # value must not be able to break a PaddleOCR run.
        if self.backend in ("rapidocr", "rapidocr_openvino"):
            tuning = rapid_tuning if rapid_tuning is not None else RapidOCRTuning.from_env()
            self._rapid_tuning: Optional[RapidOCRTuning] = tuning.resolved(
                drop_score=self.drop_score,
                use_textline_orientation=self.use_textline_orientation,
                cpu_threads=self.cpu_threads,
                backend=self.backend,
            )
        else:
            self._rapid_tuning = None

        self._ocr = None

    # -- engine lifecycle ---------------------------------------------------

    @property
    def _cache_key(self) -> tuple:
        # `backend` leads the key (plan.md task 2.2). Without it a PaddleOCR
        # engine could be served from the cache to a caller that asked for
        # RapidOCR whenever every other parameter happened to match -- a silent
        # swap of the component that reads bank account numbers.
        return (
            self.backend,
            self.lang,
            self.det_model,
            self.rec_model,
            self.use_textline_orientation,
            self.use_doc_orientation_classify,
            self.use_doc_unwarping,
            self.text_det_limit_side_len,
            self.cpu_threads,
            # Always None on the PaddleOCR path, so every position above keeps
            # its value for PaddleOCR configs and no OCR_RAPID_* setting can
            # perturb PaddleOCR engine identity. On the RapidOCR path this also
            # carries `drop_score` into the key transitively -- there it becomes
            # `text_score`, baked into the engine, rather than the pure
            # post-filter it is for PaddleOCR.
            self._rapid_tuning,
        )

    def _load(self):
        """Instantiate (or reuse) the active engine's pipeline (RapidOCR;
        PaddleOCR is the preserved but currently-disabled fallback)."""
        if self._ocr is not None:
            return self._ocr

        key = self._cache_key
        if key in _ENGINE_CACHE:
            self._ocr = _ENGINE_CACHE[key]
            return self._ocr

        if self.backend == "rapidocr":
            try:
                from rapidocr_onnxruntime import RapidOCR
            except ImportError as exc:
                raise ImportError(
                    "OCR_BACKEND=rapidocr requires the rapidocr-onnxruntime package. "
                    "Install it with:\n"
                    "    pip install -r backend/requirements.txt\n"
                    "or set OCR_BACKEND=paddleocr to use the preserved fallback engine "
                    "(also requires uncommenting its execution path -- see the "
                    "PRESERVED FALLBACK banner below in this file)."
                ) from exc

            tuning = self._rapid_tuning
            t0 = time.perf_counter()
            engine = RapidOCR(**tuning.to_rapidocr_kwargs())
            logger.info(
                "RapidOCR engine loaded in %.1fs -- ACTIVE backend, scored 19/24 "
                "(1 wrong, 0 hallucinated -- shared with PaddleOCR, see OCR_BACKEND "
                "comment above) against verified ground truth 2026-09-01. "
                "max_side_len=%d text_score=%.2f use_cls=%s det_box_thresh=%.2f "
                "intra_op=%s (machine has %s cores)",
                time.perf_counter() - t0, tuning.max_side_len, tuning.text_score,
                tuning.use_cls, tuning.det_box_thresh, tuning.intra_op_num_threads,
                os.cpu_count(),
            )
            _ENGINE_CACHE[key] = engine
            self._ocr = engine
            return engine

        if self.backend == "rapidocr_openvino":
            # ACTIVE DEFAULT as of 2026-09-09. Same detector/recognizer as the
            # rapidocr (ONNX Runtime) path, run through OpenVINO's runtime
            # instead. Measured ~2.3x faster median latency on this machine's
            # Intel CPU (10 isolated-subprocess reps: 1.71s vs 3.93s), with
            # identical accuracy on the ground-truth gate (19/24, 95.8%,
            # every GSTIN/PAN/IFSC/account-number field byte-identical to the
            # ONNX Runtime path). NOT validated on AMD or ARM64 (Apple
            # Silicon/Qualcomm) -- OpenVINO's fast paths are Intel-specific
            # kernels (AVX2/AVX-512/AMX); it is reported SLOWER than ONNX
            # Runtime on AMD Ryzen in third-party benchmarks. If this ever
            # needs to run well on non-Intel hardware, re-run this same
            # isolated-subprocess comparison there before trusting this
            # default -- see OCR_BACKEND=rapidocr to fall back to ONNX
            # Runtime, which has no such Intel-specific caveat.
            try:
                from rapidocr_openvino import RapidOCR as RapidOCROpenVINO
            except ImportError as exc:
                raise ImportError(
                    "OCR_BACKEND=rapidocr_openvino (the default) requires the "
                    "rapidocr-openvino and openvino packages (not yet in "
                    "requirements.txt -- see requirements.txt's own comment). "
                    "Install with:\n"
                    "    pip install rapidocr-openvino openvino-dev\n"
                    "or set OCR_BACKEND=rapidocr to use the ONNX Runtime path instead "
                    "(fully supported, no extra install, ~2.3x slower on this machine's "
                    "Intel CPU -- see the OpenVINO backend comment above)."
                ) from exc

            tuning = self._rapid_tuning
            # Auto-substitute the OpenVINO IR (.xml/.bin) models when the
            # tuning still holds RapidOCRTuning's plain .onnx defaults (i.e.
            # nobody explicitly overrode det/rec_model_path via OCR_RAPID_* or
            # rapid_tuning=). This is what makes a bare OCREngine() -- no env
            # vars, no kwargs -- work out of the box on this backend. An
            # explicit override (e.g. pointing at a different .onnx or a
            # different .xml) is respected as-is; only the untouched default
            # gets swapped, and only if it isn't already a .xml.
            det_default = str(_OCR_MODELS_DIR / "rapidocr_det_int8.onnx")
            rec_default = str(_OCR_MODELS_DIR / "rapidocr_rec_int8.onnx")
            if tuning.det_model_path == det_default:
                tuning = dataclasses.replace(
                    tuning,
                    det_model_path=str(_OCR_MODELS_DIR_OPENVINO / "rapidocr_det_int8_openvino.xml"),
                )
            if tuning.rec_model_path == rec_default:
                tuning = dataclasses.replace(
                    tuning,
                    rec_model_path=str(_OCR_MODELS_DIR_OPENVINO / "rapidocr_rec_int8_openvino.xml"),
                )
            if not tuning.det_model_path.endswith(".xml") or not tuning.rec_model_path.endswith(".xml"):
                raise ValueError(
                    f"OCR_BACKEND=rapidocr_openvino needs OpenVINO IR (.xml) models, got "
                    f"det_model_path={tuning.det_model_path!r} "
                    f"rec_model_path={tuning.rec_model_path!r}. Convert .onnx weights with "
                    "`ovc` first, or set OCR_BACKEND=rapidocr to use the ONNX Runtime path, "
                    "which reads the committed .onnx files directly."
                )
            t0 = time.perf_counter()
            engine = RapidOCROpenVINO(**tuning.to_rapidocr_openvino_kwargs())
            logger.info(
                "RapidOCR+OpenVINO engine loaded in %.1fs -- EXPERIMENTAL backend, "
                "not yet accuracy-gated against ground truth. "
                "text_score=%.2f use_cls=%s det_box_thresh=%.2f threads=%s "
                "det_model=%s (machine has %s cores)",
                time.perf_counter() - t0, tuning.text_score, tuning.use_cls,
                tuning.det_box_thresh, tuning.intra_op_num_threads,
                tuning.det_model_path, os.cpu_count(),
            )
            _ENGINE_CACHE[key] = engine
            self._ocr = engine
            return engine

        # =====================================================================
        # PRESERVED FALLBACK -- PREVIOUS OCR ENGINE (PaddleOCR)
        # =====================================================================
        # This is the only reachable code for backend == "paddleocr" as of the
        # 2026-09-01 switch to RapidOCR as the active engine. It is disabled,
        # not deleted: PaddleOCR remains fully supported and this whole block
        # is a straight copy of the working implementation.
        #
        # TO RESTORE PADDLEOCR AS THE ACTIVE ENGINE:
        #   1. Uncomment this block (the `from paddleocr import PaddleOCR ...`
        #      through the final `return engine` below).
        #   2. Delete/comment out the `raise RuntimeError(...)` immediately
        #      below this banner.
        #   3. Uncomment the matching PRESERVED FALLBACK block in
        #      read_image() and in _result_to_spans() further down this file.
        #   4. Un-skip the paddleocr parametrize cases in
        #      tests/test_bank_record_consistency.py.
        #   5. Either set OCR_BACKEND=paddleocr, or change the default back
        #      at this file's _REQUESTED_BACKEND line above.
        #   No reinstall is needed -- paddlepaddle/paddleocr stay pinned in
        #   requirements.txt specifically so this restore is code-only.
        #
        # from paddleocr import PaddleOCR  # imported lazily, see module docstring
        #
        # kwargs: dict[str, Any] = {
        #     "lang": self.lang,
        #     "use_textline_orientation": self.use_textline_orientation,
        #     "use_doc_orientation_classify": self.use_doc_orientation_classify,
        #     "use_doc_unwarping": self.use_doc_unwarping,
        # }
        # if self.det_model:
        #     kwargs["text_detection_model_name"] = self.det_model
        # if self.rec_model:
        #     kwargs["text_recognition_model_name"] = self.rec_model
        # if self.text_det_limit_side_len is not None:
        #     kwargs["text_det_limit_side_len"] = self.text_det_limit_side_len
        # if self.cpu_threads is not None:
        #     kwargs["cpu_threads"] = self.cpu_threads
        #
        # t0 = time.perf_counter()
        # if _REQUESTED_DEVICE == "gpu":
        #     try:
        #         engine = PaddleOCR(device="gpu", **kwargs)
        #         logger.info(
        #             "PaddleOCR engine loaded on GPU in %.1fs (%s)",
        #             time.perf_counter() - t0, key,
        #         )
        #     except Exception as exc:
        #         # Common causes: no GPU present, paddlepaddle-gpu not
        #         # installed (this is still the CPU wheel), or the CUDA
        #         # Toolkit runtime DLLs (cuBLAS etc.) aren't on PATH even
        #         # though the driver is fine. Falling back keeps the app
        #         # usable rather than crashing the whole pipeline over an
        #         # optional speedup.
        #         logger.warning(
        #             "GPU OCR engine init failed (%s: %s) -- falling back to CPU",
        #             type(exc).__name__, exc,
        #         )
        #         t0 = time.perf_counter()
        #         engine = PaddleOCR(**kwargs)
        #         logger.info(
        #             "PaddleOCR engine loaded on CPU in %.1fs (%s)",
        #             time.perf_counter() - t0, key,
        #         )
        # else:
        #     engine = PaddleOCR(**kwargs)
        #     logger.info("PaddleOCR engine loaded in %.1fs (%s)", time.perf_counter() - t0, key)
        #
        # _ENGINE_CACHE[key] = engine
        # self._ocr = engine
        # return engine
        raise RuntimeError(
            "OCR_BACKEND=paddleocr is currently disabled. PaddleOCR is PRESERVED "
            "as a fallback but its execution path is commented out in "
            "ocr_engine.py's _load() -- see the PRESERVED FALLBACK banner just "
            "above this error for the exact restore steps."
        )
        # ===== END PRESERVED FALLBACK =====

    def warmup(self) -> float:
        """Force model load up-front so the first real page isn't charged for
        it. Returns seconds spent."""
        t0 = time.perf_counter()
        self._load()
        return time.perf_counter() - t0

    # -- inference ----------------------------------------------------------

    def read_image(
        self,
        image: np.ndarray,
        source_document: str,
        page: int,
        span_source: SpanSource = SpanSource.OCR,
        order_offset: int = 0,
    ) -> list[TextSpan]:
        """Run OCR on one RGB image and return spans in reading order."""
        engine = self._load()

        # PaddleOCR expects BGR (OpenCV convention); our renders are RGB.
        if image.ndim == 3 and image.shape[2] == 3:
            image = image[:, :, ::-1]
        elif image.ndim == 3 and image.shape[2] == 4:
            image = image[:, :, [2, 1, 0]]

        image = np.ascontiguousarray(image)
        spans: list[TextSpan] = []

        if self.backend in ("rapidocr", "rapidocr_openvino"):
            # RapidOCR is callable and returns (results, elapse), where each
            # result is [polygon, text, score]. Identical contract on both the
            # rapidocr_onnxruntime and rapidocr_openvino packages (same
            # upstream project, different inference runtime underneath), so
            # the same adapter (_rapid_result_to_spans) handles both.
            #
            # NEVER pass kwargs here. RapidOCR accepts box_thresh/unclip_ratio/
            # text_score per call, but doing so (a) MUTATES the engine instance
            # permanently, and this instance is shared via _ENGINE_CACHE, so the
            # override leaks into every later caller and silently falsifies the
            # cache key; and (b) passing any one of the three resets the other
            # two to hardcoded literals (main.py:76-86), so a single override
            # would quietly undo the constructor's tuning. All configuration is
            # applied at construction, in _load(). Keep this call bare.
            results, _elapse = engine(image)
            spans.extend(
                _rapid_result_to_spans(
                    results,
                    source_document=source_document,
                    page=page,
                    span_source=span_source,
                    drop_score=self.drop_score,
                )
            )
        else:
            # ================================================================
            # PRESERVED FALLBACK -- PREVIOUS OCR ENGINE (PaddleOCR)
            # ================================================================
            # Unreachable as of 2026-09-01: self._load() above already raises
            # RuntimeError for backend == "paddleocr" before code can reach
            # here. Kept and marked so this branch's logic is not lost. To
            # restore: uncomment this loop, remove the raise below it, and
            # follow the restore steps in _load()'s matching banner.
            #
            # for result in engine.predict(image) or []:
            #     spans.extend(
            #         _result_to_spans(
            #             result,
            #             source_document=source_document,
            #             page=page,
            #             span_source=span_source,
            #             drop_score=self.drop_score,
            #         )
            #     )
            raise RuntimeError(
                "OCR_BACKEND=paddleocr is currently disabled -- see the "
                "PRESERVED FALLBACK banner in _load()."
            )
            # ===== END PRESERVED FALLBACK =====

        spans = sort_reading_order(spans)
        for i, span in enumerate(spans):
            span.order = order_offset + i
        return spans


# ---------------------------------------------------------------------------
# RESULT ADAPTER
# ---------------------------------------------------------------------------

def _rapid_result_to_spans(
    results: Any,
    source_document: str,
    page: int,
    span_source: SpanSource,
    drop_score: float,
) -> list[TextSpan]:
    """Normalize RapidOCR output into TextSpans.

    RapidOCR returns a flat list of `[polygon, text, score]`, where polygon is
    four (x, y) points. Filtering and bbox collapse deliberately mirror
    `_result_to_spans()` exactly, so the only variable between backends is the
    OCR itself and not the post-processing around it.
    """
    spans: list[TextSpan] = []
    for entry in (results or []):
        try:
            poly, text, score = entry[0], entry[1], float(entry[2])
        except (TypeError, IndexError, ValueError):
            logger.warning("Unrecognized RapidOCR entry: %r", entry)
            continue

        text = (text or "").strip()
        if not text or score < drop_score:
            continue

        bbox = _to_bbox(poly)
        if bbox is None:
            continue

        spans.append(
            TextSpan(
                text=text,
                page=page,
                bbox=bbox,
                source_document=source_document,
                confidence=score,
                source=span_source,
            )
        )
    return spans


# =============================================================================
# PRESERVED FALLBACK -- PREVIOUS OCR ENGINE (PaddleOCR) result adapter
# =============================================================================
# Unreachable as of 2026-09-01: its only caller (read_image()'s paddle branch)
# is disabled. Left intact and callable (not commented out, since it is pure
# and self-contained) so restoring PaddleOCR needs no changes here -- only the
# read_image() and _load() banners need uncommenting. See _load()'s banner
# for full restore steps.
def _result_to_spans(
    result: Any,
    source_document: str,
    page: int,
    span_source: SpanSource,
    drop_score: float,
) -> list[TextSpan]:
    """Normalize one PaddleOCR 3.x result into TextSpans.

    The 3.x result is a dict-like `OCRResult` carrying parallel arrays
    (`rec_texts`, `rec_scores`, and polygons under one of several keys
    depending on pipeline configuration). We read it defensively so a minor
    PaddleOCR point release renaming a key degrades rather than crashes.
    """
    data = result if isinstance(result, dict) else getattr(result, "json", None)
    if isinstance(data, dict) and "res" in data:  # some versions nest under "res"
        data = data["res"]
    if not isinstance(data, dict):
        logger.warning("Unrecognized PaddleOCR result type: %r", type(result))
        return []

    texts = data.get("rec_texts") or []
    scores = data.get("rec_scores") or []
    polys = (
        data.get("rec_polys")
        if data.get("rec_polys") is not None
        else data.get("rec_boxes")
        if data.get("rec_boxes") is not None
        else data.get("dt_polys")
    )
    if polys is None:
        polys = []

    spans: list[TextSpan] = []
    for i, text in enumerate(texts):
        text = (text or "").strip()
        if not text:
            continue
        score = float(scores[i]) if i < len(scores) else 1.0
        if score < drop_score:
            continue

        bbox = _to_bbox(polys[i]) if i < len(polys) else None
        if bbox is None:
            continue

        spans.append(
            TextSpan(
                text=text,
                page=page,
                bbox=bbox,
                source_document=source_document,
                confidence=score,
                source=span_source,
            )
        )
    return spans


def _to_bbox(poly) -> Optional[BBox]:
    """Accept either a 4-point polygon [[x,y]*4] or a flat [x1,y1,x2,y2] box."""
    arr = np.asarray(poly, dtype=float)
    if arr.ndim == 2 and arr.shape[0] >= 3:
        return BBox.from_polygon(arr)
    flat = arr.reshape(-1)
    if flat.size == 4:
        x1, y1, x2, y2 = flat
        return BBox(min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
    return None


# ---------------------------------------------------------------------------
# READING ORDER
# ---------------------------------------------------------------------------

def sort_reading_order(spans: list[TextSpan]) -> list[TextSpan]:
    """Sort top-to-bottom, left-to-right, grouping spans into visual lines.

    Grouping is by vertical-centre distance against the *median* span height,
    not by raw box overlap. Real documents contain tall outliers -- rotated
    sidebar text, watermarks, a stamp -- and a single one of those overlaps
    many genuine lines at once. Keying off overlap lets one such box swallow
    several rows and scramble their order; keying off centre distance with a
    median-derived tolerance keeps outliers on their own line and leaves the
    real rows intact.

    Order matters because the semantic engine reads label/value adjacency: a
    label on the left and its value on the right must stay on one line, and a
    naive sort by y alone would interleave neighbouring columns.
    """
    if not spans:
        return []

    heights = [s.bbox.height for s in spans if s.bbox.height > 0]
    median_h = float(np.median(heights)) if heights else 1.0
    tolerance = max(median_h * 0.6, 1.0)

    lines: list[list[TextSpan]] = []
    line_centres: list[float] = []
    for span in sorted(spans, key=lambda s: (s.bbox.y1, s.bbox.x1)):
        cy = span.bbox.cy
        placed = False
        for i, centre in enumerate(line_centres):
            if abs(cy - centre) <= tolerance:
                lines[i].append(span)
                # Track the line's running centre so it drifts with the row
                # rather than being pinned to whichever span arrived first.
                line_centres[i] = float(np.mean([s.bbox.cy for s in lines[i]]))
                placed = True
                break
        if not placed:
            lines.append([span])
            line_centres.append(cy)

    ordered: list[TextSpan] = []
    for _, line in sorted(zip(line_centres, lines), key=lambda pair: pair[0]):
        ordered.extend(sorted(line, key=lambda s: s.bbox.x1))
    return ordered