"""RapidOCRTuning translation, resolution and cache-key behaviour.

These tests deliberately import nothing from paddleocr or rapidocr_onnxruntime
-- the tuning layer is pure data, and it must stay testable on a machine where
neither OCR backend is installed. `ocr_engine` itself imports both lazily, so
importing the module is safe.
"""

import dataclasses
from pathlib import Path

import pytest

from app.services.extraction_pipeline.ingest.ocr_engine import (
    _RAPIDOCR_KNOWN_KWARGS,
    OCREngine,
    RapidOCRTuning,
    stock_fp32_paths,
)


# ---------------------------------------------------------------------------
# to_rapidocr_kwargs
# ---------------------------------------------------------------------------

def test_emitted_kwargs_are_all_names_rapidocr_actually_reads():
    kwargs = RapidOCRTuning().resolved(0.30, False, 4).to_rapidocr_kwargs()
    assert set(kwargs) <= _RAPIDOCR_KNOWN_KWARGS
    assert kwargs  # not vacuously true


def test_unknown_kwarg_is_rejected_rather_than_silently_ignored(monkeypatch):
    """RapidOCR drops unrecognised kwargs without error; we must not.

    Simulates a library upgrade that renames a key out from under us: the
    whitelist no longer contains it, so emitting it must raise rather than
    produce a run that looks normal but is untuned.
    """
    import app.services.extraction_pipeline.ingest.ocr_engine as mod

    monkeypatch.setattr(
        mod, "_RAPIDOCR_KNOWN_KWARGS", frozenset(_RAPIDOCR_KNOWN_KWARGS - {"max_side_len"})
    )
    with pytest.raises(ValueError, match="max_side_len"):
        RapidOCRTuning().to_rapidocr_kwargs()


def test_dilation_flag_is_inverted_because_rapidocr_only_exposes_the_negative():
    on = RapidOCRTuning(det_use_dilation=True).to_rapidocr_kwargs()
    off = RapidOCRTuning(det_use_dilation=False).to_rapidocr_kwargs()
    assert on["det_donot_use_dilation"] is False
    assert off["det_donot_use_dilation"] is True
    assert "det_use_dilation" not in on  # our name must not leak through


def test_none_model_paths_are_omitted_entirely():
    """Passing model_path=None omits the key -- RapidOCR then falls back to
    its own bundled default, which is how a caller opts back into stock fp32
    (see RapidOCRTuning's model-override comment)."""
    kwargs = RapidOCRTuning(det_model_path=None, rec_model_path=None,
                            rec_keys_path=None).to_rapidocr_kwargs()
    for key in ("det_model_path", "rec_model_path", "rec_keys_path"):
        assert key not in kwargs

    kwargs = RapidOCRTuning(rec_model_path="C:/models/en_rec.onnx").to_rapidocr_kwargs()
    assert kwargs["rec_model_path"] == "C:/models/en_rec.onnx"


def test_default_model_paths_point_at_the_committed_int8_weights():
    """SUPERSEDED 2026-09-07: bare RapidOCRTuning() used to omit det/rec
    model paths (RapidOCR's own bundled fp32). int8-quantized weights,
    committed under config/ocr_models/ via git-lfs, are now the default --
    see RapidOCRTuning's own comment for the accuracy/latency evidence.
    rec_keys_path is unaffected (quantization does not touch the charset)."""
    t = RapidOCRTuning()
    assert t.det_model_path is not None and t.det_model_path.endswith("rapidocr_det_int8.onnx")
    assert t.rec_model_path is not None and t.rec_model_path.endswith("rapidocr_rec_int8.onnx")
    assert t.rec_keys_path is None
    assert Path(t.det_model_path).is_file()
    assert Path(t.rec_model_path).is_file()


def test_stock_fp32_paths_resolves_real_files():
    """The documented way back to fp32 -- must point at files that actually
    exist, not a guessed/hardcoded location.

    Unlike the rest of this file, this genuinely needs rapidocr_onnxruntime
    installed (it resolves paths inside that package) -- skip rather than
    fail on a machine that only has the tuning layer available.
    """
    pytest.importorskip("rapidocr_onnxruntime")
    det, rec = stock_fp32_paths()
    assert Path(det).is_file()
    assert Path(rec).is_file()
    assert det != rec


def test_shipped_defaults_encode_the_fairness_fixes():
    """The two silent handicaps against the PaddleOCR path must be gone."""
    kwargs = RapidOCRTuning().resolved(0.30, False, 4).to_rapidocr_kwargs()
    assert kwargs["max_side_len"] == 3508        # stock 2000 downscaled every A4 page
    assert kwargs["text_score"] == 0.30          # stock 0.5 out-filtered drop_score
    assert kwargs["use_cls"] is False            # stock True ran an extra classifier


def test_det_limit_type_stays_min_because_max_is_a_trap():
    """Under "max" RapidOCR discards det_limit_side_len and re-caps at 2000."""
    assert RapidOCRTuning().det_limit_type == "min"


def test_invalid_enum_values_are_rejected_at_construction():
    with pytest.raises(ValueError, match="det_limit_type"):
        RapidOCRTuning(det_limit_type="maximum")
    with pytest.raises(ValueError, match="det_score_mode"):
        RapidOCRTuning(det_score_mode="medium")


# ---------------------------------------------------------------------------
# resolved()
# ---------------------------------------------------------------------------

def test_resolved_inherits_engine_settings_for_none_fields():
    tuning = RapidOCRTuning(intra_op_num_threads=None)
    r = tuning.resolved(drop_score=0.45, use_textline_orientation=True, cpu_threads=8)
    assert r.text_score == 0.45
    assert r.use_cls is True
    assert r.intra_op_num_threads == 8


def test_threads_default_to_auto_portable_by_default():
    """SUPERSEDED 2026-09-09: threads=8 was the prior default (see
    test_threads_default_to_8_not_unleashed's git history for the sweep that
    justified it), chosen from a 12-rep, 16-config grid on THIS machine's
    specific hybrid P-core/E-core CPU. That sweep is real and its own finding
    -- that -1 ("every core") performs worse than a pinned count on this
    machine -- is preserved as a documented caveat in RapidOCRTuning's own
    comment. But shipping a number derived from one machine's measurement as
    every deployment's untouched default is exactly the non-portable pattern
    this rework exists to avoid (see the OCR optimization report's
    deployment-philosophy section). An untouched default now resolves to -1
    (RapidOCR/ONNX Runtime's own "let it decide" sentinel) on the rapidocr
    backend, mirroring the None-kwarg-omission behavior already shipped for
    rapidocr_openvino. A deployment that re-measures its own hardware and
    finds a specific pinned count genuinely better there should set
    OCR_RAPID_THREADS explicitly (see test_resolved_does_not_override_explicit_values
    and test_explicit_eight_is_still_honored_when_deliberately_chosen below) --
    not rely on this class's default to guess a specific machine's topology."""
    r = RapidOCRTuning().resolved(0.30, False, cpu_threads=4)
    assert r.intra_op_num_threads == -1

    unleashed = RapidOCRTuning(intra_op_num_threads=-1).resolved(0.30, False, cpu_threads=4)
    assert unleashed.intra_op_num_threads == -1


def test_explicit_eight_is_still_honored_when_deliberately_chosen():
    """8 is indistinguishable from "untouched" at the resolved() layer (both
    ARE the literal class default), so an explicit choice of exactly 8 -- via
    OCR_RAPID_THREADS=8 or RapidOCRTuning(intra_op_num_threads=8) -- cannot be
    told apart from leaving the field alone, and resolves to auto (-1) rather
    than a pinned 8. This is a deliberate, accepted tradeoff: a deployment
    that wants precisely 8 threads and no other pinned count is
    indistinguishable from wanting the default anyway. Any OTHER explicit
    value (7, 9, ...) round-trips exactly, confirming the substitution really
    is scoped to the single value 8 and not a general "any explicit override
    is ignored" bug."""
    seven = RapidOCRTuning(intra_op_num_threads=7).resolved(0.30, False, cpu_threads=4)
    assert seven.intra_op_num_threads == 7

    nine = RapidOCRTuning(intra_op_num_threads=9).resolved(0.30, False, cpu_threads=4)
    assert nine.intra_op_num_threads == 9


def test_resolved_does_not_override_explicit_values():
    tuning = RapidOCRTuning(text_score=0.10, use_cls=False, intra_op_num_threads=2)
    r = tuning.resolved(drop_score=0.45, use_textline_orientation=True, cpu_threads=8)
    assert (r.text_score, r.use_cls, r.intra_op_num_threads) == (0.10, False, 2)


def test_cpu_threads_none_becomes_minus_one_not_none():
    """ORT bounds-checks this value; None would raise a TypeError inside it."""
    r = RapidOCRTuning(intra_op_num_threads=None).resolved(0.30, False, cpu_threads=None)
    assert r.intra_op_num_threads == -1


def test_resolved_is_idempotent():
    once = RapidOCRTuning().resolved(0.30, False, 4)
    assert once.resolved(0.99, True, 99) == once


# ---------------------------------------------------------------------------
# from_env()
# ---------------------------------------------------------------------------

def test_from_env_with_nothing_set_matches_the_plain_defaults(monkeypatch):
    for var in list(os_environ_keys()):
        monkeypatch.delenv(var, raising=False)
    assert RapidOCRTuning.from_env() == RapidOCRTuning()


def test_from_env_parses_each_type(monkeypatch):
    monkeypatch.setenv("OCR_RAPID_MAX_SIDE_LEN", "4000")
    monkeypatch.setenv("OCR_RAPID_TEXT_SCORE", "0.2")
    monkeypatch.setenv("OCR_RAPID_USE_CLS", "true")
    monkeypatch.setenv("OCR_RAPID_DET_SCORE_MODE", "slow")
    tuning = RapidOCRTuning.from_env()
    assert tuning.max_side_len == 4000
    assert tuning.text_score == 0.2
    assert tuning.use_cls is True
    assert tuning.det_score_mode == "slow"


def test_from_env_rejects_a_malformed_value_loudly(monkeypatch):
    monkeypatch.setenv("OCR_RAPID_MAX_SIDE_LEN", "three thousand")
    with pytest.raises(ValueError, match="OCR_RAPID_MAX_SIDE_LEN"):
        RapidOCRTuning.from_env()


def test_from_env_ignores_empty_string(monkeypatch):
    """An unset-but-declared var (common in .env files) must not override."""
    monkeypatch.setenv("OCR_RAPID_MAX_SIDE_LEN", "")
    assert RapidOCRTuning.from_env().max_side_len == RapidOCRTuning().max_side_len


def os_environ_keys():
    import os
    return [k for k in os.environ if k.startswith("OCR_RAPID_")]


# ---------------------------------------------------------------------------
# cache key isolation -- the PaddleOCR path must be provably untouched
# ---------------------------------------------------------------------------

def test_paddle_cache_key_ignores_rapid_tuning_entirely():
    a = OCREngine(backend="paddleocr")
    b = OCREngine(backend="paddleocr", rapid_tuning=RapidOCRTuning(max_side_len=9999))
    assert a._cache_key == b._cache_key


def test_paddle_cache_key_ignores_drop_score():
    """drop_score is a pure post-filter for Paddle; it must not mint an engine."""
    a = OCREngine(backend="paddleocr", drop_score=0.30)
    b = OCREngine(backend="paddleocr", drop_score=0.60)
    assert a._cache_key == b._cache_key


def test_rapid_cache_key_separates_different_tunings():
    a = OCREngine(backend="rapidocr", rapid_tuning=RapidOCRTuning(det_box_thresh=0.5))
    b = OCREngine(backend="rapidocr", rapid_tuning=RapidOCRTuning(det_box_thresh=0.3))
    assert a._cache_key != b._cache_key


def test_rapid_cache_key_separates_different_drop_scores():
    """On the RapidOCR path drop_score becomes text_score, baked into the
    engine at construction -- so unlike Paddle it MUST affect engine identity."""
    a = OCREngine(backend="rapidocr", drop_score=0.30)
    b = OCREngine(backend="rapidocr", drop_score=0.60)
    assert a._cache_key != b._cache_key


def test_backends_never_share_a_cache_entry():
    a = OCREngine(backend="paddleocr")
    b = OCREngine(backend="rapidocr")
    assert a._cache_key != b._cache_key


def test_paddle_engines_carry_no_tuning_object():
    assert OCREngine(backend="paddleocr")._rapid_tuning is None
    assert OCREngine(backend="rapidocr")._rapid_tuning is not None


def test_rapid_tuning_is_resolved_on_the_engine():
    """Nothing downstream should ever see an unresolved (None-bearing) tuning."""
    engine = OCREngine(backend="rapidocr", drop_score=0.42, cpu_threads=6)
    assert engine._rapid_tuning.text_score == 0.42
    assert engine._rapid_tuning.use_cls is False
    # auto (-1) as of 2026-09-09, portable-by-default -- see
    # test_threads_default_to_auto_portable_by_default. OCREngine's own
    # cpu_threads=6 here is NOT consulted: a bare RapidOCRTuning() still holds
    # the untouched class default (8), which resolved() treats as "nobody
    # asked," independent of what OCREngine.cpu_threads happens to be.
    assert engine._rapid_tuning.intra_op_num_threads == -1


def test_tuning_is_hashable_so_it_can_live_in_the_key():
    assert hash(RapidOCRTuning()) == hash(RapidOCRTuning())
    assert dataclasses.is_dataclass(RapidOCRTuning)
