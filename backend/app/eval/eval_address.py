"""Address segmentation accuracy check.

Feeds each raw address string in `address_cases.yaml` through the current
splitting logic and scores the fields it pulls out against the hand-made
expected values. Prints a per-case, per-field table and an aggregate.

    python -m app.eval.eval_address                # score current code
    python -m app.eval.eval_address --verbose      # also show every field

Exit code is non-zero when any field is `wrong` or `missed`, so this can gate
a change: run it before a phase to record the baseline, run it after to prove
the phase did not regress.

Which splitter is scored is decided by `_split(address)` below -- it currently
calls onboarding_mapper._split_trailing_location. When address_resolver lands,
point it there instead and the same cases measure the new module.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

_CASES_FILE = Path(__file__).with_name("address_cases.yaml")

# Fields we score, in display order. --multiline extends this with
# address_3/address_4 (see main()).
_FIELDS = ("address_1", "address_2", "city", "state", "pin_code")
_MULTILINE_FIELDS = ("address_1", "address_2", "address_3", "address_4", "city", "state", "pin_code")


def _split(address: str, multiline: bool = False) -> dict[str, str]:
    """Run the address through the current segmentation path and return
    {address_1, address_2, [address_3, address_4,] city, state, pin_code,
    _confidence}.

    Points at extraction_pipeline.extract.address_resolver (Phase 2). The
    pre-Phase-2 baseline pointed at onboarding_mapper._split_trailing_location;
    the 37/55 number recorded then is what every change here must not drop
    below. `_confidence` is reported but not scored -- the case files carry no
    expected confidence, it is shown so a `correct`-but-`low` row is visible.

    `multiline=True` (opt-in, matching `resolve_address_blob`'s own default)
    additionally splits the premises/locality remainder into address_3/
    address_4 via address_segmenter.py, instead of leaving them empty. Scored
    against the SAME address_cases.yaml -- these 11 hand-written cases were
    written under the legacy "everything in address_1" convention, so most are
    expected to still show address_3/address_4 as "" (correctly_absent) even
    in multiline mode; this run exists to catch any UNEXPECTED hallucination
    of a third/fourth line on cases that should not produce one, not to
    exercise the segmenter's own dedicated corpus (address_line_cases.yaml,
    scored separately by test_address_segmenter.py).
    """
    from app.services.extraction_pipeline.extract.address_resolver import (
        resolve_address_blob,
    )

    r = resolve_address_blob(address, multiline=multiline)
    out = {
        "address_1": r.address_1.strip(", ").strip(),
        "address_2": r.address_2,
        "city": r.city,
        "state": r.state,
        "pin_code": r.pin_code,
        "_confidence": r.confidence,
    }
    if multiline:
        out["address_3"] = r.address_3
        out["address_4"] = r.address_4
    return out


def _norm(s: str) -> str:
    """Comparison-normalise a string: collapse whitespace, drop a trailing
    comma, casefold. Address strings vary in spacing and comma style without
    being wrong."""
    return " ".join((s or "").replace(",", " , ").split()).strip(" ,").casefold()


def _score_field(got: str, want: str) -> str:
    got, want = (got or "").strip(), (want or "").strip()
    if want == "":
        return "correctly_absent" if got == "" else "hallucinated"
    if got == "":
        return "missed"
    return "correct" if _norm(got) == _norm(want) else "wrong"


def main() -> int:
    ap = argparse.ArgumentParser(description="Score address segmentation against address_cases.yaml")
    ap.add_argument("--verbose", action="store_true", help="show every field, not just failures")
    ap.add_argument(
        "--multiline", action="store_true",
        help="score resolve_address_blob(multiline=True) instead of the legacy "
             "single-address_1 path -- adds address_3/address_4 to the scored "
             "fields (expected '' on every existing address_cases.yaml case "
             "unless noted otherwise).",
    )
    ap.add_argument(
        "--cases",
        help="path to an alternative case file (default: address_cases.yaml). "
             "Use with --multiline to score a corpus written under the 4-line "
             "convention, e.g. address_vendor_lines.yaml. Keeping those in a "
             "separate file leaves the legacy baseline comparable.",
    )
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass

    fields = _MULTILINE_FIELDS if args.multiline else _FIELDS

    cases_file = Path(args.cases) if args.cases else _CASES_FILE
    if not cases_file.is_absolute() and args.cases:
        # Allow either a path relative to the working directory or a bare
        # filename sitting next to the default case file.
        beside = _CASES_FILE.with_name(cases_file.name)
        if not cases_file.exists() and beside.exists():
            cases_file = beside
    if not cases_file.exists():
        print(f"case file not found: {cases_file}", file=sys.stderr)
        return 2

    doc = yaml.safe_load(cases_file.read_text(encoding="utf-8"))
    cases = doc["cases"]

    tally = {"correct": 0, "wrong": 0, "missed": 0, "hallucinated": 0, "correctly_absent": 0}
    conf_tally: dict[str, int] = {"high": 0, "medium": 0, "low": 0}
    failing_cases: list[str] = []
    # cases the resolver got fully right but flagged low-confidence, and the
    # reverse -- a wrong/missed field the resolver reported as high. Both are
    # calibration problems worth seeing even though neither changes the score.
    miscalibrated: list[str] = []

    print("=" * 78)
    mode = " [--multiline]" if args.multiline else ""
    print(f"ADDRESS SEGMENTATION{mode}  --  {len(cases)} cases, {len(fields)} fields each")
    print(f"cases: {cases_file.name}")
    print("=" * 78)

    for case in cases:
        cid = case["id"]
        got = _split(case["address"], multiline=args.multiline)
        want = case["expect"]

        results = {f: _score_field(got.get(f, ""), want.get(f, "")) for f in fields}
        for r in results.values():
            tally[r] += 1
        bad = [f for f, r in results.items() if r in ("wrong", "missed", "hallucinated")]
        if bad:
            failing_cases.append(cid)

        conf = got.get("_confidence", "low")
        conf_tally[conf] = conf_tally.get(conf, 0) + 1
        # `low` is the right call when the case genuinely has nothing to find
        # (no city/state/pin expected), so only flag a low-confidence case that
        # actually resolved something, and any wrong/missed case reported high.
        resolved_something = any(want.get(f) for f in ("city", "state", "pin_code"))
        if not bad and conf == "low" and resolved_something:
            miscalibrated.append(f"{cid} (correct but low)")
        elif bad and conf == "high":
            miscalibrated.append(f"{cid} (wrong but high)")

        status = "OK  " if not bad else "FAIL"
        print(f"\n[{status}] {cid}   confidence={conf}")
        if args.verbose or bad:
            for f in fields:
                r = results[f]
                mark = "  " if r in ("correct", "correctly_absent") else "**"
                print(f"   {mark} {f:11} {r:16} got={got.get(f, '')!r}")
                if r in ("wrong", "missed"):
                    print(f"      {'':11} {'':16} want={want.get(f, '')!r}")

    total = sum(tally.values())
    print("\n" + "=" * 78)
    print("AGGREGATE")
    print("=" * 78)
    for k in ("correct", "wrong", "missed", "hallucinated", "correctly_absent"):
        print(f"  {k:18} {tally[k]:3}")
    good = tally["correct"] + tally["correctly_absent"]
    print(f"  {'-'*18}")
    print(f"  {'good / total':18} {good:3} / {total}   ({100*good/total:.1f}%)")
    print(f"  {'-'*18}")
    print(f"  {'confidence':18} high={conf_tally['high']}  "
          f"medium={conf_tally['medium']}  low={conf_tally['low']}   (reported, not scored)")
    if miscalibrated:
        print(f"  {'miscalibrated':18} {', '.join(miscalibrated)}")
    if failing_cases:
        print(f"\n  failing cases: {', '.join(failing_cases)}")
    print("=" * 78)

    return 1 if (tally["wrong"] or tally["missed"] or tally["hallucinated"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
