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

# Fields we score, in display order.
_FIELDS = ("address_1", "address_2", "city", "state", "pin_code")


def _split(address: str) -> dict[str, str]:
    """Run the address through the current segmentation path and return
    {address_1, address_2, city, state, pin_code, _confidence}.

    Points at extraction_pipeline.extract.address_resolver (Phase 2). The
    pre-Phase-2 baseline pointed at onboarding_mapper._split_trailing_location;
    the 37/55 number recorded then is what every change here must not drop
    below. `_confidence` is reported but not scored -- the case files carry no
    expected confidence, it is shown so a `correct`-but-`low` row is visible.
    """
    from app.services.extraction_pipeline.extract.address_resolver import (
        resolve_address_blob,
    )

    r = resolve_address_blob(address)
    return {
        "address_1": r.address_1.strip(", ").strip(),
        "address_2": r.address_2,
        "city": r.city,
        "state": r.state,
        "pin_code": r.pin_code,
        "_confidence": r.confidence,
    }


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
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass

    doc = yaml.safe_load(_CASES_FILE.read_text(encoding="utf-8"))
    cases = doc["cases"]

    tally = {"correct": 0, "wrong": 0, "missed": 0, "hallucinated": 0, "correctly_absent": 0}
    conf_tally: dict[str, int] = {"high": 0, "medium": 0, "low": 0}
    failing_cases: list[str] = []
    # cases the resolver got fully right but flagged low-confidence, and the
    # reverse -- a wrong/missed field the resolver reported as high. Both are
    # calibration problems worth seeing even though neither changes the score.
    miscalibrated: list[str] = []

    print("=" * 78)
    print(f"ADDRESS SEGMENTATION  --  {len(cases)} cases, {len(_FIELDS)} fields each")
    print("=" * 78)

    for case in cases:
        cid = case["id"]
        got = _split(case["address"])
        want = case["expect"]

        results = {f: _score_field(got.get(f, ""), want.get(f, "")) for f in _FIELDS}
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
            for f in _FIELDS:
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
