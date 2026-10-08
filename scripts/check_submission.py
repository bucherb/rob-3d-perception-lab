#!/usr/bin/env python3
"""Pre-submission check: are the required deliverables present and parseable?

    python scripts/check_submission.py --lab 1 --results results/

Checks FILE FORMAT, not correctness. It will not tell you your numbers are
wrong. It will tell you that tables.md is missing -- the kind of thing that
costs points on work you actually did.

Run it before you zip.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

REQUIRED = {
    1: ["tables.md"],
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lab", type=int, choices=[1], default=1)
    ap.add_argument("--results", type=Path, default=Path("results"))
    args = ap.parse_args()

    print(f"checking Lab {args.lab} deliverables in {args.results}/\n")
    if not args.results.exists():
        print(f"error: {args.results} does not exist", file=sys.stderr)
        return 2

    problems = []
    for name in REQUIRED[args.lab]:
        path = args.results / name
        if path.exists():
            print(f"  found    {name}  ({path.stat().st_size:,} bytes)")
        else:
            print(f"  MISSING  {name}")
            problems.append(f"missing required deliverable: {name}")

    if not (args.results / "manifest.json").exists():
        print("\n  note: no manifest.json. Not required, but it is where your run "
              "conditions live, and every quantitative claim needs them.")

    print()
    if problems:
        print(f"{len(problems)} problem(s):")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("deliverables look well formed. This says nothing about whether your "
          "numbers are right.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
