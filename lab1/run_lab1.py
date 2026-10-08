#!/usr/bin/env python3
"""Lab 1 driver.

    python lab1/run_lab1.py --data data/lab1 --out results/ --part all

`--data` is the directory fetch_data.py unpacked; lab1/dataset.py documents its
layout and loads it.

Produces results/tables.md containing the representation-comparison table that
the handout calls "the centre of the report", plus manifest.json.

Part E needs B, C, and D to have run -- it compares them. Run the earlier parts
first and cache their outputs to --out.
"""
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rob498 import report                          # noqa: E402
from rob498.config import RunManifest              # noqa: E402

PARTS = ["b", "c", "d", "e"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("results"))
    ap.add_argument("--part", choices=[*PARTS, "all"], default="all")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--recon-threshold", type=float, default=0.05,
                    help="One threshold for C3 and D3, so the splat and "
                         "feed-forward numbers are comparable.")
    ap.add_argument("--robot-radius", type=float, default=0.15,
                    help="Part E: radius (m) of the spherical "
                         "robot, the same for every representation and the GT.")
    args = ap.parse_args()

    if not args.data.exists():
        print(f"error: {args.data} does not exist. Run fetch_data.py. Note the "
              f"ScanNet++ subset needs you to have accepted the dataset terms, "
              f"which takes time to approve.", file=sys.stderr)
        return 2

    args.out.mkdir(parents=True, exist_ok=True)
    manifest = RunManifest(run_name="lab1", seed=args.seed)
    manifest.add_param("recon_threshold_m", args.recon_threshold)
    manifest.add_param("robot_radius_m", args.robot_radius)

    table = report.lab1_summary_table()
    parts = PARTS if args.part == "all" else [args.part]
    completed, failed = [], []

    for part in parts:
        print(f"\n=== Part {part.upper()} " + "=" * 50)
        try:
            _run_part(part, args, manifest, table)
            completed.append(part)
        except NotImplementedError as e:
            print(f"  not implemented: {e}")
            failed.append(part)
        except Exception:
            print("  FAILED with an exception:")
            traceback.print_exc()
            failed.append(part)

    manifest.save(args.out)
    table.conditions = manifest.conditions_line()
    report.write_report_bundle(args.out, [table])
    print("\n" + "=" * 62)
    print(f"completed: {completed or 'none'}   outstanding: {failed or 'none'}")
    print("\nReminder: three pipelines and a PSNR table is the failure mode this "
          "lab is built around. Part E is 40 points and it asks a different "
          "question than Parts B-D do.")
    return 0


def _run_part(part, args, manifest, table) -> None:
    if part == "b":
        import segmentation  # noqa: F401
        raise NotImplementedError(
            "Part B: build_model, train, evaluate, voxel_size_ablation. Report "
            "both fine-tuned and zero-shot numbers if you started from weights.")
    if part == "c":
        import splatting  # noqa: F401
        raise NotImplementedError(
            "Part C: fit_splats on held-out views, extract_surface, "
            "evaluate_geometry at --recon-threshold.")
    if part == "d":
        import feedforward  # noqa: F401
        raise NotImplementedError(
            "Part D: run_vggt with NO poses and NO intrinsics, "
            "align_to_ground_truth, evaluate_geometry with its sim3.")
    if part == "e":
        import downstream  # noqa: F401
        raise NotImplementedError(
            "Part E: implement SegmentationOccupancy, SplatOccupancy, "
            "FeedForwardOccupancy and the TSDFOccupancy baseline, "
            "label_ground_truth_occupancy, then evaluate_representations -- "
            "which holds the query procedure fixed, as E2 requires. Finish with "
            "find_ranking_flip and explain the mechanism.")


if __name__ == "__main__":
    raise SystemExit(main())
