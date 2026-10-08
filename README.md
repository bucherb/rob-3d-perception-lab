# ROB 498/599 — 3D Robot Perception: lab starter code

Starter code for the lab. Clone it, make the environment, and start filling in
the stubs.

```bash
git clone https://github.com/bucherb/rob-3d-perception-lab.git && cd rob-3d-perception-lab
conda env create -f environment.yml && conda activate rob498
pip install -e .
python scripts/check_env.py            # on a GPU node: python scripts/check_env.py --gpu
pytest rob498/tests lab1/tests -v
```

## What this is, and what it deliberately is not

**Provided:** the shared `rob498/` package — SE(3) conventions, evaluation
metrics, file I/O in the exact deliverable formats, and the summary-table
builders. Plus, in `lab1/`, the plumbing around the algorithms: data
structures, the runner CLI, and helpers that are not the thing being assessed.

**Not provided:** the algorithms. Every function you are graded on raises
`NotImplementedError` with its part number and point value in the docstring.

The split follows the course policy: *you may not import a library that
implements the core algorithm you were asked to write.* So the TSDF baseline in
Lab 1 E2 may be fused with Open3D because it is a baseline, and the occupancy
query on it is yours.

## Why use the provided metrics rather than your own

Two reasons, both of which have cost students points in the past.

**Comparability.** "Accuracy" and "completeness" have several definitions in the
reconstruction literature and the numbers do not transfer between them. Lab 1
compares a splat surface (C3) with a feed-forward reconstruction (D3) *at one
fixed threshold*. That comparison is meaningless if the two used different
conventions, so `rob498.metrics.accuracy_completeness` fixes one and documents it.

**Traps that are easy to miss.** `ate_rmse` aligns before measuring, because
an unaligned ATE mostly measures how two world frames differ.
`iou_from_confusion` returns NaN for classes absent from both prediction and
truth rather than 0.0, which otherwise silently drags mIoU down. `read_tum`
matches trajectories *by timestamp*, because comparing by index produces a
plausible-looking wrong number.

If you think a provided metric is wrong, say so in your report with evidence.
That is a legitimate finding and it is graded as one.

## Layout

```
rob498/                 shared, provided
  se3.py                conventions, helpers, exp/log/adjoint -- all provided
  metrics.py            ATE, RPE, accuracy/completeness, IoU, binary rates, bootstrap CI
  io3d.py               TUM trajectories, depth back-projection, PLY, JSON, mesh sampling
  report.py             the required summary tables
  config.py             seeding + RunManifest (captures your run conditions)

  tests/                must pass from day one: `pytest rob498/tests`

lab1/  dataset.py (provided loader)
       segmentation.py splatting.py feedforward.py downstream.py  run_lab1.py
```

`lab1/tests/test_contracts.py` checks your code against the conventions. Run it early and often.

## What you implement

Every function below raises `NotImplementedError` until you write it. Its
docstring gives the part, the points, and what it must return. The handout's
point values are per part, and a part's report questions are graded alongside
its code.

| Part | File | Implement | Provided, use these |
|---|---|---|---|
| B1 (12) | `lab1/segmentation.py` | `build_model`, `train`, `evaluate` | `voxelize`, `SegmentationResult.metrics` |
| B2 (7) | `lab1/segmentation.py` | `voxel_size_ablation` | |
| B3 (6) | `lab1/segmentation.py` | nothing new; report questions | `SegmentationResult.subset_miou` |
| C1 (8) | `lab1/splatting.py` | `fit_splats` | `SplatModel` (with `validate`, `memory_mb`), `psnr` |
| C2 (6) | `lab1/splatting.py` | `extract_surface` | `filter_floaters` |
| C3 (6) | `lab1/splatting.py` | `evaluate_geometry` | `rob498.metrics` |
| D1 (5) | `lab1/feedforward.py` | `run_vggt` (no poses, no intrinsics) | `FeedForwardResult` |
| D2 (5) | `lab1/feedforward.py` | nothing new; report questions | `align_to_ground_truth`, `apply_sim3` |
| D3 (5) | `lab1/feedforward.py` | `evaluate_geometry` (takes D2's `sim3`) | `rob498.metrics` |
| E1 (8) | `lab1/downstream.py` | `label_ground_truth_occupancy` | `sample_configurations` |
| E2 (18) | `lab1/downstream.py` | `__init__`, `is_occupied` and `memory_mb` on each of `SegmentationOccupancy`, `SplatOccupancy`, `FeedForwardOccupancy`, and the `TSDFOccupancy` baseline | `evaluate_representations`, `time_queries` (the fixed query harness; do not modify) |
| E3 (10) | `lab1/downstream.py` | nothing new; explain the mechanism | `find_ranking_flip` |
| E4 (4) | report only | | |

You also wire the parts together in `_run_part` in `lab1/run_lab1.py`, which
raises `NotImplementedError` with a list of what each part calls. That is
plumbing and earns no points on its own, but it is how you produce the
numbers in your report:

```bash
python lab1/run_lab1.py --data data/lab1 --out results/ --part b   # or c, d, e, all
```

## The contract tests

```bash
pytest lab1/tests -v
```

They check **conventions and shapes, not correctness**. They cannot tell you
your occupancy query is right. They can tell you that your poses are inverted,
that a splat model stores log-scales where metres are expected, or that a
query returns the wrong shape — which is where people lose points on work they actually did.
`pytest rob498/tests` covers the provided package and must pass before you start.

Tests for functions you have not written yet **skip** rather than fail, so the
suite stays useful throughout.

## Conventions, in one place

Read the `rob498/se3.py` docstring before writing any code. The short version:

- A pose `T_wc` maps points **from** the child frame **into** the parent:
  `p_w = T_wc @ p_c`.
- Twists are ordered **`(v, omega)` — translation first.** Much of the
  literature uses the opposite, and so does GTSAM. Mix them and an adjoint comes
  out silently block-transposed.
- Perturbations are right: `T = T_bar @ exp(xi)`.
- Metres, radians, right-handed.
- Quaternions are `(qx, qy, qz, qw)` — **scalar last** — because that is what
  the TUM format specifies.

## Reporting numbers

The handout is blunt about this: a row reading `VGGT: 0.03` is worth nothing; one
that names the sequence, the trial count, the seed, and the hardware is worth
full marks even if the number is bad.

`RunManifest` captures all of that automatically:

```python
from rob498.config import RunManifest
m = RunManifest(run_name="vggt_scene0", seed=0)
m.add_param("n_frames", 32)
m.add_result("ate_rmse_m", 0.031)
m.save("results/vggt_scene0")
print(m.conditions_line())
# seed=0, n_frames=32, NVIDIA A40
```

`Table.to_markdown()` **refuses to render without conditions set.** That is
deliberate. Pass `strict=False` only for a table that genuinely has no run
conditions.

## Environments

One `environment.yml` covers the lab, on Great Lakes and on a laptop. Conda
provides only Python; every package is installed by pip, with the versions
pinned that matter for reproducibility (torch 2.5.1, open3d 0.18.0, gsplat
1.5.3, a fixed VGGT commit).

**Great Lakes (where the graded runs happen).** Parts B, C and D need a GPU with
≥16 GB. Create the environment once on a login node, then on a GPU node:

```bash
module load cuda/12.x                 # any 12.x -- `module avail cuda`; needed once, for gsplat
conda activate rob498
python scripts/check_env.py --gpu     # first run compiles gsplat's kernels: several minutes
```

The torch wheel carries its own CUDA runtime, so the CUDA module matters only
for that one gsplat compile. Colab will not carry a 3DGS fit or a long VGGT
forward pass, and you will lose a day discovering that.

**Laptop (macOS on Apple silicon, or Linux).** The same commands work; gsplat
and spconv are Linux + CUDA only and are skipped on macOS. Every contract test,
the data loader, and the Part E queries run on CPU, so you can develop there and
send the heavy runs to Great Lakes. Intel Macs are not supported (PyTorch no
longer publishes wheels for them).

**VGGT weights.** Part D downloads the 1B-parameter checkpoint (~5 GB) from
Hugging Face on first use. If a compute node cannot reach Hugging Face, download
it once on a login node (`hf download facebook/VGGT-1B`) or save
`model.pt` somewhere and `export VGGT_WEIGHTS=/path/to/model.pt`.

Run `lab1/fetch_data.py` **the day the lab is released**. The ScanNet++
data needs you to have accepted the dataset terms, which takes time to approve.
`lab1/fetch_data.py` downloads, verifies and unpacks to `data/lab1/`;
`lab1/dataset.py` documents the layout and loads it -- including the camera
convention (OpenCV axes, camera-to-world), which is the one ScanNet++'s own
`transforms.json` does *not* use.

## Generative AI

Permitted, per the course policy. You are responsible for everything you submit:
if you cannot explain a line of your code in office hours, it does not count as
yours. Note in your README where you used it substantially.

Worth saying plainly: these stubs are unusually detailed about *why* each piece
is hard, and an assistant can turn a detailed docstring into plausible code very
quickly. The parts of the lab that carry the most points — the mechanistic
explanation in Lab 1 C3 and the ranking flip in Lab 1 E3 — are the parts where plausible is worth nothing and only having looked
at your own results helps.

## Getting help

Bring the failing test, the command you ran, and the traceback. "Lab 1 doesn't
work" is hard to help with; `pytest lab1/tests -k floater` and its output is easy.
