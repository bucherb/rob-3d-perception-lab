"""Lab 1 Part C -- Per-scene optimization (3D Gaussian Splatting).  [20 pts]

You are not asked to write a rasterizer. Use an existing 3DGS implementation and
record its version and commit -- C1's numbers are not reproducible without it.

C3 is the part that carries the lab's argument: report accuracy and completeness
against ground truth AT THE LAB'S FIXED THRESHOLD (``--recon-threshold``, the same
one D3 uses for the feed-forward reconstruction) so the numbers are comparable, then comment on the relationship between your PSNR and your
geometric accuracy. "They are not the same ordering, and the gap is the
finding."
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SplatModel:
    """A fitted 3DGS model, in the course conventions.

    Attributes:
        means: (N, 3) Gaussian centres, world frame, metres.
        scales: (N, 3) per-axis standard deviations in METRES (linear, > 0).
        rotations: (N, 4) unit quaternions ``(qx, qy, qz, qw)``, scalar LAST,
            mapping the Gaussian's local axes into the world frame.
        opacities: (N,) in [0, 1] (after the sigmoid).
        sh_coeffs: (N, K, 3) spherical-harmonic colour coefficients.

    Rasterizers store something else: gsplat and the reference 3DGS code keep
    LOG scales, LOGIT opacities, and SCALAR-FIRST ``(w, x, y, z)`` quaternions.
    Convert when you build this object -- ``exp``, ``sigmoid``, and a roll of
    the quaternion -- because every provided function, including
    :func:`filter_floaters`, reads these fields in the units above.
    :meth:`validate` catches the common slip of passing raw log-scales.
    """

    means: np.ndarray
    scales: np.ndarray
    rotations: np.ndarray
    opacities: np.ndarray
    sh_coeffs: np.ndarray
    fit_time_s: float = float("nan")
    peak_gpu_mb: float = float("nan")

    @property
    def n_gaussians(self) -> int:
        return len(self.means)

    def validate(self) -> None:
        """PROVIDED. Raise if a field is not in the units documented above."""
        n = len(self.means)
        shapes = {"means": (self.means, (n, 3)), "scales": (self.scales, (n, 3)),
                  "rotations": (self.rotations, (n, 4)), "opacities": (self.opacities, (n,))}
        for name, (a, want) in shapes.items():
            if np.shape(a) != want:
                raise ValueError(f"SplatModel.{name} has shape {np.shape(a)}, expected {want}")
        if np.shape(self.sh_coeffs)[:1] != (n,) or np.ndim(self.sh_coeffs) != 3:
            raise ValueError(f"SplatModel.sh_coeffs has shape {np.shape(self.sh_coeffs)}, "
                             f"expected ({n}, K, 3)")
        if n and np.min(self.scales) <= 0:
            raise ValueError("SplatModel.scales must be linear standard deviations in metres "
                             "(> 0). Negative values mean log-scales were passed: apply exp().")
        if n and (np.min(self.opacities) < 0 or np.max(self.opacities) > 1):
            raise ValueError("SplatModel.opacities must be in [0, 1]. Values outside mean "
                             "logits were passed: apply a sigmoid.")
        if n and not np.allclose(np.linalg.norm(self.rotations, axis=1), 1.0, atol=1e-3):
            raise ValueError("SplatModel.rotations must be unit quaternions (qx, qy, qz, qw).")

    def memory_mb(self) -> float:
        """PROVIDED. Resident bytes of the fitted parameters held here.

        This counts the arrays a downstream consumer must keep -- means, scales,
        rotations, opacities, SH coefficients -- at their stored dtype, which is
        the representation's cost at query time (Part E). It deliberately
        excludes training-time state (Adam moments, gradient accumulators,
        densification counters), which can be several times larger: report that
        as ``peak_gpu_mb`` from the fit, and say which number the memory column
        of your summary table holds.
        """
        return sum(np.asarray(a).nbytes for a in
                   (self.means, self.scales, self.rotations,
                    self.opacities, self.sh_coeffs)) / 1024 ** 2

    def summary(self) -> dict[str, float]:
        return {"n_gaussians": self.n_gaussians, "memory_mb": self.memory_mb(),
                "fit_time_s": self.fit_time_s, "peak_gpu_mb": self.peak_gpu_mb}


def psnr(pred: np.ndarray, gt: np.ndarray, max_val: float = 1.0) -> float:
    """PROVIDED. PSNR in dB between two images in [0, max_val]."""
    pred = np.asarray(pred, dtype=float)
    gt = np.asarray(gt, dtype=float)
    if pred.shape != gt.shape:
        raise ValueError(f"shape mismatch: {pred.shape} vs {gt.shape}")
    mse = float(np.mean((pred - gt) ** 2))
    return float("inf") if mse == 0 else float(10 * np.log10(max_val ** 2 / mse))


def filter_floaters(
    model: SplatModel, opacity_threshold: float = 0.1,
    max_scale_m: float = 0.5, k_neighbors: int = 8, max_neighbor_dist_m: float = 0.3
) -> np.ndarray:
    """PROVIDED. A defensible floater filter, returning a keep mask.

    Removes Gaussians that are (a) nearly transparent, (b) implausibly large, or
    (c) isolated from any neighbour. Provided so that everyone starts from the
    same baseline -- but C2 requires you to DOCUMENT THE PROCEDURE PRECISELY, so
    if you tune these thresholds, report the values you used and what changed.

    A splat fit always produces floaters. They barely affect PSNR, because they
    are nearly invisible from the training views, and they wreck geometric
    accuracy and collision checking. That asymmetry is the mechanism behind the
    ranking flip Part E3 asks you to find -- so do not silently tune them away
    and then report that no flip exists.
    """
    from scipy.spatial import cKDTree

    model.validate()          # linear scales and [0, 1] opacities, or this is meaningless
    keep = model.opacities > opacity_threshold
    keep &= np.max(model.scales, axis=1) < max_scale_m
    if keep.sum() > k_neighbors:
        pts = model.means[keep]
        d, _ = cKDTree(pts).query(pts, k=min(k_neighbors + 1, len(pts)))
        isolated = d[:, 1:].mean(axis=1) > max_neighbor_dist_m
        idx = np.where(keep)[0]
        keep[idx[isolated]] = False
    return keep


# --------------------------------------------------------------------------- #
# YOU WRITE THESE
# --------------------------------------------------------------------------- #
def fit_splats(images, poses, intrinsics, n_iterations: int = 30_000, **kwargs) -> SplatModel:
    """Fit 3DGS to the evaluation scene.  [C1, 8 pts]

    `images` (V, H, W, 3) in [0, 1], `poses` (V, 4, 4) camera-to-world with
    OpenCV axes (``dataset.EvalScene`` provides both), `intrinsics` (3, 3). Pass the
    TRAINING views only. gsplat's ``rasterization`` wants world-to-camera
    ``viewmats``: invert the poses with ``rob498.se3.inverse``.

    C1 wants PSNR, SSIM, and LPIPS on the HELD-OUT views, the final Gaussian
    count, wall-clock fit time, and peak GPU memory. Held-out means held out --
    if the fit ever saw those views, every number in your report is optimistic
    and the comparison against the other two representations is void.
    """
    raise NotImplementedError("Lab 1 C1: fit the 3DGS model")


def extract_surface(model: SplatModel, method: str = "opacity_threshold", **kwargs):
    """Extract a surface or occupancy estimate from the splats.  [C2, 6 pts]

    Returns (vertices, faces) or a point set -- state which and why.

    C2 requires the procedure documented precisely: the opacity threshold, how
    you treat ANISOTROPIC Gaussians, and how you handle floaters. The
    anisotropic question is the substantive one: a Gaussian that is long and
    thin is a legitimate surface element viewed edge-on, or an artefact
    stretched along a viewing ray, and distinguishing them is the whole problem.
    Whatever rule you adopt, state it as a predicate on the scales and show what
    it removes.
    """
    raise NotImplementedError("Lab 1 C2: extract a surface from the splat model")


def evaluate_geometry(surface_points: np.ndarray, gt_mesh_points: np.ndarray,
                      threshold: float = 0.05):
    """Accuracy and completeness against ground truth.  [C3, 6 pts]

    Call ``rob498.metrics.accuracy_completeness`` with the lab's fixed threshold
    (``--recon-threshold``), the same one D3 uses -- the point of holding it
    fixed is that the splat and feed-forward numbers become comparable -- and
    return its ``ReconResult``.

    Then answer the actual question: how does your PSNR ordering relate to your
    geometric accuracy ordering? If the model with the best PSNR does not have
    the best accuracy, say why. The mechanism is usually that view synthesis is
    rewarded for reproducing appearance from the training viewpoints and is
    indifferent to where the geometry actually sits along the ray.
    """
    raise NotImplementedError("Lab 1 C3: evaluate the extracted geometry")
