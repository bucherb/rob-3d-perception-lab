"""Lab 1 Part D -- Feed-forward reconstruction (VGGT).  [15 pts]

D1 is explicit: supply NO POSES AND NO INTRINSICS. The point of the comparison
is what a feed-forward model recovers from images alone. Feeding it poses makes
its numbers incomparable to the premise of the part.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class FeedForwardResult:
    """What the model returns, plus the cost of getting it.

    Everything here is in the MODEL'S frame and scale -- for VGGT, the first
    input camera's frame, in units that are not metres. Map it into the world
    with the Sim(3) from :func:`align_to_ground_truth` before comparing it with
    anything metric, and say in the report that you did.
    """

    points: np.ndarray                 # (N, 3) predicted geometry, model frame
    poses: np.ndarray                  # (V, 4, 4) predicted camera-to-world, OpenCV axes
    intrinsics: np.ndarray | None      # (V, 3, 3) if predicted
    n_views: int
    wall_clock_s: float
    peak_gpu_mb: float
    confidence: np.ndarray | None = None   # (N,) if the model provides it
    view_ids: list | None = None       # which input frames, in order -- D2 needs
                                       # them to pair predicted poses with GT poses

    def summary(self) -> dict[str, float]:
        return {"n_views": self.n_views, "n_points": len(self.points),
                "wall_clock_s": self.wall_clock_s, "peak_gpu_mb": self.peak_gpu_mb}

    def memory_mb(self) -> float:
        """PROVIDED. Resident bytes of the reconstruction a consumer keeps."""
        arrays = [self.points, self.poses]
        arrays += [a for a in (self.intrinsics, self.confidence) if a is not None]
        return sum(np.asarray(a).nbytes for a in arrays) / 1024 ** 2


def apply_sim3(S: np.ndarray, points: np.ndarray) -> np.ndarray:
    """PROVIDED. Apply a 4x4 similarity ``[[s R, t], [0, 1]]`` to (N, 3) points."""
    S = np.asarray(S, dtype=float)
    return np.asarray(points, dtype=float) @ S[:3, :3].T + S[:3, 3]


def align_to_ground_truth(pred_poses: np.ndarray, gt_poses: np.ndarray) -> dict:
    """PROVIDED. Sim(3)-align predicted poses to ground truth.  [D2, 5 pts]

    Both arguments are (V, 4, 4) camera-to-world poses for the SAME views in the
    same order. VGGT returns world-to-camera extrinsics (OpenCV "camera from
    world"); invert them before calling this.

    Returns ATE, rotation error in degrees, the recovered SCALE FACTOR, and
    ``sim3``: the (4, 4) similarity mapping the model's frame onto the world.
    Use the same ``sim3`` for the D3 geometry and the Part E queries, so that all
    three are evaluated under one alignment.

    The alignment uses camera centres only (Umeyama), so it needs at least three
    views that are not collinear; a camera path along a straight line leaves the
    rotation about that line undetermined, and the numbers will look worse than
    the reconstruction is.

    D2 asks you to note the scale factor and what it means for a robot that
    needs metric distances. The honest reading: a scale recovered by aligning to
    ground truth is not available to a robot that has no ground truth, so a
    method whose output is only correct up to an unknown scale has not solved
    the metric-reconstruction problem no matter how good its aligned ATE looks.
    Say that in your own words, with your number attached.
    """
    from rob498.metrics import ate_rmse, umeyama_alignment
    from rob498.se3 import rotation_angle

    pred = np.asarray(pred_poses, dtype=float)
    gt = np.asarray(gt_poses, dtype=float)
    if pred.shape != gt.shape or pred.ndim != 3 or pred.shape[1:] != (4, 4):
        raise ValueError(f"need matching (V, 4, 4) pose arrays, got {pred.shape}, {gt.shape}")
    if len(pred) < 3:
        raise ValueError("Sim(3) alignment needs at least 3 views")
    S, scale = umeyama_alignment(pred[:, :3, 3], gt[:, :3, 3], with_scale=True)
    R_align = S[:3, :3] / scale
    ate = ate_rmse(pred, gt, align=True, with_scale=True)
    # Rotation error AFTER alignment: the predicted rotations live in the model's
    # frame, so comparing them to GT without R_align measures the frame offset.
    rot = [np.degrees(rotation_angle((R_align @ p[:3, :3]).T @ g[:3, :3]))
           for p, g in zip(pred, gt)]
    return {
        "ate_rmse_m": ate["rmse"],
        "ate_median_m": ate["median"],
        "scale_factor": float(scale),
        "mean_rotation_error_deg": float(np.mean(rot)),
        "median_rotation_error_deg": float(np.median(rot)),
        "n_views": len(pred),
        "sim3": S,
    }


# --------------------------------------------------------------------------- #
# YOU WRITE THESE
# --------------------------------------------------------------------------- #
def run_vggt(images, n_views: int | None = None, **kwargs) -> FeedForwardResult:
    """Run VGGT on the images, with no poses and no intrinsics.  [D1, 5 pts]

    Return a :class:`FeedForwardResult` with camera-to-world ``poses``. VGGT's
    ``pose_encoding_to_extri_intri`` returns world-to-camera extrinsics, 3x4,
    OpenCV axes ("camera from world"): pad to 4x4 and invert. Record which
    frames you used in ``view_ids``.

    D1 asks how wall-clock and peak memory SCALE as you vary the number of input
    views. Run at least four view counts and plot it. The scaling is the
    interesting quantity -- attention over views is not linear, and the point at
    which the model stops fitting in memory is a deployment fact about the
    method, not an incidental detail of your GPU.
    """
    raise NotImplementedError("Lab 1 D1: run the feed-forward reconstruction")


def evaluate_geometry(result: FeedForwardResult, gt_mesh_points: np.ndarray,
                      threshold: float = 0.05, sim3: np.ndarray | None = None):
    """Accuracy and completeness against ground truth.  [D3, 5 pts]

    Same metrics and same threshold as C3 (``--recon-threshold``), so the splat
    and feed-forward numbers are comparable. `sim3` is
    ``align_to_ground_truth(...)["sim3"]``: the predicted points are in the
    model's frame and scale, and comparing them to the GT mesh unaligned
    measures the frame offset, not the geometry. Returns a
    ``rob498.metrics.ReconResult``.

    D3 asks you to IDENTIFY THE REGIONS WHERE IT FAILS AND CHARACTERIZE THEM --
    textureless, specular, thin structure, or beyond the view distribution.
    ``ReconResult.dist_pred_to_gt`` gives you per-point error; colour the cloud
    by it, look at the worst decile, and name what those regions have in common.
    A histogram alone does not answer this; you have to look at where the errors
    are.
    """
    raise NotImplementedError("Lab 1 D3: evaluate the feed-forward geometry")
