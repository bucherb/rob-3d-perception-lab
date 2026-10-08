"""Evaluation metrics for the lab.

These are PROVIDED and you should use them rather than writing your own. Two
reasons. First, "accuracy" and "completeness" have at least three definitions in
the reconstruction literature and the numbers are not comparable across them --
Lab 1 compares a splat surface (C3) against a feed-forward reconstruction (D3) at
one threshold, and that comparison is meaningless if the two used different
conventions. Second,
ATE has a well-known alignment subtlety (see :func:`ate_rmse`) that costs people
points every year.

If you believe a metric here is wrong, say so in your report with evidence. That
is a legitimate finding and it is graded as one.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rob498.se3 import inverse, relative, rotation_angle


# --------------------------------------------------------------------------- #
# Trajectory metrics
# --------------------------------------------------------------------------- #
def umeyama_alignment(
    src: np.ndarray, dst: np.ndarray, with_scale: bool = False
) -> tuple[np.ndarray, float]:
    """Least-squares similarity transform mapping `src` onto `dst`.

    Args:
        src: (N, 3) source points.
        dst: (N, 3) target points, corresponding row-wise to `src`.
        with_scale: if True solve Sim(3) and return the scale; if False solve
            SE(3) and return scale 1.0.

    Returns:
        (T, s) where T is (4, 4) and ``s * (R @ p) + t`` maps src onto dst.

    Provided because Lab 1 D2 needs trajectory alignment, which is evaluation
    rather than algorithm.
    """
    src = np.asarray(src, dtype=float)
    dst = np.asarray(dst, dtype=float)
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 3:
        raise ValueError(f"need matching (N, 3) arrays, got {src.shape}, {dst.shape}")
    n = src.shape[0]
    mu_s, mu_d = src.mean(axis=0), dst.mean(axis=0)
    sc, dc = src - mu_s, dst - mu_d
    Sigma = (dc.T @ sc) / n
    U, D, Vt = np.linalg.svd(Sigma)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1.0
    R = U @ S @ Vt
    if with_scale:
        var_s = (sc ** 2).sum() / n
        s = float(np.trace(np.diag(D) @ S) / var_s) if var_s > 0 else 1.0
    else:
        s = 1.0
    t = mu_d - s * (R @ mu_s)
    T = np.eye(4)
    T[:3, :3] = s * R
    T[:3, 3] = t
    return T, s


def ate_rmse(
    est: np.ndarray, gt: np.ndarray, align: bool = True, with_scale: bool = False
) -> dict[str, float]:
    """Absolute Trajectory Error between two pose sequences.

    Args:
        est: (N, 4, 4) estimated poses.
        gt: (N, 4, 4) ground-truth poses, time-aligned with `est`.
        align: if True, first remove the rigid (or similarity) transform between
            the two trajectories. **You almost always want this.** ATE is defined
            after alignment; an unaligned number mostly measures how the two
            world frames differ, not how good your odometry is.
        with_scale: solve Sim(3) instead of SE(3). Use for monocular or
            feed-forward methods with no metric scale -- Lab 1 D2 -- and NOT
            when a depth sensor gives you scale: hiding a scale error behind
            Sim(3) alignment is exactly the mistake to avoid.

    Returns:
        dict with rmse, mean, median, std, min, max (metres) and the applied
        scale.
    """
    est = np.asarray(est, dtype=float)
    gt = np.asarray(gt, dtype=float)
    if est.shape != gt.shape or est.ndim != 3 or est.shape[1:] != (4, 4):
        raise ValueError(f"need matching (N, 4, 4) arrays, got {est.shape}, {gt.shape}")
    p_est, p_gt = est[:, :3, 3], gt[:, :3, 3]
    scale = 1.0
    if align:
        T, scale = umeyama_alignment(p_est, p_gt, with_scale=with_scale)
        p_est = p_est @ T[:3, :3].T + T[:3, 3]
    err = np.linalg.norm(p_est - p_gt, axis=1)
    return {
        "rmse": float(np.sqrt((err ** 2).mean())),
        "mean": float(err.mean()),
        "median": float(np.median(err)),
        "std": float(err.std()),
        "min": float(err.min()),
        "max": float(err.max()),
        "scale": float(scale),
        "n": int(len(err)),
    }


def rpe(est: np.ndarray, gt: np.ndarray, delta: int = 1) -> dict[str, float]:
    """Relative Pose Error over a fixed frame gap.

    ATE is dominated by whether the trajectory drifted; RPE isolates local
    consistency and is the better diagnostic when you are debugging odometry.
    Report both when you evaluate a trajectory: they fail differently.

    Returns:
        dict with translation (metres) and rotation (degrees) RMSE.
    """
    est = np.asarray(est, dtype=float)
    gt = np.asarray(gt, dtype=float)
    n = len(est)
    if n <= delta:
        raise ValueError(f"need more than delta={delta} poses, got {n}")
    t_errs, r_errs = [], []
    for i in range(n - delta):
        d_est = relative(est[i], est[i + delta])
        d_gt = relative(gt[i], gt[i + delta])
        d = inverse(d_gt) @ d_est
        t_errs.append(np.linalg.norm(d[:3, 3]))
        r_errs.append(np.degrees(rotation_angle(d[:3, :3])))
    t_errs, r_errs = np.array(t_errs), np.array(r_errs)
    return {
        "trans_rmse_m": float(np.sqrt((t_errs ** 2).mean())),
        "rot_rmse_deg": float(np.sqrt((r_errs ** 2).mean())),
        "trans_median_m": float(np.median(t_errs)),
        "rot_median_deg": float(np.median(r_errs)),
        "delta": int(delta),
        "n": int(len(t_errs)),
    }


def final_drift(est: np.ndarray, gt: np.ndarray) -> float:
    """Final-position drift in metres, with both trajectories put at a common
    origin. A useful companion to ATE for any trajectory."""
    est = np.asarray(est, dtype=float)
    gt = np.asarray(gt, dtype=float)
    e = relative(est[0], est[-1])[:3, 3]
    g = relative(gt[0], gt[-1])[:3, 3]
    return float(np.linalg.norm(e - g))


# --------------------------------------------------------------------------- #
# Reconstruction metrics
# --------------------------------------------------------------------------- #
@dataclass
class ReconResult:
    """Accuracy/completeness at one threshold, with the distances kept.

    Keeping the raw distances matters: the handouts ask you to show *where* your
    reconstruction fails, and a single scalar cannot do that. Colour your mesh
    by ``dist_pred_to_gt`` and the failure regions are immediately visible.
    """

    accuracy_mean: float
    accuracy_median: float
    completeness_mean: float
    completeness_median: float
    precision: float     # fraction of predicted points within threshold of GT
    recall: float        # fraction of GT points within threshold of prediction
    f_score: float
    threshold: float
    dist_pred_to_gt: np.ndarray
    dist_gt_to_pred: np.ndarray

    def summary(self) -> dict[str, float]:
        return {
            "accuracy_mean_m": self.accuracy_mean,
            "accuracy_median_m": self.accuracy_median,
            "completeness_mean_m": self.completeness_mean,
            "completeness_median_m": self.completeness_median,
            "precision": self.precision,
            "recall": self.recall,
            "f_score": self.f_score,
            "threshold_m": self.threshold,
        }


def accuracy_completeness(
    pred_points: np.ndarray,
    gt_points: np.ndarray,
    threshold: float = 0.05,
) -> ReconResult:
    """Symmetric point-cloud comparison. USE THIS IN LABS 1 AND 2 BOTH.

    Definitions used here, stated explicitly because they vary in the literature:

      accuracy      mean/median distance from each PREDICTED point to its nearest
                    GT point. Low accuracy error means "what you built is real".
      completeness  mean/median distance from each GT point to its nearest
                    PREDICTED point. Low completeness error means "you built all
                    of it".
      precision     fraction of predicted points within `threshold` of GT.
      recall        fraction of GT points within `threshold` of a prediction.
      f_score       harmonic mean of precision and recall.

    A reconstruction that covers half the room perfectly scores well on accuracy
    and badly on completeness. One that fills the room with noise scores the
    reverse. Reporting only one of them hides the failure -- report both, which
    is why this returns both and why the summary table asks for both.

    Sample your mesh to points before calling this (``io3d.sample_mesh``);
    vertex-only sampling biases toward wherever your mesher happened to put
    vertices.
    """
    from scipy.spatial import cKDTree

    pred = np.asarray(pred_points, dtype=float)
    gt = np.asarray(gt_points, dtype=float)
    if pred.ndim != 2 or pred.shape[1] != 3 or len(pred) == 0:
        raise ValueError(f"pred_points must be a non-empty (N, 3) array, got {pred.shape}")
    if gt.ndim != 2 or gt.shape[1] != 3 or len(gt) == 0:
        raise ValueError(f"gt_points must be a non-empty (N, 3) array, got {gt.shape}")

    d_pred_gt, _ = cKDTree(gt).query(pred, k=1)
    d_gt_pred, _ = cKDTree(pred).query(gt, k=1)
    precision = float((d_pred_gt < threshold).mean())
    recall = float((d_gt_pred < threshold).mean())
    denom = precision + recall
    return ReconResult(
        accuracy_mean=float(d_pred_gt.mean()),
        accuracy_median=float(np.median(d_pred_gt)),
        completeness_mean=float(d_gt_pred.mean()),
        completeness_median=float(np.median(d_gt_pred)),
        precision=precision,
        recall=recall,
        f_score=float(2 * precision * recall / denom) if denom > 0 else 0.0,
        threshold=float(threshold),
        dist_pred_to_gt=d_pred_gt,
        dist_gt_to_pred=d_gt_pred,
    )


# --------------------------------------------------------------------------- #
# Segmentation and detection metrics
# --------------------------------------------------------------------------- #
def confusion_matrix(pred: np.ndarray, gt: np.ndarray, num_classes: int) -> np.ndarray:
    """(C, C) confusion matrix, rows = ground truth, columns = prediction.

    Labels outside [0, num_classes) are dropped -- that is how ScanNet-style
    ignore labels (-1 or 255) are meant to be handled. Do not remap them to a
    real class; you will inflate your mIoU.
    """
    pred = np.asarray(pred).ravel()
    gt = np.asarray(gt).ravel()
    valid = (gt >= 0) & (gt < num_classes) & (pred >= 0) & (pred < num_classes)
    idx = gt[valid].astype(np.int64) * num_classes + pred[valid].astype(np.int64)
    return np.bincount(idx, minlength=num_classes ** 2).reshape(num_classes, num_classes)


def iou_from_confusion(cm: np.ndarray) -> dict[str, np.ndarray | float]:
    """Per-class IoU and mIoU from a confusion matrix.

    Classes absent from BOTH prediction and ground truth get NaN, not zero, and
    mIoU is a nanmean. Scoring an absent class as 0.0 silently drags mIoU down
    and is a common source of unreproducible numbers -- Lab 1 B1 asks for
    per-class IoU precisely so this is visible.
    """
    cm = np.asarray(cm, dtype=float)
    tp = np.diag(cm)
    union = cm.sum(axis=1) + cm.sum(axis=0) - tp
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = np.where(union > 0, tp / union, np.nan)
    return {
        "per_class_iou": iou,
        "miou": float(np.nanmean(iou)),
        "pixel_acc": float(tp.sum() / cm.sum()) if cm.sum() > 0 else 0.0,
    }


def binary_rates(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    """TP/FP/FN/TN rates for a boolean prediction. Lab 1 Part E uses this.

    E1 asks you to state what a false negative costs a real robot, and the two
    error types are not symmetric -- for collision checking, a false negative is
    a collision and a false positive is a detour. Report both rates separately;
    a single accuracy number hides the one that matters.
    """
    pred = np.asarray(pred).astype(bool).ravel()
    gt = np.asarray(gt).astype(bool).ravel()
    if pred.shape != gt.shape:
        raise ValueError(f"shape mismatch: {pred.shape} vs {gt.shape}")
    tp = int((pred & gt).sum())
    fp = int((pred & ~gt).sum())
    fn = int((~pred & gt).sum())
    tn = int((~pred & ~gt).sum())
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "false_positive_rate": fp / max(fp + tn, 1),
        "false_negative_rate": fn / max(fn + tp, 1),
        "precision": tp / max(tp + fp, 1),
        "recall": tp / max(tp + fn, 1),
        "accuracy": (tp + tn) / max(len(pred), 1),
    }


def bootstrap_ci(
    values: np.ndarray, n_boot: int = 10_000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval for the mean.

    The handouts ask for confidence intervals on several numbers. With 20 trials
    a normal-approximation interval is not trustworthy; this makes no
    distributional assumption. Report the interval, not just the mean.
    """
    v = np.asarray(values, dtype=float).ravel()
    if len(v) == 0:
        raise ValueError("empty array")
    rng = np.random.default_rng(seed)
    means = rng.choice(v, size=(n_boot, len(v)), replace=True).mean(axis=1)
    return float(np.percentile(means, 100 * alpha / 2)), float(
        np.percentile(means, 100 * (1 - alpha / 2))
    )
