"""Lab 1 Part B -- Learned segmentation.  [25 pts]

B1 permits starting from released weights, and says: if you do, SAY SO and
report both fine-tuned and zero-shot numbers. Do that. A fine-tuned number
presented as if trained from scratch is the kind of thing that unravels in
office hours.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SegmentationResult:
    """Predictions plus everything the summary table needs."""

    pred_labels: np.ndarray            # (N,) per-point predicted class
    gt_labels: np.ndarray              # (N,) per-point ground truth, -1 = ignore
    inference_latency_ms: float
    peak_memory_mb: float
    voxel_size_m: float
    from_pretrained: bool = False      # B1: you must report this

    def metrics(self, num_classes: int) -> dict:
        """PROVIDED. mIoU and per-class IoU via the shared metrics module."""
        from rob498.metrics import confusion_matrix, iou_from_confusion

        cm = confusion_matrix(self.pred_labels, self.gt_labels, num_classes)
        return iou_from_confusion(cm)

    def subset_miou(self, class_ids: list[int], num_classes: int) -> float:
        """PROVIDED. mIoU restricted to the classes your Part E task needs. [B3]

        B3 asks whether mIoU was a good proxy for what you actually needed. Run
        this with your task's classes and compare against the full mIoU. If they
        diverge -- and they usually do, because mIoU is dominated by large easy
        classes like floor and wall -- that divergence IS the answer to B3.
        """
        iou = self.metrics(num_classes)["per_class_iou"]
        sel = iou[class_ids]
        return float(np.nanmean(sel))


def voxelize(points: np.ndarray, voxel_size: float):
    """PROVIDED. Quantize points to a voxel grid.

    Returns (coords (M, 3) int32, inverse (N,) int) where `inverse` maps each
    original point to its voxel, so you can push per-voxel predictions back to
    per-point labels for evaluation.

    Evaluate at POINT level, not voxel level. Voxel-level mIoU at 10 cm looks
    much better than it is, and B2's whole finding -- which classes degrade
    fastest as resolution coarsens -- disappears if you evaluate in the same
    coarsened space you predicted in.
    """
    pts = np.asarray(points, dtype=float)
    coords = np.floor(pts / voxel_size).astype(np.int32)
    uniq, inverse = np.unique(coords, axis=0, return_inverse=True)
    # numpy 2.0.0 returned `inverse` as (N, 1) when `axis` is given; flatten so
    # the documented (N,) shape holds on every numpy version.
    return uniq, inverse.reshape(-1)


# --------------------------------------------------------------------------- #
# YOU WRITE THESE
# --------------------------------------------------------------------------- #
def build_model(architecture: str = "minkowski", num_classes: int = 20, **kwargs):
    """Build a sparse-conv net (MinkowskiNet) or point transformer (PTv3). [B1]"""
    raise NotImplementedError("Lab 1 B1: build the segmentation model")


def train(model, train_loader, val_loader, epochs: int = 50, **kwargs):
    """Train and return the fitted model plus a training log.  [B1, 12 pts]

    Log per-epoch train and val mIoU and keep the curve for your report. If val
    mIoU peaks at epoch 12 and decays after, report the peak and say you
    early-stopped -- do not quietly report the last epoch.
    """
    raise NotImplementedError("Lab 1 B1: train the segmentation model")


def evaluate(model, loader, voxel_size: float) -> SegmentationResult:
    """Run inference and package a SegmentationResult.  [B1]

    Measure latency and peak GPU memory here rather than estimating them later:
    ``torch.cuda.max_memory_allocated`` after ``reset_peak_memory_stats``, and
    ``torch.cuda.synchronize()`` before you stop the clock. Without the
    synchronize you are timing kernel launches, not kernels, and the number will
    be absurdly good.
    """
    raise NotImplementedError("Lab 1 B1: evaluate the segmentation model")


def voxel_size_ablation(models, loader, voxel_sizes=(0.02, 0.05, 0.10)) -> list:
    """Ablate voxel size; plot mIoU and latency against it.  [B2, 7 pts]

    `models` maps each voxel size to a model TRAINED at that voxel size. Running
    one model at voxel sizes it was not trained at measures train/test mismatch
    -- its kernels' receptive field in metres changes with the voxel -- rather
    than what the resolution costs, so train once per size. Returns one record
    per voxel size: mIoU, per-class IoU, and latency.

    B2 asks which classes degrade fastest as resolution coarsens and what they
    have in common. Sort per-class IoU by the drop from 2 cm to 10 cm and look
    at the top of that list before you write the explanation -- the answer is a
    property of the objects, and it will be obvious once you see them together.
    """
    raise NotImplementedError("Lab 1 B2: implement the voxel-size ablation")
