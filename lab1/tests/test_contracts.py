"""Contract tests for Lab 1. Shapes and conventions, not correctness.

    pytest lab1/tests -v

Tests of a function you have not written yet SKIP (they catch the
NotImplementedError) rather than fail. Once you implement it, its contract test
runs.

Staff: ``ROB498_LAB1_IMPL=/path/to/solutions/lab1 pytest lab1/tests`` runs the
same contracts against another implementation of the stubs.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "lab1"))
if os.environ.get("ROB498_LAB1_IMPL"):
    sys.path.insert(0, str(Path(os.environ["ROB498_LAB1_IMPL"]).resolve()))

from rob498 import metrics, se3  # noqa: E402


def _or_skip(fn, *args, **kwargs):
    """Call a student function; skip the test if it is still a stub."""
    try:
        return fn(*args, **kwargs)
    except NotImplementedError as e:
        pytest.skip(f"not implemented yet: {e}")


def test_iou_absent_class_is_nan_not_zero():
    """A class in neither prediction nor truth must not drag mIoU down."""
    cm = np.zeros((3, 3))
    cm[0, 0] = 50
    cm[1, 1] = 50                       # class 2 never appears
    out = metrics.iou_from_confusion(cm)
    assert np.isnan(out["per_class_iou"][2])
    assert out["miou"] == pytest.approx(1.0)


def test_confusion_matrix_ignores_negative_labels():
    pred = np.array([0, 1, 0, 1])
    gt = np.array([0, 1, -1, 255])
    cm = metrics.confusion_matrix(pred, gt, num_classes=2)
    assert cm.sum() == 2


def test_binary_rates_separates_the_two_error_types():
    """Part E turns on FN and FP being reported separately, not averaged."""
    pred = np.array([1, 1, 0, 0, 0], dtype=bool)
    gt = np.array([1, 0, 1, 0, 0], dtype=bool)
    r = metrics.binary_rates(pred, gt)
    assert r["tp"] == 1 and r["fp"] == 1 and r["fn"] == 1 and r["tn"] == 2
    assert r["false_negative_rate"] == pytest.approx(0.5)
    assert r["false_positive_rate"] == pytest.approx(1 / 3)


def test_occupancy_query_interface_is_enforced():
    """A representation must implement both methods to be comparable."""
    from downstream import OccupancyQuery

    class Incomplete(OccupancyQuery):
        name = "incomplete"

        def is_occupied(self, points):
            return np.zeros(len(points), dtype=bool)

    with pytest.raises(TypeError):
        Incomplete()                     # memory_mb is still abstract


def test_evaluate_representations_holds_the_procedure_fixed():
    """The harness must run identical queries against every representation."""
    from downstream import OccupancyQuery, evaluate_representations

    class Always(OccupancyQuery):
        def __init__(self, val, name):
            self.val, self.name = val, name

        def is_occupied(self, points):
            return np.full(len(points), self.val, dtype=bool)

        def memory_mb(self):
            return 1.0

    pts = np.random.default_rng(0).normal(size=(100, 3))
    gt = np.zeros(100, dtype=bool)
    gt[:30] = True
    out = evaluate_representations([Always(True, "a"), Always(False, "b")], pts, gt)
    assert set(out) == {"a", "b"}
    assert out["a"]["false_negative_rate"] == 0.0     # never misses
    assert out["b"]["false_negative_rate"] == 1.0     # misses everything
    assert out["a"]["recall"] == 1.0


def test_evaluate_representations_rejects_a_wrong_shaped_prediction():
    from downstream import OccupancyQuery, evaluate_representations

    class Bad(OccupancyQuery):
        name = "bad"

        def is_occupied(self, points):
            return np.zeros(len(points) - 1, dtype=bool)

        def memory_mb(self):
            return 0.0

    pts = np.zeros((10, 3))
    with pytest.raises(ValueError, match="is_occupied returned"):
        evaluate_representations([Bad()], pts, np.zeros(10, dtype=bool))


def test_find_ranking_flip_detects_a_flip():
    from downstream import find_ranking_flip

    fidelity = {"splat": 30.0, "tsdf": 25.0}          # splat renders better
    task = {"splat": 0.60, "tsdf": 0.85}              # tsdf is better for the task
    assert ("splat", "tsdf") in find_ranking_flip(fidelity, task)


def test_find_ranking_flip_empty_when_orders_agree():
    from downstream import find_ranking_flip

    assert find_ranking_flip({"a": 2.0, "b": 1.0}, {"a": 0.9, "b": 0.5}) == []


def test_sample_configurations_is_reproducible():
    from downstream import sample_configurations

    a = sample_configurations([0, 0, 0], [1, 1, 1], n=100, seed=7)
    b = sample_configurations([0, 0, 0], [1, 1, 1], n=100, seed=7)
    assert np.array_equal(a, b), "same seed must give the same query points"


def test_splat_floater_filter_removes_the_obvious_cases():
    from splatting import SplatModel, filter_floaters

    rng = np.random.default_rng(0)
    means = np.vstack([rng.normal(scale=0.1, size=(200, 3)),
                       np.array([[50.0, 50.0, 50.0]])])       # one distant floater
    m = SplatModel(
        means=means,
        scales=np.full((201, 3), 0.01),
        rotations=np.tile([0, 0, 0, 1.0], (201, 1)),
        opacities=np.concatenate([np.full(200, 0.9), [0.9]]),
        sh_coeffs=np.zeros((201, 9, 3)),
    )
    keep = filter_floaters(m)
    assert not keep[-1], "the isolated distant Gaussian should be filtered"
    assert keep[:200].sum() > 150, "the filter removed too much of the real scene"


def test_floater_filter_rejects_log_scales():
    """3DGS implementations store log-scales; the filter needs metres."""
    from splatting import SplatModel, filter_floaters

    m = SplatModel(means=np.zeros((20, 3)), scales=np.full((20, 3), np.log(0.01)),
                   rotations=np.tile([0, 0, 0, 1.0], (20, 1)),
                   opacities=np.full(20, 0.9), sh_coeffs=np.zeros((20, 1, 3)))
    with pytest.raises(ValueError, match="log-scales"):
        filter_floaters(m)


def test_voxelize_inverse_is_flat():
    from segmentation import voxelize

    pts = np.random.default_rng(0).uniform(0, 1, size=(500, 3))
    coords, inverse = voxelize(pts, 0.1)
    assert inverse.shape == (500,)
    assert np.array_equal(coords[inverse], np.floor(pts / 0.1).astype(np.int32))


# --------------------------------------------------------------------------- #
# Part D conventions
# --------------------------------------------------------------------------- #
def _random_rotation(rng):
    return se3.quat_to_rot(rng.normal(size=4))


def _sim3(s, R, t):
    S = np.eye(4)
    S[:3, :3], S[:3, 3] = s * R, t
    return S


def _model_and_world_poses(seed=0, s=2.5):
    """Poses in a model frame, and the same poses mapped into the world by a
    known Sim(3) -- what VGGT's output is relative to the dataset."""
    rng = np.random.default_rng(seed)
    S = _sim3(s, _random_rotation(rng), rng.normal(size=3))
    model = np.array([se3.pose_from_rt(_random_rotation(rng),
                                       [np.cos(i), np.sin(i), 0.1 * i]) for i in range(12)])
    world = []
    for T in model:
        W = np.eye(4)
        W[:3, :3] = S[:3, :3] / s @ T[:3, :3]
        W[:3, 3] = S[:3, :3] @ T[:3, 3] + S[:3, 3]
        world.append(W)
    return model, np.array(world), S


def test_align_to_ground_truth_recovers_a_similarity():
    """An estimate that differs from truth by one Sim(3) has zero error --
    including ROTATION error, which must be measured after the alignment."""
    from feedforward import align_to_ground_truth

    model, world, S = _model_and_world_poses(s=2.5)
    out = align_to_ground_truth(model, world)
    assert out["ate_rmse_m"] < 1e-8
    assert out["scale_factor"] == pytest.approx(2.5)
    assert out["mean_rotation_error_deg"] < 1e-5, (
        "rotation error must be measured after applying the alignment rotation")
    assert np.allclose(out["sim3"], S, atol=1e-8)


def test_feedforward_geometry_is_evaluated_after_alignment():
    from feedforward import FeedForwardResult, apply_sim3, evaluate_geometry

    model, world, S = _model_and_world_poses(s=2.5)
    gt = np.random.default_rng(1).uniform(-1, 1, size=(3000, 3))
    pred_model_frame = apply_sim3(np.linalg.inv(S), gt)
    res = FeedForwardResult(points=pred_model_frame, poses=model, intrinsics=None,
                            n_views=len(model), wall_clock_s=0.0, peak_gpu_mb=0.0)
    r = _or_skip(evaluate_geometry, res, gt, threshold=0.05, sim3=S)
    assert isinstance(r, metrics.ReconResult)
    assert r.threshold == pytest.approx(0.05)
    assert r.f_score > 0.99, "points that coincide with GT after the Sim(3) must score ~1"


# --------------------------------------------------------------------------- #
# Part C contracts
# --------------------------------------------------------------------------- #
def _disc_splats(n=400, seed=0):
    """Flat Gaussians tiling the plane z = 0 -- an idealised splat surface."""
    from splatting import SplatModel

    rng = np.random.default_rng(seed)
    means = np.column_stack([rng.uniform(-1, 1, n), rng.uniform(-1, 1, n), np.zeros(n)])
    return SplatModel(means=means, scales=np.tile([0.08, 0.08, 0.004], (n, 1)),
                      rotations=np.tile([0, 0, 0, 1.0], (n, 1)), opacities=np.full(n, 0.95),
                      sh_coeffs=np.zeros((n, 1, 3)))


def test_splat_surface_lies_on_the_splats():
    from splatting import extract_surface

    out = _or_skip(extract_surface, _disc_splats())
    pts = np.asarray(out[0] if isinstance(out, tuple) else out, dtype=float)
    assert pts.ndim == 2 and pts.shape[1] == 3 and len(pts) > 0
    assert np.abs(pts[:, 2]).max() < 0.1, "surface points should lie on the z = 0 splats"


def test_splat_evaluate_geometry_uses_the_given_threshold():
    from splatting import evaluate_geometry

    gt = np.random.default_rng(0).uniform(0, 1, size=(2000, 3))
    r = _or_skip(evaluate_geometry, gt.copy(), gt, threshold=0.03)
    assert isinstance(r, metrics.ReconResult)
    assert r.threshold == pytest.approx(0.03) and r.f_score == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# Part E contracts: every query answers the same question, in metres
# --------------------------------------------------------------------------- #
def _check_query(q, occupied_pts, free_pts):
    pts = np.vstack([occupied_pts, free_pts])
    out = np.asarray(q.is_occupied(pts))
    assert out.shape == (len(pts),), f"{q.name}.is_occupied returned shape {out.shape}"
    out = out.astype(bool)
    assert out[:len(occupied_pts)].all(), f"{q.name}: missed an obviously occupied point"
    assert not out[len(occupied_pts):].any(), f"{q.name}: flagged an obviously free point"
    assert q.memory_mb() > 0


def _box_mesh(lo=(0.0, 0.0, 0.0), hi=(1.0, 1.0, 1.0)):
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    v = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1])
                  for z in (lo[2], hi[2])])
    f = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1],
                  [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]])
    return v, f


def test_ground_truth_labels_use_the_robot_radius():
    from downstream import label_ground_truth_occupancy

    v, f = _box_mesh()
    pts = np.array([[0.5, 0.5, 1.05],      # 5 cm above the top face
                    [0.5, 0.5, 2.0]])      # 1 m above it
    small = _or_skip(label_ground_truth_occupancy, pts, v, f, robot_radius_m=0.02)
    big = label_ground_truth_occupancy(pts, v, f, robot_radius_m=0.10)
    small, big = np.asarray(small), np.asarray(big)
    assert small.shape == (2,) and big.shape == (2,)
    assert big[0] and not big[1], "a 10 cm robot 5 cm from a face is in collision"
    assert not small[0], "a 2 cm robot 5 cm from a face is free"


def test_splat_occupancy_contract():
    from downstream import SplatOccupancy

    q = _or_skip(SplatOccupancy, _disc_splats(), robot_radius_m=0.1)
    _check_query(q, occupied_pts=[[0.0, 0.0, 0.0], [0.3, -0.2, 0.05]],
                 free_pts=[[0.0, 0.0, 1.0], [5.0, 5.0, 5.0]])


def test_feedforward_occupancy_works_in_metres():
    """The reconstruction is in model units; the query must apply the Sim(3)
    scale. A wall at z = 1 model unit with scale 2 is a wall at z = 2 m."""
    from feedforward import FeedForwardResult
    from downstream import FeedForwardOccupancy

    g = np.linspace(-1, 1, 41)
    wall = np.array([[x, y, 1.0] for x in g for y in g])
    res = FeedForwardResult(points=wall, poses=np.tile(np.eye(4), (3, 1, 1)),
                            intrinsics=None, n_views=3, wall_clock_s=0.0, peak_gpu_mb=0.0)
    S = _sim3(2.0, np.eye(3), np.zeros(3))
    q = _or_skip(FeedForwardOccupancy, res, S, robot_radius_m=0.1)
    _check_query(q, occupied_pts=[[0.0, 0.0, 2.0], [0.5, 0.5, 1.95]],
                 free_pts=[[0.0, 0.0, 1.0], [0.0, 0.0, 3.0]])


def test_segmentation_occupancy_uses_obstacle_classes():
    from downstream import SegmentationOccupancy

    g = np.linspace(-1, 1, 41)
    floor = np.array([[x, y, 0.0] for x in g for y in g])            # class 0
    table = np.array([[x, y, 0.8] for x in g[:10] for y in g[:10]])  # class 1
    pts = np.vstack([floor, table])
    labels = np.r_[np.zeros(len(floor), int), np.ones(len(table), int)]
    q = _or_skip(SegmentationOccupancy, pts, labels, obstacle_classes=[1], robot_radius_m=0.1)
    _check_query(q, occupied_pts=[[-0.9, -0.9, 0.8]],
                 free_pts=[[0.5, 0.5, 0.05], [0.0, 0.0, 0.4]])


def test_tsdf_occupancy_contract():
    """Three cameras at the origin looking down +z at a wall 2 m away."""
    from downstream import TSDFOccupancy

    K = np.array([[60.0, 0, 40], [0, 60.0, 30], [0, 0, 1]])
    depths = np.full((3, 60, 80), 2.0)
    poses = np.tile(np.eye(4), (3, 1, 1))
    poses[1, 0, 3], poses[2, 1, 3] = 0.05, -0.05
    q = _or_skip(TSDFOccupancy, depths, poses, K, voxel_size_m=0.04, robot_radius_m=0.1)
    _check_query(q, occupied_pts=[[0.0, 0.0, 2.0], [0.2, 0.1, 1.95]],
                 free_pts=[[0.0, 0.0, 1.0], [0.1, -0.1, 1.5]])
