"""Tests for the shared rob498 package.  These must pass from day one.

    pytest rob498/tests -v

Everything here is provided, so nothing skips: if one of these fails, the
environment is broken, not your lab.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from rob498 import io3d, metrics, se3  # noqa: E402


def test_skew_unskew_roundtrip():
    w = np.array([0.1, -0.2, 0.3])
    assert np.allclose(se3.unskew(se3.skew(w)), w)


def test_skew_is_antisymmetric():
    W = se3.skew([1.0, 2.0, 3.0])
    assert np.allclose(W, -W.T)


def test_inverse_is_exact():
    R = se3.quat_to_rot(np.array([0.1, 0.2, 0.3, 0.9]))
    T = se3.pose_from_rt(R, [1.0, -2.0, 3.0])
    assert np.allclose(T @ se3.inverse(T), np.eye(4), atol=1e-12)


def test_quaternion_roundtrip_including_180_degrees():
    """The 180-degree case is where a naive trace-based conversion divides by ~0."""
    for axis in np.eye(3):
        n = axis.reshape(3, 1)
        R = 2 * (n @ n.T) - np.eye(3)      # 180-degree rotation about `axis`
        assert se3.is_valid_rotation(R)
        q = se3.rot_to_quat(R)
        assert np.allclose(se3.quat_to_rot(q), R, atol=1e-8), f"failed for axis {axis}"


def test_transform_points_matches_manual():
    R = se3.quat_to_rot(np.array([0.0, 0.0, 0.3827, 0.9239]))  # 45 deg about z
    T = se3.pose_from_rt(R, [1.0, 2.0, 3.0])
    pts = np.random.default_rng(0).normal(size=(50, 3))
    manual = np.array([T[:3, :3] @ p + T[:3, 3] for p in pts])
    assert np.allclose(se3.transform_points(T, pts), manual)


def test_tum_roundtrip(tmp_path):
    """A trajectory written and read back must be unchanged."""
    rng = np.random.default_rng(0)
    poses = []
    for _ in range(10):
        q = rng.normal(size=4)
        poses.append(se3.pose_from_rt(se3.quat_to_rot(q), rng.normal(size=3)))
    poses = np.array(poses)
    ts = np.arange(10) * 0.1
    p = io3d.write_tum(tmp_path / "traj.txt", ts, poses)
    ts2, poses2 = io3d.read_tum(p)
    assert np.allclose(ts, ts2)
    assert np.allclose(poses, poses2, atol=1e-8)


def test_ate_of_identical_trajectories_is_zero():
    rng = np.random.default_rng(0)
    poses = np.array([se3.pose_from_rt(np.eye(3), rng.normal(size=3)) for _ in range(20)])
    assert metrics.ate_rmse(poses, poses)["rmse"] < 1e-9


def test_ate_alignment_removes_a_rigid_offset():
    """An estimate that differs from truth by one rigid transform has ~zero ATE."""
    rng = np.random.default_rng(1)
    # A helix, deliberately NOT collinear: a straight-line trajectory leaves the
    # alignment rotation underdetermined and the test would pass vacuously.
    gt = np.array([
        se3.pose_from_rt(np.eye(3),
                         [np.cos(i * 0.2), np.sin(i * 0.2), i * 0.05])
        for i in range(30)
    ])
    offset = se3.pose_from_rt(se3.quat_to_rot(rng.normal(size=4)), [5.0, -3.0, 2.0])
    est = np.array([offset @ T for T in gt])
    assert metrics.ate_rmse(est, gt, align=True)["rmse"] < 1e-8
    assert metrics.ate_rmse(est, gt, align=False)["rmse"] > 1.0


def test_accuracy_completeness_is_asymmetric():
    """Half a scene scores well on accuracy and badly on completeness."""
    rng = np.random.default_rng(0)
    gt = rng.uniform(0, 1, size=(2000, 3))
    half = gt[gt[:, 0] < 0.5]
    r = metrics.accuracy_completeness(half, gt, threshold=0.02)
    assert r.accuracy_mean < r.completeness_mean
    assert r.recall < 0.75


# --------------------------------------------------------------------------- #
# The exponential and logarithm maps, and the adjoint
# --------------------------------------------------------------------------- #
def test_exp_log_roundtrip():
    rng = np.random.default_rng(0)
    for _ in range(20):
        xi = rng.normal(size=6) * 0.5
        T = se3.exp_se3(xi)
        xi2 = se3.log_se3(T)
        assert np.allclose(xi, xi2, atol=1e-9), f"round trip failed for {xi}"


def test_exp_of_zero_is_identity():
    T = se3.exp_se3(np.zeros(6))
    assert np.allclose(T, np.eye(4), atol=1e-12)


def test_exp_small_angle_is_stable():
    """The small-angle branch must not produce NaN or lose the translation."""
    for theta in [1e-3, 1e-6, 1e-10, 0.0]:
        xi = np.array([1.0, 2.0, 3.0, 0.0, 0.0, theta])
        T = se3.exp_se3(xi)
        assert np.all(np.isfinite(T)), f"non-finite result at theta={theta}"
        assert se3.is_valid_rotation(T[:3, :3]), f"not a rotation at theta={theta}"


def test_exp_produces_valid_rotations():
    rng = np.random.default_rng(2)
    for _ in range(20):
        T = se3.exp_se3(rng.normal(size=6) * 2.0)
        assert se3.is_valid_rotation(T[:3, :3]), "exp_se3 did not return a rotation"


def test_log_near_pi():
    """theta near pi is where a naive log picks the wrong branch."""
    for theta in [np.pi - 1e-3, np.pi - 1e-6]:
        xi = np.array([0.1, 0.2, 0.3, 0.0, 0.0, theta])
        T = se3.exp_se3(xi)
        xi2 = se3.log_se3(T)
        T2 = se3.exp_se3(xi2)
        assert np.allclose(T, T2, atol=1e-7), f"log/exp inconsistent at theta={theta}"


def test_adjoint_identity():
    """Ad(T) must satisfy T exp(xi) = exp(Ad(T) xi) T -- the defining property."""
    rng = np.random.default_rng(3)
    T = se3.exp_se3(rng.normal(size=6))
    Ad = se3.adjoint(T)
    assert Ad.shape == (6, 6), f"adjoint must be 6x6, got {Ad.shape}"
    xi = rng.normal(size=6) * 0.1
    lhs = T @ se3.exp_se3(xi)
    rhs = se3.exp_se3(Ad @ xi) @ T
    assert np.allclose(lhs, rhs, atol=1e-8), (
        "Ad(T) xi is wrong. The usual cause is twist ordering: this course uses "
        "(v, omega), translation first. Check against the block form in the docstring.")


def test_depth_to_points_inverts_the_pinhole():
    K = np.array([[500.0, 0, 320], [0, 500.0, 240], [0, 0, 1]])
    depth = np.zeros((480, 640))
    depth[100, 400] = 2000.0                     # 2 m, in millimetres
    pts, pix = io3d.depth_to_points(depth, K)
    assert pts.shape == (1, 3) and pix.tolist() == [[400, 100]]
    assert np.allclose(pts[0], [(400 - 320) * 2.0 / 500, (100 - 240) * 2.0 / 500, 2.0])
