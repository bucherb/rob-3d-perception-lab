"""SE(3) conventions and helpers, all provided.

CONVENTIONS FOR THE LAB. Every function here, every dataset we ship, and
every grading script assumes:

  * A pose is a 4x4 homogeneous matrix ``T`` mapping POINTS FROM the child frame
    INTO the parent frame.  ``T_wc`` maps camera-frame points into world:
    ``p_w = T_wc @ p_c``.
  * Twist coordinates are ordered ``xi = (v, omega)`` -- TRANSLATION FIRST.
    This matters: much of the literature uses (omega, v), and if you mix them
    your adjoint is silently block-transposed. The adjoint below is stated in
    THIS ordering; match it.
  * Perturbations are RIGHT perturbations: ``T = T_bar @ exp(xi^)``.
  * Right-handed rotations, radians, metres.
  * Quaternions, where they appear (TUM trajectory files), are
    ``(qx, qy, qz, qw)`` -- scalar LAST -- because the TUM format says so.

EVERYTHING HERE IS PROVIDED
---------------------------
skew/unskew, composition, inverse, validation, quaternion conversion, error
metrics, and the exponential and logarithm maps and the adjoint. That is
bookkeeping for the lab, not what it assesses. The two places a direct
transcription of the formulas silently loses precision -- small theta in
``exp_se3`` and theta near pi in ``log_se3`` -- are handled, and
``tests/test_shared.py`` checks both.
"""
from __future__ import annotations

import numpy as np

# Below this rotation angle (radians), the Taylor form is better conditioned
# than the closed form in float64. This is our default; you may argue for
# another in your report if you justify it from the cancellation analysis.
SMALL_ANGLE_EPS = 1e-8


# --------------------------------------------------------------------------- #
# Provided: elementary operations
# --------------------------------------------------------------------------- #
def skew(w: np.ndarray) -> np.ndarray:
    """Map a 3-vector to its skew-symmetric matrix (the 'hat' operator)."""
    w = np.asarray(w, dtype=float).reshape(3)
    return np.array([[0.0, -w[2], w[1]],
                     [w[2], 0.0, -w[0]],
                     [-w[1], w[0], 0.0]])


def unskew(W: np.ndarray) -> np.ndarray:
    """Inverse of :func:`skew`. Does not verify that W is skew-symmetric."""
    W = np.asarray(W, dtype=float)
    return np.array([W[2, 1], W[0, 2], W[1, 0]])


def compose(*poses: np.ndarray) -> np.ndarray:
    """Chain poses left to right: ``compose(A, B, C) == A @ B @ C``."""
    out = np.eye(4)
    for T in poses:
        out = out @ np.asarray(T, dtype=float)
    return out


def inverse(T: np.ndarray) -> np.ndarray:
    """Exact SE(3) inverse using R.T rather than a general 4x4 solve.

    Never call ``np.linalg.inv`` on a pose: it is slower, and it lets drift out
    of SO(3) pass unnoticed instead of showing up in your validation.
    """
    T = np.asarray(T, dtype=float)
    R, t = T[:3, :3], T[:3, 3]
    out = np.eye(4)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out


def relative(T_a: np.ndarray, T_b: np.ndarray) -> np.ndarray:
    """Pose of b expressed in a: ``T_ab = inv(T_wa) @ T_wb``."""
    return inverse(T_a) @ T_b


def transform_points(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply a pose to an (N, 3) array of points, returning (N, 3)."""
    T = np.asarray(T, dtype=float)
    pts = np.asarray(pts, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"expected (N, 3) points, got {pts.shape}")
    return pts @ T[:3, :3].T + T[:3, 3]


def is_valid_rotation(R: np.ndarray, tol: float = 1e-6) -> bool:
    """True if R is orthonormal with determinant +1 to within `tol`."""
    R = np.asarray(R, dtype=float)
    if R.shape != (3, 3):
        return False
    return bool(np.allclose(R @ R.T, np.eye(3), atol=tol)
                and abs(np.linalg.det(R) - 1.0) < tol)


def project_to_so3(R: np.ndarray) -> np.ndarray:
    """Nearest rotation matrix in Frobenius norm, via SVD.

    Long ICP chains drift out of SO(3). Re-project every few hundred
    compositions rather than letting the error compound silently.
    """
    U, _, Vt = np.linalg.svd(np.asarray(R, dtype=float))
    D = np.eye(3)
    D[2, 2] = np.sign(np.linalg.det(U @ Vt))
    return U @ D @ Vt


def quat_to_rot(q: np.ndarray) -> np.ndarray:
    """Quaternion (qx, qy, qz, qw), scalar LAST, to a rotation matrix."""
    q = np.asarray(q, dtype=float).reshape(4)
    q = q / np.linalg.norm(q)
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def rot_to_quat(R: np.ndarray) -> np.ndarray:
    """Rotation matrix to quaternion (qx, qy, qz, qw), scalar LAST.

    Branches on the largest diagonal element, so it stays well conditioned for
    all rotations -- including the 180-degree cases where the naive trace
    formula divides by something near zero.
    """
    R = np.asarray(R, dtype=float)
    tr = np.trace(R)
    if tr > 0:
        s = 0.5 / np.sqrt(tr + 1.0)
        w, x = 0.25 / s, (R[2, 1] - R[1, 2]) * s
        y, z = (R[0, 2] - R[2, 0]) * s, (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w, x = (R[2, 1] - R[1, 2]) / s, 0.25 * s
        y, z = (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w, x = (R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s
        y, z = 0.25 * s, (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w, x = (R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s
        y, z = (R[1, 2] + R[2, 1]) / s, 0.25 * s
    q = np.array([x, y, z, w])
    return q / np.linalg.norm(q)


def pose_from_rt(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Assemble a 4x4 pose from a 3x3 rotation and a 3-vector translation."""
    T = np.eye(4)
    T[:3, :3] = np.asarray(R, dtype=float)
    T[:3, 3] = np.asarray(t, dtype=float).reshape(3)
    return T


def rotation_angle(R: np.ndarray) -> float:
    """Geodesic angle of a rotation in radians, clipped against round-off."""
    R = np.asarray(R, dtype=float)
    return float(np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)))


def pose_error(T_est: np.ndarray, T_gt: np.ndarray) -> tuple[float, float]:
    """(translation error in metres, rotation error in radians)."""
    dT = relative(T_gt, T_est)
    return float(np.linalg.norm(dT[:3, 3])), rotation_angle(dT[:3, :3])


# --------------------------------------------------------------------------- #
# Provided: the exponential and logarithm maps, and the adjoint
# --------------------------------------------------------------------------- #
def _coeffs(theta: float) -> tuple[float, float, float]:
    """``sin(t)/t``, ``(1 - cos(t))/t^2`` and ``(t - sin(t))/t^3``, with Taylor
    forms below ``SMALL_ANGLE_EPS`` where the closed forms cancel."""
    if theta < SMALL_ANGLE_EPS:
        t2 = theta * theta
        return 1.0 - t2 / 6.0, 0.5 - t2 / 24.0, 1.0 / 6.0 - t2 / 120.0
    s, c = np.sin(theta), np.cos(theta)
    return s / theta, (1.0 - c) / theta ** 2, (theta - s) / theta ** 3


def exp_se3(xi: np.ndarray) -> np.ndarray:
    """Exponential map from a twist to a pose.

    Args:
        xi: (6,) twist ``(v, omega)``, TRANSLATION FIRST -- see module docstring.

    Returns:
        (4, 4) pose matrix.

    With ``theta = ||omega||`` and ``w^`` the skew matrix of omega::

        R = I + (sin(theta)/theta) w^ + ((1 - cos(theta))/theta^2) (w^)^2
        V = I + ((1 - cos(theta))/theta^2) w^ + ((theta - sin(theta))/theta^3) (w^)^2
        exp(xi^) = [[R, V v], [0, 1]]

    ``V`` is not the identity: the rotation happens *during* the motion, so the
    straight-line displacement is the rotation integrated along the way. Below
    ``SMALL_ANGLE_EPS`` the coefficients use their Taylor forms, because
    ``1 - cos(theta)`` cancels catastrophically in float64.
    """
    xi = np.asarray(xi, dtype=float).reshape(6)
    v, w = xi[:3], xi[3:]
    theta = float(np.linalg.norm(w))
    A, B, C = _coeffs(theta)
    W = skew(w)
    W2 = W @ W
    T = np.eye(4)
    T[:3, :3] = np.eye(3) + A * W + B * W2
    T[:3, 3] = (np.eye(3) + B * W + C * W2) @ v
    return T


def _log_so3(R: np.ndarray) -> np.ndarray:
    """Rotation vector of R, stable at both theta -> 0 and theta -> pi."""
    cos_t = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    theta = float(np.arccos(cos_t))
    if theta < SMALL_ANGLE_EPS:
        return unskew(R - R.T) / 2.0 * (1.0 + theta * theta / 6.0)
    if np.pi - theta < 1e-4:
        # The antisymmetric part vanishes near pi; read the axis from the
        # symmetric part, then fix its sign from the antisymmetric part.
        # Symmetric part: (R + R^T)/2 = cos(t) I + (1 - cos(t)) n n^T, exactly.
        M = ((R + R.T) / 2.0 - cos_t * np.eye(3)) / (1.0 - cos_t)
        k = int(np.argmax(np.diag(M)))
        axis = M[:, k] / np.sqrt(max(M[k, k], 1e-300))
        axis /= np.linalg.norm(axis)
        anti = unskew(R - R.T)
        if axis @ anti < 0:
            axis = -axis
        return axis * theta
    return unskew(R - R.T) * theta / (2.0 * np.sin(theta))


def log_se3(T: np.ndarray) -> np.ndarray:
    """Logarithm map from a pose to a twist; the exact inverse of :func:`exp_se3`
    for rotation angles below pi.

    Args:
        T: (4, 4) pose matrix.

    Returns:
        (6,) twist ``(v, omega)``, translation first.

    ``theta`` comes from the trace, ``omega`` from the antisymmetric part of R
    (or, near pi, the symmetric part, where the antisymmetric part vanishes),
    and ``v = inv(V) @ t`` with V as in :func:`exp_se3`.
    """
    T = np.asarray(T, dtype=float)
    w = _log_so3(T[:3, :3])
    theta = float(np.linalg.norm(w))
    _, B, C = _coeffs(theta)
    W = skew(w)
    V = np.eye(3) + B * W + C * (W @ W)
    v = np.linalg.solve(V, T[:3, 3])
    return np.concatenate([v, w])


def adjoint(T: np.ndarray) -> np.ndarray:
    """Adjoint of a pose, in the (v, omega) ordering.

    Args:
        T: (4, 4) pose matrix.

    Returns:
        (6, 6) matrix satisfying ``T @ exp(xi^) == exp((Ad(T) @ xi)^) @ T``.

    For ``xi`` ordered ``(v, omega)`` and ``T = [[R, t], [0, 1]]``::

        Ad(T) = [[R, t^ R],
                 [0,    R]]

    If you find the blocks transposed in a paper or library, it uses the other
    twist convention -- GTSAM is one -- see the module docstring.
    """
    T = np.asarray(T, dtype=float)
    R, t = T[:3, :3], T[:3, 3]
    Ad = np.zeros((6, 6))
    Ad[:3, :3] = R
    Ad[:3, 3:] = skew(t) @ R
    Ad[3:, 3:] = R
    return Ad
