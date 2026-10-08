"""File I/O in the exact formats the deliverables require.

Every lab lists required output files. Use these writers and the format is
correct by construction -- the graders parse these files automatically, and a
trajectory written with the quaternion in the wrong order scores zero on a part
you actually completed.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from rob498.se3 import pose_from_rt, quat_to_rot, rot_to_quat


# --------------------------------------------------------------------------- #
# TUM trajectory format  (camera trajectories, e.g. Lab 1 D2)
# --------------------------------------------------------------------------- #
# One pose per line:  timestamp tx ty tz qx qy qz qw
# Scalar-LAST quaternion. Timestamps in seconds, ascending.
def write_tum(path: str | Path, timestamps: Iterable[float], poses: np.ndarray) -> Path:
    """Write poses in TUM format. `poses` is (N, 4, 4) camera-to-world."""
    poses = np.asarray(poses, dtype=float)
    ts = np.asarray(list(timestamps), dtype=float)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError(f"expected (N, 4, 4) poses, got {poses.shape}")
    if len(ts) != len(poses):
        raise ValueError(f"{len(ts)} timestamps for {len(poses)} poses")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write("# timestamp tx ty tz qx qy qz qw\n")
        for t, T in zip(ts, poses):
            q = rot_to_quat(T[:3, :3])
            p = T[:3, 3]
            f.write(f"{t:.9f} {p[0]:.9f} {p[1]:.9f} {p[2]:.9f} "
                    f"{q[0]:.9f} {q[1]:.9f} {q[2]:.9f} {q[3]:.9f}\n")
    return path


def read_tum(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Read a TUM trajectory. Returns (timestamps (N,), poses (N, 4, 4))."""
    ts, poses = [], []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) != 8:
            raise ValueError(f"expected 8 fields per line, got {len(parts)}: {line!r}")
        vals = [float(x) for x in parts]
        ts.append(vals[0])
        poses.append(pose_from_rt(quat_to_rot(np.array(vals[4:8])), np.array(vals[1:4])))
    if not poses:
        raise ValueError(f"no poses found in {path}")
    return np.array(ts), np.array(poses)


def associate(
    ts_a: np.ndarray, ts_b: np.ndarray, max_difference: float = 0.02
) -> tuple[np.ndarray, np.ndarray]:
    """Greedily match two timestamp sequences within `max_difference` seconds.

    Estimated and ground-truth trajectories are rarely sampled at identical
    times. Comparing them by index instead of by timestamp is a silent error
    that produces a plausible-looking, wrong ATE. Returns index arrays into a
    and b.
    """
    ts_a = np.asarray(ts_a, dtype=float)
    ts_b = np.asarray(ts_b, dtype=float)
    ia, ib = [], []
    used_b: set[int] = set()
    for i, t in enumerate(ts_a):
        d = np.abs(ts_b - t)
        for j in np.argsort(d):
            if d[j] > max_difference:
                break
            if int(j) not in used_b:
                ia.append(i); ib.append(int(j)); used_b.add(int(j))
                break
    return np.array(ia, dtype=int), np.array(ib, dtype=int)


# --------------------------------------------------------------------------- #
# Depth images
# --------------------------------------------------------------------------- #
def depth_to_points(
    depth: np.ndarray, K: np.ndarray, depth_scale: float = 1000.0,
    max_depth: float = 8.0
) -> tuple[np.ndarray, np.ndarray]:
    """Back-project a depth image to camera-frame points.

    Returns ((M, 3) points, (M, 2) integer pixel coords of the valid pixels).
    Drops zeros (no return) and anything beyond `max_depth`. Keeping the pixel
    coordinates lets you look up colour or a mask for the same points without
    recomputing the correspondence -- Lab 1 B2 needs exactly this.
    """
    d = np.asarray(depth, dtype=float) / depth_scale
    h, w = d.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    valid = (d > 0) & (d < max_depth)
    z = d[valid]
    K = np.asarray(K, dtype=float)
    x = (u[valid] - K[0, 2]) * z / K[0, 0]
    y = (v[valid] - K[1, 2]) * z / K[1, 1]
    return np.stack([x, y, z], axis=1), np.stack([u[valid], v[valid]], axis=1)


# --------------------------------------------------------------------------- #
# Point clouds and meshes
# --------------------------------------------------------------------------- #
def write_ply_points(
    path: str | Path, points: np.ndarray, colors: np.ndarray | None = None
) -> Path:
    """Write an (N, 3) point cloud as a binary PLY. `colors` is (N, 3) uint8."""
    pts = np.asarray(points, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"expected (N, 3) points, got {pts.shape}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    has_color = colors is not None
    if has_color:
        col = np.asarray(colors, dtype=np.uint8)
        if col.shape != pts.shape:
            raise ValueError(f"colors {col.shape} must match points {pts.shape}")
    header = ["ply", "format binary_little_endian 1.0", f"element vertex {len(pts)}",
              "property float x", "property float y", "property float z"]
    if has_color:
        header += ["property uchar red", "property uchar green", "property uchar blue"]
    header.append("end_header")
    with path.open("wb") as f:
        f.write(("\n".join(header) + "\n").encode())
        if has_color:
            dt = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                           ("r", "u1"), ("g", "u1"), ("b", "u1")])
            rec = np.empty(len(pts), dtype=dt)
            rec["x"], rec["y"], rec["z"] = pts[:, 0], pts[:, 1], pts[:, 2]
            rec["r"], rec["g"], rec["b"] = col[:, 0], col[:, 1], col[:, 2]
            f.write(rec.tobytes())
        else:
            f.write(pts.tobytes())
    return path


def write_ply_mesh(path: str | Path, vertices: np.ndarray, faces: np.ndarray) -> Path:
    """Write a triangle mesh as a binary PLY. `faces` is (M, 3) int indices."""
    v = np.asarray(vertices, dtype=np.float32)
    fc = np.asarray(faces, dtype=np.int32)
    if v.ndim != 2 or v.shape[1] != 3:
        raise ValueError(f"expected (N, 3) vertices, got {v.shape}")
    if fc.ndim != 2 or fc.shape[1] != 3:
        raise ValueError(f"expected (M, 3) triangle faces, got {fc.shape}")
    if fc.size and (fc.max() >= len(v) or fc.min() < 0):
        raise ValueError("face indices out of range")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = ["ply", "format binary_little_endian 1.0", f"element vertex {len(v)}",
              "property float x", "property float y", "property float z",
              f"element face {len(fc)}", "property list uchar int vertex_indices",
              "end_header"]
    with path.open("wb") as f:
        f.write(("\n".join(header) + "\n").encode())
        f.write(v.tobytes())
        dt = np.dtype([("n", "u1"), ("a", "<i4"), ("b", "<i4"), ("c", "<i4")])
        rec = np.empty(len(fc), dtype=dt)
        rec["n"] = 3
        rec["a"], rec["b"], rec["c"] = fc[:, 0], fc[:, 1], fc[:, 2]
        f.write(rec.tobytes())
    return path


def sample_mesh(
    vertices: np.ndarray, faces: np.ndarray, n_points: int = 200_000, seed: int = 0
) -> np.ndarray:
    """Uniformly sample points on a mesh surface, area-weighted.

    Call this before :func:`rob498.metrics.accuracy_completeness`. Comparing raw
    vertices instead biases the metric toward wherever your mesher placed
    vertices, which for marching cubes means toward high-curvature regions --
    and it makes your numbers incomparable with anyone else's.
    """
    v = np.asarray(vertices, dtype=float)
    fc = np.asarray(faces, dtype=int)
    a, b, c = v[fc[:, 0]], v[fc[:, 1]], v[fc[:, 2]]
    areas = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    total = areas.sum()
    if total <= 0:
        raise ValueError("mesh has zero total area")
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(fc), size=n_points, p=areas / total)
    u = rng.random((n_points, 1))
    w = rng.random((n_points, 1))
    flip = (u + w) > 1
    u[flip], w[flip] = 1 - u[flip], 1 - w[flip]
    return a[idx] + u * (b[idx] - a[idx]) + w * (c[idx] - a[idx])


# --------------------------------------------------------------------------- #
# JSON  (manifests and other structured deliverables)
# --------------------------------------------------------------------------- #
class _NumpyEncoder(json.JSONEncoder):
    """Serialise NumPy scalars and arrays, which json refuses by default.

    Without this, ``json.dump`` on a dict holding a np.float32 raises
    ``TypeError: Object of type float32 is not JSON serializable`` -- reliably,
    at 11pm on the due date.
    """

    def default(self, o: Any) -> Any:
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, Path):
            return str(o)
        return super().default(o)


def write_json(path: str | Path, obj: Any, indent: int = 2) -> Path:
    """Write JSON, handling NumPy types. Keys are sorted for stable diffs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, cls=_NumpyEncoder, indent=indent, sort_keys=True))
    return path


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text())
