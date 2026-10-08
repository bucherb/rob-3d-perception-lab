"""Lab 1 Part E -- The downstream task.  [40 pts]

THE TRAP IN THIS LAB, restated: three pipelines and a table of PSNR values,
concluding that the highest number wins, fails this part -- 40 of the lab's 100
points. The question is not which representation reconstructs best. It is which
one a robot can use.

The query harness below is PROVIDED and you should not modify it. E2 requires
the query procedure held fixed across representations so the comparison is about
the representation and not your query code. If each representation gets its own
bespoke query implementation, the comparison measures your three implementations
and the finding is worthless. What you implement is one `occupancy_query`
per representation behind a common interface.

The three representations are the ones Parts B-D built -- the segmented sensor
point cloud (B), the splats (C), and the feed-forward reconstruction (D) -- plus
a TSDF fused from the same depth, as the baseline every row is read against.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


class OccupancyQuery(ABC):
    """One representation's answer to "is the point x occupied?".

    This interface is the whole lab in miniature. "Is the point x occupied?" has
    a different procedure and a different asymptotic cost for each of the three
    representations, and for a splat model the answer is not even well defined
    -- a Gaussian has support everywhere, so any occupancy answer depends on a
    threshold you chose. Implement one subclass per representation, and state in
    your report what each is really thresholding and what it costs per query.

    ROBOT RADIUS. Every subclass takes ``robot_radius_m``. With r > 0,
    ``is_occupied(x)`` asks whether a sphere of radius r centred at x touches
    the scene -- the configuration-space obstacle of a spherical robot, which is
    what collision checking needs. With r = 0 it is point occupancy. Use the SAME
    r for every representation and for :func:`label_ground_truth_occupancy`.
    """

    name: str = "unnamed"

    @abstractmethod
    def is_occupied(self, points: np.ndarray) -> np.ndarray:
        """(N, 3) points -> (N,) boolean occupancy."""

    @abstractmethod
    def memory_mb(self) -> float:
        """Resident size of the representation, in MB. Measure, do not estimate."""

    def describe(self) -> str:
        """One sentence on what is being thresholded. Quote it in E2 and E4."""
        return f"{self.name}: no description provided"


@dataclass
class QueryTiming:
    """Latency of a batch of queries, for the summary table's latency column."""

    total_s: float
    n_queries: int

    @property
    def per_query_ms(self) -> float:
        return 1000.0 * self.total_s / max(self.n_queries, 1)


def time_queries(q: OccupancyQuery, points: np.ndarray, repeats: int = 3) -> QueryTiming:
    """PROVIDED. Time a batch of occupancy queries, best of `repeats`.

    Best-of rather than mean, to reduce the effect of an unlucky GC pause or a
    competing job on the cluster. Say `repeats` in your report.
    """
    import time

    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        q.is_occupied(points)
        best = min(best, time.perf_counter() - t0)
    return QueryTiming(total_s=best, n_queries=len(points))


def sample_configurations(
    bounds_min: np.ndarray, bounds_max: np.ndarray, n: int = 2000, seed: int = 0
) -> np.ndarray:
    """PROVIDED. Uniformly sample (n, 3) positions in an axis-aligned box.

    Provided so that every student queries the SAME distribution of points given
    the same seed, which makes your false-negative rates comparable to a
    classmate's. Report the seed.
    """
    rng = np.random.default_rng(seed)
    lo = np.asarray(bounds_min, dtype=float)
    hi = np.asarray(bounds_max, dtype=float)
    return rng.uniform(lo, hi, size=(n, 3))


def evaluate_representations(
    queries: list[OccupancyQuery],
    test_points: np.ndarray,
    gt_occupied: np.ndarray,
) -> dict[str, dict]:
    """PROVIDED. Run the identical evaluation across every representation.

    This is the fixed procedure E2 requires. Returns a dict keyed by
    representation name, each holding the binary rates from
    ``rob498.metrics.binary_rates`` plus latency and memory.

    Read the false-NEGATIVE rate first. For collision checking a false negative
    is a collision and a false positive is a detour, and E1 asks you to say what
    each costs. They are not symmetric and a single accuracy number hides it.
    """
    from rob498.metrics import binary_rates

    gt = np.asarray(gt_occupied).astype(bool)
    if len(gt) != len(test_points):
        raise ValueError(f"{len(gt)} labels for {len(test_points)} points")

    out: dict[str, dict] = {}
    for q in queries:
        pred = np.asarray(q.is_occupied(test_points)).astype(bool)
        if pred.shape != gt.shape:
            raise ValueError(
                f"{q.name}.is_occupied returned {pred.shape}, expected {gt.shape}")
        rates = binary_rates(pred, gt)
        timing = time_queries(q, test_points)
        out[q.name] = {
            **rates,
            "query_latency_ms": timing.per_query_ms,
            "memory_mb": q.memory_mb(),
            "description": q.describe(),
        }
    return out


def find_ranking_flip(
    fidelity: dict[str, float], task: dict[str, float], higher_is_better_task: bool = True
) -> list[tuple[str, str]]:
    """PROVIDED. Find pairs whose fidelity and task rankings disagree.  [E3]

    Args:
        fidelity: representation name -> fidelity score (higher is better).
        task: representation name -> task metric.

    Returns:
        Pairs (a, b) where a beats b on fidelity but loses on the task. A tie on
        either metric is not a flip.

    This finds the flip mechanically; it does not explain it, and E3 is graded
    on the explanation. If the list comes back empty, that is a reportable
    result -- E3 gives full credit for a well-supported negative -- but check
    first that your fidelity metric is the one that should rank things (PSNR
    ranks rendering, not geometry) before concluding no flip exists.
    """
    flips = []
    for a in fidelity:
        for b in fidelity:
            if a == b or a not in task or b not in task:
                continue
            better_fid = fidelity[a] > fidelity[b]
            worse_task = task[a] < task[b] if higher_is_better_task else task[a] > task[b]
            if better_fid and worse_task:
                flips.append((a, b))
    return flips


# --------------------------------------------------------------------------- #
# YOU WRITE THESE -- one subclass per representation
# --------------------------------------------------------------------------- #
class TSDFOccupancy(OccupancyQuery):
    """Occupancy from a TSDF baseline.  [E2]

    Fuse the TSDF from the dataset's training-view depth and poses with Open3D's
    ``open3d.pipelines.integration.UniformTSDFVolume``. That is permitted: the
    TSDF is the baseline here, not the thing assessed. The occupancy query on it
    is yours. ``extract_volume_tsdf()`` returns the dense grid -- (res**3, 2)
    rows of (tsdf, weight), x-major, tsdf normalised by ``sdf_trunc`` -- and
    trilinear interpolation of it at the query point, compared against the
    robot radius, is O(1) per query: the honest advantage of a grid, and worth
    saying out loud. (``ScalableTSDFVolume`` is the right class for extracting
    a mesh, but it does not expose its voxels for querying.)

    Three conventions bite here. ``integrate`` takes the WORLD-TO-CAMERA
    extrinsic, the inverse of the ``T_wc`` the dataset gives you. Voxels with
    weight 0 were never observed, and their tsdf of 0 is not "on the surface" --
    decide whether unobserved space is occupied, and say so. And the truncation
    distance must exceed the robot radius, or every query beyond it reads as free.
    """

    name = "tsdf"

    def __init__(self, depths, poses, intrinsics, voxel_size_m: float = 0.04,
                 robot_radius_m: float = 0.0, **kwargs):
        """`depths` (V, H, W) metres, `poses` (V, 4, 4) camera-to-world."""
        raise NotImplementedError("Lab 1 E2: fuse the TSDF baseline")

    def is_occupied(self, points: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Lab 1 E2: TSDF occupancy query")

    def memory_mb(self) -> float:
        raise NotImplementedError("Lab 1 E2: TSDF resident memory")


class SplatOccupancy(OccupancyQuery):
    """Occupancy from your 3DGS model.  [E2 -- and the crux of E3]

    State precisely what quantity you are thresholding, and why the answer is
    not well defined. A Gaussian has support everywhere, so any occupancy answer
    depends on a threshold you chose;
    accumulated opacity along a ray, density at the point, and distance to the
    nearest Gaussian centre scaled by its covariance are all defensible and they
    disagree. Pick one, define it in an equation, and note what the others would
    have given.

    The robot radius has a clean form here: a sphere of radius r swept against a
    Gaussian with covariance Sigma behaves, to second order, like a Gaussian with
    covariance ``Sigma + r**2 I``.
    """

    name = "3dgs"

    def __init__(self, model, robot_radius_m: float = 0.0, **kwargs):
        """`model` is a :class:`splatting.SplatModel`, floaters handled your way."""
        raise NotImplementedError("Lab 1 E2: build the splat occupancy query")

    def is_occupied(self, points: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Lab 1 E2: splat occupancy query")

    def memory_mb(self) -> float:
        raise NotImplementedError("Lab 1 E2: splat resident memory")


class FeedForwardOccupancy(OccupancyQuery):
    """Occupancy from the VGGT reconstruction.  [E2]

    Remember D2's scale factor. If you aligned with Sim(3) to report ATE, the
    raw geometry is NOT in metres, and a collision check in the wrong units will
    look like a catastrophic failure of the representation when it is actually a
    units bug in your query. Fix the scale, then evaluate, and say in the report
    that you did -- a robot that needs metric distances cannot apply a scale it
    learns from ground truth it does not have, which is itself the finding.
    """

    name = "vggt"

    def __init__(self, result, sim3, robot_radius_m: float = 0.0, **kwargs):
        """`result` is a :class:`feedforward.FeedForwardResult`; `sim3` is
        ``align_to_ground_truth(...)["sim3"]``, which puts it in metres."""
        raise NotImplementedError("Lab 1 E2: build the feed-forward occupancy query")

    def is_occupied(self, points: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Lab 1 E2: feed-forward occupancy query")

    def memory_mb(self) -> float:
        raise NotImplementedError("Lab 1 E2: feed-forward resident memory")


class SegmentationOccupancy(OccupancyQuery):
    """Occupancy from the Part B representation: the segmented sensor cloud.  [E2]

    The sensor's own points (back-projected training-view depth,
    ``dataset.EvalScene.fused_points``), labelled by your Part B model. Which
    predicted classes count as obstacles is part of your E1 definition: for a
    robot that drives on the floor the floor is not an obstacle, and this is
    the row where B3's "did mIoU measure what the task needed?" gets answered
    with a number.
    """

    name = "segmentation"

    def __init__(self, points, labels, obstacle_classes, robot_radius_m: float = 0.0,
                 **kwargs):
        """`points` (N, 3) world frame, metres; `labels` (N,) predicted classes."""
        raise NotImplementedError("Lab 1 E2: build the segmentation occupancy query")

    def is_occupied(self, points: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Lab 1 E2: segmentation occupancy query")

    def memory_mb(self) -> float:
        raise NotImplementedError("Lab 1 E2: segmentation resident memory")


def label_ground_truth_occupancy(
    points: np.ndarray, gt_mesh_vertices: np.ndarray, gt_mesh_faces: np.ndarray,
    robot_radius_m: float = 0.0,
) -> np.ndarray:
    """Label sampled configurations against the ground-truth mesh.  [E1, 8 pts]

    Returns (N,) boolean, True = in collision.

    E1 asks you to define your metric formally BEFORE running anything, and this
    function is where the definition lives. Decisions to state explicitly: is a
    point inside the mesh in collision, or within `robot_radius_m` of the
    surface? What happens for a query outside the scanned volume -- collision,
    free, or excluded? Those choices move the numbers more than the choice of
    representation does, which is exactly why they belong in the report.

    A scanned room mesh is not watertight, so an inside/outside test on it
    (ray parity, or the sign of Open3D's ``compute_signed_distance``) is not
    reliable: a closed room makes all of its free space read as "inside".
    Unsigned distance to the surface, compared against ``robot_radius_m``, is
    always well defined; restricting the sampled configurations to observed
    space is how you keep the interiors of furniture out of the count.
    """
    raise NotImplementedError("Lab 1 E1: label ground-truth occupancy")
