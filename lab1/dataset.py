"""Lab 1 data loading -- PROVIDED.

Loading the dataset is bookkeeping, not an algorithm, so it is provided: every
student reads the same views, the same poses, and the same held-out split.

Expected layout of the released subset (what ``fetch_data.py`` unpacks)::

    <data>/
      segmentation/                        Part B
        classes.txt                        one class name per line; line i = label i
        train/<scene>.npz                  coord   (N, 3) float32, metres
        val/<scene>.npz                    color   (N, 3) uint8
                                           segment (N,)  int, negative = ignore
      eval_scene/                          Parts C, D, E
        cameras.json                       intrinsics, poses, train/test split
        images/<name>.png|jpg              undistorted RGB
        depth/<name>.png                   uint16 depth aligned with RGB, 0 = no return
        mesh.ply                           ground-truth mesh, world frame, metres

``cameras.json``::

    {
      "K": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],    shared pinhole intrinsics, pixels
      "width": W, "height": H,
      "depth_scale": 1000.0,                         raw depth units per metre
      "frames": [{"name": "frame_000", "T_wc": [[...4x4...]]}, ...],
      "train": ["frame_000", ...],                   views a method may see
      "test":  ["frame_007", ...]                    held-out views (C1)
    }

CAMERA CONVENTION. ``T_wc`` maps camera-frame points into the world, as
everywhere in this course, and the camera axes are OpenCV's: x right, y down,
z forward -- the axes ``rob498.io3d.depth_to_points`` assumes. ScanNet++ ships its
DSLR poses in nerfstudio's ``transforms.json``, which uses OpenGL axes (y up,
z backward). :func:`cameras_from_nerfstudio` does that conversion; use it when
preparing a release from raw ScanNet++ rather than copying the matrices across.

Segmentation scenes may also be given as the ``.pth`` files the ScanNet++
toolbox's semantic preparation writes (keys ``vtx_coords``, ``vtx_colors``,
``vtx_labels``); ScanNet++'s ignore label is -100, which every metric here
already drops because it is negative.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from rob498.io3d import depth_to_points, sample_mesh
from rob498.se3 import transform_points

# OpenGL camera axes (x right, y up, z backward) -> OpenCV (x right, y down, z forward).
GL_TO_CV = np.diag([1.0, -1.0, -1.0, 1.0])

_COORD_KEYS = ("coord", "vtx_coords", "points", "xyz")
_COLOR_KEYS = ("color", "vtx_colors", "colors", "rgb")
_LABEL_KEYS = ("segment", "vtx_labels", "labels", "semantic_gt")


# --------------------------------------------------------------------------- #
# Part B: segmentation scenes
# --------------------------------------------------------------------------- #
@dataclass
class SegScene:
    """One labelled point cloud. ``segment`` < 0 means ignore."""

    name: str
    coord: np.ndarray           # (N, 3) float32, metres
    color: np.ndarray           # (N, 3) uint8
    segment: np.ndarray         # (N,) int64

    def __len__(self) -> int:
        return len(self.coord)


def _pick(d, keys, what, path):
    for k in keys:
        if k in d:
            return np.asarray(d[k])
    raise KeyError(f"{path}: no {what} array; looked for {keys}, found {list(d.keys())}")


def load_seg_scene(path: str | Path) -> SegScene:
    path = Path(path)
    if path.suffix == ".npz":
        with np.load(path) as z:
            d = {k: z[k] for k in z.files}
    elif path.suffix == ".pth":
        import torch

        d = torch.load(path, map_location="cpu", weights_only=False)
    else:
        raise ValueError(f"unsupported segmentation file {path}")
    coord = _pick(d, _COORD_KEYS, "coordinate", path).astype(np.float32)
    color = _pick(d, _COLOR_KEYS, "colour", path)
    if color.dtype != np.uint8:                      # some exports store [0, 1] floats
        color = np.clip(color * (255.0 if color.max() <= 1.0 else 1.0), 0, 255)
    segment = _pick(d, _LABEL_KEYS, "label", path).astype(np.int64).reshape(-1)
    if not (len(coord) == len(color) == len(segment)):
        raise ValueError(f"{path}: length mismatch {len(coord)}/{len(color)}/{len(segment)}")
    return SegScene(path.stem, coord, color.astype(np.uint8), segment)


def load_segmentation_split(data_root: str | Path, split: str) -> list[SegScene]:
    """All scenes of one split (``"train"`` or ``"val"``), sorted by name."""
    d = Path(data_root) / "segmentation" / split
    files = sorted([*d.glob("*.npz"), *d.glob("*.pth")])
    if not files:
        raise FileNotFoundError(f"no .npz or .pth scenes in {d}")
    return [load_seg_scene(f) for f in files]


def load_class_names(data_root: str | Path) -> list[str]:
    path = Path(data_root) / "segmentation" / "classes.txt"
    return [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]


# --------------------------------------------------------------------------- #
# Parts C-E: the evaluation scene
# --------------------------------------------------------------------------- #
@dataclass
class EvalScene:
    """Posed RGB-D views of the evaluation scene plus its ground-truth mesh."""

    root: Path
    K: np.ndarray                         # (3, 3)
    width: int
    height: int
    depth_scale: float
    names: list[str]                      # every frame, in file order
    T_wc: dict[str, np.ndarray]           # name -> (4, 4), OpenCV camera axes
    train: list[str]
    test: list[str]
    _mesh: tuple[np.ndarray, np.ndarray] | None = field(default=None, repr=False)

    # -- per-view access ---------------------------------------------------- #
    def image_path(self, name: str) -> Path:
        for ext in (".png", ".jpg", ".jpeg", ".JPG"):
            p = self.root / "images" / f"{name}{ext}"
            if p.exists():
                return p
        raise FileNotFoundError(f"no image for frame {name!r} in {self.root / 'images'}")

    def image(self, name: str) -> np.ndarray:
        """(H, W, 3) float32 RGB in [0, 1]."""
        from PIL import Image

        return np.asarray(Image.open(self.image_path(name)).convert("RGB"),
                          dtype=np.float32) / 255.0

    def depth_raw(self, name: str) -> np.ndarray:
        """(H, W) raw depth in sensor units (divide by ``depth_scale`` for metres)."""
        from PIL import Image

        return np.asarray(Image.open(self.root / "depth" / f"{name}.png"), dtype=np.float32)

    def depth(self, name: str) -> np.ndarray:
        """(H, W) float32 depth in metres; 0 where there was no return."""
        return self.depth_raw(name) / self.depth_scale

    def pose(self, name: str) -> np.ndarray:
        return self.T_wc[name].copy()

    def poses(self, names: list[str]) -> np.ndarray:
        return np.stack([self.T_wc[n] for n in names])

    def images(self, names: list[str]) -> np.ndarray:
        return np.stack([self.image(n) for n in names])

    def image_paths(self, names: list[str]) -> list[Path]:
        return [self.image_path(n) for n in names]

    # -- ground truth ------------------------------------------------------- #
    def gt_mesh(self) -> tuple[np.ndarray, np.ndarray]:
        """(vertices (V, 3), faces (F, 3)) of the ground-truth mesh."""
        if self._mesh is None:
            self._mesh = read_ply_mesh(self.root / "mesh.ply")
        return self._mesh

    def gt_points(self, n: int = 200_000, seed: int = 0) -> np.ndarray:
        """Area-weighted surface samples of the GT mesh -- what the metrics take."""
        v, f = self.gt_mesh()
        return sample_mesh(v, f, n_points=n, seed=seed)

    def bounds(self, pad: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
        v, _ = self.gt_mesh()
        return v.min(axis=0) - pad, v.max(axis=0) + pad

    # -- sensor geometry ---------------------------------------------------- #
    def fused_points(self, names: list[str] | None = None, stride: int = 2,
                     max_depth: float = 8.0, voxel: float | None = 0.01
                     ) -> tuple[np.ndarray, np.ndarray]:
        """World-frame points and uint8 colours back-projected from the depth of
        `names` (default: the training views). This is the SENSOR's geometry --
        what the segmenter and the TSDF see -- not the ground truth."""
        names = self.train if names is None else names
        pts, cols = [], []
        for n in names:
            d = self.depth_raw(n)[::stride, ::stride]
            K = self.K.copy()
            K[:2] /= stride                 # pixel (u', v') of the strided image is (s u', s v')
            p, pix = depth_to_points(d, K, depth_scale=self.depth_scale, max_depth=max_depth)
            img = self.image(n)[::stride, ::stride]
            pts.append(transform_points(self.T_wc[n], p))
            cols.append((img[pix[:, 1], pix[:, 0]] * 255).astype(np.uint8))
        pts = np.concatenate(pts) if pts else np.zeros((0, 3))
        cols = np.concatenate(cols) if cols else np.zeros((0, 3), np.uint8)
        if voxel and len(pts):
            key = np.floor(pts / voxel).astype(np.int64)
            _, first = np.unique(key, axis=0, return_index=True)
            pts, cols = pts[first], cols[first]
        return pts, cols


def load_eval_scene(data_root: str | Path) -> EvalScene:
    root = Path(data_root) / "eval_scene"
    meta = json.loads((root / "cameras.json").read_text())
    K = np.asarray(meta["K"], dtype=float)
    names, T = [], {}
    for fr in meta["frames"]:
        M = np.asarray(fr["T_wc"], dtype=float)
        if M.shape != (4, 4):
            raise ValueError(f"frame {fr['name']}: T_wc must be 4x4, got {M.shape}")
        names.append(fr["name"])
        T[fr["name"]] = M
    train, test = list(meta["train"]), list(meta["test"])
    leak = set(train) & set(test)
    if leak:
        raise ValueError(f"cameras.json lists {sorted(leak)} as both train and test")
    return EvalScene(root=root, K=K, width=int(meta["width"]), height=int(meta["height"]),
                     depth_scale=float(meta.get("depth_scale", 1000.0)),
                     names=names, T_wc=T, train=train, test=test)


def cameras_from_nerfstudio(transforms: dict) -> dict:
    """Convert a nerfstudio ``transforms.json`` (ScanNet++ DSLR) to ``cameras.json``.

    nerfstudio stores camera-to-world matrices with OpenGL camera axes; this
    flips them to OpenCV axes, which is what every function in this course
    assumes. ScanNet++ lists held-out views under ``test_frames``.
    """
    K = [[transforms["fl_x"], 0.0, transforms["cx"]],
         [0.0, transforms["fl_y"], transforms["cy"]],
         [0.0, 0.0, 1.0]]

    def conv(frames):
        out = []
        for fr in frames:
            name = Path(fr["file_path"]).stem
            T = np.asarray(fr["transform_matrix"], dtype=float) @ GL_TO_CV
            out.append({"name": name, "T_wc": T.tolist()})
        return out

    train = conv(transforms.get("frames", []))
    test = conv(transforms.get("test_frames", []))
    return {"K": K, "width": int(transforms["w"]), "height": int(transforms["h"]),
            "depth_scale": 1000.0, "frames": train + test,
            "train": [f["name"] for f in train], "test": [f["name"] for f in test]}


def read_ply_mesh(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Read a triangle mesh (vertices (V, 3) float64, faces (F, 3) int64)."""
    path = Path(path)
    try:
        import open3d as o3d

        m = o3d.io.read_triangle_mesh(str(path))
        v, f = np.asarray(m.vertices), np.asarray(m.triangles)
        if len(v) and len(f):
            return v.astype(float), f.astype(np.int64)
    except ImportError:
        pass
    from plyfile import PlyData

    ply = PlyData.read(str(path))
    vx = ply["vertex"]
    v = np.stack([vx["x"], vx["y"], vx["z"]], axis=1).astype(float)
    f = np.stack(ply["face"]["vertex_indices"]).astype(np.int64)
    return v, f
