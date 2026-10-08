#!/usr/bin/env python3
"""Check that the rob498 environment is complete.

    python scripts/check_env.py          # any machine: imports, versions, CPU checks
    python scripts/check_env.py --gpu    # on a Great Lakes GPU node: CUDA, gsplat, spconv

Prints one line per check and exits non-zero if anything required failed. Bring
its output to office hours if setup is the problem.

The first --gpu run compiles gsplat's CUDA kernels (several minutes), which
needs nvcc: `module load cuda/12.x` first. Later runs reuse the cached build.
"""
from __future__ import annotations

import argparse
import importlib
import platform
import sys
import traceback

RESULTS: list[tuple[str, str, str]] = []      # (status, name, detail)


def check(name: str, required: bool = True):
    def deco(fn):
        def run(*a, **k):
            try:
                detail = fn(*a, **k) or ""
                RESULTS.append(("ok", name, str(detail)))
            except Exception as e:  # noqa: BLE001
                status = "FAIL" if required else "warn"
                msg = f"{type(e).__name__}: {e}".splitlines()[0][:160]
                RESULTS.append((status, name, msg))
                if "--verbose" in sys.argv:
                    traceback.print_exc()
        return run
    return deco


def version(mod: str) -> str:
    m = importlib.import_module(mod)
    return getattr(m, "__version__", "?")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpu", action="store_true", help="also check CUDA, gsplat, spconv")
    ap.add_argument("--verbose", action="store_true", help="print tracebacks")
    args = ap.parse_args()
    linux_x86 = sys.platform == "linux" and platform.machine() == "x86_64"

    @check("python 3.10-3.12")
    def _():
        v = sys.version_info
        assert (3, 10) <= v[:2] <= (3, 12), f"found {platform.python_version()}"
        return platform.python_version()
    _()

    @check("numpy < 2")
    def _():
        import numpy as np

        assert int(np.__version__.split(".")[0]) < 2, (
            f"numpy {np.__version__}: VGGT and open3d 0.18 need numpy 1.x")
        return np.__version__
    _()

    for mod in ("scipy", "matplotlib", "skimage", "PIL", "plyfile", "pytest", "tqdm"):
        check(mod)(lambda m=mod: version(m))()

    @check("rob498 (pip install -e .)")
    def _():
        import rob498

        return rob498.__file__
    _()

    @check("torch")
    def _():
        import torch

        dev = ("cuda: " + torch.cuda.get_device_name(0)) if torch.cuda.is_available() else (
            "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
            else "cpu only")
        return f"{torch.__version__} ({dev})"
    _()
    check("torchvision")(lambda: version("torchvision"))()

    @check("open3d TSDF + ray casting")
    def _():
        import numpy as np
        import open3d as o3d

        vol = o3d.pipelines.integration.UniformTSDFVolume(
            length=1.0, resolution=8, sdf_trunc=0.2,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.NoColor)
        assert np.asarray(vol.extract_volume_tsdf()).shape == (512, 2)
        box = o3d.t.geometry.TriangleMesh.from_legacy(o3d.geometry.TriangleMesh.create_box())
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(box)
        d = scene.compute_distance(o3d.core.Tensor([[0.5, 0.5, 2.0]], dtype=o3d.core.float32))
        assert abs(float(d.numpy()[0]) - 1.0) < 1e-5
        return o3d.__version__
    _()

    check("lpips")(lambda: version("lpips"))()

    @check("vggt")
    def _():
        from vggt.models.vggt import VGGT  # noqa: F401

        try:
            from huggingface_hub import try_to_load_from_cache

            hit = try_to_load_from_cache("facebook/VGGT-1B", "config.json")
            cached = isinstance(hit, str)
        except Exception:  # noqa: BLE001
            cached = False
        import os

        if os.environ.get("VGGT_WEIGHTS"):
            return f"weights from VGGT_WEIGHTS={os.environ['VGGT_WEIGHTS']}"
        return "weights cached" if cached else "weights not cached yet (~5 GB on first use)"
    _()

    check("gsplat (Linux only)", required=linux_x86)(lambda: version("gsplat"))()
    check("spconv (Linux x86_64 only)", required=linux_x86)(
        lambda: importlib.import_module("spconv.pytorch") and version("spconv"))()

    if args.gpu:
        @check("CUDA device")
        def _():
            import torch

            assert torch.cuda.is_available(), "torch.cuda.is_available() is False -- not a GPU node?"
            p = torch.cuda.get_device_properties(0)
            gb = p.total_memory / 2 ** 30
            note = "" if gb >= 15.5 else "  <-- the labs need >= 16 GB"
            return f"{p.name}, {gb:.0f} GB, sm_{p.major}{p.minor}, torch CUDA {torch.version.cuda}{note}"
        _()

        @check("gsplat rasterization on CUDA (compiles on first run)")
        def _():
            import torch
            from gsplat import rasterization

            n = 100
            means = torch.randn(n, 3, device="cuda") * 0.3 + torch.tensor([0, 0, 3.0], device="cuda")
            quats = torch.nn.functional.normalize(torch.randn(n, 4, device="cuda"), dim=-1)
            scales = torch.full((n, 3), 0.05, device="cuda")
            opac = torch.full((n,), 0.8, device="cuda")
            cols = torch.rand(n, 3, device="cuda")
            K = torch.tensor([[100.0, 0, 64], [0, 100.0, 48], [0, 0, 1]], device="cuda")
            img, alpha, _ = rasterization(means, quats, scales, opac, cols,
                                          torch.eye(4, device="cuda")[None], K[None], 128, 96)
            assert img.shape == (1, 96, 128, 3) and float(alpha.max()) > 0.1
            return "ok"
        _()

        @check("spconv sparse convolution on CUDA")
        def _():
            import spconv.pytorch as spconv
            import torch

            idx = torch.unique(torch.randint(0, 20, (500, 3)), dim=0).int()
            ind = torch.cat([torch.zeros(len(idx), 1, dtype=torch.int32), idx], 1).cuda()
            feat = torch.randn(len(idx), 4, device="cuda", requires_grad=True)
            x = spconv.SparseConvTensor(feat, ind, [20, 20, 20], 1)
            conv = spconv.SubMConv3d(4, 8, 3, indice_key="s").cuda()
            conv(x).features.sum().backward()
            assert feat.grad is not None
            return "forward + backward ok"
        _()

    width = max(len(n) for _, n, _ in RESULTS)
    for status, name, detail in RESULTS:
        print(f"[{status:>4}] {name:<{width}}  {detail}")
    failed = [n for s, n, _ in RESULTS if s == "FAIL"]
    print()
    if failed:
        print(f"{len(failed)} required check(s) failed: {', '.join(failed)}")
        return 1
    print("environment OK" + ("" if args.gpu else
                              "  (run with --gpu on a GPU node to check CUDA, gsplat, spconv)"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
